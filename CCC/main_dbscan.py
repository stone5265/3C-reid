import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import random
import collections
import copy

from torch.backends import cudnn
from sklearn.cluster import DBSCAN

import .models
from .trainer import Trainer
from .evaluator import exact_features, Evaluator
from .configs import config
from .utils.dataloader import get_train_loader, get_test_loader
from .utils.datasets import create
from .utils.memory_table import MemoryTable
from .utils.lr_scheduler import WarmupMultiStepLR, WarmupMultiStepLRv2
from .utils.logger import Logger#, HDC_Logger
from .utils.faiss_rerank import compute_jaccard_distance


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)

    return centers


def main_work(args):
    print('\033[1;31;10m'+args.notes+'\033[0m')
    cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Warmup-checkpoint
    load_warmup_ckpt = args.warmup_iters > 0
    warmup_ckpt_dir = '[DBSCAN_warmup]'
    warmup_ckpt_dir += f'{args.warmup_iters}iters,{args.cm_mode},{args.pooling_type}'
    warmup_ckpt_dir += f',resnet{args.resnet_pretrained}' if 'ibn' not in args.arch else ',ibn_resnet'
    warmup_ckpt_dir += ',reweight' if args.loss_with_camera else ''
    warmup_ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, warmup_ckpt_dir)
    exist_warmup_ckpt = os.path.exists(warmup_ckpt_dir)

    # Model
    model = models.create(args.arch, pretrained=args.resnet_pretrained,
                          norm=True, pooling_type=args.pooling_type)
    num_features = model.num_features
    if load_warmup_ckpt and exist_warmup_ckpt:
        model.load_state_dict(torch.load(os.path.join(warmup_ckpt_dir, 'model.pth')), strict=False)
    model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])

    # Optimizer
    params = [{'params': [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.lr_weight_decay)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr

    # Scheduler
    if args.warmup_version == 'v2':
        scheduler = WarmupMultiStepLRv2(optimizer,
                                        # milestones=[x * args.lr_steps for x in range(1, 99)],
                                        milestones=[args.warmup_iters + x * args.lr_steps for x in range(1, 99)],
                                        gamma=args.lr_gamma,
                                        warmup_lr_factor=args.warmup_lr_factor,
                                        warmup_iters=args.warmup_iters,
                                        warmup_factor=args.warmup_factor,
                                        last_epoch=args.warmup_iters-1 if load_warmup_ckpt and exist_warmup_ckpt else -1)
    else:
        scheduler = WarmupMultiStepLR(optimizer,
                                      # milestones=[x * args.lr_steps for x in range(1, 99)],
                                      milestones=[args.warmup_iters + x * args.lr_steps for x in range(1, 99)],
                                      gamma=args.lr_gamma,
                                      warmup_iters=args.warmup_iters,
                                      warmup_factor=args.warmup_factor,
                                      last_epoch=args.warmup_iters-1 if load_warmup_ckpt and exist_warmup_ckpt else -1)

    # Evaluator
    evaluator = Evaluator(model, args.eval_dist)

    # Logger
    logger = Logger(args)
    logger_full = Logger(args, 'full-version')
    if load_warmup_ckpt and exist_warmup_ckpt:
        logger.resume(os.path.join(warmup_ckpt_dir, 'loss.txt'))
        logger_full.resume(os.path.join(warmup_ckpt_dir, 'full_version.txt'), cluster_results=True)

    # Dataset
    dataset = create(args.dataset, args.root_dir)
    cluster_loader = get_test_loader(args, dataset.train)
    test_loader = get_test_loader(args, dataset.gallery)
    query_loader = get_test_loader(args, dataset.query)

    # Cluster
    cluster = DBSCAN(eps=args.dbscan_eps, metric='precomputed',
                     min_samples=args.cluster_min_instances, n_jobs=args.num_workers)

    # Memory
    memory = MemoryTable(cluster=cluster, num_features=num_features,
                         temp=args.cm_temperature, momentum=args.cm_momentum, mode=args.cm_mode)

    for i_epoch in range(args.num_epochs):
        if i_epoch < args.warmup_iters and load_warmup_ckpt and exist_warmup_ckpt:
            continue
        # Exact features
        features = exact_features(model, cluster_loader)

        # Update centers & pseudo labels
        print('Updating centers & pseudo labels...', end='')
        start = time.time()

        rerank_dist = compute_jaccard_distance(features, k1=args.jaccard_k1, k2=args.jaccard_k2)
        print('(Jaccard distance using {:.5}s.)'.format(time.time() - start), end='')
        memory.pseudo_labels = cluster.fit_predict(rerank_dist)
        centroids = generate_cluster_features(memory.pseudo_labels, features)
        memory.cluster_memory.features = F.normalize(centroids, dim=1).to(device)
        memory.num_clusters = len(set(memory.pseudo_labels)) - (1 if -1 in memory.pseudo_labels else 0)

        num_outliers = 0
        new_dataset = []
        for (fname, _, cid), label in zip(dataset.train, memory.pseudo_labels):
            if label == -1:
                num_outliers += 1
            else:
                new_dataset.append((fname, int(label), cid))
        train_loader = get_train_loader(args, new_dataset, args.iters)
        train_loader.new_epoch()
        cluster_time = time.time()-start
        print(' using {:.5}s.'.format(cluster_time))
        print('Current num pseudo labels: {}. num_outliers: {}.'.format(memory.num_clusters, num_outliers))
        time.sleep(0.01)

        del features

        # Fine-tuning model
        trainer = Trainer(model, optimizer, scheduler, memory, use_camera=args.loss_with_camera)
        trainer.train(train_loader=train_loader,
                      title='[epoch {}/{}] Train'.format(i_epoch + 1, args.num_epochs))

        # Evaluate
        if (i_epoch+1) >= args.eval_start_epoch and ((i_epoch+1) % args.eval_step_epoch == 0 or i_epoch == 0):
            start = time.time()
            evaluator.evaluate(query_loader, test_loader, args.rerank)
            time.sleep(0.01)
            print('Evaluate using {:.5}s.'.format(time.time() - start))
            print('    mAP: {:4.2%}'.format(evaluator.mAP), end='')
            for cmc in evaluator.cmc_scores:
                print(',    Rank{}: {:4.2%}'.format(cmc[0], cmc[1]), end='')
            print()
            time.sleep(0.01)

            # Record
            new_best = logger.record(i_epoch, trainer, evaluator, memory.num_clusters, num_outliers, cluster_time)
            logger_full.record(i_epoch, trainer, evaluator, memory.num_clusters,
                               num_outliers, cluster_time, cluster_result=new_dataset)

            # Save checkpoint
            if args.eval_and_ckpt and i_epoch > 0:
                state_dict = collections.OrderedDict()
                for k, v in model.state_dict().items():
                    state_dict[k.strip('module.')] = copy.deepcopy(v).cpu()
                torch.save(state_dict, os.path.join(warmup_ckpt_dir, f'model_e{i_epoch+1}.pth'))
            # Save best_mAP model
            if new_best:
                state_dict = collections.OrderedDict()
                for k, v in model.state_dict().items():
                    state_dict[k.strip('module.')] = copy.deepcopy(v).cpu()
                torch.save(state_dict, os.path.join(warmup_ckpt_dir, f'model_best.pth'))
        else:
            logger_full.record(i_epoch, trainer, None, memory.num_clusters,
                               num_outliers, cluster_time, cluster_result=new_dataset)

        # Save warmup checkpoint
        if i_epoch + 1 == args.warmup_iters and load_warmup_ckpt and not exist_warmup_ckpt:
            os.makedirs(warmup_ckpt_dir)
            state_dict = collections.OrderedDict()
            for k, v in model.state_dict().items():
                state_dict[k.strip('module.')] = copy.deepcopy(v).cpu()
            torch.save(state_dict, os.path.join(warmup_ckpt_dir, 'model.pth'))
            logger.save(os.path.join(warmup_ckpt_dir, 'loss.txt'))
            logger_full.save(os.path.join(warmup_ckpt_dir, 'full_version.txt'))
            logger.save_config(os.path.join(warmup_ckpt_dir, 'config.ini'))

    print('Training complete')
    logger.finish()
    logger_full.finish()


def main():
    args = config(dbscan=True).parse_args()
    if args.no_jaccard:
        del args.jaccard_k1
        del args.jaccard_k2
    if 'ibn' in args.arch:
        args.pretrained = True
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    main_work(args)


if __name__ == '__main__':
    main()
