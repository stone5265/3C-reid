import os
import os.path as osp
import time
import shutil
import collections
import torch
# import numpy as np


class Logger(object):
    def __init__(self, args, name='loss'):
        self.log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        log_dir = osp.join(args.log_dir, args.dataset, self.log_name)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        self.log_path = osp.join(log_dir,  '{}.txt'.format(name))
        self.config_path = osp.join(log_dir, 'config.ini')
        self.log_dir = log_dir
        self.best_mAP = {
            'i_epoch': 0,
            'mAP': 0.,
            'Rank1': 0.,
            'Rank5': 0.,
            'Rank10': 0.
        }
        self.cluster_results = collections.OrderedDict()

        with open(self.log_path, 'w') as f:
            f.write('start time: {}.\n\n'.format(time.asctime(time.localtime())))
        with open(self.config_path, 'w') as f:
            for key, value in args._get_kwargs():
                f.write('{}={}\n'.format(key, value))
            f.write('\n')

    def record(self, i_epoch, trainer, evaluator, num_clusters, num_outliers, cluster_time, cluster_result=None):
        if cluster_result is not None:
            self.cluster_results[i_epoch] = cluster_result
        with open(self.log_path, 'a') as f:
            f.write('Epoch {}:\n'.format(i_epoch+1))
            f.write('    Clustering use {:.5}s.\n'.format(cluster_time))
            f.write('    Current num pesudo labels: {}. num_outliers: {}.\n'.format(num_clusters, num_outliers))
            for (loss_name, loss_value) in trainer.loss:
                f.write('    {}: {}\n'.format(loss_name, loss_value))
            if evaluator is not None:
                f.write('    mAP: {:4.2%}\n'.format(evaluator.mAP))
                for cmc in evaluator.cmc_scores:
                    f.write('    Rank{}: {:4.2%}\n'.format(cmc[0], cmc[1]))
        new_best = False
        if evaluator is not None:
            if evaluator.mAP > self.best_mAP['mAP']:
                new_best = True
                self.best_mAP['i_epoch'] = i_epoch
                self.best_mAP['mAP'] = evaluator.mAP
                for cmc in evaluator.cmc_scores:
                    self.best_mAP['Rank%d' % cmc[0]] = cmc[1]
        return new_best

    def finish(self):
        if len(self.cluster_results) > 0:
            torch.save(self.cluster_results, osp.join(self.log_dir, 'cluster_results.pt'))
        with open(self.log_path, 'a') as f:
            f.write('\n (Epoch {})'.format(self.best_mAP['i_epoch'] + 1))
            f.write('  best mAP: {:4.2%}'.format(self.best_mAP['mAP']))
            for rank in ['Rank1', 'Rank5', 'Rank10']:
                f.write(',  {}: {:4.2%}'.format(rank, self.best_mAP[rank]))
            f.write('\n\nend time: {}.\n'.format(time.asctime(time.localtime())))

        time.sleep(0.01)
        print(' (Epoch {})'.format(self.best_mAP['i_epoch'] + 1), end='')
        print('  best mAP: {:4.2%}'.format(self.best_mAP['mAP']), end='')
        for rank in ['Rank1', 'Rank5', 'Rank10']:
            print(',  {}: {:4.2%}'.format(rank, self.best_mAP[rank]), end='')
        print()
        time.sleep(0.01)

    def resume(self, checkpoint_path, cluster_results=False):
        with open(self.log_path, 'a') as f:
            with open(checkpoint_path, 'r') as src:
                lines = src.readlines()
            f.writelines(lines[2:])
        if cluster_results:
            checkpoint_dir = osp.dirname(checkpoint_path)
            if osp.exists(osp.join(checkpoint_dir, 'cluster_results.pt')):
                self.cluster_results = torch.load(osp.join(checkpoint_dir, 'cluster_results.pt'))

    def save(self, checkpoint_path):
        shutil.copyfile(self.log_path, checkpoint_path)
        if len(self.cluster_results) > 0:
            checkpoint_dir = osp.dirname(checkpoint_path)
            torch.save(self.cluster_results, osp.join(checkpoint_dir, 'cluster_results.pt'))

    def save_config(self, checkpoint_path):
        shutil.copyfile(self.config_path, checkpoint_path)


class HDC_Logger(object):
    def __init__(self, args, start_epoch=0, topk=50):
        log_dir = osp.join(args.log_dir, args.dataset, time.strftime('%Y%m%d-%H%M%S', time.localtime()))
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        self.log_path = osp.join(log_dir,  'HDC-log.txt')
        self.topk = topk
        self.ith_epoch = start_epoch
        self.iter_records = []

        with open(self.log_path, 'w') as f:
            f.write('')

    def new_epoch(self):
        self.ith_epoch += 1
        with open(self.log_path, 'a') as f:
            f.write('Epoch {}:\n'.format(self.ith_epoch))
            # f.write('    tolerance: {}\n'.format(tol))

    # def save(self, iter_record):
    #     self.iter_records.append(iter_record)

    def record(self, iter, shift):
        with open(self.log_path, 'a') as f:
            f.write('    iter{} shift: {}\n'.format(iter, shift))
            # for i, iter_record in enumerate(self.iter_records):
            #     num_relocate_repeat, count_in_clusters = iter_record
            #     f.write('    iter{}: (repeat {} relocate) | top{} {}'
            #             .format(i, num_relocate_repeat, self.topk, sorted(count_in_clusters)[:self.topk]))
            #     f.write('    {}\n'.format(np.argsort(count_in_clusters)[:self.topk].tolist()))
