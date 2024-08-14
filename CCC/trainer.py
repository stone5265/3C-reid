import numpy as np
import torch

from tqdm import tqdm
from utils.loss import SoftTripletLoss, CrossEntropyLabelSmooth
from utils import AverageMeter


class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        memory,
        # smooth_rate=0.,
        # loss_switch='001',
        # triple_loss_dist='Euclidean',
        use_camera=False
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.cluster_momery = memory.cluster_memory
        self.num_class = memory.num_clusters

        # memory.cluster_memory.epsilon = smooth_rate
        # memory.cluster_memory.num_classes = self.num_class

        # self.use_triplet = bool(int(loss_switch[0]))
        # self.use_id = bool(int(loss_switch[1]))
        # self.use_contrast = bool(int(loss_switch[2]))
        self.use_camera = use_camera

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # self.criterion_gce = CrossEntropyLabelSmooth(self.num_class, smooth_rate).to(self.device)
        # self.criterion_tri = SoftTripletLoss(distance=triple_loss_dist).to(self.device)

        self.loss_list = []

    def train(self, train_loader, title):
        self.model.train()

        train_loss = AverageMeter()
        train_loss_gce = AverageMeter()
        train_loss_tri = AverageMeter()
        train_loss_con = AverageMeter()

        train_tdqm = tqdm(range(len(train_loader)))
        train_tdqm.set_description(title)
        for _ in train_tdqm:
            images, labels, cameras = train_loader.next()
            indexes = None
            if isinstance(labels, list):
                labels, indexes = labels
                indexes = indexes.long().to(self.device)
            images = images.to(self.device)
            labels = labels.long().to(self.device)
            cameras = cameras.long().to(self.device)

            feats = self.model(images)
            # preds = preds[:, :self.num_class]

            # loss_gce = self.criterion_gce(preds, labels) if self.use_id else 0.
            # loss_tri = self.criterion_tri(feats, labels) if self.use_triplet else 0.
            loss_gce = 0.
            loss_tri = 0.

            if indexes is not None:
                loss_con = self.cluster_momery(feats, labels, cameras, reweight=self.use_camera, indexes=indexes)
            else:
                loss_con = self.cluster_momery(feats, labels, cameras, reweight=self.use_camera)

            loss = loss_gce + loss_tri + loss_con

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item())
            # train_loss_gce.update(loss_gce.item()) if self.use_id else train_loss_gce.update(np.nan)
            # train_loss_tri.update(loss_tri.item()) if self.use_triplet else train_loss_tri.update(np.nan)
            # train_loss_con.update(loss_con.item()) if self.use_contrast else train_loss_con.update(np.nan)
            train_loss_gce.update(np.nan)
            train_loss_tri.update(np.nan)
            train_loss_con.update(loss_con.item())

            train_tdqm.set_postfix(loss='{:.4}'.format(train_loss.avg),
                                   _gce='{:.4}'.format(train_loss_gce.avg),
                                   _tri='{:.4}'.format(train_loss_tri.avg),
                                   _con='{:.4}'.format(train_loss_con.avg))

        self.scheduler.step()
        self.loss_list.append((
            ('loss', train_loss.avg),
            ('loss_gce', train_loss_gce.avg),
            ('loss_tri', train_loss_tri.avg),
            ('loss_con', train_loss_con.avg)
        ))

    @property
    def loss(self):
        return self.loss_list[-1]
