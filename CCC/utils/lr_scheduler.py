# encoding: utf-8
from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    @author:  liaoxingyu
    @contact: sherlockliao01@gmail.com
    """
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupMultiStepLRv2(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_lr_factor=1.0,   # lr = lr*warmup_lr_factor in warmup last iter
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_lr_factor = warmup_lr_factor
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLRv2, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        lr_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                if not self.last_epoch == 0:
                    alpha = (self.last_epoch+1) / self.warmup_iters
                else:
                    alpha = 0
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            lr_factor = self.warmup_lr_factor
        return [
            base_lr * lr_factor
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


if __name__ == '__main__':
    import torch
    import torch.optim as optim

    w = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
    optimizer = optim.Adam([w], lr=3.5e-4)
    optimizer.param_groups[0]['initial_lr'] = 3.5e-4
    optimizer2 = optim.Adam([w], lr=3.5e-4)
    optimizer2.param_groups[0]['initial_lr'] = 3.5e-4
    scheduler = WarmupMultiStepLR(optimizer,
                                  milestones=[20, 30],
                                  gamma=0.5,
                                  warmup_iters=10,
                                  warmup_factor=0.1,
                                  last_epoch=-1)
    scheduler2 = WarmupMultiStepLRv2(optimizer2,
                                     milestones=[20, 30],
                                     gamma=0.5,
                                     warmup_lr_factor=0.1,
                                     warmup_iters=10,
                                     warmup_factor=0.1,
                                     last_epoch=-1)
    lr = optimizer.param_groups[0]['lr']
    lr2 = optimizer2.param_groups[0]['lr']
    epoch = 1
    for i in range(30):
        optimizer.step()
        scheduler.step()
        optimizer2.step()
        scheduler2.step()
        lr = optimizer.param_groups[0]['lr']
        lr2 = optimizer2.param_groups[0]['lr']
        epoch += 1
        print()
