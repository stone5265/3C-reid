# Credit to https://github.com/alibaba/cluster-contrast-reid/blob/main/clustercontrast/models/cm.py
import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


class CM_Camera(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cameras, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cameras)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cameras = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        target_camera2features = collections.defaultdict(list)
        for feature, target, camera in zip(inputs, targets.tolist(), cameras.tolist()):
            target_camera2features[f'{target}_{camera}'].append(feature)

        for y_c, features in target_camera2features.items():
            y = int(y_c.split('_')[0])
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * features[0]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        target2features = collections.defaultdict(list)
        for feature, target in zip(inputs, targets.tolist()):
            target2features[target].append(feature)

        for y, features in target2features.items():
            features = torch.stack(features, dim=0)
            similarities = features.mm(ctx.features[y].unsqueeze(1))
            ptr = similarities.argmin()

            ctx.features[y] = ctx.features[y] * ctx.momentum + (1 - ctx.momentum) * features[ptr]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


class CM_Hard_Camera(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cameras, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cameras)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cameras = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        target_camera2features = collections.defaultdict(list)
        for feature, target, camera in zip(inputs, targets.tolist(), cameras.tolist()):
            target_camera2features[f'{target}_{camera}'].append(feature)

        target2features = collections.defaultdict(list)
        for y_c, features in target_camera2features.items():
            y = int(y_c.split('_')[0])
            features = torch.stack(features, dim=0)
            similarities = features.mm(ctx.features[y].unsqueeze(1))
            ptr = similarities.argmin()
            target2features[y].append(features[ptr])

        for y, features in target2features.items():
            for feature in features:
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * feature
                ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


class TCCL(autograd.Function):
    """(NOT official) The implementation of Time-based Camera Contrastive Learning (TCCL) of
    '[TCSVT2023] Camera Contrast Learning for Unsupervised Person Re-Identification' by ZMX.
    """

    @staticmethod
    def forward(ctx, inputs, targets, cameras, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cameras)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cameras = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        target_camera2features = collections.defaultdict(list)
        for feature, target, camera in zip(inputs, targets.tolist(), cameras.tolist()):
            target_camera2features[f'{target}_{camera}'].append(feature)

        target2camera_proxies = collections.defaultdict(list)
        for y_c, features in target_camera2features.items():
            y = int(y_c.split('_')[0])
            features = torch.stack(features, dim=0)
            camera_center = features.mean(0)
            camera_center /= camera_center.norm()
            target2camera_proxies[y].append(camera_center)

        for y, proxies in target2camera_proxies.items():
            proxies = torch.stack(proxies, dim=0)
            similarities = proxies.mm(ctx.features[y].unsqueeze(1))
            ptr = similarities.argmin()
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * proxies[ptr]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


class HD_Camera(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cameras, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cameras)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cameras = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        target_camera2features = collections.defaultdict(list)
        for feature, target, camera in zip(inputs, targets.tolist(), cameras.tolist()):
            target_camera2features[f'{target}_{camera}'].append(feature)

        target2camera_proxies = collections.defaultdict(list)
        for y_c, features in target_camera2features.items():
            y = int(y_c.split('_')[0])
            features = torch.stack(features, dim=0)
            camera_center = features.mean(0, keepdim=True)
            camera_center /= camera_center.norm()

            # xc_sim = (features.mm(camera_center.t()) + 1) / 2
            # xz_dis = (1 - features.mm(ctx.features[y].unsqueeze(1))) / 2
            # harmonic_discrepancy = 2 * xz_dis * xc_sim / (xz_dis + xc_sim)
            xc_sim = features.mm(camera_center.t()) + 1
            xz_dis = 1 - features.mm(ctx.features[y].unsqueeze(1))
            harmonic_discrepancy = xz_dis * xc_sim / (xz_dis + xc_sim)
            ptr = harmonic_discrepancy.argmax()
            target2camera_proxies[y].append((features[ptr], harmonic_discrepancy[ptr]))

        for y, proxies in target2camera_proxies.items():
            proxies.sort(key=lambda x: x[1], reverse=True)
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * proxies[0][0]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


__cm_factory = {
    'cm': CM,
    'cm_hard': CM_Hard,
    'cm_camera': CM_Camera,
    'cm_hard_camera': CM_Hard_Camera,
    'tccl_camera': TCCL,
    'hd_camera': HD_Camera
}


def cm(inputs, indexes, cameras, features, momentum=0.5, mode='cm'):
    if 'camera' in mode:
        return __cm_factory[mode].apply(inputs, indexes, cameras, features, torch.Tensor([momentum]).to(inputs.device))
    else:
        return __cm_factory[mode].apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def _complex(X, alpha):
    return 1 / np.sqrt(2) * torch.complex(torch.cos(alpha * torch.pi * X), torch.sin(alpha * torch.pi * X))


def euler_cosine_distances(X, Y, alpha=1.9):
    X_ = _complex(X, alpha)
    Y_ = _complex(Y, alpha)
    D = torch.sum(X_ ** 2, dim=1).unsqueeze(1) + torch.sum(Y_ ** 2, dim=1) - 2 * torch.mm(X_, Y_.t())
    return torch.abs(D)


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='cm'):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.mode = mode

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, cameras, reweight=False):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = cm(inputs, targets, cameras, self.features, self.momentum, self.mode)

        outputs /= self.temp
        if not reweight:
            loss = F.cross_entropy(outputs, targets)
        else:
            label_idx = list(set(targets.tolist()))

            targets = torch.eye(self.features.shape[0]).cuda()[targets]
            cameras = torch.eye(cameras.max()+1).cuda()[cameras]
            cluster_camera = targets.t().mm(cameras)
            prob = cluster_camera / torch.clamp(cluster_camera.sum(1, keepdim=True), min=1e-30)
            entropy = (-prob * torch.log2(torch.clamp(prob, min=1e-30))).sum(1)

            entropy_ = torch.exp(entropy)
            entropy_ = entropy_[label_idx]
            cluster_weight = len(label_idx) * entropy_ / entropy_.sum()
            weight = targets[:, label_idx].mm(cluster_weight.unsqueeze(1)).squeeze()
            loss = (weight * F.cross_entropy(outputs, targets, reduction='none')).mean()
        return loss


if __name__ == '__main__':
    def test():
        import pdb
        B = 200
        C = 750
        P = 16
        model = ClusterMemory(1024, C, mode='cm').cuda()
        centroids = F.normalize(torch.rand(C, 1024, dtype=torch.float32).cuda())
        X = F.normalize(torch.rand(B, 1024, dtype=torch.float32, requires_grad=True).cuda())
        target = torch.randint(0, P, [B,], dtype=torch.long).cuda()
        camera = torch.randint(0, 6, [B,], dtype=torch.long).cuda()

        model.features = centroids
        loss = model(X, target, camera)
        pdb.set_trace()
        loss.backward()
        pass
    test()
