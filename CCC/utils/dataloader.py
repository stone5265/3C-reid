import torchvision.transforms as T

from torch.utils.data import DataLoader
from .data import Preprocessor, IterLoader
from .sampler import RandomMultipleGallerySamplerNoCam, RandomMultipleGallerySampler


def get_train_loader(args, train_set, num_iters):
    train_transform = T.Compose([
        T.Resize([args.height, args.width]),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # T.RandomErasing(p=0.5, value=[0.485, 0.456, 0.406])
        T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 1/0.3), value=[0.485, 0.456, 0.406])
    ])

    if args.no_cam:
        sampler = RandomMultipleGallerySamplerNoCam
    else:
        sampler = RandomMultipleGallerySampler

    train_loader = IterLoader(
        DataLoader(dataset=Preprocessor(train_set, transform=train_transform),
                   batch_size=args.batch_size,
                   sampler=sampler(train_set, args.num_instances),
                   num_workers=args.num_workers,
                   drop_last=True,
                   pin_memory=True),
        length=num_iters
    )
    return train_loader


def get_test_loader(args, test_set):
    test_transform = T.Compose([
        T.Resize([args.height, args.width]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(dataset=Preprocessor(test_set, transform=test_transform),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)
    return test_loader
