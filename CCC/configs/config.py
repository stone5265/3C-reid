# --coding:utf-8--
import argparse
import os.path as osp


def config(dbscan=False, kmeans=False):
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--notes", type=str, default='')

    parser.add_argument("--arch", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet_ibn50a', 'resnet_ibn101a'])
    parser.add_argument("--resnet-pretrained", type=str, default='V1', choices=['V1', 'V2'],
                        help='resnet_ibn has not this option')

    parser.add_argument("--root-dir", type=str, default=f'{osp.expanduser("~")}/Dataset',
                        help='datasets store in {root_dir}')
    parser.add_argument("--dataset", type=str, default='market1501',
                        choices=['market1501', 'msmt17', 'veri'],
                        help='folder name of dataset is defined in "utils/datasets/{dataset}.py"')

    parser.add_argument("--loss-with-camera", action='store_true')

    parser.add_argument("--pooling-type", type=str, default='gem', choices=['avg', 'gem'])

    parser.add_argument("--cm-mode", type=str, default='hd_camera',
                        choices=['cm', 'cm_hard', 'tccl_camera', 'hd_camera'],
                        help='cm = Vanilla,\n'
                             'cm_hard = Hard,\n'
                             'tccl_camera = Hard (TCCL), \n'
                             'hd_camera = Hard (CHD)\n'
                             ' in paper')

    parser.add_argument('--seed', type=int, default=999,
                        help='random seed')

    parser.add_argument("--warmup-with-dbscan", action='store_true')

    parser.add_argument("--warmup-version", type=str, default='v1')
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--warmup-factor", type=float, default=0.1)
    parser.add_argument("--warmup-lr-factor", type=float, default=1., help='v2 ONLY')

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)

    parser.add_argument("--cm-temperature", type=float, default=0.05)
    parser.add_argument("--cm-momentum", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=32)

    parser.add_argument("--no-cam", action='store_true')
    parser.add_argument("--rerank", action='store_true')

    parser.add_argument("--num-clusters", type=int, default=750)
    parser.add_argument("--dbscan-eps", type=float, default=0.6)
    parser.add_argument("--cluster-min-instances", type=int, default=4)
    if not dbscan and not kmeans:
        parser.add_argument("--hdc-camera-version", type=str, default=None)
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--hdc-similarity", type=str, default='Euler Cosine')
        parser.add_argument("--hdc-max-iter", type=int, default=80)
        parser.add_argument("--hdc-init", type=str, default='k-means++2', choices=['random', 'k-means++', 'k-means++2'])
        parser.add_argument("--hdc-normalize", action='store_true')
        parser.add_argument("--hdc-centroids", action='store_true')
    parser.add_argument("--hdc-outlier", action='store_true')

    parser.add_argument("--no-jaccard", action='store_true')
    parser.add_argument("--jaccard-k1", type=int, default=30)
    parser.add_argument("--jaccard-k2", type=int, default=6)

    parser.add_argument("--num-epochs", type=int, default=70)
    parser.add_argument("--eval-start-epoch", type=int, default=0)
    parser.add_argument("--eval-step-epoch", type=int, default=5)
    parser.add_argument("--eval-and-ckpt", action='store_true')

    parser.add_argument("--eval-dist", type=str, default=None, choices=[None, 'Euclidean', 'Cosine'])

    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--lr-weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-steps", type=int, default=20)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=200)

    parser.add_argument("--ckpt-dir", type=str, default=r'ckpt',
                        help='model parameter checkpoint root directory')

    if dbscan:
        parser.add_argument("--log-dir", type=str, default=r'logs/log_dbscan')
    elif kmeans:
        parser.add_argument("--log-dir", type=str, default=r'logs/log_kmeans')
    else:
        parser.add_argument("--log-dir", type=str, default=r'logs/log')

    return parser


if __name__ == '__main__':
    args = config().parse_args()
    for key, value in args._get_kwargs():
        print('{}={}\n'.format(key, value))
    pass
