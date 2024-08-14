import os


__dataset_params_factory = {
    'market1501': ' --dataset market1501 --iters 200 --num-epochs 60 --dbscan-eps 0.6 --hdc-outlier',
    'msmt17': ' --dataset msmt17 --iters 400 --num-epochs 60 --dbscan-eps 0.6 --hdc-outlier',
    'veri776': ' --dataset veri --iters 400 --num-epochs 60 --dbscan-eps 0.6 --height 224 --width 224'
}


def repeat(n, dataset, cmd):
    for _ in range(n):
        os.system(cmd + __dataset_params_factory[dataset])


def num_clusters(dataset, value_range, kmeans=False, dbscan_warmup=True, n_repeat=1):
    for num_cluster in value_range:
        repeat(n_repeat, dataset,
               f'python CCC/main{"_kmeans.py" if kmeans else ".py  --hdc-centroids --hdc-init k-means++2"}'
               ' --pooling-type gem --cm-mode hd_camera'
               f' --loss-with-camera {"--warmup-with-dbscan" if dbscan_warmup else ""}'
               f' --num-clusters {num_cluster}'
               f' --notes "[{"KMeans" if kmeans else "HDC"}],num_cluster={num_cluster}{",warmup_w/o_dbscan" if not dbscan_warmup else ""}"'
               ' --log-dir "logs/log(parameter)"')


if __name__ == '__main__':
    num_clusters('market1501', [500, 1000, 1250])
    num_clusters('market1501', [500, 1000, 1250], dbscan_warmup=False)
    num_clusters('market1501', [500, 1000, 1250], kmeans=True)
    num_clusters('market1501', [500, 1000, 1250], kmeans=True, dbscan_warmup=False)

    num_clusters('msmt17', [750, 1250, 1500])
    num_clusters('msmt17', [750, 1250, 1500], dbscan_warmup=False)
    num_clusters('msmt17', [750, 1250, 1500], kmeans=True)
    num_clusters('msmt17', [750, 1250, 1500], kmeans=True, dbscan_warmup=False)

    num_clusters('veri776', [500, 1000, 1250])
    num_clusters('veri776', [500, 1000, 1250], kmeans=True)
    num_clusters('veri776', [500, 1000, 1250], dbscan_warmup=False)
    num_clusters('veri776', [500, 1000, 1250], kmeans=True, dbscan_warmup=False)
