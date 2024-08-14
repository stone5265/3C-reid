import os


__dataset_params_factory = {
    'market1501': ' --dataset market1501 --num-clusters 750 --iters 200 --num-epochs 60 --dbscan-eps 0.6 --hdc-outlier',
    'msmt17': ' --dataset msmt17 --num-clusters 1000 --iters 400 --num-epochs 60 --dbscan-eps 0.6 --hdc-outlier',
    'veri776': ' --dataset veri --num-clusters 1000 --iters 400 --num-epochs 60 --dbscan-eps 0.6 --height 224 --width 224'
}


def repeat(n, dataset, run, cmd):
    if run == '0':
        return
    for _ in range(n):
        os.system(cmd + __dataset_params_factory[dataset])


def ablate1(dataset, switch='11 111 111', n_repeat=1):
    """
    set switch[i] to 0 which don't want to run
        [0]: DBSCAN + Hard(CHD)
        [1]: DBSCAN + Hard(CHD) + reweight
        [2]: K-means + Hard(CHD)
        [3]: K-means + Hard(CHD) + reweight
        [4]: K-means + Hard(CHD) + reweight + DBSCAN_warmup
        [5]: HDC + Hard(CHD)
        [6]: HDC + Hard(CHD) + reweight
        [7]: HDC + Hard(CHD) + reweight + DBSCAN_warmup
    """
    switch = ''.join(switch.split())
    repeat(n_repeat, dataset, switch[0],
           'python CCC/main_dbscan.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --notes "Hard(CHD),[DBSCAN],(NOT)camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation1)"')
    repeat(n_repeat, dataset, switch[1],
           'python CCC/main_dbscan.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --loss-with-camera'
           ' --notes "Hard(CHD),[DBSCAN],camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation1)"')

    repeat(n_repeat, dataset, switch[2],
           'python CCC/main_kmeans.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --notes "Hard(CHD),[KMeans],(NOT)camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation1)"')
    repeat(n_repeat, dataset, switch[3],
           'python CCC/main_kmeans.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --loss-with-camera'
           ' --notes "Hard(CHD),[KMeans],camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation1)"')
    repeat(n_repeat, dataset, switch[4],
           'python CCC/main_kmeans.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "Hard(CHD),[KMeans],camera_entropy_reweight,[dbscan_warmup],GEM"'
           ' --log-dir "logs/log(ablation1)"')

    repeat(n_repeat, dataset, switch[5],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           f' --notes "Hard(CHD),[HDC],(NOT)camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation1)"')
    repeat(n_repeat, dataset, switch[6],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --loss-with-camera'
           f' --notes "Hard(CHD),[HDC],camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation1)"')
    repeat(n_repeat, dataset, switch[7],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "Hard(CHD),[HDC],camera_entropy_reweight,[dbscan_warmup],GEM"'
           ' --log-dir "logs/log(ablation1)"')


def ablate2(dataset, switch='1111 1110', n_repeat=1):
    """
    set switch[i] to 0 which don't want to run
        [0]: Vanilla w/o reweight
        [1]: Hard w/o reweight
        [2]: Hard(TCCL) w/o reweight
        [3]: Hard(CHD) w/o reweight
        [4]: Vanilla
        [5]: Hard
        [6]: Hard(TCCL)
        [7]: Hard(CHD)
    """
    switch = ''.join(switch.split())
    repeat(n_repeat, dataset, switch[0],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode cm'
           ' --warmup-with-dbscan'
           f' --notes "[Vanilla],HDC,(NOT)camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')
    repeat(n_repeat, dataset, switch[1],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode cm_hard'
           ' --warmup-with-dbscan'
           f' --notes "[Hard],HDC,(NOT)camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')
    repeat(n_repeat, dataset, switch[2],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode tccl_camera'
           ' --warmup-with-dbscan'
           f' --notes "[Hard(TCCL)],HDC,(NOT)camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')
    repeat(n_repeat, dataset, switch[3],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --warmup-with-dbscan'
           f' --notes "[Hard(CHD)],HDC,(NOT)camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')

    repeat(n_repeat, dataset, switch[4],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode cm'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[Vanilla],HDC,camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')
    repeat(n_repeat, dataset, switch[5],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode cm_hard'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[Hard],HDC,camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')
    repeat(n_repeat, dataset, switch[6],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode tccl_camera'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[Hard(TCCL)],HDC,camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')
    repeat(n_repeat, dataset, switch[7],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[Hard(CHD)],HDC,camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation2)"')


def ablate3(dataset, switch='111 111 111', n_repeat=1):
    switch = ''.join(switch.split())
    repeat(n_repeat, dataset, switch[0],
           'python CCC/main_dbscan.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --resnet-pretrained V1'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "[pretrainV1],DBSCAN,Hard(CHD),camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation3)"')
    repeat(n_repeat, dataset, switch[1],
           'python CCC/main_dbscan.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --resnet-pretrained V2'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "[pretrainV2],DBSCAN,Hard(CHD),camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation3)"')
    repeat(n_repeat, dataset, switch[2],
           'python CCC/main_dbscan.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --arch resnet_ibn50a'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "[ibn],DBSCAN,Hard(CHD),camera_entropy_reweight,GEM"'
           ' --log-dir "logs/log(ablation3)"')

    repeat(n_repeat, dataset, switch[3],
           'python CCC/main_kmeans.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --resnet-pretrained V1'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "[pretrainV1],KMeans,Hard(CHD),camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation3)"')
    repeat(n_repeat, dataset, switch[4],
           'python CCC/main_kmeans.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --resnet-pretrained V2'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "[pretrainV2],KMeans,Hard(CHD),camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation3)"')
    repeat(n_repeat, dataset, switch[5],
           'python CCC/main_kmeans.py'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --arch resnet_ibn50a'
           ' --loss-with-camera --warmup-with-dbscan'
           ' --notes "[ibn],KMeans,Hard(CHD),camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation3)"')

    repeat(n_repeat, dataset, switch[6],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --resnet-pretrained V1'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[pretrainV1],HDC,Hard(CHD),camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation3)"')
    repeat(n_repeat, dataset, switch[7],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --resnet-pretrained V2'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[pretrainV2],HDC,Hard(CHD),camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation3)"')
    repeat(n_repeat, dataset, switch[8],
           f'python CCC/main.py --hdc-centroids --hdc-init k-means++2'
           ' --pooling-type gem --cm-mode hd_camera'
           ' --arch resnet_ibn50a'
           ' --loss-with-camera --warmup-with-dbscan'
           f' --notes "[ibn],HDC,Hard(CHD),camera_entropy_reweight,dbscan_warmup,GEM"'
           ' --log-dir "logs/log(ablation3)"')


if __name__ == '__main__':
    for dataset in ['market1501', 'msmt17', 'veri776']:
        ablate1(dataset)
        ablate2(dataset)
        ablate3(dataset)
