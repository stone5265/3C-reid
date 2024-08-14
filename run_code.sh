# Market-1501
CUDA_VISIBLE_DEVICES=0,1 python CCC/main.py  --dataset market1501 --height 256 --width 128 \
--num-clusters 750 --iters 200 --num-epochs 60 --warmup-iters 10 \
--warmup-with-dbscan --dbscan-eps 0.6 \
--arch resnet50 --resnet-pretrained V1 --pooling-type gem \
--hdc-outlier --hdc-centroids --hdc-init k-means++2 \
--cm-mode hd_camera --loss-with-camera \
--root-dir $HOME/Dataset --log-dir logs/log

# MSMT17
CUDA_VISIBLE_DEVICES=0,1 python CCC/main.py  --dataset msmt17 --height 256 --width 128 \
--num-clusters 1000 --iters 400 --num-epochs 60 --warmup-iters 10 \
--warmup-with-dbscan --dbscan-eps 0.6 \
--arch resnet50 --resnet-pretrained V1 --pooling-type gem \
--hdc-outlier --hdc-centroids --hdc-init k-means++2 \
--cm-mode hd_camera --loss-with-camera \
--root-dir $HOME/Dataset --log-dir logs/log


# VeRi-776
CUDA_VISIBLE_DEVICES=0,1 python CCC/main.py  --dataset veri --height 224 --width 224 \
--num-clusters 1000 --iters 400 --num-epochs 60 --warmup-iters 10 \
--warmup-with-dbscan --dbscan-eps 0.6 \
--arch resnet50 --resnet-pretrained V1 --pooling-type gem \
--hdc-centroids --hdc-init k-means++2 \
--cm-mode hd_camera --loss-with-camera \
--root-dir $HOME/Dataset --log-dir logs/log
