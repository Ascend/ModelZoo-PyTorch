bash train.sh 1 --data /opt/npu/imagenet/ --model CSWin_64_12211_tiny_224 -j 8 --no-prefetcher -b 256 --lr 2e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.2

