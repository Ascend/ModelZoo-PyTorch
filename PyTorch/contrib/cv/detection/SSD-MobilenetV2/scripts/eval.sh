source scripts/set_npu_env.sh
python3 -u eval_ssd.py \
 --net mb2-ssd-lite  \
 --dataset /opt/npu/voc/VOC2007_test \
 --trained_model "/home/deng/SSD-MobilenetV2/models/8p/mb2-ssd-lite-Epoch-215-Loss-2.405523674718795.pth" \
 --eval_dir  models/1p/eval-215 \
 --device npu \
 --gpu 1 \
 --label_file models/8p/voc-model-labels.txt | tee models/1p/eval-215.txt
