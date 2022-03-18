source ./test/env_npu.sh
export RANK_SIZE=1
PORT=29500 ./tools/dist_train.sh configs/yolo/yolov3_d53_320_273e_coco.py 1 --cfg-options optimizer.lr=0.001 --seed 0 --local_rank 0
