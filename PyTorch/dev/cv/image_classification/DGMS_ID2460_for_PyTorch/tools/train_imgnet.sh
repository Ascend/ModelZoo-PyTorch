DATASET="--train-dir /mount/data/ImageNet/train/ --val-dir /mount/data/ImageNet/val/ -d imagenet --num-classes 1000"
GENERAL="--lr 2e-5 --batch-size 256 --test-batch-size 256 --epochs 60 --workers 4 --base-size 256 --crop-size 224"
INFO="--checkname resnet18_4bit --lr-scheduler one-cycle"
MODEL="--network resnet18 --K 16 --weight-decay 5e-4"
PARAMS="--tau 0.01"
NORMAL="--normal"
PRETRAINED="--pretrained --rt --show-info"
#DEVICES="0,1"
GPU="--gpu-ids 0,1"
#CUDA_VISIBLE_DEVICES=$DEVICES 
python3 main.py $GPU $DATASET $GENERAL $MODEL $INFO $PARAMS $PRETRAINED