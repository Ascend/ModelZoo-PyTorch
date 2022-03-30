source env_npu.sh
if [ ! $1 ];
then
    DATA_DIR=/path/to/imagenet/val
else
    DATA_DIR="$1"
fi
if [ ! $2 ];
then
    MODEL_DIR=/path/to/checkpoint
else
    MODEL_DIR="$2"
fi
python3 validate.py $DATA_DIR  --model lvvit_s --checkpoint $MODEL_DIR/lvvit_s-26m-224-83.3.pth.tar --no-test-pool --amp  -b 64
#python3 validate.py $DATA_DIR  --model lvvit_s --checkpoint $MODEL_DIR/lvvit_s-26m-224-83.3.pth.tar --no-test-pool --amp --img-size 224 -b 64
