source scripts/npu_setenv.sh

RANK_ID_START=0
RANK_SIZE=1

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup \
taskset -c $PID_START-$PID_END python mobilenet.py --data /opt/npu/imagenet \
        -e \
        -b 512 \
        --ngpu 1 \
        --epochs 1 \
        -j $(($(nproc)/8)) \
        --lr 0.8 \
	--rank $RANK_ID \
        1>log.txt \
        2> error.txt &
done