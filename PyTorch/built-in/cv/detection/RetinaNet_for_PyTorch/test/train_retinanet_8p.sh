#source pt_set_env.sh
rm -rf kernel_meta/
/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
#export HCCL_WHITELIST_DISABLE=1
#export SCALAR_TO_HOST_MEM=1
#export RANK_SIZE=8
PORT=29500 ./tools/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 8 --cfg-options optimizer.lr=0.04 --seed 0 --gpu-ids 0 --no-validate --opt-level O1

#KERNEL_NUM=$(($(nproc)/8))
#for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
#do
#    export RANK=$RANK_ID

#    if [ $(uname -m) = "aarch64" ]
#    then
#       PID_START=$((KERNEL_NUM * RANK_ID))
#        PID_END=$((PID_START + KERNEL_NUM - 1))
 #       taskset -c $PID_START-$PID_END python3.7 ./tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py \
 #           --launcher pytorch \
 #           --cfg-options \
  #          optimizer.lr=0.038 \
  #          --seed 0 \
  #          --gpu-ids 0 \
  #          --opt-level O1 &
  #  else
  #      python3.7 ./tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py \
   #         --launcher pytorch \
   #         --cfg-options \
   #         optimizer.lr=0.038 \
   #         --seed 0 \
    #        --gpu-ids 0 \
    #        --opt-level O1 &
   # fi
#done