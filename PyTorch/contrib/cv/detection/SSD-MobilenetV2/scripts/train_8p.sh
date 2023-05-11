source scripts/set_npu_env.sh
export RANK_SIZE=8

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        kernel_num=$(($(nproc) / $RANK_SIZE))
        pid_start=$((kernel_num * rank))
        pid_end=$((pid_start + kernel_num - 1))

        taskset -c $pid_start-$pid_end python3 -u train_ssd.py \
          --dataset_type voc  \
          --data_path /opt/npu/voc \
          --net mb2-ssd-lite \
          --base_net models/mb2-imagenet-71_8.pth  \
          --scheduler cosine \
          --lr 0.08 \
          --batch_size 32 \
          --t_max 200 \
          --validation_epochs 5 \
          --checkpoint_folder models/8p  \
          --eval_dir  models/8p/eval \
          --num_epochs 200  \
          --debug_steps 1 \
          --amp \
          --distributed \
          --rank $RANK_ID  \
          --warm_up \
          --warm_up_epochs 5 \
          --stay_lr 1 \
          --device_list '0,1,2,3,4,5,6,7' \
          --dist_backend 'hccl' \
          --device npu  > models/8p/log.txt 2>&1 &
    else
        python3 -u train_ssd.py \
          --dataset_type voc  \
          --data_path /opt/npu/voc \
          --net mb2-ssd-lite \
          --base_net models/mb2-imagenet-71_8.pth  \
          --scheduler cosine \
          --lr 0.08 \
          --batch_size 32 \
          --t_max 200 \
          --validation_epochs 5 \
          --checkpoint_folder models/8p  \
          --eval_dir  models/8p/eval \
          --num_epochs 200  \
          --debug_steps 1 \
          --amp \
          --distributed \
          --rank $RANK_ID  \
          --warm_up \
          --warm_up_epochs 5 \
          --stay_lr 1 \
          --device_list '0,1,2,3,4,5,6,7' \
          --dist_backend 'hccl' \
          --device npu > models/8p/log.txt 2>&1 &
    fi
done
