# 数据集路径,保持为空,不需要修改
source test/env_npu.sh
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

ln -sb ${data_path} /root/dataset
cd src

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/$RANK_SIZE))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
WORKERS_NUM=$((KERNEL_NUM - 1))
echo "WORKERS_NUM: $WORKERS_NUM"

nohup taskset -c $PID_START-$PID_END \
      python3.7 -u train_8p.py mot --exp_id mot17_dla34  \
            --load_model '../models/ctdet_coco_dla_2x.pth' \
            --data_cfg '../src/lib/cfg/mot17.json'   \
            --world_size 8 \
            --batch_size 12 \
            --rank $RANK_ID \
            --print_iter 1 \
            --lr 30e-4 \
            --use_npu True \
            --use_amp True \
            --num_epochs 3 \
&

done
cd ..
