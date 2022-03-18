# 数据集路径,保持为空,不需要修改
source test/env_npu.sh
data_path=""
cur_path=`pwd`
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
cd src
nohup python3.7  track.py mot --exp_id mot17_dla34  \
            --load_model ${cur_path}/exp/mot/mot17_dla34/model_30.pth \
            --data_cfg '../src/lib/cfg/mot17.json'   \
            --world_size 1 \
            --gpus '0' \
            --batch_size 12 \
            --rank 0 \
            --val_mot17 True \
            --data_dir ${data_path} \
&
cd ..