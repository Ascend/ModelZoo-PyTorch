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


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################不需要修改#################
ASCEND_DEVICE_ID=0



start_time=$(date +%s)
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
cd src
nohup python3  track.py mot --exp_id mot17_dla34  \
            --load_model ${cur_path}/exp/mot/mot17_dla34/model_50.pth \
            --data_cfg '../src/lib/cfg/mot17.json'   \
            --world_size 1 \
            --gpus '0' \
            --batch_size 12 \
            --rank 0 \
            --val_mot17 True \
            --data_dir ${data_path} \
> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_val_${ASCEND_DEVICE_ID}.log 2>&1 & 

wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

Network="FairMOT"
RANK_SIZE=8
batch_size=12
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'


trainAccuracy=`grep "OVERALL" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_val_${ASCEND_DEVICE_ID}.log |  awk '{print $15}'`

echo "---------Final result----------"
echo "Final Train Accuracy : ${train_accuracy}"

sed -i "/.*TrainAccuracy*/c\TrainAccuracy=${trainAccuracy}"  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

