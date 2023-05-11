#!/bin/bash
cur_path=`pwd`
export ASCEND_SLOG_PRINT_TO_STDOUT=0
ls /npu/traindata/coco_txl >1.txt
ls /npu/traindata/coco_txt/images >2.txt
ls /npu/traindata/coco_txl/images/train2017 >3.txt
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="YOLOV4_ID0396_for_PyTorch"
# 训练batch_size
batch_size=1024
# 训练使用的npu卡数
export RANK_SIZE=32
# 数据集路径,保持为空,不需要修改
data_path=""
conf_path=""
server_index=""
fix_node_ip=""
devicesnum=""
one_node_ip=""
linux_num=""

# 训练epoch
train_epochs=1
# 图片大小
image_size=608
# 指定训练所使用的npu device卡id
device_id=0

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --fix_node_ip* ]];then
	    fix_node_ip=`echo ${para#*=}`
	elif [[ $para == --devicesnum* ]];then
	    devicesnum=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --one_node_ip* ]];then
        one_node_ip=`echo ${para#*=}`
    elif [[ $para == --linux_num* ]];then
        linux_num=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

if [[ $conf_path == "" ]];then
    one_node_ip=$one_node_ip
    linux_num=$linux_num
else 
    one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
    linux_num=`find $conf_path -name "server_*.info" |wc -l`
fi

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi


#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

if [ -d $data_path/../coco_txl/COCO2017/images/train2017/000000000009.jpg ];then
        echo "NO NEED UNTAR"
else
    mkdir -p $data_path/../coco_txl
        tar -zxvf $data_path/COCO2017.tar.gz -C  $data_path/../coco_txl/
rm -rf $data_path/../coco_txl/COCO2017/labels/*.cache
fi
wait

sed -i "s|./coco/train2017.txt|$data_path/../coco_txl/COCO2017/train2017.txt|g" data/coco.yaml
sed -i "s|./coco/val2017.txt|$data_path/../coco_txl/COCO2017/val2017.txt|g" data/coco.yaml
sed -i "s|./coco/testdev2017.txt|$data_path/../coco_txl/COCO2017/testdev2017.txt|g" data/coco.yaml
sed -i "s|./coco/annotations/instances_val|$data_path/../coco_txl/COCO2017/annotations/instances_val|g" test.py
sed -i "s|opt.notest or final_epoch:|opt.notest:|g" main.py

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip
export MASTER_PORT=29501
export HCCL_WHITELIST_DISABLE=1
device_num=${#devicesnum}
devices_num=`awk 'BEGIN{printf "%.0f\n",'${device_num}'-1}'`

NPUS=($(seq 0 $devices_num))
rank_server=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${server_index}'}'`
export NPU_WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${linux_num}'}'`

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
rank=0
KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 7)
do
    export NPU_CALCULATE_DEVICE=$i
    rankid=`awk 'BEGIN{printf "%.0f\n",'${rank}'+'${rank_server}'}'`
    if [ $(uname -m) = "aarch64" ]
    then
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3 main.py --img $image_size $image_size \
                                          --data coco.yaml \
                                          --cfg cfg/yolov4_8p.cfg \
                                          --weights '' \
                                          --name yolov4 \
                                          --batch-size ${batch_size} \
                                          --epochs=${train_epochs} \
                                          --amp \
                                          --opt-level O1 \
                                          --loss_scale 128 \
                                          --multiprocessing_distributed \
                                          --device 'npu' \
                                          --global_rank $rankid \
                                          --device_list 0,1,2,3,4,5,6,7 \
                                          --world_size ${linux_num} \
                                          --addr $one_node_ip \
                                          --dist_url 'tcp://127.0.0.1:41111' \
                                          --dist_backend 'hccl' \
                                          --stop_step_num 100 \
                                          --notest > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    else
        python3 main.py --img $image_size $image_size \
                   --data coco.yaml \
                   --cfg cfg/yolov4_8p.cfg \
                   --weights '' \
                   --name yolov4 \
                   --batch-size ${batch_size} \
                   --epochs=${train_epochs} \
                   --amp \
                   --opt-level O1 \
                   --loss_scale 128 \
                   --multiprocessing_distributed \
                   --device 'npu' \
                   --global_rank $rankid \
                   --device_list 0,1,2,3,4,5,6,7 \
                   --world_size ${linux_num} \
                   --addr $one_node_ip \
                   --dist_url 'tcp://127.0.0.1:41111' \
                   --dist_backend 'hccl' \
                   --stop_step_num 100 \
                   --notest > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
    let rank++
done


wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#参数复原
sed -i "s|$data_path/../coco_txl/COCO2017/train2017.txt|./coco/train2017.txt|g" data/coco.yaml
sed -i "s|$data_path/../coco_txl/COCO2017/val2017.txt|./coco/val2017.txt|g" data/coco.yaml
sed -i "s|$data_path/../coco_txl/COCO2017/testdev2017.txt|./coco/testdev2017.txt|g" data/coco.yaml
sed -i "s|$data_path/../coco_txl/COCO2017/annotations/instances_val|./coco/annotations/instances_val|g" test.py
sed -i "s|opt.notest:|opt.notest or final_epoch:|g" main.py

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $4}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tr '\r' '\n'|grep "${image_size}:"|awk -F " " '{print $6}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
