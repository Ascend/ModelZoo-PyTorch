#!/bin/bash

#网络名称，同目录名称
Network="MT5_ID4146_for_PyTorch"
batch_size=4
model_path=""
output_dir="./tst-translation"
# 数据集路径,保持为空,不需要修改
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --model_path* ]];then
        model_path=`echo ${para#*=}`
    elif [[ $para == --output_dir* ]];then
        output_dir=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source set_conda.sh --conda_name=$conda_name
        source activate $conda_name
    fi
done

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
	test_path_dir=${cur_path}
	cd ..
	cur_path=$(pwd)
else
	test_path_dir=${cur_path}/test
fi

ASCEND_DEVICE_ID=0

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ]; then
	rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
	source ${test_path_dir}/env_npu.sh
else
	model_path=${data_path}/mt5-small/
	cd ./transformers/
	pip3 install -e .
	cd ..
	mkdir /root/.cache/huggingface
	ln -s ${data_path}/wmt16/datasets /root/.cache/huggingface/
	ln -s ${data_path}/wmt16/modules /root/.cache/huggingface/
fi

python3 -m torch.distributed.launch --nproc_per_node 8 run_translation.py \
    --model_name_or_path $model_path \
    --do_train \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir $output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --pad_to_max_length \
    --fp16 \
    --use_combine_grad True \
    --optim adamw_apex_fused_npu \
    --use_combine_ddp True \
    --half_precision_backend apex \
    --download_max_retries 1 \
    --max_step 600 \
    --save_step 5000 >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=$(grep "it/s" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -n 2 | awk "NR==1 {print}" | awk -F "100%" '{print $NF}' | awk -F "]" '{print $1}' | awk -F ", " '{print $2}' | sed 's/^[ \t]*//g')
FPS=`echo "$FPS" | awk '{printf "%.2f\n",$1*8}'`
#打印，不需要修改
echo "Final Performance iter/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "'loss':" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "'loss':" '{print $2}'| awk -F "," '{print $1}' | sed 's/^[ \t]*//g' >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS" = ${FPS} >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
