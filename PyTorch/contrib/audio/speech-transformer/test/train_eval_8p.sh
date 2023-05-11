#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Speech-Transformer"
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""
do_delta=false
dumpdir=dump   # directory to dump full features

source path.sh

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
# if [[ $data_path == "" ]];then
#     echo "[Error] para \"data_path\" must be confing"
#     exit 1
# fi

feat_test_dir=${dumpdir}/test/delta${do_delta};
dict=data/lang_1char/train_chars.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo $test_path_dir

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

export WORLD_SIZE=$RANK_SIZE
expdir=${test_path_dir}/output
echo "Decoding"
decode_dir=${expdir}/${ASCEND_DEVICE_ID}

echo ${feat_test_dir}/data.json
echo ${dict}
echo ${decode_dir}/data.json
python3 \
    ../src/bin/recognize.py \
    --recog-json ${feat_test_dir}/data.json \
    --dict $dict \
    --result-label ${decode_dir}/data.json \
    --model-path ./output/final.pth.tar \
    >> ${decode_dir}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
# Compute CER
local/score.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))