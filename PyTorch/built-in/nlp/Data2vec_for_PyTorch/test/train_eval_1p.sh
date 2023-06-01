#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#数据集路径,保持为空,不需要修改
data_path=""
#模型权重路径
checkpoint_path=""

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 校验是否传入checkpoint_path,不需要修改
if [[ $checkpoint_path == "" ]];then
    echo "[Error] para \"checkpoint_path\" must be confing"
    exit 1
fi

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

#################启动训练脚本#################

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

python3 fairseq_cli/hydra_validate.py -m \
    --config-dir examples/data2vec/config/text/pretraining \
    --config-name base \
    task.data=$data_path \
    common.user_dir=examples/data2vec \
    common_eval.path=$checkpoint_path \
    common.cpu=true \
    common.fp16=false

