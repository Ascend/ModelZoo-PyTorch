# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================





# Readme
# 1. if the checkpoint is trained from mutil NPU/GPUs, use the option --from_multiprocessing_distributed
# 2. if the checkpoint is wanted to be tested through mutil NPU/GPUs, use the option --multiprocessing_distributed

################基础配置参数##################

arch_network="RCAN" # 网络结构选择，可以不用指定，默认即可
test_dataset_dir="/root/dataset_zzq/0_Set5/" # 测试集路径
outputs_dir="/root/RCAN_8p_npu_test/" # 输出保存路径
checkpoint_path="/root/RCAN_8p_gpu/X2/model_best_amp.pth" # 需要测试的模型
device="npu"
amp="--amp" # 是否使用amp进行训练，可以不用指定，默认即可
scale=2  # 超分辨率放大倍数，可以不用指定，默认即可
device_id=0
from_multiprocessing_distributed="" # 测试模型是否训练于多卡，使用后可以将多卡模型加载于单卡上进行测试


################接收外部输入配置参数##################
for para in $*
do
    if [[ $para == --arch_network* ]];then
        arch_network=`echo ${para#*=}`
    elif [[ $para == --test_dataset_dir* ]];then
        test_dataset_dir=`echo ${para#*=}`
    elif [[ $para == --outputs_dir* ]];then
        outputs_dir=`echo ${para#*=}`
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=`echo ${para#*=}`
    elif [[ $para == --device* ]];then
        device=`echo ${para#*=}`
    elif [[ $para == --scale* ]];then
        scale=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --from_multiprocessing_distributed ]];then
        from_multiprocessing_distributed="--from_multiprocessing_distributed"
    fi
done

#################创建日志输出目录，不需要修改#################
if [ -d ${outputs_dir}/X${scale}/ ];then
    log_path=${outputs_dir}/X${scale}/test.log
else
    mkdir -p ${outputs_dir}/X${scale}/
    log_path=${outputs_dir}/X${scale}/test.log
fi

#################激活环境，修改环境变量#################
if [ ${device} == "npu" ];then
    check_etp_flag=`env | grep etp_running_flag`
    etp_flag=`echo ${check_etp_flag#*=}`
    if [ x"${etp_flag}" != x"true" ];then
        source env_npu.sh
    fi
    export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
else
    source activate pt-1.5
fi


#################启动训练脚本#################

nohup python3 -u ../test.py  --arch ${arch_network} \
                --test_dataset_dir ${test_dataset_dir} \
                --outputs_dir ${outputs_dir} \
                --checkpoint_path ${checkpoint_path} \
                --scale ${scale} \
                --device ${device} \
                ${amp} \
                --device_id ${device_id} \
                ${from_multiprocessing_distributed} >> ${log_path} 2>&1 & 
wait

exit








