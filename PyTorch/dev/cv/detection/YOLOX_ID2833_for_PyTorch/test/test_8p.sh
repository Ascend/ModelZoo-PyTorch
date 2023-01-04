#!/bin/bash

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

# 数据集路径,保持为空,不需要修改
data_path=""
pth_path=""
# 进行验证的Device个数
nnodes=8

#网络名称,同目录名称,需要模型审视修改
Network="YOLOX_ID2833_for_PyTorch"


#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    fi
    if [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
      #echo "PATH TRAIN BEFORE: $PATH"
      #source set_conda.sh --conda_name=$conda_name
      source ${test_path_dir}/set_conda1.sh
      source activate $conda_name
      #echo "PATH TRAIN AFTER: $PATH"
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

if [[ $pth_path == "" ]];then
    echo "[Error] para \"pth_path\" must be confing"
    exit 1
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi


#进入训练脚本目录，需要模型审视修改
cd $cur_path

chmod +x ${cur_path}/tools/dist_test.sh

#runtime 2.0 enable
export ENABLE_RUNTIME_V2=1
echo "Runtime 2.0 $ENABLE_RUNTIME_V2"

sed -i "s|data/coco/|$data_path/|g" configs/yolox/yolox_s_8x8_300e_coco.py

#训练开始时间，不需要修改
start_time=$(date +%s)
#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
PORT=29500 ./tools/dist_test.sh configs/yolox/yolox_m_8x8_300e_coco.py \
    $pth_path \
    $nnodes \
    --eval bbox > ${test_path_dir}/test_8p.log 2>&1 &
wait

sed -i "s|$data_path/|data/coco/|g" configs/yolox/yolox_s_8x8_300e_coco.py

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Average Precision' ${test_path_dir}/test_8p.log|awk "NR==1"|awk -F ' ' '{print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改

DeviceType=`uname -m`
CaseName=${Network}_${DeviceType}_${nnodes}'p'_'acc'

currentTime=`date "+%Y-%m-%d %H:%M:%S"`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Date_time = ${currentTime}" >> $test_path_dir/${CaseName}.log
echo "Network = ${Network}" >> $test_path_dir/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/${CaseName}.log
echo "PTH = ${pth_path}" >> $test_path_dir/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $test_path_dir/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/${CaseName}.log
#退出anaconda环境
if [ -n "$conda_name" ];then
    echo "conda $conda_name deactivate"
    conda deactivate
fi
