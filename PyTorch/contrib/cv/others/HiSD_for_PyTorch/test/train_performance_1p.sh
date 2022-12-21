#!/bin/bash

##########################################################
#########第3行 至 90行，请一定不要、不要、不要修改##########
#########第3行 至 90行，请一定不要、不要、不要修改##########
#########第3行 至 90行，请一定不要、不要、不要修改##########
##########################################################
# shell脚本所在路径
cur_path=`echo $(cd $(dirname $0);pwd)`
# 判断当前shell是否是performance
perf_flag=`echo $0 | grep performance | wc -l`


# 当前执行网络的名称
Network="HiSD"

batch_size=8
total_iterations=1000
export RANK_SIZE=1
export RANK_ID=0
export JOB_ID=10087

# 路径参数初始化

output_path="results"

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
	  --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
    "
    exit 1
fi

# 参数校验，不需要修改
#for para in $*
#do
#    if [[ $para == --data_path* ]];then
#        data_path=`echo ${para#*=}`
#    elif [[ $para == --output_path* ]];then
#        output_path=`echo ${para#*=}`
#    elif [[ $para == --train_steps* ]];then
#        train_steps=`echo ${para#*=}`
#	elif [[ $para == --train_epochs* ]];then
#        train_epochs=`echo ${para#*=}`
#    elif [[ $para == --batch_size* ]];then
#        batch_size=`echo ${para#*=}`
#    fi
#done

#
## 校验是否传入data_path,不需要修改
#if [[ $data_path == "" ]];then
#    echo "[Error] para \"data_path\" must be config"
#    exit 1
#fi
#
## 校验是否传入output_path,不需要修改
#if [[ $output_path == "" ]];then
#    output_path="./test/output"
#fi

CaseName="HiSD"


# 跳转到code目录
cd ${cur_path}/../
#rm -rf ./test/output
#mkdir -p ./test/output

# 训练开始时间记录，不需要修改
start_time=$(date +%s)
##########################################################
#########第3行 至 90行，请一定不要、不要、不要修改##########
#########第3行 至 90行，请一定不要、不要、不要修改##########
#########第3行 至 90行，请一定不要、不要、不要修改##########
##########################################################

#=========================================================
#=========================================================
#========训练执行命令，需要根据您的网络进行修改==============
#=========================================================
#=========================================================
# 基础参数，需要模型审视修改
# 您的训练数据集在${data_path}路径下，请直接使用这个变量获取
# 您的训练输出目录在${output_path}路径下，请直接使用这个变量获取
# 您的其他基础参数，可以自定义增加，但是batch_size请保留，并且设置正确的值

# 设置打屏日志文件名，请保留，文件名为${print_log}
cd ${cur_path}/../
print_log="./test/output/train_performance.log"


python3 ./core/train.py --batchsize ${batch_size} --total_iterations ${total_iterations} >${print_log} 2>&1


# 性能相关数据计算
StepTime=`grep "sec/step : " ${print_log} | tail -n 10 | awk '{print $NF}' | awk '{sum+=$1} END {print sum/NR}'`
FPS=`grep "Total FPS = " ${print_log} | tail -n 10 | awk '{print $NF}' | awk '{sum+=$1} END {print sum/NR}'`

###########################################################
#########后面的所有内容请不要修改###########################
#########后面的所有内容请不要修改###########################
#########后面的所有内容请不要修改###########################
###########################################################

# 训练端到端耗时
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


echo "------------------ Final result ------------------"
# 输出性能FPS/单step耗时/端到端耗时
echo "Final Performance images/sec : $FPS"
echo "Final Performance sec/step : $StepTime"
echo "E2E Training Duration sec : $e2e_time"


#关键信息打印到${CaseName}.log中，不需要修改
echo "------------------ Final result ------------------" >> ${print_log}
echo "Network = ${Network}" >> ${print_log}
echo "RankSize = ${RANK_SIZE}" >> ${print_log}
echo "BatchSize = ${batch_size}" >> ${print_log}
echo "DeviceType = `uname -m`" >> ${print_log}
echo "CaseName = ${CaseName}" >> ${print_log}
echo "ActualFPS = ${FPS}" >> ${print_log}
echo "TrainingTime = ${StepTime}" >> ${print_log}
echo "E2ETrainingTime = ${e2e_time}" >> ${print_log}