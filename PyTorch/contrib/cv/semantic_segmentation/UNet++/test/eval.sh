#!/usr/bin/env python

data_path=$1
resume_path=$2

ASCEND_DEVICE_ID=0
RANK_ID=0

if [ $# != 2 ]
then
    echo "Usage: bash ./test/eval_8p.sh ./inputs/dsb2018_96/ ./models/dsb2018_96_NestedUNet_woDS/model_best.pth.tar"
exit 1
fi

if [ ! -f $resume_path ]
then
    echo "error: resume_path=$resume_path is not a file"
exit 1
fi

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

#进入训练脚本目录，需要模型审视修改
cd $cur_path

#################启动训练脚本#################
# 评估开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ./test/env_npu.sh
fi

nohup  python3 -u train.py \
        --data_path ${data_path} \
        --device npu \
        --batch_size 128 \
        --num_gpus 1 \
        --num_workers 16 \
        --evaluate \
        --resume ${resume_path}  \
        --rank_id $RANK_ID > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval.log 2>&1 &
wait


##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出训练精度,需要模型审视修改
eval_accuracy=`grep -a 'AVG'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval.log|awk -F " " '{print $8}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final eval Accuracy : ${eval_accuracy}"
echo "E2E eval Duration sec : $e2e_time"

