#!/bin/sh

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_WHITELIST_DISABLE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export PTCOPY_ENABLE=1

################ Basic Training Settings ##################
# "Must Have" Settings: Network batch_size RANK_SIZE
# Network Name, the same with dir
Network="VideoPose3D"
# training batch_size per GPU
batch_size=8192
# num of NPUs
export RANK_SIZE=8
# train epochs
train_epochs=80

############# Specify Training Directory #############
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

############# Create Log output directory ##############
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi 

#################### Start Training  #################
# start time, no modification needed
start_time=$(date +%s)

python run.py \
    -e ${train_epochs} \
    -k cpn_ft_h36m_dbb \
    -arc 3,3,3,3,3 \
    -c checkpoint/8p_lr2.2e-3 \
    -o test/output/8p_lr2.2e-3 \
    -lr 0.0022 \
    --log log/8p_lr2.2e-3 \
    --sampler \
    --device-list '0,1,2,3,4,5,6,7' &
wait

################# Gain Training Data ####################
# end training time, no modification needed
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# print results, no modification needed
echo "------------------ Final result ------------------"
# output FPS
FPS=`grep -a 'FPS'  ${test_path_dir}/output/8p_lr2.2e-3/train_rank0.log|awk -F " " '{print $15}'|awk 'END {print}'`
# print
echo "Final Performance images/sec : $FPS"

# train-accuracy
# acc=`grep -a 'Protocol #1'  ${test_path_dir}/output/8p_lr2.2e-3/train_rank0.log|awk 'END {print}'|awk -F " " '{print $7}'`
# print
# echo "Final Train Accuracy (mm) : ${acc:8:5}"
echo "E2E Training Duration sec : $e2e_time"

# Performance Summary
# Train-related information, no modification needed
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# Key information print
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log