#!/bin/bash
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
    CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
    CANN_INSTALL_PATH="/usr/local/Ascend"
fi

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ]; then
    source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
    source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

export TASK_QUEUE_ENABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=50001
batch_size=256
python3 main.py \
	-a shufflenet_v2_x1_0 \
	--world-size=1 \
	--gpu=0 \
  --rank=0 \
	-j 16 \
	--epochs=3 \
	$1 > train_1p_npu.log

if [ ! -f train_1p_npu.log ]; then
    echo "No log file"
    exit -1
fi
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
Time=`grep -P "Epoch: \[\d+\]" train_1p_npu.log | tail -n 1 | sed -E 's/Time\s+[0-9]+\.[0-9]+\s+\(\s+([0-9]+\.[0-9]+)\).*/\1/' | awk {'print $3'}`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${Time}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

grep -P " *   Acc@1.*Acc@5.*" train_1p_npu.log | tail -n 1

#输出最后一轮验证精度,需要模型审视修改，如需打印最优ACC1，需在训练脚本中修改print逻辑
train_accuracy=`grep -P " *   Acc@1.*Acc@5.*" train_1p_npu.log | tail -n 1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"