#!/usr/bin/env bash
source scripts/env.sh

export DYNAMIC_COMPILE_ENABLE=1
export EXPERIMENTAL_DYNAMIC_PARTITION=1
export DISABLE_DYNAMIC_PATH=./disable.conf

device_id=0
currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 -u ${currentDir}/train.py \
    --data='data/widerface/train/label.txt' \
    --addr=$(hostname -I |awk '{print $1}') \
    --workers=16 \
    --dist-url='tcp://127.0.0.1:50000' \
    --dist-backend='hccl' \
    --world-size=1 \
    --batch-size=32 \
    --epochs=100 \
    --rank=0 \
    --amp \
    --opt-level='O2' \
    --loss-scale=128. \
    --device-list=${device_id} > ./RetinaFace_1p.log 2>&1 &

wait
if [ ! -d "${currentDir}/outputs" ];then
  mkdir ${currentDir}/outputs
fi
cp -f weights/Resnet50_epoch_100distributed_False ${currentDir}/outputs/Resnet50_final.pth
cd ${currentDir}
python3.7 test_widerface.py -m ${train_log_dir}/weights/Resnet50_epoch_100distributed_False &
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
wait
cd ${currentDir}/widerface_evaluate
if [ -f "bbox.cpython-37m-x86_64-linux-gnu.so" ]; then
    echo 'no need to build'
else     
    python3.7 setup.py build_ext --inplace
fi
python3.7 evaluation.py >> ${train_log_dir}/RetinaFace_1p.log &
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


