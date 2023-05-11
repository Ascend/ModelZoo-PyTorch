source scripts/npu_set_env.sh
device_id=0
currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_performance_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3 -u ${currentDir}/train.py \
	--dist_url='tcp://127.0.0.1:123457' \
	--multiprocessing-distributed \
    --training_dataset=${currentDir}/data/WIDER_FACE \
	--print-freq=1 \
    --world_size=1 \
    --device_num=1 \
	--dist-backend="hccl" \
	--batch_size=256 \
	--device='npu' \
   	--rank=0 \
	--max=5 \
    --device_list=${device_id} \
    --num_workers=1 > ./1p_train.log 2>&1 &