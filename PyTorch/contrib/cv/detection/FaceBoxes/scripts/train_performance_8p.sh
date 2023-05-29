source scripts/npu_set_env.sh
export HCCL_CONNECT_TIMEOUT=3600
device_id_list=0,1,2,3,4,5,6,7

currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_performance_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3 -u ${currentDir}/train.py \
    --dist_url='tcp://127.0.0.1:40001' \
    --multiprocessing-distributed \
	--print-freq=1 \
    --world_size=1 \
    --training_dataset=${currentDir}/data/WIDER_FACE \
    --device_num=8 \
	--dist-backend='hccl' \
	--device='npu' \
   	--rank=0 \
    --device_list=${device_id_list} \
	--batch_size=256 \
	--lr=8e-3 \
	--max=5 \
	--addr=$(hostname -I |awk '{print $1}') \
	--loss_scale=32 \
    --num_workers=0 > ./8p_train.log 2>&1 &