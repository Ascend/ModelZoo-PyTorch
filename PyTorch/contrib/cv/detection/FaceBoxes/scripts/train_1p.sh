source scripts/npu_set_env.sh
device_id=0
currentDir=$(cd "$(dirname "$0")";pwd)/..
cd ${currentDir}
echo "train log path is ${currentDir}/1p_train.log"

python3 -u ${currentDir}/train.py \
	--dist_url='tcp://127.0.0.1:123456' \
	--multiprocessing-distributed \
	--print-freq=1 \
    --world_size=1 \
    --device_num=1 \
	--dist-backend="hccl" \
	--batch_size=256 \
	--device='npu' \
   	--rank=0 \
    --device_list=${device_id} \
    --num_workers=1 > ${currentDir}/1p_train.log 2>&1 &
