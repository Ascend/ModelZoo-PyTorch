source ./test/env.sh

if [ $(uname -m) = "aarch64" ]
then
  KERNEL_NUM=$(($(nproc)/8))
	for i in $(seq 0 7)
	do 
  PID_START=$((KERNEL_NUM * i))
  PID_END=$((PID_START + KERNEL_NUM - 1))
  taskset -c $PID_START-$PID_END python3.7 -u run_classification_criteo_dcn.py \
	--npu_id $i \
	--device_num 8 \
	--trainval_path='path/to/criteo_trainval.txt' \
	--test_path='path/to/criteo_test.txt' \
	--dist \
	--lr=0.0006 \
	--use_fp16 &
	done
else
   for i in $(seq 0 7)
   do
   python3.7 -u run_classification_criteo_dcn.py \
   --npu_id $i \
   --device_num 8 \
   --trainval_path='path/to/criteo_trainval.txt' \
   --test_path='path/to/criteo_test.txt' \
   --dist \
   --lr=0.0006 \
   --use_fp16 &
   done
fi