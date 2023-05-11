export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
rm -rf checkpoints_profiling
rm -rf npu1pbs1_profiling
rm -rf npu1p.prof
nohup python3 train.py --pu_ids='0' \
	 --prof=1 \
   --prof_file="npu1p.prof"  \
	 --multiprocessing_distributed=0 \
	 --distributed=1 \
	 --npu=1 \
	 --dataroot=./datasets/maps \
	 --checkpoints_dir=./checkpoints_profiling \
   --batch_size=1 \
   --isapex=True \
   --apex_type="O1"  \
   --loss_scale=dynamic \
   --log_path="npu8pbs1_profiling.txt" \
   --num_epoch_start=0  \
   --num_epoch=11 \
   --n_epochs=100 \
   --lr=1e-1   \
   --line_scale=1  \
   --n_epochs=100 \
   --n_epochs_decay=100 \
   --pool_size=50  \
   --lambda_A=10   \
   --lambda_B=10   \
   --loadweight=latest   >>npu1pbs1_profiling.log 2>&1 &     \
   tail -f npu1pbs1_profiling.log  
	  #  --continue_train \ # #if want to continue to train it,please no ignore it