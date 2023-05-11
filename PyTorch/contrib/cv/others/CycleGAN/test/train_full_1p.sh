export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
rm -rf npu1pbs1.*
rm -rf checkpoints_1pbs1
nohup python3 train.py --pu_ids='0' \
	 --prof=0 \
	 --multiprocessing_distributed=0 \
	 --distributed=1 \
	 --npu=1 \
	 --dataroot=./datasets/maps \
	 --checkpoints_dir=./checkpoints_1pbs1 \
   --batch_size=1 \
   --isapex=True \
   --apex_type="O1"  \
   --loss_scale=dynamic \
   --log_path="npu1pbs1.txt" \
   --num_epoch_start=0  \
   --num_epoch=200 \
   --n_epochs=100 \
   --lr=1e-4   \
   --line_scale=1  \
   --n_epochs=100 \
   --n_epochs_decay=100 \
   --pool_size=50  \
   --lambda_A=10   \
   --lambda_B=10   \
   --loadweight=latest   >>npu1pbs1.log 2>&1 &
tail -f npu1pbs1.log  
     
     
     
     
     