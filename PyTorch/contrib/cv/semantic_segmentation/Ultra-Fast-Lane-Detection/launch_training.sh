export NPU_VISIBLE_DEVICES=0,1,2,3,5,6,7
export NGPUS=8
export OMP_NUM_THREADS=4 # you can change this value according to your number of cpu cores


python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/culane.py
# python train.py configs/tusimple.py
