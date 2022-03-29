python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 moby_linear.py \
--cfg configs/moby_swin_tiny.yaml --data-path /data/imagenet > linear.log 2>&1 &