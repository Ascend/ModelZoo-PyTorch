source UFLD/test/env_npu.sh&&nohup python -u -m torch.distributed.launch --nproc_per_node 8 UFLD/train.py UFLD/configs/tusimple.py --batch_size=128 --epoch=10 --learning_rate=16e-4>> 8p_performance.log &