source UFLD/test/env_npu.sh&&nohup python -u UFLD/train.py UFLD/configs/tusimple.py --epoch=10 --finetune=UFLD/UFLD_LOG/model_par/ep099.pth>> fintune.log &