#!/usr/bin/env bash
source scripts/env_set.sh
python3.7.5 ./main.py recognition\
       -c config/st_gcn/kinetics-skeleton/train.yaml\
       --device 0\
       --batch_size 64\
       --test_batch_size 64\
       --use_gpu_npu npu\
       --amp True\
       --num_worker $(nproc)\
       --train_feeder_args data_path=\'./data/Kinetics/kinetics-skeleton/train_data.npy\'\
       --train_feeder_args label_path=\'./data/Kinetics/kinetics-skeleton/train_label.pkl\'\
       --test_feeder_args data_path=\'./data/Kinetics/kinetics-skeleton/val_data.npy\'\
       --test_feeder_args label_path=\'./data/Kinetics/kinetics-skeleton/val_label.pkl\'\
       --num_epoch 50\