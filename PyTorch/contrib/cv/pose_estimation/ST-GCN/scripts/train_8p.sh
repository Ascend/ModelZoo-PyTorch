#!/usr/bin/env bash
source scripts/env_set.sh
python3.7.5 ./main.py recognition\
         -c config/st_gcn/kinetics-skeleton/train.yaml\
         --device 0 1 2 3 4 5 6 7\
         --batch_size 64\
         --test_batch_size 64\
         --base_lr 0.8\
         --use_gpu_npu npu\
         --amp True\
         --num_worker 0\
         --train_feeder_args data_path=\'./data/Kinetics/kinetics-skeleton/train_data.npy\'\
         --train_feeder_args label_path=\'./data/Kinetics/kinetics-skeleton/train_label.pkl\'\
         --test_feeder_args data_path=\'./data/Kinetics/kinetics-skeleton/val_data.npy\'\
         --test_feeder_args label_path=\'./data/Kinetics/kinetics-skeleton/val_label.pkl\'\
         --num_epoch 50\ 