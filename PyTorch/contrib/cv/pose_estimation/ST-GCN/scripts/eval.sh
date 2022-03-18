#!/usr/bin/env bash
source scripts/env_set.sh
python3.7.5 ./main.py recognition\
       -c config/st_gcn/kinetics-skeleton/test.yaml\
       --weights ./work_dir/recognition/kinetics_skeleton/ST_GCN/best_model_8p.pt\
       --test_feeder_args data_path=\'./data/Kinetics/kinetics-skeleton/val_data.npy\'\
       --test_feeder_args label_path=\'./data/Kinetics/kinetics-skeleton/val_label.pkl\'\
       --use_gpu_npu npu\
       --amp True\
       --device 0\