#!/usr/bin/env bash
source npu_set_env.sh
export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1


python dscnn_train_pytorch.py --device_id=0 \
                             --train_data_path=../datasets/train_data \
                             --label_data_path=../datasets/label_data \
                             --batch_size=8 \
                             --epoch=60 \
                             --lr=0.00001 \
                             --model_save_dir=./models \
                             # --pre_trained_model_path=./models/DCNN_bilinear_0226_l1_0.025154941563113792_99.pkl