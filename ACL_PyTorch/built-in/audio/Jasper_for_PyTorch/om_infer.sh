#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python om_infer_acl.py \
        --batch_size 4 \
        --model ./jasper.om \
        --val_manifests /path/to/LibriSpeech/LibriSpeech-test-other-wav.json \
        --model_config=configs/jasper10x5dr_speedp-online_speca.yaml \
        --dataset_dir=/path/to/LibriSpeech \
        --max_duration 4 \
        --pad_to_max_duration \
        --save_predictions ./preds.txt
