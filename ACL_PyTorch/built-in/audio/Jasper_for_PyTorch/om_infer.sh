#!/bin/bash

export install_path="/usr/local/Ascend/ascend-toolkit/latest"
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

python om_infer_acl.py \
        --batch_size 4 \
        --model ./jasper.om \
        --val_manifests /path/to/LibriSpeech/LibriSpeech-test-other-wav.json \
        --model_config=configs/jasper10x5dr_speedp-online_speca.yaml \
        --dataset_dir=/path/to/LibriSpeech \
        --max_duration 4 \
        --pad_to_max_duration \
        --save_predictions ./preds.txt
