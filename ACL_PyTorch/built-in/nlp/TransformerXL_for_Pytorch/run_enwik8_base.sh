#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3.7 train.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00000001 \
        --warmup_step 0 \
        --eval-interval 500\
        --max_step 10000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 11 \
        --log-interval 10 \
        --gpu0_bsz 4 \
        --restart \
        --restart_dir workdir0-enwik8/check_point \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3.7 eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 160 \
        --clamp_len 80 \
        --same_length \
        --batch_size 1 \
        --split test \
        ${@:2}
elif [[ $1 == 'om_eval' ]]; then
    echo 'Run evaluation...'
    python3.7 om_eval.py \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 160 \
        --clamp_len 80 \
        --same_length \
        --batch_size 1 \
        --split test \
        ${@:2}
elif [[ $1 == 'onnx' ]]; then
    echo 'Run evaluation...'
    python3.7 eval.py \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 160 \
        --clamp_len 80 \
        --same_length \
        --batch_size 1 \
        --split onnx \
        ${@:2}
else
    echo 'unknown argment 1'
fi
