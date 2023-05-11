#!/usr/bin/env bash
source npu_set_env.sh

python3 bleu_score.py \
    --workers 40 \
    --dist-url 'tcp://127.0.0.1:50000' \
    --world-size 1 \
    --npu 0 \
    --batch-size 512 \
    --epochs 10 \
    --rank 0 \
    --amp \
    --bleu-npu 0 \
    --ckptpath ./seq2seq-gru-model.pth.tar

