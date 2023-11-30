#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_10B.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-10b \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --seq-length 512 \
       --max-position-embeddings 1024 \
       --save ./checkpoints \
       --log-interval 1 \
       --eval-interval 1000 \
       --save-interval 2000 \
       --train-iters 200 \
       --train-data pile \
       --resume-dataloader \
       --filter-english \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 175000 \
       --warmup 0.04 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --hidden-dropout 0. \
       --attention-dropout 0. \
       --output-dropout 0. \
"       
#       --fp16 \

gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"