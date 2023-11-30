#!/bin/bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

LD_PRELOAD=$LD_PRELOAD:/home/.local/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 WANDB_DISABLED=true python run_mlm.py --model_name_or_path ./checkpoint/reformer-crime-and-punishment --tokenizer_name ./lf_token/ --train_file ./datasets/train_corpus.txt --validation_file ./datasets/test_corpus.txt --debug_file ./datasets/debug_corpus.txt --per_device_train_batch_size 16 --max_seq_length 5120 --learning_rate 5e-5 --num_train_epochs 10 --save_steps 50000 --pad_to_max_length --line_by_line True --do_train --overwrite_output_dir --output_dir ./checkpoint/pretrain_model --debug_mode