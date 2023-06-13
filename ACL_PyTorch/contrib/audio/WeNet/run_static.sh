#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=5 # start from 0 if you need to start from data preparation
stop_stage=5
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0
# data
data=/export/data/asr-data/OpenSLR/33/
data_url=`sed '/^data_url=/!d;s/.*=//' url.ini`

nj=16
feat_dir=raw_wav
dict=data/dict/lang_char.txt

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_conformer_no_pos.yaml: Conformer without relative positional encoding
# 6. conf/train_u2++_conformer.yaml: U2++ conformer
# 7. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=false
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        python3 wenet/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir  \
            --num ${average_num} \
            --val_best
    fi
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.3
    reverse_weight=0.3
    for mode in ${decode_modes}; do
    {
        test_dir=$dir/test_${mode}
        mkdir -p $test_dir
        python3 wenet/bin/static.py --gpu -1 \
            --mode $mode \
            --config $dir/train.yaml \
            --test_data $feat_dir/test/format.data \
            --checkpoint $decode_checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --ctc_weight $ctc_weight \
            --reverse_weight $reverse_weight \
            --result_file $test_dir/text \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
         python3 tools/compute-wer.py --char=1 --v=1 \
            $feat_dir/test/text $test_dir/text > $test_dir/wer
    } &
    done
    wait

fi

