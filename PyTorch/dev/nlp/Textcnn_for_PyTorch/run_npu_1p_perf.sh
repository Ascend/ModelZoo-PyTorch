#!/bin/bash
rm -rf ./tensorboard/textcnn/*
# set env

export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/python3.7.5/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/fwkacllib/python/site-packages:/usr/local/Ascend/atc/python/site-packages:/usr/local/python3.7.5/lib/python3.7/site-packages/
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin

export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.71.T5.0.B060
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910

#export PRINT_MODEL=0
#export DISABLE_REUSE_MEMORY=1

# for fast training
#unset DUMP_OP
#unset PRINT_MODEL
#unset DUMP_GE_GRAPH
export DISABLE_REUSE_MEMORY=0
export SLOG_PRINT_TO_STDOUT=0
#export DUMP_OP=1
#export DISABLE_REUSE_MEMORY=1
#export PROFILING_MODE=true
#export PROFILING_OPTIONS=training_trace:task_trace
# for fast training
#unset DUMP_OP
#unset PRINT_MODEL
#unset DUMP_GE_GRAPH
#export DISABLE_REUSE_MEMORY=0
export TF_CPP_MIN_LOG_LEVEL=1
export EXPERIMENTAL_DYNAMIC_PARTITION=1
export ASCEND_GLOBAL_LOG_LEVEL=1
/usr/local/Ascend/driver/tools/msnpureport -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
export RANK_ID=0
export RANK_SIZE=1
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID

python3 ./run_cnn_modify.py train


