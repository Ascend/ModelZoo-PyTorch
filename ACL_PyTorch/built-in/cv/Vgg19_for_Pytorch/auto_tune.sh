export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/:/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/schedule_search.egg:/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
# export DUMP_GE_GRAPH=2
export SLOG_PRINT_TO_STDOUT=1
# export REPEAT_TUNE=True

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=info --soc_version=Ascend310

# --input_fp16_nodes="actual_input_1"
# --enable_small_channel=1
# --auto_tune_mode="RL,GA"