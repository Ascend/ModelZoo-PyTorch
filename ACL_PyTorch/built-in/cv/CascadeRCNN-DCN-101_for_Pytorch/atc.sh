export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
# export DUMP_GE_GRAPH=2
export ASCEND_SLOG_PRINT_TO_STDOUT=1

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=info --soc_version=Ascend710 --out_nodes="Concat_1036:0;Reshape_1038:0"