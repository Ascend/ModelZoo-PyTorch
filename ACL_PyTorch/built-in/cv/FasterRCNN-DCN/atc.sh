export PATH=/usr/local/python3.7.5/bin:/home/cann5.0.2.alpha003/ascend-toolkit/latest/atc/ccec_compiler/bin:/home/cann5.0.2.alpha003/ascend-toolkit/latest/atc/bin:$PATH
export PYTHONPATH=/home/cann5.0.2.alpha003/ascend-toolkit/latest/atc/python/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=/home/cann5.0.2.alpha003/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/home/cann5.0.2.alpha003/ascend-toolkit/latest/opp
# export DUMP_GE_GRAPH=2
export ASCEND_SLOG_PRINT_TO_STDOUT=1

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=info --soc_version=Ascend310 --out_nodes="Concat_1168:0;Reshape_1170:0"