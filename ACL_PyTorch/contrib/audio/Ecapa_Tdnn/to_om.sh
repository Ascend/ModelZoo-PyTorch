export ASCEND_DEVICE_ID=0
export TUNE_BANK_PATH=./aoe_result_bs4
export TE_PARALLEL_COMPILER=8
export REPEAT_TUNE=False

#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=0

mkdir om

chmod 777 ./aoe_result_bs4

atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=./om/ecapa_tdnn_bs4 --input_format=ND --input_shape="mel:4,80,200" --log=debug  --soc_version=Ascend710