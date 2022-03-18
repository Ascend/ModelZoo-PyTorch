
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1

python test_npu.py \
	--long_size 2240 \
	--npu 1\
	--resume "/PATH/TO/CONFIGED/PSENet/8p/best/npu8pbatch64lr4_0.3401_0.9416_0.8407_0.9017_521.pth"\
	--data_dir '/PATH/TO/CONFIGED/data/ICDAR/Challenge/' \
	--output_file 'npu8p64r4521'