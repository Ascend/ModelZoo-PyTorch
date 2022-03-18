#!/bin/bash

atc --model=$1 \
	--framework=5 \
	--output=$2 \
	--input_format=NCHW \
	--input_shape="actual_input_1:1,3,304,304" \
	--enable_small_channel=1 \
	--log=error \
	--soc_version=Ascend310 \
	--insert_op_conf=densenet121_pt_aipp.config
