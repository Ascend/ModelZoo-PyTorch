#!/bin/bash

if [ ! -d "./mg_om_dir" ]
then
	mkdir ./mg_om_dir
fi

for i in 4 8 16 32 64
do
	atc --model=mg_onnx_dir/textcnn_${i}bs_mg.onnx --framework=5 --output=mg_om_dir/textcnn_${i}bs_mg --output_type=FP16 --soc_version=Ascend710 --enable_small_channel=1
done
