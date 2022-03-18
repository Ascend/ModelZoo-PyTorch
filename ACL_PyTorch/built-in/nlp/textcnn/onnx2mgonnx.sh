#!/bin/bash

if [ ! -d "./mg_onnx_dir" ]
then
	mkdir ./mg_onnx_dir
fi

for i in 4 8 16 32 64
do
	python3 ./fix_onnx.py ${i}
done
