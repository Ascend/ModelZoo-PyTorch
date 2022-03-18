#!/usr/bin/env bash
cd '/usr/local/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin'
./trtexec --onnx=GaitSet.onnx --fp16 --shapes=image:1*100*64*44 --threads