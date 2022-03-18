#!/bin/bash

# T4上执行
trtexec --onnx=deeplabv3_sim_bs1.onnx --fp16 --shapes=input:1,3,1024,2048 \
--threads --workspace=10000 > deeplabv3_bs1.log
perf_str=`grep "GPU.* mean.*ms$" deeplabv3_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" deeplabv3_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'