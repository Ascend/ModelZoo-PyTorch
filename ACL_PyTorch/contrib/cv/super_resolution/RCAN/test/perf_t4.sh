#!/bin/bash

# T4上执行：
trtexec --onnx=rcan.onnx --fp16 --shapes=image:1x3x256x256 > rcan_1bs.log
perf_str=`grep "GPU.* mean.*ms$" rcan_1bs.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" rcan_1bs.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'


