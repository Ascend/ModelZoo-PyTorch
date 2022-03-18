#!/bin/bash

# T4上执行：
trtexec --onnx=vnet.onnx --fp16 --shapes=actual_input_1:1x1x64x80x80 --threads > vnet_bs1.log
perf_str=`grep "GPU.* mean.*ms$" vnet_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" vnet_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=vnet.onnx --fp16 --shapes=actual_input_1:16x1x64x80x80 --threads > vnet_bs16.log
perf_str=`grep "GPU.* mean.*ms$" vnet_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" vnet_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'