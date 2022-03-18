#!/bin/bash

rm -rf perf_bs1.log
trtexec --onnx=MSPN.onnx --fp16 --shapes='input:1x3x256x192' –-fp16 --dumpProfile > perf_bs1.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

rm -rf perf_bs16.log
trtexec --onnx=MSPN.onnx --fp16 --shapes='input:16x3x256x192' > perf_bs16.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
