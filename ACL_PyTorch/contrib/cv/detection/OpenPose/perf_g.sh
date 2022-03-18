#!/bin/bash

# T4上执行：
trtexec --onnx=human-pose-estimation.onnx --fp16 --shapes=data:1x3x368x640 > OpenPose_b1.log
perf_str=`grep "GPU.* mean.*ms$" OpenPose_b1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" OpenPose_b1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=human-pose-estimation.onnx --fp16 --shapes=data:16x3x368x640 > OpenPose_b16.log
perf_str=`grep "GPU.* mean.*ms$" OpenPose_b16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" OpenPose_b16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
