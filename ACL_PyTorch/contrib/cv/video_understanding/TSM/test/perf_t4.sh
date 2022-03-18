#!/bin/bash
# T4上执行：
trtexec --onnx=onnx_sim/tsm_bs1.onnx --fp16 --shapes=video:1x8x3x224x224 --threads > tsm_bs1.log
perf_str=`grep "GPU.* mean.*ms$" tsm_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" tsm_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=onnx_sim/tsm_bs16.onnx --fp16 --shapes=video:16x8x3x224x224 --threads > tsm_bs16.log
perf_str=`grep "GPU.* mean.*ms$" tsm_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" tsm_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
