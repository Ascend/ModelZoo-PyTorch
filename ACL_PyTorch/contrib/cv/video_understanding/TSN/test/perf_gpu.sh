#!/bin/bash
# GPU上执行：
trtexec --onnx=onnx_sim/tsn_1.onnx --fp16 --shapes=video:1x75x3x224x224 > tsn_1.log
perf_str=`grep "GPU.* mean.*ms$" tsn_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" tsn_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=onnx_sim/tsn_4.onnx --fp16 --shapes=video:4x75x3x224x224 > tsn_4.log
perf_str=`grep "GPU.* mean.*ms$" tsn_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" tsn_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs4 fps:%.3f\n", 1000*1/('$perf_num'/16)}'

trtexec --onnx=onnx_sim/tsn_8.onnx --fp16 --shapes=video:8x75x3x224x224 > tsn_8.log
perf_str=`grep "GPU.* mean.*ms$" tsn_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" tsn_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs8 fps:%.3f\n", 1000*1/('$perf_num'/16)}'