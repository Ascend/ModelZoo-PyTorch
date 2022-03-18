#!/bin/bash
trtexec --onnx=hrnet.onnx --fp16 --shapes=image:1x3x1024x2048 --threads > hrnet_bs1.log
perf_str=`grep "GPU.* mean.*ms$" hrnet_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" hrnet_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=hrnet.onnx --fp16 --shapes=image:4x3x1024x2048 --threads > hrnet_bs4.log
perf_str=`grep "GPU.* mean.*ms$" hrnet_bs4.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" hrnet_bs4.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs4 fps:%.3f\n", 1000*1/('$perf_num'/4)}'
