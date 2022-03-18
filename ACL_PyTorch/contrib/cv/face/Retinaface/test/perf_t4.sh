#!/bin/bash

trtexec --onnx=retinaface.onnx --fp16 --shapes=image:1x3x1000x1000 --threads > retinaface_bs1.log
perf_str=`grep "GPU.* mean.*ms$" retinaface_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" retinaface_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=retinaface.onnx --fp16 --shapes=image:16x3x1000x1000 --threads > retinaface_bs16.log
perf_str=`grep "GPU.* mean.*ms$" retinaface_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" retinaface_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
