#!/bin/bash
trtexec --onnx=genet_gpu.onnx --fp16 --shapes=image:1x3x32x32 --threads > genet_bs1.log
perf_str=`grep "GPU.* mean.*ms$" genet_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" genet_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=genet_gpu.onnx --fp16 --shapes=image:16x3x32x32 --threads > genet_bs16.log
perf_str=`grep "GPU.* mean.*ms$" genet_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" genet_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
