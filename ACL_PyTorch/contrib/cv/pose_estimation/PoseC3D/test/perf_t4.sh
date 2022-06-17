#!/bin/bash

rm -rf perf_bs1.log
trtexec --onnx=posec3d_bs1.onnx --fp16 --threads > perf_bs1.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'
