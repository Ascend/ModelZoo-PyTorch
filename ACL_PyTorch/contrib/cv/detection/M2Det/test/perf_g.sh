#!/bin/bash
rm -rf m2det512_bs1.log m2det512_bs16.log
# T4ִУ
trtexec --onnx=m2det512.onnx --fp16 --shapes=image:1x3x512x512 --threads > m2det512_bs1.log

perf_str=`grep "GPU.* mean.*ms$" m2det512_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" m2det512_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=m2det512.onnx --fp16 --shapes=image:16x3x512x512 --threads > m2det512_bs16.log

perf_str=`grep "GPU.* mean.*ms$" m2det512_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" m2det512_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
