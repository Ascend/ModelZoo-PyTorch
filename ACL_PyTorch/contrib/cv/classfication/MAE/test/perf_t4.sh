#!/bin/bash


rm -rf perf_bs1.log
trtexec --onnx=./mae_dynamicbs.onnx --shapes=image:1x3x224x224 --fp16 --threads > perf_bs1.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
echo "==> mae bs=1."
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'



rm -rf perf_bs8.log
trtexec --onnx=./mae_dynamicbs.onnx --shapes=image:8x3x224x224 --fp16 --threads > perf_bs8.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs4.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs8.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
echo "==> mae bs=8."
awk 'BEGIN{printf "t4 bs8 fps:%.3f\n", 1000*1/('$perf_num'/1)}'





