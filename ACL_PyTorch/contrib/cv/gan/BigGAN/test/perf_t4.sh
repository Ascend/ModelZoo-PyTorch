#!/bin/bash

rm -rf perf_bs1.log
trtexec --onnx=biggan_sim_bs1.onnx --fp16 --shapes='noise:1x1x20;label:1x5x148' --threads --dumpProfile > perf_bs1.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

rm -rf perf_bs16.log
trtexec --onnx=biggan_sim_bs16.onnx --fp16 --shapes='noise:16x1x20;label:16x5x148' --threads > perf_bs16.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'

rm -rf perf_bs4.log
trtexec --onnx=biggan_sim_bs4.onnx --fp16 --shapes='noise:4x1x20;label:4x5x148' --threads --dumpProfile > perf_bs4.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs4.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs4.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs4 fps:%.3f\n", 1000*1/('$perf_num'/4)}'

rm -rf perf_bs8.log
trtexec --onnx=biggan_sim_bs8.onnx --fp16 --shapes='noise:8x1x20;label:8x5x148' --threads --dumpProfile > perf_bs8.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs8.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs8.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs8 fps:%.3f\n", 1000*1/('$perf_num'/8)}'

rm -rf perf_bs32.log
trtexec --onnx=biggan_sim_bs32.onnx --fp16 --shapes='noise:32x1x20;label:32x5x148' --threads --dumpProfile > perf_bs32.log
perf_str=`grep "GPU.* mean.*ms$" perf_bs32.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf_bs32.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs32 fps:%.3f\n", 1000*1/('$perf_num'/32)}'

