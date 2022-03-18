#!/bin/bash

# T4ÐÔÄÜ²âÊÔ
rm -rf perf1.log
trtexec --onnx=fast_scnn_bs1_sim.onnx --fp16 --shapes=image:1x3x1024x2048 --threads > perf1.log
perf_str=`grep "GPU.* mean.*ms$" perf1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

rm -rf perf2.log
trtexec --onnx=fast_scnn_bs4.onnx --fp16 --shapes=image:4x3x1024x2048 --threads > perf2.log
perf_str=`grep "GPU.* mean.*ms$" perf2.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf2.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs4 fps:%.3f\n", 1000*1/('$perf_num'/4)}'

rm -rf perf3.log
trtexec --onnx=fast_scnn_bs8.onnx --fp16 --shapes=image:8x3x1024x2048 --threads > perf3.log
perf_str=`grep "GPU.* mean.*ms$" perf3.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf3.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs8 fps:%.3f\n", 1000*1/('$perf_num'/8)}'

rm -rf perf16.log
trtexec --onnx=fast_scnn_bs16.onnx --fp16 --shapes=image:16x3x1024x2048 --threads > perf4.log
perf_str=`grep "GPU.* mean.*ms$" perf4.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf4.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'

rm -rf perf32.log
trtexec --onnx=fast_scnn_bs32.onnx --fp16 --shapes=image:32x3x1024x2048 --threads > perf5.log
perf_str=`grep "GPU.* mean.*ms$" perf5.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf5.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs32 fps:%.3f\n", 1000*1/('$perf_num'/32)}'
