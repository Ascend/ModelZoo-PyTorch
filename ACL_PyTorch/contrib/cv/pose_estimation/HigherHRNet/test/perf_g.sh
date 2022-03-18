#! /bin/bash

trtexec --onnx=pose_higher_hrnet_w32_512_bs1_dynamic.onnx --fp16 --shapes=image:1x3x1024x512 > 1024x512_bs1.log
perf_str=`grep "GPU.* mean.*ms$" 1024x512_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" 1024x512_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=pose_higher_hrnet_w32_512_bs1_dynamic.onnx --fp16 --shapes=image:1x3x512x512 > 512x512_bs1.log
perf_str=`grep "GPU.* mean.*ms$" 512x512_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" 512x512_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'