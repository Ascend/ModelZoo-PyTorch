#!/bin/bash

# T4上执行
trtexec --onnx=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx --fp16 --shapes=input:1,3,500,500 --threads > pspnet_r50-d8_512x512_20k_voc12aug_sim_bs1.log
perf_str=`grep "GPU.* mean.*ms$" pspnet_r50-d8_512x512_20k_voc12aug_sim_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" pspnet_r50-d8_512x512_20k_voc12aug_sim_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx --fp16 --shapes=input:16,3,500,500 --threads > pspnet_r50-d8_512x512_20k_voc12aug_sim_bs16.log
perf_str=`grep "GPU.* mean.*ms$" pspnet_r50-d8_512x512_20k_voc12aug_sim_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" pspnet_r50-d8_512x512_20k_voc12aug_sim_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'