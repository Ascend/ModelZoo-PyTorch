#! /bin/bash

trtexec --onnx=RawNet2_sim_bs1.onnx --fp16 --shapes=wav:1x59049 > RawNet2_bs1.log
perf_str=`grep "GPU.* mean.*ms$" ReID_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" ReID_bs1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=RawNet2_sim_bs16.onnx --fp16 --shapes=wav:16x59049 > RawNet2_bs16.log
perf_str=`grep "GPU.* mean.*ms$" ReID_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" ReID_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "gpu bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
