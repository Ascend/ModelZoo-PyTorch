#!/bin/bash

# T4ÐÔÄÜ²âÊÔ
rm -rf perf.log
trtexec --onnx=CGAN.onnx --fp16 --shapes=image:100,72 --threads > perf.log
perf_str=`grep "GPU.* mean.*ms$" perf.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" perf.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 CGAN fps:%.3f\n", 1000*1/('$perf_num'/1)}'
