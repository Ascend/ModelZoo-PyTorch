#!/bin/bash

# exec on T4
trtexec --onnx=search.onnx --fp16 --shapes=actual_input_1:1x9x255x255 > siamfc_search_b1.log
perf_str=`grep "GPU.* mean.*ms$" siamfc_search_b1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" siamfc_search_b1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 of search branch fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=exemplar.onnx --fp16 --shapes=actual_input_1:1x3x127x127 > siamfc_exemplar_b1.log
perf_str=`grep "GPU.* mean.*ms$" siamfc_exemplar_b1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" siamfc_exemplar_b1.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs1 of exemplar branch fps:%.3f\n", 1000*1/('$perf_num'/1)}'