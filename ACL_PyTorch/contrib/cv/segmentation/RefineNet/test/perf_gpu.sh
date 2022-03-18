batch=$1
echo "batch size: $batch"
trtexec --onnx=RefineNet_910.onnx --fp16 --shapes=input:${batch}x3x500x500 > RefineNet_910_bs${batch}.log
perf_str=`grep "GPU.* mean.*ms$" RefineNet_910_bs${batch}.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" RefineNet_910_bs${batch}.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs'$batch' fps:%.3f\n", 1000*1/('$perf_num'/'$batch')}'