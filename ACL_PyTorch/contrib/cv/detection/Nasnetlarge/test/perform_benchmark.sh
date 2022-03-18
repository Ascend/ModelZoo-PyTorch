
trtexec --onnx=nasnetlarge_sim.onnx --fp16 --shapes=image:1x3x331x331 --threads > nasnetlarge_bs1.log

perf_str=`grep "GPU.* mean.*ms$" nasnetlarge_bs1.log`
if [ -n "$perf_str" ]; then
	perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
	perf_str=`grep "mean.*ms$" nasnetlarge_bs1.log`
	perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "benchmark bs1 fps:%.3f\n", 1000*1/('$perf_num'/1)}'

trtexec --onnx=nasnetlarge_sim.onnx --fp16 --shapes=image:16x3x331x331 --threads > nasnetlarge_bs16.log

perf_str=`grep "GPU.* mean.*ms$" nasnetlarge_bs16.log`
if [ -n "$perf_str" ]; then
	perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
	perf_str=`grep "mean.*ms$" nasnetlarge_bs16.log`
	perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "benchmark bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'

