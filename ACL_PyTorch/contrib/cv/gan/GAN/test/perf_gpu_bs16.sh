trtexec --onnx=GAN.onnx --fp16 --shapes=Z:16x100 > GAN_bs16.log
perf_str=`grep "GPU.* mean.*ms$" GAN_bs16.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F' ' '{print $16}'`
else
    perf_str=`grep "mean.*ms$" GAN_bs16.log`
    perf_num=`echo $perf_str | awk -F' ' '{print $4}'`
fi
awk 'BEGIN{printf "t4 bs16 fps:%.3f\n", 1000*1/('$perf_num'/16)}'
