#!/bin/bash

source env.sh


rm -rf images
rm -rf vectors

python3.7 GAN_testdata.py --online_path=images --offline_path=vectors --pth_path=generator_8p_0.0008_128.pth --iters 100 --batch_size 1


rm -rf out/

echo "Begin to infer!"

./msame --model "GAN_bs1.om"  --input "./vectors" --output "out" --outfmt TXT > GAN_bs1.log

DIR="out"
if [ "$(ls -A $DIR)" ]; then

  echo "Turn to txt success!"
else
  echo "Turn to txt fail!"
  exit -1
fi

perf_str=`grep "Inference average time : " GAN_bs1.log`
if [ -n "$perf_str" ]; then
    perf_num=`echo $perf_str | awk -F ' ' '{print $5}'`
    echo $perf_num
else
  echo "fail!"
  exit -1
fi

awk 'BEGIN{printf "310 bs1 fps:%.3f\n", 1000*1/('$perf_num'/4)}'



rm -rf genimg

python3.7 GAN_txt2jpg.py --txt_path=out --infer_results_path=genimg

DIR2="genimg"
if [ "$(ls -A $DIR2)" ]; then
  echo "Turn to jpg success!"
  echo "You can see the online_infer result in images and the offline_infer result in genimg .Then you can compare their differences"
else
  echo "Turn to jpg fail!"
  exit -1
fi
