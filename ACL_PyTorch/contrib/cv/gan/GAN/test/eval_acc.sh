#!/bin/bash

source env.sh
rm -rf images
rm -rf vectors

python3.7 GAN_testdata.py --online_path=images --offline_path=vectors --pth_path=generator_8p_0.0008_128.pth --iters 1 --batch_size 64


rm -rf out/

./msame --model "GAN_bs64.om"  --input "vectors" --output "out" --outfmt TXT > run_msame.log

DIR="out"
if [ "$(ls -A $DIR)" ]; then
  echo "Turn to txt success!"
  echo "Begin to infer!"
else
  echo "Turn to txt fail!"
  exit -1
fi


rm -rf genimg
python3.7 GAN_txt2jpg.py --txt_path=out --infer_results_path=genimg

DIR2="genimg"
if [ "$(ls -A $DIR2)" ]; then
  echo "Turn to jpg success!"
  echo "You can see the online_infer result in images and the offline_infer result in genimg.Then you can compare their differences"
else
  echo "Turn to jpg fail!"
  exit -1
fi
