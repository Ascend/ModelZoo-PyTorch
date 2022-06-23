#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

h_list=(240 480 720 1080)
w_list=(320 640 1280 1920)
length=${#h_list[@]}

mkdir -p logs
for ((i=0; i<${length}; i++));
do
    h=${h_list[$i]}
    w=${w_list[$i]}
    ./msame --model $1 --input inputs/${h}_${w}.bin --dymShape "image:1,3,${h},${w}" --loop 20 &> logs/${h}_${w}.log
    time=`cat logs/${h}_${w}.log | grep 'Inference average time without' | sed 's#^.* .*without first time: ##' | sed 's# ms##'`
    fps=$(echo "scale=4; 1000/${time}*1"|bc)
    echo "H:${h},W:${w}: time: ${time} fps: $fps"
done
