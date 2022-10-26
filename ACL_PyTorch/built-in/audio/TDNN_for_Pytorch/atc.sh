#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

model=$1
bs=`echo  ${model} | tr -cd "[0-9]" `

if [ `echo $model | grep "mod"` ]
then
   atc --model=$model --framework=5 --input_format=ND --input_shape="feats:${bs},-1,23;random:${bs},1500" --dynamic_dims='200;300;400;500;600;700;800;900;1000;1100;1200;1300;1400;1500;1600;1700;1800' --output=./tdnn_bs${bs}_mods --soc_version=$2 --log=error
else
   atc --model=$model --framework=5 --input_format=ND --input_shape="feats:${bs},-1,23" --dynamic_dims='200;300;400;500;600;700;800;900;1000;1100;1200;1300;1400;1500;1600;1700;1800' --output=./tdnn_bs${bs}s --soc_version=$2 --log=error
fi

