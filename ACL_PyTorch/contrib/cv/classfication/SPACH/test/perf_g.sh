export OM_MODEL=$1 
#! /bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# inference with TensorRT
./msame --model ${OM_MODEL} --output "./output/" --outfmt BIN --loop 100
