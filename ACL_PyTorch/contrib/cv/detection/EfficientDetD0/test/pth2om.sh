#!/bin/bash
rm -rf result
mkdir result

batch_size=1
chip_name=310P3

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --chip_name* ]]; then
        chip_name=`echo ${para#*=}`
    fi
done
arch=`uname -m`

python3.7 pth2onnx.py --batch_size=${batch_size} --checkpoint=./model/d0.pth --out=./model/d0.onnx 
python3.7 -m onnxsim --input-shape="${batch_size},3,512,512" --dynamic-input-shape ./model/d0.onnx ./model/d0_sim.onnx --skip-shape-inference
python3.7 modify_onnx.py --model=./model/d0_sim.onnx --out=./model/d0_modify.onnx

atc --framework=5 \
--model=./model/d0_modify.onnx \
--output=./model/d0_bs${batch_size} \
--input_format=NCHW \
--input_shape="x.1:${batch_size},3,512,512" \
--log=debug \
--soc_version=Ascend${chip_name} \
--precision_mode=allow_mix_precision \
--modify_mixlist=ops_info.json 