#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

batch_size=1
not_skip_onnx=true
chip_name=310P3

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --not_skip_onnx* ]]; then
        not_skip_onnx=`echo ${para#*=}`
    fi
    if [[ $para == --chip_name* ]]; then
        chip_name=`echo ${para#*=}`
    fi
done

# ======================= convert onnx =======================================
if [ $not_skip_onnx == true ]; then
    rm -rf posec3d.onnx
    python ./posec3d_pytorch2onnx.py  \
        ./mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py \
        ./slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth \
        --shape ${batch_size} 20 17 48 56 56 \
        --verify \
        --output-file ./posec3d_bs${batch_size}.onnx
    if [ -f "posec3d_bs${batch_size}.onnx" ]; then
      echo "==> 1. creating onnx model successfully."
    else
      echo "onnx export failed"
      exit -1
    fi
fi


# ======================= convert om =========================================
rm -rf posec3d_bs${batch_size}.om
export TUNE_BANK_PATH="./aoe_result_bs1"
atc --framework=5 --model=./posec3d_bs${batch_size}.onnx \
    --output=./posec3d_bs${batch_size} \
    --input_format=ND --input_shape="invals:${batch_size},20,17,48,56,56" \
    --log=debug --soc_version=Ascend${chip_name}
if [ -f "posec3d_bs${batch_size}.om" ] ; then
  echo "==> 2. creating om model successfully."
else
  echo "sim_om export failed"
fi
echo "==> 3. Done."