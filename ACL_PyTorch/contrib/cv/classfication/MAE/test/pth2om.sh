#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

batch_size=1
not_skip_onnx=true
chip_name==$3

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --not_skip_onnx* ]]; then
        not_skip_onnx=`echo ${para#*=}`
    fi
done

# ======================= convert onnx =======================================
if [ $not_skip_onnx == true ]; then
    rm -rf mae_dynamicbs.onnx
    python MAE_pth2onnx.py  \
        --source "./mae_finetuned_vit_base.pth" \
        --target "./mae_dynamicbs.onnx"
    if [ -f "./mae_dynamicbs.onnx" ]; then
      echo "==> 1. creating onnx model successfully."
    else
      echo "onnx export failed"
      exit -1
    fi
fi

# ======================= convert om =========================================
rm -rf mae_bs${batch_size}.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=mae_dynamicbs.onnx --output=mae_bs${batch_size} --input_format=NCHW --input_shape="image:${batch_size},3,224,224" --log=debug --soc_version=$3 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance --enable_small_channel=1
if [ -f "mae_bs${batch_size}.om" ] ; then
  echo "==> 2. creating om model successfully."
else
  echo "om export failed"
fi
echo "==> 3. Done."
