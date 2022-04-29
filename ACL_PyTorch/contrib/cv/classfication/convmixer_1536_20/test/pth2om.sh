#!/bin/bash

source env.sh

batch_size=1
not_skip_onnx=true

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
    rm -rf convmixer_1536_20.onnx
    python convmixer_pth2onnx.py  \
        --source "./convmixer_1536_20_ks9_p7.pth.tar" \
        --target "./convmixer_1536_20.onnx"
    if [ -f "convmixer_1536_20.onnx" ]; then
      echo "==> 1. creating onnx model successfully."
    else
      echo "onnx export failed"
      exit -1
    fi
fi


# ======================= convert om =========================================
rm -rf convmixer_1536_20_bs${batch_size}.om
atc --framework=5 --model=./convmixer_1536_20.onnx \
    --output=./convmixer_1536_20_bs${batch_size} \
    --input_format=NCHW --input_shape="image:${batch_size},3,224,224" \
    --log=error --soc_version=Ascend710
if [ -f "convmixer_1536_20_bs${batch_size}.om" ] ; then
  echo "==> 2. creating om model successfully."
else
  echo "sim_om export failed"
fi
echo "==> 3. Done."
