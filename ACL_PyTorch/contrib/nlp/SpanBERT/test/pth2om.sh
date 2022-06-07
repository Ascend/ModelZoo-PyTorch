#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

batch_size=1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done

# ======================= convert onnx =======================================

rm -rf spanBert_dynamicbs.onnx
python spanBert_bin2onnx.py  \
    --config_file ./model_dir/squad1/bert_config.json  \
    --bin_file ./model_dir/squad1/pytorch_model.bin
if [ -f "spanBert_dynamicbs.onnx" ]; then
  echo "==> 1. creating onnx model successfully."
else
  echo "onnx export failed"
  exit -1
fi
# ======================= convert om =========================================
rm -rf spanBert_bs${batch_size}.om
atc --framework=5 --model=./spanBert_dynamicbs.onnx \
    --output=./spanBert_bs${batch_size} \
    --input_format=ND --input_shape="input_ids:1,512;token_type_ids:1,512;attention_mask:1,512" \
    --log=error --soc_version=Ascend710
if [ -f "spanBert_bs${batch_size}.om" ] ; then
  echo "==> 2. creating om model successfully."
else
  echo "sim_om export failed"
fi
echo "==> 3. Done."