#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

batch_size=1
datasets_path="/opt/npu/squad1"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done
# ======================= generate prep_dataset ==============================
rm -rf ./input_ids
rm -rf ./input_mask
rm -rf ./segment_ids
python spanBert_preprocess.py \
    --dev_file ${datasets_path}/dev-v1.1.json \
    --batch_size ${batch_size}
if [ $? != 0 ]; then
    echo "spanBert_preprocess fail!"
    exit -1
fi
echo "==> 1. spanBert preprocess successfully."
# =============================== msame ======================================
if [ ! -d ./result ]; then
    mkdir ./result
fi
rm -rf ./result/outputs_bs${batch_size}_om
./msame --model "./spanBert_bs${batch_size}.om" \
        --input "./input_ids,./segment_ids,./input_mask" \
        --output "./result/outputs_bs${batch_size}_om" \
        --outfmt BIN > ./msame_bs${batch_size}.txt
if [ $? != 0 ]; then
    echo "msame bs${batch_size} fail!"
    exit -1
fi
echo "==> 2. conducting spanBert_bs${batch_size}.om successfully."
# ============================ evaluate ======================================
python spanBert_postprocess.py \
	--do_eval \
	--model spanbert-base-cased \
	--dev_file ${datasets_path}/dev-v1.1.json \
	--max_seq_length 512 \
	--doc_stride 128 \
	--eval_metric f1 \
	--fp16 \
	--bin_dir ./result/outputs_bs${batch_size}_om \
    --eval_batch_size ${batch_size}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "==> 3. evaluating spanBert on bs${batch_size} successfully."
