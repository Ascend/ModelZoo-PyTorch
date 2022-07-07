#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

batch_size=1
datasets_path="/opt/npu/imageNet"

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
rm -rf ./prep_image_bs${batch_size}
python MAE_preprocess.py \
    --image-path ${datasets_path}/val \
    --prep-image ./prep_dataset_bs${batch_size} \
    --batch-size ${batch_size}
if [ $? != 0 ]; then
    echo "convmixer preprocess fail!"
    exit -1
fi
echo "==> 1. creating ./prep_image_bs${batch_size} successfully."

# =============================== msame ======================================

if [ ! -d ./result ]; then
    mkdir ./result
fi
rm -rf ./result/outputs_bs${batch_size}_om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "==> conducting. Please wait a moment!"
./msame --model ./mae_bs${batch_size}.om  --output ./result/outputs_bs${batch_size}_om --outfmt TXT --input ./prep_dataset_bs${batch_size} > msame_bs${batch_size}.txt
if [ $? != 0 ]; then
    echo "msame bs${batch_size} fail!"
    exit -1
fi
echo "==> 2. conducting mae_bs${batch_size}.om successfully."


# ============================ evaluate ======================================
python3 MAE_postprocess.py \
    --folder-davinci-target ./result/outputs_bs${batch_size}_om/ \
    --annotation-file-path ${datasets_path}/val_label.txt \
    --result-json-path ./result \
    --json-file-name result_bs${batch_size}.json \
    --batch-size ${batch_size}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "==> 3. evaluating mae on bs${batch_size} successfully."


# =========================== print performance data =========================
echo "====performance data===="
python test/parse.py --result-file ./msame_bs${batch_size}.txt --batch-size ${batch_size}
if [ $? != 0 ]; then
    echo "parse bs${batch_size} performance fail!"
    exit -1
fi


# =========================== print accuracy data ============================
echo "====accuracy data===="
python test/parse.py --result-file ./result/result_bs${batch_size}.json
if [ $? != 0 ]; then
    echo "parse bs${batch_size} accuracy fail!"
    exit -1
fi


echo '==> 4. printing performance and accuracy data successfully.'
echo '==> 5. Done.'
