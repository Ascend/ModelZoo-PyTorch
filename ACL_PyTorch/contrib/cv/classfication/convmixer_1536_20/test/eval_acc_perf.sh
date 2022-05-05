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
python convmixer_preprocess.py \
    --image-path ${datasets_path}/val \
    --prep-image ./prep_image_${batch_size} \
    --batch-size ${batch_size}
if [ $? != 0 ]; then
    echo "convmixer preprocess fail!"
    exit -1
fi
echo "==> 1. creating ./prep_image_bs${batch_size} successfully."

# =============================== msame ======================================
rm -rf ./result/outputs_bs${batch_size}_om
./msame --model "./convmixer_1536_20_bs${batch_size}.om" \
        --input "./prep_image_bs${batch_size}" \
        --output "./result/outputs_bs${batch_size}_om" \
        --outfmt TXT > ./msame_bs${batch_size}.txt
if [ $? != 0 ]; then
    echo "msame bs${batch_size} fail!"
    exit -1
fi
echo "==> 2. conducting convmixer_1536_20_bs${batch_size}.om successfully."


# ============================ evaluate ======================================
python convmixer_eval_acc.py \
    --folder-davinci-target ./result/outputs_bs${batch_size}_om/ \
    --annotation-file-path ${datasets_path}/val_label.txt \
    --result-json-path ./result \
    --json-file-name result_bs${batch_size}.json \
    --batch-size ${batch_size}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "==> 3. evaluating convmixer on bs${batch_size} successfully."


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
