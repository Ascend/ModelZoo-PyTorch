#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

batch_size=1
datasets_path="/opt/npu/hmdb51"

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
rm -rf ./prep_hmdb51_bs${batch_size}
python posec3d_preprocess.py \
    --batch-size ${batch_size} \
    --data_root ${datasets_path}/rawframes/ \
    --ann_file ./hmdb51.pkl \
    --name ./prep_hmdb51_bs${batch_size}
if [ $? != 0 ]; then
    echo "posec3d preprocess fail!"
    exit -1
fi
echo "==> 1. creating ./prep_hmdb51_bs${batch_size} successfully."

# =============================== msame ======================================
if [ ! -d ./result ]; then
    mkdir ./result
fi
rm -rf ./result/outputs_bs${batch_size}_om
./msame --model "./hmdb51_bs${batch_size}.om" \
        --input "./prep_hmdb51_bs${batch_size}" \
        --output "./result/outputs_bs${batch_size}_om" \
        --outfmt TXT > ./msame_bs${batch_size}.txt
if [ $? != 0 ]; then
    echo "msame bs${batch_size} fail!"
    exit -1
fi
echo "==> 2. conducting hmdb51_bs${batch_size}.om successfully."


# ============================ evaluate ======================================
python postprocess.py \
    --result_path ./result/outputs_bs${batch_size}_om/ \
    --info_path /opt/npu/hmdb51/hmdb51.txt

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "==> 3. evaluating hmda51 on bs${batch_size} successfully."
echo '==> 4. Done.'