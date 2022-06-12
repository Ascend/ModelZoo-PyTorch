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

# ============================= prepare dataset ==============================
cd ./mmaction2/tools/data/hmdb51
bash download_annotations.sh
bash download_videos.sh
bash extract_rgb_frames_opencv.sh
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
../../../data/hmdb51 /opt/npu
cd ../../../..

# ======================= generate prep_dataset ==============================
rm -rf ./prep_hmdb51_bs${batch_size}
chmod u+x msame
python posec3d_preprocess.py \
    --batch_size ${batch_size} \
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
./msame --model "./posec3d_bs${batch_size}.om" \
        --input "./prep_hmdb51_bs${batch_size}" \
        --output "./result/outputs_bs${batch_size}_om" \
        --outfmt TXT > ./msame_bs${batch_size}.txt
if [ $? != 0 ]; then
    echo "msame bs${batch_size} fail!"
    exit -1
fi
echo "==> 2. conducting hmdb51_bs${batch_size}.om successfully."


# ============================ evaluate ======================================
python posec3d_postprocess.py \
    --result_path ./result/outputs_bs${batch_size}_om/20220612_142755 \
    --info_path ./hmdb51.info

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "==> 3. evaluating hmda51 on bs${batch_size} successfully."
echo '==> 4. Done.'