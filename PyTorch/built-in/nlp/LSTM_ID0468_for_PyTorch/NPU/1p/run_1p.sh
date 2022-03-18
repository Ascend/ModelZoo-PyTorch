#!/bin/bash

source pt_set_env.sh
#Author: Ruchao Fan
#2017.11.1     Training acoustic model and decode with phoneme-level bigram
#2018.4.30     Replace the h5py with ark and simplify the data_loader.py
#2019.12.20    Update to pytorch1.2 and python3.7

device_id=0

. path.sh
stage=0
timit_dir='../TIMIT'
phoneme_map='60-39'
feat_dir='data'                            #dir to save feature
feat_type='fbank'                          #fbank, mfcc, spectrogram
config_file='conf/ctc_config.yaml'

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
echo "train log path is ${train_log_dir}"


if [ ! -z $1 ]; then
    stage=$1
fi

if [ $stage -le 0 ]; then
    echo "Step 0: Data Preparation ..."
    chmod +x local/timit_data_prep.sh
    local/timit_data_prep.sh $timit_dir $phoneme_map || exit 1;
    python3 steps/get_model_units.py $feat_dir/train/phn_text
fi

if [ $stage -le 1 ]; then
    echo "Step 1: Feature Extraction..."
    chmod +x steps/make_feat.sh
    steps/make_feat.sh $feat_type $feat_dir || exit 1;
fi

if [ $stage -le 2 ]; then
    echo "Step 2: Acoustic Model(CTC) Training..."
    taskset -c 0-95 python3 steps/train_ctc.py \
    --device_id 0 \
    --apex \
    --loss_scale 128 \
    --opt_level O2 \
    --conf 'conf/ctc_config.yaml' > ${train_log_dir}/lstm_1p.log 2>&1 & 
    exit 1 
fi

if [ $stage -le 3 ]; then
    echo "Step 3: LM Model Training..."
    steps/train_lm.sh $feat_dir || exit 1;
fi

if [ $stage -le 4 ]; then
    echo "Step 4: Decoding..."
    python3 steps/test_ctc.py  --use_npu True --device_id $device_id --conf $config_file >> ${train_log_dir}/lstm_1p.log 2>&1 &
    exit 1
fi

