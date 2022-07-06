#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# generate prep_dataset
rm -rf ./prep_noise_bs1 ./prep_label_bs1 gen_y_bs1.npz
python3.7 biggan_preprocess.py --batch-size 1 --num-inputs 50000
if [ $? != 0 ]; then
    echo "bs1 preprocess fail!"
    exit -1
fi
echo '==> 1. creating ./prep_noise_bs1 ./prep_label_bs1 successfully.'

rm -rf ./prep_noise_bs16 ./prep_label_bs16 gen_y_bs16.npz
python3.7 biggan_preprocess.py --batch-size 16 --num-inputs 50000
if [ $? != 0 ]; then
    echo "bs16 preprocess fail!"
    exit -1
fi
echo '==> 2. creating ./prep_noise_bs16 ./prep_label_bs16 successfully.'

# msame bs1
rm -rf ./outputs_bs1_om
./main --model "./biggan_sim_bs1.om" \
        --input "./prep_noise_bs1,./prep_label_bs1" \
        --output "./outputs_bs1_om" \
        --outfmt BIN > ./msame_bs1.txt
if [ $? != 0 ]; then
    echo "msame bs1 fail!"
    exit -1
fi
echo '==> 3. conducting biggan_sim_bs1.om successfully.'

# msame bs16
rm -rf ./outputs_bs16_om
./main --model "./biggan_sim_bs16.om" \
        --input "./prep_noise_bs16,./prep_label_bs16" \
        --output "./outputs_bs16_om" \
        --outfmt BIN > ./msame_bs16.txt
if [ $? != 0 ]; then
    echo "msame bs16 fail!"
    exit -1
fi
echo '==> 4. conducting biggan_sim_bs16.om successfully.'

# print performance data
echo "====performance data===="
python3.7 test/parse.py --txt-file "./msame_bs1.txt" --batch-size 1 > bs1_perf.log
if [ $? != 0 ]; then
    echo "parse bs1 fail!"
    exit -1
fi

python3.7 test/parse.py --txt-file "./msame_bs16.txt" --batch-size 16 > bs16_perf.log
if [ $? != 0 ]; then
    echo "parse bs16 fail!"
    exit -1
fi

echo '==> 5. printing performance data successfully.'
echo '==> 6. Done.'
