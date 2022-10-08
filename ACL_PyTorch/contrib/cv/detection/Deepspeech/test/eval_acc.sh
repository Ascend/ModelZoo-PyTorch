#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

arch=`uname -m`
currentDir=$(cd "$(dirname "$0")";pwd)/..


python3 deepspeech_preprocess.py --data_file $currentDir/deepspeech.pytorch/data/an4_test_manifest.json --save_path $currentDir/deepspeech.pytorch/data/an4_dataset/test --label_file $currentDir/deepspeech.pytorch/labels.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result
./msame --model "$currentDir/deepspeech_bs1.om" --input "$currentDir/deepspeech.pytorch/data/an4_dataset/test/spect,$currentDir/deepspeech.pytorch/data/an4_dataset/test/sizes" --output "$currentDir/deepspeech.pytorch/result" --outfmt TXT
if [ $? != 0 ]; then
    echo "fail to gen bs1 result!"
    exit -1
fi


python3 deepspeech_postprocess.py --out_path $currentDir/deepspeech.pytorch/result --info_path $currentDir/deepspeech.pytorch/data/an4_dataset/test
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"     