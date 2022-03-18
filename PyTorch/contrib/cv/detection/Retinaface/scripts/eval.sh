#!/usr/bin/env bash
source scripts/env.sh
weights=`find -name 'Resnet50*' | xargs ls -t | head -1`
python3.7 test_widerface.py -m $weights &
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
wait
cd widerface_evaluate
python3.7 evaluation.py > eval_result.txt &
wait
cat eval_result.txt
