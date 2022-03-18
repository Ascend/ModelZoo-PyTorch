#!/bin/bash

currentDir=$(cd "$(dirname "$0")";pwd)
export SET_RESULT_FILE=$currentDir/set_result.py
export RESULT_FILE=$currentDir/result.txt

function prepare() {
    # download dataset

    # verify  dataset

    # preprocess
    return 0
}

function exec_train() {

    # pytorch lenet5 sample
    #python3.7 $currentDir/pytorch_lenet5_train.py

    # tensorflow-1.15 wide&deep sample
    #python3.7 $currentDir/tensorflow_1_15_wide_deep.py

    # test sample
    cd $currentDir
    bash run_vitbase_1p.sh 
    
    sleep 5
    accuracy="81"
    Accuracy=`grep 'Best Accuracy' vitbase.log | awk '{print $10}'`
    FPS=`grep "s/it" vitbase.log | tail -1 | awk '{print $10}'`

    python3.7 $currentDir/set_result.py training "accuracy" $Accuracy
    python3.7 $currentDir/set_result.py training "throughput_ratio" $FPS
    python3.7 $currentDir/set_result.py training "result" "NOK"
}

function main() {

    prepare

    exec_train

}

main "$@"
ret=$?
exit $?
