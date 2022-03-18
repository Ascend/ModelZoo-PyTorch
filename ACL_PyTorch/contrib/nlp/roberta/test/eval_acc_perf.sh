# !/bin/bash

DEVICE=1
DATAPATH=./data/SST-2-bin
PADLENGTH=70
OUTPUT=outputs


if [[ $1 = 'benchmark' ]];then
    echo [INFO] Benchmark begin.
    for i in 1 4 8 16 32
        do
            echo [INFO] Benchmark on batch size $i
            ./benchmark.x86_64 -device_id=$DEVICE -om_path=./${OUTPUT}/roberta_base_batch_${i}.om -round=30 -batch_size=$i
        done
elif [[ $1 = 'msame' ]];then
    echo [INFO] Msame begin.
    for i in 1 16
        do 
            echo [INFO] Preprocessing data on batch size $i
            python3.7 RoBERTa_preprocess.py --data_path $DATAPATH --data_kind valid --batch_size $i --pad_length $PADLENGTH
            echo [INFO] Evaluating on batch size $i
            ./msame --model ./${OUTPUT}/roberta_base_batch_${i}.om --input ./${DATAPATH}/batch_size_${i}/roberta_base_bin --output ./result --device $DEVICE --outfmt BIN > msame_res_bs${i}_device${DEVICE}.log
            python3.7 RoBERTa_postprocess.py --batch_size=${i} --device=$DEVICE --data_path=$DATAPATH
        done
fi