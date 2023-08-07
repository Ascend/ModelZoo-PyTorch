#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#集合通信参数,不需要修改
export RANK_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29688
# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="Fairseq_Transformer_wmt18_for_PyTorch"

#训练epoch
train_epochs=20
#训练batch_size,,需要模型审视修改
token_size=4000

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        token_size=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

ASCEND_DEVICE_ID=0
#创建DeviceID输出目录，不需要修改
if [ -d $test_path_dir/output/${ASCEND_DEVICE_ID} ];then
    rm -rf $test_path_dir/output/$ASCEND_DEVICE_ID
    mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
else
    mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

##################创建日志输出目录，根据模型审视##################
# 模型采用非循环方式启动多卡训练，创建日志输出目录如下；采用循环方式启动多卡训练的模型，在循环中创建日志输出目录，可参考CRNN模型
# 非循环方式下8卡训练日志输出路径中的ASCEND_DEVICE_ID默认为0，只是人为指定文件夹名称， 不涉及训练业务
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/*
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

##############执行训练##########
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi


REPLACE_UNICODE_PUNCT=$SCRIPTS/tokenizer/replace-unicode-punctuation.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py
HYP=hyp
REF=ref
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

# 使用NPU融合优化器开关, True为开启, False为关闭.
export NPU_FUSED_MODE=True

# 开启分桶策略，提升多卡场景下的性能
export _DEFAULT_FIRST_BUCKET_BYTES=50

nohup fairseq-train $data_path \
                    --distributed-world-size 8 \
                    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
                    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
                    --dropout 0.3 --weight-decay 0.0001 \
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                    --max-tokens 4000 \
                    --eval-bleu \
                    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                    --eval-bleu-detok moses \
                    --eval-bleu-remove-bpe \
                    --eval-bleu-print-samples \
                    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                    --save-dir checkpoints/transformer_wmt_en_de \
                    --max-epoch 20 \
                    --bucket-cap-mb=300 \
                    > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
WPS=`grep 'train_inner' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "wps=" '{print $NF}'|awk -F "wps" '{print $1}'|awk -F "," '{print $1}'|awk 'END {print}'`
train_wall=`grep 'train_inner' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "train_wall=" '{print $NF}'|awk -F "," '{print $1}'|awk 'NR==1{min=$1;next}{min=min<$1?min:$1}END{print min}'`
TRAIN_WALL=`grep 'train_inner' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "train_wall=" '{print $NF}'|awk -F "," '{print $1}'|tail -n  20|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

#打印，不需要修改
echo "Final Performance words/sec : $WPS"
echo "train_wall : $TRAIN_WALL"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'valid ' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "best_bleu" '{print $NF}'|awk -F "best_bleu" '{print $1}'|awk -F "," '{print $1}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
TokenSize=${token_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${TokenSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualWPS=${WPS}
#单迭代训练时长
TrainingTime=${train_wall}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -r "loss :" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $19}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TokenSize = ${TokenSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualWPS = ${ActualWPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}">> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log