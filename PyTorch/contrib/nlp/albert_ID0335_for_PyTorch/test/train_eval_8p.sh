encoding=utf-8

source ./test/env_npu.sh

# 数据集路径,保持为空,不需要修改
data_path=""
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

export BERT_BASE_DIR=./prev_trained_model/albert_base_v2
export DATA_DIR=${data_path}
export OUTPUR_DIR=./outputs
export PYTHONPATH=./:$PYTHONPATH
export DEVICE=npu
TASK_NAME="SST-2"
for i in $(seq 7 -1 0)
    do
      python3.7 ./run_classifier.py \
        --device=$DEVICE \
        --model_type=SST \
        --model_name_or_path=$BERT_BASE_DIR/ \
        --task_name=$TASK_NAME \
        --data_dir=$DATA_DIR/$TASK_NAME/ \
        --spm_model_file=$BERT_BASE_DIR/30k-clean.model \
        --output_dir=$OUTPUR_DIR/$TASK_NAME/ \
        --do_eval \
        --do_lower_case \
        --max_seq_length=128 \
        --batch_size=440 \
        --learning_rate=180e-5 \
        --num_train_epochs=7.0 \
        --logging_steps=10 \
        --save_steps=10 \
        --overwrite_output_dir \
        --seed=42 \
        --local_rank=$i \
        --fp16 \
        --fp16_opt_level=O2  &
    done

wait
