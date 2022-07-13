export DATA_SET=$1
export VAL_LABEL=$2
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir prep_dataset
python ./SPACH_preprocess.py --src-path=${DATA_SET} --save-path="./prep_dataset" --batch-size=1

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mkdir output
./msame --model "./spach_ms_conv_s_1.om" --input " ./prep_dataset" --output "./output" --outfmt TXT

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python SPACH_postprocess.py --txt-path="./output/$(ls output)/" --label-path=${VAL_LABEL}

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
