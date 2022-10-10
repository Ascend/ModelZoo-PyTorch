datasets_path="/root/datasets"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./total-text-bin
echo "preprocess"
python TextSnake_preprocess.py --src_path ${datasets_path}/data/total-text/Images/Test --save_path ./total-text-bin
if [ $? != 0 ]; then
    echo "fail!"				        
    exit -1
fi
echo "get_info"
python get_info.py bin ./total-text-bin ./textsnake_prep_bin.info 512 512
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
echo "./benchmark.x86_64 bs1"
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=TextSnake_bs1.om -input_text_path=./textsnake_prep_bin.info -input_width=512 -input_height=512 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "eval acc bs1"
python TextSnake_postprocess.py bs1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "./benchmark.x86_64 bs16"
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=TextSnake_bs16.om -input_text_path=./textsnake_prep_bin.info -input_width=512 -input_height=512 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "eval acc bs16"
python TextSnake_postprocess.py bs16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
