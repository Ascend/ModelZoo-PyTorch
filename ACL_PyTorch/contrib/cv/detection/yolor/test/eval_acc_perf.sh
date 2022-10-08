
arch=`uname -m`
rm -rf ./val2017_bin
echo "preprocess"
python yolor_preprocess.py --save_path ./val2017_bin --data ./coco.yaml
if [ $? != 0 ]; then
    echo "fail!"				        
    exit -1
fi
echo "get_info"
python get_info.py bin ./val2017_bin ./yolor_prep_bin.info 1344 1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
echo "./benchmark.x86_64 bs1"
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=yolor_bs1.om -input_text_path=./yolor_prep_bin.info -input_width=1344 -input_height=1344 -output_binary=true -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "eval acc bs1"
python yolor_postprocess.py --data ./coco.yaml --img 1280 --batch 1 --conf 0.001 --iou 0.65 --npu 0 --name yolor_p6_val --names ./yolor/data/coco.names
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"