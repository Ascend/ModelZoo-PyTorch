# 数据集路径,保持为空,不需要修改
data_path=""

# 参数校验，data_path为必传参数， 其他参数的增删由模型自身决定；此处若新增参数需在上面有定义并赋值；
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#解决mlperf_logging包的调用问题
PYTHONPATH=../../../../../SIAT/benchmarks/resnet/implementations/tensorflow_open_src:$PYTHONPATH

source env.sh

echo "====json===="
chmod -R 777 prepare-json.py
python prepare-json.py --keep-keys ${data_path}/coco/annotations/instances_val2017.json ${data_path}/coco/annotations/bbox_only_instances_val2017.json

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "=======get bin======="
rm -rf ssd_bin

python ssd_preprocess.py --data=${data_path}/coco --bin-output=./ssd_bin

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "=======get info======="
python get_info.py bin ./ssd_bin ssd.info 300 300

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "=======benchmark======="
rm -rf result

chmod u+x benchmark.x86_64

./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./ssd.info -input_width=300 -input_height=300 -useDvpp=False -output_binary=true -om_path=./ssd_bs1.om

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

./benchmark.x86_64 -model_type=vision -batch_size=16 -device_id=1 -input_text_path=./ssd.info -input_width=300 -input_height=300 -useDvpp=False -output_binary=true -om_path=./ssd_bs16.om

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "=======eval bs1======="
python ssd_postprocess.py --data=${data_path}/coco --bin-input=./result/dumpOutput_device0

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "=======eval bs16======="
python ssd_postprocess.py --data=${data_path}/coco --bin-input=./result/dumpOutput_device1

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"