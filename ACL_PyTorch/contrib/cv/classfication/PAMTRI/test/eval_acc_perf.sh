#!/bin/bash

datasets_path="./data/veri"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset_query
rm -rf ./prep_dataset_gallery
python3.7 PAMTRI_preprocess.py

rm ./prep_query_bin.info
rm ./prep_gallery_bin.info
python3.7 gen_dataset_info.py bin ./prep_dataset_query ./prep_query_bin.info 256 256
python3.7 gen_dataset_info.py bin ./prep_dataset_gallery ./prep_gallery_bin.info 256 256
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1_query
rm -rf result/dumpOutput_device0_bs1_gallery
rm -rf result/dumpOutput_device0_bs16_query
rm -rf result/dumpOutput_device0_bs16_gallery

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./PAMTRI_bs1.om -input_text_path=./prep_query_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1_query

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./PAMTRI_bs1.om -input_text_path=./prep_gallery_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1_gallery

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./PAMTRI_bs16.om -input_text_path=./prep_query_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16_query

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./PAMTRI_bs16.om -input_text_path=./prep_gallery_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16_gallery

python3.7 PAMTRI_postprocess.py --queryfeature_path=./result/dumpOutput_device0_bs1_query --galleryfeature_path=./result/dumpOutput_device0_bs1_gallery > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 PAMTRI_postprocess.py --queryfeature_path=./result/dumpOutput_device0_bs16_query --galleryfeature_path=./result/dumpOutput_device0_bs16_gallery > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 ./test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ./test/parse.py result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 ./test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ./test/parse.py result/perf_vision_batchsize_16_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"