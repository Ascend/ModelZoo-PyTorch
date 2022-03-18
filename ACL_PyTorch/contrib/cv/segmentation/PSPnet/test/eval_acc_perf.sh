#!/bin/bash

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./voc12_bin
python3.7 mmsegmentation_voc2012_preprocess.py --image_folder_path=${datasets_path}VOCdevkit/VOC2012/JPEGImages/ --split=${datasets_path}VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --bin_folder_path=./voc12_bin/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin  ./voc12_bin voc12.info 500 500
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
chmod +x ben*
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -om_path=pspnet_r50-d8_512x512_20k_voc12aug_sim_fp16_bs1.om -device_id=0 -batch_size=1 -input_text_path=voc12.info -input_width=500 -input_height=500 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -om_path=pspnet_r50-d8_512x512_20k_voc12aug_sim_fp16_bs16.om -device_id=1 -batch_size=16 -input_text_path=voc12.info -input_width=500 -input_height=500 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py jpg ${datasets_path}VOCdevkit/VOC2012/JPEGImages/ voc12_jpg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 mmsegmentation_voc2012_postprocess.py --bin_data_path=./result/dumpOutput_device0 --test_annotation=./voc12_jpg.info --img_dir=${datasets_path}VOCdevkit/VOC2012/JPEGImages --ann_dir=${datasets_path}VOCdevkit/VOC2012/SegmentationClass --split=${datasets_path}VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --net_input_width=500 --net_input_height=500
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 mmsegmentation_voc2012_postprocess.py --bin_data_path=./result/dumpOutput_device1 --test_annotation=./voc12_jpg.info --img_dir=${datasets_path}VOCdevkit/VOC2012/JPEGImages --ann_dir=${datasets_path}VOCdevkit/VOC2012/SegmentationClass --split=${datasets_path}VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --net_input_width=500 --net_input_height=500
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"