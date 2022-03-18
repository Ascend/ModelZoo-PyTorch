#!/bin/bash
set -eu
set -x

#ls|xargs rm -f ./pre_dataset
#coco_imgs_path="/root/data/coco/images"
#coco_anns_path="/root/data/coco/annotations"
for para in $*
do
    if [[ $para == --coco_imgs_path* ]]; then
        coco_imgs_path=`echo ${para#*=}`
    fi
done

for para in $*
do
    if [[ $para == --coco_anns_path* ]]; then
        coco_anns_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`

rm -rf pre_dataset
python3.7 M2Det_preprocess.py --config=M2Det/configs/m2det512_vgg.py --save_folder=pre_dataset --COCO_imgs=${coco_imgs_path} --COCO_anns=${coco_anns_path}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 1. creating prep_dataset successfully.'

rm -rf coco_prep_bin.info coco_images.info

python3.7 gen_dataset_info.py bin pre_dataset coco_prep_bin.info 512 512
python3.7 gen_dataset_info.py jpg ${coco_imgs_path}/val2014 coco_images.info

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 2. creating coco_prep_bin.info coco_images.info successfully.'

rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=0 -om_path=m2det512_bs1.om -input_text_path=coco_prep_bin.info -input_width=512 -input_height=512 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 3. conducting m2det512_bs1.om on device 0 successfully.'

rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -batch_size=16 -device_id=1 -om_path=m2det512_bs16.om -input_text_path=coco_prep_bin.info -input_width=512 -input_height=512 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 4. conducting m2det512_bs16.om on device 1 successfully.'

rm -rf result/detection-results_0/COCO/
python3.7 M2Det_postprocess.py --bin_data_path=result/dumpOutput_device0/ --test_annotation=coco_images.info --det_results_path=result/detection-results_0 --net_out_num=2 --prob_thres=0.1 --COCO_imgs=${coco_imgs_path} --COCO_anns=${coco_anns_path}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 5. calculate acc on bs1 successfully.'

rm -rf detection-results_1/COCO/
python3.7 M2Det_postprocess.py --bin_data_path=result/dumpOutput_device1/ --test_annotation=coco_images.info --det_results_path=result/detection-results_1 --net_out_num=2 --prob_thres=0.1 --COCO_imgs=${coco_imgs_path} --COCO_anns=${coco_anns_path}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 6. calculate acc on bs16 successfully.'

echo "success"
