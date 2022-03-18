#! /bin/bash

#test

#清除上次运行数据
rm -r ./result/*
rm -r ./query_preproc_data_Ascend310
rm -r ./gallery_preproc_data_Ascend310
#数据预处理
echo "preprocess......"
python3.7 ../PCB_pth_preprocess.py -d market -b 1 --height 384 --width 128 --data-dir /opt/npu/Market_1501/ -j 4
#生成数据集信息文件
echo "get_info......"
python3.7 ../get_info.py bin ./query_preproc_data_Ascend310 ./query_preproc_data_Ascend310.info 128 384
python3.7 ../get_info.py bin ./gallery_preproc_data_Ascend310 ./gallery_preproc_data_Ascend310.info 128 384
#离线推理  bs = 1
echo "off-line inference  bs = 1......"
#gallery 
../benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./models/PCB_sim_split_bs1_autotune.om -input_text_path=./gallery_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False >> gallary_bs1.log
mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_gallery_bs1
mv ./result/perf_vision_batchsize_1_device_0.txt ./result/gallery_perf_vision_batchsize_1_device_0.txt
#query 
../benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./models/PCB_sim_split_bs1_autotune.om -input_text_path=./query_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False >> query_bs1.log
mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_query_bs1
mv ./result/perf_vision_batchsize_1_device_0.txt ./result/query_perf_vision_batchsize_1_device_0.txt
#离线推理  bs = 16
echo "off-line inference  bs = 16......"
#gallery 
../benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./models/PCB_sim_split_bs16_autotune.om -input_text_path=./gallery_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False >> gallary_bs16.log
mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_gallery_bs16
mv ./result/perf_vision_batchsize_16_device_0.txt ./result/gallery_perf_vision_batchsize_16_device_0.txt
#query 
../benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./models/PCB_sim_split_bs16_autotune.om -input_text_path=./query_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False >> query_bs16.log
mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_query_bs16
mv ./result/perf_vision_batchsize_16_device_0.txt ./result/query_perf_vision_batchsize_16_device_0.txt
###数据后处理
echo "postprocess......"
python3.7 ../PCB_pth_postprocess.py -q ./result/dumpOutput_device0_query_bs1 -g ./result/dumpOutput_device0_gallery_bs1 -d market --data-dir /opt/npu/Market_1501/
echo "====performance data===="
echo "bs1 : "
python3.7 parse.py ./result/gallery_perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "bs16 : "
python3.7 parse.py ./result/gallery_perf_vision_batchsize_16_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
