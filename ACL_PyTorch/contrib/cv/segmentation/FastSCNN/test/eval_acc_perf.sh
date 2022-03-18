 #!/bin/bash
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "====data prp_treatment starting===="
rm -rf ./datasets
ln -s opt/npu/datasets datasets
python3.7.5 Fast_SCNN_preprocess.py 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====data prp_treatment finished===="


echo "====creating info file===="
rm -rf ./fast_scnn_prep_bin.info
rm -rf prep_dataset
ln -s /opt/npu/prep_dataset prep_dataset
python3.7.5 gen_dataset_info.py 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====creating info file done===="


echo "====bs1 benchmark start===="
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./fast_scnn_bs1.om -input_text_path=./fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 benchmark finished===="


echo "====bs4 benchmark start===="
source env.sh
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=4 -om_path=./fast_scnn_bs4.om -input_text_path=./fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs4 benchmark finished===="


echo "====bs16 benchmark start===="
source env.sh
rm -rf result/dumpOutput_device3
./benchmark.${arch} -model_type=vision -device_id=2 -batch_size=16 -om_path=./fast_scnn_bs16.om -input_text_path=./fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs16 benchmark finished===="




echo "====bs1 evaluate start; this cost long time===="
rm -rf fast_scnn_bs1.log
python3.7.5 -u Fast_SCNN_postprocess.py ./result/dumpOutput_device0 ./out >fast_scnn_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 evaluate finished===="


echo "====bs4 evaluate start; this cost long time===="
rm -rf fast_scnn_bs4.log
python3.7.5 -u Fast_SCNN_postprocess.py ./result/dumpOutput_device1 ./out >fast_scnn_bs4.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs4 evaluate finished===="


echo "====bs16 evaluate start; this cost long time===="
rm -rf fast_scnn_bs16.log
python3.7.5 -u Fast_SCNN_postprocess.py ./result/dumpOutput_device2 ./out >fast_scnn_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs16 evaluate finished===="


echo "====accuracy data===="
echo "pth mIoU:64.46"
python3.7.5 test/parse.py fast_scnn_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7.5 test/parse.py fast_scnn_bs4.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


python3.7.5 test/parse.py fast_scnn_bs16.log
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
echo "success"


python3.7.5 test/parse.py result/perf_vision_batchsize_4_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"

python3.7.5 test/parse.py result/perf_vision_batchsize_16_device_2.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
