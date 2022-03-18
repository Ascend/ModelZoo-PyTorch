arch=`uname -m`
datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

rm -rf bin1 bin16
python3.7 Efficient-3DCNNs_preprocess.py --video_path=${datasets_path}/ucf101/rawframes --annotation_path=ucf101_01.json --output_path=bin1 --info_path=ucf101_bs1.info --inference_batch_siz=1
python3.7 Efficient-3DCNNs_preprocess.py --video_path=${datasets_path}/ucf101/rawframes --annotation_path=ucf101_01.json --output_path=bin16 --info_path=ucf101_bs16.info --inference_batch_siz=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf out1 out16
./msame --model Efficient-3DCNNs_bs1.om --input bin1 --output out1 --outfmt BIN --device 0
./msame --model Efficient-3DCNNs_bs16.om --input bin16 --output out16 --outfmt BIN --device 0
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 Efficient-3DCNNs_postprocess.py --result_path=out1 --info_path=ucf101_bs1.info --annotation_path=ucf101_01.json --acc_file=result_bs1.json
python3.7 Efficient-3DCNNs_postprocess.py --result_path=out16 --info_path=ucf101_bs16.info --annotation_path=ucf101_01.json --acc_file=result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
./benchmark.${arch} -round=20 -om_path=Efficient-3DCNNs_bs1.om -device_id=0 -batch_size=1
./benchmark.${arch} -round=20 -om_path=Efficient-3DCNNs_bs16.om -device_id=0 -batch_size=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/PureInfer_perf_of_Efficient-3DCNNs_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/PureInfer_perf_of_Efficient-3DCNNs_bs16_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
