datasets_path=/root/datasets
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

mkdir image_pre_test
python3.7 PraNet_preprocess.py   ${datasets_path}/Kvasir ./image_pre_test
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi
python3.7 get_info.py bin  ./image_pre_test ./pre_bin.info 352 352
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./PraNet-19_bs1.om -input_text_path=./pre_bin.info -input_width=352 -input_height=352 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=./PraNet-19_bs16.om -input_text_path=./pre_bin.info -input_width=352 -input_height=352 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi

python3.7 PraNet_postprocess.py  ${datasets_path}/Kvasir ./result/dumpOutput_device0 ./bs1_test/Kvasir/  
python3.7 Eval.py   ${datasets_path} ./bs1_test/Kvasir/  ./bs1_test/result_bs1.json 
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi
python3.7 PraNet_postprocess.py  ${datasets_path}/Kvasir ./result/dumpOutput_device1 ./bs16_test/Kvasir/  
python3.7 Eval.py  ${datasets_path}  ./bs16_test/Kvasir/  ./bs16_test/result_bs16.json  
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi
python3.7 test/parse.py ./bs1_test/result_bs1.json
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi
python3.7 test/parse.py ./bs16_test/result_bs16.json
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi
echo "====310 performance data===="
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
