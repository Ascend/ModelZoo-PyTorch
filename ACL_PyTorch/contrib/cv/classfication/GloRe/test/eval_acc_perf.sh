datasets_path="/root/datasets/UCF-101"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
# 数据预处理
if [ -d "bin" ]; then
    rm -rf bin
fi

python3.7 GloRe_preprocess.py --data-root ${datasets_path} --save-path bin/bs1 --batch-size 1
if [ $? != 0 ]; then
    echo "fail preprocess"
fi
python3.7 GloRe_preprocess.py --data-root ${datasets_path} --save-path bin/bs16  --batch-size 16
if [ $? != 0 ]; then
    echo "fail preprocess"
fi

#om推理
rm -rf om_res
./msame --model "GloRe_bs1.om" --input "bin/bs1" --output "om_res_bs1" --outfmt TXT
if [ $? != 0 ]; then
    echo "fail msame!"
fi
./msame --model "GloRe_bs16.om" --input "bin/bs16" --output "om_res_bs16" --outfmt TXT
if [ $? != 0 ]; then
    echo "fail msame!"
fi

#om精度判断
if [ -f "om_res_bs1.json" ]; then
    rm om_res_bs1.json
fi
if [ -f "om_res_bs16.json" ]; then
    rm om_res_bs16.json
fi
python3.7 GloRe_postprocess.py --i om_res_bs1 --t bs1_target.json --o res_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
fi
echo "success"
python3.7 GloRe_postprocess.py --i om_res_bs16 --t bs16_target.json --o res_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
fi
echo "success"
arch=`uname -m`
./benchmark.${arch} -round=20 -om_path=./GloRe_bs1.om -device_id=0 -batch_size=1
./benchmark.${arch} -round=20 -om_path=./GloRe_bs16.om -device_id=0 -batch_size=16

echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
fi
echo "success"
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
fi
echo "success"