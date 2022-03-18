arch=`uname -m`

rm -rf ./predata_bts1
rm -rf ./predata_bts16
python3.7 r2plus1d_preprocess.py --config=./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py --bts=1 --output_path=./predata_bts1/
python3.7 r2plus1d_preprocess.py --config=./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py --bts=16 --output_path=./predata_bts16/

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf predata_bts1.info
rm -rf predata_bts16.info
python3.7 get_info.py bin ./data/predata_bts1 ./predata_bts1.info 256 256
python3.7 get_info.py bin ./data/predata_bts16 ./predata_bts16.info 256 256
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====inference with msame===="
rm -rf ./predata_bts1_om_out
rm -rf ./predata_bts16_om_out

./msame --model "r2plus1d_bs1.om" --input "./predata_bts1" --output "./predata_bts1_om_out"  --outfmt TXT
./msame --model "r2plus1d_bs16.om" --input "./predata_bts16" --output "./predata_bts16_om_out"  --outfmt TXT



echo "====post postprocess with inference and accuracy===="
echo "predata-bs1-acc:"
python3.7 r2plus1d_postprocess.py --result_path=./predata_bts1_om_out > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "predata-bs16-acc:"
python3.7 r2plus1d_postprocess.py --result_path=./predata_bts1_om_out > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====benchmark===="
rm -rf result/PureInfer_perf_of_r2plus1d_bs1_in_device_0.txt
rm -rf result/PureInfer_perf_of_r2plus1d_bs16_in_device_0.txt
chmod +x benchmark.${arch}
./benchmark.${arch} -round=20 -om_path=r2plus1d_bs1.om  -device_id=0 -batch_size=1
./benchmark.${arch} -round=20 -om_path=r2plus1d_bs16.om  -device_id=0 -batch_size=16
echo "benchmark success"

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
echo "predata-bs1-perf:"
python3.7 test/parse.py result/PureInfer_perf_of_r2plus1d_bs1_in_device_0.txt
echo "predata-bs16-perf:"
python3.7 test/parse.py result/PureInfer_perf_of_r2plus1d_bs16_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
