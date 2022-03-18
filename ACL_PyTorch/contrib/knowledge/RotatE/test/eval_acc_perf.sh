arch=`uname -m`
echo "=======pre_process-bs1======="
rm -rf bin
python3.7 rotate_preprocess.py --test_batch_size=1 --output_path='bin-bs1/'


if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====inference with msame bs=1===="
rm -rf  out_bs1
./msame --model "./kge_1_head.om" --input "./bin-bs1/head/pos,./bin-bs1/head/neg"  --output "./out_bs1" --outfmt TXT --device=0
./msame --model "./kge_1_tail.om" --input "./bin-bs1/tail/pos,./bin-bs1/tail/neg" --output "./out_bs1" --outfmt TXT --device=0

echo "====post postprocess with inference and accuracy===="
echo "predata-bs1-acc:"
python3.7  rotate_postprocess.py  --result_path='./out_bs1'  --data_head='./bin-bs1/head'  --data_tail='./bin-bs1/tail' > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "=======pre_process-bs16======="
rm -rf bin
python3.7 rotate_preprocess.py --test_batch_size=16 --output_path='bin-bs16/'

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====inference with msame bs=16===="
rm -rf  out_bs16

./msame --model "./kge_16_head.om" --input "./bin-bs16/head/pos,./bin-bs16/head/neg"  --output "./out_bs16" --outfmt TXT --device=0
./msame --model "./kge_16_tail.om" --input "./bin-bs16/tail/pos,./bin-bs16/tail/neg" --output "./out_bs16" --outfmt TXT --device=0

echo "====post postprocess with inference and accuracy===="
echo "predata-bs16-acc:"
python3.7  rotate_postprocess.py  --result_path='./out_bs16'  --data_head='./bin-bs16/head'  --data_tail='./bin-bs16/tail' > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====benchmark===="
rm -rf result/PureInfer_perf_of_kge_1_head_in_device_0.txt
chmod +x benchmark.x86_64
./benchmark.x86_64 -round=20 -om_path=./kge_1_head.om  -device_id=0 -batch_size=1
echo "benchmark success"

echo "====benchmark===="
rm -rf result/PureInfer_perf_of_kge_16_head_in_device_0.txt
./benchmark.x86_64 -round=20 -om_path=./kge_16_head.om  -device_id=0 -batch_size=16
echo "benchmark success"


echo "====accuracy data bs1===="
python3.7 test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data bs16===="
python3.7 test/parse.py result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
echo "predata-bs1-perf:"
python3.7 test/parse.py result/PureInfer_perf_of_kge_1_head_in_device_0.txt
echo "predata-bs16-perf:"
python3.7 test/parse.py result/PureInfer_perf_of_kge_16_head_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"