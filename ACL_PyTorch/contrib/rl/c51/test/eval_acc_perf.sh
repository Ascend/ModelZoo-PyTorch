echo "====inference with msame bs=1===="
./msame --model "c51_bs1.om" --input "dataset/bin" --output "dataset/out" --outfmt TXT


echo "====benchmark bs=1===="
rm -rf result/PureInfer_perf_of_c51_bs1_in_device_0.txt
chmod u+x ./benchmark.x86_64
./benchmark.x86_64 -round=20 -om_path=c51_bs1.om -device_id=0 -batch_size=1
echo "benchmark success"


echo "====accuracy data===="
python3.7 acc_compare.py dataset/actions dataset/out 1000

echo "====performance data===="
echo "c51-bs1-perf:"
python3.7 test/parse.py result/PureInfer_perf_of_c51_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
