datasets_path="./data/modelnet40_ply_hdf5_2048"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 PointNetCNN_preprocess.py ./prep_dataset ./labels
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
./msame --model pointnetcnn_bs1.om --input "prep_dataset,prep_dataset" --output ./output --outfmt TXT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device0
./benchmark.${arch} -round=20 -om_path=pointnetcnn_bs1.om -device_id=0 -batch_size=1

echo "benchmark success"
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


python3.7 PointNetCNN_postprocess.py  ./labels/label ./output > result_bs1.json
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
echo "====performance data===="
python3.7 test/parse.py result/PureInfer_perf_of_pointnetcnn_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
