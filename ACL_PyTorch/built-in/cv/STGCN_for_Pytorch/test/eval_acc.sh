#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# om模型batchsize1生成结果与基准对比结果
rm -rf ./st_gcn_bs1.log
python3.7 st_gcn_postprocess.py -result_dir=./result/dumpOutput_device0/ -label_dir=./data/Kinetics/kinetics-skeleton/val_label.pkl > ./st_gcn_bs1.log
if [ $? != 0 ]; then
    echo "bs1 comparison fail!"
    exit -1
fi
# om模型batchsize16生成结果与基准对比结果
rm -rf ./st_gcn_bs16.log
python3.7 st_gcn_postprocess.py -result_dir=./result/dumpOutput_device1/ -label_dir=./data/Kinetics/kinetics-skeleton/val_label.pkl > ./st_gcn_bs16.log
if [ $? != 0 ]; then
    echo "bs1 comparison fail!"
    exit -1
fi

echo "----bs1 acc result----"
cat ./st_gcn_bs1.log
echo "----bs16 acc result----"
cat ./st_gcn_bs16.log
echo "success"