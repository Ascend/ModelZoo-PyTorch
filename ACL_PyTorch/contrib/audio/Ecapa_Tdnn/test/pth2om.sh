cd ..
python pytorch2onnx.py checkpoint ecapa_tdnn.onnx 

python fix_conv1d.py ecapa_tdnn.onnx ecapa_tdnn_sim.onnx

rm -rf ./om
mkdir om

echo om_bs=1
atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=om/ecapa_tdnn_bs1 --input_format=ND --input_shape="mel:1,80,200" --log=debug  --soc_version=Ascend710>after_bs1.log

echo om_bs=8
atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=om/ecapa_tdnn_bs8 --input_format=ND --input_shape="mel:8,80,200" --log=debug  --soc_version=Ascend710>after_bs8.log

echo om_bs=16
atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=om/ecapa_tdnn_bs16 --input_format=ND --input_shape="mel:16,80,200" --log=debug --soc_version=Ascend710>after_bs16.log 

echo om_bs=32
atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=om/ecapa_tdnn_bs32 --input_format=ND --input_shape="mel:32,80,200" --log=debug  --soc_version=Ascend710>after_bs32.log 

echo om_bs=64
atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=om/ecapa_tdnn_bs64 --input_format=ND --input_shape="mel:64,80,200" --log=debug  --soc_version=Ascend710>after_bs64.log 

echo om_bs=4
rm -rf ./om_aoe
mkdir om_aoe
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:$PATH

export ASCEND_DEVICE_ID=0
export TUNE_BANK_PATH=./aoe_result_bs4
export TE_PARALLEL_COMPILER=8
export REPEAT_TUNE=False

mkdir ./aoe_result_bs4
chmod 777 ./aoe_result_bs4

# 这是aoe调优命令，已验证bs4调出来最优，只需要调这一个就行了， 生成的ecapa_tdnn_bs4_jt12.om是结合子图和算子调优出来的。
aoe --model=ecapa_tdnn_sim.onnx --framework=5 --input_format=ND --output=./om_aoe/ecapa_tdnn_bs4_jt1 --job_type=1 --input_shape="mel:4,80,200"
aoe --model=ecapa_tdnn_sim.onnx --framework=5 --input_format=ND --output=./om_aoe/ecapa_tdnn_bs4_jt12 --job_type=2 --input_shape="mel:4,80,200"

atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=./om/ecapa_tdnn_bs4 --input_format=ND --input_shape="mel:4,80,200" --log=debug  --soc_version=Ascend710
