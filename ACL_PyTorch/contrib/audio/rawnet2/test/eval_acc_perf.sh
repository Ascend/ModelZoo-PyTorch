datasets_path="/root/datasets/VoxCeleb1/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# 数据预处理输出bin文件夹
if [ -e "bin_out_bs1" ]; then
    rm -r bin_out_bs1
fi
if [ -e "bin_out_bs16" ]; then
    rm -r bin_out_bs16
fi
mkdir bin_out_bs1
mkdir bin_out_bs16

python3.7 RawNet2_preprocess.py --input=${datasets_path}  --batch_size=1 --output="bin_out_bs1/"

if [ $? != 0 ]; then
    echo "fail preprocess"
fi

python3.7 RawNet2_preprocess.py --input=${datasets_path}  --batch_size=16 --output="bin_out_bs16/"

if [ $? != 0 ]; then
    echo "fail preprocess"
else
    echo "success"
fi

#om推理
if [ -e "om_bs1" ]; then
    rm -r om_bs1
fi
if [ -e "om_bs16" ]; then
    rm -r om_bs16
fi
if [ -e "result" ]; then
    rm -r result
fi

mkdir om_bs1
mkdir om_bs16
mkdir result

source /usr/local/Ascend/ascend-toolkit/set_env.sh

./msame --model RawNet2_sim_bs1.om --input bin_out_bs1 --output om_bs1 --outfmt TXT --device 0
if [ $? != 0 ]; then
    echo "fail msame!"
fi
./msame --model RawNet2_sim_bs16.om --input bin_out_bs16 --output om_bs16 --outfmt TXT --device 0

if [ $? != 0 ]; then
    echo "fail msame!"
fi

#om精度判断

python3.7 RawNet2_postprocess.py --input="om_bs1/" --batch_size=1

if [ $? != 0 ]; then
    echo "fail!"
else
    echo "success"
fi

python3.7 RawNet2_postprocess.py --input="om_bs16/" --batch_size=16

if [ $? != 0 ]; then
    echo "fail!"
else
    echo "success"
fi

arch=`uname -m`
./benchmark.${arch} -round=10 -om_path=RawNet2_sim_bs1.om -device_id=0 -batch_size=1
./benchmark.${arch} -round=10 -om_path=RawNet2_sim_bs16.om -device_id=0 -batch_size=16
python3.7 test/parse.py result/PureInfer_perf_of_RawNet2_sim_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 test/parse.py result/PureInfer_perf_of_RawNet2_sim_bs16_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
