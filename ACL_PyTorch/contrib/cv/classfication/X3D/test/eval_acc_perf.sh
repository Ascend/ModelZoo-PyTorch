datasets_path="/root/datasets/Knetics-400"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# 数据预处理
if [ -e "data_bin_bs1" ]; then
    rm -r data_bin_bs1
fi
if [ -e "data_bin_bs16" ]; then
    rm -r data_bin_bs16
fi
mkdir data_bin_bs1
mkdir data_bin_bs16

sed -i 's/ENABLE: True/ENABLE: False/g' SlowFast/configs/Kinetics/X3D_S.yaml
python3 X3d_preprocess.py --cfg SlowFast/configs/Kinetics/X3D_S.yaml DATA.PATH_TO_DATA_DIR ${datasets_path} DATA.PATH_PREFIX "${datasets_path}/val" TEST.BATCH_SIZE 1 X3D_PREPROCESS.ENABLE True X3D_PREPROCESS.DATA_OUTPUT_PATH "data_bin_bs1/"
if [ $? != 0 ]; then
    echo "fail preprocess"
fi
python3 X3d_preprocess.py --cfg SlowFast/configs/Kinetics/X3D_S.yaml DATA.PATH_TO_DATA_DIR ${datasets_path} DATA.PATH_PREFIX "${datasets_path}/val" TEST.BATCH_SIZE 16 X3D_PREPROCESS.ENABLE True X3D_PREPROCESS.DATA_OUTPUT_PATH "data_bin_bs16/"
if [ $? != 0 ]; then
    echo "fail preprocess"
else
    echo "success"
fi

#om推理
if [ -e "om_res_bs1" ]; then
    rm -r om_res_bs1
fi
if [ -e "om_res_bs16" ]; then
    rm -r om_res_bs16
fi
mkdir om_res_bs1
mkdir om_res_bs16

./msame --model x3d_s1.om --input data_bin_bs1 --output om_res_bs1 --outfmt BIN --device 0
if [ $? != 0 ]; then
    echo "fail msame!"
fi
./msame --model x3d_s16.om --input data_bin_bs16 --output om_res_bs16 --outfmt BIN --device 0

if [ $? != 0 ]; then
    echo "fail msame!"
fi

#om精度判断

sed -i 's/ENABLE: True/ENABLE: False/g' SlowFast/configs/Kinetics/X3D_S.yaml
python3 X3d_postprocess.py --cfg SlowFast/configs/Kinetics/X3D_S.yaml TEST.BATCH_SIZE 1 X3D_POSTPROCESS.ENABLE True X3D_POSTPROCESS.OM_OUTPUT_PATH "om_res_bs1/"

if [ $? != 0 ]; then
    echo "fail!"
else
    echo "success"
fi

python3 X3d_postprocess.py --cfg SlowFast/configs/Kinetics/X3D_S.yaml TEST.BATCH_SIZE 16 X3D_POSTPROCESS.ENABLE True X3D_POSTPROCESS.OM_OUTPUT_PATH "om_res_bs16/"

if [ $? != 0 ]; then
    echo "fail!"
else
    echo "success"
fi

arch=`uname -m`
./benchmark.${arch} -round=10 -om_path=x3d_s1.om -device_id=0 -batch_size=1
./benchmark.${arch} -round=10 -om_path=x3d_s16.om -device_id=0 -batch_size=16