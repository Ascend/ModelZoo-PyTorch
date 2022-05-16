#! /bin/bash
export profile_path=/home/zhangyifan/mmocr

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done

python ${profile_path}/tools/deployment/pytorch2onnx.py \
--model_config ${profile_path}/configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
--model_ckpt ${profile_path}/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
det \
--dynamic-export

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -f fcenet.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=fcenet.onnx --output=fcenet  --input_format=NCHW \
--input_shape="input:$batch_size,3,1280,2272" --log=error --soc_version=Ascend710

if [ -f "fcenet.om" ]; then
    echo "success"
else
    echo "fail!"
fi