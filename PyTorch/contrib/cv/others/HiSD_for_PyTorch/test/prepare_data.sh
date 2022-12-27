#!/bin/bash

##########################################################
#########第3行 至 90行，请一定不要、不要、不要修改##########
#########第3行 至 90行，请一定不要、不要、不要修改##########
#########第3行 至 90行，请一定不要、不要、不要修改##########
##########################################################
# shell脚本所在路径
cur_path=`echo $(cd $(dirname $0);pwd)`
# 判断当前shell是否是performance
perf_flag=`echo $0 | grep performance | wc -l`

CelebA_data_path="" 
CelebA_laebl_path=""
train_dataset="datasets"
test_dataset="datasets_test"
test_img_dir="test_imgs"


# 参数校验，不需要修改
for para in $*
do
    if [[ $para == --CelebA_data_path* ]];then
        CelebA_data_path=`echo ${para#*=}`
    elif [[ $para == --CelebA_laebl_path* ]];then
        CelebA_laebl_path=`echo ${para#*=}`
    fi
done

echo $CelebA_data_path
echo $CelebA_laebl_path

# 校验是否传入data_path,不需要修改
if [[ $CelebA_data_path == "" ]];then
    echo "[Error] para \"CelebA_data_path\" must be config"
    exit 1
fi

if [[ $CelebA_laebl_path == "" ]];then
    echo "[Error] para \"CelebA_laebl_path\" must be config"
    exit 1
fi




python3 preprocessors/celeba-hq.py --img_path $CelebA_data_path --label_path $CelebA_laebl_path --target_path $train_dataset --start 3002 --end 30002
python3 preprocessors/celeba-hq.py --img_path $CelebA_data_path --label_path $CelebA_laebl_path --target_path $test_dataset --start 2 --end 3002

python3 ./split_img.py --test_path $test_dataset --source_path $CelebA_data_path --target_path $test_img_dir

echo "parpare dataset successful!"