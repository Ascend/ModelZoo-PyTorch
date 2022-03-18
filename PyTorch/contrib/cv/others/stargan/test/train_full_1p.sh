INPUT_PATH=$1

workdir=$(cd $(dirname $0); pwd)

source $workdir/env_npu.sh

if [ ! $INPUT_PATH ]; then
    nohup python3 -u ./main.py  --mode train --folder_dir stargan_NPU_1p \
                                         --batch_size 16 --epoch 50 --npus 1 --distributed True \
                                         --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young &
else
    nohup python3 -u ./main.py  --mode train --folder_dir stargan_NPU_1p \
                                         --batch_size 16 --epoch 50 --npus 1 --distributed True \
                                         --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                                         --dataset_dir $INPUT_PATH &
fi