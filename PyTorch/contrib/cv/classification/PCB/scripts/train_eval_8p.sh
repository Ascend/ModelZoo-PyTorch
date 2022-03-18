data_path_info=$1
data_path=`echo ${data_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="../data/Market-1501/"
    # exit 1
fi
nohup python3 PCB_amp.py --npu -d market -a resnet50 -b 64 -j 4 --epochs 60 --lr 0.8 --log logs/market-1501/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 \
--data-dir ${data_path} --evaluate &
