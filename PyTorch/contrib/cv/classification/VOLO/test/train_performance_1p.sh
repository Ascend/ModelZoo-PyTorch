source env_npu.sh

data_path_info=$1
label_path_info=$2
data_path=`echo ${data_path_info#*=}`
label_path=`echo ${label_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="/home/imagenet"
fi
if [[ $label_path == "" ]];then
    echo "[Warning] para \"label_path\" not set"
    echo "[Warning] use default label_path"
    label_path="/home/label_top5_train_nfnet"
fi

python3 main.py  \
"${data_path}" \
--model volo_d1 \
--img-size 224 \
-j 20  --no-prefetcher \
-b 128 --lr 2e-4 \
--epochs 1 \
--drop-path 0.1 --apex-amp \
--loss-scale 1.0 \
--token-label --token-label-size 14 \
--token-label-data "${label_path}"
