source env_npu.sh

data_path_info=$1
weight_path_info=$2
data_path=`echo ${data_path_info#*=}`
weight_path=`echo ${weight_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="/home/imagenet"
fi
if [[ $weight_path == "" ]];then
    echo "[Warning] para \"weight_path\" not set"
    echo "[Warning] use default weight_path"
    weight_path="/home/VOLO"
fi
python3 validate.py "${data_path}"  --model volo_d1 \
  --checkpoint "${weight_path}" --no-test-pool --apex-amp --img-size 224 -b 128
