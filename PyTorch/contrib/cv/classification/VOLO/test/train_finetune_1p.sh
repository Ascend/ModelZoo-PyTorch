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

nohup python3 main.py \
	"${data_path}" \
	--model volo_d1 \
	--img-size 224 \
	-j 20  --no-prefetcher \
	-b 128 --lr 1.6e-3 \
	--min-lr 4.0e-6 \
	--epochs 100 \
	--drop-path 0.1 --apex-amp \
	--weight-decay 1.0e-8 --warmup-epochs 5  --ground-truth \
	--finetune "${weight_path}" \
&
