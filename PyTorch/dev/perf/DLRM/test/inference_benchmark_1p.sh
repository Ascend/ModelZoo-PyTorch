data_path=""

cur_path=$(pwd)
source ${cur_path}/test/env_npu.sh

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]]; then
  echo "[Error] para \"data_path\" must be confing"
  exit 1
fi

python3 ./dlrm/scripts/main.py --mode inference_benchmar --dataset ${data_path}  --interaction_op dot --embedding_type joint