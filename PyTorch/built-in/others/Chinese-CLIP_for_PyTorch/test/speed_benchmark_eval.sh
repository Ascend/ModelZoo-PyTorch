
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

DATAPATH=${1}

source ${test_path_dir}/env_npu.sh

python3 cn_clip/deploy/speed_benchmark.py \
        --model-arch ViT-B-16 \
        --pytorch-ckpt ${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt \
        --pytorch-precision fp16

