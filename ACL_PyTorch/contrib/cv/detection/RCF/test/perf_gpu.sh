#!/bin/bash
batch_size=1
# usage
if [ $# -ne 4 ]
then
    echo "usage: bash test/perf_gpu.sh datasets_path data/BSR/BSDS500/data/images/test batch_size 1"
else
    datasets_path=$2
    batch_size=$4
fi
python3.7 -u performance_gpu.py --onnx_name=rcf_bs${batch_size}_change_sim --imgs_dir data/BSR/BSDS500/data/images/test \
--batch_size ${batch_size} ${batch_size} --height 321 481 --width 481 321 > performance_gpu_bs${batch_size}.log
cat performance_gpu_bs${batch_size}.log