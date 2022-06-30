#!/bin/bash

get_lscpu_value() {
    awk -F: "(\$1 == \"${1}\"){gsub(/ /, \"\", \$2); print \$2; found=1} END{exit found!=1}"
}

lscpu_out=$(lscpu)
n_sockets=4
n_cores_per_socket=$(get_lscpu_value 'Core(s) per socket' <<< "${lscpu_out}")

echo "num_sockets = ${n_sockets} cores_per_socket=${n_cores_per_socket}"

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi

}
export PYTHONPATH=../:$PYTHONPATH

python3.7 -u -m bind_pyt \
    --nsockets_per_node ${n_sockets} \
    --ncores_per_socket ${n_cores_per_socket} \
    --master_addr $(hostname -I |awk '{print $1}') \
    --no_hyperthreads \
    --no_membind "$@" main.py\
    --data /secHome/imagenet/ --model CSWin_64_12211_tiny_224 -j 16 --pin-mem -b 256 --lr 2e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.2; check_status
exit ${EXIT_STATUS}
# --no-prefetcher