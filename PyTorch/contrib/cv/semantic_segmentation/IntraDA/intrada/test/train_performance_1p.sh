source ./test/env_npu.sh
nohup python3 -u train.py \
        --cfg ./intrada.yml \
        --device_type npu \
        --device_id 0 \
        --performance_log &>performance_1p.log &
