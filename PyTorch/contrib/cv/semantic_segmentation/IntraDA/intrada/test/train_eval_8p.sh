source ./test/env_npu.sh
nohup python3 -u test.py \
        --cfg ./intrada_8p.yml \
        --device_type npu \
        --device_id 0 &>eval.log &