source scripts/npu_setenv.sh

nohup python3.7 mobilenet.py --data /opt/npu/imagenet \
        -b 512 \
        --ngpu 1 \
        --epochs 1 \
        -j $(($(nproc))) \
        --lr 0.8 \
        --device_id 0  \
	--device_id 0 \
        1>log.txt \
        2> error.txt &