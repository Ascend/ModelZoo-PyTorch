python volo_pth2onnx.py --src $1 --des $2 --batchsize $5
python modify.py --src $2 --des $3
atc --model=$3 \
    --framework=5 \
    --output=$4 \
    --input_format=NCHW \
    --input_shape=$6 \
    --log=debug \
    --soc_version=Ascend310 \
    --buffer_optimize=off_optimize
