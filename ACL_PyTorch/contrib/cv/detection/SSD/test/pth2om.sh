#!/bin/bash
rm -rf ssd_300_coco.onnx
python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/ssd/ssd300_coco.py ./ssd300_coco_20200307-a92d2092.pth --output-file=ssd_300_coco.onnx --shape=300 --verify --show --mean 123.675 116.28 103.53 --std 1 1 1
echo "convert onnx"
source env.sh
rm -rf ssd_300_coco.om
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=ssd_300_coco.onnx --framework=5 --output=ssd_300_coco --input_format=NCHW --input_shape="input:1,3,300,300" --log=info --soc_version=Ascend310 --out_nodes="Concat_637:0;Reshape_639:0" --buffer_optimize=off_optimize --precision_mode allow_mix_precision
echo "convert om"
if [ -f "ssd_300_coco.om" ]; then
    echo "success"
else
    echo "fail!"
fi