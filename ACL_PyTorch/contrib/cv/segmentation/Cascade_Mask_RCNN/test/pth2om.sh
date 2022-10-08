#!/bin/bash

python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS model_final_e9d89b.pkl MODEL.DEVICE cpu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# ${chip_name}可通过 npu-smi info指令查看
atc --model=output/model.onnx --framework=5 --output=output/cascade_maskrcnn_bs1 --input_format=NCHW --input_shape="0:1,3,1344,1344" --out_nodes="Cast_1835:0;Gather_1838:0;Reshape_1829:0;Slice_1862:0" --log=debug --soc_version=Ascend${chip_name}
if [ -f "output/cascade_maskrcnn_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi