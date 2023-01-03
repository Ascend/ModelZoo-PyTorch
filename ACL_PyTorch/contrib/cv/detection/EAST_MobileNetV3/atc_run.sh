chip_name=310P3
bs=1

atc --framework=5 \
    --model=east.onnx \
    --input_shape="x:${bs},3,704,1280" \
    --output=east_aoe_aipp_bs${bs} \
    --log=error \
    --soc_version=Ascend${chip_name} \
    --insert_op_conf=east_aipp.cfg \
    --enable_small_channel=1