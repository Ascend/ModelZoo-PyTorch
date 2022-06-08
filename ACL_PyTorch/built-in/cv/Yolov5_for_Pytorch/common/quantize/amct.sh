#!/bin/bash

onnx=$1

## 量化配置参数
config=common/quantize/config.cfg
echo 'activation_offset: true' > ${config}
echo 'do_fusion: true' >> ${config}

if [[ ${onnx} == yolov5n_cali.onnx ]]; then
    echo 'skip_layers: ["Conv_0","Conv_3","Conv_155","Conv_198","Conv_175","Conv_199","Conv_195","Conv_200"]' >> ${config}
    echo 'skip_fusion_layers: ["Conv_0","Conv_3","Conv_155","Conv_198","Conv_175","Conv_199","Conv_195","Conv_200"]' >> ${config}
elif [[ ${onnx} == yolov5l_cali.onnx ]]; then
    echo 'skip_layers: ["Conv_0","Conv_3","Conv_277","Conv_344","Conv_309","Conv_345","Conv_341","Conv_346"]' >> ${config} 
    echo 'skip_fusion_layers: ["Conv_0","Conv_3","Conv_277","Conv_344","Conv_309","Conv_345","Conv_341","Conv_346"]' >> ${config}
fi

## onnx量化
if [[ ${onnx} == *_cali.onnx ]]; then
    amct_onnx calibration \
            --model ${onnx} \
            --save_path ./output/result \
            --input_shape "images:-1,3,640,640" \
            --data_dir "./amct_data" \
            --data_types "float32" \
            --calibration_config=common/quantize/config.cfg
else
    amct_onnx calibration \
            --model ${onnx} \
            --save_path ./output/result \
            --input_shape "images:-1,3,640,640" \
            --data_dir "./amct_data" \
            --data_types "float32"
fi