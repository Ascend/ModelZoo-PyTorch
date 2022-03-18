source env.sh

input_file='./RefineDet320_VOC_final.pth'
output_file='RefineDet320_VOC_final_no_nms.onnx'
dataset_path='/root/datasets/VOCdevkit/'

python3.7 RefineDet_pth2onnx.py ${input_file} ${output_file} ${dataset_path}


atc --framework=5 \
--out_nodes="Reshape_224:0;Softmax_231:0;Reshape_237:0;Softmax_244:0" \
--model=${output_file} \
--output=refinedet_voc_320_non_nms_bs1 \
--input_format=NCHW \
--input_shape="image:1,3,320,320" \
--log=debug \
--soc_version=Ascend310 \
--precision_mode allow_fp32_to_fp16 \


atc --framework=5 \
--out_nodes="Reshape_224:0;Softmax_231:0;Reshape_237:0;Softmax_244:0" \
--model=${output_file} \
--output=refinedet_voc_320_non_nms_bs16 \
--input_format=NCHW \
--input_shape="image:16,3,320,320" \
--log=debug \
--soc_version=Ascend310 \
--precision_mode allow_fp32_to_fp16 \