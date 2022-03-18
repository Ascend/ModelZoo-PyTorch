bash env_npu.sh
echo "atc model_Ga-_bs1.om atc model_Gb-_bs1.om start"
atc --framework=5 --model=onnxmodel/model_Ga.onnx --output=model_Ga-b0_bs1 --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310
atc --framework=5 --model=onnxmodel/model_Gb.onnx --output=model_Gb-b0_bs1 --input_format=NCHW --input_shape="img_maps_sat:1,3,256,256"  --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310
echo "atc model_Ga-_bs16.om atc model_Gb-_bs16.om start"
atc --framework=5 --model=onnxmodel/model_Ga.onnx --output=model_Ga-b0_bs16 --input_format=NCHW --input_shape="img_sat_maps:16,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310
atc --framework=5 --model=onnxmodel/model_Gb.onnx --output=model_Gb-b0_bs16 --input_format=NCHW --input_shape="img_maps_sat:16,3,256,256"  --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310
