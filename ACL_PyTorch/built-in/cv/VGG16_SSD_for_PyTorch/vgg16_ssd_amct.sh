
amct_onnx calibration --model ./vgg16_ssd.onnx --save_path ./result_mact/vgg16_ssd --input_shape "actual_input_1:-1,3,300,300" --data_dir "./data_amct" --data_types "float32"