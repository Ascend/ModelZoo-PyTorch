amct_onnx calibration \
          --model yolov5s_sim.onnx \
          --save_path ./result/yolov5s \
          --input_shape "images:1,3,640,640" \
          --data_dir "./amct_data" \
          --data_types "float32"
