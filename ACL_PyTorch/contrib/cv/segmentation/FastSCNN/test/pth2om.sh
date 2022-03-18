#!/usr/bin/env bash
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs1 --batch_size 1
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs4 --batch_size 4
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs8 --batch_size 8
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs16 --batch_size 16
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs32 --batch_size 32
atc --framework=5 --model=fast_scnn_bs1.onnx --output=fast_scnn_bs1  --output_type=FP16 --input_format=NCHW --input_shape="image:1,3,1024,2048" --log=debug --soc_version=Ascend310
atc --framework=5 --model=fast_scnn_bs4.onnx --output=fast_scnn_bs4  --output_type=FP16 --input_format=NCHW --input_shape="image:4,3,1024,2048" --log=debug --soc_version=Ascend310
atc --framework=5 --model=fast_scnn_bs8.onnx --output=fast_scnn_bs8  --output_type=FP16 --input_format=NCHW --input_shape="image:8,3,1024,2048" --log=debug --soc_version=Ascend310
atc --framework=5 --model=fast_scnn_bs16.onnx --output=fast_scnn_bs16  --output_type=FP16 --input_format=NCHW --input_shape="image:16,3,1024,2048" --log=debug --soc_version=Ascend310
atc --framework=5 --model=fast_scnn_bs32.onnx --output=fast_scnn_bs32  --output_type=FP16 --input_format=NCHW --input_shape="image:32,3,1024,2048" --log=debug --soc_version=Ascend310
