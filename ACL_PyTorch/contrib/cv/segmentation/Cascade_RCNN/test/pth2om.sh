source /usr/local/Ascend/ascend-toolkit/set_env.sh


atc --model=model_py1.8.onnx --framework=5 --output=cascadercnn_detectron2_npu --input_format=NCHW --input_shape="0:1,3,1344,1344" --out_nodes="Cast_1853:0;Gather_1856:0;Reshape_1847:0;Slice_1886:0" --log=debug --soc_version=Ascend310