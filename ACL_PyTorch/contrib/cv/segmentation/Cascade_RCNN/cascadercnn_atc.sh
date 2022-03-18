export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest


atc --model=model_py1.8.onnx --framework=5 --output=cascadercnn_detectron2_npu --input_format=NCHW --input_shape="0:1,3,1344,1344" --out_nodes="Cast_1853:0;Gather_1856:0;Reshape_1847:0;Slice_1886:0" --log=debug --soc_version=Ascend310