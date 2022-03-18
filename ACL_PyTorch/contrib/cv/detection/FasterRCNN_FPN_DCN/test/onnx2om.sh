 
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
atc --framework=5 --model=./faster_rcnn_r50_fpn_1x_coco.onnx --output=faster_rcnn_r50_fpn_1x_coco  --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=debug --soc_version=Ascend310
