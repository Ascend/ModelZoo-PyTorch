
export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=${install_path}
# export DUMP_GE_GRAPH=2
export SLOG_PRINT_TO_STDOUT=1
export REPEAT_TUNE=True

atc --model=./vgg16_ssd.onnx --framework=5 --output=vgg16_ssd --input_format=NCHW --input_shape="actual_input_1:1,3,300,300" --log=info --soc_version=Ascend710

atc --model= ./result_amct/vgg16_ssd_deploy_model.onnx --framework=5 --output=vgg16_ssd_deploy_model --input_format=NCHW --input_shape="actual_input_1:1,3,300,300" --log=info --soc_version=Ascend710