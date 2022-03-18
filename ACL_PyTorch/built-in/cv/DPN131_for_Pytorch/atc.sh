export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/bin:${install_path}/atc/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/fwkacllib/lib64:${install_path}/acllib/lib64:${install_path}/atc/lib64/plugin/opskernel/:/usr/local/Ascend/aoe/lib64/:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}

atc --model=./dpn131.onnx --framework=5 --output=dpn131_fp16_bs8 --input_format=NCHW --input_shape="image:8,3,224,224" --log=error --soc_version=Ascend710
