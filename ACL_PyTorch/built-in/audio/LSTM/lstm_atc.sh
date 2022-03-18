export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATHexport ASCEND_OPP_PATH=${install_path}/opp

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --input_format=ND --framework=5 --model=lstm_ctc_16batch.onnx --input_shape="actual_input_1:16,390,243" --output=lstm_ctc_16batch_auto --auto_tune_mode="RL,GA" --log=info --soc_version=Ascend310