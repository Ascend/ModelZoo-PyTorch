source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --input_format=ND --framework=5 --model=lstm_ctc_16batch.onnx --input_shape="actual_input_1:16,390,243" --output=lstm_ctc_16batch_auto --auto_tune_mode="RL,GA" --log=info --soc_version=Ascend310