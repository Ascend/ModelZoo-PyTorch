#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=1

export REPEAT_TUNE=True
# export ENABLE_TUNE_BANK=True
# export TUNE_BANK_PATH=/home/ljt/deepspeech.pytorch/AT/custom_tune_bank
export TUNE_OPS_NAME=LSTM_124/DynamicRNNForward,LSTM_124/DynamicRNNReverse,LSTM_182/DynamicRNNForward,LSTM_182/DynamicRNNReverse,LSTM_240/DynamicRNNForward,LSTM_240/DynamicRNNReverse,LSTM_298/DynamicRNNForward,LSTM_298/DynamicRNNReverse,LSTM_356/DynamicRNNForward,LSTM_356/DynamicRNNReverse
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_DEVICE_ID=3

atc --framework=5 --model=deepspeech810_sim.onnx --input_format=ND --input_shape="spect:1,1,161,621;transcript:1" --output=deepspeech_sim_AT --log=info --soc_version=Ascend310 --auto_tune_mode="RL"
