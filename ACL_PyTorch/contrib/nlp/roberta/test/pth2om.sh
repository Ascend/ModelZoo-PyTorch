#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

CHECKPOINTPATH=checkpoints/
CHECKPOINTFILE=checkpoint.pt
DATAPATH=./data/SST-2-bin
ONNXPATH=outputs
PADLENGTH=70

if [ ! -d $ONNXPATH ];
then
    mkdir $ONNXPATH
fi

for i in 1 4 8 16 32
do
    echo [INFO] Generating roberta_base_batch_${i}.onnx
    python3.7 RoBERTa_pth2onnx.py --checkpoint_path $CHECKPOINTPATH --checkpoint_file $CHECKPOINTFILE --data_name_or_path $DATAPATH --onnx_path $ONNXPATH --batch_size $i --pad_length $PADLENGTH
    echo [INFO] Simplifying roberta_base_batch_${i}.onnx
    python3.7 -m onnxsim ./${ONNXPATH}/roberta_base_batch_${i}.onnx ./${ONNXPATH}/roberta_base_batch_${i}_sim.onnx
    echo [INFO] Generating roberta_base_batch_${i}.om
    atc --framework=5 --model=./${ONNXPATH}/roberta_base_batch_${i}_sim.onnx --output=./${ONNXPATH}/roberta_base_batch_$i --input_format=ND --input_shape="src_tokens:${i},${PADLENGTH}" --log=info --soc_version=Ascend310
done


