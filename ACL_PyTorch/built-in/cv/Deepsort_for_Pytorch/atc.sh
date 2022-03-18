export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
# export DUMP_GE_GRAPH=2
export ASCEND_SLOG_PRINT_TO_STDOUT=1

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./yolov3-sim.onnx --framework=5 --output=yolov3-sim --input_format=NCHW --input_shape="actual_input_1:1,3,416,416" --log=info --soc_version=Ascend310
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./deep.onnx --framework=5 --output=deep_dims --input_format=ND --input_shape="actual_input_1:-1,3,128,64" \
--dynamic_dims="1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47;48;49;50" \
--log=info --soc_version=Ascend310