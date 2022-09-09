export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

atc --model=decoder.onnx --framework=5 --output=decoder_static --input_format=ND \
    --input_shape="encoder_out:32,-1,256;encoder_out_lengths:32;hyps_pad_sos_eos:32,10,-1;hyps_lens_sos:32,10;r_hyps_pad_sos_eos:32,10,-1;ctc_score:32,10" --log=error \
    --dynamic_dims="96,5,5;96,6,6;96,7,7;96,8,8;96,9,9;96,10,10;96,11,11;96,12,12;96,13,13;96,14,14;96,15,15;96,16,16;96,17,17;96,18,18;96,19,19;96,20,20;144,5,5;
    144,6,6;144,7,7;144,8,8;144,9,9;144,10,10;144,11,11;144,12,12;144,13,13;144,14,14;144,15,15;144,16,16;144,17,17;144,18,18;144,19,19;144,20,20;
    144,21,21;144,22,22;144,23,23;144,24,24;144,25,25;144,26,26;144,27,27;144,28,28;144,29,29;384,5,5;384,6,6;384,7,7;384,8,8;384,9,9;384,10,10;384,11,11;
    384,12,12;384,13,13;384,14,14;384,15,15;384,16,16;384,17,17;384,18,18;384,19,19;384,20,20;384,21,21;384,22,22;384,23,23;384,24,24;384,25,25;384,26,26;
    384,27,27;384,28,28;384,29,29;384,30,30;384,31,31;384,32,32;384,33,33;384,34,34;384,35,35;384,36,36;384,37,37;384,38,38;384,39,39;384,40,40;384,41,41;
    384,42,42;384,43,43;384,44,44;" \
    --soc_version=$1
