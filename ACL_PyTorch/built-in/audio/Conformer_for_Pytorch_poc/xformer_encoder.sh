atc --model=xformer_encoder.onnx --framework=5 --output=xformer_encoder_dynamic --input_format=ND \
--input_shape="feats:1,-1,80" --log=error --op_select_implmode=high_performance --optypelist_for_implmode="Sigmoid" \
--soc_version=$1

atc --model=xformer_encoder_multibatch.onnx --framework=5 --output=xformer_encoder_rank --input_format=ND \
    --input_shape="feats:$2,-1,80;mask:$2,-1;position:$2,-1;conv_mask:$2,1,-1" --log=error --op_select_implmode=high_performance --optypelist_for_implmode="Sigmoid" \
    --dynamic_dims="259,64,4096,64;387,96,9216,96;515,128,16384,128;643,160,25600,160;771,192,36864,192;899,224,50176,224;1027,256,65536,256;1155,288,82944,288;1283,320,102400,320;1411,352,123904,352;1539,384,147456,384;1667,416,173056,416;1795,448,200704,448;1923,480,230400,480" \
    --soc_version=$1
