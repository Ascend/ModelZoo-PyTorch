qmodel_path=$1
output_model_name=$2

/usr/local/Ascend/atc/bin/atc \
       	--model=$model_path \
       	--framework=5 \
	--output=$output_model_name \
	--input_format=NCHW 
        --input_shape="actual_input_1:1,3,299,299" \
       	--enable_small_channel=1 \
	--log=error \
       	--soc_version=Ascend310 \
        --insert_op_conf=./convert/inception_v4_pt_aipp.cfg
       
