export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/fwkacllib/lib64:${install_path}/acllib/lib64:${install_path}/atc/lib64/plugin/opskernel/:/usr/local/Ascend/aoe/lib64/:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}

# modify model_name run_model soc_version
model_name="ssd_vgg16"
run_model=(vgg16_ssd.onnx result_amct/vgg16_ssd_deploy_model.onnx)     ## FP16  INT8   
soc_version="Ascend710"                      ## 710

batch_sizes="1 4 8 16 32 64"
format_types=(fp16, int8)

if [ ! -d log ];then
    mkdir log
fi

for ((i=0;i<2;i++))
do
    format_type=${format_types[i]}
    for batch_size in $batch_sizes
    do
        output="${model_name}_${format_type}_bs${batch_size}_npu"
        echo "$output"
        if [ -f "$output".log ];then
            rm "$output".log
        fi
        
        ## modify  input shape
        atc --model=${run_model[i]} --framework=5 --output=./om/$output --input_format=NCHW \
            --input_shape="actual_input_1:${batch_size},3,300,300" --log=error --soc_version=$soc_version

        benchmark.x86_64 --model_type=vision -batch_size=${batch_size} -device_id=0 -om_path="./om/${output}.om" \
        -input_text_path=vgg16_ssd.info -input_width=300 -input_height=300 -useDvpp=false -output_binary=true >> "./log/${output}".log

        python3.7.5 ssd_pth_postprocess.py ./VOC2007/ ./voc-model-labels.txt ./result/dumpOutput_device0/ ./eval_results/ >> "./log/${output}".log
        value_info=`cat "./log/$output".log | grep ave_throughputRate | awk -F ': ' '{print $2}' | tr -cd "[0-9|.]"`
        echo ${format_type},${batch_size},${value_info} >> "${model_name}_npu".csv

    done
done
