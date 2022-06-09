## 帮助信息
### === Model Options ===
###  --version      yolov5 tags [2.0/3.1/4.0/5.0/6.0/6.1], default: 6.1
###  --model        yolov5[n/s/m/l/x], default: yolov5s
###  --bs           batch size, default: 4
### === Build Options ===
###  --type         data type [fp16/int8], default: fp16
###  --calib_bs     batch size of calibration data (int8 use only), default: 16
### === Inference Options ===
###  --mode         infer/val, default: infer
###  --conf         confidence threshold, default: 0.4
###  --iou          NMS IOU threshold, default: 0.5
###  --output_dir   output dir, default: output
### === Environment Options ===
###  --soc          soc version [Ascend310/Ascend710], default: Ascend710
### === Help Options ===
###  -h             print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}

## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al version:,model:,bs:,type:,calib_bs:,mode:,conf:,iou:,output_dir:,soc: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;; 
        --version) version=$2; shift 2;;
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --type) type=$2; shift 2;;
        --calib_bs) calib_bs=$2; shift 2;;
        --mode) mode=$2; shift 2;;
        --conf) conf=$2; shift 2;;
        --iou) iou=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $version ]]; then version=6.1; fi
if [[ -z $model ]]; then model=yolov5s; fi
if [[ -z $bs ]]; then bs=4; fi
if [[ -z $type ]]; then type=fp16; fi
if [[ -z $calib_bs ]]; then calib_bs=16; fi
if [[ -z $mode ]]; then mode=infer; fi
if [[ -z $conf ]]; then conf=0.4; fi
if [[ -z $iou ]]; then iou=0.5; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $soc ]]; then soc=Ascend710; fi

if [[ ${type} == fp16 ]] ; then
    args_info="=== pth2om args === \n version: $version \n model: $model \n bs: $bs \n type: $type \n 
               mode: $mode \n conf: $conf \n iou: $iou \n output_dir: $output_dir \n soc: $soc"
    echo -e $args_info
else
    args_info="=== pth2om args === \nversion: $version \n model: $model \n bs: $bs \n type: $type \n calib_bs: $calib_bs \n 
               mode: $mode \n conf: $conf \n iou: $iou \n output_dir: $output_dir \n soc: $soc"
    echo -e $args_info
fi

if [ ! -d ${output_dir} ]; then
  mkdir ${output_dir}
fi

## pt导出om模型
echo "Starting 修改pytorch源码"
git apply v${version}/v${version}.patch

echo "Starting 导出onnx模型并简化"
if [[ ${version} == 6* ]] ; then
    python3 export.py --weights=${model}.pt --imgsz=640 --batch-size=${bs} --opset=11 --dynamic || exit 1
else
    python3 models/export.py --weights=${model}.pt --img-size=640 --batch-size=${bs} --opset=11 --dynamic || exit 1
fi
python3 -m onnxsim ${model}.onnx ${model}.onnx --dynamic-input-shape --input-shape images:${bs},3,640,640 || exit 1
model_tmp=${model}

if [ ${type} == int8 ] ; then
    echo "Starting 生成量化数据"
    python3 common/quantize/generate_data.py --img_info_file=common/quantize/img_info_amct.txt --save_path=amct_data --batch_size=${calib_bs} || exit 1
    
    if [[ ${version} == 6.1 && ${model} == yolov5[nl] ]] ; then
        echo "Starting pre_amct"
        python3 common/quantize/calibration_scale.py --input=${model}.onnx --output=${model}_cali.onnx --mode=pre_amct || exit 1

        echo "Starting onnx模型量化"
        bash common/quantize/amct.sh ${model}_cali.onnx || exit 1
        if [[ -f ${output_dir}/result_deploy_model.onnx ]];then
            mv ${output_dir}/result_deploy_model.onnx ${model}_amct.onnx
        fi
        rm -rf ${model}_cali.onnx

        echo "Starting after_amct"
        python3 common/quantize/calibration_scale.py --input=${model}_amct.onnx --output=${model}_amct.onnx --mode=after_amct || exit 1
    else
        echo "Starting onnx模型量化"
        bash common/quantize/amct.sh ${model}.onnx || exit 1
        if [[ -f ${output_dir}/result_deploy_model.onnx ]];then
            mv ${output_dir}/result_deploy_model.onnx ${model}_amct.onnx
        fi
    fi

    model_tmp=${model}_amct
    if [[ -f ${output_dir}/result_* ]];then
        rm -rf  ${output_dir}/result_*
    fi
fi

echo "Starting 修改onnx模型，添加NMS后处理算子"
python3 common/util/modify_model.py --pt=${model}.pt --onnx=${model_tmp}.onnx --conf-thres=${conf} --iou-thres=${iou} || exit 1

echo "Starting onnx导出om模型（有后处理）"
bash common/util/atc.sh infer ${model_tmp}_nms.onnx ${output_dir}/${model_tmp}_nms ${bs} ${soc} || exit 1
rm -rf ${model_tmp}_nms.onnx

if [[ ${mode} == val ]] ; then
    echo "Starting onnx导出om模型（无后处理）"
    bash common/util/atc.sh val ${model_tmp}.onnx ${output_dir}/${model_tmp} ${bs} ${soc} || exit 1
    rm -rf ${model_tmp}.onnx
fi

echo -e "pth导出om模型 Success \n"
