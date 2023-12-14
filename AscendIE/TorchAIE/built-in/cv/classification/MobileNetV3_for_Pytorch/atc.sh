## 帮助信息
### === Model Options ===
###  --model        generator, default: mbv3_small
###  --bs           batch size, default: 1
### === Inference Options ===
###  --output_dir   output dir, default: output
### === Environment Options ===
###  --soc          soc version [Ascend310P?], default: Ascend310P3
### === Help Options ===
###  -h             print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}

## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al model:,bs:,output_dir:,soc: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;;
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $model ]]; then model=generator_v1; fi
if [[ -z $bs ]]; then bs=1; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi


atc --model=${output_dir}/${model}.onnx \
    --framework=5 \
    --output=${output_dir}/${model}_bs${bs} \
    --input_format=NCHW \
    --input_shape="input:${bs},3,224,224" \
    --log=error \
    --soc_version=${soc} \
    --input_fp16_nodes="input" \
    --output_type=FP16
