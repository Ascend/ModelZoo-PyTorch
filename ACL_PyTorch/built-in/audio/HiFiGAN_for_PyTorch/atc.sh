## 帮助信息
### === Model Options ===
###  --model        generator, default: generator_v1
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


atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${output_dir}/${model}.onnx --output=${output_dir}/${model}_bs${bs} \
    --input_shape="mel_spec:${bs},80,-1" \
    --dynamic_dims="250;500;750;1000;1250;1500;1750;2000"