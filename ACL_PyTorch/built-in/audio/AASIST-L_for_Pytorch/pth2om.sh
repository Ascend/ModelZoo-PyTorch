## 参数设置
GETOPT_ARGS=`getopt -o '' -al model:,bs:,output_dir:,soc: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $model ]]; then model=aasist_bs1; fi
if [[ -z $bs ]]; then bs=1; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi

if [ ! -d ${output_dir} ]; then
  mkdir ${output_dir}
fi

echo "Starting pth导出onnx"
python3 util/pth2onnx.py --onnx=${output_dir}/${model}.onnx --batch=${bs} || exit 1

echo "Starting 修改onnx模型"
python3 util/modify.py -m1=${output_dir}/${model}.onnx -m2=${output_dir}/${model}.onnx -bs=${bs} || exit 1

echo "Starting atc转模型"
bash util/atc.sh ${soc} ${output_dir}/${model} ${output_dir}/${model} ${bs} || exit 1

echo -e "pth导出om模型 Success \n"

