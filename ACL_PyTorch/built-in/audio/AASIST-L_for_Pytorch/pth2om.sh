## 参数设置
GETOPT_ARGS=`getopt -o '' -al model:,bs:,output_dir: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc_version) soc_version=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $model ]]; then model=aasist_bs1; fi
if [[ -z $bs ]]; then bs=1; fi
if [[ -z $output_dir ]]; then output_dir=output; fi

if [ ! -d ${output_dir} ]; then
  mkdir ${output_dir}
fi

echo "Starting pth导出onnx"
python3 util/pth2onnx.py --onnx=${output_dir}/${model}.onnx --batch=${bs} || exit 1

echo "Starting 修改onnx模型"
python3 util/modify_onnx.py --input=${output_dir}/${model}.onnx --output=${output_dir}/${model}.onnx || exit 1

echo "Starting atc转模型"
bash util/atc.sh ${output_dir}/${model} ${output_dir}/${model} ${bs} ${soc_version} || exit 1

echo -e "pth导出om模型 Success \n"

