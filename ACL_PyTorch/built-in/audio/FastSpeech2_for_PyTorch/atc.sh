## 帮助信息
### === Model Options ===
###  --bs           batch size, default: 1
### === Environment Options ===
###  --soc          soc version [Ascend310P?], default: Ascend310P3
### === Help Options ===
###  -h             print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}

## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al soc:,bs: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;;
        --soc) soc=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi
if [[ -z $bs ]]; then bs=1; fi

max_src_len=256
max_mel_len=2000
onnx_path=output/onnx
om_path=output/om

if [ ! -d om_path ];then
  mkdir -p om_path
fi

echo "1) 导出om：encoder"
atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${onnx_path}/encoder.onnx --output=${om_path}/encoder_bs${bs} \
    --input_shape_range="texts:[${bs},1~${max_src_len}];src_masks:[${bs},1~${max_src_len}]"
mv ${om_path}/encoder_bs${bs}* ${om_path}/encoder_bs${bs}.om

echo "2) 导出om：variance_adaptor"
atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${onnx_path}/variance_adaptor.onnx --output=${om_path}/variance_adaptor_bs${bs} \
    --input_shape_range="enc_output:[${bs},1~${max_src_len},256];src_masks:[${bs},1~${max_src_len}]"
mv ${om_path}/variance_adaptor_bs${bs}* ${om_path}/variance_adaptor_bs${bs}.om

echo "3) 导出om：decoder"
atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${onnx_path}/decoder.onnx --output=${om_path}/decoder_bs${bs} \
    --input_shape_range="output:[${bs},1~${max_mel_len},256];mel_masks:[${bs},1~${max_mel_len}]"
mv ${om_path}/decoder_bs${bs}* ${om_path}/decoder_bs${bs}.om

echo "4) 导出om：postnet"
atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${onnx_path}/postnet.onnx --output=${om_path}/postnet_bs${bs} \
    --input_shape_range="dec_output:[${bs},1~${max_mel_len},256]"
mv ${om_path}/postnet_bs${bs}* ${om_path}/postnet_bs${bs}.om

echo "5) 导出om：hifigan"
atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${onnx_path}/hifigan.onnx --output=${om_path}/hifigan_bs${bs} \
    --input_shape="mel_output:${bs},-1,80" \
    --dynamic_dims="250;500;750;1000;1250;1500;1750;2000"
