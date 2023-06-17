#!/bin/bash
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

max_seq_len=256
onnx_path=output/onnx
om_path=output/om

if [ ! -d om_path ];then
  mkdir -p om_path
fi


echo "导出om：encoder"
atc --framework=5 --input_format=ND --soc_version=${soc} \
    --model=${onnx_path}/encoder_bs${bs}.onnx --output=${om_path}/encoder_dyn \
    --input_shape_range="sequences:[${bs},1~${max_seq_len}];sequence_lengths:[${bs}]" \
    --log=error

echo "导出om：decoder"
atc --framework=5 --input_format=ND --soc_version=${soc} \
    --model=${onnx_path}/decoder_iter_bs${bs}.onnx --output=${om_path}/decoder_iter_dyn \
    --input_shape_range="decoder_input:[${bs},80];attention_hidden:[${bs},1024];attention_cell:[${bs},1024];decoder_hidden:[${bs},1024];decoder_cell:[${bs},1024];attention_weights:[${bs},1~${max_seq_len}];attention_weights_cum:[${bs},1~${max_seq_len}];attention_context:[${bs},512];memory:[${bs},1~${max_seq_len},512];processed_memory:[${bs},1~${max_seq_len},128];mask:[${bs},1~${max_seq_len}]" \
    --log=error

echo "导出om：postnet"
atc --framework=5 --input_format=ND --soc_version=${soc} \
    --model=${onnx_path}/postnet_bs${bs}.onnx --output=${om_path}/postnet_dyn \
    --input_shape_range="mel_outputs:[${bs},80,1~2000]" \
    --log=error
