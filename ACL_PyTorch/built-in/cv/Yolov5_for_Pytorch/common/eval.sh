## 帮助信息
### === Model Options ===
###  --version       yolov5 tags [2.0/3.1/4.0/5.0/6.0/6.1], default: 6.1
###  --model         yolov5[n/s/m/l/x], default: yolov5s
###  --bs            batch size, default: 4
### === Build Options ===
###  --type         data type [fp16/int8], default: fp16
### === Inference Options ===
###  --mode          infer/val, default: infer
###  --output_dir    output_dir dir, default: output
###  --dataset       dataset dir, default: val2017
### === Environment Options ===
###  --install_path  CANN install path, default: /usr/local/Ascend/ascend-toolkit
###  --arch          platform [x86_64/aarch64], default: x86_64
### === Help Options ===
###  -h              print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}


## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al version:,model:,bs:,type:,mode:,output_dir:,dataset:,install_path:,arch: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;; 
        --version) version=$2; shift 2;;
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --type) type=$2; shift 2;;
        --mode) mode=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --dataset) dataset=$2; shift 2;;
        --install_path) install_path=$2; shift 2;;
        --arch) arch=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $version ]]; then version=6.1; fi
if [[ -z $model ]]; then model=yolov5s; fi
if [[ -z $bs ]]; then bs=4; fi
if [[ -z $type ]]; then type=fp16; fi
if [[ -z $mode ]]; then mode=infer; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $dataset ]]; then dataset=val2017; fi
if [[ -z $install_path ]]; then install_path=/usr/local/Ascend/ascend-toolkit; fi
if [[ -z $arch ]]; then arch=x86_64; fi

args_info="=== eval args === \n version: $version \n model: $model \n bs: $bs \n type: $type \n 
           mode: $mode \n output_dir: $output_dir \n dataset: $dataset \n 
           install_path: $install_path \n arch: $arch"
echo -e $args_info

if [ ${type} == int8 ] ; then
    model=${model}_amct
fi

## 推理模型
echo "Starting om模型推理精度，日志写入val.log" | tee ${output_dir}/val.log
python3 common/om_infer.py --img-path=${dataset} --model=${output_dir}/${model}_nms_bs${bs}.om --batch-size=${bs} | tee -a ${output_dir}/val.log
acc_value=`cat ${output_dir}/val.log | grep Precision | sed '2,$d' | awk -F '] = ' '{print $2}'`
echo "acc: " ${acc_value} > ${output_dir}/results.txt

if [[ ${mode} == infer && -f ./benchmark.${arch} ]] ; then
    echo "Starting om模型纯推理性能(有后处理)，日志写入val.log" | tee -a ${output_dir}/val.log
    chmod 777 ./benchmark.${arch}
    ./benchmark.${arch} -device_id=0 -round=10 -batch_size=${bs} -om_path=${output_dir}/${model}_nms_bs${bs}.om | tee -a ${output_dir}/val.log
fi


## 性能评估
if [[ ${mode} == val ]] ; then
    echo "Starting om模型纯推理性能(无后处理)，日志写入val.log" | tee -a ${output_dir}/val.log
    chmod 777 ./benchmark.${arch}
    ./benchmark.${arch} -device_id=0 -round=10 -batch_size=${bs} -om_path=${output_dir}/${model}_bs${bs}.om | tee -a ${output_dir}/val.log
    perf_value=`cat ${output_dir}/val.log | grep ave_throughputRate | awk -F ': ' '{print $2}' | tr -cd "[0-9|.]"`
    echo "perf: " ${perf_value} >> ${output_dir}/results.txt

    echo "Starting om模型性能profiling，日志写入profile.log"
    ${install_path}/latest/toolkit/tools/profiler/bin/msprof --output=${output_dir}/profiling \
        --aicpu=on --dvpp-profiling=on --runtime-api=on --task-time=on \
        --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on \
        --model-execution=on \
        --application="./benchmark.${arch} -device_id=0 -round=10 -batch_size=${bs} -om_path=${model}_bs${bs}.om" > ${output_dir}/profile.log
fi

