soc=${1}
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi
model=${2:-"aasist_bs1"}
bs=${3:-"1"}
output_dir=${4:-"output"}
mode=${5:-"infer"}
install_path=${6:-"/usr/local/Ascend/ascend-toolkit"}


## pth导出om模型
bash pth2om.sh --model=${model} --bs=${bs} --output_dir=${output_dir} --soc=${soc}

if [ $? -ne 0 ]; then
    echo -e "pth导出om模型 Failed \n" 
    exit 1
fi

## om模型推理
echo "Starting om模型推理精度，日志写入val.log" | tee ${output_dir}/val.log
python3 om_infer.py --om=${output_dir}/${model}.om --batch=${bs} | tee -a ${output_dir}/val.log
EER=`cat ${output_dir}/val.log | grep EER | sed '2,$d' | awk -F '=' '{print $2}' | tr -cd "[0-9|.]"`
echo "EER: " ${EER} > ${output_dir}/results.txt
tDCF=`cat ${output_dir}/val.log | grep tDCF | awk -F '=' '{print $2}'`
echo "min-tDCF: " ${tDCF} >> ${output_dir}/results.txt

if [[ ${mode} == val ]] ; then
    echo "Starting om模型纯推理性能，日志写入val.log" | tee -a ${output_dir}/val.log
    ./benchmark.x86_64 -device_id=0 -round=10 -batch_size=${bs} -om_path=${output_dir}/${model}.om | tee -a ${output_dir}/val.log
    perf_value=`cat ${output_dir}/val.log | grep ave_throughputRate | awk -F ': ' '{print $2}' | tr -cd "[0-9|.]"`
    echo "perf: " ${perf_value} >> ${output_dir}/results.txt

    echo "Starting om模型性能profiling，日志写入profile.log"
    ${install_path}/latest/toolkit/tools/profiler/bin/msprof --output=${output_dir}/profiling \
        --aicpu=on --dvpp-profiling=on --runtime-api=on --task-time=on \
        --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on \
        --model-execution=on \
        --application="./benchmark.x86_64 -device_id=0 -round=10 -batch_size=${bs} -om_path=${output_dir}/${model}.om" > ${output_dir}/profile.log
fi

