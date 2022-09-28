soc=${1}
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi
model=${2:-"mbv3_small"}
bs=${3:-"1"}
output_dir=${4:-"output"}
mode=${5:-"infer"}
arch=`uname -m`

if [ ! -d ${output_dir} ];then
  mkdir ${output_dir}
fi

echo "Starting pth导出onnx"
python3 pth2onnx.py --pth=${model}.pth.tar --onnx=${model}.onnx --batch=${bs} --dynamic --simplify
if [ $? -ne 0 ]; then
    echo -e "pth导出onnx模型 Failed \n" 
    exit 1
fi

echo "Starting onnx导出om模型"
bash atc.sh ${soc} ${output_dir} ${model} ${bs} 
if [ $? -ne 0 ]; then
    echo -e "onnx导出om模型 Failed \n" 
    exit 1
fi

echo "Starting 推理om模型精度，日志写入val.log" | tee ${output_dir}/val.log
python3 val.py --dataset=imagenet --checkpoint=${output_dir}/${model}_bs${bs}.om --batch=${bs} | tee -a ${output_dir}/val.log
ACC=`cat output/val.log | grep ACC | awk -F '[@|]' '{print $2}' | tr -cd "[0-9|.]"`
echo "ACC:" ${ACC} > ${output_dir}/results.txt

if [[ ${mode} == val ]] ; then
  echo "Starting 推理om模型性能，日志写入val.log" | tee -a ${output_dir}/val.log
  ./benchmark.${arch} -device_id=0 -round=1000 -batch_size=${bs} -om_path=${output_dir}/${model}_bs${bs}.om | tee -a ${output_dir}/val.log
  perf_value=`cat ${output_dir}/val.log | grep ave_throughputRate | awk -F ': ' '{print $2}' | tr -cd "[0-9|.]"`
  echo "perf:" ${perf_value} >> ${output_dir}/results.txt
fi
