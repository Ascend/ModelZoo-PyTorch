model_name=$1
batch_size=$2
max_seq_len=$3
soc_version=$4
device_id=$5
task_name=$6

atc --model=model.onnx --framework=5 --output=model --input_format=ND --input_shape="token_type_ids:${batch_size},${max_seq_len};input_ids:${batch_size},${max_seq_len}" --log=error --soc_version=${soc_version} --op_precision_mode=op_precision.ini
rm -rf results.txt
echo "acc test"
python3 infer.py --task_name ${task_name} --model_path model.om --use_pyacl 1 --device npu --device_id ${device_id} --batch_size ${batch_size} --model_name_or_path ${model_name}
echo "performance test"
rm -rf performance.log
arch=`uname -m`
./benchmark.${arch} -om_path=model.om -device_id=${device_id} -batch_size=${batch_size} -round=1000 > performance.log
value_info=`cat performance.log | grep ave_throughputRate | awk -F ': ' '{print $2}' | tr -cd "[0-9|.]"`
echo "perf: ${value_info}" >> results.txt
echo "done"