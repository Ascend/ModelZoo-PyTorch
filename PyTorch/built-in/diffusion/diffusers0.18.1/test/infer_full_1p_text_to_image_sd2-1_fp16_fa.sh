# 微调生成的ckpt路径
ckpt_path="/path/ckpt"
mixed_precision="fp16"

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

#创建输出目录，不需要修改
output_path=${test_path_dir}/output/infer_res
mkdir -p ${output_path}

#推理开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

python3 examples/text_to_image/infer_text_to_image.py \
  --mixed_precision=${mixed_precision} \
  --enable_npu_flash_attention \
  --ckpt_path=${ckpt_path} \
  --output_path=${output_path} \
  --device_id=0
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))