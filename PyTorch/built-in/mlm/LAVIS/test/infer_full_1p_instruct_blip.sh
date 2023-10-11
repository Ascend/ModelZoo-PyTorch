img_path=""
prompt=""

for para in $*
do
    if [[ $para == --img_path* ]];then
        img_path=`echo ${para#*=}`
    elif [[ $para == --prompt* ]];then
        prompt=$((`echo ${para#*=}`))
    fi
done

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

#校验是否传入img_path,不需要修改
if [[ $img_path == "" ]];then
    echo "[Error] para \"img_path\" must be confing"
    exit 1
fi

#校验是否传入prompt,不需要修改
if [[ $prompt == "" ]];then
    echo "[Error] para \"prompt\" must be confing"
    exit 1
fi

source ${test_path_dir}/env_npu.sh

#推理开始时间，不需要修改
start_time=$(date +%s)

echo "------------------ Final result ------------------"
python projects/instructblip/infer.py \
      --img_path $img_path \
      --prompt "$prompt"
wait

#推理结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))
echo "E2E Training Duration sec : $e2e_time"