python data_preprocess.py data ISource INoisy  #执行预处理脚本，生成数据集预处理后的bin文件
python get_info.py bin INoisy DnCNN_bin.info 481 481  #生成数据集信息文件脚本get_info.py
source /usr/local/Ascend/ascend-lastest/set_env.sh  #设置环境变量
chmod u+x benchmark.x86_64  #增加benchmark.{arch}可执行权限
for i in 1 16;do
./benchmark.x86_64 -model_type=vision -om_path=DnCNN-S-15_bs"$i".om -device_id=0 -batch_size="$i" -input_text_path=DnCNN_bin.info -input_width=481 -input_height=481 -useDvpp=false -output_binary=true  #benchmark离线推理
python postprocess.py result/dumpOutput_device0/  #调用postprocess.py脚本推理结果进行PSRN计算
done