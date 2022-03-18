# 使用二进制输入时，执行如下命令
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./AlignedReID_bs1.om -input_text_path=./alignedreid.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=4 -om_path=./AlignedReID_bs4.om -input_text_path=./alignedreid.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=8 -om_path=./AlignedReID_bs8.om -input_text_path=./alignedreid.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./AlignedReID_bs16.om -input_text_path=./alignedreid.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=32 -om_path=./AlignedReID_bs32.om -input_text_path=./alignedreid.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False

# 参数1:推理结果路径
python3.7 AlignedReID_acc_eval.py ./result/dumpOutput_device0

