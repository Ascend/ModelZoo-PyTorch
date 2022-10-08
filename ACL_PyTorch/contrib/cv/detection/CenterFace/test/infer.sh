cd ../
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./CenterFace_bs1.om -input_text_path=./CenterFace_pre_bin.info -input_width=800 -input_height=800 -output_binary=True -useDvpp=False
