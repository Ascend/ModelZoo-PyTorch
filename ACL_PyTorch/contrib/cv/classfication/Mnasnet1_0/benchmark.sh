for i in {1,4,8,16,32,64}
do
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size="$i" -om_path=mnasnet1.0_bs"$i".om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
done