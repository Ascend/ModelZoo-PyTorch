./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mnasnet1.0_bs1.om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=4 -om_path=mnasnet1.0_bs4.om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=8 -om_path=mnasnet1.0_bs8.om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=mnasnet1.0_bs16.om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=32 -om_path=mnasnet1.0_bs32.om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=64 -om_path=mnasnet1.0_bs64.om -input_text_path=mnasnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False