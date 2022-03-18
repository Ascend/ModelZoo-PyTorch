cd ../
source atc.sh
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./pyramidbox_1000_bs1.om -input_text_path=./pyramidbox_pre_bin_1000_1.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./pyramidbox_1000_bs1.om -input_text_path=./pyramidbox_pre_bin_1000_2.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False