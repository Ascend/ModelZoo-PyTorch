source ./test/env.sh
chmod +x ./benchmark.x86_64

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./FOTS_bs1.om -input_text_path=./FOTS_prep_bin.info -input_width=2240 -input_height=1248 -output_binary=True -useDvpp=False