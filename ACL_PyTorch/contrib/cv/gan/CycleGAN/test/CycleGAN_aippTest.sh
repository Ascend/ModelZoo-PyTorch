bash env.sh
echo "start to prepare info file"
python3.7 gen_dataset_info.py
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=Cons_Ga_aipp512_b0_bs1.om -input_text_path=testA_prep.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=true
echo "Cons_Gb_aipp512_b0_bs1.om benchmark  start"
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=Cons_Gb_aipp512_b0_bs1.om -input_text_path=testB_prep.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=true
