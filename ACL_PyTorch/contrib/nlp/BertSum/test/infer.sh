cd ../
#./msame --model "./bertsum_13000_9_sim_bs1.om" --input "./pre_data/src/data_1.bin,./pre_data/segs/data_1.bin,./pre_data/clss/data_1.bin,./pre_data/mask/data_1.bin,./pre_data/mask_cls/data_1.bin" --output "./result" --outfmt BIN
./msame --model "./bertsum_13000_9_sim_bs1.om" --input "./pre_data/src,./pre_data/segs,./pre_data/clss,./pre_data/mask,./pre_data/mask_cls" --output "./result" --outfmt bin
./msame --model "./bertsum_13000_9_sim_bs1.om" --input "./pre_data_1/src,./pre_data_1/segs,./pre_data_1/clss,./pre_data_1/mask,./pre_data_1/mask_cls" --output "./result" --outfmt bin
