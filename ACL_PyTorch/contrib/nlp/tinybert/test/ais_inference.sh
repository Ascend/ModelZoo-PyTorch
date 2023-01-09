. /usr/local/Ascend/ascend-toolkit/set_env.sh &&
python3.7 -m ais_bench --model ./TinyBERT_bs$1.om \
--input "./bert_bin/input_ids,./bert_bin/segment_ids,./bert_bin/input_mask" \
--output ./result --outfmt TXT --batchsize $1
