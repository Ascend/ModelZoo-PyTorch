source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
./msame --model "./TinyBERT.om" \
--input "/home/TinyBERT/bert_bin/input_ids,/home/TinyBERT/bert_bin/segment_ids,/home/TinyBERT/bert_bin/input_mask" \
--output "./result/msame_output" --outfmt TXT
