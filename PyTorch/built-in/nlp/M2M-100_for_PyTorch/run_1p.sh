fairseq-generate data_bin --fp16 --batch-size 224 --path 1.2B_last_checkpoint.pt \
    --fixed-dictionary model_dict.128k.txt -s en -t zh --remove-bpe 'sentencepiece' \
    --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt \
    --decoder-langtok --encoder-langtok src --gen-subset test