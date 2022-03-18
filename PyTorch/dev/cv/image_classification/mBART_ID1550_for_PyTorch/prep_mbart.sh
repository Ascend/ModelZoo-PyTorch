#!/bin/bash
mkdir mbartdata-bin -p

python preprocess.py \
	-train_src examples/translation/iwslt14.tokenized.de-en/src-train-mbart.txt \
	-train_tgt examples/translation/iwslt14.tokenized.de-en/tgt-train-mbart.txt \
	-valid_src examples/translation/iwslt14.tokenized.de-en/src-valid-mbart.txt \
	-valid_tgt examples/translation/iwslt14.tokenized.de-en/tgt-valid-mbart.txt \
	-save_data mbartdata-bin/demo \
	--mbart_masking \
	--subword_prefix_is_added \
	--share_vocab # this is very important to use this switch
