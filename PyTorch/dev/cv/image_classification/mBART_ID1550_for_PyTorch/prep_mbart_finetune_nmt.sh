#!/bin/bash

mkdir data-bin -p
python preprocess.py \
	-train_src examples/translation/iwslt14.tokenized.de-en/src-train-finetune-mbart.txt \
	-train_tgt examples/translation/iwslt14.tokenized.de-en/tgt-train-finetune-mbart.txt \
	-valid_src examples/translation/iwslt14.tokenized.de-en/src-valid-finetune-mbart.txt \
	-valid_tgt examples/translation/iwslt14.tokenized.de-en/tgt-valid-finetune-mbart.txt \
	-save_data data-bin/demo --overwrite \
	--subword_prefix_is_added \
	-src_vocab mbartdata-bin/demo.vocab.pt # This is the vocab used for mbart training
