#!/bin/bash 
mkdir data-bin -p
python preprocess.py \
	-train_src examples/translation/iwslt14.tokenized.de-en/train.de \
	-train_tgt examples/translation/iwslt14.tokenized.de-en/train.en \
	-valid_src examples/translation/iwslt14.tokenized.de-en/valid.de \
	-valid_tgt examples/translation/iwslt14.tokenized.de-en/valid.en \
	-save_data data-bin/demo --overwrite \
