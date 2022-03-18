#!/bin/bash
mkdir roberta-data-bin -p

# We use a hack to use the same preprocessing pipeline by putting both the src and target the same corpus

python preprocess.py \
	-train_src examples/translation/iwslt14.tokenized.de-en/train-roberta.txt \
	-train_tgt examples/translation/iwslt14.tokenized.de-en/train-roberta.txt \
	-valid_src examples/translation/iwslt14.tokenized.de-en/valid-roberta.txt \
	-valid_tgt examples/translation/iwslt14.tokenized.de-en/valid-roberta.txt \
	-save_data roberta-data-bin/demo \
	--bert_masking \
	--share_vocab # It is very important to use this switch
