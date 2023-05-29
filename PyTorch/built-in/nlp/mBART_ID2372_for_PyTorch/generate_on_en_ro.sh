source ./test/env_npu.sh
DATA_PATH=path_of_data                     # fix it to your own train data path
BPE_PATH=/path/sentence.bpe.model         # fix it to your own sentence.bpe.model path
SCRIPTS=mosesdecoder/scripts              # fix it to your own mosesdecoder path
WMT16_SCRIPTS=wmt16-scripts               # fix it to your own wmt16-scripts path

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
model_dir=$1
REPLACE_UNICODE_PUNCT=$SCRIPTS/tokenizer/replace-unicode-punctuation.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py
HYP=hyp
REF=ref

fairseq-generate $DATA_PATH \
  --fp16 --path $model_dir --max-tokens 1024 \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t ro_RO -s en_XX \
  --bpe 'sentencepiece' --sentencepiece-model $BPE_PATH \
  --scoring sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs $langs > en_ro
sed -i '$d' en_ro
cat en_ro | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > $HYP".txt"
cat en_ro | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > $REF".txt"

for f in $HYP $REF
	do
	rm -rf "en_ro."$f
	cat $f".txt" | \
	perl $REPLACE_UNICODE_PUNCT | \
	perl $NORM_PUNC -l ro | \
	perl $REM_NON_PRINT_CHAR | \
	python3 $NORMALIZE_ROMANIAN | \
	python3 $REMOVE_DIACRITICS | \
	perl $TOKENIZER -no-escape -threads 16 -a -l ro >"en_ro."$f
	done
sacrebleu -tok 'none' -s 'none' en_ro.ref < en_ro.hyp