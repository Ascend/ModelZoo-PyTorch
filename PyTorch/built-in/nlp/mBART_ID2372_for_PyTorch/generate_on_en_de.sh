source ./test/env_npu.sh
DATA_PATH=path_of_data                     # fix it to your own train data path
BPE_PATH=/path/sentence.bpe.model         # fix it to your own sentence.bpe.model path
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
model_dir=$1
DETOKENIZER=mosesdecoder/scripts/tokenizer/detokenizer.perl
HYP=hyp
REF=ref

fairseq-generate $DATA_PATH \
  --fp16 --path $model_dir --max-tokens 1024 \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t de_DE -s en_XX \
  --bpe 'sentencepiece' --sentencepiece-model $BPE_PATH \
  --scoring sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs $langs > en_de
sed -i '$d' en_de
cat en_de | grep -P "^H" |sort -V |cut -f 3- > $HYP".txt"
cat en_de | grep -P "^T" |sort -V |cut -f 2- > $REF".txt"

$DETOKENIZER -l de < $HYP".txt" >test.detok.hyp
sacrebleu -t wmt20 -l en-de -i test.detok.hyp -b