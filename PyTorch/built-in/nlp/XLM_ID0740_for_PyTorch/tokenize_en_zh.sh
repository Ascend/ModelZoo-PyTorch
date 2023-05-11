#!/bin/bash
FASTBPE=tools/fastBPE/fast
OUTPATH=data/processed/XLM_en_zh/50k
mkdir -p $OUTPATH

cat data/wiki/txt/en.train | ./tools/tokenize.sh en > data/wiki/txt/token_en.train
cat data/wiki/txt/en.test | ./tools/tokenize.sh en > data/wiki/txt/token_en.test
cat data/wiki/txt/en.valid | ./tools/tokenize.sh en > data/wiki/txt/token_en.valid

cat data/wiki/txt/zh.train | ./tools/tokenize.sh zh > data/wiki/txt/token_zh.train
cat data/wiki/txt/zh.test | ./tools/tokenize.sh zh > data/wiki/txt/token_zh.test
cat data/wiki/txt/zh.valid | ./tools/tokenize.sh zh > data/wiki/txt/token_zh.valid

shuf -r -n 10000000 data/wiki/txt/token_en.train >> $OUTPATH/bpe.train.en
shuf -r -n 10000000 data/wiki/txt/token_zh.train >> $OUTPATH/bpe.train.zh

$FASTBPE learnbpe 50000 $OUTPATH/bpe.train.en $OUTPATH/bpe.train.zh > $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/train.en data/wiki/txt/token_en.train $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/test.en data/wiki/txt/token_en.test $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/valid.en data/wiki/txt/token_en.valid $OUTPATH/codes

$FASTBPE applybpe $OUTPATH/train.zh data/wiki/txt/token_zh.train $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/test.zh data/wiki/txt/token_zh.test $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/valid.zh data/wiki/txt/token_zh.valid $OUTPATH/codes

$FASTBPE getvocab $OUTPATH/train.en $OUTPATH/train.zh > $OUTPATH/vocab

python3 preprocess.py $OUTPATH/vocab $OUTPATH/train.en
python3 preprocess.py $OUTPATH/vocab $OUTPATH/test.en
python3 preprocess.py $OUTPATH/vocab $OUTPATH/valid.en

python3 preprocess.py $OUTPATH/vocab $OUTPATH/train.zh
python3 preprocess.py $OUTPATH/vocab $OUTPATH/test.zh
python3 preprocess.py $OUTPATH/vocab $OUTPATH/valid.zh
