#!/bin/bash

oridata_path=$1
data_bin=$2
spm_path=$3
dict_path=$4
mose_http=$5
iws_url=$6
bash examples/prepare_iwslt14.sh $oridata_path $mose_http $iws_url
wait

orill_dir=$oridata_path/iwslt14.tokenized.de-en
spm_encode --model=$spm_path --output_format=piece < $orill_dir/train.de > train.spm.src
spm_encode --model=$spm_path --output_format=piece < $orill_dir/train.en > train.spm.tgt
spm_encode --model=$spm_path --output_format=piece < $orill_dir/valid.de > valid.spm.src
spm_encode --model=$spm_path --output_format=piece < $orill_dir/valid.en > valid.spm.tgt
spm_encode --model=$spm_path --output_format=piece < $orill_dir/test.de > test.spm.src
spm_encode --model=$spm_path --output_format=piece < $orill_dir/test.en > test.spm.tgt

wait
python preprocess.py  \
    --trainpref train.spm \
    --validpref valid.spm \
    --testpref test.spm \
    --source-lang src --target-lang tgt \
    --destdir $data_bin \
    --srcdict $dict_path \
    --tgtdict $dict_path \
    --workers 40

