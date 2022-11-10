sacrebleu --echo src -l en-zh -t wmt20 | head -n 2048 > raw_input.en-zh.en
sacrebleu --echo ref -l en-zh -t wmt20 | head -n 2048 > raw_input.en-zh.zh

python3 scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=raw_input.en-zh.en --outputs=spm.en-zh.en

python3 scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=raw_input.en-zh.zh --outputs=spm.en-zh.zh

fairseq-preprocess --source-lang en --target-lang zh --testpref spm.en-zh --thresholdsrc 0 --thresholdtgt 0 --destdir data_bin --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt