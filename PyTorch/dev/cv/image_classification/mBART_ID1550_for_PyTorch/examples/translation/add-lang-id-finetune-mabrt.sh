#!/bin/bash

# This will add language tag at the end of each segment in the corpus
sed -e 's/$/ <\/s> <EN>/' train.en > src-train-finetune-mbart.txt
sed -e 's/^/<DE> /' train.de > temp-file.de
sed -e 's/$/ <\/s> <DE>/' temp-file.de > tgt-train-finetune-mbart.txt

sed -e 's/$/ <\/s> <EN>/' valid.en > src-valid-finetune-mbart.txt
sed -e 's/^/<DE> /' valid.de > temp-file.de
sed -e 's/$/ <\/s> <DE>/' temp-file.de > tgt-valid-finetune-mbart.txt
rm temp-file.de
