#!/bin/bash 

# This will add language tag at the end of each segment in the corpu
sed -e 's/$/ <\/s> <EN>/' train.en > src-train-mbart.txt
sed -e 's/$/ <\/s> <DE>/' train.de >> src-train-mbart.txt

sed -e 's/^/<EN> /' train.en > temp-file.en
sed -e 's/^/<DE> /' train.de > temp-file.de

sed -e 's/$/ <\/s> <EN>/' temp-file.en > tgt-train-mbart.txt
sed -e 's/$/ <\/s> <DE>/' temp-file.de >> tgt-train-mbart.txt

sed -e 's/$/ <\/s> <EN>/' valid.en > src-valid-mbart.txt
sed -e 's/$/ <\/s> <DE>/' valid.de >> src-valid-mbart.txt

sed -e 's/^/<EN> /' valid.en > temp-file.en
sed -e 's/^/<DE> /' valid.de > temp-file.de

sed -e 's/$/ <\/s> <EN>/' temp-file.en > tgt-valid-mbart.txt
sed -e 's/$/ <\/s> <DE>/' temp-file.de >> tgt-valid-mbart.txt
rm temp-file.en temp-file.de
