source ./npu_set_env.sh
source ./env_new.sh
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Binarize the dataset:
TEXT=examples/translation/iwslt14.tokenized.de-en
python3 preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en