export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:$PYTHONPATH
export OM_PATH=`pwd`/vqa-vilbert_bs1.om
export OM_EVAL=True
export OM_DEVICE=3
python3.7 -m allennlp evaluate vilbert-vqa-pretrained.2021-03-15.tar.gz balanced_real_val --batch-size 1
