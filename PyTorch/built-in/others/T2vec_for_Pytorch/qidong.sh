#!/bin/bash

log=$1
if [ x"$log" == x ];then
  echo "pdb"
  python -u t2vec.py -vocab_size 18866 -criterion_name "KLDIV" -knearestvocabs "data/porto-vocab-dist-cell100.h5"
else
  nohup python -u t2vec.py -vocab_size 18866 -criterion_name "KLDIV" -knearestvocabs "data/porto-vocab-dist-cell100.h5" > $log 2>&1 &
fi
wait

Loss=`grep -a "validation loss" $log | awk -F "validation loss " '{print $2}' | awk 'NR==1{min=$1};{if ($1<min)min=$1}END{print min}'`
echo "minloss: $Loss"
