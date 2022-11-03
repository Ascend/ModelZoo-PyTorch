#!/bin/bash

t_log="8p"

ll_log=`ls 8p_*.log`

max=-1
#echo $ll_log | xargs -n1
for log in ${ll_log};
do
  i=`echo ${log:2} | tr -cd [0-9]`
  if ((i > max));then
  #if [ $i -gt $max ];then
    max=$i
  fi
done
echo "max: $max"
max=$((max+1))
train_log="8p_${max}.log"
echo "log: ${train_log}"

