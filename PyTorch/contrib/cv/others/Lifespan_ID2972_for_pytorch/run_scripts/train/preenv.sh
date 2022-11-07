#!/bin/bash
#Usage: ./preenv data_url train_url
CUR_DIR=$( cd `dirname $0` ; pwd )
SCRIPT_DIR=$( cd `dirname $CUR_DIR` ; pwd )
ROOT_DIR=$( cd `dirname $SCRIPT_DIR` ; pwd )
REQUIREMENT_FILE=$ROOT_DIR/requirements.txt
echo "SCRIPT_DIR  :$SCRIPT_DIR"
echo "CUR_DIR     :$CUR_DIR"
echo "ROOT_DIR    :$ROOT_DIR"
DATA_URL=$1
TRAIN_URL=$2
echo "data_url    :$DATA_URL"
echo "train_url   :$TRAIN_URL"
echo "data url sub dirs :-------------------"
ls -ahl $DATA_URL
# check requirements.txt is exists.
if [ ! -f $REQUIREMENT_FILE ];then
  echo  "requirements file : $REQUIREMENT_FILE check : Failed"
  exit 1
fi

#bash $CUR_DIR/apex.sh $ROOT_DIR
#if [ '$?' = 0 ];then
#  exit 1
#fi


pip3 install -r $REQUIREMENT_FILE

if [ '$?' = 0 ];then
  exit 1
fi

echo "------------installed whl:"
pip3 list



