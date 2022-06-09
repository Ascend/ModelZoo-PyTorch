#!/bin/bash
id_dir='/home/jenkins/share-data/gitee/ascend/modelzoo/code/compile/20_bak'
file_log='log1/py_train.log'
dir_log='log1'
if [ -d $dir_log ];
then
  if [ -f $file_log ];
  then
    rm -f $file_log
  fi
else
  mkdir $dir_log
fi
#=========功能列表=============
#1、py/cpp文件中license检查
#2、垃圾文件检查,后缀为so/log/h5/event
#3、首层目录必要文件检查：LINCENSE,README.md,requirements.txt
#4、文件大小检查，不超过2M
#5、内部链接扫描
#6、敏感信息扫描，如wx/00开头的工号
#7、网络功能、性能、精度扫描；
#py功能实现
python3 access_upline.py --pr_filelist_dir=$id_dir/pr_filelist.txt >$file_log 2>&1
#结果呈现
license_check=`grep -ri  "license_check=1" ${file_log} | wc -l`
filesize_check=`grep -ri  "filesize_check=1" ${file_log} | wc -l`
firstlevel_check=`grep -ri  "firstlevel_check=1" ${file_log} | wc -l`
funkfile_check=`grep -ri  "funkfile_check=1" ${file_log} | wc -l`
internal_link_check=`grep -ri  "internal_link_check=1" ${file_log} | wc -l`
sensitive_check=`grep -ri  "sensitive_check=1" ${file_log} | wc -l`
modelzoo_level_check=`grep -ri  "modelzoo_level_check=1" ${file_log} | wc -l`
#echo "========== $sensitive_check"
cat $file_log | grep -v "check=1" | grep -v "check=0"
if [[ $license_check -ge 1 || $filesize_check -ge 1 || $firstlevel_check -ge 1 || $funkfile_check -ge 1 || $internal_link_check -ge 1 || $sensitive_check -ge 1 || modelzoo_level_check -ge 1 ]];
then
  echo "check fail"
  exit 1
else
  echo "check success"
fi

