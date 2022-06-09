#!/bin/bash

echo "####################################################################"
echo "#                   Start Modelzoo Network Test....                 "
echo "####################################################################"

top_dir=`pwd`

hostname="worker-121-36-69-71"
config_dir=/root/lava_workspace/$hostname
log_dir=/root/lava_workspace/$hostname/log
test_dir=/root/lava_workspace/$hostname/git
modelzoo_dir=$1/modelzoo
ascend310_ip="183.129.213.69"
ascend910_ip="218.2.129.25"

if [ -f $top_dir/result.xml ]
then
  #echo "clear $top_dir/result.xml"
  rm -rf $top_dir/result.xml
fi

if [ -f $top_dir/result.txt ]
then
  #echo "clear $top_dir/result.txt"
  rm -rf $top_dir/result.txt
fi

if [ -f $top_dir/result.bak ]
then
  #echo "clear $top_dir/result.bak"
  rm -rf $top_dir/result.bak
fi

if [ -d $top_dir/log ]
then
  #echo "clear $top_dir/log/*"
  rm -rf $top_dir/log/*
fi

echo "=================Modified files in this PR: ================="
cat $1/pr_filelist.txt



#如果PR只涉及到.MD文件的修改，则无需执行用例，直接返回OK
if [[ `grep -ciE ".MD|.txt|.doc|.docx|LICENSE" "$1/pr_filelist.txt"` -ne '0' && `grep -cE ".py|.sh|.cpp" "$1/pr_filelist.txt"` -eq '0' ]] ;then
   echo "Only .MD|.txt|.doc|.docx|LICENSE in pr_filelist, No need to run testcases!"
   exit 0
fi
#=========功能列表=============
#1、py/cpp文件中license检查
#2、垃圾文件检查,后缀为so/log/h5/event
#3、首层目录必要文件检查：LINCENSE,README.md,requirements.txt
#4、文件大小检查，不超过2M
#5、内部链接扫描
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
python3 access_upline.py --pr_filelist_dir=$1/pr_filelist.txt >$file_log 2>&1
license_check=`grep -ri  "license_check=1" ${file_log} | wc -l`
filesize_check=`grep -ri  "filesize_check=1" ${file_log} | wc -l`
firstlevel_check=`grep -ri  "firstlevel_check=1" ${file_log} | wc -l`
funkfile_check=`grep -ri  "funkfile_check=1" ${file_log} | wc -l`
internal_link_check=`grep -ri  "internal_link_check=1" ${file_log} | wc -l`
sensitive_check=`grep -ri  "sensitive_check=1" ${file_log} | wc -l`
cat $file_log | grep -v "check=1" | grep -v "check=0"
if [[ $license_check -ge 1 || $filesize_check -ge 1 || $firstlevel_check -ge 1 || $funkfile_check -ge 1 ||  $internal_link_check -ge 1 || sensitive_check -ge 1 ]];
then
  echo "check fail"
  exit 1 
else
  echo "check success"
fi
#exit $status


echo "=================Start to Check License ================="
#license检查
lincense_check=0
while read line
do
    a=`echo $line |awk -F "_for_" '{print $1}' | awk -F "/" '{print $NF}'`
    b=`echo $line |awk -F "_for_" '{print $2}' | awk -F "/" '{print $1}'`
    result=`echo $a`_for_`echo $b`
    lise_dir=$(echo ${line%$result*}/$result/LICENSE)
	directory=$(echo ${line%$result*}/$result/)
    if [ -n "$b" ] && [ -d $1/modelzoo/$directory ];
    then
        if [ -f $1/modelzoo/$lise_dir ];
        then
            true
        else
            echo "$result license is not exist!"
			let lincense_check=1
        fi
     else
	    true
        #echo "$result name -ERROR"
    fi
done < $1/pr_filelist.txt


#py/cpp文件检查
while read line
do
    function checkfile()
    {
     result=$(echo $1 | grep -E "\.py|\.cpp" | grep -v "__init__.py")
     if [ -n "$result" ];
     then
         Hw_result=`cat $1 | grep -i "License"`
         if [ -n "$Hw_result" ];
          then
              true
          else
               echo "$1 license check fail!"
			   let lincense_check=1
          fi
      else
          #echo "$1 no need check"
		  true
      fi
    }
    function getAllFiles()
    {
        for fileName in `ls $1`;
        do
           dir_or_file=$1"/"$fileName
           if [ -d $dir_or_file ]
           then
              getAllFiles $dir_or_file
           else
              checkfile $dir_or_file
           fi
         done
    }
    if [ -f $1/modelzoo/$line ];
    then
       #echo $line
       checkfile $1/modelzoo/$line
    else
       getAllFiles $1/modelzoo/$line
    fi

done < $1/pr_filelist.txt

if [ $lincense_check -eq '1' ] ;then
   echo "License check failed, Please follow the guide to add License:"
   echo "https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md"
   exit 1
fi

#如果新增的都是目录，则无需执行用例，直接返回OK
check_res=0
while read line
do 
   if [[ ! $line =~ ".keep" ]];
   then
       let check_res=1
   fi
done < $1/pr_filelist.txt

if [ $check_res -eq '0' ] ;then
   echo "Add directorys in contrib/Research, No need to run testcases!"
   exit 0
fi

#代码安全检查模块
while read line
do 
   if [ -d $1/modelzoo/$line ];
   then
       rmresult=`grep -rn -w "rm " $1/modelzoo/$line/*.sh | wc -l`
       cpresult=`grep -rn -w "cp " $1/modelzoo/$line/*.sh | wc -l`
       toresult=`grep -rn -w "touch " $1/modelzoo/$line/*.sh | wc -l`
       if [ $rmresult -gt 0 ] || [ $cpresult -gt 0 ] || [ $toresult -gt 0 ] ;
       then
           echo "Please do not use rm/cp/touch in .sh, tks!"
       fi
   elif [[ $1/modelzoo/$line =~ ".sh" ]];
   then
       rmresult=`cat $1/modelzoo/$line | grep -w "rm " | wc -l`
       cpresult=`cat $1/modelzoo/$line | grep -w "cp " | wc -l`
       toresult=`cat $1/modelzoo/$line | grep -w "touch " | wc -l`
       if [ $rmresult -gt 0 ] || [ $cpresult -gt 0 ] || [ $toresult -gt 0 ] ;
       then
           echo "Please do not use rm/cp/touch in .sh, tks!"
       fi
    fi
done < $1/pr_filelist.txt

#文件大小检查模块，超过10M则报错
filesize_check=0
maxsize=$((1024*1024*2))
while read line
do 
   if [ -d $1/modelzoo/$line ];
   then
       #PR提交里面不存在目录，如果是空目录，则为.keep
	   echo "directory"
   else
       filesize=`ls -l $1/modelzoo/$line | awk '{ print $5 }'`
       if [[ $filesize -gt $maxsize ]];
	   then
	       echo "File size of $1/modelzoo/$line greater than 2M, Please remove it!"
		   let filesize_check=1
		fi
    fi
done < $1/pr_filelist.txt

if [ $filesize_check -eq '1' ] ;then
   echo "File size check failed, exit!"
   exit 1
fi

#如果PR不涉及contrib/TensorFlow/Research目录，则无需执行用例，直接返回OK
if [ `grep -c "contrib/TensorFlow/Research" "$1/pr_filelist.txt"` -eq '0' ] ;then
   echo "This pr dosn't have changes in contrib/TensorFlow/Research, No need to run testcases!"
   exit 0
fi

date_time=`date +%Y%m%d`"."`date +%H%M%S`
echo "===================================================================="
echo "$date_time : start run test case , please wait ..."
echo "===================================================================="

python3 createCases.py $1/pr_filelist.txt $modelzoo_dir

#如果case.txt中没有生成用例，则报错退出
if [ `grep -c ".sh" "$top_dir/cases.txt"` -eq '0' ] ;then
   echo "No testcases was found, Please check your PR!"
   exit 1
fi

date_time=`date +%Y%m%d`"."`date +%H%M%S`
echo "===================================================================="
echo "$date_time : copy source code to Ascend310&Ascend910 , please wait ..."
echo "===================================================================="

if [ `grep -c "_offline_inference" "$top_dir/cases.txt"` -ne '0' ] ;then
   ./auto_scp.sh "$modelzoo_dir/contrib" "$ascend310_ip" "/home/HwHiAiUser/modelzoo" "Root@123" "22"  >/dev/null 2>&1
fi
if [[ `grep -c "_online_inference" "$top_dir/cases.txt"` -ne '0' || `grep -c "_train" "$top_dir/cases.txt"` -ne '0' ]] ;then
   ./auto_scp.sh "$modelzoo_dir/contrib" "$ascend910_ip" "/home/HwHiAiUser/modelzoo" "Root@123" "7745"  >/dev/null 2>&1
fi

date_time=`date +%Y%m%d`"."`date +%H%M%S`
echo "===================================================================="
echo "$date_time : cat testcase info"
echo "===================================================================="
cat cases.txt 

echo "===================================================================="
num=1
cat cases.txt | while read line
do
  date_time=`date +%Y%m%d`"."`date +%H%M%S`
  echo "$date_time : start run test case num [ $num ] : [ $line ]"
  echo "====================================================================" 
  array=(${line//,/ })
  case=${array[0]}
  echo $case
  
  if [ -f "$test_dir/$case" ]
	then	
    chmod +x $test_dir/$case
    sleep 1
    $test_dir/$line
	date_time=`date +%Y%m%d`"."`date +%H%M%S`
    echo "$date_time : finished test case num [ $num ] : [ $line ]"
    wc -l $top_dir/result.txt
    let num=num+1
    if [ -s $top_dir/result.txt ]
    then
      cp -rf $top_dir/result.txt $top_dir/result.bak
    else
      echo "####################################################################"
  	  echo "$date_time ERROR : Check Test Result FAIL"
  	  echo "ERROR INFO : $top_dir/result.txt is empty , please check..."
      echo "####################################################################"
    fi
    echo "===================================================================="
  else
      echo "####################################################################"
  	  echo "$date_time ERROR : Run Testcase FAIL"
  	  echo "ERROR INFO : Could not find testcase [ $test_dir/$case ]"
      echo "####################################################################"
  fi
done

cp -r $top_dir/log $1/modelzoo_log

date_time=`date +%Y%m%d`"."`date +%H%M%S`
echo "####################################################################"
echo "#                   Modelzoo Network Test Finished!                 "
echo "####################################################################"

if [ `grep -c "fail" "$top_dir/result.txt"` -ne '0' ] ;then
   exit 1
else
  exit 0
fi