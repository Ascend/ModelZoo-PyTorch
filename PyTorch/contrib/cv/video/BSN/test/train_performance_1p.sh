data_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi


taskset -c 0-23 python3 -u main_1p.py --module PERF --tem_epoch 2 --pem_epoch 2 --data_path ${data_path} > ${test_path_dir}/output/0/train_perfomance_1p.log 2>&1 &

wait

echo "------------------ Final result ------------------"
FPS_TEM=`grep -m 1 'FPS(TEM)'  ${test_path_dir}/output/0/train_perfomance_1p.log | awk -F " " '{print$4}'`
FPS_PEM=`grep -m 1 'FPS(PEM)'  ${test_path_dir}/output/0/train_perfomance_1p.log | awk -F " " '{print$4}'`
echo "Final TEM Performance images/sec : $FPS_TEM"
echo "Final PEM Performance images/sec : $FPS_PEM"
