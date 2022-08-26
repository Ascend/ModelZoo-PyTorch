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

source /usr/local/Ascend/ascend-toolkit/set_env.sh


echo "=======get bin======="
rm -rf ssd_bin

python ssd_preprocess.py --data=${data_path}/coco --bin-output=./ssd_bin

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "=======ais_infer======="
# rm -rf result
cd tools/ais-bench_workload/tool/ais_infer/

python ais_infer.py --model ../../../../ssd_bs1.om --input ../../../../ssd_bin/ --output ../../../../

cd ../../../../

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd tools/ais-bench_workload/tool/ais_infer/

python ais_infer.py --model ../../../../ssd_bs16.om --input ../../../../ssd_bin/ --output ../../../../

cd ../../../../

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "=======eval bs1======="

echo "please run *python ssd_postprocess.py --data=${data_path}/coco --bin-input=./20xx(year)_xx(month)_xx(day)-xx_xx/ "

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

