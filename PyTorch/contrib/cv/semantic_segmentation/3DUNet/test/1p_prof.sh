for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
   data_path="./datasets"
fi

source ./test/env_npu.sh
export PYTHONPATH=/usr/lib:$PYTHONPATH
rm -rf ./runs



python3 -u train_brats2018_new.py --amp --data_path $data_path  --world_size 1 --rank 0 --prof