data_path="/root/data/VOCdevkit"

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

echo ${data_path}



python -u train_1p.py \
    --dataset_root ${data_path} \
    --save_folder ./RefineDet320/ \
    --num_workers 8 \
    --batch_size 32 \
    --num_epochs 232 \
    --amp True \
    --device_id 1 \

