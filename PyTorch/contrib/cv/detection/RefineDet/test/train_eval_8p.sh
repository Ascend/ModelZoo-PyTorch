data_path="/root/data/VOCdevkit"

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

echo ${data_path}


python eval_refinedet.py './RefineDet320_bn/RefineDet320_VOC_231.pth' ${data_path}
