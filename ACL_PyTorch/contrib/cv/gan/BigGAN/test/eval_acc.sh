#!/bin/bash

source env.sh

# generate prep_dataset
rm -rf ./postprocess_img gen_img_bs1.npz gen_img_bs16.npz
python3.7 biggan_postprocess.py --result-path "./outputs_bs1_om" --batch-size 1 --save-img --save-npz
python3.7 biggan_postprocess.py --result-path "./outputs_bs16_om" --batch-size 16 --save-img --save-npz


# IS FID
rm -rf biggan_acc_eval_bs1.log
python3.7 -u biggan_eval_acc.py --num-inception-images 50000 --batch-size 1 > biggan_acc_eval_bs1.log
if [ $? != 0 ]; then
    echo "generate pth result fail!"
    exit -1
fi

rm -rf biggan_acc_eval_bs16.log
python3.7 -u biggan_eval_acc.py --num-inception-images 50000 --batch-size 16 > biggan_acc_eval_bs16.log
if [ $? != 0 ]; then
    echo "generate pth result fail!"
    exit -1
fi

echo "----bs1 acc result----"
cat biggan_acc_eval_bs1.log
echo "----bs16 acc result----"
cat biggan_acc_eval_bs16.log
echo "success"
