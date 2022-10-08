#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# generate prep_dataset
rm -rf ./pth_result/
python3.7 dcgan_pth_result.py --checkpoint_path ./checkpoint-amp-epoch_200.pth \
                           --dataset_path ./prep_dataset/ \
                           --save_path ./pth_result/
if [ $? != 0 ]; then
    echo "generate pth result fail!"
    exit -1
fi
# om模型batchsize1生成结果与基准对比结果
rm -rf ./dcgan_acc_eval_bs1.log
python3.7 dcgan_acc_eval.py --pth_result_path ./pth_result/ \
                            --om_result_path ./result/dumpOutput_device0/ \
                            --log_save_name ./dcgan_acc_eval_bs1.log
if [ $? != 0 ]; then
    echo "bs1 comparison fail!"
    exit -1
fi
# om模型batchsize16生成结果与基准对比结果
rm -rf ./dcgan_acc_eval_bs16.log
python3.7 dcgan_acc_eval.py --pth_result_path ./pth_result/ \
                            --om_result_path ./result/dumpOutput_device1/ \
                            --log_save_name ./dcgan_acc_eval_bs16.log
if [ $? != 0 ]; then
    echo "bs16 comparison fail!"
    exit -1
fi
echo "----bs1 acc result----"
cat ./dcgan_acc_eval_bs1.log
echo "----bs16 acc result----"
cat ./dcgan_acc_eval_bs16.log
echo "success"