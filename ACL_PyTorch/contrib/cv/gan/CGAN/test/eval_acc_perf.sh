 #!/bin/bash
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "====data pre_treatment starting===="
rm -rf ./prep_dataset
python3.7  CGAN_preprocess.py --save_path ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====data pre_treatment finished===="


echo "====creating info file===="
rm -rf ./CGAN_prep_bin.info
python3.7 gen_dataset_info.py --dataset_bin ./prep_dataset --info_name CGAN_prep_bin.info --width 72 --height 100
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====creating info file done===="


echo "==== msame start===="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf ./out
./msame --model "./CGAN_bs1.om" --input "./prep_dataset/input.bin" --output "./out" --outfmt BIN --loop 1 >>perf.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====msame finished===="


echo "====start post_process===="
python3.7 CGAN_postprocess.py --bin_out_path ./out/20211113_073952 --save_path ./result   
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====end post_process===="
echo "success"
  
  
echo "====performance data===="
python3.7 test/parse.py perf.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"