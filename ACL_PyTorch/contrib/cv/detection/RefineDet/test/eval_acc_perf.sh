source /usr/local/Ascend/ascend-toolkit/set_env.sh

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done




python3.7 RefineDet_preprocess.py  ${datasets_path} voc07test_bin

python3.7 get_info.py voc07test_bin voc07test.info


python3.7 get_prior_data.py


rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
rm -rf result/dumpOutput_device0_bs16

./benchmark.x86_64  -model_type=vision \
-device_id=0 \
-batch_size=1 \
-om_path=./refinedet_voc_320_non_nms_bs1.om \
-input_text_path=./voc07test.info \
-input_width=320 -input_height=320 \
-output_binary=True \
-useDvpp=False

mv result/dumpOutput_device0 result/dumpOutput_device0_bs1
 python3.7 RefineDet_postprocess.py --datasets_path ${datasets_path} --result_path .result/dumpOutput_device0_bs1 > result_bs1.json

./benchmark.x86_64  -model_type=vision \
-device_id=0 \
-batch_size=16 \
-om_path=./refinedet_voc_320_non_nms_bs16.om \
-input_text_path=./voc07test.info \
-input_width=320 -input_height=320 \
-output_binary=True \
-useDvpp=False


mv result/dumpOutput_device0 result/dumpOutput_device0_bs16
 python3.7 RefineDet_postprocess.py --datasets_path ${datasets_path} --result_path .result/dumpOutput_device0_bs16 > result_bs16.json

echo "====accuracy data===="
python3.7 test/parse.py result_bs1.json

python3.7 test/parse.py result_bs16.json

echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt

python3.7 test/parse.py result/perf_vision_batchsize_16_device_0.txt

echo "success"