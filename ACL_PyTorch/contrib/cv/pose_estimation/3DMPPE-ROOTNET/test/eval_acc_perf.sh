arch=`uname -m`
datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

rm -rf data_image_bs1 data_cam_bs1 data_image_bs16 data_cam_bs16
python3.7 3DMPPE-ROOTNET_preprocess.py --img_path=${datasets_path}/MuPoTS/MultiPersonTestSet --ann_path=${datasets_path}/MuPoTS/MuPoTS-3D.json --save_path_image=data_image_bs1 --save_path_cam=data_cam_bs1 --inference_batch_siz=1
python3.7 3DMPPE-ROOTNET_preprocess.py --img_path=${datasets_path}/MuPoTS/MultiPersonTestSet --ann_path=${datasets_path}/MuPoTS/MuPoTS-3D.json --save_path_image=data_image_bs16 --save_path_cam=data_cam_bs16 --inference_batch_siz=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf out_bs1 out_bs16
./msame --model 3DMPPE-ROOTNET_bs1.om --input data_image_bs1,data_cam_bs1 --output out_bs1 --outfmt BIN --device 0
./msame --model 3DMPPE-ROOTNET_bs16.om --input data_image_bs16,data_cam_bs16 --output out_bs16 --outfmt BIN --device 0
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 3DMPPE-ROOTNET_postprocess.py --input_path=out_bs1 --result_file=result_bs1
python3.7 3DMPPE-ROOTNET_postprocess.py --input_path=out_bs16 --result_file=result_bs16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
./benchmark.${arch} -round=20 -om_path=3DMPPE-ROOTNET_bs1.om -device_id=0 -batch_size=1
./benchmark.${arch} -round=20 -om_path=3DMPPE-ROOTNET_bs16.om -device_id=0 -batch_size=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_bs1/result_score.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16/result_score.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/PureInfer_perf_of_3DMPPE-ROOTNET_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/PureInfer_perf_of_3DMPPE-ROOTNET_bs16_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"