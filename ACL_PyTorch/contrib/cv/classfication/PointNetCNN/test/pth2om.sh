rm -rf pointnetcnn.onnx
python3.7 PointNetCNN_pth2onnx.py  pointcnn_epoch240.pth pointnetcnn.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 PointNetCNN_DelSplit.py
python3.7 PointNetCNN_DelConcat.py
python3.7 -m onnxsim pointnetcnn.onnx pointnetcnn_sim.onnx  --input-shape P_sampled:1,1024,3 P_patched:1,1024,3
rm -rf pointnetcnn_bs1.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=./pointnetcnn_sim.onnx --input_format=ND --input_shape="P_sampled:1,1024,3;P_patched:1,1024,3" --output=pointnetcnn_bs1 --log=error --soc_version=Ascend310
if  [ -f "pointnetcnn_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
