rm -rf model_py1.8.onnx
python3.7 ./detectron2/tools/deploy/export_model.py --config-file ./detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS model_final.pkl MODEL.DEVICE cpu
mv ./output/model.onnx model_py1.8.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf retinanet_detectron2_npu.om 
atc --model=model_py1.8.onnx --framework=5 --output=retinanet_detectron2_npu --input_format=NCHW --input_shape="input0:1,3,1344,1344"  --log=debug --soc_version=Ascend310
if [ -f "retinanet_detectron2_npu.om" ] ; then
    echo "success"
else
    echo "fail!"
fi