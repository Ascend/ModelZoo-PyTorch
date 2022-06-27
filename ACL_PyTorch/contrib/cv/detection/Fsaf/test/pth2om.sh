rm -rf fsaf.onnx
python3.7 ./mmdetection/tools/deployment/pytorch2onnx.py ./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py ./fsaf_r50_fpn_1x_coco-94ccc51f.pth --output-file fsaf.onnx --shape 800 1216 --input-img ./mmdetection/demo/demo.jpg --show
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf fsaf_bs1.om 
atc --framework=5 --model=./fsaf.onnx --output=fsaf_bs1 --input_format=NCHW --input_shape="input:1,3,800,1216" --log=debug --soc_version=Ascend310 --out_nodes="dets;labels"
atc --framework=5 --model=./fsaf.onnx --output=fsaf_bs16 --input_format=NCHW --input_shape="input:16,3,800,1216" --log=debug --soc_version=Ascend310 --out_nodes="dets;labels"
if [ -f "fsaf_bs1.om" ] && [ -f "fsaf_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi