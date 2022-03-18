if [ ! -d "./checkpoints"]; then 
mkdir "./checkpoints" 
fi 

if [ ! -f "checkpoints/jasper_fp16.pt"]; then 
wget -P ./checkpoints https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/audio/Jasper/jasper_fp16.pt
fi 

rm -rf jasper_1batch.onnx && rm -rf jasper_16batch.onnx

python3.7 Jasper_pth2onnx.py checkpoints/jasper_fp16.pt jasper_1batch.onnx 1
python3.7 Jasper_pth2onnx.py checkpoints/jasper_fp16.pt jasper_16batch.onnx 16

source env.sh

rm -rf jasper_1batch.om && rm -rf jasper_16batch.om

atc --model=jasper_1batch.onnx \
    --framework=5 \
    --input_format=ND \
    --input_shape="feats:1,64,4000;feat_lens:1" \
    --output=jasper_1batch \
    --soc_version=Ascend310 \
    --log=error

atc --model=jasper_16batch.onnx \
    --framework=5 \
    --input_format=ND \
    --input_shape="feats:16,64,4000;feat_lens:1" \
    --output=jasper_6batch \
    --soc_version=Ascend310 \
    --log=error

if [ -f "jasper_1batch.om" ] && [ -f "jasper_16batch.om" ]; then
    echo "success"
else
    echo "fail!"
fi