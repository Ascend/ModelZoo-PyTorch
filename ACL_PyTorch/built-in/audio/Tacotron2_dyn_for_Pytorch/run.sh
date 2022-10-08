SOC_VERSION=$1
bs=$2
device_id=$3

echo "Starting pytorch导出onnx"
python3 tensorrt/cvt_tacotron2onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/ -bs ${bs}
python3 tensorrt/cvt_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 -o output/ --config-file config.json

echo "Starting onnx导出om"
bash atc.sh ${SOC_VERSION} ${bs} 256

echo "Starting 推理tacotron2 om"
python3 val.py -i filelists/ljs_audio_text_test_filelist.txt -bs ${bs} -device_id ${device_id}

echo "Starting 推理waveglow生成wav文件"
python3 val.py -i filelists/ljs_audio_text_test_filelist.txt -o output/audio -bs ${bs} -device_id ${device_id} --gen_wav