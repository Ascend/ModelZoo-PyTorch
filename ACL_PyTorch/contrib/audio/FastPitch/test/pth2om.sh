source /usr/local/Ascend/ascend-toolkit/set_env.sh

python pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 1
python pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 4
python pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 8
python pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 16
python pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 32

python -m onnxsim ./test/models/FastPitch_bs1.onnx ./test/models/FastPitch_bs1_sim.onnx
python -m onnxsim ./test/models/FastPitch_bs4.onnx ./test/models/FastPitch_bs4_sim.onnx
python -m onnxsim ./test/models/FastPitch_bs8.onnx ./test/models/FastPitch_bs8_sim.onnx
python -m onnxsim ./test/models/FastPitch_bs16.onnx ./test/models/FastPitch_bs16_sim.onnx
python -m onnxsim ./test/models/FastPitch_bs32.onnx ./test/models/FastPitch_bs32_sim.onnx

atc --framework=5 --model=./test/models/FastPitch_bs1_sim.onnx --output=./test/models/FastPitch_bs1 --input_format=ND --input_shape="input:1,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend310
atc --framework=5 --model=./test/models/FastPitch_bs4_sim.onnx --output=./test/models/FastPitch_bs4 --input_format=ND --input_shape="input:4,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend310
atc --framework=5 --model=./test/models/FastPitch_bs8_sim.onnx --output=./test/models/FastPitch_bs8 --input_format=ND --input_shape="input:8,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend310
atc --framework=5 --model=./test/models/FastPitch_bs16_sim.onnx --output=./test/models/FastPitch_bs16 --input_format=ND --input_shape="input:16,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend310
atc --framework=5 --model=./test/models/FastPitch_bs32_sim.onnx --output=./test/models/FastPitch_bs32 --input_format=ND --input_shape="input:32,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend310

