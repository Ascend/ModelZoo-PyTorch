trtexec --onnx=/home/common/vit/vit_bs1_sim.onnx --fp16 --shapes="image:1x3x224x224" --threads
trtexec --onnx=/home/common/vit/vit_bs16_sim.onnx --fp16 --shapes="image:16x3x224x224" --threads