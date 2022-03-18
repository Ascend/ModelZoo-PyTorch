trtexec --onnx=./pointnet_bs1_sim.onnx --fp16 --shapes=image:1x3x2500 --threads
trtexec --onnx=./pointnet_bs4_sim.onnx --fp16 --shapes=image:4x3x2500 --threads
trtexec --onnx=./pointnet_bs8_sim.onnx --fp16 --shapes=image:8x3x2500 --threads
trtexec --onnx=./pointnet_bs16_sim.onnx --fp16 --shapes=image:16x3x2500 --threads
trtexec --onnx=./pointnet_bs32_sim.onnx --fp16 --shapes=image:32x3x2500 --threads
