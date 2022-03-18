trtexec --onnx=AlignedReID_bs1.onnx --fp16 --shapes=image:1x3x256x128
trtexec --onnx=AlignedReID_bs4.onnx --fp16 --shapes=image:4x3x256x128
trtexec --onnx=AlignedReID_bs8.onnx --fp16 --shapes=image:8x3x256x128
trtexec --onnx=AlignedReID_bs16.onnx --fp16 --shapes=image:16x3x256x128
trtexec --onnx=AlignedReID_bs32.onnx --fp16 --shapes=image:32x3x256x128