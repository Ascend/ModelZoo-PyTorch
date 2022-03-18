# encoding=utf-8

trtexec --onnx=./outputs/albert_bs1s.onnx --fp16 --threads
trtexec --onnx=./outputs/albert_bs4s.onnx --fp16 --threads
trtexec --onnx=./outputs/albert_bs8s.onnx --fp16 --threads
trtexec --onnx=./outputs/albert_bs16s.onnx --fp16 --threads
trtexec --onnx=./outputs/albert_bs32s.onnx --fp16 --threads
