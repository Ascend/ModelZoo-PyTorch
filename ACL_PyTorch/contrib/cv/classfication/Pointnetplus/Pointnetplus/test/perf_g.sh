#!/bin/bash

# bs1
trtexec --onnx=Pointnetplus_part1_bs1.onnx --fp16 --threads
trtexec --onnx=Pointnetplus_part2_bs1.onnx --fp16 --threads

# bs16
trtexec --onnx=Pointnetplus_part1_bs16.onnx --fp16 --threads
trtexec --onnx=Pointnetplus_part2_bs16.onnx --fp16 --threads
