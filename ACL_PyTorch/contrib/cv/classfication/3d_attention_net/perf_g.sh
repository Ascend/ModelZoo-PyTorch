#!/bin/bash

trtexec --onnx=3d_attention_net.onnx --fp16 --shapes=image:1x3x32x32
trtexec --onnx=3d_attention_net.onnx --fp16 --shapes=image:16x3x32x32