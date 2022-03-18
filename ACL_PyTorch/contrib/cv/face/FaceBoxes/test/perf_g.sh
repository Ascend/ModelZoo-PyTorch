#!/bin/bash

trtexec --onnx=faceboxes-b0_bs1.onnx --fp16 --shapes=image:1x3x1024x1024 --threads
trtexec --onnx=faceboxes-b0_bs1.onnx --fp16 --shapes=image:16x3x1024x1024 --threads