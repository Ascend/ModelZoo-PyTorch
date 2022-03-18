#!/bin/bash

trtexec --onnx=transpose_sim_bs1.onnx --fp16 --shapes=image:1x3x156x192
trtexec --onnx=transpose_sim_bs16.onnx --fp16 --shapes=image:16x3x156x192