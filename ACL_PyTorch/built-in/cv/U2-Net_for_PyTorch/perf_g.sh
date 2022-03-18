#!/bin/bash
set -eu


trtexec --onnx=u2net_sim_bs1_fixv2.onnx --fp16 --threads --workspace=30000
trtexec --onnx=u2net_sim_bs4_fixv2.onnx --fp16 --threads --workspace=30000
trtexec --onnx=u2net_sim_bs8_fixv2.onnx --fp16 --threads --workspace=30000
trtexec --onnx=u2net_sim_bs16_fixv2.onnx --fp16 --threads --workspace=30000
trtexec --onnx=u2net_sim_bs32_fixv2.onnx --fp16 --threads --workspace=30000
trtexec --onnx=u2net_sim_bs64_fixv2.onnx --fp16 --threads --workspace=30000
