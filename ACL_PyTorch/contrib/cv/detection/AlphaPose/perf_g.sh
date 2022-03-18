#!/bin/bash

set -eu

trtexec --onnx=./outputs/roberta_base_batch_1_sim.onnx --fp16 --threads
trtexec --onnx=./outputs/roberta_base_batch_4_sim.onnx --fp16 --threads
trtexec --onnx=./outputs/roberta_base_batch_8_sim.onnx --fp16 --threads
trtexec --onnx=./outputs/roberta_base_batch_16_sim.onnx --fp16 --threads
trtexec --onnx=./outputs/roberta_base_batch_32_sim.onnx --fp16 --threads
