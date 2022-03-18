#!/bin/bash
trtexec --onnx=st-gcn_kinetics-skeleton_bs1.onnx --fp16 --shapes=image:1x3x300x18x2
trtexec --onnx=st-gcn_kinetics-skeleton_bs4.onnx --fp16 --shapes=image:4x3x300x18x2
trtexec --onnx=st-gcn_kinetics-skeleton_bs8.onnx --fp16 --shapes=image:8x3x300x18x2
trtexec --onnx=st-gcn_kinetics-skeleton_bs16.onnx --fp16 --shapes=image:16x3x300x18x2
trtexec --onnx=st-gcn_kinetics-skeleton_bs32.onnx --fp16 --shapes=image:32x3x300x18x2