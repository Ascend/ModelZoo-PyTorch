#! /bin/bash
./trtexec --onnx=model/model_best_bs1_sim.onnx --fp16 --shapes=image:1x3x112x112 --device=0 > sim_onnx_bs1.log;
./trtexec --onnx=model/model_best_bs16_sim.onnx --fp16 --shapes=image:16x3x112x112 --device=0 > sim_onnx_bs16.log;
echo "====accuracy data===="

bs1_mean=`cat sim_onnx_bs1.log | grep 'mean' | tail -n 1 | awk '{print $4}'`
bs1_fps=`awk 'BEGIN{printf "%.2f\n", 1*1000/'${bs1_mean}'}'`
echo "bs1_fps: ${bs1_fps}"

bs16_mean=`cat sim_onnx_bs16.log | grep 'mean' | tail -n 1 | awk '{print $4}'`
bs16_fps=`awk 'BEGIN{printf "%.2f\n", 16*1000/'${bs16_mean}'}'`
echo "bs16_fps: ${bs16_fps}"