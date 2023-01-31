nohup python3 eval_onnx.py \
      --cuda  \
      --data=data/enwik \
      --batch_size=${1} \
      --split=valid \
      --onnx_file=${2} > onnx_gpu_perf.log 2>&1 &
