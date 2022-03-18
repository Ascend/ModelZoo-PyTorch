# on npu
python ./utils/gen_test_data.py ./test/test_data/test_1_3_1500_1500.bin 1,3,1500,1500
python ./utils/gen_test_data.py ./test/test_data/test_4_3_1500_1500.bin 4,3,1500,1500
python ./utils/gen_test_data.py ./test/test_data/test_8_3_1500_1500.bin 8,3,1500,1500

python ./utils/gen_test_data.py ./test/test_data/test_100_3_24_24.bin 100,3,24,24
python ./utils/gen_test_data.py ./test/test_data/test_800_3_24_24.bin 800,3,24,24
python ./utils/gen_test_data.py ./test/test_data/test_1600_3_24_24.bin 1600,3,24,24

python ./utils/gen_test_data.py ./test/test_data/test_50_3_48_48.bin 50,3,48,48
python ./utils/gen_test_data.py ./test/test_data/test_400_3_48_48.bin 400,3,48,48
python ./utils/gen_test_data.py ./test/test_data/test_800_3_48_48.bin 800,3,48,48

./msame --model ./weights/PNet_dynamic.om --input ./test/test_data/test_1_3_1500_1500.bin --dymShape "image:1,3,1500,1500" --outputSize "8880400,4440200"
./msame --model ./weights/PNet_dynamic.om --input ./test/test_data/test_4_3_1500_1500.bin --dymShape "image:4,3,1500,1500" --outputSize "35521600,17760800"
./msame --model ./weights/PNet_dynamic.om --input ./test/test_data/test_8_3_1500_1500.bin --dymShape "image:8,3,1500,1500" --outputSize "71043200,35521600"

./msame --model ./weights/RNet_dynamic.om --input ./test/test_data/test_100_3_24_24.bin --dymShape "image:100,3,24,24" --outputSize "1600,800"
./msame --model ./weights/RNet_dynamic.om --input ./test/test_data/test_800_3_24_24.bin --dymShape "image:800,3,24,24" --outputSize "12800,6400"
./msame --model ./weights/RNet_dynamic.om --input ./test/test_data/test_1600_3_24_24.bin --dymShape "image:1600,3,24,24" --outputSize "25600,12800"

./msame --model ./weights/ONet_dynamic.om --input ./test/test_data/test_50_3_48_48.bin --dymShape "image:50,3,48,48" --outputSize "800,2000,400"
./msame --model ./weights/ONet_dynamic.om --input ./test/test_data/test_400_3_48_48.bin --dymShape "image:400,3,48,48" --outputSize "6400,16000,3200"
./msame --model ./weights/ONet_dynamic.om --input ./test/test_data/test_800_3_48_48.bin --dymShape "image:800,3,48,48" --outputSize "12800,32000,6400"

./benchmark.x86_64 -round=20 -om_path=./weights/Inception_facenet_vggface2_bs1.om -device_id=0 -batch_size=1
./benchmark.x86_64 -round=20 -om_path=./weights/Inception_facenet_vggface2_bs4.om -device_id=0 -batch_size=4
./benchmark.x86_64 -round=20 -om_path=./weights/Inception_facenet_vggface2_bs8.om -device_id=0 -batch_size=8
./benchmark.x86_64 -round=20 -om_path=./weights/Inception_facenet_vggface2_bs16.om -device_id=0 -batch_size=16
./benchmark.x86_64 -round=20 -om_path=./weights/Inception_facenet_vggface2_bs32.om -device_id=0 -batch_size=32


# on gpu
python -m onnxsim ./weights/PNet_truncated_fix.onnx ./weights/PNet_sim_1_3_1500_1500.onnx --input-shape 'image:1,3,1500,1500'
python -m onnxsim ./weights/PNet_truncated_fix.onnx ./weights/PNet_sim_4_3_1500_1500.onnx --input-shape 'image:4,3,1500,1500'
python -m onnxsim ./weights/PNet_truncated_fix.onnx ./weights/PNet_sim_8_3_1500_1500.onnx --input-shape 'image:8,3,1500,1500'

python -m onnxsim ./weights/RNet_truncated_fix.onnx ./weights/RNet_sim_100_3_24_24.onnx --input-shape 'image:100,3,24,24'
python -m onnxsim ./weights/RNet_truncated_fix.onnx ./weights/RNet_sim_800_3_24_24.onnx --input-shape 'image:800,3,24,24'
python -m onnxsim ./weights/RNet_truncated_fix.onnx ./weights/RNet_sim_1600_3_24_24.onnx --input-shape 'image:1600,3,24,24'

python -m onnxsim ./weights/ONet_truncated_fix.onnx ./weights/ONet_sim_50_3_48_48.onnx --input-shape 'image:50,3,48,48'
python -m onnxsim ./weights/ONet_truncated_fix.onnx ./weights/ONet_sim_400_3_48_48.onnx --input-shape 'image:400,3,48,48'
python -m onnxsim ./weights/ONet_truncated_fix.onnx ./weights/ONet_sim_800_3_48_48.onnx --input-shape 'image:800,3,48,48'

trtexec --onnx=./weights/PNet_sim_1_3_1500_1500.onnx --fp16 --shapes=image:1×3×1500×1500
trtexec --onnx=./weights/PNet_sim_4_3_1500_1500.onnx --fp16 --shapes=image:4×3×1500×1500
trtexec --onnx=./weights/PNet_sim_8_3_1500_1500.onnx --fp16 --shapes=image:8×3×1500×1500

trtexec --onnx=./weights/RNet_sim_100_3_24_24.onnx --fp16 --shapes=image:100×3×24×24
trtexec --onnx=./weights/RNet_sim_800_3_24_24.onnx --fp16 --shapes=image:800×3×24×24
trtexec --onnx=./weights/RNet_sim_1600_3_24_24.onnx --fp16 --shapes=image:1600×3×24×24

trtexec --onnx=./weights/ONet_sim_50_3_48_48.onnx --fp16 --shapes=image:50×3×48×48
trtexec --onnx=./weights/ONet_sim_400_3_48_48.onnx --fp16 --shapes=image:400×3×48×48
trtexec --onnx=./weights/ONet_sim_800_3_48_48.onnx --fp16 --shapes=image:800×3×48×48

python -m onnxsim ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_vggface2_sim_1_3_160_160.onnx --input-shape 'image:1,3,160,160'
python -m onnxsim ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_vggface2_sim_4_3_160_160.onnx --input-shape 'image:4,3,160,160'
python -m onnxsim ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_vggface2_sim_8_3_160_160.onnx --input-shape 'image:8,3,160,160'
python -m onnxsim ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_vggface2_sim_16_3_160_160.onnx --input-shape 'image:16,3,160,160'

trtexec --onnx=./weights/Inception_vggface2_sim_1_3_160_160.onnx --fp16 --shapes=image:1×3×160×160
trtexec --onnx=./weights/Inception_vggface2_sim_4_3_160_160.onnx --fp16 --shapes=image:4×3×160×160
trtexec --onnx=./weights/Inception_vggface2_sim_8_3_160_160.onnx --fp16 --shapes=image:8×3×160×160
trtexec --onnx=./weights/Inception_vggface2_sim_16_3_160_160.onnx --fp16 --shapes=image:16×3×160×160
