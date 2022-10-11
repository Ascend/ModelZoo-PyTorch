source /usr/local/Ascend/ascend-toolkit/set_env.sh
python MTCNN_pth2onnx.py --model PNet --output_file ./weights/PNet_truncated.onnx
python MTCNN_pth2onnx.py --model RNet --output_file ./weights/RNet_truncated.onnx
python MTCNN_pth2onnx.py --model ONet --output_file ./weights/ONet_truncated.onnx
python FaceNet_pth2onnx.py --pretrain vggface2 --model ./weights/Inception_facenet_vggface2.pt --output_file ./weights/Inception_facenet_vggface2.onnx

python ./utils/fix_prelu.py ./weights/PNet_truncated.onnx ./weights/PNet_truncated_fix.onnx
python ./utils/fix_prelu.py ./weights/RNet_truncated.onnx ./weights/RNet_truncated_fix.onnx
python ./utils/fix_prelu.py ./weights/ONet_truncated.onnx ./weights/ONet_truncated_fix.onnx
python ./utils/fix_prelu.py ./weights/Inception_facenet_vggface2.onnx ./weights/Inception_facenet_vggface2_fix.onnx
# python ./utils/fix_clip.py ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_facenet_vggface2_fix.onnx

atc --framework=5 --model=./weights/PNet_truncated_fix.onnx --output=./weights/PNet_dynamic --input_format=NCHW --input_shape_range='image:[1~32,3,1~1500,1~1500]' --log=debug --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/RNet_truncated_fix.onnx --output=./weights/RNet_dynamic --input_format=NCHW --input_shape_range='image:[1~2000,3,24,24]' --log=debug --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/ONet_truncated_fix.onnx --output=./weights/ONet_dynamic --input_format=NCHW --input_shape_range='image:[1~1000,3,48,48]' --log=debug --soc_version=Ascend310 --log=error > atc1.log

atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs1 --input_format=NCHW --input_shape="image:1,3,160,160" --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs4 --input_format=NCHW --input_shape="image:4,3,160,160" --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs8 --input_format=NCHW --input_shape="image:8,3,160,160" --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs16 --input_format=NCHW --input_shape="image:16,3,160,160" --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs32 --input_format=NCHW --input_shape="image:32,3,160,160" --soc_version=Ascend310 --log=error > atc1.log

