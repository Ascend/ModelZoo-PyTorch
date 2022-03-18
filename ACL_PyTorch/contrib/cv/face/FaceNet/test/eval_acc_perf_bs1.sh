python MTCNN_preprocess.py --model Pnet --data_dir ./data/lfw --batch_size 1
python MTCNN_preprocess.py --model Rnet --data_dir ./data/lfw --batch_size 1
python MTCNN_preprocess.py --model Onet --data_dir ./data/lfw --batch_size 1
python FaceNet_preprocess.py --crop_dir ./data/lfw_split_om_cropped_1 --save_dir ./data/input/Facenet_1 --batch_size 1

./msame --model ./weights/Inception_facenet_vggface2_bs1.om --input ./data/input/Facenet_1/xb_results --output ./data/output
mv ./data/output/2022* ./data/output/Facenet_vggface2_1
mkdir ./data/output/Facenet_vggface2_1_2
python ./utils/batch_utils.py --batch_size 1 --data_root_path ./data/output/Facenet_vggface2_1 --save_root_path ./data/output/Facenet_vggface2_1_2
python FaceNet_postprocess.py  --ONet_output_dir ./data/output/split_bs1/onet.json --test_dir ./data/output/Facenet_vggface2_1_2 --crop_dir ./data/lfw_split_om_cropped_1
