python MTCNN_preprocess.py --model Pnet --data_dir ./data/lfw --batch_size 16
python MTCNN_preprocess.py --model Rnet --data_dir ./data/lfw --batch_size 16
python MTCNN_preprocess.py --model Onet --data_dir ./data/lfw --batch_size 16

python FaceNet_preprocess.py --crop_dir ./data/lfw_split_om_cropped_16 --save_dir ./data/input/Facenet_16 --batch_size 16

./msame --model ./weights/Inception_facenet_vggface2_bs16.om --input ./data/input/Facenet_16/xb_results --output ./data/output
mv ./data/output/2022* ./data/output/Facenet_vggface2_16
mkdir ./data/output/Facenet_vggface2_16_2
python ./utils/batch_utils.py --batch_size 16 --data_root_path ./data/output/Facenet_vggface2_16 --save_root_path ./data/output/Facenet_vggface2_16_2
python FaceNet_postprocess.py  --ONet_output_dir ./data/output/split_bs16/onet.json --test_dir ./data/output/Facenet_vggface2_16_2 --crop_dir ./data/lfw_split_om_cropped_16
