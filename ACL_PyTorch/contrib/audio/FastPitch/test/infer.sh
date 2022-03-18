source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod +x ./test/benchmark.x86_64


cd test
./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=1 -om_path=./models/FastPitch_bs1.om -input_text_path=./input_bin_info.info -output_binary=True -useDvpp=False
cd ..
echo batch_size=1
python3.7 infer_test.py


cd test
./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=4 -om_path=./models/FastPitch_bs4.om -input_text_path=./input_bin_info.info -output_binary=True -useDvpp=False
cd ..
echo batch_size=4
python3.7 infer_test.py


cd test
./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=8 -om_path=./models/FastPitch_bs8.om -input_text_path=./input_bin_info.info -output_binary=True -useDvpp=False
cd ..
echo batch_size=8
python3.7 infer_test.py


cd test
./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=16 -om_path=./models/FastPitch_bs16.om -input_text_path=./input_bin_info.info -output_binary=True -useDvpp=False
cd ..
echo batch_size=16
python3.7 infer_test.py


cd test
./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=32 -om_path=./models/FastPitch_bs32.om -input_text_path=./input_bin_info.info -output_binary=True -useDvpp=False
cd ..
echo batch_size=32
python3.7 infer_test.py