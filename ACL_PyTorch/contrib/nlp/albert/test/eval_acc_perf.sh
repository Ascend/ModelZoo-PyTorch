# encoding=utf-8

source env.sh
# make sure benchmark.x86_64 is in the dir
echo batch_size 1
./benchmark.x86_64 -model_type=bert -device_id=0 -batch_size=1 -om_path=./outputs/albert_bs1s.om -input_text_path=albert.info -output_binary=false
python3.7 Albert_postprocess.py --dump_output=./result/dumpOutput_device0 --dump_perf=./result/perf_bert_batchsize_1_device_0.txt

echo batch_size 4
./benchmark.x86_64 -model_type=bert -device_id=0 -batch_size=4 -om_path=./outputs/albert_bs4s.om -input_text_path=albert.info -output_binary=false
python3.7 Albert_postprocess.py --dump_output=./result/dumpOutput_device0 --dump_perf=./result/perf_bert_batchsize_4_device_0.txt

echo batch_size 8
./benchmark.x86_64 -model_type=bert -device_id=0 -batch_size=8 -om_path=./outputs/albert_bs8s.om -input_text_path=albert.info -output_binary=false
python3.7 Albert_postprocess.py --dump_output=./result/dumpOutput_device0 --dump_perf=./result/perf_bert_batchsize_8_device_0.txt

echo batch_size 16
./benchmark.x86_64 -model_type=bert -device_id=0 -batch_size=16 -om_path=./outputs/albert_bs16s.om -input_text_path=albert.info -output_binary=false
python3.7 Albert_postprocess.py --dump_output=./result/dumpOutput_device0 --dump_perf=./result/perf_bert_batchsize_16_device_0.txt

echo batch_size 32
./benchmark.x86_64 -model_type=bert -device_id=0 -batch_size=32 -om_path=./outputs/albert_bs32s.om -input_text_path=albert.info -output_binary=false
python3.7 Albert_postprocess.py --dump_output=./result/dumpOutput_device0 --dump_perf=./result/perf_bert_batchsize_32_device_0.txt
