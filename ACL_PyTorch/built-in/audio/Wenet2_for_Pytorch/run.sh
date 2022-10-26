soc_version=$1
rm -rf onnx
python3 wenet/bin/export_onnx_npu.py --config exp/20210601_u2++_conformer_exp/train.yaml --checkpoint exp/20210601_u2++_conformer_exp/final.pt --output_onnx_dir \
./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 
python3 wenet/bin/export_onnx_npu.py --config exp/20210601_u2++_conformer_exp/train.yaml --checkpoint exp/20210601_u2++_conformer_exp/final.pt --output_onnx_dir \
./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 --streaming
rm offline_encoder.om online_encoder.om encoder_static.om
atc --model=./onnx/offline_encoder.onnx --framework=5 --output=offline_encoder --input_format=ND \
--input_shape_range="speech:[1~64,1~1500,80];speech_lengths:[1~64]" --log=error  --soc_version=$soc_version
atc --model=./onnx/online_encoder.onnx --framework=5 --output=online_encoder --input_format=ND \
--input_shape="chunk_xs:64,67,80;chunk_lens:64;offset:64,1;att_cache:64,12,4,64,128;cnn_cache:64,12,256,7;cache_mask:64,1,64" \
--log=error  --soc_version=$soc_version
atc --model=./onnx/offline_encoder.onnx --framework=5 --output=encoder_static --input_format=ND \
--input_shape="speech:32,-1,80;speech_lengths:32" --log=error \
--dynamic_dims="262;326;390;454;518;582;646;710;774;838;902;966;1028;1284;1478" \
--soc_version=$soc_version
rm -rf offline_wer static_wer offline_test_result.txt offline_res_result.txt onlinelog.log online_test_result.txt static_res_result.txt static_test_result.txt
# offline
python3 wenet/bin/recognize_om.py --config=exp/20210601_u2++_conformer_exp/train.yaml --test_data=exp/20210601_u2++_conformer_exp/data.list --dict=exp/20210601_u2++_conformer_exp/units.txt --mode=ctc_greedy_search \
--result_file=offline_res_result.txt --encoder_om=offline_encoder.om --decoder_om=xx.om --batch_size=1 --device_id=0 --test_file=offline_test_result.txt
python3 tools/compute-wer.py --char=1 --v=1 exp/20210601_u2++_conformer_exp/text offline_res_result.txt > offline_wer
value_info=`cat offline_wer | grep Overall | awk -F ' ' '{print $3}' `
echo "acc: " $value_info >> offline_test_result.txt
 

arch=`uname -m`
#online
./benchmark.${arch} -batch_size=64 -om_path=online_encoder.om -round=1000 -device_id=0 > onlinelog.log
value_info=`cat onlinelog.log | grep ave_throughputRate | awk -F ': ' '{print $2}' | tr -cd "[0-9|.]"`
echo "perf: " $value_info > online_test_result.txt
python3 cosine_similarity.py >> online_test_result.txt

#static
python3 wenet/bin/recognize_om.py --config=exp/20210601_u2++_conformer_exp/train.yaml --test_data=exp/20210601_u2++_conformer_exp/data.list --dict=exp/20210601_u2++_conformer_exp/units.txt --mode=ctc_greedy_search \
--result_file=static_res_result.txt --encoder_om=encoder_static.om --decoder_om=xx.om --batch_size=32 --device_id=0 --static --test_file=static_test_result.txt
python3 tools/compute-wer.py --char=1 --v=1 exp/20210601_u2++_conformer_exp/text static_res_result.txt > static_wer
value_info=`cat static_wer | grep Overall | awk -F ' ' '{print $3}' `
echo "acc: " $value_info >> static_test_result.txt


