echo 'pth to onnx start'
python3  CycleGAN_onnx_export.py \
--model_ga_path=./checkpoints/maps_cycle_gan/latest_net_G_A.pth \
--model_gb_path=./checkpoints/maps_cycle_gan/latest_net_G_B.pth \
--onnx_path=./onnxmodel/ \
--model_ga_onnx_name=model_Ga.onnx \
--model_gb_onnx_name=model_Gb.onnx 
echo 'onnx to om start'
echo "atc Cons_Ga_aipp512_b0_bs1.om start"
atc --framework=5 --model=./onnxmodel/model_Ga.onnx --output=Cons_Ga_aipp512_b0_bs1 --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config
echo "atc Cons_Gb_aipp512_b0_bs1.om start"
atc --framework=5 --model=./onnxmodel/model_Gb.onnx --output=Cons_Gb_aipp512_b0_bs1 --input_format=NCHW --input_shape="img_maps_sat:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config
echo "Cons_Ga_aipp512_b0_bs1.om benchmark  start"