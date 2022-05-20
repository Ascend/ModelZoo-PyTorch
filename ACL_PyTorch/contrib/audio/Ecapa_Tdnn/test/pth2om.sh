python pytorch2onnx.py checkpoint.pt ecapa_tdnn.onnx 

python fix_conv1d.py ecapa_tdnn.onnx ecapa_tdnn_sim.onnx


echo om_bs=1
atc --framework=5 --model=/home/zhn/ecapa_tdnn_sim.onnx --output=om1/ecapa_tdnn_bs1 --input_format=ND --input_shape="mel:1,80,200" --log=debug --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend710>after_bs1.log

echo om_bs=4
atc --framework=5 --model=/home/zhn/ecapa_tdnn_sim.onnx --output=om1/ecapa_tdnn_bs4 --input_format=ND --input_shape="mel:4,80,200" --log=debug --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend710>after_bs4.log  

echo om_bs=8
atc --framework=5 --model=/home/zhn/ecapa_tdnn_sim.onnx --output=om1/ecapa_tdnn_bs8 --input_format=ND --input_shape="mel:8,80,200" --log=debug --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend710>after_bs8.log

echo om_bs=16
atc --framework=5 --model=/home/zhn/ecapa_tdnn_sim.onnx --output=om1/ecapa_tdnn_bs16 --input_format=ND --input_shape="mel:16,80,200" --log=debug --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend710>after_bs16.log 

echo om_bs=32
atc --framework=5 --model=/home/zhn/ecapa_tdnn_sim.onnx --output=om1/ecapa_tdnn_bs32 --input_format=ND --input_shape="mel:32,80,200" --log=debug --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend710>after_bs32.log 

echo om_bs=64
atc --framework=5 --model=/home/zhn/ecapa_tdnn_sim.onnx --output=om1/ecapa_tdnn_bs64 --input_format=ND --input_shape="mel:64,80,200" --log=debug --fusion_switch_file=fusion_switch.cfg --soc_version=Ascend710>after_bs64.log 