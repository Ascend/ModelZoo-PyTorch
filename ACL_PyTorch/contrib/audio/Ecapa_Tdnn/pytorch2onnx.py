import torch
import sys

from ECAPA_TDNN.main import  ECAPA_TDNN,load_checkpoint
from torch import optim
from functools import partial

def pth2onnx(checkpoint, output_file):
    device = torch.device('cpu')
    model = ECAPA_TDNN(1211, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-5)

    model, optimizer, step = load_checkpoint(model, optimizer, checkpoint, rank='cpu')

    model.forward = partial(model.forward, infer = True)
    # 调整模型为eval mode
    model.eval()
    # 输入节点名
    #input_names = ["mel","speaker"]
    input_names = ["mel"]
    # 输出节点名
    output_names = ["output1","output2"]
    dynamic_axes = {'mel': {0: '-1'}, 'output1': {0: '-1'},'output2': {0: '-1'}}
    dummy_input1 = torch.randn(1,80,200).to(device)
    
    torch.onnx.export(model, dummy_input1, output_file, input_names=input_names, dynamic_axes=dynamic_axes,output_names=output_names, opset_version=12, verbose=True)



if __name__ == "__main__":
    checkpoint = sys.argv[1]
    save_path = sys.argv[2]
    pth2onnx(checkpoint, save_path)
