import torch
import crnn
import onnx
import torch.onnx

def convert():
    checkpoint = torch.load("./checkpoint_16_CRNN_acc_0.7963.pth", map_location='cpu')
    model = crnn.CRNN(32,1,37,256)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 1, 32, 100)
    dynamic_axes = {'actual_input_1':{0:'-1'},'output1':{1:'-1'}}
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, "crnn_npu_dy.onnx", input_names=input_names,dynamic_axes = dynamic_axes, output_names=output_names, opset_version=11)


if __name__ == "__main__":
    convert()
