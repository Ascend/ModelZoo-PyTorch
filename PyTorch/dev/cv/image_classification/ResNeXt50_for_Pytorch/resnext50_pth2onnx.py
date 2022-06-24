import sys
import ssl
import torch
import torch.onnx
import torchvision.models as models

def convert():
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    if (len(sys.argv) == 3):
        model = models.resnext50_32x4d(pretrained=False)
        checkpoint = torch.load(input_file, map_location=None)
        model.load_state_dict(checkpoint)
    else:
        model = models.resnext50_32x4d(pretrained=True)

    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11)

if __name__ == "__main__":
    if (len(sys.argv) == 3):
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        output_file = "./resnext50.onnx"
    ssl._create_default_https_context = ssl._create_unverified_context
    convert()
