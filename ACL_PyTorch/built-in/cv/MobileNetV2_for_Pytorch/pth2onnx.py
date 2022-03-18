import torch
import torch.onnx
import torchvision.models as models


def convert():
    model = models.mobilenet_v2(pretrained=False)
    pthfile = './mobilenet_v2-b0353104.pth'
    mobilenet_v2 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(mobilenet_v2)
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input,
        "mobilenet_v2_16.onnx",
        input_names=input_names, 
        output_names=output_names, 
        opset_version=11)


if __name__ == "__main__":
    convert()

