import torch
import torchvision.models as models


def convert():
    model = models.densenet121(pretrained=False)
    pthfile = './densenet121-a639ec97.pth'
    densenet121 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(densenet121)
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input,
        "densenet121.onnx",
        input_names=input_names, 
        output_names=output_names, 
        opset_version=11)


if __name__ == "__main__":
    convert()

