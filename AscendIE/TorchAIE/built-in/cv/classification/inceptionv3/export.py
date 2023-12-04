import ssl
import sys

import torch
import torchvision.models as models


def convert(checkpoint=None, output_file='./inceptionv3.pt'):
    if (checkpoint):
        model = models.inception_v3(
            pretrained=False,
            transform_input=True,
            init_weights=False
        )
        checkpoint = torch.load(checkpoint, map_location=None)
        model.load_state_dict(checkpoint)
    else:
        model = models.inception_v3(pretrained=True)

    model.eval()

    dummy_input = torch.randn(1, 3, 299, 299)
    ts_model = torch.jit.trace(model,dummy_input)
    ts_model.save(output_file)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('data preprocess.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to PyTorch pretrained file(.pth)')
    parser.add_argument('--output', type=str, default='./inceptionv3.pt',
                        help='path to save ts model(.pt)')
    args = parser.parse_args()

    ssl._create_default_https_context = ssl._create_unverified_context
    convert(args.checkpoint, args.output)
