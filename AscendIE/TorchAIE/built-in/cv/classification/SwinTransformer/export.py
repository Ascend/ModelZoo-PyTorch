import argparse

import timm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='SwinTransformer TorchScript export.')
    parser.add_argument('-n', '--model_name', type=str, default='swin_base_patch4_window12_384',
                        help='model name for swintransformer')
    parser.add_argument('-i', '--image_size', type=int, default=384,
                        help='input image size for swintransformer')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='./swin_base_patch4_window12_384_22kto1k.pth',
                        help='checkpoint path for pth model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = timm.create_model(args.model_name, checkpoint_path=args.checkpoint_path)
    model.eval()

    input = torch.ones(1, 3, args.image_size, args.image_size)
    output = model(input)
    print(f'output shape: {output.shape}')

    script_model = torch.jit.trace(model, input)
    script_model.save(f'{args.model_name}.ts')


if __name__ == '__main__':
    main()
