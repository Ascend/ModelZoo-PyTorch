from allennlp.modules.elmo import Elmo
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', default='elmo.onnx')
    parser.add_argument('word_len', default=8, type=int)
    opt = parser.parse_args()
    pth2onnx(opt)
    

def pth2onnx(opt):
    batch_size = 1
    options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1)
    elmo.eval()
    dummy_input = torch.randint(1, 10, (batch_size, opt.word_len, 50), dtype=torch.int32)
    torch.onnx.export(elmo, dummy_input, opt.output_file, input_names=["input"], output_names=["output"], opset_version=11, verbose=True)


if __name__ == '__main__':
        main()
