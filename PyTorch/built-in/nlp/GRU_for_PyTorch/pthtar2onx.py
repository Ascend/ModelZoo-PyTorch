import torch
import torch.onnx
import numpy

from collections import OrderedDict
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq

# hyperparameter
CALCULATE_DEVICE = "npu:0"
device = CALCULATE_DEVICE
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
INPUT_DIM = 7854
OUTPUT_DIM = 5893
CLIP = 1
MAX = 2147483647
seed_init = 0


def gen_seeds(num):
    return torch.randint(1, MAX, size=(num,), dtype=torch.float)


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert():
    checkpoint = torch.load("seq2seq-gru-model.pth.tar", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint)

    # model backbone
    seed_init = gen_seeds(32 * 1024 * 12)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT, seed=seed_init)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, seed=seed_init)
    model = Seq2Seq(enc, dec, device)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    # input
    input_names = ["src"]
    output_names = ["trg"]
    src = numpy.random.randint(0, 7854, (46, 512))
    trg = numpy.random.randint(0, 5893, (46, 512))
    src = torch.LongTensor(src)
    trg = torch.LongTensor(trg)

    # to onnx
    torch.onnx.export(model, (src, trg), "gru.onnx", verbose=True, input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert()
