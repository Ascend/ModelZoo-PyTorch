import sys
import torch

from tool.utils import *
from models import Yolov4


def transform_to_ts(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):

    model = Yolov4(n_classes=n_classes, inference=False)

    pretrained_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)
    model.eval()

    ts_file_name = "yolov4_{}_3_{}_{}.ts".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
    # Export the model
    print('Export the torch script model ...')
    input_data = torch.ones(batch_size, 3, IN_IMAGE_H, IN_IMAGE_W)
    ts_model = torch.jit.trace(model, input_data)
    ts_model.save(ts_file_name)
    print('ts model exporting done')

if __name__ == '__main__':
    print("Converting to torch script ...")
    if len(sys.argv) == 6:

        weight_file = sys.argv[1]
        batch_size = int(sys.argv[2])
        n_classes = int(sys.argv[3])
        IN_IMAGE_H = int(sys.argv[4])
        IN_IMAGE_W = int(sys.argv[5])

        transform_to_ts(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
    else:
        print('Please run this way:\n')
        print('  python yolov4_pth2ts.py <weight_file> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>')
