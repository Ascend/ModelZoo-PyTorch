import sys
from timm.data import create_loader, ImageDataset
import os
import numpy as np
import argparse

os.environ['device'] = 'cpu'

def preprocess_volo(data_dir, save_path, batch_size):
    f = open("volo_val_bs"+str(batch_size)+".txt", "w")

    loader = create_loader(
        ImageDataset(data_dir),
        input_size=(3, 224, 224),
        batch_size=batch_size,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=4,
        crop_pct=0.96,
        pin_memory=False,
        tf_preprocessing=False)

    for batch_idx, (input, target) in enumerate(loader):
        img = np.array(input).astype(np.float32)
        save_name = os.path.join(save_path, "test_" + str(batch_idx) + ".bin")
        print(save_name)
        img.tofile(save_name)
        info = "%s " % ("test_" + str(batch_idx) + ".bin")
        for i in range(batch_size):
            info = info + str(int(target[i])) + " "
        info = info + "\n"
        f.write(info)

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imagenet val_dataset preprocess')
    parser.add_argument('--src', type=str, default='./',
                        help='imagenet val dir.')
    parser.add_argument('--des', type=str, default='./',
                        help='preprocess dataset dir.')
    parser.add_argument('--batchsize', type=int, default='1',
                        help='batchsize.')
    args = parser.parse_args()
    src = args.src
    des = args.des
    bs = args.batchsize
    files = None
    if not os.path.exists(src):
        print('this path not exist')
        exit(0)
    os.makedirs(des, exist_ok=True)
    preprocess_volo(src, des, bs)

    # python volo_224_preprocess.py --src /opt/npu/val --des /opt/npu/data_bs1 --batchsize 1
