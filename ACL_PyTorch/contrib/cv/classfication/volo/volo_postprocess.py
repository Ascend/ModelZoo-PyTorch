import os
import numpy as np
import argparse

def read_txt_data(path):
    line = ""
    with open(path, 'r') as f:
        line = f.read()
    if line != "":
        return np.array([float(s) for s in line.split(" ") if s != "" and s != "\n"])
    return None

def read_label(path, bs):
    with open(path, 'r') as f:
        content = f.read()
    lines = [line for line in content.split('\n')]
    if lines[-1] == "":
        lines = lines[:-1]
    if bs == 16:
        total_label = np.zeros((len(files) * bs))
        base = 0
        for line in lines:
            labels = line.split(' ')[1:-1]
            labels = [int(label) for label in labels]
            for i in range(len(labels)):
                total_label[base * bs + i] = labels[i]
            base = base + 1
        total_label = np.expand_dims(total_label, 1)
        return total_label
    if bs == 1:
        labels = [int(line.split(' ')[-2]) for line in lines]
        labels = np.array(labels)
        labels = np.expand_dims(labels, 1)
        return labels

def get_topK(files, topk, bs):
    if bs == 1:
        matrix = np.zeros((len(files), topk))
    if bs ==16:
        matrix = np.zeros((len(files) * bs, topk))
    for file in files:
        data = read_txt_data(root + file)
        if bs == 1:
            line = np.argsort(data)[-topk:][::-1]
            index = int(file.split('_')[1])
            matrix[index-1, :] = line[:topk]
        if bs == 16:
            base_index = int(file.split('_')[1])
            newdata = data.reshape(bs, 1000)
            for i in range(bs):
                line = np.argsort(newdata[i,:])[-topk:][::-1]
                matrix[base_index * bs + i, :] = line[:topk]
    return matrix.astype(np.int64)

def get_topK_acc(matrix, labels, k):
    matrix_tmp = matrix[:, :k]
    match_array = np.logical_or.reduce(matrix_tmp==labels, axis=1)
    topk_acc = match_array.sum() / match_array.shape[0]
    return topk_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VOLO validation')
    parser.add_argument('--batchsize', type=int, default='1',
                        help='batchsize.')
    parser.add_argument('--result', type=str, default='./',
                        help='output dir of msame')
    parser.add_argument('--label', type=str, default='./volo_val_bs1.txt',
                        help='label txt dir')
    args = parser.parse_args()
    root = args.result
    bs = args.batchsize
    label_dir = args.label
    files = None
    if os.path.exists(root):
        files=os.listdir(root)
    else:
        print('this path not exist')
        exit(0)
    matrix = get_topK(files, 6, bs)
    labels = read_label(label_dir, bs)
    for i in range(1, 6):
        acc = get_topK_acc(matrix, labels, i)
        print("acc@top{}: {:.3f}%".format(i, 100*acc))