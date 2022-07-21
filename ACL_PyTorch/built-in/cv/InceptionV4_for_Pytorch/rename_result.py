import os
from tqdm import tqdm
import sys


def rename_result_txt(result_path):
    txts = os.listdir(result_path)
    for txt in tqdm(txts):
        if not txt.endswith('.txt'):
            continue
        old = txt
        txt = txt.partition('_output')[0]
        index = txt.partition('put')[2]
        #print("txt:",txt)
        #print("index:",index)
        index = int(index) + 1
        new = "ILSVRC2012_val_" + '%08d'%index + "_1.txt"
        os.rename(os.path.join(result_path, old), os.path.join(result_path, new))


if __name__ == "__main__":
    result_path = os.path.abspath(sys.argv[1])
    rename_result_txt(result_path)
