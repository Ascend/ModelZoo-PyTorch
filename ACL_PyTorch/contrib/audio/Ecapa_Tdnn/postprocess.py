from collections import defaultdict

import torch

from ECAPA-TDNN.main import inference_embeddings_to_plt_hist_and_roc
import sys
from tqdm import tqdm

if __name__ == "__main__":
    result_path = sys.argv[1]
    speakers_path = sys.argv[2]
    embedding_holder = defaultdict(list)
    for i in tqdm(range(290)):
        index = i+1
        speakers = torch.load(speakers_path+ 'speakers'+str(index)+".pt")
#        print(speakers)
        batch = []
        with open(result_path+ "mels"+str(index)+"_output_0.txt", 'r') as file:
            data_read = file.read()
            data_line = str(data_read).split('\n')
#            print(data_line[0])
            for j in range(16):
                data_list = []
                num = data_line[j].split(' ')
#                print(float(num[191]))
                for k in range(192):
                    data_list.append(float(num[k]))
                batch.append(data_list)
            h_tensor = torch.Tensor(batch)
#        print(h_tensor)
        for h, s in zip(h_tensor.detach().cpu(), speakers):
            embedding_holder[s.item()].append(h.numpy())
    infer_hist, infer_roc, scores = inference_embeddings_to_plt_hist_and_roc(embedding_holder, 88600)
    tp, tn, roc_auc = scores
    print("roc_auc: ")
    print(roc_auc)