from collections import defaultdict

import torch

from ECAPA_TDNN.main import inference_embeddings_to_plt_hist_and_roc
import sys
from tqdm import tqdm

if __name__ == "__main__":
    result_path = sys.argv[1]
    speakers_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    total_nums = int(sys.argv[4])
    result_nums = int(total_nums/batch_size)
    print(result_nums)
    embedding_holder = defaultdict(list)
    #样本总数为4648，bs为16,drop_last=True，所以循环4648/16且向下取整即290
    for i in tqdm(range(result_nums)):
        index = i+1
        speakers = torch.load(speakers_path+ 'speakers'+str(index)+".pt")

        batch = []
        with open(result_path+ "mels"+str(index)+"_output_0.txt", 'r') as file:
            data_read = file.read()
            data_line = str(data_read).split('\n')


            for j in range(batch_size):
                data_list = []
                num = data_line[j].split(' ')

                for k in range(batch_size):
                    data_list.append(float(num[k]))
                batch.append(data_list)
            h_tensor = torch.Tensor(batch)

        for h, s in zip(h_tensor.detach().cpu(), speakers):
            embedding_holder[s.item()].append(h.numpy())
    infer_hist, infer_roc, scores = inference_embeddings_to_plt_hist_and_roc(embedding_holder, 88600)
    tp, tn, roc_auc = scores
    print("roc_auc: ")
    print(roc_auc)