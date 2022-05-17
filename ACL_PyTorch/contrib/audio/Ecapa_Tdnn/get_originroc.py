from collections import defaultdict
from time import *
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ECAPA-TDNN.main import ECAPA_TDNN, load_checkpoint, inference_embeddings_to_plt_hist_and_roc
from preprocess import get_dataloader
import sys

if __name__ == "__main__":
    device = torch.device("cuda")
    model = ECAPA_TDNN(1211, device).to(device)
    checkpoint = sys.argv[1]
    # model.load_state_dict(torch.load('runs/Mar04_12-48-27_f2506d594d1f/checkpoint.pt'))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-5)

    model, optimizer, step = load_checkpoint(model, optimizer, checkpoint , rank='cpu')
    model.eval()
    acc_list = list()
    dataset_test, test_speakers = get_dataloader('vox1', 19, sys.argv[2])
    embedding_holder = defaultdict(list)
    for mels, mel_length, speakers in tqdm(dataset_test):
        begin = time()
        h_tensor, info_tensors = model(mels.to(device), infer=True)
        end = time()
        print("time: ",end-begin)
#        print(h_tensor.dtype)
        print(h_tensor.shape)
#        print(speakers)
#        break
        for h, s in zip(h_tensor.detach().cpu(), speakers):
            embedding_holder[s.item()].append(h.numpy())
    
    infer_hist, infer_roc, scores = inference_embeddings_to_plt_hist_and_roc(embedding_holder, step)
    tp, tn, roc_auc = scores
#88600
#    print(step)
#    print(h_tensor)
#    print(tp)
#    print(tn)
    print('origin roc_auc:')
    print(roc_auc)
    #     pred_tensor, info_tensors = model(mels.to(device), speakers.to(device))
    #     prediction = torch.argmax(pred_tensor, axis=-1)
    #     print(prediction)
    #     print(speakers.to(device))
    #     print(torch.sum((prediction == speakers.to(device)), dtype=torch.float32))
    #     acc = (torch.sum((prediction == speakers.to(device)), dtype=torch.float32) / len(
    #         speakers)).detach().cpu().numpy()
    #     acc_list.append(acc)
    # acc_mean = np.mean(acc_list)
    # print(acc_mean)
