import onnxruntime as ort
import torch
import time
import cv2
import numpy as np
import argparse

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def make_inf_dummy_input(bs):
    org_input_ids = torch.ones(bs, 512).long()
    org_token_type_ids = torch.ones(bs, 512).long()
    org_input_mask = torch.ones(bs, 512).long()

    return (org_input_ids, org_token_type_ids, org_input_mask)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=1, type=int, required=True)
    args = parser.parse_args()
    ort_session = ort.InferenceSession('/home/zxy/spanBertWeight_batch1.onnx',providers=['CUDAExecutionProvider'])
    inf_dummy_input = make_inf_dummy_input(args.bs)
    inf_dummy_input = [t.cpu().numpy() for t in inf_dummy_input]
    totaltime = 0
    count = 10
    for i in range(count):
        t1 = time_sync()
        pred_onnx = ort_session.run(None, {'input_ids': inf_dummy_input[0], 'token_type_ids': inf_dummy_input[1],'attention_mask': inf_dummy_input[2]})
        t2 = time_sync()
        totaltime += (t2 - t1)
    onnx_meantime = totaltime / 10.0 * 1000
    fps = 1000 / (onnx_meantime / args.bs)
    print('spanBert inference by onnxruntime mean time is {}ms, bs{}, {}fps'.format(onnx_meantime, args.bs, fps))
