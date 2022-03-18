import time
import os

from impl.edges_eval_dir import edges_eval_dir
from impl.edges_eval_plot import edges_eval_plot


def eval_edge(alg, model_name_list, result_dir, gt_dir, workers=1):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]

    for model_name in model_name_list:
        tic = time.time()
        res_dir = os.path.join(result_dir, model_name)
        print(res_dir)
        edges_eval_dir(res_dir, gt_dir, thin=1, max_dist=0.0075, workers=workers)
        toc = time.time()
        print("TIME: {}s".format(toc - tic))
        edges_eval_plot(res_dir, alg)


if __name__ == '__main__':
    eval_edge("HED", "hed", "NMS_RESULT_FOLDER", "test", 10)
