# coding=utf-8
import argparse  
from evaluation import MmluEval

def main(test_dir , model_path , device , num_gpus , task):
    if(task=="Mmlu"):
        mmlu_eval = MmluEval(test_dir , model_path , device , num_gpus)
        res,score = mmlu_eval.eval()
        print(score)
    else:
        print("Wrong task")
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_path", type = str)  
    parser.add_argument("--test_dir", type = str)  
    parser.add_argument("--device", type = str,default = "npu")  
    parser.add_argument("--num_gpus", type = int,default = 1)
    parser.add_argument("--task", type = str) 
    args = parser.parse_args()
    main(args.test_dir,args.model_path,args.device,args.num_gpus,args.task)