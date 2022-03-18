# 3DMPPE_ROOTNET

This implements training of 3DMPPE_ROOTNET on the MuCo, MPII and MuPoTS datasets.
- Reference implementation:
```
url=https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
branch=master 
commit_id=a199d50be5b0a9ba348679ad4d010130535a631d
```

## 3DMPPE_ROOTNET Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, 3DMPPE_ROOTNET is re-implemented using semantics such as custom OP. 

## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- The training datasets are MuCo and MPII, the evaluation dataset is MoPoTS. The datasets are large. Please ensure sufficient hard disk space when downloading and decompressing.
- You need to follow directory structure of the `data` as below. If you connect to the prepared data folder, you don't need to build the following directory.
```
${3DMPPE_ROOTNET}
|-- data
|   |-- MPII
|   |   |-- images
|   |   |   |-- ...   ## image files
|   |   |-- annotations
|   |   |   |-- test.json
|   |   |   |-- train.json
|   |-- MuCo
|   |   |-- data
|   |   |   |-- augmented_set
|   |   |   |   |-- ...   ## image files
|   |   |   |-- unaugmented_set
|   |   |   |   |-- ...   ## image files
|   |   |   |-- MuCo-3DHP.json
|   |-- MuPoTS
|   |   |-- data
|   |   |   |-- MultiPersonTestSet
|   |   |   |   |-- ...   ## image files
|   |   |   |-- MuPoTS-3D.json
```
- You need to follow directory structure of the `output` as below.
```
${3DMPPE_ROOTNET}
|-- output
|   |-- model_dump
|   |-- result
|   |-- log
|   |   |-- train_logs.txt   ## The training log is saved here
|   |   |-- test_logs.txt    ## The evaluation log is saved here
|   |-- vis
|   |-- prof
```

## Training #

- Note that the `output` folder under the `test` directory will also save the code running log.
- To train a model, run `train_1p.py` or `train_8p.py`: 

```bash
# 1p train perf
bash test/train_performance_1p.sh --data_path=xxx

# 8p train perf
bash test/train_performance_8p.sh --data_path=xxx

# 1p train full
bash test/train_full_1p.sh --data_path=xxx

# 8p train full
bash test/train_full_8p.sh --data_path=xxx

```

## 3DMPPE_ROOTNET training result 

| AP_25(percentage)    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 33(average) 35.78(highest）        | 190      | 1        | 20      | O1       |
| -        | 230      | 2        | 3      | O1       |
| -        | 425      | 4        | 3      | O1       |
| 37(average) 41.05(highest）        | 855      | 8        | 20      | O1       |

# Else #

- run `demo.py`：
Enter the demo folder. The input file for running the demo has been provided(`input.jpg`). After running, the output pictures will be obtained in this directory. 
First, place `snapshot_XX.pth` in directory `./output/model_dump/`. Then, Change the parameter `test_epoch` of `run_demo.sh` to `XX` ,which corresponds to the number of `.pth` file just now. Finally, run the command:
```
bash demo/run_demo.sh
```
You can also run the following command:
```
python demo.py --test_epoch XX
```
- Model evaluation only：
Place `snapshot_XX.pth` in directory `./output/model_dump/`. cd to `main` folder and run the command:
```
python test.py --test_epoch XX
```
The log will be saved in`./output/log/test_logs.txt`.