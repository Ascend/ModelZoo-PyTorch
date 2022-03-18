# CenetrNet

This implements training of CenetrNet on the Pascal VOC dataset, mainly modified from [CenterNet](https://github.com/xingyizhou/CenterNet).

## CenetrNet Detail

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point -- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. 



## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py install
    ~~~
-  Install Dependencies
    ~~~
    pip install -r requirements.txt
    ~~~
- Compile deformable convolutional (from DCNv2).

    ~~~
    cd ./src/lib/models/networks/DCNv2
    ./make.sh
    ~~~
- Compile NMS
    ~~~
    cd ./src/lib/external
    make
    ~~~


## Download the Pscal VOC dataset 

  - Run
      ~~~
      cd ./src/tools/
      bash get_pascal_voc.sh
      ~~~
  - The above script includes:
      - Download, unzip, and move Pascal VOC images from the VOC website. 
      - Download Pascal VOC annotation in COCO format (from Detectron). 
      - Combine train/val 2007/2012 annotation files into a single json. 
  - Move the created `voc` folder to `data` (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- voc
      `-- |-- annotations
          |   |-- pascal_trainval0712.json
          |   |-- pascal_test2017.json
          |-- images
          |   |-- 000001.jpg
          |   ......
          `-- VOCdevkit
  
  ~~~
  The `VOCdevkit` folder is needed to run the evaluation script from [faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/reval.py).

## Training

To train a model, run `main_npu_1p/8p.py` with the desired model architecture:

```bash
# training 1p accuracy
bash test/train_full_1p.sh  --data_path=YourDataPath

# training 1p performance
bash test/train_performance_1p.sh   --data_path=YourDataPath

# training 8p accuracy
bash test/train_full_8p.sh  --data_path=YourDataPath

# training 8p performance
bash test/train_performance_8p.sh  --data_path=YourDataPath

# evaluate 8p accuracy
bash test/train_eval.sh  --data_path=YourDataPath
```

## Related Files Path:
- log path:
  ~~~
    exp/task(default:ctdet)/exp_id/date_and_time/log.txt  # training detail and performance result log    
    test/output/devie_id/test_0.log # 8p training accuracy result log
  ~~~
- model path:
  ~~~
    exp/task(default:ctdet)/exp_id/model_last.pth  #The completed training model(Use this model for testing by default)
    exp/task(default:ctdet)/exp_id/model_best.pth  #Training the model with the lowest loss
    exp/task(default:ctdet)/exp_id/model_45.pth  #Model of the 45th epoch, Training at this epoch will drop the learn rate
    exp/task(default:ctdet)/exp_id/model_60.pth  #Model of the 60th epoch, Training at this epoch will drop the learn rate
    exp/task(default:ctdet)/exp_id/model_75.pth  #Model of the 75th epoch, Training at this epoch will drop the learn rate
  ~~~

## Test other models
If you want to test another model (e.g. model_best), enter the command in the terminal
  ~~~
    python src/test.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --resume --flip_test --load_model YourModelPath

  ~~~



## CenterNet training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 140       | 1        | 5        | O1       |
| 71.16   | 1080      | 8        | 90      | O1       |
