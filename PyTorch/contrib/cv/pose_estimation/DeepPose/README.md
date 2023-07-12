# DeepPose

This implements training of DeepPose on the COCO2017 dataset, mainly modified from [mmpose](https://github.com/open-mmlab/mmpose).

## DeepPose Detail 

For details, see mmpose/models/backbones/resnet.py. 

## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org)) and apex

- `pip install -r requirements.txt`

  **Caution**: If your cpu is based on **ARM**，you need to download the source code of package **xtcocoapi**. You can download the source code from [GoogleDrive](https://drive.google.com/file/d/1xdMmB9t4NVYsApQ-Mi2OexabUMMVvkOV/view?usp=sharing) and do like this:

  ```
  unzip xtcocoapi.zip
  cd xtcocoapi
  python3 setup.py build_ext install
  ```

- [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://onedrive.live.com/?cid=56b9f9c97f261712&id=56B9F9C97F261712%2110160&ithint=folder,&authkey=!ANejPkF4WXyxYz4) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing). Download and extract them , and place **COCO_val2017_detections_AP_H_56_person.json** under **$DeepPose/person_detection_results**.


  

## Training 

```bash
# default work directory is work_dirs/npu_deeppose_res50_coco_256x192/
# O2 training , defalut device 0
bash test/train_full_1p.sh --data_path=coco2017_data_path

# O2 training 8p
bash test/train_full_8p.sh --data_path=coco2017_data_path

# eval 8p
# default ckpt path is work_dirs/npu_deeppose_res50_coco_256x192/epoch_210.pth
# you need to choose the correct path of checkpoint
bash bash test/train_full_8p.sh --data_path=coco2017_data_path --checkpoint=ckpt_path

# online inference demo
python3 demo.py configs/top_down/deeppose/coco/npu_deeppose_res50_coco_256x192.py work_dirs/npu_deeppose_res50_coco_256x192/epoch_210.pth

# onnx
python3 pthtar2onnx.py
```

## DeepPose training result 

|  名称  | 精度  | 性能    | AMP_Type |
| :----: | ----- | ------- | -------- |
| GPU-1p | -     | 194     | O2       |
| GPU-8p | 52.50 | 1160    | O2       |
| NPU-1p | -     | 117     | O2       |
| NPU-8p | 52.65 | 650-830 | O2       |

# Statement
For details about the public address of the code in this repository, you can get from the file public_address_statement.md
