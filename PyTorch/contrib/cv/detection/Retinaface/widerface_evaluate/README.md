# WiderFace-Evaluation
Python Evaluation Code for [Wider Face Dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)


## Usage


##### before evaluating ....
you should make sure that there are wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat`
if you donot have them,please download them from https://github.com/biubug6/Pytorch_Retinaface.git.Once you have downloaded,please copy the groud_truth directory from widerface_evaluate/groud_truth to current directory groud_truth 
````
python3 setup.py build_ext --inplace
````

##### evaluating

**GroungTruth:** `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat`

````
python3 evaluation.py -p <your prediction dir> -g <groud truth dir>
````

## Bugs & Problems
please issue

