# 1. Problem

Single Stage Object Detection.


# 2. Directions

### Steps to download data
```
cd reference/single_stage_detector/
source download_dataset.sh
```

## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to
run our container.

### NVIDIA DGX-1/DGX-2/DGX A100 (single node)

Launch configuration and system-specific hyperparameters for the appropriate
NVIDIA DGX single node submission are in the `config_DGX1.sh`,
`config_DGX2.sh`, or `config_DGXA100.sh` script respectively.

Steps required to launch single node training:

1. Build the container and push to a docker registry:
```
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch
```
2. Launch the training:

```
source config_DGXA100.sh # or config_DGX1.sh or config_DGX2.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-pytorch DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX-1/DGX-2/DGX A100 (single
node)](#nvidia-dgx-1dgx-2dgx-a100-single-node) explain how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:single_stage_detector-pytorch .
source config_DGXA100.sh # or config_DGX1.sh or config_DGX2.sh
CONT=mlperf-nvidia:single_stage_detector-pytorch DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```


## Steps to launch training on multiple nodes

For multi-node training, we use Slurm for scheduling, and the Pyxis plugin to
Slurm to run our container, and correctly configure the environment for Pytorch
distributed execution.

### NVIDIA DGX A100/DGX-2H (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX
A100 2 node submission is in the `config_DGXA100_multi_2x8x56.sh` script.
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
4 node submission is in the `config_DGX2_multi_4x16x24.sh` script.

Steps required to launch multi node training on NVIDIA DGX A100

1. Build the docker container and push to a docker registry
```
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch
```

2. Launch the training
```
source config_DGXA100_multi_2x8x56.sh # or config_DGX2_multi_4x16x24.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ImageNet (from torchvision).

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.23

### Evaluation frequency
The model is evaluated at epochs 40, 50, 55, and then every 5th epoch.

### Evaluation thoroughness
All the images in COCO 2017 val data set.
