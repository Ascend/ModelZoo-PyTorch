_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=100))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)

#optimizer
optimizer_config = dict(grad_clip=None)
#learning policy
runner = dict(type='EpochBasedRunner', max_epochs=2)
lr_config = dict(policy='CosineAnnealing', min_lr=0)