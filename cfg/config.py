# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : configs.py
# Description :config parameters
# --------------------------------------
import os

path_params = {
    'train_data_path': '/home/chenwei/HDD/Project/datasets/object_detection/VOCdevkit/voc_train.txt',
    'pretrain_weights': '/home/chenwei/HDD/Project/CenterNet/weights/resnet34.npy',
    'checkpoints_path': '/home/chenwei/HDD/Project/CenterNet/checkpoints',
    'logs_path': '/home/chenwei/HDD/Project/CenterNet/logs'
}

model_params = {
    'num_classes': 20,
    'input_height': 448,
    'input_width': 448,
    'downsample': 4,
}

solver_params = {
    'batch_size': 8,
    'epochs': 10000,
    'lr_type': "exponential",
    'lr': 1e-3,             # exponential
    'decay_steps': 5000,    # exponential
    'decay_rate': 0.95,     # exponential
    'warm_up_epochs': 2,    # CosineAnnealing
    'init_lr': 1e-4,        # CosineAnnealing
    'end_lr': 1e-6,         # CosineAnnealing
    'pre_train': True
}

test_params = {
    'score_threshold': 0.3,
    'nms_threshold': 0.4
}

classes_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']