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
    'train_data_path': '',
    'test_data_path': '',
    'pretrain_weights': '',
}

model_params = {
    'num_classes': 2,
    'input_height': 448,
    'input_width': 448,
    'downsample': 4,
    'max_object': 150
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

classes_map = {'person': 0, 'hat': 1}