# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :config parameters
# --------------------------------------

import cv2
import os
import math
import numpy as np
import tensorflow as tf
from data.augmentation import *
from cfg.config import *

def process_data(line):
    if 'str' not in str(type(line)):
        line = line.decode()
    data = line.split()
    image_path = data[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = np.array(cv2.imread(image_path))
    labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in data[1:]])

    image, labels = random_horizontal_flip(image, labels)
    image, labels = random_crop(image, labels)
    image, labels = random_translate(image, labels)

    input_height, input_width = model_params['input_height'], model_params['input_width']
    image_rgb = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB).astype(np.float32)
    image_rgb, labels = letterbox_resize(image_rgb, (input_height, input_width), np.copy(labels), interp=0)
    image_norm = image_rgb / 255.

    output_h = model_params['input_height'] // model_params['downsample']
    output_w = model_params['input_width'] // model_params['downsample']
    hm = np.zeros((output_h, output_w, cfg.num_classes), dtype=np.float32)
    wh = np.zeros((cfg.max_objs, 2), dtype=np.float32)
    reg = np.zeros((cfg.max_objs, 2), dtype=np.float32)
    ind = np.zeros((cfg.max_objs), dtype=np.float32)
    reg_mask = np.zeros((cfg.max_objs), dtype=np.float32)

    for idx,


def get_data(batch_lines):
    for num, line in enumerate(batch_lines):
        pass

def create_dataset(filen_path, batch_num, batch_size=1, is_shuffle=False):
    """
    :param filenames: train file path
    :param batch_size: batch size
    :param is_shuffle: whether shuffle
    :param n_repeats: number of repeats
    :return:
    """
    dataset = tf.data.TextLineDataset(filen_path)
    if is_shuffle:
        dataset = dataset.shuffle(batch_num)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(get_data, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(batch_size)

    return dataset