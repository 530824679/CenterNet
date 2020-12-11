# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :config parameters
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
import math
import numpy as np
import tensorflow as tf
from data.augmentation import *
from cfg.config import *
from utils.process import *

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
    hm = np.zeros((output_h, output_w, model_params['num_classes']), dtype=np.float32)
    wh = np.zeros((150, 2), dtype=np.float32)
    reg = np.zeros((150, 2), dtype=np.float32)
    ind = np.zeros((150), dtype=np.float32)
    reg_mask = np.zeros((150), dtype=np.float32)

    for idx, label in enumerate(labels):
        bbox = label[:4] / model_params['downsample']
        class_id = label[4]
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[:, :, class_id], ct_int, radius)
        wh[idx] = 1. * w, 1. * h
        ind[idx] = ct_int[1] * output_w + ct_int[0]
        reg[idx] = ct - ct_int
        reg_mask[idx] = 1

    return image_norm, hm, wh, reg, reg_mask, ind

def get_data(batch_lines):
    batch_image = np.zeros((solver_params['batch_size'], model_params['input_height'], model_params['input_width'], 3), dtype=np.float32)
    batch_hm = np.zeros((solver_params['batch_size'], model_params['input_height'] // model_params['downsample'], model_params['input_width'] // model_params['downsample'], model_params['num_classes']), dtype=np.float32)
    batch_wh = np.zeros((solver_params['batch_size'], 150, 2), dtype=np.float32)
    batch_reg = np.zeros((solver_params['batch_size'], 150, 2), dtype=np.float32)
    batch_reg_mask = np.zeros((solver_params['batch_size'], 150), dtype=np.float32)
    batch_ind = np.zeros((solver_params['batch_size'], 150), dtype=np.float32)

    for num, line in enumerate(batch_lines):
        image, hm, wh, reg, reg_mask, ind = process_data(line)
        batch_image[num, :, :, :] = image
        batch_hm[num, :, :, :] = hm
        batch_wh[num, :, :] = wh
        batch_reg[num, :, :] = reg
        batch_reg_mask[num, :] = reg_mask
        batch_ind[num, :] = ind

    return batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind

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
    dataset = dataset.map(lambda x: tf.py_func(get_data, inp=[x], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(batch_size)

    return dataset