import numpy as np
import tensorflow as tf
from model.ops import *
from cfg.config import *

class CenterNet():
    def __init__(self, is_train):
        self.is_train = is_train
        self.num_classes = model_params['num_classes']
        self.inplanes = 64

    def _block(self, inputs, filters, strides=1):
        expansion = 1
        conv1_bn_relu = conv2d(inputs, filters, [3, 3], strides, 'same', activation=tf.nn.relu, is_training=self.is_train, use_bn=True)
        conv2_bn = conv2d(conv1_bn_relu, filters, [3, 3], 1, 'same', activation=None, is_training=self.is_train, use_bn=True)
        if strides != 1 or self.inplanes != filters * expansion:
            inputs = conv2d(inputs, filters, [1, 1], strides, 'valid', activation=None, is_training=self.is_train, use_bn=True)
            self.inplanes = filters * expansion
        out = tf.nn.relu(conv2_bn + inputs)
        return out

    def _module(self, x, num_channels, layers, strides=1):
        for i in range(layers):
            if i == 0:
                x = self._block(x, num_channels, strides=strides)
            else:
                x = self._block(x, num_channels)
        return x

    def _resnet34(self, inputs):
        net = conv2d(inputs, 64, [7, 7], 2, 'same', activation=tf.nn.relu, is_training=self.is_train, use_bn=True)
        net = tf.layers.max_pooling2d(net, pool_size=3, strides=2, padding='same')

        layer1 = self._module(net, 64, 3, 1)
        layer2 = self._module(layer1, 128, 4, 2)
        layer3 = self._module(layer2, 256, 6, 2)
        layer4 = self._module(layer3, 512, 3, 2)

        return layer1, layer2, layer3, layer4

    def build_model(self, inputs):
        c2, c3, c4, c5 = self._resnet34(inputs)

        p5 = conv2d(c5, 128, [1, 1], is_training=self.is_train)

        up_p5 = upsampling(p5, method='resize')
        reduce_dim_c4 = conv2d(c4, 128, [1, 1], is_training=self.is_train)
        p4 = 0.5 * up_p5 + 0.5 * reduce_dim_c4

        up_p4 = upsampling(p4, method='resize')
        reduce_dim_c3 = conv2d(c3, 128, [1, 1], is_training=self.is_train)
        p3 = 0.5 * up_p4 + 0.5 * reduce_dim_c3

        up_p3 = upsampling(p3, method='resize')
        reduce_dim_c2 = conv2d(c2, 128, [1, 1], is_training=self.is_train)
        p2 = 0.5 * up_p3 + 0.5 * reduce_dim_c2

        features = conv2d(p2, 128, [3, 3], is_training=self.is_train)

        with tf.variable_scope('detector'):
            hm = conv2d(features, 64, [3, 3], is_training=self.is_train)
            hm = tf.layers.conv2d(hm, self.num_classes, 1, 1, padding='valid', activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')

            wh = conv2d(features, 64, [3, 3], is_training=self.is_train)
            wh = tf.layers.conv2d(wh, 2, 1, 1, padding='valid', activation=None, name='wh')

            reg = conv2d(features, 64, [3, 3], is_training=self.is_train)
            reg = tf.layers.conv2d(reg, 2, 1, 1, padding='valid', activation=None, name='reg')

        return hm, wh, reg

    def topk(self, hm, K=150):
        batch, height, width, cat = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
        # [b,h*w*c]
        scores = tf.reshape(hm, (batch, -1))
        # [b,k]
        topk_scores, topk_inds = tf.nn.top_k(scores, k=K)
        # [b,k]
        topk_clses = topk_inds % cat
        topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
        topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
        topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode(self, hm, wh, reg, k=150):
        batch, height, width, channel = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]

        hmax = tf.layers.max_pooling2d(hm, 3, 1, padding='same')
        keep = tf.cast(tf.equal(hm, hmax), tf.float32)
        heat = hm * keep

        scores, inds, clses, ys, xs = self.topk(heat, K=k)

        if reg is not None:
            reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
            # [b,k,2]
            reg = tf.batch_gather(reg, inds)
            xs = tf.expand_dims(xs, axis=-1) + reg[..., 0:1]
            ys = tf.expand_dims(ys, axis=-1) + reg[..., 1:2]
        else:
            xs = tf.expand_dims(xs, axis=-1) + 0.5
            ys = tf.expand_dims(ys, axis=-1) + 0.5

            # [b,h*w,2]
        wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
        # [b,k,2]
        wh = tf.batch_gather(wh, inds)

        clses = tf.cast(tf.expand_dims(clses, axis=-1), tf.float32)
        scores = tf.expand_dims(scores, axis=-1)

        xmin = xs - wh[..., 0:1] / 2
        ymin = ys - wh[..., 1:2] / 2
        xmax = xs + wh[..., 0:1] / 2
        ymax = ys + wh[..., 1:2] / 2

        bboxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

        # [b,k,6]
        detections = tf.concat([bboxes, scores, clses], axis=-1)
        return detections

    def calc_loss(self, pred_hm, pred_wh, pred_reg, true_hm, true_wh, true_reg, reg_mask, ind):
        hm_loss = self.focal_loss(pred_hm, true_hm)
        wh_loss = 0.05 * self.reg_l1_loss(pred_wh, true_wh, ind, reg_mask)
        reg_loss = self.reg_l1_loss(pred_reg, true_reg, ind, reg_mask)
        total_loss = hm_loss + wh_loss + reg_loss
        return total_loss, hm_loss, wh_loss, reg_loss

    def focal_loss(self, hm_pred, hm_true):
        pos_mask = tf.cast(tf.equal(hm_true, 1.), dtype=tf.float32)
        neg_mask = tf.cast(tf.less(hm_true, 1.), dtype=tf.float32)
        neg_weights = tf.pow(1. - hm_true, 4)

        pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(1. - hm_pred, 2) * pos_mask
        neg_loss = -tf.log(tf.clip_by_value(1. - hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(hm_pred, 2.0) * neg_weights * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
        return loss

    def reg_l1_loss(self, y_pred, y_true, indices, mask):
        b = tf.shape(y_pred)[0]
        k = tf.shape(indices)[1]
        c = tf.shape(y_pred)[-1]
        y_pred = tf.reshape(y_pred, (b, -1, c))
        indices = tf.cast(indices, tf.int32)
        y_pred = tf.batch_gather(y_pred, indices)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
        total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
        return loss

    def bbox_giou(self, boxes_1, boxes_2):
        """
        calculate regression loss using giou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou
        iou = inter_area / tf.maximum(union_area, 1e-5)

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate area of the minimun closed convex surface
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        # calculate the giou
        giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-5)

        return giou
