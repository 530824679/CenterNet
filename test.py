import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import tensorflow as tf
from model.network import *
from cfg.config import *
from utils.process import *

def predict_image():
    image_path = "/home/chenwei/HDD/Project/datasets/object_detection/VOC2028/JPEGImages/000006.jpg"
    image = cv2.imread(image_path)
    image_size = image.shape[:2]
    input_shape = [model_params['input_height'], model_params['input_width']]
    image_data = pre_process(image, input_shape)
    image_data = image_data[np.newaxis, ...]

    input = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)

    model = CenterNet(is_train=False)
    hm, wh, reg = model.build_model(input)
    det = model.decode(hm, wh, reg, 100)

    checkpoints = "./checkpoints/model.ckpt-99"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)
        detections = sess.run(det, feed_dict={input: image_data})

    detections = post_process(detections, image_size, input_shape, model_params['downsample'], test_params['score_threshold'])
    bboxes = detections[:, 0:4]
    scores = detections[:, 4]
    class_id = detections[:, 5]
    visualization(image, bboxes, scores, class_id, classes_name)

if __name__ == "__main__":
    predict_image()