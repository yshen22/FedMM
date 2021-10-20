import tensorflow as tf
from utils import *
# from slim.nets.mobilenet import mobilenet_v2
# slim = tf.contrib.slim

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def toybase(input, is_train):
    W_conv0 = weight_variable([5, 5, 3, 32])
    b_conv0 = bias_variable([32])
    h_conv0 = tf.nn.relu(conv2d(input, W_conv0) + b_conv0)
    h_pool0 = max_pool_2x2(h_conv0)
    
    W_conv1 = weight_variable([5, 5, 32, 48])
    b_conv1 = bias_variable([48])
    h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    _, width, heights, channels = h_pool1.get_shape().as_list()
    # print(width)
    # print(heights)
    # print(channels)
    # The domain-invariant feature
    feature = tf.reshape(h_pool1, [-1, width*heights*channels])
    return feature
