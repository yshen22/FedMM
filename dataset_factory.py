import tensorflow as tf
import numpy as np
import pickle as pkl
from utils import *
from functools import partial
import os

def load_data_batch(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist_batch
    else :
        return load_realworld_batch


def load_mnist_batch(batch_size, source_ratio, target_ratio, dataset_name_source=None, dataset_name_target=None, trainable= None):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    # Load MNIST-M
#    print(np.argmax(mnist.train.labels[:100], axis=-1))
#    print(np.argmax(mnistm.train.labels[:100], axis=-1))
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    mnistm_valid = mnistm['valid']
    num_train_source_client1 = int(55000*source_ratio)
    num_train_source_client2 = 55000 - num_train_source_client1
    num_train_target_client1 = int(55000*target_ratio)
    num_train_target_client2 = 55000 - num_train_target_client1
    client1_source_batch = int(batch_size *  num_train_source_client1 / (num_train_source_client1 + num_train_target_client1))
    client1_target_batch = batch_size - client1_source_batch
    client2_source_batch = int(batch_size *  num_train_source_client2 / (num_train_source_client2 + num_train_target_client2))
    client2_target_batch = batch_size - client2_source_batch
    train_source_loader_client1 =  batch_generator(
        [mnist_train[:num_train_source_client1], mnist.train.labels[:num_train_source_client1]], client1_source_batch)
    train_source_loader_client2 =  batch_generator(
        [mnist_train[num_train_source_client1:], mnist.train.labels[num_train_source_client1:]], client2_source_batch)
    train_target_loader_client1 = batch_generator(
        [mnistm_train[:num_train_target_client1], mnist.train.labels[:num_train_target_client1]], client1_target_batch)
    train_target_loader_client2 = batch_generator(
        [mnistm_train[num_train_target_client1:], mnist.train.labels[num_train_target_client1:]], client2_target_batch)
    test_target_loader = (mnistm_test, mnist.test.labels)
    input_tensor = tf.placeholder(tf.uint8, [None, 28, 28, 3])

    return train_source_loader_client1, train_source_loader_client2, train_target_loader_client1, train_target_loader_client2, test_target_loader, input_tensor, input_tensor

