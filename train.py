from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from pdgd import PDGradientDescent
from pgd import PerturbedGradientDescent
from model import adv_loss
import argparse
import importlib
import matplotlib
import pandas as pd
matplotlib.use('Agg') 
import dataset_factory
import json
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def restore_checkpoint(checkpoint_filename,sess):
    checkpoint_reader = tf.train.NewCheckpointReader(checkpoint_filename)
    var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
    restored_tensor_list = []
    global_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    global_var_name_list = [v.name for v in global_var_list]
#    print(global_var_name_list)
    tensorname_list = {}
    for key in var_to_shape_map:
        if ('feature_extractor/' + key +':0') in global_var_name_list:
            print (key)
#            print('feature_extractor/' + key +':0')
            tensorname_list[key] = tf.get_default_graph().get_tensor_by_name('feature_extractor/' + key +':0')
#            restored_tensor_list.append(tf.get_default_graph().get_tensor_by_name('feature_extractor/' + key +':0'))
    saver = tf.train.Saver(var_list=tensorname_list)
    saver.restore(sess, checkpoint_filename)
    print('checkpoint loaded')


from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-adv_loss', type=str, default= 'DANN' )
parser.add_argument('-base_net', type=str, default= 'toybase' )
parser.add_argument('-batch_size', type=int, default= 64)
parser.add_argument('-fed_trainer', type=str, default='fedmm')
parser.add_argument('-dataset', type=str, default='mnist')
parser.add_argument('-source_ratio', type=float, default = 0.)
parser.add_argument('-target_ratio', type=float, default = 1.)
parser.add_argument('-lambda1', type = float, default = 1.)
parser.add_argument('-lr', type =float, default = 0.01)
parser.add_argument('-num_local_steps', type = int, default = 20)
parser.add_argument('-lambda1_decay', type=float , default =1.02 )
parser.add_argument('-max_iter', type=int , default =5000 )
parser.add_argument('-dataset_source', type=str, default ='amazon')
parser.add_argument('-dataset_target', type=str, default='dslr')
parser.add_argument('-pretrain', type=str2bool, default= True)
parser.add_argument('-l2_decay', type=float, default= 0.0001)
parser.add_argument('-use_l2', type=str2bool, default= False)
parser.add_argument('-train_mod', type=str, default= 'stratch_train')
parser.add_argument('-decay_step', type =str, default = '200000,400000')
parser.add_argument('-num_class', type=int, default = 10)
parser.add_argument('-num_hidden1', type=int, default = 100)
args=parser.parse_args()
print(args)

# class ImageList(object):
#     """A generic data loader where the images are arranged in this way: ::
#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/xxz.png
#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/asd932_.png
#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """

#     def __init__(self, image_list, labels=None, transform=None, target_transform=None,
#                  loader=default_loader):
#         imgs = make_dataset(image_list, labels)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         path, target = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)


if __name__ == '__main__':
 # if args.dataset == 'Office-31':
 #        class_num = 31
 #        width = 1024
 #        srcweight = 4
 #        is_cen = False
 #    elif args.dataset == 'Office-Home':
 #        class_num = 65
 #        width = 2048
 #        srcweight = 2
 #        is_cen = False

#    pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

#    mnist_train = (mnist_train - pixel_mean) / 255
#    mnistm_train = (mnistm_train - pixel_mean) / 255
#    mnistm_test = (mnistm_test - pixel_mean) /255
    print(vars(args))
    lambdat = tf.placeholder(tf.float32, [])
    learning_rate = tf.placeholder(tf.float32, [])
    is_train = tf.get_variable('is_train', [],
                        tf.bool, trainable=False)
    decay_step = list(map(int, args.decay_step.split(',')))
    domain_adaptation_instance = getattr(adv_loss, args.adv_loss)
    batch_size = args.batch_size
    if args.fed_trainer == 'fedmm':
        optimizer = PDGradientDescent(learning_rate, lambdat)
    if args.fed_trainer == 'fedavg':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if args.fed_trainer == 'fedprox':
        optimizer = PerturbedGradientDescent(learning_rate, lambdat)
    if args.dataset == 'mnist':
        batch_eval = False
    else :
        batch_eval = True
    dataset_batchloader = dataset_factory.load_data_batch(args.dataset)
    train_source_loader_client1, train_source_loader_client2, train_target_loader_client1, train_target_loader_client2, test_target_loader, input_placeholder, input_tensor = \
        dataset_batchloader(batch_size = batch_size, source_ratio =args.source_ratio, target_ratio = args.target_ratio, dataset_name_source=args.dataset_source, 
        dataset_name_target=args.dataset_target, trainable=is_train)
    # train_source_loader_client1, train_source_loader_client2, test_target_loader, input_placeholder, input_tensor = \
    #     dataset_batchloader(batch_size = batch_size, source_ratio =args.source_ratio, target_ratio = args.target_ratio, dataset_name_source=args.dataset_source, 
    #     dataset_name_target=args.dataset_target)
    model_instance = domain_adaptation_instance(input_tensor=input_tensor, input_placeholder= input_placeholder, optimizer=optimizer, train_mod=args.train_mod,
    	base_net=args.base_net, class_num=args.num_class, is_train= is_train)
    lr = args.lr
    model_path = '%s.%s' % ('trainer', args.fed_trainer)
    trainer_mod = importlib.import_module(model_path)
    train_fn = getattr(trainer_mod, 'train')
    init_op = tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer())
    variables_names = [v.name for v in tf.trainable_variables()]
    print(variables_names)
    with tf.Session() as sess:
        sess.run(init_op)
        if args.pretrain :
            if args.base_net == 'mobileNet':
                restore_checkpoint('checkpoint/mobilenet_v2_1.4_224.ckpt', sess)
#        pretrain(model_instance, train_source_loader_client2, train_source_loader_client2, tf.train.GradientDescentOptimizer(0.01), sess)
        train_record= train_fn(model_instance, train_source_loader_client1, train_source_loader_client2, train_target_loader_client1, train_target_loader_client2, test_target_loader, 
        max_iter=args.max_iter, num_local_steps=args.num_local_steps, optimizer=optimizer, lr=lr, decay_epoch = decay_step,
        lr_placeholder= learning_rate ,eval_interval=100, batch_eval = batch_eval, lambda1= 2., lambdat = lambdat, lambda1_decay=args.lambda1_decay, batch_size = batch_size, sess=sess)
    print('finish train')
    train_df = pd.DataFrame.from_records(train_record)
    if not(os.path.exists('acc_result_office31')):
        os.makedir('acc_result_office31')
    now =datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
#    filenames = ''
    
    # for key, value in vars(args).items():
    #     if key== 'fed_trainer' or key == 'source_ratio' or key == 'lambda1' or key =='lr' or key == 'num_local_steps' or key =='lambda1_decay':
    #         filenames += '|' + str(key) + '|' + str(value)
    train_df.to_csv('acc_result/result_' + dt_string +'.csv')
    with open('acc_result/result_' + dt_string +'.txt', 'w') as outfile:
        json.dump(args.__dict__, outfile)




