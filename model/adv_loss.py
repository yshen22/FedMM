import tensorflow as tf
import model.backbone as backbone
from utils import *

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

class DANNNet(object):
    def __init__(self, input, is_train, base_net='toybase', use_bottleneck=True, class_num=31):
        net = getattr(backbone, base_net)
        self.endpoint = {}
#        input_normalized = tf.map_fn(lambda x:  tf.image.per_image_standardization(x), tf.cast(input, tf.float32))
#        input_normalized = (tf.cast(input, tf.float32) -78.)/ 255.
        input1 = tf.cast(input, tf.float32)
        if base_net == 'toybase':
            input_normalized = (input1 - tf.reduce_mean(input1)) / 255.
        else :
            input_normalized = input1
        with tf.variable_scope('feature_extractor'):
            self.endpoint['feature'] = net(input_normalized, is_train)
        if base_net == 'toybase':
            self.task_classifier = task_classifier_light
            self.domain_classifier = domain_classifier_light
        else:
            self.bottleneck = bottleneck
            self.task_classifier = task_classifier_medium
            self.domain_classifier = domain_classifier_medium
        if use_bottleneck :
            with tf.variable_scope('bottleneck'):
                self.endpoint['feature'] = self.bottleneck(self.endpoint['feature'], is_train)
        with tf.variable_scope('label_predictor'):
            self.endpoint['logits'] = self.task_classifier(self.endpoint['feature'], class_num= class_num, is_train= is_train)
        with tf.variable_scope('domain_predictor'):
            self.endpoint['d_logits'] =self.domain_classifier(self.endpoint['feature'], is_train)


    def get_endpoint(self):
        return self.endpoint


        # features = self.base_network(inputs)

        # features_adv = self.grl_layer(features)
        # outputs_adv = self.classifier_layer_2(features_adv)
        
        # outputs = self.classifier_layer(features)
        # softmax_outputs = self.softmax(outputs)

#        return features, outputs, softmax_outputs, outputs_adv

class MDDNet(object):
    def __init__(self, input, is_train, base_net='toybase', use_bottleneck=True, class_num=31):
        net = getattr(backbone, base_net)
        self.endpoint = {}
#        input_normalized = tf.map_fn(lambda x:  tf.image.per_image_standardization(x), tf.cast(input, tf.float32))
#        input_normalized = (tf.cast(input, tf.float32) -78.)/ 255.
        input1 = tf.cast(input, tf.float32)
        if base_net == 'toybase':
            input_normalized = (input1 - tf.reduce_mean(input1)) / 255.
        else :
            input_normalized = input1
        with tf.variable_scope('feature_extractor'):
            self.endpoint['feature'] = net(input_normalized, is_train)
        if base_net == 'toybase':
            self.task_classifier = mdd_task_classifier_light
#            self.domain_classifier = task_classifier_light
        else:
            self.bottleneck = bottleneck
            self.task_classifier = task_classifier_medium
#            self.domain_classifier = task_classifier_medium
        if use_bottleneck :
            with tf.variable_scope('bottleneck'):
                self.endpoint['feature'] = self.bottleneck(self.endpoint['feature'], is_train)
        with tf.variable_scope('label_predictor'):
            self.endpoint['logits'] = self.task_classifier(self.endpoint['feature'], class_num= class_num, is_train = is_train)
        with tf.variable_scope('domain_predictor'):
            self.endpoint['d_logits'] =self.task_classifier(self.endpoint['feature'], class_num= class_num, is_train = is_train)


    def get_endpoint(self):
        return self.endpoint


class CDANNet(object):
    def __init__(self, input, is_train, base_net='toybase', use_bottleneck=True, class_num=31):
        net = getattr(backbone, base_net)
        self.endpoint = {}
#        input_normalized = tf.map_fn(lambda x:  tf.image.per_image_standardization(x), tf.cast(input, tf.float32))
#        input_normalized = (tf.cast(input, tf.float32) -78.)/ 255.
#        R_g = tf.random.normal([7 * 7 * 48, 4096])
#        R_f = tf.random.normal([class_num, 4096])
        input1 = tf.cast(input, tf.float32)
        if base_net == 'toybase':
            input_normalized = (input1 - tf.reduce_mean(input1)) / 255.
        else :
            input_normalized = input1
        with tf.variable_scope('feature_extractor'):
            self.endpoint['feature'] = net(input_normalized, is_train)
        if base_net == 'toybase':
            self.task_classifier = task_classifier_light
            self.domain_classifier = domain_classifier_light
        else:
            self.bottleneck = bottleneck
            self.task_classifier = task_classifier_medium
            self.domain_classifier = domain_classifier_medium
        if use_bottleneck :
            with tf.variable_scope('bottleneck'):
                self.endpoint['feature'] = self.bottleneck(self.endpoint['feature'], is_train)
        with tf.variable_scope('label_predictor'):
            self.endpoint['logits'] = self.task_classifier(self.endpoint['feature'], class_num= class_num, is_train = is_train)
#        classification_logits = tf.stop_gradient(tf.nn.softmax(self.endpoint['logits'], axis=1))
#        d_feature = tf.matmul(classification_logits, R_f) * tf.matmul(self.endpoint['feature'], R_g) / np.sqrt(4096)
        d_feature = tf.matmul(tf.expand_dims(tf.nn.softmax(self.endpoint['logits'], axis=1), axis=-1), tf.expand_dims(self.endpoint['feature'], axis=1))
        feature_dims =  self.endpoint['feature'].get_shape().as_list()[-1]
        with tf.variable_scope('domain_predictor'):
           self.endpoint['d_logits'] =self.domain_classifier(tf.reshape(d_feature, [-1, class_num * feature_dims] ), is_train= is_train)
#            self.endpoint['d_logits'] =self.get_domain_classifier(d_feature)


    def get_endpoint(self):
        return self.endpoint




class DANN(object):
    def __init__(self, input_tensor, input_placeholder, optimizer, train_mod, base_net='ResNet50', class_num=31, is_train=None):
        self.inputs = input_placeholder
        self.labels = tf.placeholder(tf.float32, [None, class_num])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.is_train =  is_train
        # self.step_ratio = tf.placeholder(tf.float32, [])
        # self.weight_decay = weight_decay
        # self.use_l2 = use_l2
        if base_net == 'toybase':
            self.c_net = DANNNet(input_tensor, self.is_train, base_net, False, class_num)
        else :
            self.c_net = DANNNet(input_tensor, self.is_train, base_net, True, class_num)
        endpoint = self.c_net.get_endpoint()
        self.class_num = class_num
        logits = endpoint['logits']
        d_logits = endpoint['d_logits']
#        coef = 2.0 * 0.15  / (1.0 + tf.exp(-1.0 * self.step_ratio)) - 0.15
        self.pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
#        self.supervised_training = finetune1(self.pretrain_loss, optimizer)
        if base_net == 'toybase':
            coef = 1.
        else :
            coef = 0.25
#        print('coef= ' + str(coef))
        self.total_loss = self.get_loss(logits, d_logits, self.domain, self.labels, coef)
        train_tensor_fn = globals()[train_mod]
        self.loss_training = train_tensor_fn(self.total_loss, optimizer, coef)
        self.label_acc = self.get_acc(logits, self.labels)
        


    def get_loss(self, logits, d_logits, domain_label, labels, coef):

        domain_label1 = tf.argmax(domain_label, axis=-1)
        source_logits = tf.gather(logits, tf.squeeze(tf.where(tf.equal(domain_label1, tf.zeros_like(domain_label1))), axis=-1) ,axis=0) 
        classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=source_logits, labels=labels))
        transfer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=domain_label))
        total_loss = classifier_loss - transfer_loss * coef 
        # if self.use_l2 :
        #     l2_costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY)]
        #     l2_loss = tf.multiply(self.weight_decay, tf.add_n(l2_costs))
        #     total_loss += l2_loss




        return total_loss

    def get_acc(self, outputs, labels):
        correct_label_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        return label_acc

    def set_train(self, mode):
#        self.c_net.train(mode)
        self.is_train.load(mode)

class CDAN(object):
    def __init__(self, input_tensor, input_placeholder, optimizer, train_mod, base_net='ResNet50',class_num=31, is_train=None):
        self.inputs = input_placeholder
        self.labels = tf.placeholder(tf.float32, [None, class_num])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.is_train =  is_train
        if base_net == 'toybase':
            self.c_net = CDANNet(input_tensor, self.is_train, base_net, False, class_num)
        else :
            self.c_net = CDANNet(input_tensor, self.is_train, base_net, True, class_num)

        endpoint = self.c_net.get_endpoint()
        self.class_num = class_num
        logits = endpoint['logits']
        d_logits = endpoint['d_logits']
        if base_net == 'toybase':
            coef = 1.
        else :
            coef = 0.25
        self.pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        self.total_loss = self.get_loss(logits, d_logits, self.domain, self.labels, coef)
        train_tensor_fn = globals()[train_mod]
        self.loss_training = train_tensor_fn(self.total_loss, optimizer, coef)
        self.label_acc = self.get_acc(logits, self.labels)
        


    def get_loss(self, logits, d_logits, domain_label, labels, coef):

        domain_label1 = tf.argmax(domain_label, axis=-1)
#        print(domain_label1.get_shape().as_list())
#        print((domain_label1  tf.zeros_like(domain_label1)).get_shape().as_list())
        source_logits = tf.gather(logits, tf.squeeze(tf.where(tf.equal(domain_label1, tf.zeros_like(domain_label1))), axis=-1) ,axis=0) 
        classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=source_logits, labels=labels))
        transfer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=domain_label))
        total_loss = classifier_loss - transfer_loss * coef
        # if self.use_l2 :
        #     l2_costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY)]
        #     l2_loss = tf.multiply(self.weight_decay, tf.add_n(l2_costs))
        #     total_loss += l2_loss




        return total_loss

    def get_acc(self, outputs, labels):
        correct_label_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        return label_acc



    def set_train(self, mode):
#        self.c_net.train(mode)
        self.is_train.load(mode)


class MDD(object):
    def __init__(self, input_tensor, input_placeholder, optimizer, train_mod, base_net='ResNet50',  class_num=31, is_train=None):
        self.inputs = input_placeholder
        self.labels = tf.placeholder(tf.float32, [None, class_num])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.is_train = is_train
        if base_net == 'toybase':
            self.c_net = MDDNet(input_tensor, self.is_train, base_net, False, class_num)
        else :
            self.c_net = MDDNet(input_tensor, self.is_train, base_net, True, class_num)
        endpoint = self.c_net.get_endpoint()
        self.class_num = class_num
        logits = endpoint['logits']
        d_logits = endpoint['d_logits']
        self.pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        if base_net == 'toybase':
            coef = 0.5
        else :
            coef = 0.1
        train_tensor_fn = globals()[train_mod]
        self.total_loss = self.get_loss(logits, d_logits, self.domain, self.labels, coef)
        self.loss_training = train_tensor_fn(self.total_loss, optimizer, coef)
        self.label_acc = self.get_acc(logits, self.labels)
        


    def get_loss(self, logits, d_logits, domain_label, labels, coef):

        domain_label1 = tf.argmax(domain_label, axis=-1)
#        print(domain_label1.get_shape().as_list())
#        print((domain_label1  tf.zeros_like(domain_label1)).get_shape().as_list())
        source_logits = tf.gather(logits, tf.squeeze(tf.where(tf.equal(domain_label1, tf.zeros_like(domain_label1))), axis=-1) ,axis=0) 
#        print(source_logits.get_shape().as_list())
        classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=source_logits, labels=labels))
#        classifier_label = tf.one_hot(tf.stop_gradient(tf.argmax(logits, 1)), self.class_num)
        classifier_label = tf.stop_gradient(tf.cast(tf.argmax(logits, 1), tf.int32))
        cat_idx = tf.stack([tf.range(0, tf.shape(classifier_label)[0]), classifier_label], axis=1)
        source_transfer_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= d_logits, labels= classifier_label)
        target_transfer_logits = - tf.gather_nd(tf.log(1.00001 - tf.nn.softmax(logits = d_logits, axis=1)), cat_idx)

        transfer_loss = tf.reduce_mean(tf.where(tf.equal(domain_label1, tf.zeros_like(domain_label1)), source_transfer_logits, target_transfer_logits))
        total_loss = classifier_loss - coef * transfer_loss
        # if self.use_l2 :
        #     l2_costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY)]
        #     l2_loss = tf.multiply(self.weight_decay, tf.add_n(l2_costs))
        #     total_loss += l2_loss 


        return total_loss

    def get_acc(self, outputs, labels):
        correct_label_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        return label_acc

    def set_train(self, mode):
#        self.c_net.train(mode)
        self.is_train.load(mode)

def finetune(total_loss, optimizer, coef):
    grads_and_vars1 = optimizer.compute_gradients(total_loss)
    grads_and_vars1_1 = []
    for grad, var in grads_and_vars1:
        if var.name.split('/')[0] == 'domain_predictor':
            grads_and_vars1_1.append( (- grad / (coef + 0.01), var))
        elif var.name.split('/')[0] == 'feature_extractor':
            grads_and_vars1_1.append((0.05*grad, var))
        else :
            grads_and_vars1_1.append((grad, var))
    all_loss_training = optimizer.apply_gradients(grads_and_vars1_1)
    return all_loss_training

def stratch_train(total_loss, optimizer, coef):
    grads_and_vars1 = optimizer.compute_gradients(total_loss)
    grads_and_vars1_1 = []
    for grad, var in grads_and_vars1:
        if var.name.split('/')[0] == 'domain_predictor':
            grads_and_vars1_1.append( (- grad / coef, var))
        else :
            # print(var.name)
            # print(grad is None)
            grads_and_vars1_1.append((grad, var))
    all_loss_training = optimizer.apply_gradients(grads_and_vars1_1)
    return all_loss_training

# def finetune1(total_loss, optimizer):
#     grads_and_vars1 = optimizer.compute_gradients(total_loss)
#     grads_and_vars1_1 = []
#     for grad, var in grads_and_vars1:
#         if var.name.split('/')[0] == 'feature_extractor':
#             grads_and_vars1_1.append((0.1*grad, var))
#         else :
#             grads_and_vars1_1.append((grad, var))
#     all_loss_training = optimizer.apply_gradients(grads_and_vars1_1)
#     return all_loss_training

def _bn(input_feat, is_train, name="bn"):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay

        batch_mean, batch_var = tf.nn.moments(input_feat, [0,1])
#        print(batch_mean.get_shape().as_list())
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable= True)
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable= True)
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                           lambda: (mu, sigma))
#        mean, var = mu, sigma
        bn = tf.nn.batch_normalization(input_feat, mean, var, beta, gamma, 1e-5)
    return bn
# def freeze_train(total_loss, optimizer):
#     grads_and_vars1 = optimizer.compute_gradients(total_loss, trainable_var)
#     grads_and_vars1_1 = []
#     for grad, var in grads_and_vars1:
#         if var.name.split('/')[0] == 'domain_predictor':
#             grads_and_vars1_1.append( (- grad, var))
#         elif var.name.split('/')[0] == 'feature_extractor':
#             pass
#         else :
#             grads_and_vars1_1.append((grad, var))
#     all_loss_training = optimizer.apply_gradients(grads_and_vars1_1)
#     return all_loss_training
def bottleneck(feature, is_train):
    print('use_bottleneck')
    feature_dims = feature.get_shape().as_list()[-1]
    W_fc0 = weight_variable([feature_dims, 1024])
    b_fc0 = bias_variable([1024])
    h_fc0 = tf.matmul(feature, W_fc0) + b_fc0
    x = tf.contrib.layers.batch_norm(inputs=h_fc0, decay=0.9,
                                      updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                      scale=True, epsilon=1e-5, is_training=is_train,
                                      trainable=True)
#        x = _bn(h_fc0, is_train)
    x = tf.nn.relu(x)
#        x = tf.nn.dropout(x, 0.8)
    x = tf.cond(is_train, lambda : tf.nn.dropout(x, 0.5), lambda: x*0.5)
# The domain-invariant feature
#        x = tf.reshape(h_pool1, [-1, width*heights*channels])
    return x



def domain_classifier_medium(feature, is_train):
    feat = feature
    feature_dims = feat.get_shape().as_list()[-1]
        
    d_W_fc0 = weight_variable([feature_dims, 1024])
    d_b_fc0 = bias_variable([1024])
    d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)
#        d_h_fc0 = tf.nn.dropout(d_h_fc0, 0.8)
    d_h_fc0 = tf.cond(is_train, lambda : tf.nn.dropout(d_h_fc0, 0.5), lambda: d_h_fc0 *0.5)
    
    d_W_fc1 = weight_variable([1024, 2])
    d_b_fc1 = bias_variable([2])
    d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1
    return d_logits

def domain_classifier_light(feature, is_train):
    feat = feature
    feature_dims = feat.get_shape().as_list()[-1]
    d_W_fc0 = weight_variable([feature_dims, 100])
    d_b_fc0 = bias_variable([100])
    d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)
    d_W_fc1 = weight_variable([100, 2])
    d_b_fc1 = bias_variable([2])
    d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1
    return d_logits

def task_classifier_light(feature, class_num, is_train):
    classify_feats = feature
    feature_dims = classify_feats.get_shape().as_list()[-1]
    W_fc0 = weight_variable([feature_dims, 100])
    b_fc0 = bias_variable([100])
    h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)
    W_fc1 = weight_variable([100, 100])
    b_fc1 = bias_variable([100])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    W_fc2 = weight_variable([100, class_num])
    b_fc2 = bias_variable([class_num])
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    return logits

def mdd_task_classifier_light(feature, class_num, is_train):
    classify_feats = feature
    feature_dims = classify_feats.get_shape().as_list()[-1]
    W_fc0 = weight_variable([feature_dims, 200])
    b_fc0 = bias_variable([200])
    h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)
    h_fc0 = tf.cond(is_train, lambda : tf.nn.dropout(h_fc0, 0.5), lambda: h_fc0*0.5)
    W_fc1 = weight_variable([200, 200])
    b_fc1 = bias_variable([200])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
#    h_fc1 = h_fc0
    h_fc1 = tf.cond(is_train, lambda : tf.nn.dropout(h_fc1, 0.5), lambda: h_fc1*0.5)
    W_fc2 = weight_variable([200, class_num])
    b_fc2 = bias_variable([class_num])
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2

    return logits

def task_classifier_medium(feature, class_num, is_train):
    classify_feats = feature
    feature_dims = classify_feats.get_shape().as_list()[-1]
        

        
    W_fc0 = weight_variable([feature_dims, 1024])
    b_fc0 = bias_variable([1024])
    h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)
#        h_fc0 = tf.nn.dropout(h_fc0, 0.8)
    h_fc0 = tf.cond(is_train, lambda : tf.nn.dropout(h_fc0, 0.5), lambda: h_fc0*0.5)
    h_fc1 = h_fc0

    # W_fc1 = weight_variable([1024, 100])
    # b_fc1 = bias_variable([100])
    # h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    # x = tf.cond(is_train, lambda : tf.nn.dropout(x, 0.5), lambda: x)

    W_fc2 = weight_variable([1024, class_num])
    b_fc2 = bias_variable([class_num])
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    return logits

