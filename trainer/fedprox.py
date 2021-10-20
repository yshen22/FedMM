import tensorflow as tf
import numpy as np
import pandas as pd

def get_params(sess):
    model_params = sess.run(tf.trainable_variables())
    return model_params

def set_params(model_params, sess):
    if model_params is not None: 
      all_vars = tf.trainable_variables()
      for variable, value in zip(all_vars, model_params):
          variable.load(value, sess)

def aggregate(wsolns):
    total_weight = 0.0
    base = [0]*len(wsolns[0][1])
    for (w, soln) in wsolns:  # w is the number of local samples
        total_weight += w
        for i, v in enumerate(soln):
            base[i] += w*v.astype(np.float32)

    averaged_soln = [v / total_weight for v in base]

    return averaged_soln
def train(model_instance, train_source_loader_client1, train_source_loader_client2, train_target_loader_client1, train_target_loader_client2, test_target_loader,
          max_iter, num_local_steps, optimizer, lr, decay_epoch, lr_placeholder, eval_interval, batch_eval, lambda1, lambdat, lambda1_decay, batch_size, sess):
    model_instance.set_train(True)
    model = model_instance
    print("start train...")
    iter_num = 0
    epoch = 0
#    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    label_acc = model_instance.label_acc
    train_op = model_instance.loss_training
    X0, y0 = next(train_target_loader_client1)
    X1, y1 = next(train_source_loader_client1)
    client1_target_batch_size = np.shape(X0)[0]
    client1_source_batch_size = np.shape(X1)[0]
    client1_domain_labels = np.concatenate([np.tile([0., 1.], [client1_target_batch_size, 1]), np.tile([1., 0.], [client1_source_batch_size, 1])], axis =0)
    X2, y2 = next(train_target_loader_client2)
    X3, y3 = next(train_source_loader_client2)
    client2_target_batch_size = np.shape(X2)[0]
    client2_source_batch_size = np.shape(X3)[0]
    client2_domain_labels = np.concatenate([np.tile([0., 1.], [client2_target_batch_size, 1]), np.tile([1., 0.], [client2_source_batch_size, 1])], axis =0)

    client_params = []
    train_record = []
    variable_set = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i in range(2):
        client_model = []
        for var in variable_set:
            client_model.append(np.zeros(shape = var.get_shape().as_list(), dtype = np.float32))
        client_params.append(client_model)


    lr_t = lr
    latest_model = get_params(sess)
    for i in range(max_iter):
#            print('====================Round {0}===================='.format(i))            
        # Adaptation param and learning rate schedule as described in the paper

        lambdat_i = lambda1 * (lambda1_decay ** (i/100))

    


        # Training step
        if i == decay_epoch[0] or i == decay_epoch[1]:
            lr_t  = lr_t * 0.1
        set_params(latest_model, sess)
        optimizer.set_params(latest_model, sess)

#            sess.run(grads_zeros_ops)
        for j in range(num_local_steps):
            X0, y0 = next(train_target_loader_client1)
            X1, y1 = next(train_source_loader_client1)
            X = np.concatenate([X0, X1], axis=0)
#                print(y0)
            _  = sess.run(
            [train_op],
            feed_dict={model.inputs: X, model.labels: y1, model.domain: client1_domain_labels,
                        lr_placeholder: lr_t, lambdat:lambda1 })

        client_params[0] = get_params(sess)


        set_params(latest_model, sess)

        optimizer.set_params(latest_model, sess)
#           sess.run(grads_zeros_ops)
        for j in range(num_local_steps):
            X2, y2 = next(train_target_loader_client2)
            X3, y3 = next(train_source_loader_client2)
            X = np.concatenate([X2, X3], axis=0)
            _= sess.run(
            [train_op],
            feed_dict={model.inputs: X, model.labels: y3, model.domain: client2_domain_labels,
                        lr_placeholder: lr_t, lambdat:lambda1} )
        client_params[1] = get_params(sess)

        latest_model = aggregate([(w , soln) for w, soln in zip(np.ones(2)/2, client_params)])
        if i % eval_interval == 0:
            print('====================Round {0}===================='.format(i))    
#                source_acc = sess.run(label_acc,
#                feed_dict={model.X: mnist_test, model.y: mnist.test.labels,
#                model.train: False})
            model_instance.set_train(False)
            if batch_eval :
                target_acc_all = 0
                for k in range(10):
                    target_test_data, target_test_label = next(test_target_loader)
                    # print(np.shape(target_test_data))
                    # print(np.shape(target_test_label))
                    target_acc = sess.run(label_acc,
                    feed_dict={model.inputs: target_test_data, model.labels: target_test_label,
                    })
                    target_acc_all+= target_acc
                target_acc_all = target_acc_all / 10
            else :
                target_test_data, target_test_label = test_target_loader
                target_acc_all = sess.run(label_acc,
                    feed_dict={model.inputs: target_test_data, model.labels: target_test_label,
                    })
            model_instance.set_train(True)
#                print('Source (MNIST) accuracy:', source_acc)
            print('Target accuracy:', target_acc_all)
            train_record.append({'acc': target_acc_all})

            # val
    return train_record

    # print('finish train')
    # train_df = pd.DataFrame.from_records(train_record)
    # train_df.to_csv('acc_result/fedpd_acc.csv')