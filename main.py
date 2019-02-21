import models
import _pickle as cPickle
import numpy as np
import tensorflow as tf
import os
import pandas as pd

batch_size = 128
image_w = 32
image_h = 32
image_channel = 3
num_classes = 10
epoch = 1
train_samples = 5000
test_samples = 1000

ddir = 'data/'
o_f_ext = '.csv'
m_name = 'myModel'
m_dir = './model/checkpoint/'
m_file = m_name + '.ckpt'
m_pred_fl = ddir + 'pred_' + m_name + o_f_ext

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def one_hot_vec(label):
    vec = np.zeros(num_classes)
    vec[label] = 1
    return vec

def load_data():
    x_all = []
    y_all = []
    for i in range (5):
        print("loading cifar-10-batches-py/data_batch_" + str(i+1))
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))
        x_ = d['data']
        y_ = d['labels']
        x_all.append(x_)
        y_all.append(y_)

    d = unpickle('cifar-10-batches-py/test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:train_samples],axis=0)
    x -= pixel_mean

    y = list(map(one_hot_vec, y))
    X_train = x[0:train_samples,:,:,:]
    Y_train = y[0:train_samples]
    X_test = x[train_samples:,:,:,:]
    Y_test = y[train_samples:]

    return (X_train, Y_train, X_test, Y_test)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('batch_size', 25, 'Batch size')

X_train, Y_train, X_test, Y_test = load_data()

X = tf.placeholder("float", [None, image_w, image_h, image_channel])
Y = tf.placeholder("float", [None, num_classes])
learning_rate = tf.placeholder("float", [])

def reshapebatch(_batch_size):
  #return tf.reshape( X, tf.stack([_batch_size,image_w,image_h,image_channel])), tf.reshape( Y, tf.stack([_batch_size,num_classes]))
  return tf.reshape( X, [_batch_size,image_w,image_h,image_channel]), tf.reshape( Y, [_batch_size,num_classes])

# ResNet Models
net = models.resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cross_entropy = -tf.reduce_sum(Y*tf.log(net))
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_op = opt.minimize(cross_entropy)

#sess = tf.Session()
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  
  #correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  #pred = tf.argmax(net,1)
  pred_ind, pred_val = tf.nn.top_k(net,k=1,sorted=True,name='pred')
  #labels = tf.argmax(Y,1)
  
  print("CWD [{}]".format(os.getcwd()))
  if not os.path.exists(m_dir):
    os.makedirs(m_dir)
  save_path = os.path.join( m_dir, m_file)
  
  saver = tf.train.Saver()
  checkpoint = tf.train.latest_checkpoint(m_dir)
  if checkpoint:
    print("Restoring from checkpoint", checkpoint)
    saver.restore(sess, checkpoint)
  else:
    print("Couldn't find checkpoint to restore from. Starting over.")
    for j in range (epoch):
        for i in range (0, train_samples, batch_size):
            feed_dict={
                X: X_train[i:i + batch_size], 
                Y: Y_train[i:i + batch_size],
                learning_rate: 0.001}
            sess.run([train_op], feed_dict=feed_dict)
            if i % 512 == 0:
                print("training on image #%d" % i)
                saver.save(sess, save_path, global_step=i)
  
  #X, Y = reshapebatch(batch_size) 
  #o_fd = open(m_pred_fl,'w')
  y_hat = []
  y_pred = []
  y_pred_prob = []
  col_names = ['pred_label','prob','label']
  pred_df = pd.DataFrame(columns=col_names)
   
  for i in range (0, test_samples, batch_size):
      p_val, p_ind = sess.run([ pred_ind, pred_val],feed_dict={
         X: X_test[i:i+batch_size],
         Y: Y_test[i:i+batch_size]
        })
      #accuracy_summary = tf.summary.scalar("accuracy", accuracy)
      cnt = 0
      yhat = np.argmax(Y_test[i:i+batch_size],1)
      y_hat.extend(yhat)
      #print(type(yhat),yhat.shape,yhat)
      for b_ind,b_val in zip(p_ind,p_val):
        #print(len(b_ind),len(labels),labels,type(labels))
        for ind,val in zip(b_ind,b_val):
          #o_fd.write("%s|%s|%s\n" % (ind,val,label))
          #o_fd.write("%s|%s\n" % (ind,val))
          y_pred.append(ind)
          y_pred_prob.append(val)
          cnt += 1
  
  pred_df[col_names[0]] = y_pred
  pred_df[col_names[1]] = y_pred_prob
  pred_df[col_names[2]] = y_hat
  pred_df['prediction'] = pred_df['pred_label'] == pred_df['label']
  pred_df.to_csv(m_pred_fl,index=False)
  print("------------------------------------------------------------------------")
  print(" Prediction Accuracy for #[%d] - [%.3f%%]" % (pred_df.label.count(),pred_df['prediction'].mean()*100))
  print("------------------------------------------------------------------------")
  labels = pred_df.label.unique()
  for label in labels:
    print(" label[%s] #[%7d] Accuracy - [%3.2f%%]" % (label,pred_df[pred_df.label == label].label.count(),pred_df[pred_df.label == label].prediction.mean()*100))
  
  #o_fd.close() 
  sess.close()
