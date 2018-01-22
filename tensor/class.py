import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def add_layer(input,insize,outsize,active_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weight = tf.Variable(tf.random_normal([insize,outsize]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,outsize])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input,Weight)+biases
        if active_function == None:
            return Wx_plus_b
        else:
            return active_function(Wx_plus_b)
#init data add noise
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
#create l1,prediction
prediction = add_layer(xs,784,10,active_function = tf.nn.softmax)
#imp loss and train
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#run session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#train and print
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

