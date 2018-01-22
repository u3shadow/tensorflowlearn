import tensorflow as tf
import numpy as np
def add_layer(input,insize,outsize,name,active_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weight = tf.Variable(tf.random_normal([insize,outsize]),name='W')
            tf.summary.histogram(name+'/Weight',Weight)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,outsize])+0.1,name='b')
            tf.summary.histogram(name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input,Weight)+biases
        if active_function == None:
            out = Wx_plus_b
        else:
            out = active_function(Wx_plus_b)

        tf.summary.histogram(name+'/out',out)
        return out
#init data add noise
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise
xs = tf.placeholder(tf.float32,[None,1],name='x_in')
ys = tf.placeholder(tf.float32,[None,1],name='y_in')
#create l1,prediction
l1 = add_layer(xs,1,10,'1',active_function = tf.nn.relu)
prediction  = add_layer(l1,10,1,'2',active_function=None)
#imp loss and train
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]),name='loss')
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#run session
init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
#train and print
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        rs = (sess.run(merged,feed_dict={xs:x_data,ys:y_data}))
        print rs
        writer.add_summary(rs,i)


