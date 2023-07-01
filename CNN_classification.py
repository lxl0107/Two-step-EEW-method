import time
import datetime
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import read_original_data_t


# Dataset
train5_record = open('train_larger_than52.txt')
train5_record_read = train5_record.readlines()
train5_record.close()

train_record = open('train_smaller_than45.txt')
train_record_read = train_record.readlines()
train_record.close()

train_unsure = open('train_unsure45_52.txt')
train_unsure_read = train_unsure.readlines()
train_unsure.close()

test5_record = open('test_larger_than5.txt')
test5_record_read = test5_record.readlines()
test5_record.close()

test_record = open('test_smaller_than5.txt')
test_record_read = test_record.readlines()
test_record.close()



TRAINNUM = 500
batch_size = 64
test_size = 32
unsure_batch_size = 12
unsure_test_size = 6

# Save model
model_info = 'Classification model'
training_time = datetime.datetime.now().strftime("%y-%m-%d+%h-%M-%S")
if not os.path.exists(r'./svntin'):
    os.makedirs(r'./svntin')
strsvin = r'./svntin' + '/' + model_info + training_time   #生成模型位置
if not os.path.exists(strsvin):
    os.makedirs(strsvin)

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 1, 500, 3], name="x-sword")
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32, name='kp')

xs = tf.reshape(x, [-1, 1, 1000, 3])


W_conv1 = weight_variable([1, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)
m_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 1, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(m_pool1, W_conv2) + b_conv2)

W_conv3 = weight_variable([1, 5, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
m_pool3 = max_pool_2x2(h_conv3) 

W_conv4 = weight_variable([1, 1, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(m_pool3, W_conv4) + b_conv4)

W_conv5 = weight_variable([1, 3, 64, 128]) 
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
m_pool5 = max_pool_2x2(h_conv5) 

W_conv6 = weight_variable([1, 1, 128, 128])
b_conv6 = bias_variable([128])
h_conv6 = tf.nn.relu(conv2d(m_pool5, W_conv6) + b_conv6)

W_conv7 = weight_variable([1, 3, 128, 64])
b_conv7 = bias_variable([64])
h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)
m_pool7 = max_pool_2x2(h_conv7)

W_conv8 = weight_variable([1, 1, 64, 64])
b_conv8 = bias_variable([64])
h_conv8 = tf.nn.relu(conv2d(m_pool7, W_conv8) + b_conv8)

W_conv9= weight_variable([1, 3, 64, 32]) 
b_conv9 = bias_variable([32])
h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)
m_pool9 = max_pool_2x2(h_conv9) 

W_conv10 = weight_variable([1, 1, 32, 32])
b_conv10 = bias_variable([32])
h_conv10 = tf.nn.relu(conv2d(m_pool9, W_conv10) + b_conv10)


W_fc1 = weight_variable([16*32, 2])
b_fc1 = bias_variable([2])
h_conv10_flat = tf.reshape(h_conv10, [-1, 16*32])
h_conv10_drop = tf.nn.dropout(h_conv10_flat, keep_prob)
y_conv = tf.nn.softmax(tf.matmul(h_conv10_drop, W_fc1) + b_fc1)


print(tf.trainable_variables())

tf.add_to_collection("sword", y_conv)


cross_entropy = -tf.reduce_sum(ys * tf.log(tf.clip_by_value(y_conv, 1e-8, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()

test_acc_max = 0.0
cost_x = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for i in range(TRAINNUM + 1):

    X_train, Y_train = read_original_data_t.load_train_data(train5_record_read, train_record_read, train_unsure_read, batch_size, unsure_batch_size)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    if i % 5 == 0:
        cost_x.append(i)
        train_accuracy = accuracy.eval(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})
        train_entropy = cross_entropy.eval(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})
        train_loss.append(train_entropy)
        train_acc.append(train_accuracy)
    if i % 5 == 0:
        X_test, Y_test = read_original_data_t.load_test_data(test5_record_read, test_record_read, test_size)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        # print(X_test.shape, Y_test.shape)
        test_entropy = cross_entropy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0})
        test_loss.append(test_entropy)
        test_acc.append(test_accuracy)
        print("step %d, training loss %g, training accuracy %g" % (i, train_entropy, train_accuracy))
        print("step %d, testing loss %g, testing accuracy %g" % (i, test_entropy, test_accuracy))
        # print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0}))
        frecord.write("step %d, training loss %g, training accuracy %g" % (i, train_entropy, train_accuracy) + '\n')
        frecord.write("step %d, testing loss %g, testing accuracy %g" % (i, test_entropy, test_accuracy) + '\n')

        if test_accuracy > 0.9 and train_accuracy > 0.9:
            test_acc_madel = strsvin + '\\'+ str(test_accuracy)
            if not os.path.exists(test_acc_madel):
                os.makedirs(test_acc_madel)
            saver = tf.train.Saver()
            saver.save(sess, strsvin + '\\' + str(test_accuracy) + '/model.ckpt')

    train_step.run(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})


plt.xlabel('Iterations')
plt.ylabel('Training loss')
plt.plot(cost_x, train_loss)
plt.savefig(strsvin + '\\' + 'Training_loss.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Training accuracy')
plt.plot(cost_x, train_acc)
plt.savefig(strsvin + '\\' + 'Training_accuracy.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Testing loss')
plt.plot(cost_x, test_loss)
plt.savefig(strsvin + '\\' + 'Testing_loss.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Testing accuracy')
plt.plot(cost_x, test_acc)
plt.savefig(strsvin + '\\' + 'Testing_accuracy.jpg')
plt.show()

if not os.path.exists(strsvin + '/final'):
    os.makedirs(strsvin + '/final')
saver = tf.train.Saver()
saver.save(sess, strsvin + '/final' + '/model.ckpt')

end_time = time.time()
fsvin.write('endtime is ' + str(end_time) + '\n')
fsvin.write('costtime is ' + str(end_time - start_time) + '\n')
fsvin.close()
frecord.close()

print("Finished")