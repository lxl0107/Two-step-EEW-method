import time
import datetime
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import read_data_regression

# Large regression model
train5 = open('train_larger_than5.txt')
reg_train5 = train5.readlines()
train5.close()
test5 = open('test_larger_than5.txt')
reg_test5 = test5.readlines()
test5.close()

# Small regression model
# train = open('train_smaller_than5.txt')
# reg_train = train.readlines()
# train.close()
# test = open('test_smaller_than5.txt')
# reg_test = test.readlines()
# test.close()


TRAINNUM = 4000
batch_size = 128
test_size = 64


# 保存模型及运行过程信息
model_info = 'Large regression model' # train large regression model
training_time = datetime.datetime.now().strftime("%y-%m-%d+%h-%M-%S")
if not os.path.exists(r'./svntin'):
    os.makedirs(r'./svntin')
strsvin = r'./svntin' + '/' + model_info + training_time   #生成模型位置
if not os.path.exists(strsvin):
    os.makedirs(strsvin)


sess = tf.InteractiveSession()  # 创建session

def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
def avg_pool_6x6(x):
   return tf.nn.avg_pool(x,strides=[1, 1, 10, 1],ksize=[1, 1, 10, 1],padding="SAME")

x = tf.placeholder(tf.float32, [None, 1, 600, 3], name="x-sword")
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32, name='kp')

xs = tf.reshape(x, [-1, 1, 600, 3])


W_conv1 = weight_variable([1, 5, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([1, 5, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([1, 3, 64, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([1, 3, 32, 16])
b_conv5 = bias_variable([16])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

W_conv6 = weight_variable([1, 3, 16, 8])
b_conv6 = bias_variable([8])
h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)
print(h_pool6.eval)

W_fc1 = weight_variable([10 * 8, 64]) # 减小全连接层的参数个数[512,128/64/32]
b_fc1 = bias_variable([64])
h_pool1_flat = tf.reshape(h_pool6, [-1, 10 * 8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([64, 1]) # 对应128/64/32修改一下
b_fc2 = bias_variable([1])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

print(tf.trainable_variables())

tf.add_to_collection("sword", y_conv)

mse_loss = tf.reduce_mean(tf.square(ys-y_conv))
mae_loss = tf.reduce_mean(tf.abs(ys-y_conv))



train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(mse_loss)
tf.global_variables_initializer().run()

test_acc_max = 0.0
cost_x = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for i in range(TRAINNUM + 1):

    X_train, Y_train = read_data_regression.load_train_data(reg_train5, batch_size)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    if i % 5 == 0:
        cost_x.append(i)
        MSE = mse_loss.eval(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})
        train_loss.append(MSE)
    if i % 5 == 0:
        X_test, Y_test = read_data_regression.load_test_data(reg_test5, test_size)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        # print(X_test.shape, Y_test.shape)
        MAE = mae_loss.eval(feed_dict={x: X_train, ys: Y_train, keep_prob: 1.0})
        test_loss.append(MAE)
        print("step %d, training mse %g" % (i, MSE))

        print("step %d, testing mae %g" % (i, MAE))
        # print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0}))
        frecord.write("step %d, training mse %g" % (i, MSE) + '\n')
        frecord.write("step %d, testing mae %g" % (i, MAE) + '\n')


    train_step.run(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.65})


plt.xlabel('Iterations')
plt.ylabel('Training loss')
plt.plot(cost_x, train_loss)
plt.savefig(strsvin + '\\' + 'Training_loss.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Testing loss')
plt.plot(cost_x, test_loss)
plt.savefig(strsvin + '\\' + 'Testing_loss.jpg')
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