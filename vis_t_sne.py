import tensorflow as tf
import numpy as np
import math
import time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn import preprocessing


def detrend10_sensitivity(ori_list, ear):
    # detrend
    a = np.polyfit(range(len(ori_list)), ori_list[:], 10)
    b = np.poly1d(a)
    c = b(range(len(ori_list)))
    qushi = [(ori_list[i] - c[i]) for i in range(len(ori_list))]
    # sensitivity
    sen = []
    ear_tz = ear.split('\\')[-1].split('.')[0] + '.' + ear.split('\\')[-1].split('.')[1]
    ear_tz = ear_tz.upper()
    if 'BHE' in ear:
        for i in range(len(qushi)):
            sen.append(qushi[i] / tz_cata[ear_tz]['sensitivity']['E'])
    elif 'BHN' in ear:
        for i in range(len(qushi)):
            sen.append(qushi[i] / tz_cata[ear_tz]['sensitivity']['N'])
    else:
        for i in range(len(qushi)):
            sen.append(qushi[i] / tz_cata[ear_tz]['sensitivity']['Z'])
    return sen


sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('D:/final/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint("D:/final/"))
output = tf.get_collection("sword")
last_conv = tf.get_collection("last_layer")
flatten_vector = tf.get_collection("flatten")

wave_length = 400
test_list = []
# 5+
train5_record = open('test_larger_than5.txt')
train5_record_read = train5_record.readlines()
train5_record.close()
test_list.append(train5_record_read)

train_record = open('test_smaller_than5.txt')
train_record_read = train_record.readlines()
train_record.close()
test_list.append(train_record_read)

fv = []
fv_label = []
ori = []
for i in range(len(test_list)):
    random.shuffle(test_list[i])
    for j in range(len(test_list[i])):
        if j > 200:
            break
        ear = test_list[i][j].split()[1].replace('\n', '')
        onset = int(test_list[i][j].split()[0])
        event_name = ear.split('\\')[-2]
        ear_tz = ear.split('\\')[-1].split('.')[0] + '.' + ear.split('\\')[-1].split('.')[1]
        fr = open(ear)
        fread = fr.read()
        fr.close()
        ori_data = [float(k) for k in fread.split()[:]]
        data = ori_data[onset:onset + wave_length]
        detrend10_sen = detrend10_sensitivity(data, ear)
        X_test = np.array(detrend10_sen).reshape([1, 1, wave_length, 1])
        result = sess.run(flatten_vector, feed_dict={'x-sword:0': X_test, 'kp:0': 1.0})[0]
        fv_vector = result.reshape(13*32)
        if i == 0:
            ori.append(data)
            fv.append(fv_vector)
            fv_label.append(0)
        else:
            ori.append(data)
            fv.append(fv_vector)
            fv_label.append(1)
fv5 = np.array(fv)
ori = np.array(ori)

# iris = datasets.load_iris()
# x, y = iris["data"], iris["target"]
# print(type(y[1]))
#
tsne = TSNE(
    perplexity=50,
    n_iter=500,
    metric="euclidean",
    # callbacks=ErrorLogger(),
    n_jobs=8,
    random_state=42,
)
# embedding = tsne.fit(ori) # fv5
# plot(embedding, fv_label) # , colors=MOUSE_10X_COLORS

embedding = tsne.fit(fv5) # fv5
plot(embedding, fv_label) # , colors=MOUSE_10X_COLORS