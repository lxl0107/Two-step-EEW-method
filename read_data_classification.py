import math
import random
import numpy as np


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


wave_length = 400

def load_train_data(train5_record, train_record, train_unsure_record, unsure_num, num):
    X_train = []
    Y_train = []

    # 5+
    index5_list = list(range(0, int(len(train5_record))))
    random.shuffle(index5_list)
    for i in range(num):
        index = index5_list[i]
        onset = int(train5_record[index].split()[0])
        ear = train5_record[index].split()[1].replace('\n', '')
        fr = open(ear)
        fread = fr.read()
        fr.close()
        ori_data = [float(l) for l in fread.split()[:]]
        data = ori_data[onset:onset + wave_length]
        detrend10_sen = detrend10_sensitivity(data, ear)
        X_train.append(np.array(detrend10_sen).reshape(1, wave_length, 1))
        Y_train.append([1, 0])

    # 5-
    index3_list = list(range(0, int(len(train_record))))
    random.shuffle(index3_list)
    for k in range(num):
        index = index3_list[k]
        onset = int(train_record[index].split()[0])
        ear = train_record[index].split()[1].replace('\n', '')
        fr = open(ear)
        fread = fr.read()
        fr.close()
        ori_data = [float(l) for l in fread.split()[:]]
        data = ori_data[onset:onset + wave_length]
        detrend10_sen = detrend10_sensitivity(data, ear)
        X_train.append(np.array(detrend10_sen).reshape(1, wave_length, 1))
        Y_train.append([0, 1])
    
    # 45-52
    index_unsure_list = list(range(0, int(len(train_unsure_record))))
    random.shuffle(index_unsure_list)
    for k in range(unsure_num):
        index = index_unsure_list[k]
        onset = int(train_unsure_record[index].split()[0])
        ear = train_unsure_record[index].split()[1].replace('\n', '')
        fr = open(ear)
        fread = fr.read()
        fr.close()
        ori_data = [float(l) for l in fread.split()[:]]
        data = ori_data[onset:onset + wave_length]
        detrend10_sen = detrend10_sensitivity(data, ear)
        X_train.append(np.array(detrend10_sen).reshape(1, wave_length, 1))
        Y_train.append([0.5, 0.5])
    return X_train, Y_train


def load_test_data(test5_record, test_record, num):
    X_test = []
    Y_test = []
    # 5+
    index5_list = list(range(0, int(len(test5_record))))
    random.shuffle(index5_list)
    for i in range(num):
        index = index5_list[i]
        onset = int(test5_record[index].split()[0])
        ear = test5_record[index].split()[1].replace('\n', '')
        fr = open(ear)
        fread = fr.read()
        fr.close()
        ori_data = [float(l) for l in fread.split()[:]]
        data = ori_data[onset:onset + wave_length]
        detrend10_sen = detrend10_sensitivity(data, ear)
        X_test.append(np.array(detrend10_sen).reshape(1, wave_length, 1))
        Y_test.append([1, 0])

    # 5-
    index3_list = list(range(0, int(len(test_record))))
    random.shuffle(index3_list)
    for k in range(num):
        index = index3_list[k]
        onset = int(test_record[index].split()[0])
        ear = test_record[index].split()[1].replace('\n', '')
        fr = open(ear)
        fread = fr.read()
        fr.close()
        ori_data = [float(l) for l in fread.split()[:]]
        data = ori_data[onset:onset + wave_length]
        detrend10_sen = qushi10_sensitivity(data, ear)
        X_test.append(np.array(detrend10_sen).reshape(1, wave_length, 1))
        Y_test.append([0, 1])
    return X_test, Y_test