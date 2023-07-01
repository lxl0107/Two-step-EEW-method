import math
import random
import numpy as np


def detrend10_standardization(ori_list):
    # detrend
    a = np.polyfit(range(len(ori_list)), ori_list[:], 10) 
    b = np.poly1d(a)  
    c = b(range(len(ori_list)))  
    qushi = [(ori_list[i] - c[i]) for i in range(len(ori_list))]
    # standardization
    mean = np.mean(np.array(qushi))
    std = np.std(np.array(qushi), ddof=1)
    nor = []
    for i in range(len(qushi)):
        nor.append((qushi[i] - mean) / std)
    return nor

wave_length = 400

def load_train_data(train5_record, num):
    usesize = [1, wave_length]
    lsep = np.zeros([1, wave_length, 3])
    X_train = []
    Y_train = []
    # 5+
    index5_list = list(range(0, int(len(train5_record)/3)))
    random.shuffle(index5_list)
    for i in range(num):
        for j in range(3):
            index = index5_list[i] * 3 + j
            onset = int(train5_record[index].split()[0])
            ear = train5_record[index].split()[1].replace('\n', '')
            event_name = ear.split('\\')[-2]
            level = catalog_more_than_5[event_name]['lev'] # catalog_more_than_5: the catalog of earthquakes larger than 5
            fr = open(ear)
            fread = fr.read()
            fr.close()
            ori_data = [float(l) for l in fread.split()[:]]
            data = ori_data[onset:onset + wave_length]
            detrend10_nor = detrend10_standardization(data)
            lsep[:, :, j] = np.array(detrend10_nor).reshape(usesize)
        X_train.append(np.array(lsep).reshape(1, wave_length, 3))
        Y_train.append([level])

    # # 5-
    # index3_list = list(range(0, int(len(train_record)/3)))
    # random.shuffle(index3_list)
    # for k in range(num):
    #     for l in range(3):
    #         index = index3_list[k] * 3 + l
    #         onset = int(train_record[index].split()[0])
    #         ear = train_record[index].split()[1].replace('\n', '')
    #         event_name = ear.split('\\')[-2]
#             level = catalog_less_than_5[event_name]['lev'] # catalog_less_than_5: the catalog of earthquakes smaller than 5
    #         fr = open(ear)
    #         fread = fr.read()
    #         fr.close()
    #         ori_data = [float(l) for l in fread.split()[:]]
    #         data = ori_data[onset:onset + wave_length]
    #         detrend10_nor = detrend10_standardization(data)
    #         lsep[:, :, l] = np.array(detrend10_nor).reshape(usesize)
    #     X_train.append(np.array(lsep).reshape(1, wave_length, 3))
    #     Y_train.append([level])
    return X_train, Y_train


def load_test_data(test5_record, num):
    usesize = [1, wave_length]
    lsep = np.zeros([1, wave_length, 3])
    X_test = []
    Y_test = []
    # 5+
    index5_list = list(range(0, int(len(test5_record)/3)))
    random.shuffle(index5_list)
    for i in range(num):
        for j in range(3):
            index = index5_list[i] * 3 + j
            onset = int(test5_record[index].split()[0])
            ear = test5_record[index].split()[1].replace('\n', '')
            event_name = ear.split('\\')[-2]
            level = catalog_more_than_5[event_name]['lev']
            fr = open(ear)
            fread = fr.read()
            fr.close()
            ori_data = [float(l) for l in fread.split()[:]]
            data = ori_data[onset:onset + wave_length]
            detrend10_nor = detrend10_standardization(data)
            lsep[:, :, j] = np.array(detrend10_nor).reshape(usesize)
        X_test.append(np.array(lsep).reshape(1, wave_length, 3))
        Y_test.append([level])

    # # 5-
    # index3_list = list(range(0, int(len(test_record)/3)))
    # random.shuffle(index3_list)
    # for k in range(num):
    #     for l in range(3):
    #         index = index3_list[k] * 3 + l
    #         onset = int(test_record[index].split()[0])
    #         ear = test_record[index].split()[1].replace('\n', '')
    #         event_name = ear.split('\\')[-2]
    #         level = catalog_less_than_5[event_name]['lev']
    #         event_name = ear.split('\\')[-2]
    #         level = catalog_less_than_5[event_name]['lev']
    #         fr = open(ear)
    #         fread = fr.read()
    #         fr.close()
    #         ori_data = [float(l) for l in fread.split()[:]]
    #         data = ori_data[onset:onset + wave_length]
    #         detrend10_nor = detrend10_standardization(data)
    #         lsep[:, :, l] = np.array(detrend10_nor).reshape(usesize)
    #     X_test.append(np.array(lsep).reshape(1, wave_length, 3))
    #     Y_test.append([level])
    return X_test, Y_test