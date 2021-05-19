  
# -*- coding: utf-8 -*-
# @Time : 2021/5/17 
# @Author : Jielong Tang & Shumeng Jia
# @File : train.py
# @function : It is a file to train the 2D CNN+Bi-GRU model on patients 1 to 10


import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Flatten, LSTM, Bidirectional, ConvLSTM2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, TimeDistributed, SimpleRNN, GRU
from keras.optimizers import SGD

import os
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import collections
from PIL import Image
from numpy.random import seed
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, fbeta_score

warnings.filterwarnings("ignore")

pathDataSet = 'physionet.org/files/chbmit/1.0.0/'  # path of the dataset
# the record of the patients we use to train
patients = ["01","02","03","04","05","06","07","08","09","10"]


def generatePathList(patients, test_size):
     """
     It is a function to shuttle the dataset and return appropriate index for training and testing
     :param patients: the patients list
     :param test_size: the test set rate 
     :return index[:test_index]: the index for testing
     :return index[test_index:]: the index for training
     :return pathList: the shuttle pathList
     """
    parent_path = 'chbfig/'
    pathList = []
    for indexPatient in range(0, len(patients)):
        sub_path = 'chb' + patients[indexPatient] + '/'
        directory_name = parent_path + sub_path
        for filename in os.listdir(directory_name):
            pathList.append(directory_name + filename)
    L = len(pathList)
    test_index = int(L * test_size)
    index = random.sample(range(L), L)
    return index[:test_index], index[test_index:], pathList


class DataGenerator(Sequence):
    '''
    It is a class implement keras Sequence, which is to generate image dataset with appropriate batch and use for model fit_generator function

    :param batch_size: the batch size
    :param parent_path: the dataset parent path
    :param pathList: the list to store all image path
    :param index: the training index or testing index
    :param L: the dataset length
    
    '''
    def __init__(self, index, pathList, parent_path='chbfig/', batch_size=32):
        self.batch_size = batch_size
        self.parent_path = parent_path
        self.pathList = pathList
        self.index = index
        self.L = len(self.index)

    def __len__(self):
        return self.L - self.batch_size

    def __getitem__(self, idx):
        batch_indexs = self.index[idx:(idx + self.batch_size)]
        image_path = [self.pathList[k] for k in batch_indexs]

        return self._load_image(image_path)

    def _load_image(self, image_path):
        features = np.zeros(((len(image_path)), 23, 256))
        labels = np.zeros((len(image_path)), dtype=int)
        i = 0  # the feature index
        for name in image_path:
            # print(name)
            if '_seizure_' in name:
                features[i] = np.array(Image.open(name))[0:23, :]
                labels[i] = 1
            elif '_nonseizure_' in name:
                features[i] = np.array(Image.open(name))[0:23, :]
                labels[i] = 0
            i = i + 1
        # print(features)
        # print(labels)
        # print(np.expand_dims(np.array(features), axis=3).shape)
        # print(labels.shape)
        return np.expand_dims(np.array(features), axis=3), labels


model = Sequential()

model.add(Conv2D(64, (2, 4), input_shape=((23, 256, 1))))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Dropout(0.15))

model.add(Conv2D(32, (2, 4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Dropout(0.15))

model.add(Conv2D(32, (2, 4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Dropout(0.15))

model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(GRU(32)))
model.add(Dropout(0.15))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(metrics=['accuracy',keras.metrics.Recall(name='sen')],
              loss='binary_crossentropy', optimizer=sgd)


seed(1)

save_dir = os.path.join(os.getcwd(), 'cks')
if os.path.isdir(save_dir) is not True:
        os.makedirs(save_dir)
filepath = "model_{epoch:03d}-{val_accuracy:.2f}-{val_sen:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, filepath), verbose=0,
                             save_best_only=False, save_weights_only=False)

test_id, train_id, path_list = generatePathList(patients,test_size=0.3)  # 30% use for validation 
train_data = DataGenerator(train_id, path_list)
test_data = DataGenerator(test_id, path_list)

model.fit_generator(generator=train_data, epochs=50, verbose=1, callbacks=[checkpoint], steps_per_epoch=None,
                    validation_data=test_data, class_weight={0:1, 1:50}, workers=4, use_multiprocessing=True,
                    shuffle=False, initial_epoch=0)
