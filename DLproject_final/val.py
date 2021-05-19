# -*- coding: utf-8 -*-
# @Time : 2021/5/17 
# @Author : Jielong Tang & Shumeng Jia
# @File : val.py
# @function : It is a file to validate the 2D CNN+Bi-GRU model on 30% patients 1 to 10

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


model_test = keras.models.load_model('model_290-0.99.hdf5')

patients = ["01"]  # You may change the patient number you want to validate (1-10)
val_id, train_id, path_list = generatePathList(patients,test_size=0.3)
val_data = DataGenerator(val_id, path_list)

model_test.evaluate(val_data, batch_size=32)

y_hat = model_test.predict_classes(val_data)
y_val = []
for i in range(len(val_data)):
  list_val = list(val_data[i])
  y_val = np.hstack((y_val,list_val[1]))


tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
print('acuuracy \t', accuracy_score(y_val, y_hat))
sensitivity = int(tp)/int(tp+fn)
print('sensitivity \t', sensitivity)
specificity = int(tn)/int(fp+tn)
print('specificity \t', specificity)
print('F1-score \t', f1_score(y_val, y_hat))
