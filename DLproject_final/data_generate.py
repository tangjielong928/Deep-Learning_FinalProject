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


def loadSummaryPatient(index):
    f = open(pathDataSet+'chb'+patients[index]+'/chb'+patients[index]+'-summary.txt', 'r')
    parent = 'chb'+patients[index]+'/'
    return f, parent

def seizureImageGenerate(secSt, secEn, name_F, parent):
    file1 = pyedflib.EdfReader(pathDataSet+parent+name_F)
    n = file1.signals_in_file
    signal_labels = file1.getSignalLabels()
    signal_headers = file1.getSignalHeaders()
    rate = signal_headers[0]['sample_rate']
    dur = file1.getFileDuration()
    x = np.zeros((n, file1.getNSamples()[0]))
    for i in range(n):
        x[i,:] = file1.readSignal(i)
        label = file1.getLabel(i)
    file1.close()
    # a = os.getcwd()
    path = 'chbfig/'+ parent
    if os.path.isdir(path) is not True:
        os.makedirs(path)
    picnum = int(dur*rate/256)
    for i in range(picnum):
        img = x[:,i*256:(i+1)*256]
        Img = Image.fromarray(np.uint8(img))
        if secSt <= i+1 <= secEn:
            filename = '_seizure_'+ str(i)
            Img.save(path + name_F.split('.')[0] + filename+'.jpg')
        else:
            filename = '_nonseizure_'+ str(i)
            Img.save(path + name_F.split('.')[0] + filename+'.jpg')


def createDataset():
    print("START \n")
    for indexPatient in range(0, len(patients)):
        # fileList = []

        f, parent = loadSummaryPatient(indexPatient)
        line = f.readline()
        while (line):
            data = line.split(':')
            if (data[0] == "File Name"):
                name_F = data[1].strip()
                # print(name_F)
                for i in range(3):
                    line = f.readline()
                for j in range(0, int(line.split(': ')[1])):
                    secSt = int(f.readline().split(': ')[1].split(' ')[0])
                    secEn = int(f.readline().split(': ')[1].split(' ')[0])
                    seizureImageGenerate(secSt, secEn, name_F, parent)

            line = f.readline()
        f.close()

    print("END \n")


createDataset()

