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


def testlabelGenerate():
     """
     It is a function to generate test dataset for fine tune
     """
    parent_path = 'chbfig/'
    featureList = []
    labelList = []
    for indexPatient in range(0, len(test_patients)):  # len(patients)
        sub_path = 'chb' + test_patients[indexPatient] + '/'
        print(sub_path)
        directory_name = parent_path + sub_path
        features = np.zeros((len(os.listdir(directory_name)), 23, 256))
        labels = np.zeros((len(os.listdir(directory_name))), dtype=int)
        i = 0  # the feature index
        for filename in os.listdir(directory_name):
            # print(filename)
            if '_seizure_' in filename:
                im_features = np.array(Image.open(directory_name + filename))
                features[i] = np.vstack((im_features[0:23, :], np.zeros(256)))[0:23, :]
                labels[i] = 1
            elif '_nonseizure_' in filename:
                im_features = np.array(Image.open(directory_name + filename))
                features[i] = np.vstack((im_features[0:23, :], np.zeros(256)))[0:23, :]
                labels[i] = 0
            i = i + 1

        featureList.append(features)
        labelList.append(labels)

    X = featureList[0]
    Y = labelList[0]
    for j in range(1, len(featureList)):
        X = np.vstack((X, featureList[j]))
        Y = np.hstack((Y, labelList[j]))

    return X, Y

model_test = keras.models.load_model('model_290-0.99.hdf5')

test_patients = ["11"] # You may change the patient number you want to test (11-23)

test_features, test_labels = testlabelGenerate()
if test_features.ndim == 3:
    test_features = np.expand_dims(test_features, axis=3)

# train last 4 layers Bi-GRU and 2 dense layer
model_test.trainable = True
for layer in model_test.layers[:16]:
    layer.trainable = False

x_train, x_test, y_train, y_test = train_test_split(test_features, test_labels, test_size=0.4, random_state=1)
weights = collections.Counter(y_train)
weight = int(weights[0]/weights[1])

# Before fine tune

y_hat = model_test.predict_classes(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
print('acuuracy \t', accuracy_score(y_test, y_hat))
sensitivity = int(tp)/int(tp+fn)
print('sensitivity \t', sensitivity)
specificity = int(tn)/int(fp+tn)
print('specificity \t', specificity)
print('F1-score \t', f1_score(y_test, y_hat))

seed(1)
model_test.fit(x_train, y_train, batch_size=32, epochs=10, class_weight = {0:1, 1:weight}, shuffle=False)

# After fine tune

y_hat = model_test.predict_classes(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
print('acuuracy \t', accuracy_score(y_test, y_hat))
sensitivity = int(tp)/int(tp+fn)
print('sensitivity \t', sensitivity)
specificity = int(tn)/int(fp+tn)
print('specificity \t', specificity)
print('F1-score \t', f1_score(y_test, y_hat))

y_score = model_test.predict_proba(x_test)
fpr, tpr, threshold = roc_curve(y_test, y_score)

confusion_matrix(y_test, y_hat).ravel()
roc_auc = auc(fpr, tpr)
print('     AUC     \t', roc_auc)
plt.figure()

plt.figure(figsize=(5, 3))
A, = plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
B, = plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='AUC = 0.5')

legend = plt.legend(handles=[A, B], loc="lower right")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC AUC')
plt.show()
