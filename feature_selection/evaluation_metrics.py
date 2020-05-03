# coding='UTF-8'
import io
import sys
import os
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import keras
from keras import regularizers
from keras import backend as K
from keras.utils import np_utils

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')  #解决print输出报编码错问题


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # K.clip(y_true * y_pred, 0, 1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
 
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    # beta<1时,模型选对正确的标签更加重要,而beta>1时,模型对选错标签有更大的惩罚.
    return fbeta_score(y_true, y_pred, beta=1)

def getAccuracy(prediction, y_test): ### prediction and y_test are both encoded.
    sample_size = prediction.shape[0]
    col_num = prediction.shape[1]
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    wrong_num = 0
    for i in range(sample_size):
        if round(prediction[i][0]) == 1 and round(y_test[i][0]) == 1:
            true_positive = true_positive + 1
        elif round(prediction[i][0]) == 1 and round(y_test[i][0]) == 0:
            false_positive = false_positive + 1
        elif round(prediction[i][0]) == 0 and round(y_test[i][0]) == 1:
            false_negative = false_negative + 1
        elif round(prediction[i][0]) == 0 and round(y_test[i][0]) == 0:
            true_negative = true_negative + 1
    precision = float(true_positive) / (float(true_positive) + float(false_positive))
    recall = float(true_positive) / (float(true_positive) + float(false_negative))
    f1_score = (2 * precision * recall) / (precision + recall)
    acc = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    TNR = float(true_negative) / (float(false_positive) + float(true_negative))
    return precision, recall, f1_score, acc, TNR
