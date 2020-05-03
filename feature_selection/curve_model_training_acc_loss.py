# coding='UTF-8'
import io
import sys
import os
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # training acc loss curve of gru on different feature set
    data_file = './experiment_log.csv'
    data = pd.read_csv(data_file, encoding='utf-8-sig')

    curve_type = 'loss'  # acc, loss
    for flaw_index in range(6):
        curve_data = data[data.flaw_index==flaw_index]
        # set0:text statistic  set1: structure feature  set2: writing style  set3: readability score  set4: edit history
        text_statistic_data = curve_data[curve_data.feature_index=='set0']
        text_statistic = [float(item) for item in text_statistic_data['training_' + curve_type].tolist()[0].replace('[', '').replace(']', '').split()]
        
        structure_data = curve_data[curve_data.feature_index=='set1']
        structure = [float(item) for item in structure_data['training_' + curve_type].tolist()[0].replace('[', '').replace(']', '').split()]

        writing_style_data = curve_data[curve_data.feature_index=='set2']
        writing_style = [float(item) for item in writing_style_data['training_' + curve_type].tolist()[0].replace('[', '').replace(']', '').split()]

        readability_data = curve_data[curve_data.feature_index=='set3']
        readability = [float(item) for item in readability_data['training_' + curve_type].tolist()[0].replace('[', '').replace(']', '').split()]

        edit_history_data = curve_data[curve_data.feature_index=='set4']
        edit_history = [float(item) for item in edit_history_data['training_' + curve_type].tolist()[0].replace('[', '').replace(']', '').split()]
        x_data = [int(item) for item in range(len(text_statistic))]

        #开始画ROC曲线
        lw = 1.5
        plt.plot(x_data, text_statistic, color='#FD821A', linestyle='-', lw=lw, label='Text statistics')
        plt.plot(x_data, structure, color='#2BA02D', linestyle='--', lw=lw,label='Structure feature')
        plt.plot(x_data, writing_style, color='#1F77B4', linestyle='-', lw=lw,label='Writing style')
        plt.plot(x_data, readability, color='#D82625', linestyle='--', lw=lw,label='Readability scores')
        plt.plot(x_data, edit_history, color='#87564D', linestyle='-', lw=lw,label='Edit history')
        plt.legend(loc='upper right')  # lower right
        plt.locator_params('x',nbins=len(text_statistic))
        plt.xlim([-1,len(text_statistic)])
        plt.ylim([-0.05, 0.55])  # acc: 0.5, 1,1  loss: -0.05, 0.55
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.title('Model traning loss') # accuracy
        plt.show()

