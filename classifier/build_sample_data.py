# coding='UTF-8'
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import keras
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Flatten, Dropout, Multiply, Permute, concatenate
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

from evaluation_metrics import  precision, recall, fbeta_score, fmeasure, getAccuracy


def prepare_input():
    # 仅统计数据部分不需要进行文件内容读取
    # sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    total_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    # para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    total_label_data = pd.read_csv(total_para_txt_list, encoding='utf-8-sig')
    
    sample_data = total_label_data.iloc[:1000, :].drop(['Unnamed: 0'],axis=1)
    
    sample_data.to_csv('labels_feature_stats_merged_sampled1000.csv')
    



if __name__ == '__main__':
    prepare_input()
