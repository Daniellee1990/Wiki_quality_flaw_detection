# coding='UTF-8'
import io
import sys
import os
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import tensorflow
import keras
from keras import regularizers, optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Bidirectional, Flatten, Dropout, Multiply, Add, Permute, concatenate
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')  #解决print输出报编码错问题

def prepare_input(file_num_limit):
    # 仅统计数据部分不需要进行文件内容读取
    # sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    sampled_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    # para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    sampled_label_data = pd.read_csv(sampled_para_txt_list, encoding='utf-8-sig')
    onehotlabels = sampled_label_data.iloc[:file_num_limit,2:8].values
    stats_features = sampled_label_data.iloc[:file_num_limit,10:].values

    return onehotlabels, stats_features

def getAccuracy(prediction, y_test): ### prediction and y_test are both encoded.
    sample_size = len(prediction)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    wrong_num = 0
    for i in range(sample_size):
        if np.round(prediction[i]) == 1 and np.round(y_test[i]) == 1:
            true_positive = true_positive + 1
        elif np.round(prediction[i]) == 1 and np.round(y_test[i]) == 0:
            false_positive = false_positive + 1
        elif np.round(prediction[i]) == 0 and np.round(y_test[i]) == 1:
            false_negative = false_negative + 1
        elif np.round(prediction[i]) == 0 and np.round(y_test[i]) == 0:
            true_negative = true_negative + 1
    print('tp:', true_positive, 'fp:', false_positive)
    print('fn:', false_negative, 'tn:', true_negative)
    precision = float(true_positive) / (float(true_positive) + float(false_positive))
    recall = float(true_positive) / (float(true_positive) + float(false_negative))
    f1_score = (2 * precision * recall) / (precision + recall)
    acc = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    TNR = float(true_negative) / (float(false_positive) + float(true_negative))
    return precision, recall, f1_score, acc, TNR


# target params
max_depth = Integer(low=1, high=32, name='max_depth')
min_samples_split = Real(low=0.1, high=1, name='min_samples_split')
min_samples_leaf = Real(low=0.1, high=0.5, name='min_samples_leaf')
max_features = Integer(low=1,high=65, name="max_features")

dimensions = [max_depth,
              min_samples_split,
              min_samples_leaf,
              max_features
             ]
default_parameters = [5, 0.2, 0.2, 15]

# input prepare
# 输入限制，文章数量及每篇文章输入的句子/段落数
file_num_limit = 45614  # total 45614
paras_limit = 20

onehotlabels, stats_features = prepare_input(file_num_limit)

# stats_features 标准化
scaler = preprocessing.StandardScaler() #实例化
scaler = scaler.fit(stats_features)
stats_features = scaler.transform(stats_features)

# 换算成二分类
# no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
no_good_flaw_type = 5 # finished
# 找出FA类的索引
FA_indexs = [index for index in range(len(onehotlabels)) if sum([int(item) for item in onehotlabels[index]]) == 0]
# 找出二分类另外一类的索引
not_good_indexs = [index for index in range(len(onehotlabels)) if onehotlabels[index][no_good_flaw_type] > 0]
binary_classification_indexs = FA_indexs + not_good_indexs
print('FA count:', len(FA_indexs), 'no good count:', len(not_good_indexs))
onehotlabels = np.array([onehotlabels[index] for index in binary_classification_indexs])
stats_features = np.array([stats_features[index] for index in binary_classification_indexs])

# 变成二分类的标签
onehotlabels = onehotlabels[:,no_good_flaw_type]
onehotlabels = np.array([[label] for label in onehotlabels])
### split data into training set and label set
# X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
X_train_stats, y_train = stats_features, onehotlabels

# 打乱操作
index = [i for i in range(len(y_train))]
np.random.shuffle(index)

X_train_stats = stats_features[index]
y_train = onehotlabels[index]

X_train = X_train_stats
y_train = y_train

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# skopt
@use_named_args(dimensions=dimensions)
def fitness(max_depth, min_samples_split, min_samples_leaf, max_features):

    model = tree.DecisionTreeClassifier(
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         max_features=max_features
                        )
    #named blackbox becuase it represents the structure
    blackbox = model.fit(X_train, y_train)
    #calculate the validation accuracy for the last epoch.
    prediction = model.predict(X_test)
    precision, recall, f1_score, acc, TNR = getAccuracy(prediction, y_test)

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.3%}".format(acc))
    print()


    # Delete the DT model with these hyper-parameters from memory.
    del model
    
    return -acc


K.clear_session()
tensorflow.reset_default_graph()
gp_result = gp_minimize(func=fitness,
                    dimensions=dimensions,
                    n_calls=12,
                    noise= 0.01,
                    n_jobs=-1,
                    kappa = 5,
                    x0=default_parameters)

print("best accuracy was " + str(round(gp_result.fun *-100,2))+"%.")
print(gp_result.x)
print(gp_result.func_vals)
