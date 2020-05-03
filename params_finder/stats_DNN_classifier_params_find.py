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
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

from evaluation_metrics import  precision, recall, fbeta_score, fmeasure, getAccuracy

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


# target params
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                     name='learning_rate')
dim_epochs = Integer(low=10, high=30, name='epochs')
dim_batch_size = Integer(low=16, high=256, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2, name="adam_decay")

dimensions = [dim_learning_rate,
              dim_epochs,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-3, 20, 100, 1e-3]

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

# model
def DNN_stats(X_train_stats, y_train, learning_rate, adam_decay):
    # stats part
    stats_input = Input(shape=(65,), name='stats_input')
    x = Dense(128, activation='relu', name='merged_feedforward_1')(stats_input)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=stats_input, outputs=possibility_outputs)  # stats_input
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    return model

# skopt
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, epochs, batch_size, adam_decay):

    model = DNN_stats(
                         X_train_stats=X_train,
                         y_train=y_train,
                         learning_rate=learning_rate,
                         adam_decay=adam_decay
                        )
    #named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15,
                        shuffle=True,
                        )
    #return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.3%}".format(accuracy))
    print()


    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.reset_default_graph()
    
    return -accuracy


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
