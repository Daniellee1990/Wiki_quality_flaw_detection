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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')  #解决print输出报编码错问题

def prepare_input(file_num_limit):
    # 仅统计数据部分不需要进行文件内容读取
    # sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    sampled_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    # para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    sampled_label_data = pd.read_csv(sampled_para_txt_list, encoding='utf-8-sig')
    onehotlabels = sampled_label_data.iloc[:file_num_limit,2:8]
    stats_features = sampled_label_data.iloc[:file_num_limit,10:]

    return onehotlabels, stats_features


def DNN_stats(X_train_stats, y_train, X_val_stats, y_val, batch_size, epochs):
    # stats part
    stats_input = Input(shape=(65,), name='stats_input')
    concat_layer = concatenate([content_feedforward, stats_input])
    x = Dense(128, activation='relu', name='merged_feedforward_1')(concat_layer)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=stats_input, outputs=possibility_outputs)  # stats_input
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    history = model.fit(X_train_stats, y_train, batch_size, epochs, validation_data=(X_val_stats, y_val), shuffle=True)  # callbacks=[TensorBoard(log_dir='./tmp/log')]

    return model, history


if __name__ == '__main__':
    # 输入限制，用于分类的文章数量
    file_num_limit = 45614  # total 45614
    paras_limit=20

    onehotlabels, stats_features = prepare_input(file_num_limit)

    # stats_features 标准化
    scaler = preprocessing.StandardScaler() #实例化
    scaler = scaler.fit(stats_features)
    stats_features = scaler.transform(stats_features)
    
    # 换算成二分类
    # no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
    no_good_flaw_type = 0 # finished
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
    ### create the deep learning models
    epochs = 20
    batch_size = 100
    # 训练模型
    X_train_stats, y_train = stats_features, onehotlabels

    # 引入十折交叉验证
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    kfold_accuracy, kfold_recall, kfold_f1_score = [], [], []
    fold_counter = 0
    for train, test in kfold.split(X_train_stats, y_train):
        print('folder comes to:', fold_counter)
        X_test_stats_kfold, y_test_kfold = X_train_stats[test], y_train[test]
        X_val_stats_kfold, y_val_kfold = X_train_stats[train[-1000:]], y_train[train[-1000:]]
        X_train_stats_kfold, y_train_kfold = X_train_stats[train[:-1000]], y_train[train[:-1000]]

        # 采用后1000条做验证集
        # X_val, y_val = X_train[-1000:], y_train[-1000:]
        # X_train, y_train = X_train[:-1000], y_train[:-1000]
        model, history = DNN_stats(X_train_stats_kfold, y_train_kfold, X_val_stats_kfold, y_val_kfold, batch_size, epochs)
        prediction = model.predict(X_test_content_kfold)  # {'content_bert_input': X_test_content, 'stats_input': X_test_stats}
        _precision, _recall, _f1_score = getAccuracy(prediction, y_test_kfold)
        print('precision:', _precision, 'recall', _recall, 'f1_score', _f1_score)
        kfold_accuracy.append(_precision)
        kfold_recall.append(_recall)
        kfold_f1_score.append(_f1_score)
        fold_counter += 1
    print('10 k average evaluation is:', 'precision:', np.mean(kfold_accuracy), 'recall', np.mean(kfold_recall), 'f1_score', np.mean(kfold_f1_score))


