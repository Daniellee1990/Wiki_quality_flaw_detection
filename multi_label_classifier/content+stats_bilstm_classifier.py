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

from evaluation_metrics_multilabel import  precision, recall, fbeta_score, fmeasure, getAccuracy


def prepare_input(file_num_limit, paras_limit):
    sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    sampled_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    sampled_label_data = pd.read_csv(sampled_para_txt_list, encoding='utf-8-sig')

    # 取得全部文章的encoded content, 并将其解析成为(paras_limit, 768) 的shape
    contents = np.zeros((file_num_limit, paras_limit, 768))
    labels = np.zeros((file_num_limit, 6))
    stats_features = np.zeros((file_num_limit, 65))
    file_number_count = 0
    for file_index in range(len(para_encoded_txts)):
        file = para_encoded_txts[file_index]
        # 调整X的输入
        content = np.loadtxt(sampled_para_encoded_dir + file)
        content = content.reshape(-1, 768)
        # 调整Y的输入
        label_index = sampled_label_data[(sampled_label_data['file_name']==int(file.replace('.txt', '')))].index
        label = np.array(sampled_label_data.iloc[label_index, 2:8])
        if len(label) != 0:
            # 截断超过paras_limit个元素的content （para中指section超过paras_limit个的文章）
            for para_index in range(min(paras_limit, content.shape[0])):
                contents[file_number_count, para_index, :] = content[para_index, :]
            stats_feature = np.array(sampled_label_data.iloc[label_index, 10:75])
            stats_features[file_number_count, :] = stats_feature[:]
            # print(label_index, sampled_label_data['article_names'][label_index], label)
            labels[file_number_count, :] = label[:]
            file_number_count += 1

        if file_number_count + 1 >= file_num_limit:
            break
   
    # 仅存储有值部分的文本，去除后面的空值
    contents = contents[:file_number_count,:,:]
    labels = labels[:file_number_count, :]
    stats_features = stats_features[:file_number_count, :]

    return contents, labels, stats_features


# model
def Bidirectional_LSTM_sematic_stats(X_train_content, X_train_stats, y_train, X_val_content, X_val_stats, y_val, learning_rate, adam_decay, dropout_rate, batch_size, epochs):
    # content part
    hidden_lstm_dim = 64
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    x = Bidirectional(LSTM(hidden_lstm_dim, return_sequences=True))(content_input)
    x = Dropout(dropout_rate)(x)
    # 添加attention层
    x = Flatten()(x)
    attention_probs = Dense(2 * hidden_lstm_dim * X_train_content.shape[1], activation='softmax', name='attention_vec')(x)  # 200*2 * X_train_content.shape[1]
    attention_mul = Multiply()([attention_probs, x])
    content_feedforward_1 = Dense(256, name='main_feedforward_1')(attention_mul)
    content_feedforward_2 = Dense(128, activation='relu', name='main_feedforward_2')(content_feedforward_1)
    content_feedforward_3 = Dense(64, activation='relu', name='main_feedforward_3')(content_feedforward_2)

    # content_feedforward_linear = Dense(64, name='main_feedforward_linear')(attention_mul)

    # content_feedforward_total = Multiply()([content_feedforward_3, content_feedforward_linear])

    # stats part
    stats_input = Input(shape=(65,), name='stats_input')
    concat_layer = concatenate([content_feedforward_3, stats_input])
    x = Dense(128, activation='relu', name='merged_feedforward_1')(concat_layer)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(6, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    history = model.fit([X_train_content, X_train_stats], [y_train], batch_size, epochs, validation_data=([X_val_content, X_val_stats], [y_val]), shuffle=True, callbacks=[TensorBoard(log_dir='./tmp/log')])

    return model, history


if __name__ == '__main__':
    # 定义超参数
    learning_rate = 1e-3
    adam_decay = 1e-3
    dropout_rate = 0.5
    batch_size = 100
    epochs = 20

    # input prepare
    # 输入限制，文章数量及每篇文章输入的句子/段落数
    file_num_limit = 45614  # total 45614
    paras_limit = 20

    encoded_contents, onehotlabels, stats_features = prepare_input(file_num_limit, paras_limit)

    # stats_features 标准化
    scaler = preprocessing.StandardScaler() #实例化
    scaler = scaler.fit(stats_features)
    stats_features = scaler.transform(stats_features)

    ### split data into training set and label set
    # X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
    X_train_content, X_train_stats, y_train = np.array(encoded_contents), np.array(stats_features), np.array(onehotlabels)

    # 打乱操作
    index = [i for i in range(len(y_train))]
    np.random.shuffle(index)

    X_train_content = encoded_contents[index, :, :]
    X_train_stats = stats_features[index, :]
    y_train = onehotlabels[index, :]
    tmp_y_train = onehotlabels[index, 0]

    # 引入十折交叉验证
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    kfold_precision, kfold_recall, kfold_f1_score, kfold_acc, kfold_TNR = [], [], [], [], []
    fold_counter = 0
    for train, test in kfold.split(X_train_content, tmp_y_train):
        print('folder comes to:', fold_counter)
        _precision, _recall, _f1_score, _acc, _TNR = 0, 0, 0, 0, 0
        X_test_content_kfold, X_test_stats_kfold, y_test_kfold = X_train_content[test], X_train_stats[test], y_train[test]
        X_val_content_kfold, X_val_stats_kfold, y_val_kfold = X_train_content[train[-1000:]], X_train_stats[train[-1000:]], y_train[train[-1000:]]
        X_train_content_kfold, X_train_stats_kfold, y_train_kfold = X_train_content[train[:-1000]], X_train_stats[train[:-1000]], y_train[train[:-1000]]

        # 采用后1000条做验证集
        # X_val, y_val = X_train[-1000:], y_train[-1000:]
        # X_train, y_train = X_train[:-1000], y_train[:-1000]
        model, history = Bidirectional_LSTM_sematic_stats(X_train_content_kfold, X_train_stats_kfold, y_train_kfold, 
                                                                X_val_content_kfold, X_val_stats_kfold, y_val_kfold, 
                                                                learning_rate, adam_decay, dropout_rate, batch_size, epochs)
        prediction = model.predict([X_test_content_kfold, X_test_stats_kfold])  # {'content_bert_input': X_test_content, 'stats_input': X_test_stats}
        _precision, _recall, _f1_score, _acc, _TNR = getAccuracy(prediction, y_test_kfold)
        print('precision:', _precision, 'recall', _recall, 'f1_score', _f1_score, 'accuracy', _acc, 'TNR', _TNR)
        kfold_precision.append(_precision)
        kfold_recall.append(_recall)
        kfold_f1_score.append(_f1_score)
        kfold_acc.append(_acc)
        kfold_TNR.append(_TNR)
        fold_counter += 1
        # Delete the Keras model with these hyper-parameters from memory.
        del model
    
        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()
        tensorflow.reset_default_graph()
    print('10 k average evaluation is:', 'precision:', np.mean(kfold_precision), 'recall', np.mean(kfold_recall), 'f1_score', np.mean(kfold_f1_score), 'accuracy', np.mean(kfold_acc), 'TNR', np.mean(kfold_TNR))

