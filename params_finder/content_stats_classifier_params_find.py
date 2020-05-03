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

def prepare_input(file_num_limit, paras_limit):
    sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    sampled_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    sampled_label_data = pd.read_csv(sampled_para_txt_list, encoding='utf-8-sig')

    # 取得全部文章的encoded content, 并将其解析成为(paras_limit, 768) 的shape
    contents = np.zeros((file_num_limit, paras_limit, 768))
    labels = []
    stats_features = []
    file_number_count = 0
    for file_index in range(len(para_encoded_txts)):
        file = para_encoded_txts[file_index]
        # 调整X的输入
        content = np.loadtxt(sampled_para_encoded_dir + file)
        content = content.reshape(-1, 768)
        # 调整Y的输入
        label_index = sampled_label_data[(sampled_label_data['file_name']==int(file.replace('.txt', '')))].index
        label = np.array(sampled_label_data.iloc[label_index, 2:8]).tolist()
        if len(label) != 0:
            # 截断超过paras_limit个元素的content （para中指section超过paras_limit个的文章）
            for para_index in range(min(paras_limit, content.shape[0])):
                contents[file_number_count, para_index, :] = content[para_index, :]
            stats_feature = np.array(sampled_label_data.iloc[label_index, 10:]).tolist()
            # print(label_index, sampled_label_data['article_names'][label_index], label)
            labels.append(label)
            stats_features.append(stats_feature)
            file_number_count += 1

        if file_number_count + 1 >= file_num_limit:
            break
   
    # 仅存储有值部分的文本，去除后面的空值
    contents = contents[:file_number_count,:,:]
    
    # 将Y转化为numpy表达，把格式调整成： n*1*6 -> n*6
    onehotlabels = [label[0] for label in labels]
    stats_features = [stats_feature[0] for stats_feature in stats_features]

    return contents, onehotlabels, stats_features


# target params
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                     name='learning_rate')
# dim_hidden_rnn = Integer(low=32, high=512, name='hidden_rnn')  # 此项不进行调整，GRU和RNN会报错
dropout_rate = Real(low=1e-1, high=6e-1, name='dropout_rate')
dim_epochs = Integer(low=10, high=30, name='epochs')
dim_batch_size = Integer(low=16, high=256, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2, name="adam_decay")

dimensions = [dim_learning_rate,
              dropout_rate,
              dim_epochs,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-3, 0.5, 20, 100, 1e-3]

# input prepare
# 输入限制，文章数量及每篇文章输入的句子/段落数
file_num_limit = 45614  # total 45614
paras_limit = 20

encoded_contents, onehotlabels, stats_features = prepare_input(file_num_limit, paras_limit)

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
encoded_contents = np.array([encoded_contents[index,:,:] for index in binary_classification_indexs])
onehotlabels = np.array([onehotlabels[index] for index in binary_classification_indexs])
stats_features = np.array([stats_features[index] for index in binary_classification_indexs])

# 变成二分类的标签
onehotlabels = onehotlabels[:,no_good_flaw_type]
onehotlabels = np.array([[label] for label in onehotlabels])
### split data into training set and label set
# X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
X_train_content, X_train_stats, y_train = encoded_contents, stats_features, onehotlabels

# 打乱操作
index = [i for i in range(len(y_train))]
np.random.shuffle(index)

X_train_content = encoded_contents[index]
X_train_stats = stats_features[index]
y_train = onehotlabels[index]

X_train = [X_train_content, X_train_stats]
y_train = y_train

# model
def Bidirectional_LSTM_sematic_stats(X_train, y_train, learning_rate, dropout_rate, adam_decay):
    # content part
    X_train_content = X_train[0]
    X_train_stats = X_train[1]
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
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    return model


def Bidirectional_RNN_sematic_stats(X_train, y_train, learning_rate, dropout_rate, adam_decay):
    # content part
    X_train_content = X_train[0]
    X_train_stats = X_train[1]
    hidden_rnn_dim = 64
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    x = Bidirectional(SimpleRNN(hidden_rnn_dim, return_sequences=True), input_shape=(X_train_content.shape[1],X_train_content.shape[2]))(content_input)
    x = Dropout(dropout_rate)(x)
    # 添加attention层
    x = Flatten()(x)
    attention_probs = Dense(2 * hidden_rnn_dim * X_train_content.shape[1], activation='softmax', name='attention_vec')(x)  # 200*2 * X_train_content.shape[1]
    attention_mul = Multiply()([attention_probs, x])
    content_feedforward_1 = Dense(256, activation='relu', name='main_feedforward_1')(attention_mul)
    content_feedforward_2 = Dense(128, activation='relu', name='main_feedforward_2')(content_feedforward_1)
    content_feedforward_3 = Dense(64, activation='relu', name='main_feedforward_3')(content_feedforward_2)

    # content_feedforward_linear = Dense(64, name='main_feedforward_linear')(attention_mul)

    # content_feedforward_total = Multiply()([content_feedforward_3, content_feedforward_linear])

    # stats part
    stats_input = Input(shape=(65,), name='stats_input')
    concat_layer = concatenate([content_feedforward_3, stats_input])
    x = Dense(128, activation='relu', name='merged_feedforward_1')(concat_layer)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    return model


def Bidirectional_GRU_sematic_stats(X_train, y_train, learning_rate, dropout_rate, adam_decay):
    # content part
    X_train_content = X_train[0]
    X_train_stats = X_train[1]
    hidden_gru_dim = 64
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    x = Bidirectional(GRU(hidden_gru_dim, return_sequences=True), input_shape=(X_train_content.shape[1],X_train_content.shape[2]))(content_input)
    x = Dropout(dropout_rate)(x)
    # 添加attention层
    x = Flatten()(x)
    attention_probs = Dense(2 * hidden_gru_dim * X_train_content.shape[1], activation='softmax', name='attention_vec')(x)  # 200*2 * X_train_content.shape[1]
    attention_mul = Multiply()([attention_probs, x])
    content_feedforward_1 = Dense(256, activation='relu', name='main_feedforward_1')(attention_mul)
    content_feedforward_2 = Dense(128, activation='relu', name='main_feedforward_2')(content_feedforward_1)
    content_feedforward_3 = Dense(64, activation='relu', name='main_feedforward_3')(content_feedforward_2)

    # content_feedforward_linear = Dense(64, name='main_feedforward_linear')(attention_mul)

    # content_feedforward_total = Multiply()([content_feedforward_3, content_feedforward_linear])

    # stats part
    stats_input = Input(shape=(65,), name='stats_input')
    concat_layer = concatenate([content_feedforward_3, stats_input])
    x = Dense(128, activation='relu', name='merged_feedforward_1')(concat_layer)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    return model

# skopt
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, dropout_rate,
            epochs, batch_size, adam_decay):

    model = Bidirectional_RNN_sematic_stats(
                         X_train=X_train,
                         y_train=y_train,
                         learning_rate=learning_rate,
                         dropout_rate=dropout_rate,
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
