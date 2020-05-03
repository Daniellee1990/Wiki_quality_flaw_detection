# coding='UTF-8'
import time
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
from keras.layers import Input, Dense, LSTM, Bidirectional, Flatten, Dropout, Multiply, Permute, concatenate
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc

from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

from evaluation_metrics import  precision, recall, fbeta_score, fmeasure, getAccuracy


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


def Bidirectional_LSTM(X_train_content, X_train_stats, y_train, X_val_content, X_val_stats, y_val, learning_rate, adam_decay, dropout_rate, batch_size, epochs):
    X_train = X_train_content
    X_val = X_val_content
    hidden_lstm_dim = 64
    content_input = Input(shape=(X_train.shape[1],X_train.shape[2]), name='bert_sent_encoded_input')
    x = Bidirectional(LSTM(hidden_lstm_dim, return_sequences=True), input_shape=(X_train.shape[1],X_train.shape[2]))(content_input)
    x = Dropout(dropout_rate)(x)
    # 添加attention层
    x = Flatten()(x)
    attention_probs = Dense(2 * hidden_lstm_dim * X_train.shape[1], activation='softmax', name='attention_vec')(x)  # 200*2 * X_train.shape[1]
    attention_mul = Multiply()([attention_probs, x])
    content_feedforward = Dense(256, name='main_feedforward_0')(attention_mul)
    x = Dense(128, activation='relu', name='merged_feedforward_1')(content_feedforward)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(y_train.shape[1], activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    model = Model(inputs=content_input, outputs=possibility_outputs)
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())
        
    # 当评价指标不在提升时，减少学习率
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001) callbacks=[reduce_lr]
    history = model.fit(X_train, y_train, batch_size, epochs, validation_data=(X_val_content, y_val), shuffle=True, callbacks=[TensorBoard(log_dir='./tmp/log')])

    return model, history


def Bidirectional_LSTM_best(X_train_content, X_train_stats, y_train, X_val_content, X_val_stats, y_val, learning_rate, adam_decay, dropout_rate, batch_size, epochs):
    # 不同于前一个模型，采用和融合模型一样架构的网络，此模型专注于调优至最佳表现效果
    X_train = X_train_content
    X_val = X_val_content
    hidden_lstm_dim = 64
    content_input = Input(shape=(X_train.shape[1],X_train.shape[2]), name='bert_sent_encoded_input')
    x = Bidirectional(LSTM(hidden_lstm_dim, return_sequences=True))(content_input)
    x = Dropout(dropout_rate)(x)
    # 添加attention层
    x = Flatten()(x)
    attention_probs = Dense(2 * hidden_lstm_dim * X_train.shape[1], activation='softmax', name='attention_vec')(x)  # 200*2 * X_train.shape[1]
    attention_mul = Multiply()([attention_probs, x])
    content_feedforward = Dense(256, name='main_feedforward_0')(attention_mul)
    x = Dense(128, activation='relu', name='merged_feedforward_1')(content_feedforward)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(y_train.shape[1], activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    model = Model(inputs=content_input, outputs=possibility_outputs)
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    # 当评价指标不在提升时，减少学习率
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001) callbacks=[reduce_lr]
    history = model.fit(X_train, y_train, batch_size, epochs, validation_data=(X_val_content, y_val), shuffle=True, callbacks=[TensorBoard(log_dir='./tmp/log')])

    return model, history


if __name__ == '__main__':
    # 输入限制，文章数量及每篇文章输入的句子/段落数
    file_num_limit = 45614  # total 45614
    paras_limit=20

    # params get through skopt
    params = [[0.001571269434366171, 0.36268946225484333, 16, 106, 0.00935379023523214], 
        [0.0008292897335159474, 0.1274048133984453, 16, 110, 0.002030669517857267],
        [0.0005817702460382074, 0.1484524521866158, 20, 122, 6.435619657173066e-05],
        [0.0048937511285175725, 0.45477104723215256, 28, 175, 0.00923480472456522],
        [0.001, 0.5, 20, 100, 0.001],
        [0.004364933044776042, 0.4030139928338391, 22, 112, 0.007795486655988679]]

    encoded_contents, onehotlabels, stats_features = prepare_input(file_num_limit, paras_limit)

    # stats_features 标准化
    scaler = preprocessing.StandardScaler() #实例化
    scaler = scaler.fit(stats_features)
    stats_features = scaler.transform(stats_features)
    
    # 换算成二分类
    # no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
    flaw_evaluation = []
    flaw_evaluation_values = []
    for flaw_index in range(6):
        no_good_flaw_type = flaw_index # finished
        # 找出FA类的索引
        FA_indexs = [index for index in range(len(onehotlabels)) if sum([int(item) for item in onehotlabels[index]]) == 0]
        # 找出二分类另外一类的索引
        not_good_indexs = [index for index in range(len(onehotlabels)) if onehotlabels[index][no_good_flaw_type] > 0]
        binary_classification_indexs = FA_indexs + not_good_indexs
        print('FA count:', len(FA_indexs), 'no good count:', len(not_good_indexs))
        X_contents = np.array([encoded_contents[index,:,:] for index in binary_classification_indexs])
        y_train = np.array([onehotlabels[index] for index in binary_classification_indexs])
        X_stats = np.array([stats_features[index] for index in binary_classification_indexs])
        
        # 变成二分类的标签
        y_train = y_train[:,no_good_flaw_type]
        y_train = np.array([[label] for label in y_train])
        ### split data into training set and label set
        # X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
        ### params set choose
        target_param = params[no_good_flaw_type]

        learning_rate = target_param[0]
        dropout_rate = target_param[1]
        epochs = target_param[2]
        batch_size = target_param[3]
        adam_decay = target_param[4]
        ### create the deep learning models
        # 训练模型
        X_train_content, X_train_stats, y_train = X_contents, X_stats, y_train

        # 引入十折交叉验证
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        kfold_precision, kfold_recall, kfold_f1_score, kfold_acc, kfold_loss, kfold_time = [], [], [], [], [], []
        fold_counter = 0
        for train, test in kfold.split(X_train_content, y_train):
            print('folder comes to:', fold_counter)
            time_start=time.time()
            _precision, _recall, _f1_score, _acc, _loss = 0, 0, 0, 0, 0
            X_test_content_kfold, X_test_stats_kfold, y_test_kfold = X_train_content[test], X_train_stats[test], y_train[test]
            X_val_content_kfold, X_val_stats_kfold, y_val_kfold = X_train_content[train[-1000:]], X_train_stats[train[-1000:]], y_train[train[-1000:]]
            X_train_content_kfold, X_train_stats_kfold, y_train_kfold = X_train_content[train[:-1000]], X_train_stats[train[:-1000]], y_train[train[:-1000]]

            # 采用后1000条做验证集
            # X_val, y_val = X_train[-1000:], y_train[-1000:]
            # X_train, y_train = X_train[:-1000], y_train[:-1000]
            model, history = Bidirectional_LSTM_best(X_train_content_kfold, X_train_stats_kfold, y_train_kfold, 
                                                                X_val_content_kfold, X_val_stats_kfold, y_val_kfold, 
                                                                learning_rate, adam_decay, dropout_rate, batch_size, epochs)
            prediction = model.predict(X_test_content_kfold)  # {'content_bert_input': X_test_content, 'stats_input': X_test_stats}
            
            print(history.history['loss'], history.history['acc'], history.history['val_loss'], history.history['val_acc'])
            _precision = [history.history['precision'][0], history.history['precision'][-1], history.history['val_precision'][0], history.history['val_precision'][-1]]
            _recall = [history.history['recall'][0], history.history['recall'][-1], history.history['val_recall'][0], history.history['val_recall'][-1]]
            _f1_score = [history.history['fmeasure'][0], history.history['fmeasure'][-1], history.history['val_fmeasure'][0], history.history['val_fmeasure'][-1]]
            _acc = [history.history['acc'][0], history.history['acc'][-1], history.history['val_acc'][0], history.history['val_acc'][-1]]
            _loss = [history.history['loss'][0], history.history['loss'][-1], history.history['val_loss'][0], history.history['val_loss'][-1]]
            # _precision, _recall, _f1_score, _acc, _TNR = getAccuracy(prediction, y_test_kfold)
            print('precision:', _precision, 'recall', _recall, 'f1_score', _f1_score, 'accuracy', _acc, 'loss', _loss)
            kfold_precision.append(_precision)
            kfold_recall.append(_recall)
            kfold_f1_score.append(_f1_score)
            kfold_acc.append(_acc)
            kfold_loss.append(_loss)
            fold_counter += 1

            time_end=time.time()
            _time = time_end - time_start
            kfold_time.append(_time)
            print('totally cost',_time)
            # Delete the Keras model with these hyper-parameters from memory.
            del model

            # Clear the Keras session, otherwise it will keep adding new
            # models to the same TensorFlow graph each time we create
            # a model with a different set of hyper-parameters.
            K.clear_session()
            tensorflow.reset_default_graph()
        print('10 k average evaluation is:', 'precision:', np.mean(kfold_precision, axis=0), 'recall:', np.mean(kfold_recall, axis=0), 'f1_score:', np.mean(kfold_f1_score, axis=0), 'accuracy:', np.mean(kfold_acc, axis=0), 'loss:', np.mean(kfold_loss, axis=0))
        print('10 k average time is:', np.mean(kfold_time))
        
        evaluation_metrics_value = '10 k average evaluation is:' + ' precision:' + str(np.mean(kfold_precision, axis=0)) + 'recall:' + str(np.mean(kfold_recall, axis=0)) + 'f1_score:' + str(np.mean(kfold_f1_score, axis=0)) + 'accuracy:' + str(np.mean(kfold_acc, axis=0)) + 'loss:' + str(np.mean(kfold_loss, axis=0))
        evaluation_time_value = '10 k average time is:' + str(np.mean(kfold_time))
        evaluation_value = str(no_good_flaw_type) + ' ' + evaluation_metrics_value + '\n' + evaluation_time_value

        evaluation_pure_value  = np.mean(kfold_precision, axis=0).tolist() + np.mean(kfold_recall, axis=0).tolist() + np.mean(kfold_f1_score, axis=0).tolist() + np.mean(kfold_acc, axis=0).tolist() + np.mean(kfold_loss, axis=0).tolist() + np.mean(kfold_time).tolist()
        flaw_evaluation.append(evaluation_value)
        flaw_evaluation_values.append(evaluation_pure_value)

    for index in range(len(flaw_evaluation)):
        print(flaw_evaluation[index])
        for item in flaw_evaluation_values[index]:
            print(item)
    

