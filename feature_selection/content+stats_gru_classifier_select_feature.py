# coding='UTF-8'
import csv
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


def Bidirectional_GRU_sematic_stats(X_train_content, X_train_stats, y_train, X_val_content, X_val_stats, y_val, learning_rate, adam_decay, dropout_rate, batch_size, epochs):
    # content part
    hidden_gru_dim = 64
    content_input = Input(shape=(X_train_content.shape[1],X_train_content.shape[2]), name='content_bert_input')
    x = Bidirectional(GRU(hidden_gru_dim, return_sequences=True))(content_input)
    x = Dropout(dropout_rate)(x)
    # 添加attention层
    x = Flatten()(x)
    attention_probs = Dense(2 * hidden_gru_dim * X_train_content.shape[1], activation='softmax', name='attention_vec')(x)  # 200*2 * X_train_content.shape[1]
    attention_mul = Multiply()([attention_probs, x])
    content_feedforward_1 = Dense(256, name='main_feedforward_1')(attention_mul)
    content_feedforward_2 = Dense(128, activation='relu', name='main_feedforward_2')(content_feedforward_1)
    content_feedforward_3 = Dense(64, activation='relu', name='main_feedforward_3')(content_feedforward_2)

    # content_feedforward_linear = Dense(64, name='main_feedforward_linear')(attention_mul)

    # content_feedforward_total = Multiply()([content_feedforward_3, content_feedforward_linear])

    # stats part
    stats_input = Input(shape=(X_train_stats.shape[1],), name='stats_input')
    concat_layer = concatenate([content_feedforward_3, stats_input])
    x = Dense(128, activation='relu', name='merged_feedforward_1')(concat_layer)
    x = Dense(64, activation='relu', name='merged_feedforward_2')(x)
    possibility_outputs = Dense(1, activation='sigmoid', name='label_output', kernel_regularizer=regularizers.l2(0.01))(x)  # softmax  sigmoid
    
    model = Model(inputs=[content_input, stats_input], outputs=[possibility_outputs])  # stats_input
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision, recall, fmeasure])  # categorical_crossentropy  binary_crossentropy
    # print(model.summary())

    history = model.fit([X_train_content, X_train_stats], [y_train], batch_size, epochs, validation_data=([X_val_content, X_val_stats], [y_val]), shuffle=True, callbacks=[TensorBoard(log_dir='./tmp/log')])

    return model, history


if __name__ == '__main__':
    # 输入限制，文章数量及每篇文章输入的句子/段落数
    file_num_limit = 45614  # total 45614
    paras_limit=20
    
    # params get through skopt
    params = [[0.004557667500673525, 0.46474188753552637, 14, 225, 0.007771022239399846],
        [0.004576829527206503, 0.27664127235576896, 24, 229, 0.00684474968144855],
        [0.0016436320384269347, 0.43346422137319895, 18, 220, 0.005361316304365063],
        [0.0014289083912333465, 0.2692776633940966, 22, 117, 0.0026793135929090483],
        [0.001, 0.5, 20, 100, 0.001],
        [0.001, 0.5, 20, 100, 0.001]]

    encoded_contents, onehotlabels, stats_features = prepare_input(file_num_limit, paras_limit)

    # stats_features 标准化
    scaler = preprocessing.StandardScaler() #实例化
    scaler = scaler.fit(stats_features)
    stats_features = scaler.transform(stats_features)
    
    # 换算成二分类
    # no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
    flaw_evaluation = []
    for flaw_index in range(3, 5):
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

        # 在每个flaw下分别计算指标
        for feature_index in range(35, 65):
            X_train_stats_del_one = np.delete(X_train_stats, feature_index, axis=1)
            # 引入十折交叉验证
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            kfold_precision, kfold_recall, kfold_f1_score, kfold_acc, kfold_TNR, kfold_training_acc, kfold_training_loss = [], [], [], [], [], [], []
            fold_counter = 0
            for train, test in kfold.split(X_train_content, y_train):
                print('folder comes to:', fold_counter)
                _precision, _recall, _f1_score, _acc, _TNR, _model_training_acc, _model_training_loss = 0, 0, 0, 0, 0, 0, 0
                X_test_content_kfold, X_test_stats_kfold, y_test_kfold = X_train_content[test], X_train_stats_del_one[test], y_train[test]
                X_val_content_kfold, X_val_stats_kfold, y_val_kfold = X_train_content[train[-1000:]], X_train_stats_del_one[train[-1000:]], y_train[train[-1000:]]
                X_train_content_kfold, X_train_stats_kfold, y_train_kfold = X_train_content[train[:-1000]], X_train_stats_del_one[train[:-1000]], y_train[train[:-1000]]

                # 采用后1000条做验证集
                # X_val, y_val = X_train[-1000:], y_train[-1000:]
                # X_train, y_train = X_train[:-1000], y_train[:-1000]
                model, history = Bidirectional_GRU_sematic_stats(X_train_content_kfold, X_train_stats_kfold, y_train_kfold, 
                                                                    X_val_content_kfold, X_val_stats_kfold, y_val_kfold, 
                                                                    learning_rate, adam_decay, dropout_rate, batch_size, epochs)
                prediction = model.predict([X_test_content_kfold, X_test_stats_kfold])  # {'content_bert_input': X_test_content, 'stats_input': X_test_stats}
                _model_training_acc = history.history['acc']
                _model_training_loss = history.history['loss']
                _precision, _recall, _f1_score, _acc, _TNR = getAccuracy(prediction, y_test_kfold)
                print('precision:', _precision, 'recall', _recall, 'f1_score', _f1_score, 'accuracy', _acc, 'TNR', _TNR)
                kfold_precision.append(_precision)
                kfold_recall.append(_recall)
                kfold_f1_score.append(_f1_score)
                kfold_acc.append(_acc)
                kfold_TNR.append(_TNR)
                kfold_training_acc.append(_model_training_acc)
                kfold_training_loss.append(_model_training_loss)
                fold_counter += 1
                # Delete the Keras model with these hyper-parameters from memory.
                del model
        
                # Clear the Keras session, otherwise it will keep adding new
                # models to the same TensorFlow graph each time we create
                # a model with a different set of hyper-parameters.
                K.clear_session()
                tensorflow.reset_default_graph()

            with open('experiment_log.csv', 'a', newline='', encoding='utf-8-sig') as t_file:
                csv_writer = csv.writer(t_file)
                csv_writer.writerow((str(flaw_index), 'set' + str(feature_index), np.mean(kfold_precision), np.mean(kfold_recall), np.mean(kfold_f1_score), np.mean(kfold_acc), np.mean(kfold_TNR), np.mean(kfold_training_acc, axis=0), np.mean(kfold_training_loss, axis=0)))


