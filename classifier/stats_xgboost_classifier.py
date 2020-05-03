# coding='UTF-8'
import re
import pandas as pd
import nltk
import nltk.data
import numpy as np
import keras
from sklearn import svm 
from sklearn import tree
from sklearn import metrics
import xgboost as xgb
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


def prepare_input(file_num_limit):
    # 仅统计数据部分不需要进行文件内容读取
    # sampled_para_encoded_dir = './encoded_para_sep/'  # encoded_para_sep_sampled
    sampled_para_txt_list = './labels_feature_stats_merged_cleaned20200223.csv'  # plain_txt_name_index_sampled.csv

    # para_encoded_txts = os.listdir(sampled_para_encoded_dir)
    sampled_label_data = pd.read_csv(sampled_para_txt_list, encoding='utf-8-sig')
    onehotlabels = sampled_label_data.iloc[:file_num_limit,2:8].values
    stats_features = sampled_label_data.iloc[:file_num_limit,10:].values

    return onehotlabels, stats_features


def xgboost_model(X_train, y_train, X_test, y_test, max_depth, min_child_weight):
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtest = xgb.DMatrix(X_test)
    # 参数设置
    params={'booster':'gbtree','objective': 'binary:logistic','eval_metric': 'auc','max_depth':max_depth,'lambda':10,'subsample':0.75,'colsample_bytree':0.75,'min_child_weight':min_child_weight,'eta': 0.025,'seed':0,'nthread':8,'silent':1}
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
    ypred=bst.predict(dtest)
    # 设置阈值, 输出一些评价指标
    # 0.5为阈值，ypred >= 0.5输出0或1
    y_pred = (ypred >= 0.5)*1
    # ROC曲线下与坐标轴围成的面积
    print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
    # 准确率
    print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
    print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
    # 精确率和召回率的调和平均数
    print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
    print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
    metrics.confusion_matrix(y_test,y_pred)

    return y_test, y_pred


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


if __name__ == '__main__':
    # 输入限制，用于分类的文章数量
    file_num_limit = 45614  # total 45614
    paras_limit=20

    # params get through manual
    params = [[7, 1],
        [12, 1],
        [9, 1],
        [11, 1],
        [9, 4],
        [10, 1]]

    onehotlabels, stats_features = prepare_input(file_num_limit)

    # stats_features 标准化
    scaler = preprocessing.StandardScaler() #实例化
    scaler = scaler.fit(stats_features)
    stats_features = scaler.transform(stats_features)
    
    # 换算成二分类
    # no_footnotes-0, primary_sources-1, refimprove-2, original_research-3, advert-4, notability-5
    flaw_evaluation = []
    for flaw_index in range(6):
        no_good_flaw_type = flaw_index # finished
        # 找出FA类的索引
        FA_indexs = [index for index in range(len(onehotlabels)) if sum([int(item) for item in onehotlabels[index]]) == 0]
        # 找出二分类另外一类的索引
        not_good_indexs = [index for index in range(len(onehotlabels)) if onehotlabels[index][no_good_flaw_type] > 0]
        binary_classification_indexs = FA_indexs + not_good_indexs
        print('FA count:', len(FA_indexs), 'no good count:', len(not_good_indexs))
        y_train = np.array([onehotlabels[index] for index in binary_classification_indexs])
        X_stats = np.array([stats_features[index] for index in binary_classification_indexs])
        
        # 变成二分类的标签
        y_train = y_train[:,no_good_flaw_type]
        y_train = np.array([[label] for label in y_train])
        ### split data into training set and label set
        # X_train, X_test, y_train, y_test = train_test_split(encoded_contents, onehotlabels, test_size=0.1, random_state=42)
        ### params set choose
        target_param = params[no_good_flaw_type]

        max_depth = target_param[0]
        min_child_weight = target_param[1]
        ### create the deep learning models
        # 训练模型
        X_train_stats, y_train = X_stats, y_train

        # 引入十折交叉验证
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        kfold_precision, kfold_recall, kfold_f1_score, kfold_acc, kfold_TNR = [], [], [], [], []
        fold_counter = 0
        for train, test in kfold.split(X_train_stats, y_train):
            print('folder comes to:', fold_counter)
            _precision, _recall, _f1_score, _acc, _TNR = 0, 0, 0, 0, 0
            X_test_stats_kfold, y_test_kfold = X_train_stats[test], y_train[test]
            X_val_stats_kfold, y_val_kfold = X_train_stats[train[-1000:]], y_train[train[-1000:]]
            X_train_stats_kfold, y_train_kfold = X_train_stats[train[:-1000]], y_train[train[:-1000]]

            # 采用后1000条做验证集
            # X_val, y_val = X_train[-1000:], y_train[-1000:]
            # X_train, y_train = X_train[:-1000], y_train[:-1000]
            ### Decision Tree ###
            ### xgboost ###
            y_test, y_pred = xgboost_model(X_train_stats_kfold, y_train_kfold, X_test_stats_kfold, y_test_kfold, max_depth, min_child_weight)
            print(y_test, y_pred)
            _precision, _recall, _f1_score, _acc, _TNR = getAccuracy(y_pred, y_test)
            print('precision:', _precision, 'recall', _recall, 'f1_score', _f1_score, 'accuracy', _acc, 'TNR', _TNR)
            kfold_precision.append(_precision)
            kfold_recall.append(_recall)
            kfold_f1_score.append(_f1_score)
            kfold_acc.append(_acc)
            kfold_TNR.append(_TNR)
            fold_counter += 1

        print('10 k average evaluation is:', 'precision:', np.mean(kfold_precision), 'recall', np.mean(kfold_recall), 'f1_score', np.mean(kfold_f1_score), 'accuracy', np.mean(kfold_acc), 'TNR', np.mean(kfold_TNR))

        evaluation_value = str(no_good_flaw_type) + ' 10 k average evaluation is: ' + ' precision: ' + str(np.mean(kfold_precision)) + ' recall ' + str(np.mean(kfold_recall)) + ' f1_score ' + str(np.mean(kfold_f1_score)) + ' accuracy ' + str(np.mean(kfold_acc)) + ' TNR ' + str(np.mean(kfold_TNR))
        flaw_evaluation.append(evaluation_value)

    for item in flaw_evaluation:
        print(item)
