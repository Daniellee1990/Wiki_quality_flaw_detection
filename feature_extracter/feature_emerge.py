# coding='UTF-8'
from bs4 import BeautifulSoup
import time
import numpy as np
import re
import os
import pandas as pd
import csv

def reshape_feature_structure(feature_structure, file_name_index):
    target_indexes = []
    article_names = file_name_index['article_names'].tolist()
    for index in range(len(feature_structure)):
        # 将命名全部换为标准的index命名
        file_name = str(feature_structure['file_name'][index])[:-5]
        if file_name in article_names:
            target_indexes.append(index)
            
            tmp_index = article_names.index(file_name)
            index_name = str(file_name_index['file_name'][tmp_index]) + '.txt'
            feature_structure.loc[feature_structure['file_name']==feature_structure['file_name'][index], 'file_name'] = index_name

    reshaped_feature_structure = feature_structure.iloc[target_indexes, :]

    return reshaped_feature_structure


def reshape_feature_edit_history(feature_edit_history, file_name_index):
    target_indexes = []
    article_names = file_name_index['article_names'].tolist()
    for index in range(len(feature_edit_history)):
        # 将命名全部换为标准的index命名
        file_name = str(feature_edit_history['file_name'][index])[9:-5]
        if file_name in article_names:
            target_indexes.append(index)
            
            tmp_index = article_names.index(file_name)
            index_name = str(file_name_index['file_name'][tmp_index]) + '.txt'
            feature_edit_history.loc[feature_edit_history['file_name']==feature_edit_history['file_name'][index], 'file_name'] = index_name

    reshaped_feature_edit_history = feature_edit_history.iloc[target_indexes, :]

    return reshaped_feature_edit_history


def merge_features():
    feature_file_dir_text_stats = './feature_text_stats20200214.csv'
    feature_file_dir_structure = './feature_structure20200214.csv'
    feature_file_dir_writing_style = './feature_writing_style20200215.csv'
    feature_file_dir_readability = './feature_readbility20200214.csv'
    feature_file_dir_edit_history = './feature_edit_history20200110.csv'

    file_name_index_dir = './data/pages/plain_txt_name_index_with_labels.csv'

    feature_text_stats = pd.read_csv(feature_file_dir_text_stats, encoding='utf-8-sig')
    feature_structure = pd.read_csv(feature_file_dir_structure, encoding='utf-8-sig')  # 命名需要调整
    feature_writing_style = pd.read_csv(feature_file_dir_writing_style, encoding='utf-8-sig')
    feature_readability = pd.read_csv(feature_file_dir_readability, encoding='utf-8-sig')
    feature_edit_history = pd.read_csv(feature_file_dir_edit_history, encoding='utf-8-sig')

    file_name_index = pd.read_csv(file_name_index_dir, encoding='utf-8-sig')
    file_name_index = file_name_index.loc[:, ['article_names', 'file_name']]

    print('feature_text_stats', len(feature_text_stats))
    print('feature_structure', len(feature_structure))
    print('feature_writing_style', len(feature_writing_style))
    print('feature_readability', len(feature_readability))
    print('feature_edit_history', len(feature_edit_history))

    print(feature_text_stats['file_name'][0], feature_structure['file_name'][0], feature_writing_style['file_name'][0], feature_readability['file_name'][0])

    reshaped_feature_structure = reshape_feature_structure(feature_structure, file_name_index)
    reshaped_feature_text_edit_history = reshape_feature_edit_history(feature_edit_history, file_name_index)
    """
    for key in reshaped_feature_text_edit_history:
        print(reshaped_feature_text_edit_history[key][0])

    print(len(reshaped_feature_structure), len(reshaped_feature_text_edit_history))
    """
    
    to_merge_datas = [feature_text_stats, reshaped_feature_structure, feature_writing_style, feature_readability, reshaped_feature_text_edit_history]
    
    merged_file = to_merge_datas[0]
    
    for file in to_merge_datas[1:]:
        merged_file = pd.merge(merged_file, file, on=['file_name'], how='outer')

    merged_file.to_csv('./feature_stats_merged.csv')


def clean_labels(labels):
    # 去除labels 表中重复的项
    target_indexes = []
    file_counter = {'total_count' : 0}
    for index in range(len(labels)):
        # 将命名全部换为标准的index命名
        if labels['file_name'][index] in file_counter:
            file_counter[labels['file_name'][index]] += 1
        if labels['file_name'][index] not in file_counter:
            file_counter[labels['file_name'][index]] = 1
        if file_counter[labels['file_name'][index]] == 1:
            target_indexes.append(index)

    cleaned_labels = labels.iloc[target_indexes, :]

    return cleaned_labels


def merge_features_labels():
    label_dir = './data/pages/plain_txt_name_index_with_labels.csv'
    features_dir = './feature_stats_merged_cleaned20200219.csv'
    labels = pd.read_csv(label_dir, encoding='utf-8-sig')
    features = pd.read_csv(features_dir, encoding='utf-8-sig')

    print(len(labels), len(features))

    cleaned_labels = clean_labels(labels)
    cleaned_labels["file_name_with_suffix"] =[ '%i.txt' % i for i in cleaned_labels["file_name"]]

    merged_file = pd.merge(cleaned_labels, features, left_on=['file_name_with_suffix'], right_on=['file_name'], how='outer')
    merged_file.drop(['discussion_count', 'Unnamed: 0_x', 'Unnamed: 0_y', 'article_names', 'file_name_y'], axis=1, inplace=True)
    merged_file.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    merged_file.rename(columns={'file_name_x':'file_name'},inplace=True) 

    merged_file.to_csv('./labels_feature_stats_merged_cleaned20200223.csv')

if __name__ == '__main__':
    # 融合特征文件
    # merge_features()
    """
    final_merged_data = pd.read_csv('./feature_stats_merged.csv', encoding='utf-8-sig')
    final_merged_data.drop(['page_id', 'wikidata_id', 'Unnamed: 0'], axis=1, inplace=True)
    final_merged_data.eval('section_cnt_include_abstract = section_cnt + 1' , inplace=True)
    final_merged_data.eval('subsection_per_section_avg = sub_section_cnt / section_cnt_include_abstract' , inplace=True)
    final_merged_data.eval('citation_count_per_text_length = ref_cnt / word_cnt' , inplace=True)
    final_merged_data.eval('links_per_text_length = external_links_cnt / word_cnt' , inplace=True)
    counter = 0
    for key in final_merged_data:
        counter += 1
        print(key)
    print(len(final_merged_data), counter)
    final_merged_data.to_csv('./feature_stats_merged_cleaned20200219.csv')
    """
    merge_features_labels()


