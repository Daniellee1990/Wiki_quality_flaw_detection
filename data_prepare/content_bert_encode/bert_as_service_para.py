# -*- coding: utf-8 -*-
from bert_serving.client import BertClient
import os
import numpy as np


# 通过bert-as-service编译输入的句子列表，以列表形式返回每个句子一个768维向量
"""
开始程序前需要开启bert as servce的服务
terminal进入bert预训练模型的父级目录，输入
bert-serving-start -model_dir uncased_L-12_H-768_A-12 -num_worker=4 -max_seq_len=None
运行报错可以尝试：export PATH=$HOME/bin:/usr/local/python3/bin:/usr/local/bin:$PATH
"""

# 创建解析好的文章目录
EXTRACTED_FILE_NAME = 'articles_extracted_para.txt'

if not os.path.exists(EXTRACTED_FILE_NAME):
    with open(EXTRACTED_FILE_NAME, 'w', encoding='utf-8-sig') as t_file:
        t_file.write('article_indexs' + '\n')

with open(EXTRACTED_FILE_NAME, 'r') as f:
    lines = f.readlines()
    print(len(lines))


def encode_input_sents(input_sents):
    bc = BertClient()
    encoded_sents = bc.encode(input_sents)
    return encoded_sents


if __name__=='__main__':
    source_article_dir = './data/plain_txt_para_sep_multiple/'
    target_encoded_article_dir = './data/encoded_para_sep_multiple/'

    source_articles = os.listdir(source_article_dir)
    # source_articles = source_articles[:20000]
    # 遍历源文章，将其分别变换成按其句子划分的n * 768 向量格式
    process_counter = len(lines)
    for article in source_articles:
        if article in lines or article+'\n' in lines:
            print(article)
            continue
        with open(source_article_dir + article, 'r', encoding='utf-8-sig') as fs:
            article_content = fs.readlines()
        article_content = [line.replace('\n', ' ').replace('\xa0', ' ') for line in article_content if len(line.replace('\n', ' ').replace('\xa0', ' ').split())>3]
        if process_counter % 1000 == 0:
            print('Encoding comes to:', article, 'process comes to :', str(process_counter), '/20000')
        encoded_content = encode_input_sents(article_content)
        process_counter += 1
        with open(EXTRACTED_FILE_NAME, 'a') as f:
            f.write(article + '\n')

        
        np.savetxt(target_encoded_article_dir + article, encoded_content)
