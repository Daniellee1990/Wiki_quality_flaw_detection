# coding='UTF-8'
import pandas as pd
import os


def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize

    return float(fsize)

if __name__ == '__main__':
    plain_txt_list = './plain_txt_list_multiple.csv'
    plain_txt_dir = './plain_txt_multiple/'
    plain_txt_name_index = './plain_txt_name_index_multiple/'
    article_list = pd.read_csv(plain_txt_list)
    # 将文章命名转化为数字编号，添加的multiple从原文档的结尾开始计数
    for index in range(56167, 56167+len(article_list)):
        file_size = get_FileSize(plain_txt_dir + article_list['article_names'][index-56167] + '.txt')
        if file_size > 1024:
            with open(plain_txt_dir + article_list['article_names'][index-56167] + '.txt', 'r', encoding='utf-8') as fs:
                content = fs.read()
                with open(plain_txt_name_index + str(index) + '.txt', 'w', encoding='utf-8') as ft:
                    ft.write(content)