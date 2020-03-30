# coding='UTF-8'
import os
import pandas as pd
import csv


if __name__ == '__main__':
    # 为新添加的multiple label标签
    base_list = './plain_txt_list_multiple.csv'
    plain_txt_name_index_dir = './plain_txt_name_index_multiple/'
    cleaned_list = './plain_txt_name_index_multiple.csv'
    plain_name_index_txts = os.listdir(plain_txt_name_index_dir)

    # 为新的整理后的文件构建csv list
    if not os.path.exists(cleaned_list):
        with open(cleaned_list, 'a', newline='', encoding='utf-8-sig') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow(('file_name', 'article_names', 'flaw_type'))

    base_data = pd.read_csv(base_list, encoding='utf-8-sig')
    for index in range(56167, 56167+len(base_data)):
        if (str(index) + '.txt') in plain_name_index_txts:
            with open(cleaned_list, 'a', newline='', encoding='utf-8-sig') as t_file:
                csv_writer = csv.writer(t_file)
                csv_writer.writerow((index,
                                     base_data['article_names'][index-56167],
                                     base_data['flaw_type'][index-56167]
                                     ))

    print('Mission completed!')
