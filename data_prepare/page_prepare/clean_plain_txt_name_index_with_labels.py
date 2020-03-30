# coding='UTF-8'
import os
import pandas as pd


if __name__ == '__main__':
    # 为新添加的multiple label标签
    base_list = './plain_txt_name_index_with_labels.csv'
    encoded_files_dir = './encoded_para_sep/'
    cleaned_list = './cleaned_plain_txt_name_index_with_labels.csv'
    encoded_index_txts = os.listdir(encoded_files_dir)

    # 去除重复项 45719
    base_data = pd.read_csv(base_list)
    dictionary = []
    indexs = []
    for index in range(len(base_data)):
        if base_data['article_name'][index] not in dictionary:
            dictionary.append(base_data['article_name'][index])
            indexs.append(index)
        else:
            print(base_data['article_name'][index])

    cleaned_base_data = base_data.iloc[indexs,:].reset_index(drop=True)
    cleaned_base_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(len(cleaned_base_data))
    print(cleaned_base_data.head())
    
    # cleaned_base_data.to_csv(cleaned_list)
    tst = pd.read_csv(cleaned_list)
    for key in tst:
        print(key)

    print('Mission completed!')
