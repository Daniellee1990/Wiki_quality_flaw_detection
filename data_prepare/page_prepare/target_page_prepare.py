# coding='utf-8'

import pandas as pd
import random
import csv


if __name__ == '__main__':

    target_url_file = 'Article_labels.csv'  # pages with single label: 517691 multi: 64160
    target_url_data = pd.read_csv(target_url_file, encoding='utf-8-sig')

    # 'no_footnotes', 'primary_sources', 'refimprove', 'original_research', 'advert', 'notability'
    flaws = ['notability']
    # no_footnotes=[], primary_sources=[], refimprove=[], original_research=[], advert=[], notability=[]
    flaw_urls_dict = dict(notability=[])
    
    # 按照flaw list， 只要具备该flaw属性，则在该条的字典中对应flaw 列表中加入其article name
    for data_index in range(len(target_url_data)):
        for flaw in flaws:
            if target_url_data[flaw][data_index] > 0:
                flaw_urls_dict[flaw].append(target_url_data['article_name'][data_index])

    # 确定每个flaw所要采集的文章总数/数值由 original research 决定
    # sample_number_for_each_flaw = 7608
    sample_number_for_each_flaw = 10000
    for flaw_urls in flaw_urls_dict:
        random.shuffle(flaw_urls_dict[flaw_urls])
        print(flaw_urls, ' Count is: ', len(flaw_urls_dict[flaw_urls]))
        flaw_urls_dict[flaw_urls] = flaw_urls_dict[flaw_urls][0:sample_number_for_each_flaw]

    # 目标采集的article名录创建
    # target_download_article.csv
    with open('target_download_article.csv', 'a', newline='', encoding='utf-8-sig') as t_file:
        csv_writer = csv.writer(t_file)
        csv_writer.writerow(('name', 'flaw_type'))

    for flaw_type in flaw_urls_dict:
        for index in range(10000):  # 7608
            # target_download_article.csv
            with open('target_download_article.csv', 'a', newline='', encoding='utf-8-sig') as t_file:
                csv_writer = csv.writer(t_file)
                csv_writer.writerow((flaw_urls_dict[flaw_type][index], flaw_type))

    target_download_articles = pd.read_csv('target_download_article.csv', encoding='utf-8-sig')
    print('page to be downloaded, the number is: ', len(target_download_articles))
