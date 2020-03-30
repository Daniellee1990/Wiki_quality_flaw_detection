# coding=utf-8

from bs4 import BeautifulSoup
import os
import pandas as pd

IRRELAVANT_TAGS = ['References', 'External links', 'See also', 'Notes', 'Further reading', 'Bibliography', 'Sources', 'Footnotes', 'Contents']


def is_content_irrelavant_tag(tag_name):
    for ir_tag in IRRELAVANT_TAGS:
        if ir_tag in tag_name:
            return True
    return False


def parse_wiki_content(page_info_html):
    soup = BeautifulSoup(page_info_html, 'html.parser')
    content = soup.find('div', id='content')

    # 移除引文标记
    citation_tags = content.select('a[href^="#cite"]')
    print(len(citation_tags))
    for each in citation_tags:
        each.decompose()

    # 移除table
    [s.extract() for s in content('table')]

    tags = content.find_all(name={'h2', 'h3', 'p', 'ul'})
    plain_text = ''
    h2 = []

    # 用h2标签切割content
    for index in range(len(tags)):
        if tags[index].name == 'h2':
            h2.append(index)

    # 存在h2标签
    if len(h2) > 0:
        # 先找到section之前的概要：
        for index in range(h2[0]):
            plain_text += (tags[index].get_text())
        # 逐个添加各个section
        for index, h2_index in enumerate(h2):
            if is_content_irrelavant_tag(tags[h2_index].get_text()):
                # 如果是与正文不相干标签，pass掉该标题之后的所有内容
                continue
            elif index < len(h2)-1:
                # 如果是正文相关的标签，
                plain_text += '\n'
                for i in range(h2[index], h2[index+1]):
                    plain_text += tags[i].get_text()
                    if tags[i].name == 'h2' or tags[i].name == 'h3':
                        plain_text += '\n'
    else:
        # 不存在h2标签，直接加入
        for index in range(len(tags)):
            plain_text += (tags[index].get_text())
    return plain_text


if __name__ == '__main__':
    # 补充multiple label的数据集
    WIKI_PAGE_DIR = './data/pages/article_page/'
    PLAIN_TXT_DIR = './data/pages/plain_txt/'

    page_list = os.listdir(WIKI_PAGE_DIR)
    for page_html in page_list:
        print(page_html)
        with open(WIKI_PAGE_DIR + page_html, 'r', encoding='UTF-8') as fs:
            page_info_html = fs.read()
            _name = page_html.split('.htm')[0]
            plain_text = parse_wiki_content(page_info_html)
            with open(PLAIN_TXT_DIR + str(_name) + '.txt', 'w', encoding="utf-8") as ft:
                ft.write(plain_text)


