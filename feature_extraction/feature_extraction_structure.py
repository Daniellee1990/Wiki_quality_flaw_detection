# coding='UTF-8'
from bs4 import BeautifulSoup
import time
import numpy as np
import re
import os
import pandas as pd
import csv
import math
from nltk.corpus import stopwords
from nltk import word_tokenize,Text,pos_tag
from nltk.stem import SnowballStemmer


### structure feature ###
ref_sec_names = {
            "references", "notes", "footnotes", "sources", "citations", 
            "bibliography", "works cited", "external references", "reference notes", 
            "references cited", "bibliographical references", "cited references", 
            "notes, references", "sources, references, external links", 
            "sources, references, external links, quotations", "notes & references", 
            "references & notes", "external links & references", "references & external links", 
            "references & footnotes", "footnotes & references", "citations & notes", "notes & sources", 
            "sources & notes", "notes & citations", "footnotes & citations", "citations & footnotes", 
            "reference & notes", "footnotes & sources", "note & references", "notes & reference", 
            "sources & footnotes", "notes & external links", "references & further reading", 
            "sources & references", "references & sources", "references & links", "links & references", 
            "references & bibliography", "references & resources", "bibliography & references", 
            "external articles & references", "references & citations", "citations & references", 
            "references & external link", "external link & references", "further reading & references", 
            "notes, sources & references", "sources, references & external links", "references/notes", 
            "notes/references", "notes/further reading", "references/links", "external links/references", 
            "references/external links", "references/sources", "external links / references", 
            "references / sources", "references / external links"
                }

citations = {
            "web", "book", "news", "journal"
            }

trivia_sections = {
           "facts", "miscellanea", "other facts", "other information", "trivia"
                    }

def isRefSec(headline):
    for ref_sec_name in ref_sec_names:
        if headline.find(ref_sec_name) != -1 and headline.find("resources") == -1:
            return True
    return False

def isCitation(line):
    target = line.lower()
    tp = "citation "
    for cite in citations:
        tp = tp + cite
        if target.find(tp) != -1:
            return True
    return False

def strCompare(str1, str2):
    str1 = str1.strip( ' ' )
    str2 = str2.strip( ' ' )
    if len(str1) != len(str2):
        return False
    for id in range(len(str1)):
        if str1[id] != str2[id]:
            return False
    return True

def isTriviaSec(headline):
    hdln = headline.strip('=')
    hdln = hdln.strip(' ')
    chars = list(hdln)
    length = len(chars)
    end = length - 1
    for index, char in enumerate(chars):
        cur_id = length - index - 1
        if chars[cur_id] != '\n' and chars[cur_id] != ' ' and chars[cur_id] != '=':
            end = cur_id
            break
    hdln = hdln[0:end + 1]
    for ts in trivia_sections:
        if strCompare(ts, hdln) == True:
            return True
    return False

def getCitationsCnt(line):
    citation_cnt = 0
    target = line.lower()
    for cite in citations:
        tp = "{{" + "cite " + cite
        if target.find(tp) != -1:
            citation_cnt = citation_cnt + target.count(tp)
    return citation_cnt

def get_structure_features(content):
    file_cnt = 0
    cate_cnt = 0
    heading_cnt = 0
    image_cnt = 0
    section_cnt = 0
    sub_section_cnt = 0
    infobox_cnt = 0
    list_cnt = 0
    structure_feature_per_page = list()
    hasLead = 0 # does not have lead paragraph
    ref_cnt = 0
    ref_sec_cnt = 0
    ref_per_sect = 0
    table_cnt = 0
    trivia_sec_cnt = 0
    external_links_cnt = 0
    ext_link_per_sect = 0
    title = ""
    lead_section = list()
    toDelete = False
    external_start = False
    plain_text = list()
    #first_heading = True
    for line in content.readlines():
        lead_start = False
        # 标题
        if line.find("<title>") != -1:
           nameEnd = line.find("</title>")
           nameStart = line.find("<title>") + 7
           title = line[nameStart:nameEnd]
        line = line.lower()
        if line.find("file") != -1:
           file_cnt = file_cnt + 1
           toDelete = True
        if line.find('class="category"') != -1:
           cate_cnt = cate_cnt + 1
           toDelete = True
        # 分节计数
        if line.find("mw-headline") != -1:    
           heading_cnt = heading_cnt + 1
           # 是否为引文节
           if isRefSec(line) == True:
               ref_sec_cnt = ref_sec_cnt + 1
           if isTriviaSec(line) == True:
               trivia_sec_cnt = trivia_sec_cnt + 1
           # 是否为外部链接
           if line.find("external text") != -1:
               external_start = True
        # 外部链接计数
        if line.find("external text") != -1:
            # if line.startswith('*') or line.startswith('{{'):
            external_links_cnt = external_links_cnt + 1
        if len(list(line)) <= 1:
            external_start = False
        # 图像数量计数
        if line.find("<img") != -1:
           image_cnt = image_cnt + 1
           toDelete = True
        if line.find("<table") != -1:
           table_cnt = table_cnt + 1
        if line.find("toclevel-1") != -1:
           section_cnt = section_cnt + 1
        # 如果没有目录，则直接计数
        if section_cnt == 0:
           section_cnt = heading_cnt
        if line.find("toclevel-2") != -1:
           sub_section_cnt = sub_section_cnt + 1
        if line.find("infobox") != -1:
           infobox_cnt = infobox_cnt + 1
           #toDelete = True
        lt_ref_start = False
        # 引文计数
        if line.find("cite_ref") != -1:
            ref_cnt = ref_cnt + 1
            lt_ref_start = True
            #toDelete = True
        """
        if isCitation(line) == True and lt_ref_start == False:
            ref_cnt = ref_cnt + getCitationsCnt(line)
        """
        curName = "'''" + title + "'''"
        # get lead section
        if line.find(curName) != -1 and heading_cnt == 0:
            lead_start = True
        if heading_cnt >= 1:
            lead_start = False
        if lead_start == True:
            lead_section.append(line)
        # get lists
        if line.startswith( '*' ):
            list_cnt = list_cnt + 1
            #toDelete = True
        # get the table numbers
        if line.startswith("{|"):
            table_cnt = table_cnt + 1
    if len(lead_section) != 0:
        hasLead = 1
    # 分节统计数据
    if section_cnt != 0:
        images_per_section = float(image_cnt) / section_cnt
        ref_per_sect = float(ref_cnt) / section_cnt
        ext_link_per_sect = float(external_links_cnt) / section_cnt
    else:
        images_per_section = 0

    structure_feature_per_page.append(file_cnt)
    structure_feature_per_page.append(section_cnt)
    structure_feature_per_page.append(sub_section_cnt)
    structure_feature_per_page.append(image_cnt)
    structure_feature_per_page.append(images_per_section)
    structure_feature_per_page.append(ref_cnt)
    structure_feature_per_page.append(ref_per_sect)
    structure_feature_per_page.append(table_cnt)
    structure_feature_per_page.append(external_links_cnt)
    structure_feature_per_page.append(ext_link_per_sect)

    return title, structure_feature_per_page


if __name__ == '__main__':
    # test_file_dir = './Ahmad_Aali.html'
    files_dir = './data/pages/article_page/'
    files_names = os.listdir(files_dir)
    print('total len is:', len(files_names))

    structure_feature_file_name = './feature_structure20200214.csv'

    if not os.path.exists(structure_feature_file_name):
        with open(structure_feature_file_name, 'w', newline='', encoding='utf-8-sig') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow(('file_name', 'file_cnt', 'section_cnt', 'sub_section_cnt', 'image_cnt', 'images_per_section',
                                 'ref_cnt', 'ref_per_sect', 'table_cnt', 'external_links_cnt', 'ext_link_per_sect'))      

    for index in range(len(files_names)):  # len(files_name)
        file_name = files_names[index]
        data = open(files_dir + file_name)
        title, structure_feature_per_page = get_structure_features(data)
        print(title, structure_feature_per_page)

        with open(structure_feature_file_name, 'a', newline='', encoding='utf-8-sig') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow((file_name, structure_feature_per_page[0], structure_feature_per_page[1], 
                                 structure_feature_per_page[2], structure_feature_per_page[3],
                                 structure_feature_per_page[4], structure_feature_per_page[5], 
                                 structure_feature_per_page[6], structure_feature_per_page[7], 
                                 structure_feature_per_page[8], structure_feature_per_page[9]))  

    print('Mission finished!!!')
    

