# coding='UTF-8'
import io
import sys
import os
import re
import pandas as pd
import nltk
import nltk.data

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')  #解决print输出报编码错问题
 
def splitParas(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    paras = tokenizer.tokenize(paragraph)
    paras = [item + '\n' for item in paras]
    return sentences


def splitSections(content):
    sections = content.split('\n\n')
    sections = [section.replace('\n', '') + '\n' for section in sections]
    return sections


def splitByLen512(content):
    content = content.replace('\n', '')
    content_list = content.split()
    divide_counter = 1
    len512_contents = []
    if len(content_list) > 512:
        len512_contents.append(' '.join(content_list[:512]))
        while len(content_list) > (divide_counter + 1) * 512:
            len512_contents.append(' '.join(content_list[divide_counter * 512:(divide_counter + 1) * 512]))
            divide_counter += 1
        len512_contents.append(' '.join(content_list[(divide_counter - 1) * 512:]))
    else:
        len512_contents.append(' '.join(content_list))
    len512_contents = [item + '\n' for item in len512_contents]
    return len512_contents



if __name__ == '__main__':
	# 添加对multiple 的段落切分
    plain_txt_dir = './plain_txt_name_index_multiple/'
    plain_txt_sent_sep_dir = './plain_txt_para_sep_multiple/'
    plain_txts = os.listdir(plain_txt_dir)

    # 取得全部文章的content, 并将其解析成为分句后的list
    for file in plain_txts:
        with open(plain_txt_dir + file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = ''.join(splitParas(content))
        with open(plain_txt_sent_sep_dir + file, 'w', encoding='utf-8') as f:
            f.write(content)

    print('Mission completed!')
