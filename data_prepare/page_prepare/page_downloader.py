# coding='utf-8'

import urllib
import pandas as pd
import random
import time
import ssl
import threading
import os
import csv

# 创建下载好的文章目录
DOWNLOADED_FILE_NAME = 'articles_downloaded.csv'

if not os.path.exists(DOWNLOADED_FILE_NAME):
    with open(DOWNLOADED_FILE_NAME, 'a', newline='', encoding='utf-8-sig') as t_file:
        csv_writer = csv.writer(t_file)
        csv_writer.writerow(('article_names', 'flaw_type'))

DOWNLOADED_DATA = pd.read_csv(DOWNLOADED_FILE_NAME, encoding='utf-8-sig')
print(len(DOWNLOADED_DATA))

DOWNLOADED_NAMES = DOWNLOADED_DATA['article_names'].tolist()


def fetch_page(threadName, name, flaw_type):
    if name in DOWNLOADED_NAMES:
        print(name)
        return
    # https://xtools.wmflabs.org/articleinfo/en.wikipedia.org/7_World_Trade_Center?editorlimit=200000
    # 字符转码，避免特殊字符不能读取
    """
    # xtools_page_url:
    url = 'https://xtools.wmflabs.org/articleinfo/en.wikipedia.org/' + urllib.parse.quote(name) + '?editorlimit=2000000'
    """
    # wiki_article_url
    url = 'https://en.wikipedia.org/wiki/' + urllib.parse.quote(name)
    print(threadName, ': ', url)
    context = ssl._create_unverified_context()  # 解决certificate验证问题
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0')
    resp = urllib.request.urlopen(req, context=context)
    content = resp.read()
    with open('./pages/article_page/' + name + '.html', 'w', encoding="utf-8") as f:
        f.write(str(content, encoding="utf-8"))
    with open(DOWNLOADED_FILE_NAME, 'a', newline='', encoding='utf-8-sig') as t_file:
        csv_writer = csv.writer(t_file)
        csv_writer.writerow((name, flaw_type))
    time.sleep(0.1)


def download_pages(threadName, urls_list, flaws_type_list):
    urls_list = urls_list.tolist()
    flaws_type_list = flaws_type_list.tolist()
    for index in range(len(urls_list)):
        name = urls_list[index].replace(' ', '_')
        flaw_type = flaws_type_list[index]
        try:
            fetch_page(threadName, name, flaw_type)
        except Exception as e:
            print(threadName, e, '  ', name, 'not downloaded')


class myThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, name, urls_list, flaws_type_list):
        threading.Thread.__init__(self)
        self.name = name
        self.urls_list = urls_list
        self.flaws_type_list = flaws_type_list

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print("Starting " + self.name)
        download_pages(self.name, self.urls_list, self.flaws_type_list)
        print("Exiting " + self.name)


if __name__ == '__main__':
    target_download_articles = pd.read_csv('target_download_article.csv', encoding='utf-8-sig')

    amount_each_thread = 500  # 将目标拆成12个线程

    # 创建新线程
    thread1 = myThread("Thread-1.1", target_download_articles['article_names'][:amount_each_thread],
                       target_download_articles['flaw_type'][:amount_each_thread])
    thread2 = myThread("Thread-2.1", target_download_articles['article_names'][amount_each_thread:2*amount_each_thread],
                       target_download_articles['flaw_type'][amount_each_thread:2*amount_each_thread])
    thread3 = myThread("Thread-3.1", target_download_articles['article_names'][2*amount_each_thread:3*amount_each_thread],
                       target_download_articles['flaw_type'][2*amount_each_thread:3*amount_each_thread])
    thread4 = myThread("Thread-4.1", target_download_articles['article_names'][3*amount_each_thread:4*amount_each_thread],
                       target_download_articles['flaw_type'][3*amount_each_thread:4*amount_each_thread])
    thread5 = myThread("Thread-5.1", target_download_articles['article_names'][4*amount_each_thread:5*amount_each_thread],
                       target_download_articles['flaw_type'][4*amount_each_thread:5*amount_each_thread])
    thread6 = myThread("Thread-6.1", target_download_articles['article_names'][5*amount_each_thread:6*amount_each_thread],
                       target_download_articles['flaw_type'][5*amount_each_thread:6*amount_each_thread])

    thread7 = myThread("Thread-1.2", target_download_articles['article_names'][6*amount_each_thread:7*amount_each_thread],
                       target_download_articles['flaw_type'][6*amount_each_thread:7*amount_each_thread])
    thread8 = myThread("Thread-2.2", target_download_articles['article_names'][7*amount_each_thread:8*amount_each_thread],
                       target_download_articles['flaw_type'][7*amount_each_thread:8*amount_each_thread])
    thread9 = myThread("Thread-3.2", target_download_articles['article_names'][8*amount_each_thread:9*amount_each_thread],
                       target_download_articles['flaw_type'][8*amount_each_thread:9*amount_each_thread])
    thread10 = myThread("Thread-4.2", target_download_articles['article_names'][9*amount_each_thread:10*amount_each_thread],
                        target_download_articles['flaw_type'][9*amount_each_thread:10*amount_each_thread])
    thread11 = myThread("Thread-5.2", target_download_articles['article_names'][10*amount_each_thread:11*amount_each_thread],
                        target_download_articles['flaw_type'][10*amount_each_thread:11*amount_each_thread])
    thread12 = myThread("Thread-6.2", target_download_articles['article_names'][11*amount_each_thread:],
                        target_download_articles['flaw_type'][11*amount_each_thread:])

    # 开启线程
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()

    thread7.start()
    thread8.start()
    thread9.start()
    thread10.start()
    thread11.start()
    thread12.start()

    # 子线程结束前主线程等待
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()

    thread7.join()
    thread8.join()
    thread9.join()
    thread10.join()
    thread11.join()
    thread12.join()

    print("Exiting Main Thread")

"""
3.1
4.1
5.1
6.1

1.2
2.2
"""
