# coding='UTF-8'
from bs4 import BeautifulSoup
import time
import numpy as np
import re
# import pymysql as MySQLdb
import os
import pandas as pd
import csv

"""
MySQLdb.converters.encoders[np.float64] = MySQLdb.converters.escape_float
MySQLdb.converters.conversions = MySQLdb.converters.encoders.copy()
MySQLdb.converters.conversions.update(MySQLdb.converters.decoders)
"""

EDIT_HISTORY_FEATURE_FILE_NAME = './feature_edit_history20200110.csv'


def fetch_page_info(page_info_html):
    soup = BeautifulSoup(page_info_html, 'html.parser')
    page_creation_date = soup.find('tr', id='mw-pageinfo-firsttime').find_all('td')[-1].text
    return page_creation_date


def fetch_page_history(page_history_html):
    soup = BeautifulSoup(page_history_html, 'html.parser')
    table1_tds = soup.find('div', class_='col-lg-5 col-lg-offset-1 stat-list clearfix').find_all('td')
    page_id = ''
    wikidata_id = ''
    total_edits_count = ''
    editors_count = ''
    for index, td in enumerate(table1_tds):
        if td.text.strip() == 'ID':
            page_id = table1_tds[index+1].find('a').text.strip()
        if td.text.strip() == 'Wikidata ID':
            wikidata_id = table1_tds[index+1].find('a').text.strip()
        if td.text.strip() == 'Total edits':
            total_edits_count = table1_tds[index+1].text.strip()
        if td.text.strip() == 'Editors':
            editors_count = table1_tds[index+1].text.strip()

    table2_tds = soup.find('div', class_='col-lg-6 stat-list clearfix').find_all('td')
    first_edit = table2_tds[11].find('a').text.strip()
    s = len(soup.find_all('td', class_='sort-entry--month'))
    last3_mounth_edit_count = ['0', '0', '0']
    if s >= 3:
        last3_mounth_edit_count = [soup.find_all('td', class_='sort-entry--month')[-1].find_next_sibling().text.strip(), soup.find_all('td', class_='sort-entry--month')[-2].find_next_sibling().text.strip(), soup.find_all('td', class_='sort-entry--month')[-3].find_next_sibling().text.strip()]
    else:
        for i in range(0, s):
            last3_mounth_edit_count[i] = soup.find_all('td', class_='sort-entry--month')[-i].find_next_sibling().text.strip()

    reviewer_tb = soup.find('table', class_='table table-bordered table-hover table-striped top-editors-table')
    user_names =[reviewer_td.text.strip() for reviewer_td in reviewer_tb.find_all('td', class_='sort-entry--username')]
    user_edits = [reviewer_td.text.strip() for reviewer_td in reviewer_tb.find_all('td', class_='sort-entry--edits')]
    return page_id, wikidata_id, total_edits_count, editors_count, first_edit, last3_mounth_edit_count, user_names, user_edits


def fetch_discussion_count(page_talk_html):
    soup = BeautifulSoup(page_talk_html, 'html.parser')
    discussions = soup.find_all('a', class_='new')
    return len(discussions)


def fetch_review_log(review_log_html):
    soup = BeautifulSoup(review_log_html, 'html.parser')
    review_log_list = soup.find_all('li', class_='mw-logline-review')
    for li in review_log_list:
        print(li.text.strip())


def convert2exist_days(now, creation_time):
    t1 = time.mktime(time.strptime(creation_time, "%Y-%m-%d %H:%M"))
    t2 = time.mktime(time.strptime(now, "%Y-%m-%d %H:%M"))
    return int((t2 - t1)/(60*60*24))


def parse_features(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        page_info_html = f.read()
        page_id, wikidata_id, total_edits_count, editors_count, first_edit, last3_mounth_edit_count, user_names, user_edits = fetch_page_history(page_info_html)
        # print(page_id, wikidata_id, total_edits_count, editors_count, first_edit, last3_mounth_edit_count, user_names, user_edits)
    '''
    with open('./page_demo/7_World_Trade_Center_talkpage.html', 'r', encoding='UTF-8') as f:
        html = f.read()
        discussion_count = fetch_discussion_count(html)
    '''
    page_id = int(page_id.replace(',', ''))
    total_edits_count = int(total_edits_count.replace(',', ''))
    editors_count = int(editors_count.replace(',', ''))
    user_edits = [int(e.replace(',', '')) for e in user_edits]
    last3_mounth_edit_count = [int(e.replace(',', '')) for e in last3_mounth_edit_count]

    # The edit history features
    edit_history_feature = {}

    edit_history_feature['page_id'] = page_id
    edit_history_feature['wikidata_id'] = wikidata_id

    # Age of the article
    edit_history_feature['age'] = convert2exist_days('2019-11-15 23:59', first_edit)
    # Mean time between two reviews over the past 30 days
    if last3_mounth_edit_count[0] != 0:
        edit_history_feature['age_per_review_30_days'] = 30/last3_mounth_edit_count[0]
    else:
        edit_history_feature['age_per_review_30_days'] = -1
    # Ratio between the total number of edits and the number of users
    edit_history_feature['avg_edit_per_usr'] = total_edits_count/editors_count
    # Total number of discussions by the user for an article
    edit_history_feature['discussion_count'] = -1
    # Total number of anonymous users represented by Internet protocol (IP) who make a comment/modification or review an article
    edit_history_feature['ip_count'] = len([user_name for user_name in user_names if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", user_name)])
    # Number of reviews made by registered and anonymous users
    edit_history_feature['review_count'] = total_edits_count
    # Number of registered users
    edit_history_feature['user_count'] = editors_count
    # TODO Ratio between the number of modified lines and the total number of lines
    edit_history_feature['Modified_lines_rate'] = None
    # Percentage of reviews made by users which edited the article less than four times.
    edit_history_feature['occasional_usr_rvw_rate'] = sum([review_count for review_count in user_edits if review_count < 4])/total_edits_count
    # Ratio between the number of reviews made during the past 3 months and the total number of reviews
    edit_history_feature['rev_porc_Last3Months'] = sum(last3_mounth_edit_count)/total_edits_count
    # Percentage of reviews made by the most active users (that is, users in the top 5%)
    edit_history_feature['Most_active_users_rvw_rate'] = sum(user_edits[0:int(len(user_edits)*0.05)])/total_edits_count
    # TODO Quality of the review, which is based on the quality of the reviewers
    edit_history_feature['prob_review'] = None
    # Standard deviation of the number of edits made by a user
    edit_history_feature['std_dev_edit_per_usr'] = np.std(user_edits)
    # Ratio between the number of reviews and the number of days an article has existed
    if edit_history_feature['age'] > 0:
        edit_history_feature['Reviews_per_day'] = total_edits_count/edit_history_feature['age']
    else:
        edit_history_feature['Reviews_per_day'] = 0

    return edit_history_feature


def save_feature_2_db(edit_history_feature, filepath, main_category, sub_category):
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='lmy718..',
        db='wiki',
        use_unicode=True,
        charset='utf8'
    )
    sqli = "INSERT INTO featured_samples(page_id, wikidata_id, age,age_per_review_30_days,avg_edit_per_usr,discussion_count,ip_count,review_count,user_count, occasional_usr_rvw_rate,rev_porc_Last3Months,Most_active_users_rvw_rate, std_dev_edit_per_usr,Reviews_per_day,filepath,main_category,sub_category) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    cur = conn.cursor()
    cur.execute(sqli, (edit_history_feature['page_id'], edit_history_feature['wikidata_id'], edit_history_feature['age'], edit_history_feature['age_per_review_30_days'], edit_history_feature['avg_edit_per_usr'], edit_history_feature['discussion_count'], edit_history_feature['ip_count'], edit_history_feature['review_count'], edit_history_feature['user_count'], edit_history_feature['occasional_usr_rvw_rate'], edit_history_feature['rev_porc_Last3Months'], edit_history_feature['Most_active_users_rvw_rate'], edit_history_feature['std_dev_edit_per_usr'], edit_history_feature['Reviews_per_day'], filepath, main_category, sub_category))
    conn.commit()


if __name__ == '__main__':
    """
    with open('./history_feature/analyzed', 'r') as f:
        lines = f.readlines()
        lines = [i.split('\n')[0] for i in lines]
        print('pages already has been parsed, count is:', len(lines))
    """
    if not os.path.exists(EDIT_HISTORY_FEATURE_FILE_NAME):
        with open(EDIT_HISTORY_FEATURE_FILE_NAME, 'w', newline='', encoding='utf-8-sig') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow(('page_id', 'wikidata_id', 'age', 'age_per_review_30_days', 'avg_edit_per_usr',
                                 'discussion_count', 'ip_count', 'review_count', 'user_count', 'occasional_usr_rvw_rate',
                                 'rev_porc_Last3Months', 'Most_active_users_rvw_rate', 'std_dev_edit_per_usr',
                                 'Reviews_per_day', 'filepath'))

    target_download_data = pd.read_csv('./data/pages/plain_txt_name_index_with_labels.csv', encoding= 'utf-8-sig')
    print('Total count of pages is :', len(target_download_data))

    add_count = 0
    total_count = 0
    exist_count = 0

    # 按照目标list遍历取出特征集合
    for index in range(len(target_download_data)):
        name = target_download_data['article_names'][index]
        total_count += 1
        print('article to be parse:', name)
        """
        if name in lines or name + '\n' in lines:
            exist_count += 1
            print(name + ' has been parsed!')
            continue
        """
        file_path = './data/pages/xtools_add_notability/' + name + '.html'

        if os.path.exists(file_path):
            try:
                print(file_path)
                edit_history_feature = parse_features(file_path)
                page_id = edit_history_feature['page_id']
                add_count += 1
                # print(page_id)
                # save_feature_2_db(edit_history_feature, file_path, main_category, sub_category)
                with open(EDIT_HISTORY_FEATURE_FILE_NAME, 'a', newline='', encoding = 'utf-8-sig') as t_file:
                    csv_writer = csv.writer(t_file)
                    csv_writer.writerow((edit_history_feature['page_id'], edit_history_feature['wikidata_id'],
                                         edit_history_feature['age'], edit_history_feature['age_per_review_30_days'],
                                         edit_history_feature['avg_edit_per_usr'], edit_history_feature['discussion_count'],
                                         edit_history_feature['ip_count'], edit_history_feature['review_count'],
                                         edit_history_feature['user_count'],
                                         edit_history_feature['occasional_usr_rvw_rate'],
                                         edit_history_feature['rev_porc_Last3Months'],
                                         edit_history_feature['Most_active_users_rvw_rate'],
                                         edit_history_feature['std_dev_edit_per_usr'],
                                         edit_history_feature['Reviews_per_day'], file_path))
                """
                with open('./history_feature/analyzed', 'a') as f:
                    f.write(name + '\n')
                """
            except Exception as e:
                print('page not parsed:', name, 'reason is: ', e)
    print('pages_newly_parsed: ', add_count, 'pages_already_parsed: ', exist_count, 'total_pages: ', total_count)

