import random
import sys
import time

import requests
from pymongo import MongoClient
import pandas as pd

sys.path.append('../')
from analysis.sentiment import cal_sentiment
from spider.m_lagou_spider import init_cookies
from util import log

client = MongoClient()


def crawl_interviewee_comments(company_id):
    request_url = 'https://www.lagou.com/gongsi/searchInterviewExperiences.json'
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Host': 'www.lagou.com',
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0'
                      ' Mobile/13B143 Safari/601.1',
        'Referer': 'https://www.lagou.com/gongsi/interviewExperiences.html?companyId=%s' % company_id
    }
    maxpage_no = get_max_page_no(company_id)

    if maxpage_no > 0:
        for pn in range(maxpage_no):
            params = {
                'companyId': company_id,
                'positionType': '',
                'pageSize': '10',
                'pageNo': str(pn + 1)
            }

            response = requests.post(request_url, headers=headers, params=params, cookies=init_cookies())
            log.info('Crawl page %s successfully~' % response.url)
            if response.status_code == 200:
                comment_list = response.json()['content']['data']['page']['result']
                for comment in comment_list:
                    insert_item(comment)
                    log.info('insert one item successfully~')
                    """
                    intervieweeComment = IntervieweeComment()
                    intervieweeComment.id = comment['id']
                    intervieweeComment.companyId = comment['companyId']
                    intervieweeComment.companyName = comment['companyName']
                    intervieweeComment.companyScore = comment['companyScore']
                    intervieweeComment.comprehensiveScore = comment['comprehensiveScore']
                    intervieweeComment.interviewerScore = comment['interviewerScore']
                    intervieweeComment.describeScore = comment['describeScore']
                    intervieweeComment.myScore = comment['myScore']
                    intervieweeComment.content = comment['content']
                    intervieweeComment.createTime = comment['createTime']
                    intervieweeComment.hrId = comment['hrId']
                    intervieweeComment.positionId = comment['positionId']
                    intervieweeComment.positionName = comment['positionName']
                    intervieweeComment.positionStatus = comment['positionStatus']
                    intervieweeComment.positionType = comment['positionType']
                    intervieweeComment.tagArray = comment['tagArray']
                    intervieweeComment.usefulCount = comment['usefulCount']
    
                    insert_item(intervieweeComment)
                    """
            else:
                log.error('Error code is ' + str(response.status_code))

            time.sleep(random.randint(3, 6))


def get_max_page_no(company_id):
    """
    return the max page number of interviewees' comments based on particular company 
    :param company_id: 
    :return: 
    """
    request_url = 'https://www.lagou.com/gongsi/searchInterviewExperiences.json'
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Host': 'www.lagou.com',
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0'
                      ' Mobile/13B143 Safari/601.1',
        'Referer': 'https://www.lagou.com/gongsi/interviewExperiences.html?companyId=%s' % company_id
    }

    params = {
        'companyId': company_id,
        'positionType': '',
        'pageSize': '10',
        'pageNo': '1'
    }

    response = requests.post(request_url, headers=headers, params=params, cookies=init_cookies())
    if response.status_code == 200:
        maxpage = int(response.json()['content']['data']['page']['totalCount'])
    else:
        log.error('Error code is ' + str(response.status_code))
        maxpage = 0

    return int(maxpage / 10) + 1


def insert_item(item):
    db = client.lagou.intervieweeComment
    result = db.insert_one(item)


def query_document(query_dict):
    db = client.lagou.intervieweeComment
    cursor = db.find(query_dict)
    document_list = list()
    for document in cursor:
        document_list.append(document)

    return document_list


def update_item():
    db = client.lagou.intervieweeComment
    query_dict = {}
    for document in query_document(query_dict):
        sentiment_value = cal_sentiment(document['content'])
        db.update_many(
            {"orderId": document['orderId']},
            {"$set": {"sentiment": sentiment_value}}
        )
        print('insert sentiment value %f successfully~' % sentiment_value)


if __name__ == '__main__':
    df = pd.read_excel('./data/company.xlsx')
    for _ in df['公司编码']:
        crawl_interviewee_comments(_)
    log.info('All interviewee comments have been stored in mongodb...')
