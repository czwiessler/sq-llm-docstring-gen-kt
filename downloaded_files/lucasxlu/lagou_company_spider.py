import random
import time

from util import log
from config import config
import os
from spider import m_lagou_spider

import requests
from bs4 import BeautifulSoup
import pandas as pd


def crawl_company(havemark=0):
    """
    crawl company's info 
    :param havemark: 0 for not showing interviewees' remark; 1 for showing interviewees' remark; the default value is 0 
    :return: 
    """
    COMPANY_LIST = list()

    req_url = 'https://www.lagou.com/gongsi/0-0-0.json?havemark=%d' % havemark
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Host': 'www.lagou.com',
        'Origin': 'https://www.lagou.com',
        'Referer': 'https://www.lagou.com/gongsi/0-0-0?havemark=0',
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 '
                      'Mobile/13B143 Safari/601.1'
    }

    for pn in range(20):
        params = {
            'first': 'false',
            'pn': str(pn),
            'sortField': '0',
            'havemark': str(havemark)
        }

        response = requests.post(req_url, headers=headers, params=params, cookies=m_lagou_spider.init_cookies(),
                                 timeout=10)
        print(response.url)
        if response.status_code == 200:
            company_list_per_page = response.json()['result']
            for company in company_list_per_page:
                COMPANY_LIST.append([company['companyId'], company['companyShortName'],
                                     company['city'], company['companyFeatures'],
                                     company['companyFullName'], company['financeStage'], company['industryField'],
                                     company['interviewRemarkNum'], company['positionNum'], company['processRate']])
            log.info('page %d has been crawled down~' % (pn + 1))
        elif response.status_code == 403:
            log.error('403 forbidden...')
        else:
            log.error(response.status_code)
        time.sleep(random.randint(3, 6))

    return COMPANY_LIST


def crawl_company_stage(company_id):
    req_url = 'https://m.lagou.com/gongsi/%s.html' % str(company_id)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Host': 'm.lagou.com',
        'Referer': 'https://m.lagou.com',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'
    }
    response = requests.get(req_url, headers=headers, cookies=m_lagou_spider.init_cookies(), timeout=20)
    print(response.url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html5lib')
        company_desc = soup.find_all(class_="desc")[0].get_text().strip()
        industryField = company_desc.split('/')[0].strip()
        financeStage = company_desc.split('/')[1].strip()
        staffNum = company_desc.split('/')[2].strip()

    elif response.status_code == 403:
        log.error('403 forbidden...')
    else:
        log.error(response.status_code)
    time.sleep(random.randint(3, 6))

    return [company_id, industryField, financeStage, staffNum]


if __name__ == '__main__':
    company_level_list = list()
    visited_company_id_list = list()
    for job in os.listdir('./data'):
        for company_id in pd.read_excel(os.path.join('./data', job))['公司编码']:
            if not company_id in visited_company_id_list:
                try:
                    company = crawl_company_stage(company_id)
                    company_level_list.append(company)
                    visited_company_id_list.append(company_id)
                except:
                    pass
                finally:
                    cols = [u'公司编码', u'所属行业', u'融资阶段', u'员工数量']
                    df = pd.DataFrame(company_level_list, columns=cols)
                    df.to_excel('./company.xlsx', 'Company', index=False)
            else:
                log.info('%d has been visited before...' % company_id)
    log.info('Processing done!')
