import pandas as pd
import requests
from bs4 import BeautifulSoup


def crawl_proxies():
    """
    collect Proxy
    :return:
    """
    prox_list = []
    for i in range(1, 20):
        req_url = 'https://www.xicidaili.com/nn/%d' % i
        response = requests.get(req_url, headers={
            'Host': 'www.xicidaili.com',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36',
            'Cache-Control': 'no-cache',
        })
        if response.status_code == 200:
            bs = BeautifulSoup(response.text, 'html5lib')
            table = bs.find_all('table', id="ip_list")[0]
            for tr in table.find_all('tr', class_='odd'):
                tds = tr.find_all('td')
                ip = tds[1].text
                port = tds[2].text
                anonymous_ext = tds[4].text
                http_type = tds[5].text
                delay = float(tds[6].div['title'].replace('秒', ''))
                survival = tds[8].text
                validation_date = tds[9].text

                print([ip, port, anonymous_ext, http_type, delay, survival, validation_date])
                prox_list.append([ip, port, anonymous_ext, http_type, delay, survival, validation_date])
        else:
            print('Error occurs during visiting Lagou. The ERROR_CODE is {0}. Return: {1}'.format(
                response.status_code, response.text))

        col = [
            'IP',
            'Port',
            'Anonymous',
            'HTTP',
            'Delay',
            'Survival',
            'ValidationDate'
        ]
        df = pd.DataFrame(prox_list, columns=col)
        df.to_excel("./Proxy.xlsx", sheet_name='Proxy', index=False)


if __name__ == '__main__':
    crawl_proxies()
