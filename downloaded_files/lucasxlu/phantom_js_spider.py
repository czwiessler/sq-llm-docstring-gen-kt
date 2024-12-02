import requests
import platform
import os
import math
import sys
import zipfile

from tqdm import tqdm
from selenium import webdriver
from phantomjs_bin import executable_path

sys.path.append('../')
from util.excel_helper import mkdirs_if_not_exists


def download_phantom_js():
    download_url_map = {
        'Windows': 'https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-windows.zip',
        'Linux': 'https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-i686.tar.bz2'
    }

    if platform.system() == 'Windows':
        if not os.path.exists(
                './PhantomJS/%s/bin/phantomjs.exe' % download_url_map[platform.system()].split('/')[-1]
                        .replace('.zip', '')):
            print('Downloading PhantomJS, please wait...')
            mkdirs_if_not_exists('./PhantomJS/')
            response = requests.get(download_url_map[platform.system()], stream=True, timeout=100)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            wrote = 0

            with open('./PhantomJS/%s' % download_url_map[platform.system()].split('/')[-1], mode='wb') as f:
                for data in tqdm(response.iter_content(block_size), total=math.ceil(total_size // block_size),
                                 unit='KB', unit_scale=True):
                    wrote = wrote + len(data)
                    f.write(data)

            if total_size != 0 and wrote != total_size:
                print("ERROR, something went wrong")

            zip_ref = zipfile.ZipFile('./PhantomJS/%s' % download_url_map[platform.system()].split('/')[-1], 'r')
            zip_ref.extractall('./PhantomJS')
            zip_ref.close()
        else:
            print('PhantomJS exists.')
            driver = webdriver.PhantomJS(executable_path=executable_path)


if __name__ == '__main__':
    download_phantom_js()
