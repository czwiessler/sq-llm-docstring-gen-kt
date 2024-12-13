import sys

import jieba
import jieba.analyse
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from config.config import *

STOPWORDS_PATH = BASE_PATH + '/config/stopwords.txt'
USER_CORPUS = BASE_PATH + '/config/usercorpus.txt'


class HotWordsGenerator:
    def concat_all_text(self, text_dir):
        """
        read and concatenate all text content in a specific text_dir
        :param text_dir:
        :return:
        """
        all_txt = list()
        for each_txt in os.listdir(text_dir):
            filepath = text_dir + os.path.sep + each_txt
            with open(filepath, mode='rt', encoding='UTF-8') as f:
                text = ''.join(f.readlines())
                all_txt.append(text)

        return ''.join(all_txt)

    def cal_and_show_job_impression_hot_words(self, interviewee_comments_dir='../spider/impression'):
        """
        calculate and show hot words of Job Impression
        :param interviewee_comments_dir:
        :return:
        """
        if not os.path.exists(interviewee_comments_dir) or len(os.listdir(interviewee_comments_dir)) == 0:
            print('Error! No valid content in {0}'.format(interviewee_comments_dir))
            sys.exit(0)
        else:
            job_and_dir = {_: os.path.join(interviewee_comments_dir, _) for _ in os.listdir(interviewee_comments_dir)}

            for k, v in job_and_dir.items():
                text = self.concat_all_text(v)
                jieba.analyse.set_stop_words(STOPWORDS_PATH)
                jieba.load_userdict(USER_CORPUS)
                hot_words_with_weights = jieba.analyse.extract_tags(text, topK=30, withWeight=True, allowPOS=())

                frequencies = {_[0]: _[1] for _ in hot_words_with_weights}

                print(frequencies)

                x, y = np.ogrid[:300, :300]
                mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
                mask = 255 * mask.astype(int)

                wordcloud = WordCloud(font_path='./msyh.ttf', width=600, height=300, background_color="white",
                                      repeat=False,
                                      mask=mask)
                wordcloud.generate_from_frequencies(frequencies)

                import matplotlib.pyplot as plt
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()

    def cal_and_show_jd_hot_words(self, jd_dir='../spider/jd'):
        """
        calculate and show hot words of Job Description (JD)
        :param jd_dir:
        :return:
        """
        if not os.path.exists(jd_dir) or len(os.listdir(jd_dir)) == 0:
            print('Error! No valid content in {0}'.format(jd_dir))
            sys.exit(0)
        else:
            jd_and_dir = {_.split('.')[0]: os.path.join(jd_dir, _) for _ in os.listdir(jd_dir)}

            for k, v in jd_and_dir.items():
                text = "".join(pd.read_excel(v)['详情描述'])
                jieba.analyse.set_stop_words(STOPWORDS_PATH)
                jieba.load_userdict(USER_CORPUS)
                hot_words_with_weights = jieba.analyse.extract_tags(text, topK=30, withWeight=True, allowPOS=())

                frequencies = {_[0]: _[1] for _ in hot_words_with_weights}

                print(frequencies)

                x, y = np.ogrid[:300, :300]
                mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
                mask = 255 * mask.astype(int)

                wordcloud = WordCloud(font_path='./msyh.ttf', width=600, height=300, background_color="white",
                                      repeat=False,
                                      mask=mask)
                wordcloud.generate_from_frequencies(frequencies)

                import matplotlib.pyplot as plt
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()


if __name__ == '__main__':
    hot_words_generator = HotWordsGenerator()
    hot_words_generator.cal_and_show_jd_hot_words()
