
# coding: utf-8

# In[3]:


import jieba
import unicodedata


def stop_words_list():
    # 停用词文件
    data_path = '../utils/stopwords.txt'

    with open(data_path, 'r', encoding='utf-8') as data:
        stopwords = [line.strip() for line in data.readlines()]

    temp_stop_list = ['\u3000', '\xa0', '\t']
    stop_words = stopwords + temp_stop_list
    return stop_words


def stock_code_dict():
    # 股票及股票代码表
    data_path = '../data/stock_list.csv'

    stock_name = []  # 提取出的股票名
    stock_code = []  # 提取出的股票代码

    with open(data_path, 'r', encoding='utf-8') as data:
        for line in data:
            stock_name.append(line[0:-9])
            stock_code.append(line[-8:-2])

    dict_code = dict(zip(stock_code, stock_name))

    return dict_code


def is_number(s):
    # 判断是否为数字
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


class WordSegmentation(object):

    def __init__(self):
        self.dict_code = stock_code_dict()
        self.stopword_list = stop_words_list()

    def word_segmentation(self, str_title_content):

        # 结巴分词词库加载股票名词
        jieba.load_userdict('../data/user_dict.txt')

        # 分词结果列表
        news_list = []

        str_content = str(str_title_content).replace('\t', '').replace('\n', '').replace('\r', '').replace(' ', '')
        str_words = ','.join(jieba.cut_for_search(str_content)).split(',')

        for word in str_words:
            if word not in self.stopword_list:
                if word[-1] != '%':
                    if is_number(word):
                        if word in self.dict_code:
                            news_list.append(word)
                    else:
                        news_list.append(word)

        return news_list

if __name__ == "__main__":
    str_title_content = "从历史数据来看，黄金和债券收益率之间的负相关关系是比较明显的，债券收益率的走高会打压金价。因此，10年期美国国债收益率的低迷对黄金将是一个支撑。另外，Spivak指出，从技术面来看，技术面动能指标表明了黄金即将迎来反弹。中质含硫原油密度粘度凝固点胶质和沥青质硫含量蜡含量析蜡点水含量"
    
    # 加载分词类
    ws = WordSegmentation()
    news_seg = ws.word_segmentation(str_title_content)
    print("分词结果：")
    print(news_seg)
    

