
# coding: utf-8

# In[2]:


import time

import word_segmentation

def load_future():
    """
      加载数据库里的期货关键字和种类对应字典
      dict_future({关键字：期货种类})
    """

    data_path = '../data/future_list.txt'

    dict_future = {}

    with open(data_path, 'r', encoding='utf-8') as data:
        for line in data.readlines():
            str_line = line.replace('\t', '').replace('\n', '').replace('\r', '').replace(' ', '')
            list_future_key = str_line.split('、')
            for word in list(set(str_line.split('、'))):
                if word not in dict_future:
                    dict_future[word] = list_future_key[0]
                else:
                    dict_future[word] = dict_future[word] + ',' + list_future_key[0]


    return dict_future

class ExtractFuture(object):

    def __init__(self):
        self.__dict_future = load_future()

    def extract_future(self, news_list):
        """
        读取一条新闻的标题和正文，提取其中出现的期货关键词，将期货关键词转换成期货类别，最后返回所有的期货类别（有重复）
        :param: 新闻的标题和正文分词结果（list）
        :returns：包含的所有期货类别（string：code1,code2...）
        """

        list_future = []  # 提取的期货类别列表

        for word in news_list:
            if word in self.__dict_future:
                word = self.__dict_future[word]
                list_future.append(word)
                
        str_future_list = ','.join(list_future)

        return str_future_list


# Extract_future test
if __name__ == "__main__":
    # 新推送的新闻
    str_title_content = "从历史数据来看，黄金和债券收益率之间的负相关关系是比较明显的，债券收益率的走高会打压金价。因此，10年期美国国债收益率的低迷对黄金将是一个支撑。另外，Spivak指出，从技术面来看，技术面动能指标表明了黄金即将迎来反弹。中质含硫原油密度粘度凝固点胶质和沥青质硫含量蜡含量析蜡点水含量"

    # 新闻分词
    ws = word_segmentation.WordSegmentation()
    news_list = ws.word_segmentation(str_title_content)

    print("分词结果：")
    print(news_list)

    time_start = time.time()

    # 加载ExtractStockCode类
    extract_future = ExtractFuture()

    # 提取股票代码
    str_future_list = extract_future.extract_future(news_list)

    time_elapsed = time.time() - time_start
    print('totally cost {:.0f}ms'.format(time_elapsed * 1000))

    print("期货类别：")
    print(str_future_list)

