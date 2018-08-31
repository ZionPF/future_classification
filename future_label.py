
# coding: utf-8

# # 在线标记资讯关联板块
# 
# * 加载LDA模型
# * 加载RF分类模型
# * 对于资讯，用LDA变为Vector
# * 将Vector进行分类，输出前三的板块名
# 

# In[38]:



#标记数据路径
LABEL_DIR = '/data/jupyter/stock/future_classification/data/'
# LDA Model 路径：
LDA_PATH = '/data/jupyter/stock/future_classification/model/lda.model'
# Dictionary 路径
DICT_PATH = '/data/jupyter/stock/future_classification/model/dictionary.txtdic'
# Random Forest 模型路径
RF_PATH = '/data/jupyter/stock/future_classification/model/rf.model'


# stock_plate对应文件
STOCK_PLATE_PATH = '/data/jupyter/stock/data/stock_plate.csv'

# 新增新闻标记数据文件
STOCK_TRDATA_PATH = '/data/jupyter/stock/future_classification/labels/future_training_data.csv'

from sklearn.externals import joblib
#lr是一个LogisticRegression模型

import time
import word_segmentation as ws
from gensim import corpora  
from gensim.models import LdaModel  
from gensim.corpora import Dictionary  
import numpy as np  
import pandas as pd

class FutureLabel():
    def __init__(self,dict_path = DICT_PATH,lda_path = LDA_PATH, rf_path = RF_PATH ):
        self.lda = LdaModel.load(lda_path)
        self.dictionary = corpora.Dictionary.load(dict_path)
        self.rf = joblib.load(rf_path)
#         self.word_plate_data = pd.read_csv(STOCK_PLATE_PATH, dtype=str)
        
    
    # turn lda result into list
    def lda2list(self,lda,topic_n):
        lda_dict = dict(lda)
        lda_list = [0] * topic_n
        for i in range(topic_n):
            lda_list[i] = lda_dict.get(i,0)
        return lda_list
    
    def get_max_keyword(self,future_key_str):
        # 输入：关键词得到的分类列表，格式：字符串，逗号分隔
        # 输出：关键词列表中出现次数最多的分类名
        counter = {}
        for i in future_key_str.split(","):
            if i in counter:
                counter[i] += 1
            else:
                counter[i] = 1
        print(counter)
        return max(counter,key=counter.get)
    
    def future_forcast(self,news_words_list):
        news_bow = self.dictionary.doc2bow(news_words_list)      #文档转换成bow  
        news_lda = self.lda2list(self.lda[news_bow],301)
        return self.rf.predict([np.array(news_lda)])

    def future_classification(self, news_words_list, future_key_str):
        '''
        输入：
        news_words_list：新闻分词list
        future_key_str: 根据新闻中关键词对应的类别列表
    
        输出：期货新闻分类决策
        
        思路：
        1. 统计future_key中每个分类匹配次数
        2. 如果future_key能够跟forcast结果匹配，则输出forcast结果
        3. 如果future_key不能跟forcast结果匹配，则输出future_key_list中得分最高的
        '''

        future_classes = future_key_str.split(",")
        future_forcast = ''.join(self.future_forcast(news_words_list))
        
        print("提取分类为:", future_classes)
        print("预测分类为:", future_forcast)
        
        if future_forcast in future_classes:
            return str(future_forcast)
        else:
            return str(self.get_max_keyword(future_key_str))
        
    


# In[40]:




import extract_future_class
import word_segmentation

if __name__ == "__main__":

    
    str_title_content = "从历史数据来看，黄金和债券收益率之间的负相关关系是比较明显的，债券收益率的走高会打压金价。因此，10年期美国国债收益率的低迷对黄金将是一个支撑。另外，Spivak指出，从技术面来看，技术面动能指标表明了黄金即将迎来反弹。中质含硫原油密度粘度凝固点胶质和沥青质硫含量蜡含量析蜡点水含量"
    ws = word_segmentation.WordSegmentation()
    news_list = ws.word_segmentation(str_title_content)

    print("分词结果：")
    print(news_list)

    time_start = time.time()

    # 加载ExtractStockCode类
    extract_future = extract_future_class.ExtractFuture()

    # 提取股票代码
    str_future_list = extract_future.extract_future(news_list)

    time_elapsed = time.time() - time_start
    print('totally cost {:.0f}ms'.format(time_elapsed * 1000))

    print("期货类别提取：")
    print(str_future_list)
    
    print("starting classification")
    time_start = time.time()
    
    labeler = FutureLabel()
    
    label = labeler.future_classification(news_list,str_future_list)
    print('totally cost {:.0f}ms'.format(time_elapsed * 1000))
    print("期货分类预测：")
    print(label)
    
    

