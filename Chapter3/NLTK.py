#encoding:utf-8
#导入新闻数据抓取器fetch_20newsgroups,需要即时从互联网上抓取新闻
from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
X,y=news.data,news.target
from  bs4 import  BeautifulSoup
import  nltk
import re
#定义一个函数名为news_to_sentence将每条新闻中的句子逐一剥离出来，并返回一个句子列表
def news_to_sentence(news):
    news_text=BeautifulSoup(news).get_text()
    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences=tokenizer.tokenize(news_text)
    sentences=[]
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]','',sent.lower().strip()).split()) #将不是字母的字符都用‘’替换
    return sentences
sentences=[]
for x in X:
    sentences+=news_to_sentence(x)
from gensim.models import word2vec

#配置词向量的维度
num_feature=30
#保证被考虑的词汇的频度
min_word_count=20
#设定并行化训练使用CPU计算核心的数量，多核多用
num_workers=2
#定义训练词向量的上下文窗口大小
context=5
downsampling=3
#训练词向量模型
model=word2vec(sentences,wokers=num_workers,size=num_feature,min_count=min_word_count,windows=context,sample=downsampling)

model.init_sims(replace=True)
model.most_similar('morning')
