#encoding:utf-8
#导入新闻数据抓取器fetch_20newsgroups,需要即时从互联网上抓取新闻
from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
print  len(news.data)
print news.data[0]

#分割数据得到训练数据和测试数据
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)
print  y_train.shape

#导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)
print "The Accuracy of Bayes Classification is",mnb.score(X_test,y_test)
from  sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=news.target_names)