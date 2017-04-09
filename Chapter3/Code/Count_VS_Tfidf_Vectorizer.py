#encoding:utf-8
#导入新闻数据抓取器fetch_20newsgroups,需要即时从互联网上抓取新闻
from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
#print  len(news.data)
#print news.data[0]

#分割数据得到训练数据和测试数据
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)
print  y_train.shape

#导入用于文本特征向量转化模块CountVectorizer,没有去除停用词
from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer()
X_count_train=count_vec.fit_transform(X_train)
X_count_test=count_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_count=MultinomialNB()
mnb_count.fit(X_count_train,y_train)
y_count_predict=mnb_count.predict(X_count_test)
print "The Accuracy of classifying 20newsgroup using Naive Bayes(CountVectorizer without filtering stopwords)：",mnb_count.score(X_count_test,y_test)
from  sklearn.metrics import classification_report
print classification_report(y_test,y_count_predict,target_names=news.target_names)

#导入用于文本特征向量转化模块TfidfVectorizer,没有去除停用词
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer()
X_tfidf_train=tfidf_vec.fit_transform(X_train)
X_tfidf_test=tfidf_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_tfidf=MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train)
y_tfidf_predict=mnb_tfidf.predict(X_tfidf_test)
print "The Accuracy of classifying 20newsgroup using Naive Bayes(TfidfVectorizer without filtering stopwords)：",mnb_tfidf.score(X_tfidf_test,y_test)
from  sklearn.metrics import classification_report
print classification_report(y_test,y_tfidf_predict,target_names=news.target_names)

#在训练文本量较多的时候，利用TfidfVectorizer压制这些常用词汇对分类决策的干扰，往往可以起到提升模型性能的作用

#去除停用词性能对比测试
count_filter_vec,tfidf_filter_vec=CountVectorizer(analyzer='word',stop_words='english'),TfidfVectorizer(analyzer='word',stop_words='english')

#使用带有停用词过滤的CountVectorizer对训练和测试文本分别进行量化处理
X_count_filter_train=count_filter_vec.fit_transform(X_train)
X_count_filter_test=count_filter_vec.transform(X_test)

#使用带有停用词过滤的TfidfVectorizer对训练和测试文本分别进行量化处理
X_tfidf_filter_train=tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test=tfidf_filter_vec.transform(X_test)

mnb_count_filter=MultinomialNB()
mnb_count_filter.fit(X_count_filter_train,y_train)
y_count_filter_predict=mnb_count_filter.predict(X_count_filter_test)
print "The Accuracy of classifying 20newsgroup using Naive Bayes(TfidfVectorizer by filtering stopwords)：",mnb_count_filter.score(X_count_filter_test,y_test)
from  sklearn.metrics import classification_report
print classification_report(y_test,y_count_filter_predict,target_names=news.target_names)

mnb_tfidf_filter=MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train,y_train)
y_tfidf_filter_predict=mnb_tfidf_filter.predict(X_tfidf_filter_test)
print "The Accuracy of classifying 20newsgroup using Naive Bayes(TfidfVectorizer by filtering stopwords)：",mnb_tfidf_filter.score(X_tfidf_filter_test,y_test)
from  sklearn.metrics import classification_report
print classification_report(y_test,y_tfidf_filter_predict,target_names=news.target_names)

#对停用词进行过滤的文本特征抽取方法，平均要比不过滤停用词的模型性能提升3%-4%