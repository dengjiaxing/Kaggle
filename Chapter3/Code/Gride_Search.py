#encoding:utf-8
#导入新闻数据抓取器fetch_20newsgroups,需要即时从互联网上抓取新闻
from sklearn.datasets import fetch_20newsgroups
import  numpy as np
news=fetch_20newsgroups(subset='all')
#print  len(news.data)
#print news.data[0]

#分割数据得到训练数据和测试数据
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(news.data[:3000],news.target[:3000],test_size=0.25,random_state=33)
from sklearn.svm import  SVC
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
#使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf=Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])
#这里需要验证的两个超参数的个数分别是4,3，svc_gamma的参数为10^-2,10^-1,10^0,10^1,svc_C的参数为10^-1,10^0,10^1,
#一共有12中超参数组合
parameters={'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}

from sklearn.grid_search import GridSearchCV
#将12组参数组合和初始化的Pipeline包含3折交叉验证的要求全部告知GridSearchCV，
#使用单线程，运行时间较长
#gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3)
#使用多线程，n_jobs=-1表示使用该计算机全部CPU
gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3,n_jobs=-1)
gs.fit(X_train,y_train)
print gs.best_params_,gs.best_score_
#输出最佳模型在测试集上的准确性
print gs.score(X_test,y_test)
