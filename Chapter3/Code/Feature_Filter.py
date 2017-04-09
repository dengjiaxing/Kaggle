#encoding:utf-8
import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print  titanic.head
#print  titanic.info()
#分离数据特征与预测目标
X=titanic.drop(['row.names','name','survived'],axis=1)
y=titanic[['survived']]

#对缺失数据进行填充
X['age'].fillna(X['age'].mean(),inplace=True)
X.fillna('UNKNOW',inplace=True)
#print X.info()
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#类别向量特征化
from sklearn.feature_extraction import  DictVectorizer
vec=DictVectorizer()
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

print len(vec.feature_names_)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
print dt.score(X_test,y_test)

# #导入特征选择器
from sklearn import feature_selection
# #选择前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能
# fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
# X_train_fs=fs.fit_transform(X_train,y_train)
# dt.fit(X_train_fs,y_train)
# X_test_fs=fs.transform(X_test)
# print  dt.score(X_test_fs,y_test)

#通过交叉验证方法，按照固定间隔的百分比选择特征，并作图展示性能随待选比例的变化
from  sklearn.cross_validation import cross_val_score
import numpy as np
percentitles=range(1,100,2)
results=[]
for i in percentitles:
    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_fs=fs.fit_transform(X_train,y_train)
    scores=cross_val_score(dt, X_train_fs, y_train, cv=5)
    results=np.append(results,scores.mean())
print results
