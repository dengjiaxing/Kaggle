#encoding:utf-8
import  pandas as pd
import numpy as np

#读取数据，并初步处理数据
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Shape',
              'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli',
              'Mitoses','Class']
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data=data.replace(to_replace='?',value=np.nan) #将？替换为标准缺失值表示
print data.shape
data=data.dropna(how='any')  #删除带有缺失值的数据（只要有一个维度有缺失）
print data.shape

#分割数据得到训练数据和测试数据
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

print y_train.value_counts()  #查看训练样本的数量和类别分布
print y_test.value_counts()    #查看测试样本的数量和类别分布

#使用分类模型从事良/恶性肿瘤预测任务
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression;  #逻辑斯蒂回归
from sklearn.linear_model import SGDClassifier;   #随机梯度下降
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
#初始化
lr=LogisticRegression()
sgdc=SGDClassifier()
#进行训练和测试
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)

from sklearn.metrics import classification_report
#获得准确性结果
print  'Accuracy of LR Classifier:',lr.score(X_test,y_test)
#获得准确率，召回率，F1调和平均数
print classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])

print  'Accuracy of SGDC Classifier:',lr.score(X_test,y_test)
#获得准确率，召回率，F1调和平均数
print classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])