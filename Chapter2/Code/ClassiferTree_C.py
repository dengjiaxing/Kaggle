#encoding:utf-8
import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print  titanic.head
#print  titanic.info()
X=titanic[['pclass','age','sex']]
y=titanic[['survived']]
#print X.info()
#X中age只有633条记录,需要补充，sex和pclass是类别型的，用0/1代替
#首先使用平均数或者中位数补充age里面的值
X['age'].fillna(X['age'].mean(),inplace=True)
#分割数据得到训练数据和测试数据
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#引入特征转换器
from sklearn.feature_extraction import  DictVectorizer
vec=DictVectorizer(sparse=False)
#转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
#print  vec.feature_names_
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

#使用单一决策树进行模型训练和预测分析
from  sklearn.tree import  DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict=dtc.predict(X_test)

#使用随机森林进行集成模型训练和预测分析
from sklearn.ensemble import  RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)

#使用梯度提升决策树进行集成模型训练和预测分析
from  sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict=gbc.predict(X_test)


from  sklearn.metrics import classification_report
print  "the accuracy of decision tree is",dtc.score(X_test,y_test)
print classification_report(dtc_y_predict,y_test,target_names=['died','survived'])

print  "the accuracy of random forest classifier  is",dtc.score(X_test,y_test)
print classification_report(rfc_y_predict,y_test,target_names=['died','survived'])

print  "the accuracy of gradient tree boosting is",dtc.score(X_test,y_test)
print classification_report(gbc_y_predict,y_test,target_names=['died','survived'])
