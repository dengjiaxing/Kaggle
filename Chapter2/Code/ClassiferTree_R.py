#encoding:utf-8
from sklearn.datasets import load_boston
boston=load_boston()
#print  boston.DESCR
from sklearn.cross_validation import train_test_split
import numpy as np
#按1:4划分训练数据和测试数据
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

from  sklearn.preprocessing import  StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

from  sklearn.tree import  DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict=dtr.predict(X_test)

#比较不同参数的性能评价
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print "The value of R_squared of DecisionTreeRegressor is ",dtr.score(X_test,y_test)
print "The mean squared value of DecisionTreeRegressor is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict))
print "The mean absoluate value of DecisionTreeRegressor is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict))
