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
# print "The max target value is",np.max(boston.target)
# print "The min target value is",np.min(boston.target)
# print "The average target value is",np.mean(boston.target)

from  sklearn.preprocessing import  StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

from  sklearn.linear_model import  LinearRegression
#使用默认设置初始化线性回归器LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

from sklearn.linear_model import  SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict=sgdr.predict(X_test)

#使用LinearRegression模型自带的评估模块，并输出评价结果
print "The value of default measurement of LinearRegression is ",lr.score(X_test,y_test)
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print "The value of R_squared of LinearRegression is ",r2_score(y_test,lr_y_predict)
print "The mean squared value of LinearRegression is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print "The mean absoluate value of LinearRegression is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))

print "The value of default measurement of SGDRegressor is ",sgdr.score(X_test,y_test)
print "The value of R_squared of SGDRegressor is ",r2_score(y_test,sgdr_y_predict)
print "The mean squared value of SGDRegressor is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict))
print "The mean absoluate value of SGDRegressor is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict))
