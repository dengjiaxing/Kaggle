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

from sklearn.svm import  SVR
#使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict=linear_svr.predict(X_test)

#使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr=SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict=poly_svr.predict(X_test)

#使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr=SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict=rbf_svr.predict(X_test)

from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print "The value of R_squared of linear SVR is ",linear_svr.score(X_test,y_test)
print "The mean squared value of LinearRegression is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict))
print "The mean absoluate value of LinearRegression is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict))

print "The value of R_squared of poly SVR is ",poly_svr.score(X_test,y_test)
print "The mean squared value of poly SVR is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict))
print "The mean absoluate value of poly SVR is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict))

print "The value of R_squared of rbf SVR is ",rbf_svr.score(X_test,y_test)
print "The mean squared value of rbf SVR is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict))
print "The mean absoluate value of rbf SVR is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict))

#比较上面三个不同核函数下的性能评估，发现使用径向基核函数具有更好的预测性能