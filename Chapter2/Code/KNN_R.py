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

from sklearn.neighbors import  KNeighborsRegressor
#使预测方式为平均回归即不同点的权重相同
uni_knr=KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict=uni_knr.predict(X_test)

#使预测方式为根据距离加权回归，距离越近权重越大
dis_knr=KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict=dis_knr.predict(X_test)

#比较不同参数的性能评价
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print "The value of R_squared of uniform-weighted  is ",uni_knr.score(X_test,y_test)
print "The mean squared value of uniform-weighted is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict))
print "The mean absoluate value of uniform-weighted is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict))

print "The value of R_squared of uniform_distance is ",dis_knr.score(X_test,y_test)
print "The mean squared value of uniform_distance is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict))
print "The mean absoluate value of uniform_distance is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict))

#由结果可知，使用KNN加权平均的策略能获得较高的模型性能