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

#导入集成模型
from  sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict=rfr.predict(X_test)

efr=ExtraTreesRegressor()
efr.fit(X_train,y_train)
efr_y_predict=efr.predict(X_test)

gfr=GradientBoostingRegressor()
gfr.fit(X_train,y_train)
gfr_y_predict=gfr.predict(X_test)

#比较不同集成模型的性能评价
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print "The value of R_squared of RandomForestRegressor is ",rfr.score(X_test,y_test)
print "The mean squared value of RandomForestRegressor is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict))
print "The mean absoluate value of RandomForestRegressor is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict))

print "The value of R_squared of ExtraTreesRegressor is ",efr.score(X_test,y_test)
print "The mean squared value of ExtraTreesRegressor is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(efr_y_predict))
print "The mean absoluate value of ExtraTreesRegressor is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(efr_y_predict))

print "The value of R_squared of GradientBoostingRegressor is ",gfr.score(X_test,y_test)
print "The mean squared value of GradientBoostingRegressor is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gfr_y_predict))
print "The mean absoluate value of GradientBoostingRegressor is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gfr_y_predict))
