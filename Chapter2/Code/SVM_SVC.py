#encoding:utf8
#导入数字加载器
from sklearn.datasets import  load_digits
digits=load_digits()
print digits.data.shape

#分割数据得到训练数据和测试数据
from sklearn.cross_validation import train_test_split
#按1:4划分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
print  y_train.shape

from sklearn.preprocessing import StandardScaler
from  sklearn.svm import LinearSVC
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
lsvc=LinearSVC()
lsvc.fit(X_train,y_train);
y_predict=lsvc.predict(X_test)
#使用模板自带评估函数进行准确性测评
print "The Accuracy of Linear SVC is",lsvc.score(X_test,y_test)
from  sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))