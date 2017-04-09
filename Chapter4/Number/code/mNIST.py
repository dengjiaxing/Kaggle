# encoding:utf-8
import pandas as pd
import re
import tensorflow as tf

train = pd.read_csv('../data/train.csv')
# print (train.shape)
test = pd.read_csv('../data/test.csv')
# print(test.shape)
# print (train[0:10])
# 将训练数据中的数据特征与对应标记分离
y_train = train['label']
X_train = train.drop('label', 1)
X_test = test
import skflow
#
# classifier = skflow.TensorFlowLinearClassifier(n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
# print('fadsfa')
# classifier.fit(X_train, y_train)
#
# linear_y_predict = classifier.predict(X_test)
# linear_submission=pd.DataFrame({'ImageId':range(1,28001),'Label':linear_y_predict})
# linear_submission.to_csv('../data/linear_submission.csv',index=False)

#使用skflow中已经封装好的基于TensorFlow搭建的全连接深度神经网络TensorFlowDNNClassifier进行学习预测
classifier=skflow.TensorFlowDNNClassifier(hidden_units=[200,20,10],n_classes=10,steps=5000,learning_rate=0.01,batch_size=50)
classifier.fit(X_train,y_train)
dnn_y_predict=classifier.predict(X_test)
dnn_submission=pd.DataFrame({'ImageId':range(1,28001),'Label':dnn_y_predict})
dnn_submission.to_csv('../data/dnn_submission.csv',index=False)

#使用Tensorflow中的算子自行搭建更为复杂的卷积神经网络，并使用skflow的程序接口从事MNIST数据的学习与预测。
def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
