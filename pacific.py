# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:39:22 2019

@author: chinmay
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
data=pd.read_excel("E:\\python\\classes\\datasets\\Pacific1.xlsx")
print(data.head())

#Data wrangling starts here
data.Status= pd.Categorical(data.Status)
data['Status']=data.Status.cat.codes #assigns int values to all different categories
print(data.Status)#shows data in categorial form
sns.countplot(data['Status'],label='count')
plt.show()

pred_columns=data[:]
pred_columns.drop(['Status'],axis=1,inplace=True)

pred_columns.drop(['Event'],axis=1,inplace=True)

pred_columns.drop(['Latitude'],axis=1,inplace=True)

pred_columns.drop(['Longitude'],axis=1,inplace=True)

pred_columns.drop(['Name'],axis=1,inplace=True)

pred_columns.drop(['ID'],axis=1,inplace=True)
prediction_var= pred_columns.columns
print(list(prediction_var))

#Spliting the data for testing and testing
train,test= train_test_split(data,test_size=0.3)
print(train.shape)
print(test.shape)

#Creating a resposnse and target variable 

train_X= train[prediction_var]
train_Y= train['Status']
print(list(train.columns))

#taking the testing data input 
test_X= test[prediction_var]
test_Y= test['Status']
print(list(test.columns))

#creating a decision tree model tree based on the training data

model=tree.DecisionTreeClassifier()
model.fit(train_X,train_Y)
#now,prediction using trained model

prediction=model.predict(test_X)

#now displaying the predicted vs actual values
df=pd.DataFrame(prediction,test_Y)
print(df)
#checking the accuracy of the model 
print(metrics.accuracy_score(prediction,test_Y))
