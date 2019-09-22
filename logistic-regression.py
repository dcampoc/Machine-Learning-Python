# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:41:21 2019

@author: dcamp
"""

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
wine = datasets.load_wine()
X = wine.data
X = scaler.fit_transform(X)
y = wine.target


##### adapt this algorithm 

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3, random_state=42)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression()) 

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))
print('probabilistic response'.upper())
print(clf.predict_proba(X_test))