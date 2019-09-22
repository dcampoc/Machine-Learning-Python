# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 02:19:29 2019

@author: dcamp
"""

import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

os.chdir(r'C:\Users\dcamp\Documents\python-practice\Machine-Learning-Python')
df = pd.read_csv('auto.csv')
print(df.head())
#   Preprocessing from non-numerical data to numerical
df_origin = pd.get_dummies(df)
print(df_origin.head())
#   'origin_Asia' is dropped because if cars are not from US or Europe it is implicit the car is from Asia
df_origin = df_origin.drop('origin_Asia', axis=1)
print(df_origin.head())

from sklearn.linear_model import Ridge
df_X = df_origin.drop('origin_US', axis=1)
X = df_X.values
y = df_origin['origin_US'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-100, 0, 100)
ridge_scores_train = []
ridge_scores_test = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha_val in alpha_space:
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge = Ridge(alpha=alpha_val, normalize=True).fit(X_train, y_train)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores_test.append(ridge.score(X_test, y_test))
    ridge_scores_train.append(ridge.score(X_train, y_train))

plt.figure()
plt.plot(alpha_space, ridge_scores_train, color='green', label='training')
plt.plot(alpha_space, ridge_scores_test, color='blue', label='testing')
plt.xlabel('Alpha value')
plt.ylabel('Score')
plt.legend()

######################
print('optimization by grid search'.upper())
from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid
param_grid = {'alpha': alpha_space}

# Instantiate a the regression: logreg
ridge = Ridge(normalize=True)

# Instantiate the GridSearchCV object: logreg_cv
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)

# Fit it to the data
ridge_cv.fit(X,y)
print("Tuned Logistic Regression Parameters: {}".format(ridge_cv.best_params_)) 
print("Best score is {}".format(ridge_cv.best_score_))


#######################
# Handling missing values by removing rows of data

df = pd.read_csv('house-votes-84.csv')
# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df_clean = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

########################################
# Use SVMs to classify data that has missing values
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print('Second exercise: Pipeline'.upper())
df = pd.read_csv('house-votes-84.csv')
# Setup the Imputation transformer: imp
imp = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]
pipeline = Pipeline(steps)
y = df['republican']
X = df.drop(['republican'], axis=1)
X[X=='n'] = 0
X[X=='y'] = 1
X[X=='?'] = np.nan
#   Make each the distribution of data of feature components with mean zero and std=1
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


