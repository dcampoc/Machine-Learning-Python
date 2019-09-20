# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:11:19 2019

@author: dcamp
"""

import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns

os.chdir(r'C:\Users\dcamp\Documents\python-practice\Machine-Learning-Python')
boston = pd.read_csv('boston.csv')
print(boston.head())

# Creating feature and target arrays (Remember that the methof .values transform informations into numpy arrays)
# The key 'MEDV' is eliminated from the dataset to define the feature vector X
dfX = boston.drop('MEDV', axis=1)
X = boston.drop('MEDV', axis=1).values
print(str(X.shape[0]) + ' instances and ' + str(X.shape[1]) + ' features')

# The key 'MEDV' is defined as the target variable  y
y = boston['MEDV'].values

X_rooms = X[:,5]
X_rooms = X_rooms.reshape(-1,1)
y = y.reshape(-1,1)
plt.figure()
plt.scatter(X_rooms, y)
plt.xlabel('Number of rooms')
plt.ylabel('Value of the house /1000 ($)')
plt.show()

plt.figure()
sns.heatmap(dfX.corr(), square=True, cmap='RdYlGn')


reg = linear_model.LinearRegression()
reg.fit(X_rooms, y)
pred_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)
y_pred = reg.predict(pred_space)
accuracy = reg.score(X_rooms,y)
print('The accuracy of the regression model (including one variable) is: ' + str(accuracy))

plt.figure()
plt.scatter(X_rooms,y, color='blue')
plt.plot(pred_space ,y_pred, color= 'black', linewidth=3, label='Regression')
plt.legend()
plt.xlabel('Number of rooms')
plt.ylabel('Value of the house /1000 ($)')
plt.show()

# USING ALL FEATURES FOR PREDICTING
reg_all = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
accuracy = reg_all.score(X_test, y_test)
print('The accuracy of the regression model (including all variables) is: ' + str(accuracy))

#from sklearn.metrics import mean_squared_error
#print('MSE: ' + str(mean_squared_error(y_pred, y_test)))

from sklearn.model_selection import cross_val_score
reg_all = linear_model.LinearRegression()
# perform a 5-fold cross-validation (cv)
cv_results = cross_val_score(reg, X, y, cv=5)
print('The mean value for the cross-validation process is: ' + str(np.mean(cv_results)))


#df = pd.DataFrame(X)
#y_1 = y.reshape(-1)
#plt.figure()
#pd.plotting.scatter_matrix(df, c = y_1, figsize= [8, 8], s=150, marker = 'D')

################## 
# CALCULATE THE IMPORTANCE OF FEATURES WHEN PREDICTING A TARGET VARIABLE
# REGULARIZE REGRESSION by lasso
from sklearn.linear_model import Lasso
names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1, normalize=True)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()
print('The most important feature is: ' + names[np.argmax(abs(lasso_coef))])


###################################################################
# REGULARIZE REGRESSION by Rigde
from sklearn.linear_model import Ridge

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
print('The best (manual) alpha for the ridge regularization is: ' + str(alpha_space[np.argmax(np.array(ridge_scores))]))

#########
print('optimization by grid search'.upper())
from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid
alpha_space = np.logspace(-4, 0, 50)
param_grid = {'alpha': alpha_space}

# Instantiate a the regression: logreg
ridge = Ridge(normalize=True)

# Instantiate the GridSearchCV object: logreg_cv
ridge_cv = GridSearchCV(ridge, param_grid, cv=10)

# Fit it to the data
ridge_cv.fit(X,y)
print("Tuned Logistic Regression Parameters: {}".format(ridge_cv.best_params_)) 
print("Best score is {}".format(ridge_cv.best_score_))



