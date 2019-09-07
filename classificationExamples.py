# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 00:32:35 2019

@author: dcamp
"""

from sklearn import datasets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# Exploratory data analysis (EDA, for short)
plt.style.use('ggplot')
iris = datasets.load_iris()
print(type(iris))
print(iris.keys())
print(type(iris.data), type(iris.target))
print(iris.data.shape)
print(iris.target_names)

X = iris.data
y = iris.target 
df = pd.DataFrame(X, columns=iris.feature_names)
print(df)
plt.figure()
pd.plotting.scatter_matrix(df, c = y, figsize= [8, 8], s=150, marker = 'D')

from sklearn.neighbors import KNeighborsClassifier
#   Traning model based on training data with 6 neighbors 
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
#   Proposed data for classification 2 samples with the four features
X_new = [[5, 1, 3, 2],
         [1, 5, 2, 2]]
pred = knn.predict(X_new)
print('Predictions:'.upper())
print(pred)

############

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print('Performance of clasifier (train/testing) split:'.upper())
print(accuracy)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


##########
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits['DESCR'])

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.figure()
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Print the accuracy
print('Aaccuracy'.upper())
print(knn.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


###########################################
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.figure()
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.plot(neighbors, (train_accuracy + test_accuracy)/2, label = 'Average Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()



