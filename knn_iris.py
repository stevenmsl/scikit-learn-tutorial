#%%
import numpy as np
from sklearn import datasets

#%% Understand the data 
iris = datasets.load_iris()
iris_X = iris.data  # pylint: disable=maybe-no-member
#(150, 4)
print (iris_X.shape)
# 150
print (len(iris_X))
#(150)
iris_Y = iris.target # pylint: disable=maybe-no-member
print (iris_Y.shape)
np.unique(iris_Y)

#%% KNN (k nearest neighbors) classification
iris = datasets.load_iris()
iris_X = iris.data  # pylint: disable=maybe-no-member 
iris_Y = iris.target # pylint: disable=maybe-no-member
indices = np.random.permutation(len(iris_X)) 
#should be (150,)
print(indices.shape)
#split data into train and test data
iris_X_train = iris_X[indices[:-10]]
iris_Y_train = iris_Y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_Y_test = iris_Y[indices[-10:]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_Y_train)
print(knn.predict(iris_X_test))
print(iris_Y_test)
