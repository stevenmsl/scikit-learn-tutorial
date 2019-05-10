#%%
from sklearn import datasets
diabetes = datasets.load_diabetes()
#%% investigate data
#check the shape
#(442, 10)
print (diabetes.data.shape) # pylint: disable=maybe-no-member
#print (diabetes.data)
#(442,)
print (diabetes.target.shape) # pylint: disable=maybe-no-member
#%% see what the data look like
print (diabetes.data) # pylint: disable=maybe-no-member
print (diabetes.target) # pylint: disable=maybe-no-member
#%% linear regression
# y = xb + c  target = data x coefficients + observation noise
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20] # pylint: disable=maybe-no-member
diabetes_X_test  = diabetes.data[-20:] # pylint: disable=maybe-no-member
diabetes_y_train = diabetes.target[:-20] # pylint: disable=maybe-no-member
diabetes_y_test  = diabetes.target[-20:] # pylint: disable=maybe-no-member
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_.shape)
print(regr.coef_) 
import numpy as np
#2004.5676026898223
np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
#variance score: 1 perfect; 0 no linear relationship
print(regr.score(diabetes_X_test, diabetes_y_test))
