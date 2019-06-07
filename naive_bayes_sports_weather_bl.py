# https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
#%%
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
'Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#%% Encode features
from sklearn import preprocessing
import numpy as np
le = preprocessing.LabelEncoder()
# should look like this: 
#[ 2 2 0 ... 1]
weather = le.fit_transform(weather)

temp = le.fit_transform(temp)
# turn tuple into 2-d array
features =[list(item) for item in zip(weather, temp)] 

labels = le.fit_transform(play)
#%% Train and Predict
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(features,labels)
predicted = model.predict([[0,2],[0,1]])
print (predicted)
