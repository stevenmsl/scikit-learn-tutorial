# https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a
#%%
import pandas as pd
#%% the data
df = pd.read_csv('diabetes_data.csv')
#%% exploring the data
df.head()
#%% splitting the data
features = df.drop(columns=['diabetes'])
labels = df['diabetes']
from sklearn.model_selection import train_test_split
# stratify=labels
#if ‘labels’ has 25% patients have diabetes, 
# the training data will also have 25% patients have diabetes. 
feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.3, stratify=labels, random_state = 10)

#%% K-NN 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier() 
# different k values from 1 to 24
params_knn = {'n_neighbors': np.arange(1,25)}
# vc=5 5-fold validation 
# Split the data into 5 groups. Use one group as the test set and the reaming 4 groups as the training set. 
# You then train on the training set and score on the test set. 
# Repeat the process until each group has been used as the test set. 
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(feature_train, label_train)
knn_best = knn_gs.best_estimator_
#%% 
print (knn_gs.best_params_)
#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
params_rf = {'n_estimators': [50, 100, 200]}
rf_gs = GridSearchCV(rf, params_rf, cv=5)
rf_gs.fit(feature_train, label_train)
rf_best = rf_gs.best_estimator_
#%%
print(rf_gs.best_params_) 
#%% logistic regression 
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(feature_train, label_train)

#%% accuracy scores
print ('knn: {}'.format(knn_best.score(feature_test, label_test)))
print ('rf: {}'.format(rf_best.score(feature_test, label_test)))
print ('log_reg: {}'.format(log_reg.score(feature_test, label_test)))

#%% voting classifier
from sklearn.ensemble import VotingClassifier
estimators = [('knn', knn_best), ('rf', rf_best),('log_reg', log_reg)]

ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(feature_train, label_train)
# The score should be higher comparing to all three individual models 
ensemble.score(feature_test, label_test)
