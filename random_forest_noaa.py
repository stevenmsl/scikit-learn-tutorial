# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#%%
import pandas as pd

#%% Explore data
features = pd.read_csv('temps.csv')
#%%
features.head(n=5)
#%% 
features.describe()
#%% one-hot encoding to encode the 'week' column
features = pd.get_dummies(features)
features.iloc[:,5:].head(5)
#%% Features and Targets
import numpy as np 
labels = np.array(features['actual'])
features = features.drop('actual', axis= 1)
features_list = list(features.columns) 
features = np.array(features)

#%% split data
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size = 0.25, random_state = 42)


#%% Establish baseline
baseline_preds = test_features[:, features_list.index('average')]
baseline_errors = abs(baseline_preds - test_labels)
print ('Average baseline error: ', round(np.mean(baseline_errors),2))


#%% Train Model 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)

#%% Prediction 
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print ('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.' )
print (errors)
#%% Performance metrics
# Mean average percentage error
mape = 100 * (errors/test_labels)
print (mape)
accuracy = 100 - np.mean(mape)
print('Accuracy: ', round(accuracy, 2), '%.')

#%%
from sklearn.tree import export_graphviz
import graphviz
import pydot
# pull out one tree
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names=features_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
# Other than pip install Graphviz and pip install pydotplus. 
# You also need to install Graphviz binary. 
# For example, check if the path like the following exists: C:\Program Files (x86)\Graphviz2.38\bin
# You also need to add that path to the System Variable Path

graph.write_png('tree.png')

#%% Build forest with reduced size trees 

rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
rf_small.fit(train_features, train_labels)
tree_small = rf_small.estimators_[6]
export_graphviz(tree_small, out_file='small_tree.dot', feature_names = features_list, rounded=True, precision =1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')
#%% Variable importance
importances = list(rf.feature_importances_)
feature_importances = [ (feature, round(importance, 2)) for feature, importance in zip(features_list, importances) ]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#%% Model with most important variables only
rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)
important_indices = [features_list.index('temp_1'), features_list.index('average')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

rf_most_important.fit(train_important, train_labels)
predictions = rf_most_important.predict(test_important)
errors = abs (predictions - test_labels)
print ('Mean Absolute Error:', round(np.mean(errors), 2), 'degree.')

mape = np.mean(100* (errors/test_labels))
accuracy = 100 - mape
print ('Accuracy:', round(accuracy, 2), '%.')

#%% visualizations 
import matplotlib.pyplot as plt 
# To solve the problem where in VS Code the default dark theme makes the labels hard to see 
plt.style.use('dark_background')
# This would produce [0,1,2,…,16]
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation ='vertical')
plt.xticks(x_values, features_list,  rotation ='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable')
plt.title('Variable Importances')

#%%
