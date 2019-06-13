#https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
#%% generating data
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
#%% exploring data  
print (X.shape)
print (X)
#%%
print (y.shape)
print (y)