#https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
#%% generating data
from sklearn.datasets.samples_generator import make_blobs
#generate isotropic Gaussian blobs for clustering
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
#%% exploring data  
print (X.shape)
print (X)
#%%
print (y.shape)
print (y)
#%% linear discriminative classifier
xfit = np.linspace(-1, 3.5) # return evenly spaced numbers over a specified interval
plt.scatter(X[:,0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5) # set x-axis limits  
#%%
print (xfit)
#%%
