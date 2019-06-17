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
#%% maximizing the margin
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
for m,b,d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
    color='#AAAAAA', alpha=0.4
    ) 
plt.xlim(-1, 3.5)
#%% fitting SVM
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)
#%%  function that plots SVM
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca() # get the current Axes instance
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #create a grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30) # 30 samples
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)

#%% meshgrid
nx, ny = (3, 2)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
print (x.shape); print (x); print (y.shape); print(y)
# There will be 6 points 3 x 2 = 6 in the rectangular grid
xv, yv = np.meshgrid(x, y)
print (xv.shape, xv)
print (yv.shape, yv)
