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
ax = plt.gca()
# (0.0, 1.0)
xlim = ax.get_xlim()
# (0.0, 1.0)
ylim = ax.get_ylim()
x = np.linspace(xlim[0], xlim[1], 30) # x coordinates 
y = np.linspace(ylim[0], ylim[1], 30) # y coordinates
# There will be 900 points 30 x 30 = 900 in the rectangular grid
# xv yv have the same dimensions - (number of y coordinates, number of x coordinates)  
xv, yv = np.meshgrid(x, y)
print (xv.shape)
print (yv.shape)
#%% plot meshgrid
plt.plot(xv, yv, marker='.', linestyle='none')
#%% no difference from the previous plot as both xv and yv have the same extent and number of examples
plt.plot(yv, xv, marker='.', linestyle='none')

#%%   xy = np.vstack([X.ravel(), Y.ravel()]).T
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 3)
Y, X = np.meshgrid(y, x)
#plt.plot(Y, X, marker='.', linestyle='none')
xr = X.ravel()
yr = Y.ravel()
xy = np.vstack([xr, yr])
xyt = xy.T
#%% see how the data is transformed
print (X, '\n', xr)
print (Y, '\n', yr)
print (xy, '\n')
print (xyt, '\n')
#%% after transformation
# You are creating a 2-D array with the shape (15, 2) 
# where 15 = 5 x 3 and the two columns are x and y coordinates. 
# The array is arranged in such that you take the first element from x 
# and pair it with every element in y to produce the first 3 rows 
# and you then move on to the second element in x and repeat the same step 
# to produce the next 3 rows. Continue the process until you exhaust all elements in x.  
print ("x:", x, '\n')
print ("y:", y, '\n')
print ("xyt:\n",xyt.shape, '\n', xyt, '\n')

#%%
