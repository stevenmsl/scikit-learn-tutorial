#%% https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

#%%
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10+y*x)*np.cos(x)
#%% 
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
#%% exploring data 
# The shape should be the same for all three: (40, 50)
print(X.shape, Y.shape, Z.shape)
print(Z)
#%% contour plot
plt.contour(X, Y, Z, colors='black')

#%%
plt.contour(X, Y, Z, 20, cmap='RdGy')
#%% color bar
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()

#%% eliminate splotches – color steps are discrete rather than continuous
plt.imshow(Z, extent=[0,5,0,5], # must manually specify the extent
origin='lower', # change it to lower left to show the gridded data
cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image') #make x and y units match
#%% combine contour and image plots
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5],
origin='lower',
cmap='RdGy', alpha=0.5)
plt.colorbar()
#%%
