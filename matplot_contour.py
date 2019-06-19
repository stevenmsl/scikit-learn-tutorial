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

#%% 
