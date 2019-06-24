#https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10+y*x)*np.cos(x)

#%%   xy = np.vstack([X.ravel(), Y.ravel()]).T
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 3)
#It really doesn’t mater how the meshgrid is created.
#There is still the same number of combinations; only their location in the array are different.  
#The x coordinates are still the x coordinates, 
#and the y coordinates are still the y coordinates 
#whether you are feeding them into a function or drawing a contour.  
Y, X = np.meshgrid(y, x) 
Z = f(X, Y)
#%% plot it
plt.contour(X, Y, Z, colors='black')
#%%
X2, Y2 = np.meshgrid(x, y)
Z2 = f(X2, Y2)
#%% this is should be exactly the same as the previous one.
plt.contour(X2, Y2, Z2, colors='black')
