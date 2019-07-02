#https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
#%% 
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
#%%
# Axes is the area your plot appears in. 
# Don’t confuse it with axis. 
ax1 = plt.axes()
# [Left, bottom, width, height]
# starts at 65% of the width and 65% of the height of the figure
# the size of the axes is 20% of the width and 20% of the height of the figure
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

#%% vertically stack two axes
fig = plt.figure()
# The bottom of the upper panel (ax1) 0.5 is the bottom of the lower panel (ax2) 0.1 + 0.4 (height). 
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
ylim=(-1.2, 1.2))
x= np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
#%%
print (np.linspace(0, 10))
#%%
