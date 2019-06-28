#https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
#%% 
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
#%%
# Axes is the area your plot appears in. 
# Don’t confuse it with axis. 
ax1 = plt.axes()
# starts at 65% of the width and 65% of the height of the figure
# the size of the axes is 20% of the width and 20% of the height of the figure
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

#%% vertically stack two axes
fig = plt.figure()


#%%
