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
#%% subplot
for i in range(1, 7):
    plt.subplot(2, 3, i) # (Row, column, index) 
    plt.text(0.5, 0.5, str((2, 3, i)),
    fontsize=18, ha='center'
    )

#%% specify spacing
fig = plt.figure()
# specify the spacing along the height and width of the figure 
fig.subplots_adjust(hspace=0.4, wspace=0.4) #40% of the subplot width and height
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
    fontsize=18, ha='center')
#%% The whole grid in one go
fig, ax = plt.subplots(2, 3, 
sharex='col', # remove inner labels on the grid
sharey='row'
)
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),fontsize=18, ha='center')

#%% more complicated arrangements
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])

#%%
mean = [0, 0]
cov = [[1, 1], [1, 2]]
# Check this link to see why you need to use .T  
# https://stackoverflow.com/questions/5741372/syntax-in-python-t
x, y = np.random.multivariate_normal(mean, cov, 3000).T
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

#scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
# histogram 
# the histogram on the right bottom
x_hist.hist(x, 40, histtype='stepfilled',
orientation='vertical', color='gray')
x_hist.invert_yaxis()

# the histogram on the left
y_hist.hist(y, 40, histtype='stepfilled',
orientation='horizontal', color='gray')
y_hist.invert_xaxis()

#%%
