import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import seaborn as sns

with open("./V.dict", "rb") as fin:
    V = pickle.load(fin)
with open("./Pi.dict", "rb") as fin:
    Pi = pickle.load(fin)

X = np.array([(i+1) for i in range(10)])
Y = np.array([(i+1) for i in range(21)])
X, Y = np.meshgrid(X, Y)

Z = []
A = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z.append(V[(X[i][j], Y[i][j])])
        A.append(Pi[(X[i][j], Y[i][j])])
Z = np.array(Z).reshape(X.shape[0], X.shape[1])
A = np.array(A).reshape(X.shape[0], X.shape[1])

fig = plt.figure()
plt.xlabel("Dealer's Points")
plt.ylabel("Your Points")
#surf0 = ax.plot_surface(X, Y, Z0, cmap = plt.get_cmap("Blues"))
#surf1 = ax.plot_surface(X, Y, Z1, cmap = plt.get_cmap("Reds"))
'''
ax = mplot3d.Axes3D(fig)
ax.set_zlabel("Your Win Rate")
surf = ax.plot_surface(X, Y, Z, cmap="rainbow")
'''
cmap = sns.cubehelix_palette(start = 2.5, rot = 1, gamma=0.7, as_cmap = True)
sns.heatmap(A, cmap=cmap)
plt.show()
