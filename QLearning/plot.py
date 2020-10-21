import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pickle

with open("./Q.dict.pkl", "rb") as fin:
    Q = pickle.load(fin)

X = np.array([(i+1) for i in range(10)])
Y = np.array([(i+1) for i in range(21)])
X, Y = np.meshgrid(X, Y)

Z0 = []
Z1 = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z0.append(Q[((X[i][j], Y[i][j]), 0)])
        Z1.append(Q[((X[i][j], Y[i][j]), 1)])
Z0 = np.array(Z0).reshape(X.shape[0], X.shape[1])
Z1 = np.array(Z1).reshape(X.shape[0], X.shape[1])

fig = plt.figure()
ax = mplot3d.Axes3D(fig)
surf0 = ax.plot_surface(X, Y, Z0, cmap = plt.get_cmap("Blues"))
surf1 = ax.plot_surface(X, Y, Z1, cmap = plt.get_cmap("Reds"))
plt.show()
