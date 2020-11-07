import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import seaborn as sns

with open("./Q.dict.pkl", "rb") as fin:
    Q = pickle.load(fin)

X = np.array([(i+1) for i in range(10)])
Y = np.array([(i+1) for i in range(21)])
X, Y = np.meshgrid(X, Y)

Z0 = []
Z1 = []
Z = []
A = []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z0.append(Q[((X[i][j], Y[i][j]), 0)])
        Z1.append(Q[((X[i][j], Y[i][j]), 1)])
        Z.append(max((Q[((X[i][j], Y[i][j]), 0)], Q[((X[i][j], Y[i][j]), 1)])))
        if Z0[-1] > Z1[-1]:
            A.append(0)
        else:
            A.append(1)
    
Z0 = np.array(Z0).reshape(X.shape[0], X.shape[1])
Z1 = np.array(Z1).reshape(X.shape[0], X.shape[1])
Z = np.array(Z).reshape(X.shape[0], X.shape[1])
A = np.array(A).reshape(X.shape[0], X.shape[1])

fig = plt.figure()
plt.xlabel("Dealer's Points")
plt.ylabel("Your Points")
# 绘制3D max Q(s,a) 图
'''
ax = mplot3d.Axes3D(fig)
ax.set_zlabel("Your Win Rate")
#surf0 = ax.plot_surface(X, Y, Z0, cmap = plt.get_cmap("Blues"))
#surf1 = ax.plot_surface(X, Y, Z1, cmap = plt.get_cmap("Reds"))
surf = ax.plot_surface(X, Y, Z, cmap="rainbow")
'''
# 绘制2D A(s,a) 图
cmap = sns.cubehelix_palette(start = 2.5, rot = 1, gamma=0.7, as_cmap = True)
sns.heatmap(A, cmap=cmap)
plt.show()
