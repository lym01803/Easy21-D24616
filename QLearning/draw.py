import os
import re
from matplotlib import pyplot as plt 
import numpy as np
import random

colors = ["r", "g", "b", "c", "m", "k"]
def draw(x, y):
    plt.plot(x, y, colors[random.randint(0, len(colors)-1)])

for root, _, filenames in os.walk("./No-Alpha-Variable"):
    for filename in filenames:
        res = re.findall("([0-9|.]+)", filename)
        eps = float(res[0])
        alpha = float(res[1])
        print(eps, alpha)
        if np.abs(alpha - 0.7) < 1e-8:
            with open(os.path.join(root, filename), "r") as fin:
                e = []
                r = []
                for line in fin.readlines():
                    li = line.strip().split()
                    e.append(float(li[0]))
                    r.append(float(li[1]))
                    draw(e, r)
    plt.show()