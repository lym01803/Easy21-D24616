import os
import re
import sys
from matplotlib import pyplot as plt 
import numpy as np
import random

eps0 = None
no0 = None
alpha0 = 0.7
plt.figure(figsize=(12, 8))
plt.title("Alpha={}, AverageWinRate-Episode".format(alpha0))
plt.xlabel("Episode")
plt.ylabel("Rate of winning")
for root, _, filenames in os.walk("./No-Alpha-Variable"):
    for filename in filenames:
        res = re.findall("([0-9|.]+)", filename)
        no = int(float(res[0]))
        alpha = float(res[1])
        print(no, alpha)
        if np.abs(alpha - alpha0) < 1e-8:
            with open(os.path.join(root, filename), "r") as fin:
                e = []
                r = []
                r2 = []
                idx = 0.0
                summ = 0.0
                for line in fin.readlines():
                    li = line.strip().split()
                    idx += 1.0
                    summ += float(li[1])
                    e.append(float(li[0]))
                    r.append(float(li[1]))
                    r2.append(summ / idx)
                plt.plot(e, r2, label="No={}".format(no))
plt.legend()
#plt.show()
plt.savefig("./figures/no-alpha/Alpha={}_average.png".format(alpha0))
str = ".\\figures\\no-alpha\\\"Alpha={}_average.png\"".format(alpha0)
print(str)
os.system(str)
