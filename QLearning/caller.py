import os
import threading

def call_the_python_script():
    global param, index
    while True:
        lock.acquire()
        if index >= len(param):
            lock.release()
            return 
        p = param[index]
        index += 1
        print(p[0], p[1], 0)
        lock.release()
        os.system("python easy21.py --eps 0.0 --no {} --alpha {} --maxiteration 2500000 --store_log --clear_dict --Not_print_to_screen".format(p[0], p[1]))
        print(p[0], p[1], 1)

global index, lock, param
No = [i for i in [200, 300, 400, 500]]
Alpha = [0.1 * (i+1) for i in range(10)]
param = []
for i in range(4):
    for j in range(10):
        param.append((No[i], Alpha[j]))
print(param)
nums = 6
index = 0
lock = threading.Lock()
T = list()
for i in range(nums):
    T.append(threading.Thread(target=call_the_python_script))
    T[i].setDaemon(True)
    T[i].start()
for i in range(nums):
    T[i].join()
