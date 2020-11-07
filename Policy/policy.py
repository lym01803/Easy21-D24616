import os
import math
import random
import numpy as np
import pickle
from tqdm import tqdm
from easy21_env import *

global state_space
state_space = [(i, j) for i in range(1, 11) for j in range(1, 22)]

def init_V_Pi(V, Pi):
    global state_space
    for state in state_space:
        V[state] = 0
        Pi[state] = random.randint(0, 1)
    V[(0, 0)] = 0   # terminal state; its value keeps 0.

def load_pickle(pth):
    if os.path.exists(pth):
        with open(pth, "rb") as fin:
            return pickle.load(fin)
    return dict()

def calc_expected_v(P_s_a, V):
    v = 0
    for s_r in P_s_a:
        s, r = s_r[0], s_r[1]
        v += P_s_a[s_r] * (r + V[s])
    return v

def Policy_eval(V, Pi, P, theta = 0.1):
    global state_space
    while True:
        delta = 0
        for state in state_space:
            v = calc_expected_v(P[(state, Pi[state])], V)
            delta = max((delta, math.fabs(v - V[state])))
            V[state] = v
        if delta < theta:
            break

def save_pickle(obj, pth):
    with open(pth, "wb") as fout:
        pickle.dump(obj, fout)

def Policy_train():
    global state_space
    v_pth = "./V.dict"
    pi_pth = "./Pi.dict"
    P_pth = "./p.pkl"
    V = load_pickle(v_pth)
    Pi = load_pickle(pi_pth)
    P = load_pickle(P_pth)
    if len(V) == 0:
        init_V_Pi(V, Pi)
    
    iter_time = 0
    while True:
        iter_time += 1
        unstable = 0
        for s in state_space:
            v0 = calc_expected_v(P[(s, 0)], V)
            v1 = calc_expected_v(P[(s, 1)], V)
            if v0 > v1:
                a = 0
            else:
                a = 1
            if Pi[s] != a:
                unstable += 1
            Pi[s] = a
        print(iter_time, unstable)
        if unstable == 0 or iter_time >= 10000:
            break
        else:
            Policy_eval(V, Pi, P, theta=0.1)
        
    
    save_pickle(V, v_pth)
    save_pickle(Pi, pi_pth)

def Policy_test():
    v_pth = "./V.dict"
    pi_pth = "./Pi.dict"
    V = load_pickle(v_pth)
    Pi = load_pickle(pi_pth)
    #print(V)
    #print(Pi)
    win = 0
    loss = 0
    tie = 0
    for i in range(10000000):
        state = (Draw_a_card(is_first_card=True), Draw_a_card(is_first_card=True))
        while True:
            action = Pi[state]
            next_state, reward, is_terminal = Step(state, action)
            if is_terminal:
                if reward > 0:
                    win += 1
                if reward < 0:
                    loss += 1
                if reward == 0:
                    tie += 1
                break
            state = next_state
    print(win, loss, tie)

if __name__ == "__main__":
    Policy_train()
    Policy_test()