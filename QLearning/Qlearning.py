import os, sys
import pickle
import numpy as np
import random
from tqdm import tqdm
import argparse
from easy21_env import *

global Q, N, parser

def load_Q_N():
    global Q, N
    if os.path.exists("./Q.dict.pkl"):
        with open("./Q.dict.pkl", "rb") as fin:
            Q = pickle.load(fin)
    else:
        Q = dict()
    if os.path.exists("./N.dict.pkl"):
        with open("./N.dict.pkl", "rb") as fin:
            N = pickle.load(fin)
    else:
        N = dict()

def save_Q_N():
    global Q, N
    with open("./Q.dict.pkl", "wb") as fout:
        pickle.dump(Q, fout)
    with open("./N.dict.pkl", "wb") as fout:
        pickle.dump(N, fout)

def get_val(func, state, action):
    if not (state, action) in func:
        func[(state, action)] = 0
    return func[(state, action)]

def modify_val(func, state, action, delta, alpha):
    val = get_val(func, state, action)
    func[(state, action)] = val + alpha * delta

def E_greedy(state, eps = 0, No = 300):
    if np.abs(eps) > 1e-8:
        p = eps
    else:
        p = No / (No + get_val(N, state, 0) + get_val(N, state, 1))
    if random.random() < p:
        return int(random.randint(0, 1))
    if get_val(Q, state, 0) > get_val(Q, state, 1):
        return 0
    else:
        return 1
    '''
    q0 = get_val(Q, state, 0)
    q1 = get_val(Q, state, 1)
    p0 = np.exp(k*q0) / (np.exp(k*q0) + np.exp(k*q1))
    if random.random() < p0:
        return 0
    else:
        return 1
    '''

def train(max_iter=10000000, store_log=""):
    global parser
    args = parser.parse_args()
    load_Q_N()
    win = 0
    iter_times = 0
    while iter_times < max_iter:
        state = (Draw_a_card(is_first_card=True), Draw_a_card(is_first_card=True))
        is_terminal = False
        while not is_terminal:
            action = E_greedy(state, eps=args.eps, No=args.no) 
            q = get_val(Q, state, action)
            next_state, reward, is_terminal = Step(state, action)
            if is_terminal:     # 判定是否终止
                delta = reward - q
                if reward > 0:
                    win += 1
            else:
                delta = reward + np.max([get_val(Q, next_state, 0), get_val(Q, next_state, 1)]) - q
            modify_val(N, state, action, 1, 1)  # 更新 N
            if args.variable:
                alpha = args.alpha / get_val(N, state, action)
            else:
                alpha = args.alpha      # 根据参数确定学习率
            modify_val(Q, state, action, delta, alpha)
            state = next_state
        iter_times += 1
        interval = 100000
        if iter_times % interval == 0:
            if not args.Not_print_to_screen:
                print(iter_times, win/interval)
            if store_log != "":
                with open(store_log, "a") as fout:
                    fout.write("{} {}\n".format(iter_times, win/interval))
            win = 0
            if args.save_dict:
                save_Q_N()

    with open("./table.txt", "w", encoding="utf8") as fout:
        for item in Q:
            fout.write("{} {} {}\n".format(item, get_val(Q, item[0], item[1]), get_val(N, item[0], item[1])))

def test():
    global parser
    args = parser.parse_args()
    load_Q_N()
    win = 0
    loss = 0
    tie = 0
    for i in range(100000):
        state = (Draw_a_card(is_first_card=True), Draw_a_card(is_first_card=True))
        while True:
            action = E_greedy(state, eps=args.eps, No=0)
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

def calc_p(save_pth, sample):
    sample = max((sample, 100))
    states = [(s1, s2) for s1 in range(1, 11) for s2 in range(1, 22)]
    actions = [0, 1]
    stop_state = (0, 0)
    P = dict()
    for s in tqdm(states):
        for a in actions:
            P[(s, a)] = dict()
            D = P[(s, a)]
            for i in range(sample):
                s_, rwd, is_terminal = Step(s, a)
                if is_terminal:
                    s_ = stop_state
                if (s_, rwd) in D:
                    D[(s_, rwd)] += 1
                else:
                    D[(s_, rwd)] = 1
            for i in D:
                D[i] /= sample
    with open(save_pth, "wb") as fout:
        pickle.dump(P, fout)

if __name__ == "__main__":
    global parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--no", type=float, default=0.0)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--variable", action="store_true")
    parser.add_argument("--maxiteration", type=int, default=10000000)
    parser.add_argument("--store_log", action="store_true")
    parser.add_argument("--clear_dict", action="store_true")
    parser.add_argument("--save_dict", action="store_true")
    parser.add_argument("--Not_print_to_screen", action="store_true")
    parser.add_argument("--calc_p", action="store_true")
    parser.add_argument("--p_save_path", type=str)
    parser.add_argument("--p_sample", type=int)

    if parser.parse_args().calc_p:
        calc_p(parser.parse_args().p_save_path, parser.parse_args().p_sample)
    elif parser.parse_args().test:
        test()
    else:
        if parser.parse_args().clear_dict:
            if os.path.exists("./Q.dict.pkl"):
                os.remove("./Q.dict.pkl")
            if os.path.exists("./N.dict.pkl"):
                os.remove("./N.dict.pkl")

        log_file = ""
        if parser.parse_args().store_log:
            if np.abs(parser.parse_args().eps) > 1e-8:
                log_file = "Eps{}Alpha{}".format(parser.parse_args().eps, parser.parse_args().alpha)
            else:
                log_file = "No{}Alpha{}".format(parser.parse_args().no, parser.parse_args().alpha)
            if parser.parse_args().variable:
                log_file += "variable"
            log_file = ".\\{}.txt".format(log_file)
        
        train(parser.parse_args().maxiteration, store_log = log_file)
