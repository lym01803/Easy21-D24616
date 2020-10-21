import os, sys
import pickle
import numpy as np
import random
from tqdm import tqdm
import argparse
from easy21_env import *

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
    parser.add_argument("--p_save_path", type=str)
    parser.add_argument("--p_sample", type=int)

    calc_p(parser.parse_args().p_save_path, parser.parse_args().p_sample)
