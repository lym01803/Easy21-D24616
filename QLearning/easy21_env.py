import os, sys
import pickle
import numpy as np
import random
from tqdm import tqdm


# is_first_card : 是否是第一张牌

def Draw_a_card(is_first_card):
    num = int(random.randint(1,10))
    if is_first_card:
        return num
    elif random.random() < 1/3:
        return -num
    else:
        return num


# action: 0 -- hit ; 1 -- stick
# state: (dealer's first card, player's current sum)

def Step(state, action):
    fst_card, summ = state
    if action == 0:
        summ += Draw_a_card(is_first_card=False)
        if summ > 21 or summ < 1:
            return ((fst_card, summ), -1) # bust reward = -1
        else:
            return ((fst_card, summ), 0)  # reward = 0
    else:
        if summ < 1 or summ > 21:
            return ((fst_card, summ), -1) # bust reward = -1
        dealer_summ = fst_card
        while dealer_summ < 16:
            dealer_summ += Draw_a_card(is_first_card=False)
            if dealer_summ > 21 or dealer_summ < 1:
                return ((fst_card, summ), 1) # dealer bust reward = 1
        if dealer_summ > summ:
            return ((fst_card, summ), -1)
        if dealer_summ < summ:
            return ((fst_card, summ), 1)
        if dealer_summ == summ:
            return ((fst_card, summ), 0)
        
# params
alpha = 0.8
gamma = 1.0
epsilon = 0.1

global Q, N

def load_Q_N():
    global Q, N
    if os.path.exists("./Q_dict.pkl"):
        with open("./Q_dict.pkl", "rb") as fin:
            Q = pickle.load(fin)
            print("Q loaded")
    else:
        Q = dict()

    if os.path.exists("./N_dict.pkl"):
        with open("./N_dict.pkl", "rb") as fin:
            N = pickle.load(fin)
            print("N loaded")
    else:
        N = dict()
    

def save_Q_N():
    global Q, N
    with open("./Q_dict.pkl", "wb") as fout:
        pickle.dump(Q, fout)
    with open("./N_dict.pkl", "wb") as fout:
        pickle.dump(N, fout)

def Q_func(state, action):
    global Q
    if not tuple((state, action)) in Q:
        Q[tuple((state, action))] = 0
    return Q[tuple((state, action))]

def Q_update(state, action, val):
    global Q
    q = Q_func(state, action)
    Q[(state, action)] = (1.0-alpha) * q + alpha*val



def Epsilon_greedy(state, k = 0.1):
    if random.random() < k:
        return int(np.random.randint(0,2))
    else:
        if Q_func(state, 0) > Q_func(state, 1):
            return 0
        else:
            return 1
    '''
    # p(i) = exp(score_i)/sigma(exp(score_i))
    p0 = np.exp(k*Q_func(state, 0)) / (np.exp(k*Q_func(state, 0)) + np.exp(k*Q_func(state, 1)))
    if random.random() < p0:
        return 0
    else:
        return 1
    '''


def test():
    total = 0
    for i in range(10000):
        #print(i)
        state = (Draw_a_card(is_first_card=True), Draw_a_card(is_first_card=True))
        #print("start:{}".format(state))
        while True:
            action = Epsilon_greedy(state, k = 0.0) #int(input()) #Epsilon_greedy(state)
            #print(["hit","stick"][action], end=" ")
            new_state, reward = Step(state, action)
            #print(new_state)
            if action == 1 or reward != 0:
                #print(reward)
                total += reward
                break
            state = new_state
    print("total:{}".format(total))

def make_table():
    with open("./table.txt", "w", encoding="utf8") as fout:
        fout.write("\t")
        for i in range(1, 22):
            fout.write("{}\t".format(i))
        fout.write("\n")
        for i in range(1, 11):
            fout.write("{}\t".format(i))
            for j in range(1,22):
                if Q_func((i, j), 0) > Q_func((i, j), 1):
                    fout.write("要\t")
                else:
                    fout.write("不\t")
            fout.write("\n")


if __name__ == "__main__":
    load_Q()
    
    total_reward = 0
    for i in tqdm(range(1500000)):
        state = (Draw_a_card(is_first_card=True), Draw_a_card(is_first_card=True))
        alpha = 1.0 * np.exp(-i/100000)
        #iter_times = 0
        while True:
            action = Epsilon_greedy(state, k = 0.5 * np.exp(-i/100000))
            new_state, reward = Step(state, action)
            if action == 1 or reward != 0:
                Q_update(state, action, reward)
                total_reward += reward
                break
            else:
                Q_update(state, action, reward + gamma * np.max([Q_func(new_state, 0), Q_func(new_state, 1)]))
            state = new_state
            #iter_times += 1
        if (i+1) % 100000 == 0:
            print(total_reward / 100000)
            with open("./train_log.txt", "a") as fout:
                fout.write("iter: {} ; average reward: {}\n".format(i+1, total_reward/100000))
            save_Q()
            total_reward = 0

    with open("Q_dict.txt", "w") as fout:
        for i in Q:
            fout.write("{} {}\n".format(i, Q[i]))
    
    test()
    
    make_table()

