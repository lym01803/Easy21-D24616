import os, sys
import numpy as np
import random

def Draw_a_card(is_first_card = False):
    num = int(random.randint(1, 10))
    if is_first_card:
        return num # 第一张牌为黑牌
    elif random.random() < 1/3:
        return -num # 1/3 概率红牌
    return num  # 2/3 概率黑牌

def is_bust(num):
    return num < 1 or num > 21

# state : (dealer_first_card_number, sum of player's cards)
# action : 0 --- hit ; 1 --- stick
# return 
#   (next_state, reward, is_terminal)
def Step(state, action):
    dealer_card, summ = state
    # hit
    if action == 0:
        summ += Draw_a_card()
        if is_bust(summ):
            return ((dealer_card, summ), -1, True)
        else:
            return ((dealer_card, summ), 0, False)
    # stick
    if action == 1:
        summ2 = dealer_card
        while summ2 < 16:
            summ2 += Draw_a_card()
            if is_bust(summ2):
                return ((dealer_card, summ), 1, True)
        if summ2 < summ:
            return ((dealer_card, summ), 1, True)
        if summ2 == summ:
            return ((dealer_card, summ), 0, True)
        if summ2 > summ:
            return ((dealer_card, summ), -1, True)
