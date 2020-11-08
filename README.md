# Easy-21

## Introduction

Easy-21 is a single-player game evolving from Blackjack. The main differences between Easy-21 and Blackjack are:

- There are only two players, you and the dealer.
- There are two types of card：red card (1/3) and black card (2/3). Red card has negative points while black card has positive points. 
- Underflow card blasting (because of red cards)

State $s=(p_d,p_p)$, where $p_d$ is the dealer's point, $p_p$ is the player's point. State Space $\mathcal{S}=\{1,2,\cdots,10\}\times\{1,2,\cdots,21\}$ .

Action $a\in\{0,1\}$, $0$ means Hit, $1$ means Stick.

## Q-learning

### Algorithm

**Update Equation**

$$Q_{t+1}(s,a)=Q_t(s,a) + \alpha\cdot[R(s,a,s')+\gamma\cdot\max_{a}Q_t(s',a)-Q_t(s,a)]$$

**Process**

<img src="./Report_graph/graph0.png" alt="image-20201107231106262" style="zoom: 80%;" />

- Initialization: For all $a\in\mathcal{A}, s\in\mathcal{S}$, assign a value to $Q(s,a)$ arbitrarily.
- Sampling: Generate training data (a set of $(s,a,s')$) through simulating Easy-21 game. 

- Update: Using the training data to update the $Q$ function according to the update equation.

**Implement**

Here is [our code](https://github.com/lym01803/Easy21-D24616/tree/master/QLearning) for Easy-21 game using Q-learning method.

### Experiment

**Exploration and exploitation**

There are many method to balance exploration and exploitation , and we have attempted some:

- $\epsilon$-greedy : Take best action (according to current Q value) with a probability of $1-\epsilon$, and take random action (must be valid) with a probability of $\epsilon$ .
- Take action according to the probability distribution: $P(a|s,Q)=\dfrac{e^{Q(s,a)/\tau}}{\sum_{a\in\mathcal{A}}e^{Q(s,a)/\tau}}$ , where $\tau$ is a positive const number.
- Take the best action (according to current Q value) with a probability of $1-\dfrac{n}{N(s)+n}$, and take random action (must be valid) with a probability of $\dfrac{n}{N(s)+n}$, where $n$ is a positive const number, and $N(s)$ means the number of times that $s$ has been visited. In other words,  every state has its own $\epsilon$ and the $\epsilon$ here is variable (the more times $s$ is visited, the smaller $\epsilon$ is) .

We mainly show the result of the third method.

**Learning rate**

We consider two method :

- Using a fixed learning rate all the time and for all the $(s,a)$.

- Make learning rate variable, and every $(s,a)$ has its own learning rate according to the visit times, $\alpha(s,a)=\dfrac{\alpha_0}{N(s,a)}$ .

The second method is better. We mainly show the result of the second one. 

#### $\frac{n}{N(s)+n}$-greedy

##### Fixed $n$

With 2.5M episodes:

<img src="./Report_graph/graph1.png" alt="image-20201107231106262" style="zoom: 100%;" />

The left column shows the average win rate of the recent $100k$ episodes, and the right column shows the average win rate of all the episodes from the beginning.

From the graphs above, we find $\alpha\approx0.5$ performs well. When the learning rate is too high, for example $\alpha=1.0$, the win rate go up quickly in the early stages because it learns from episodes quickly. However the final win rate is relatively low, because a model with higher learning rate are more sensitive to the noise data. When the learning rate is too low, for example $\alpha=0.1$, the model learning slowly and the win rate is low. We are going to test $\alpha=0.3,0.5,0.7$ then.

**Fixed $\alpha$**

With 2.5M episodes:

<img src="./Report_graph/graph2.png" alt="image-20201107231106262" style="zoom: 100%;" />

The left column shows the average win rate of the recent $100k$ episodes, and the right column shows the average win rate of all the episodes from the beginning.

A higher $n$ leads to more exploration while a lower $n$ leads to more exploitation. We find that when $\alpha$ is fixed, $n=200$ or $n=300$ performs well. 

#### Set $\alpha=0.5$, $n=300$, train with 10M episodes

##### The final win rate

We test the model with 10M episodes, it wins in 4783569 episodes, loses in 5009724 episodes and tie in 206707 episodes.

Win rate: $47.8\%$  

 ##### The graph of $\max_aQ(s,a)$



<img src="./Report_graph/max_Q_s_a.svg"/>

## Policy Iteration

### Algorithm

**Bellman Equation**

$$V(s)=\max_a \sum_{s'\in\mathcal{S}}P(s'|s,a)[R(s,a,s')+\gamma V(s')]$$

$\pi(s)=\mathtt{argmax}_a \sum_{s'\in\mathcal{S}}P(s'|s,a)[R(s,a,s')+\gamma V(s')]$

**Process**

<img src="./Report_graph/graph3.png" style="zoom:80%;" >

- Preparation:
  - Calculate $P(s'|s,a)$. We use simulation to get the distribution.
- Initialization: Assign a random value to $V(s)$ and $\pi(S)$ for every $s$. 
- According to current $V$, update the policy $\pi$, and count the unstable states.
- If there is any unstable state, evaluate $V$ according to current $\pi$.
- The iteration will be executed until the policy $\pi$ is convergent.

**Implement**

Here is [our code](https://github.com/lym01803/Easy21-D24616/tree/master/Policy) for Easy-21 game using Policy Iteration method.

### Experiment

The policy iteration has a very clear training process and almost no hyper-parameter.

In our experiment, the policy iteration method becomes to be convergent in only 5 iteration.