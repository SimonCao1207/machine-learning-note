# Reinforcement learning

- Trade off of **exploitation** and **exploration**: 
    - agent has to **exploit** what it has experienced in order to obtain reward
    - agent has to **explore** in order to make better action selections in the future. 

- A **policy** defines the learning agent's way of behaving at a given time. 
    - It is a map from state to action
    - Deterministic policy : $a = \pi(s)$
    - Stochastic policy : $\pi(a|s) = \mathbb{P}[A_t = a | S_t = s] $ 

- A **reward** $R_t$ : a scalar feedback signal that defines the goal of a RL problem.  

- **Discounted return** : a reward received
k time steps in the future is worth only $\gamma^{k-1}$ times what it would be worth if it were received immediately. 
$$G_t = \sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$
- A **value function** specifies what is good in the long run (while reward signal indicates what is good in an immediate sense).
    - $v_{\pi}(s)$ : state-value function of a state s under policy $\pi$ : the expected return when starting in $s$ and following $\pi$ thereafter.
    $$v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s]$$
    - Action choices are made based on value judments.
    - $q_{\pi}(s)$ : action-value function under policy $\pi$ : the expected return when starting in $s$, taking action $a$ and following $\pi$ thereafter.
    $$q_{\pi}(s) = E_{\pi}[G_t | S_t = s, A_t = a] = E_{\pi}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s, A_t=a]$$
    - Values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime.

- A **model** : agent's representation of the environment.
    - given a state and action, the model might predict the resultant next state and next action and used for planning.

- 
