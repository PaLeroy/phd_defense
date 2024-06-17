class: middle, center, title-slide
count: false

# Contributions to Multi-Agent Reinforcement Learning

PhD Defense

<br><br>

Pascal Leroy<br>
June 28, 2024

---
# Today
<br><br>

- What is reinforcement learning (RL)?
<br><br>

- What are the challenges when multiple agents learn together?
<br><br>

- How to train a team of agents to cooperate with RL?
<br><br>

- How to train a team to compete against another one?

---

class: section

# Reinforcement learning

---
class: middle


.italic[Reinforcement learning is learning what to do, how to map situations to .bold[actions], so as to .bold[maximize] a numerical .bold[reward] signal.]

.pull-right[Sutton & Barto, 1998.] <br>

.center[.circle.width-25[![](figures/sutton_pic.png)]
.circle.width-25[![](figures/barto_pic.jpg)]]


---
class: middle

.center[
<video controls preload="auto" height="500" width="700">
  <source src="./figures/chicken1.mp4" type="video/mp4">
</video>]

.footnote[Video credits: [Megan Hayes](https://twitter.com/PigMegan), [@YAWScience](https://twitter.com/YAWScience/status/1304199719036444672), 2020. Taken from [Intro to IA](https://github.com/glouppe/info8006-introduction-to-ai/) by Gilles Louppe.]

---
class: middle

.center[
<video controls preload="auto" height="500" width="700">
  <source src="./figures/chicken2.mp4" type="video/mp4">
</video>]

.footnote[Video credits: [Megan Hayes](https://twitter.com/PigMegan), [@YAWScience](https://twitter.com/YAWScience/status/1304199719036444672), 2020. Taken from [Intro to IA](https://github.com/glouppe/info8006-introduction-to-ai/) by Gilles Louppe.]

---

# Markov decision process (MDP)

.italic["RL is an agent learning by .bold[interacting] in its .bold[environment] to maximise a reward."]

<br>

.center[.width-100[![](figures/MDP_words.png)]]

.footnote[R. S. Sutton and A. G. Barto. Reinforcement learning, second edition: An introduction. 2018]

---

An .bold[MDP] is defined by a tuple ($\mathcal{S}$, $\mathcal{U}$,${P}$,${R}$):



- State space $s_t \in \mathcal{S}$.
- Action space $u_t \in \mathcal{U}$.
- Transition function $s\_{t+1} = P(s\_t, u\_t)$ defines the state $s_{t+1}$ reached after taking the action $u_t$ in state $s_t$.
- Reward function $r\_t=R(s\_{t+1})$ defines the reward obtained.

.center[.width-100[![](figures/MDP_words.png)]]


---
class: middle

Agent selects action $u$ in state $s$ based on its deterministic policy $\pi(s) : \mathcal{S} \rightarrow \mathcal{U}$.

.bold[Goal]: learn the policy $\pi$ that maximises the sum of rewards $\mathbb{E}_{\pi} \left[ \sum_t r_t \right]$.

.center[.width-100[![](figures/MDP_words.png)]]

---

# How to reach the nest?

.center[.width-70[![](figures/chicken_env.png)]]

---

# Chicken-MDP

- State space: location $(x, y)$ where $x$ and $y \in [1,...,4]$.

    The initial state is $(1, 1)$.

- Action space: $[Left, Right, Up, Down]$.

- Transition function: $P(s_{t+1},s_t, u_t)$ always follow the action.

- Reward: 
    - +1 if it reaches the nest in (4, 4),
    - -1 if it reaches the fox in (2, 4),
    - 0 otherwhise.


---

# The optimal policy

For each $(x, y)$, we want to choose the best direction!

For example, the best action in $(3, 4)$ is to go up to reach the nest.

.center[.width-70[![](figures/chicken_env_policy.png)]]

---

# How good is the policy?

- If we know $P$ and $R$, it is easy to evaluate $\pi$.
- $\pi(s\_t)$ provides the action $u_t$ played $\forall t$.
- $P(s\_t, u\_t)$ provides $s_{t+1}$ reached after taking the action $u_t$.
- $R(s\_{t+1})$ provides the reward obtained in $s_{t+1}$.

- Repeat this from the initial state state $s_0$ to obtain the return of the policy
$$ G\_0 = \sum\_t r\_t  = r\_0 + r\_1 + .. + r\_t$$

--

- This is the .bold[state value function!]
$$ V^{\pi}(s) = \mathbb{E}\_{\pi} \left[ \sum\_t r\_t | s\_t = s \right] $$

    The expected sum of reward from the state $s$ by following the policy $\pi$.

---

# The optimal value functions

- We can evaluate policies given a state:

    $$ V^{\pi}(s\_t) = R(s\_{t+1}) + V^\pi(s\_{t+1})$$
    where $s\_{t+1} = P(s\_t, \pi(s\_t)) $

- The optimal policy $\pi^*$ is the one maximising the value function:

$$ \pi^* (s) = \arg\max\_\pi V^{\pi}(s)$$

- In (3, 4), going $Up$ has a .bold[higher] value function then the one going $Down$.

--

<br>
Can we do better?

---
# Consider action !

- Evaluate the return given a state $s$ and .bold[an action] $u$!
- This is the .bold[state action value function!]:

$$ Q^{\pi}(s\_t, u\_t) = R(s\_{t+1}) + V^\pi(s\_{t+1}) \text{ where } s\_{t+1} = P(s\_t, u\_t) $$

- We also have $Q^{\pi^*}(s, u) = \max_{\pi}Q^\pi(s, u)$

- Therefore, if we know $Q^{\pi^*}$, we obtain

$$ \pi^\* (s) =\arg\max\_u  Q^{\pi^{\*}}(s, u)$$

---
class: middle

.italic[How to learn the optimal policy .bold[without knowing] the MDP component?]

---

# Model based or model free?

- $P$ and $R$ are most of the time unknown.
- The agent interacts with the environment to learn!
- By playing, it is possible to learn the model.
- By trial and error, it is possible to learn the policy.
- How to learn a policy?

---

# Q Learning

- Learn directly the $Q$ function.
- Store $Q$ values for each state-action pair.
- Play and update:

$$ Q(s\_t, u\_t) \leftarrow Q(s\_t, u\_t) + \alpha \left[ r\_t + \gamma \max\_u Q(s\_{t+1}, u) - Q(s\_t, u\_t) \right] $$

--

- How agent selects action?
    - Exploit: $\pi^\* (s) =\arg\max\_u  Q^{\pi^{\*}}(s, u)$
    - Explore: plays randomly with a probability $\epsilon$.

--

- Limitations:
    - Store $|\mathcal{S}| * |\mathcal{U}|$ values.
    - What if $\mathcal{S}$ is continuous?


---

# Deep Q Network

Q-learning with a neural network parametrised by $\theta$.

- Minimise the loss function:

$$
\mathcal{L}(\theta) = \mathbb{E}\_{\langle s\_{t},u\_{t},r\_{t},s\_{t+1} \rangle \sim B} \big(r\_{t} + \gamma  \underset{u \in \mathcal{U}}{\max} Q(s\_{t+1}, u; \theta') - Q(s\_{t}, u\_{t}; \theta)\big)^2$$


- The replay buffer $B$ is a collection of transitions.
- Sampling transitions allows to update the network.
- $\theta'$ are the parameters of the target network, a copy of $\theta$ periodically updated.

--

- To play Atari games:
    - $\theta$ is a CNN.
- When the environment is partially observable (POMDP):
    - $\theta$ is a recurrent network (DRQN).
    - $B$ stores sequences of transitions.

---
# In practice

- Discount reward with $\gamma \in [0, 1]$ to encourage direct reward.

$$ G\_0 = \sum\_t \gamma^t r\_t  = r\_0 + \gamma r\_1 + .. + \gamma^t r\_t$$

- Transition function defines a probability to reach $s\_{t+1} \sim P(s\_{t+1}|s\_t, u\_t)$.

$$ V^{\pi}(s\_t) = \sum\_{s'} P(s'|s\_t, \pi(s\_t)) R(s') + \gamma V^\pi(s\_{t+1}) $$

- Stochastic policies $\pi(u|s)$ exist.

$$ V^{\pi}(s\_t) = \sum\_u \pi(u|s) \sum\_{s'} P(s'|s\_t, u) R(s') + \gamma V^\pi(s\_{t+1}) $$

- Value-based method is learning $Q(s, u;\theta)$.
- Policy-based method approximates directly $\pi(u|s;\theta)

---
Examples of RL nowadays:

- Chip design
- Drone control
- Content recommendation
- Supply chain management, warehouses
- Robotics

---
# Summary

- RL is an agent learning by .bold[interacting] in its .bold[environment] to maximise a reward.

- It is possible to evaluate policies when knowing the model.

- Model free RL learn by trial and error the policy.

- Value based method learn $Q(s, u)$ in order to select the best action $\arg\max\_u Q(s, u)$ in state $s$.

- It is possible to approximate $Q(s, u)$ or $\pi(u|s)$ with neural networks.


---
class: section

# Multi-Agent

# Reinforcement Learning


---
# Stochastic game 

Stochastic game (also referred to as Markov game) $[n, \mathcal{S}, O, \mathcal{Z}, \mathcal{U}, r, P, \gamma]$:

- A set of $n$ agents, each one is represented by $a$ or $a\_i, i \in \\{1,...,n\\}$.

- A set of states $s \in \mathcal{S}$.

- An observation function $O:\mathcal{S} \times \{1,...,n\} \rightarrow \mathcal{Z}$.

- A set of action spaces $\mathcal{U}={\mathcal{U}\_1} \times ... \times \mathcal{U}\_n$, one per agent $u^{a\_i}\_t \in \mathcal{U}\_i$.

- A transition function: $ s\_{t+1} \sim P( s\_{t+1} | s\_t , \mathbf{u\_t})$ with $\mathbf{u\_t} =(u^{a\_1}\_t, ..., u^{a\_n}\_t)$.

- A reward function per agent:  $r^{a\_i}\_t = R^{a\_i}(s\_{t+1}, s\_t, \mathbf{u\_t})$.

- Agents sometimes store their history $\tau^a\_t \in (\mathcal{Z} \times \mathcal{U})^t$.

- The *goal* of each agent $ a\_i $ is to maximize its total expected sum of (discounted) rewards 

<center> $\mathbb{E}_{\mathbf{\pi}} \left[ \sum_t \gamma^t r_t^{a_i} \right]$ with $\gamma \in [0, 1)$ </center>

---
# Challenges

- Non-stationarity: 
    
    Every agent are learning and each updates its policy.
- Finding and evaluating equilibrium:
    
    Does any agent benefits from chaning its policy?
- Credit assignment:
    
    The reward of an agent is function of all agents action.
- The number of agents: 
    
    $|\mathbf{u}|$ scales exponentionnaly with $n$. 
---

In Multi-agent settings, the goal of each agent may differ:


1. *Cooperative setting*: all agents share a common goal.

    Examples: traffic control, robotics teams,...

2. *Competitive setting*: the gain of an agent equals the loss of other agents.

    Often referred as zero-sum setting, because the sum of rewards of all agents sums to zero.

    Examples: 1v1 board games, 1v1 video games,...

3. *General sum setting*: lies in between the two others.

    Examples: everything else that is not cooperative or competitive, 5v5 video games,...

---
class: section
# Learn to Cooperate

---
# Dec-POMDP

In a cooperative setting, it is possible to have a single reward function, each agent receives a same global reward:
$$
r^{a\_1}\_t = r^{a\_n}\_t=r\_t = R(s\_{t+1}, s\_t, \mathbf{u\_t}): \mathcal{S}^2 \times \mathcal{U} \rightarrow \mathbb{R}
$$

Such Markov Games are called Decentralised-POMDP.

.center.width-85[![](figures/decpomdp.png)]

---
# Centralised controller

Centralised controller:
- One agent controls all actions.
- A single joint actions space $\mathcal{U}\_1 \times ... \times \mathcal{U}\_n$.

Problems:

--

- Joint actions space scales exponentially with $n$.
- What about the partial observability?
    - Not possible to centralise.

---

# Solutions:

- Decentralised controller.
    - Naive learner: train each agent with SARL methods.
- Centralised training with decentralised execution (CTDE). 
    - Benefit from supplementary information during training, such as the entire state of the game.



---

# Naive learner
Naive learning:

- Ignore the fact that there are multiple learning agents.
- Provide a first baseline to compare algorithms.
- Easy to implement.
- Not so young: a tabular version with IQL (Tan 1993).


Challenges:

--

- Non-stationarity:
    - Other agents are also learning and their policy changes over time.

- Credit assessment: 
    - How an agent learns whether its actions is the one that lead to good (or bad) reward?
    - How an agent maximises the joint actions reward knowing only its action?


---
# Value-based methods in CTDE

Independent Q-Learning (IQL):

- Each agent learns its individual $Q\_a(\tau^a\_t, u^a\_t)$ independently.



Problem:

--


- How to ensure that $\underset{u^a_t}{\arg\max} Q\_a(\tau^a\_t, u^a\_t)$ maximises $Q(s\_t, \mathbf{u\_t})$ ?


Solution: 

--

- Learn $Q(s\_t, \mathbf{u\_t})$ as a function of all $Q\_a(\tau^a\_t, u^a\_t)$ during training.


---

# Individual Global Max

Learn $Q(s\_t, \mathbf{u\_t})$ as a function of all $Q\_a(\tau^a\_t, u^a\_t)$ during training.

*Condition: Individual Global Max (IGM)*
$$
\underset{\mathbf{u\_t}}{\arg\max} Q(s\_t, \mathbf{u\_t}) 
=
\begin{pmatrix}
\underset{u^{a\_1}\_t}{\arg\max} Q\_1(\tau^{a\_1}\_t, u^{a\_1}\_t)\\\\
.\\\\
.\\\\
.\\\\
\underset{u^{a\_n}\_t}{\arg\max} Q\_n(\tau^{a\_n}\_t, u^{a\_n}\_t) 
\end{pmatrix}
$$


$Q_a$ is not a $Q$ function anymore, but a utility function used to select actions.

 

Question: How to satisfy IGM?

---

# VDN

How to satisfy IGM?

*Value Decomposition Network*:
$$
    Q(s\_t, \mathbf{u\_t}) = \sum\_{i=1}^n Q\_{a\_i}(\tau^{a\_i}\_t, u^{a\_i}\_t) 
$$

Problems:

--

- Addition does not allow to build complex functions.
- Current state information $s\_t$ is not considered.

How can we build non-linear factorisation satisfying IGM? 

---
# QMIX

How to build non-linear factorisation satisfying IGM? 


QMIX enforces monotonicity:

$$
    \frac{\partial Q(s\_t, \mathbf{u\_t})}{\partial Q\_{a}(\tau^{a}\_t, u\_t^{a})} \geq 0 \text{ } \forall a \in \{a\_1,..,a\_n\}
$$



How to build a non-linear monotonic factorisation of $Q(s\_t, \mathbf{u\_t})$ as a function of every  $Q\_a(\tau^{a}\_t, u^a\_t)$ and $s\_t$ with neural networks?

--

In QMIX, monotonicity is ensured by constraining a *hypernetwork* that computes the weights of a second neural network.

---

#QMIX architecture

.center.width-100[![](figures/figures_lesson/qmix_archi.png)]
 

.footnote[Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., Whiteson, S. (2018). Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning.]

---
# QMIX

In QMIX:

- <span style="color:red">A hypernetwork $h\_p$</span> takes the state $s\_t$ as input and computes the weights <span style="color:red">W1</span> and <span style="color:red">W2</span> of a  *second neural network*.
- <span style="color:red">These weights</span> are constrained to be positive and then used in a *feed forward network $h\_o$* to factorise $Q(s\_t,  \mathbf{u\_t})$ with the individual $Q\_a$.
- A neural network made of monotonic functions and strictly positive weights is monotonic with respect to its inputs.
$$\rightarrow Q\_{mix}(s\_t, \mathbf{u\_t}) = h\_o\left(Q\_{a\_1}(),..,Q\_{a\_n}(), h\_p(s\_t)\right)$$

The optimisation procedure follows the same principles of DQN algorithm:
$$
    \mathcal{L}(\theta) = \mathbb{E}\_{ \langle . \rangle \sim B }
    \bigg[  \big(r\_{t} + \gamma \underset{\mathbf{u} \in \mathcal{U}}{\max} Q\_{mix}(s\_{t+1}, \mathbf{u}; \theta') - Q\_{mix}(s\_{t}, \mathbf{u\_{t}}; \theta)\big)^{2} \bigg]
$$

Parameters of individual networks are commonly shared to speed up learning.




---
# QMIX results

.center.width-100[![](figures/figures_lesson/qmix_results.png)]

.footnote[Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., Whiteson, S. (2018). Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning.]




---
# DQV

Deep-Quality Value: learn $Q(.;\theta)$ and $V(.;\phi)$ at the same time.

$$
\mathcal{L}(\theta) = \mathbb{E}\_{\langle s\_t,u\_t,r\_t,s\_{t+1}\rangle\sim B} \bigg[ \big(r\_t + \gamma V(s\_{t+1}; \phi') - Q(s\_t, u\_t; \theta) \big)^{2} \bigg]
$$
$$
\mathcal{L}(\phi) = \mathbb{E}\_{\langle  s\_{t},u\_{t},r\_{t},s\_{t+1}  \rangle\sim B} \bigg[\big(r\_{t} + \gamma V(s\_{t+1}; \phi') - V(s\_{t}; \phi)\big)^{2}\bigg]
$$

One benefit of DQV:
- Reduce the overestimation problem of DQN, linked to the max operator.

DQN loss reminder:
$$
    \mathcal{L}(\theta) = \mathbb{E}\_{\langle s\_{t},u\_{t},r\_{t},s\_{t+1}\rangle \sim B}
    \bigg[ \big(r\_{t} + \gamma \max\_{u \in \mathcal{U}} Q(s\_{t+1}, u; \theta') - Q(s\_{t}, u\_{t}; \theta)\big)^{2}\bigg]
$$

.footnote[M. Sabatelli, G. Louppe, P. Geurts, and M. A. Wiering. Deep quality-value (DQV)
learning. 2018]
---
# QVMix
QVMix and QVMix-Max are extensions of the Deep-Quality value family of algorithms to cooperative multi-agent.

QVMix:
$$
    \mathcal{L}(\theta) = \mathbb{E}\_{\langle . \rangle \sim B}
    \bigg[\big(r\_{t} + \gamma V(s\_{t+1}; \phi') - Q(s\_{t}, \mathbf{u\_{t}}; \theta)\big)^{2}\bigg]
$$
$$
    \mathcal{L}(\phi) = \mathbb{E}\_{\langle . \rangle\sim B} 
    \bigg[\big(r\_{t} + \gamma V(s\_{t+1}; \phi') - V(s\_{t}; \phi)\big)^{2}\bigg]
$$

QVMix-Max:

$$
    \mathcal{L}(\phi) = \mathbb{E}\_{\langle . \rangle\sim B} 
    \bigg[\big(r\_{t} + \gamma \max\_{\mathbf{u} \in \mathcal{U}} Q(s\_{t+1}, \mathbf{u}; \theta') - V(s\_{t}; \phi)\big)^{2}\bigg]
$$

The architecture of $V$ and $Q$ are the same as that of $Q$ in QMIX, $V$ has a single output.


---
#QVMix results

.center.width-100[![](figures/figures_lesson/qvmix_results.png)]

.footnote[Leroy, P., Ernst, D., Geurts, P., Louppe, G., Pisane, J.,  Sabatelli, M. (2020). QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning.]


---
# QVMix overestimation bias

Estimated $Q$ are higher than the one obtained for QMIX and MAVEN.


.center.width-100[![](figures/figures_lesson/2m1z3s5zQ.png)]

.footnote[Leroy, P., Ernst, D., Geurts, P., Louppe, G., Pisane, J.,  Sabatelli, M. (2020). QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning.]

---

# QVMix conclusions

- A new CTDE value based method for Dec-POMDP.
- Achieve similar and better performance than QMIX.
- Reduce the overestimation bias.

---
class: section

# Infrastructure Management Planning

---

Infrastructure management planning (IMP) is an application for CTDE methods.

- Decide of inspection and repair actions over time to reduce costs and risks to maintain a system.

- Managing infrastructure can be a Dec-POMDP.

    - SARL: a single centralised controller,
    - MARL: one agent for each part to be inspected or repaired.


- This can be applied to any type of system: Civil, maritime, transportation, and urban management setting.

- In this work: simulated deterioration and off-shore wind turbines.

- Goals: 

    - Compare MARL against an expert rule-based heuristic policy (SOTA).
    - Study the scalability of CTDE methods (up to 100 agents).




.footnote[
P Leroy, PG Morato, J Pisane, A Kolios, D Ernst. IMP-MARL: a suite of environments for large-scale infrastructure management planning via MARL. 2023 
]


---

# IMP  representation

.center.width-100[![](figures/imp_intro.png)]

---

# IMP definition
- State and observations:
    - Belief on the deterioration state of each part.
    - Typically discredited probabilities on a crack size.
    - Belief is updated whether or not you inspected a component of the system. 

- Actions: 

    1. Do-nothing
    2. Inspect
    3. Replace/repair

- The return of an episode is $ \sum\_{t=0}^{T-1} \gamma^t \left[ R\_{t,f}+ \sum\_{a=1}^n \left({R\_{t,ins}^a} + {R\_{t,rep}^a}\right)+R\_{t,camp} \right]$.

    - Failure cost is $R\_f = c\_F \times p\_{Fsys}$ encompassing economic, environmental, and societal losses.



---

# IMP variations
.grid[
.kol-1-2[
.center.width-100[![](figures/environments_v2_a.png)A k-out-of-n system environment.]
]
.kol-1-2[
.center.width-100[![](figures/environments_v2_b.png)An offshore wind farm environment.]
]]

.grid[
.kol-1-2[
.center.width-100[![](figures/environments_v2_c.png)Uncorrelated and correlated initial damage distribution.]
]
.kol-1-2[
.center.width-100[![](figures/environments_v2_d.png)A campaign cost environment.]
]]


---

# IMP performance

Results are reported as boxplot against the expert rule-based heuristic policy.

.center.width-100[![](figures/plot_explain_plot_gaussian_2.png)Box plot construction for one setting with 10 seeds.]



---

Every boxplot gathers the best policies from each of 10 executed training realisations, indicating the 25th-75th percentile range, median, minimum, and maximum obtained results.

.center.width-100[![](figures/box_plot_small_dec.png)Performance plot with two different number of agents in three settings.]


Vertically arranging environments with an increasing number of $n$ agents

Note that the results are clipped at -100\%.


---

.center.width-100[![](figures/boxplot_perc_limit_up.png)]

---

.center.width-100[![](figures/boxplot_perc_limit_down.png)]

---

# Results:

- MARL-based strategies outperform expert-based heuristic policies.
- IMP challenge.
- Campaign cost environments.
- Centralised RL methods do not scale with the number of agents.
- IMP demands cooperation among agents.
- Overall, CTDE methods generate more effective IMP policies than the other investigated methods.

---

# IMP conclusions

 
- CTDE methods generally outperform heuristics.

- Centralised RL methods do not scale well with the number of agents.

- IMP environments demand cooperation among agents: CTDE $>>$ Decentralised.

- Remaining challenges: correlated environments and campaign costs.



---
class: section

# Training a team to compete

---
# Competition

- In a two player competition, gain of one agent is equal loss of the other.
- In AlphaGo, self-play with MCTS.
- In AlphaStar, population based training.
- The goal: train against many strategies.
- What if we do the same with two teams instead of two agents?

---
# Two team Markov game
Two symmetric teams:
- A set of $n$ agents, each represented by $a$ or $a\_{i, j}, i \in  \{1,...,n\}, j \in \{1, 2\}$.
- A set of states $s \in \mathcal{S}$.
- An observation function $O:\mathcal{S} \times \{1,...,n\} \rightarrow \mathcal{Z}$.
- A set of action spaces $\mathcal{U}=\mathcal{U}\_1 \times ... \times \mathcal{U}\_n$, one per agent $u^{a\_{i,j}}\_{t} \in \mathcal{U}\_i \forall j$.
- A transition function: $s\_{t+1} \sim P(s\_{t+1} | s\_t, \mathbf{u\_t})$,  $\mathbf{u\_t}=(u^{a\_{i,j}}\_t)\_{i \in \{1,..,n\};j \in \{1,2\}} $.
- A reward function per team:  $r^j\_t = R^j(s\_{t+1}, s\_t, \mathbf{u\_t})$.
- Agents store their history $\tau^a\_t \in (\mathcal{Z} \times \mathcal{U})^t$.
- The goal of each agent $a\_i$ is to maximize its total expected sum of (discounted) rewards $\sum\_{t=0}^{T} \gamma^t r^{a\_i}\_t$.



---
# Two team Markov game
How to train a team of agents to compete?

Train against multiple evolving strategies!

Teams are trained with CTDE methods: QMIX, QVMix and MAVEN.

Teams are trained with three different learning scenarios:
1. Against a stationary strategy (heuristic) like a Dec-POMDP.
2. Self-play, against itself, using transitions as both teams to train.
3. Within a population of five training teams.
    Uniform chance to play against itself or one of the four other team.

---
# The empirical study

- New Competitive StarCraft Multi-Agent Challenge.
- Teams trained with $10^7$ timesteps in 2 environments ($3m$ and $3s5z$).
- Each method/scenario pair has been executed 10 times.
- Evaluation is performed with Elo scores and win rates.

---
# Competitive StarCraft Multi-Agent Challenge

- SMAC is initially a Dec-POMDP environment based on StarCraft 2.
- Two teams learn to compete against each other.
- We transformed it in a competitive setting and train both teams.

<center><iframe width="450" height="300" src="https://www.youtube.com/embed/VZ7zmQ_obZ0" title="SMAC: The StarCraft Multi-Agent Challenge" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></center>

.footnote[https://github.com/oxwhirl/smac  Samvelyan, M., Rashid, T., De Witt, C. S., Farquhar, G., Nardelli, N., Rudner, T. G., ...  Whiteson, S. (2019). The starcraft multi-agent challenge.]



---
# Elo score

The Elo rating system assign each player of a population with a rating $R$ to rank them.

- We can compute the probability that a player will win against B.
- Let $R\_A$ and $R\_B$ be the ELO scores of player A and B, $E\_A$ = proba A wins.
$$
    E\_A=\frac{10^{R\_A/400}}{10^{R\_A/400} + 10^{R\_B/400}}
    \text{ and }
    E\_B=\frac{10^{R\_B/400}}{10^{R\_A/400} + 10^{R\_B/400}}
$$
--

- $400$ is a parameter. If the Elo score of player A is 400 points above that of B, it has a ten-times greater chance of defeating B.
- New score where $cst$ is a constant that defines the maximum possible update of the Elo score (10 in our paper, typically 32).
$$
    R'\_A = R\_A + cst \* (S\_A - E\_A)
$$
- $S\_A$ is equal to $1$ for a win, $0$ for a loss and $0.5$ for a draw.


---
# Two team Markov game: results

*H* = trained against heuristic,
*S* = self-play,
*P* = within a population,
*BP* = the 10 bests of each population.

Elo after training (Elo score obtained in different test populations) in the $3m$ map.
.center.width-100[![](figures/figures_lesson/2team1.png)]

.footnote[Leroy, P., Pisane, J., Ernst, D. (2022). Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition.]

---
# Two team Markov game: results
All together:
.center.width-80[![](figures/figures_lesson/2teams3.png)]

Conclusions:
- Training teams within a population of learning teams is the best learning scenario, when each team plays the same number of timesteps for training purposes.
- This is irrespective of whether or not the stationary strategy was better than all trained teams.
- A selection procedure is required in the same training population.

.footnote[Leroy, P., Pisane, J., Ernst, D. (2022). Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition.]

---
class: section

# Conclusions

---
- What is reinforcement learning.
- What are the challenges when several agents learn in the environment.
- How to train a team of agents to cooperate.
- How to train a team of agents to compete against several strategies.

---
class: end-slide, center
count: false

The end.
