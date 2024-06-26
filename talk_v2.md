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

- What happens when multiple agents learn together?
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

???

To provide some intuitions, let us watch this video of a chicken trained to pick the pink paper.

Everytime the chicken 

---
class: middle

.center[
<video controls preload="auto" height="500" width="700">
  <source src="./figures/chicken2.mp4" type="video/mp4">
</video>]

.footnote[Video credits: [Megan Hayes](https://twitter.com/PigMegan), [@YAWScience](https://twitter.com/YAWScience/status/1304199719036444672), 2020. Taken from [Intro to IA](https://github.com/glouppe/info8006-introduction-to-ai/) by Gilles Louppe.]

???

Let us look at another example with a cone.

The chicken has to circle around it to receive a reward.

You see here, it understands that the reward is in the hand.

But it has to figure out that it needs to circle around the cone!

And even if it succeeded, it can still fail.

---

class: middle

An agent is learning by .bold[interacting] in its .bold[environment] to maximise a reward.

<br>

.center[.width-100[![](figures/MDP_words.png)]]

.footnote[R. S. Sutton and A. G. Barto. Reinforcement learning, second edition: An introduction. 2018]

<!-- ---

An .bold[MDP] is defined by a tuple ($\mathcal{S}$, $\mathcal{U}$,${P}$,${R}$):


- State space $s_t \in \mathcal{S}$.
- Action space $u_t \in \mathcal{U}$.
- Transition function $s\_{t+1} = P(s\_t, u\_t)$ defines the state $s_{t+1}$ reached after taking the action $u_t$ in state $s_t$.
- Reward function $r\_t=R(s\_{t+1})$ defines the reward obtained.

.center[.width-100[![](figures/MDP_words.png)]] -->

---

# Chicken environment

.center[States $s$: chicken location $(x, y)$.]

.center[.width-80[![](figures/chicken_env.png)]]

---

# Chicken environment

.center[States $s$: chicken location $(x, y)$.]

.center[.width-80[![](figures/chicken_env.png)]]


---

# Chicken environment

.center[Actions $u$: $[Left, Right, Up, Down]$.]

.center[.width-80[![](figures/chicken_env.png)]]


---

# Chicken environment

.center[
Transition function: $s_{t+1} = P(s_t, u_t)$ is deterministic.
]

.center[.width-80[![](figures/chicken_env.png)]]

---

# Chicken environment

.center[
Reward function: $+1$ if $s = (4,4)$, $-1$ if $s = (2, 4)$, $0$ otherwise.
]

.center[.width-80[![](figures/chicken_env.png)]]


---

# Goal ?

For each $s=(x, y)$, we want to choose the .bold[best action!]


.center[.width-80[![](figures/chicken_env_policy.png)]]

---

class: middle

The agent learns the policy $\pi:\mathcal{S}\to\mathcal{U}$ that maximises

$$\mathbb{E}_{\pi} \left[ \sum_t \gamma^t r_t \right].$$

The discount factor $\gamma \in [0, 1]$ encourages direct reward.

???
How can I know which is the best action?

---
class: middle

$\sum_t \gamma^t r_t$ if $\gamma=0.9$:
- Green:  $0+0+0+0+0+0.9^6$,
- Yellow: $0+0+0+0+0+0+0+0.9^8$.

.center[.width-70[![](figures/chicken_env_discount.png)]]
---
class: middle

In a state $s$, after playing an action $u$, what is the best sum of discounted rewards?

--

.center[.width-80[![](figures/chicken_env_policy.png)]]

???
Show on the environment from (2, 3) that going right and up?

---
class: middle

In state $s_t$, after playing action $u_t$, what is the best $\sum_t \gamma^t r_t$ possible?

--

$$ r\_t + \text{the best sum of discounted rewards from } s_{t+1}$$

--

$$ r\_t + \max\_{s\_{t+1}} \big[ \gamma R(s\_{t+1}) + \text{the best sum of discounted rewards from } s_{t+2} \big] $$

--

$$ r\_t + \max\_{s\_{t+1}} \big[ \gamma R(s\_{t+1}) + \max\_{s\_{t+2}} \big[ \gamma^2 R(s\_{t+2}) + \text{the best sum of discounted rewards from } s_{t+3} \big] $$

--

$$ r\_t + \gamma \max\big[  r\_{t+1} + \gamma^2 \max \big[  r\_{t+2} + .... \big] \big] $$


---
class: middle

# State action value functions

For each state and action ($s_t, u_t$), the best $\sum_t \gamma^t r_t$ possible is the Q function:

$$ Q(s\_t, u\_t) = r\_t + \gamma \max\_{u'} Q(s\_{t+1}, u').$$

--

The optimal policy is

$$ \pi^\* (s_t) =\arg\max\_u  Q(s_t, u) .$$

???
But how to learn them?

---

# Q Learning

- Start with initial Q values to 0:

$$ \forall (s, u),  Q(s, u) = 0$$

--

- Play and update:

$$ Q(s\_t, u\_t) \leftarrow Q(s\_t, u\_t) + \alpha \left[ r\_t + \gamma \max\_u Q(s\_{t+1}, u) - Q(s\_t, u\_t) \right]$$


---

# Epsilon greedy

How agent selects action?
- Take the best action most of the time
    
$$\pi^\* (s) =\arg\max\_u  Q(s, u),$$

- sometimes, with a probability $\epsilon$, take a random one!

--

# Exploration vs exploitation!

---

# Chicken Q values:

How much updates to know $Q((3,4), Up) = 1$ and $Q((3,4), Down) = -1$?

.center[.width-70[![](figures/chicken_env_discount.png)]]

???
If no luck, never!

---
class: middle

Limitations
- Need to explore a lot!
- Store $|\mathcal{S}| * |\mathcal{U}|$ values.
    - Chicken MDP: $16 \times 4$
    - If fox can be anywhere: $16 \times 16 \times 4$
    - If fox and nest can be anywhere:  $16 \times 16 \times 16 \times 4$
    - If diagional actions: $16 \times 16 \times 16 \times 8$
    - If grid is larger: $10000 \times 10000 \times 10000 \times 8$
- What if $\mathcal{S}$ is continuous?

---
class: middle

# Deep Q Network

Q-learning with a neural network parametrised by $\theta$.

Minimise the loss

$$
\mathcal{L}(\theta) = \mathbb{E}\_{\langle s\_{t},u\_{t},r\_{t},s\_{t+1} \rangle \sim B} \big(r\_{t} + \gamma  \underset{u \in \mathcal{U}}{\max} Q(s\_{t+1}, u; \theta') - Q(s\_{t}, u\_{t}; \theta)\big)^2$$


???
- The replay buffer $B$ is a collection of transitions.
- Sampling transitions allows to update the network.
- $\theta'$ are the parameters of the target network, a copy of $\theta$ periodically updated.
- To play Atari games:
    - $\theta$ is a CNN.
- When the environment is partially observable (POMDP):
    - $\theta$ is a recurrent network (DRQN).
    - $B$ stores sequences of transitions.

---
# In practice

Transition function defines a probability to reach $s\_{t+1} \sim P(s\_{t+1}|s\_t, u\_t)$.

$$ Q(s\_t, u\_t) = r\_t + \gamma \sum\_{s\_{t+1}} P(s\_{t+1}|s\_t, u') \max\_{u'} Q(s_{t+1}, u')$$

--

.bold[Markov Decision Process (MDP).]

.center[.width-100[![](figures/MDP_words.png)]]

???
Talk about the markov property.

---
# In practice

We evaluate any policy $\pi$ with the .bold[state value function] $V$:

$$ V(s)=\mathbb{E}\_{\pi}\left[r\_t + \gamma V(s\_{t+1})|s\_t=s\right] $$ 

--



- .bold[Value-based method] is learning $Q(s, u;\theta)$.
    - Most of the time used for deterministic policy $\pi = \arg\max\_u Q(s, u)$.

--

- .bold[Policy-based method] approximates directly $\pi(u|s;\theta)$.
    - Allows stochastic policies $\pi(u|s)$.

--

- .bold[Partial observability] in real-world.
    - Agent has an observation $o$ of $s$.
    - $\pi(u|\tau)$, where $\tau$ is action-observation history.
    - Recurrent neural network.


---
# Examples of RL application

- Chip design
- Drone control
- Content recommendation
- Supply chain management, warehouses
- Robotics

---
# Summary

- In RL, an agent learns by interacting in its environment to maximise a reward.

- Value based method provides $Q(s, u)$, the best sum of discounted rewards.

- The agent select the best action $\arg\max\_u Q(s, u)$ in state $s$.

- It is possible to approximate $Q(s, u)$ or $\pi(u|s)$ with neural networks.


---
class: section

# Multi-Agent

# Reinforcement Learning

???
What changes if the fox can move and also learn?


---

# Stochastic game

All agents learn a policy $\pi^{a}$ to maximise

$$ \mathbb{E}\_\pi \left[ \sum\_t \gamma^t r_t^{a\_i} \right] \text{ where }\pi = { \pi^{a\_1}, .. ,\pi^{a\_n} \} $$

.center[.width-100[![](figures/SG.png)]]


???
We still have the transition

---
class: middle

# **.bold[Rewards depend on actions of all agents]**

---

# Different goals


1. .bold[Cooperation]

--

2. .bold[Competition]

--

3. .bold[General sum]

---
class: section

# Learn to cooperate

---
# Decentralised Partially Observable Markov Decision Process (Dec-POMDP)

A single global reward

$$
r^{a\_1}\_t = r^{a\_n}\_t=r\_t = R(s\_{t+1}, s\_t, \mathbf{u\_t}): \mathcal{S}^2 \times \mathcal{U} \rightarrow \mathbb{R}
$$



.center.width-85[![](figures/decpomdp.png)]

---
class: middle

# Robotic warehouses

.center.width-85[![](figures/rware.gif)]

.footnote[Add citation]

---
class: middle

# Multi-Agent Tracking

.center.width-65[![](figures/mate.gif)]

.footnote[Add citation]

---
class: middle

# StarCraft Multi-Agent Challenge

<center><iframe width="450" height="300" src="https://www.youtube.com/embed/VZ7zmQ_obZ0" title="SMAC: The StarCraft Multi-Agent Challenge" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></center>

.footnote[https://github.com/oxwhirl/smac  Samvelyan, M., Rashid, T., De Witt, C. S., Farquhar, G., Nardelli, N., Rudner, T. G., ...  Whiteson, S. (2019). The starcraft multi-agent challenge.]

---
class: middle

# StarCraft Multi-Agent Challenge

3M

.center.width-75[![](figures/3m.png)]

3S5Z

.center.width-75[![](figures/3s5z.png)]

.footnote[https://github.com/oxwhirl/smac  Samvelyan, M., Rashid, T., De Witt, C. S., Farquhar, G., Nardelli, N., Rudner, T. G., ...  Whiteson, S. (2019). The starcraft multi-agent challenge.]

---
class: section

# How to train a team to cooperate?


---
# Centralised controller


.grid[
.kol-2-5[
- Learn $Q(s\_t, \mathbf{u\_t})$.

.bold[Problems:]

- $|\mathcal{U}\_1 \times ... \times \mathcal{U}\_n|$.

- Partial observability?

]
.kol-3-5[
.center.width-100[![](figures/central_control.png)]
]
]

---

#  Decentralised controller

.grid[
.kol-2-5[
- Independent Q-Learning: learn $Q\_a(\tau^a\_t, u^a\_t)$

.bold[Problems:]

- Non-stationarity

- Credit assessment


]
.kol-3-5[
.center.width-100[![](figures/dec_control.png)]
]
]


---

# Centralised training with <br> decentralised execution

It is possible to learn $Q(s\_t, \mathbf{u\_t})$ during training.

- Training in simulator.

- We known $s$ at training.

- We have access to all actions.


---

# CTDE Value-based methods

Only $Q\_a(\tau^a\_t, u^a\_t)$ during the execution.

--


.bold[GOAL]: $\underset{u^a_t}{\arg\max} Q\_a(\tau^a\_t, u^a\_t)$ maximises $Q(s\_t, \mathbf{u\_t})$.

--

.bold[Solution]: Learn $Q(s\_t, \mathbf{u\_t})$ as a function of all $Q\_a(\tau^a\_t, u^a\_t)$ .


---

# Individual Global Max

Learn $Q(s\_t, \mathbf{u\_t})$ as a function of all $Q\_a(\tau^a\_t, u^a\_t)$ during training.

.bold[Individual Global Max:]
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

???

$Q_a$ is not a $Q$ function anymore, but a utility function used to select actions.

 

Question: How to satisfy IGM?

---

# Value Decomposition Network
$$
    Q(s\_t, \mathbf{u\_t}) = \sum\_{i=1}^n Q\_{a\_i}(\tau^{a\_i}\_t, u^{a\_i}\_t) 
$$

--

The optimisation procedure follows the Deep Q Network algorithm
$$
    \mathcal{L}(\theta) = \mathbb{E}\_{ \langle . \rangle \sim B }
    \bigg[  \big(r\_{t} + \gamma \underset{\mathbf{u} \in \mathcal{U}}{\max} Q(s\_{t+1}, \mathbf{u}; \theta') - Q(s\_{t}, \mathbf{u\_{t}}; \theta)\big)^{2} \bigg]
$$

--

Limitations

- Addition is not complex.
- Where is $s\_t$? 

---
# QMIX

We want non-linear factorisation of $Q(s\_t, \mathbf{u\_t})$!

.footnote[Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., Whiteson, S. (2018). Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning.]

--

$$
    \frac{\partial Q(s\_t, \mathbf{u\_t})}{\partial Q\_{a}(\tau^{a}\_t, u\_t^{a})} \geq 0 \text{ } \forall a \in \{a\_1,..,a\_n\}
$$

--

.center.width-50[![](figures/qmix.png)]


---
# Results in 3M

.center.width-50[![](figures/qmix_fig/base_legend.png)]
.center.width-80[![](figures/qmix_fig/base.png)]

.footnote[Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., Whiteson, S. (2018). Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning.]


---
# Deep-Quality Value


Deep Q Network:
$$
    \mathcal{L}(\theta) = \mathbb{E}\_{\langle s\_{t},u\_{t},r\_{t},s\_{t+1}\rangle \sim B}
    \bigg[ \big(r\_{t} + \gamma \max\_{u \in \mathcal{U}} Q(s\_{t+1}, u; \theta') - Q(s\_{t}, u\_{t}; \theta)\big)^{2}\bigg]
$$

Motivation:

$$ V(s\_{t+1}) =\max\_{u} Q(s\_{t+1}, u; \theta')$$

---
# Deep Quality-Value

Learn $Q(.;\theta)$ and $V(.;\phi)$ at the same time.

$$
\mathcal{L}(\theta) = \mathbb{E}\_{\langle s\_t,u\_t,r\_t,s\_{t+1}\rangle\sim B} \bigg[ \big(r\_t + \gamma V(s\_{t+1}; \phi') - Q(s\_t, u\_t; \theta) \big)^{2} \bigg]
$$
$$
\mathcal{L}(\phi) = \mathbb{E}\_{\langle  s\_{t},u\_{t},r\_{t},s\_{t+1}  \rangle\sim B} \bigg[\big(r\_{t} + \gamma V(s\_{t+1}; \phi') - V(s\_{t}; \phi)\big)^{2}\bigg]
$$

--

Benefit :
- Reduce the overestimation problem of DQN.

.footnote[M. Sabatelli, G. Louppe, P. Geurts, and M. A. Wiering. Deep quality-value (DQV)
learning. 2018]

---
class: middle

# Contribution:

# Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning.

---
# QVMix and QVMix-Max

Take the architecture of $Q$ in QMIX to compute both $Q$ and $V$. 

--

.bold[QVMix:]

$$
    \mathcal{L}(\theta) = \mathbb{E}\_{\langle . \rangle \sim B}
    \bigg[\big(r\_{t} + \gamma V(s\_{t+1}; \phi') - Q(s\_{t}, \mathbf{u\_{t}}; \theta)\big)^{2}\bigg]
$$
$$
    \mathcal{L}(\phi) = \mathbb{E}\_{\langle . \rangle\sim B} 
    \bigg[\big(r\_{t} + \gamma V(s\_{t+1}; \phi') - V(s\_{t}; \phi)\big)^{2}\bigg]
$$

.bold[QVMix-Max:]

$$
    \mathcal{L}(\phi) = \mathbb{E}\_{\langle . \rangle\sim B} 
    \bigg[\big(r\_{t} + \gamma \max\_{\mathbf{u} \in \mathcal{U}} Q(s\_{t+1}, \mathbf{u}; \theta') - V(s\_{t}; \phi)\big)^{2}\bigg]
$$



---
#QVMix results

.grid[

.kol-1-2[
.center.width-100[3M![](figures/qvmix_3m.png)]
]
.kol-1-2[
.center.width-100[3S5Z![](figures/qvmix_3s5z.png)]
]]
.center.width-100[![](figures/qvmix_legend.png)]


.footnote[Leroy, P., Ernst, D., Geurts, P., Louppe, G., Pisane, J.,  Sabatelli, M. (2020). QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning.]


---
# QVMix overestimation bias

.center.width-90[![](figures/qvmix_overstim.jpg)]
.center.width-100[![](figures/qvmix_overestim_leg.jpg)]


.footnote[Leroy, P., Ernst, D., Geurts, P., Louppe, G., Pisane, J.,  Sabatelli, M. (2020). QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning.]

---

# QVMix and QVMix-Max

- Learning $V$ as a target for learning $Q$.

- New value based methods for Dec-POMDP:
    - Indepent learner: IQV and IQV-Max
    - CTDE: QVMix and QVMix-Max

- Achieve similar and better performance than QMIX.

- Reduce the overestimation bias.

---
class: section

# Contribution

# IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL

---

# Infrastructure management planning

- Maintain a system composed of different parts.
    - Bridges
    - Wind farms

--

- .bold[TASK:] Decide which component needs to be inspected or repaired.

--

- .bold[GOAL:] Reduce maintenance cost and failure risks.

--

- .bold[CONTRIBUTION:]
    - Today, solved with expert-based heuristic.
    - Can (MA)RL solve this problem?
    - Do CTDE methods scale well?



---
class: middle

.center.width-100[![](figures/imp_intro.png)]


.footnote[
P Leroy, PG Morato, J Pisane, A Kolios, D Ernst. IMP-MARL: a suite of environments for large-scale infrastructure management planning via MARL. 2023 
]


???


- State and observations:
    - Probability on the deterioration state of each part.

- Actions: 

    1. Do-nothing
    2. Inspect
    3. Replace/repair

- Maximise the reward:

$$ \sum\_{t=0}^{T-1} \gamma^t \left[ R\_{t,f}+ \sum\_{a=1}^n \left({R\_{t,ins}^a} + {R\_{t,rep}^a}\right)+R\_{t,camp} \right]$$

- Failure cost is $R\_f = c\_F \times p\_{Fsys}$.

encompassing economic, environmental, and societal losses

---
# A k-out-of-n system

- Fail if less than 4-out-of-5 components work.

- In our work, simulated deterioration of a crack size.

- No correlation between parts.

.width-100[![](figures/environments_v2_a.png)]

.footnote[
P Leroy, PG Morato, J Pisane, A Kolios, D Ernst. IMP-MARL: a suite of environments for large-scale infrastructure management planning via MARL. 2023 
]


---

# How to solve with (MA)RL?

- One agent decide for one part of the system.

- They all cooperate to maximise the reward.

$$ \sum\_{t=0}^{T-1} \gamma^t \left[ R\_{t,f}+ \sum\_{a=1}^n \left({R\_{t,ins}^a} + {R\_{t,rep}^a}\right)+R\_{t,camp} \right]$$

- .bold[GOAL]: Can (MA)RL be better than the heuristic?

---

# How better than the heuristic?

Normalised boxplot against the expert rule-based heuristic policy.

.center.width-60[![](figures/plot_explain_top.png)]
.center.width-60[![](figures/plot_explain_bottom.png)]

---

# QMIX in the k-out-of-n system.

- $n$: from 3 to 100 agents.
- $H$: the score of the heuristic.


.center.width-100[![](figures/boxplot_perc_limit_up_qmix.png)]

---
.center.width-100[![](figures/box_plot_legend.png)]
.center.width-40[![](figures/boxplot_perc_limit_up_koutofn.png)]

???
Better than heuristic

Less better when n is big

huge variance with large n

independent does not work

centralised work well (but more parameter)



---

# Correlated initial damage distribution

.center.width-100[![](figures/environments_v2_c.png)]

---

# Offshore wind farm 

.center.width-100[![](figures/environments_v2_b.png)]

---

# Campaign costs

.center.width-100[![](figures/environments_v2_d.png)]


---
class: middle

.center.width-60[![](figures/boxplot_perc_limit_up.png)]
.center.width-60[![](figures/boxplot_perc_limit_down.png)]


---

# IMP-MARL

- Six new real-world environments.

- CTDE methods can perform better than the expert-based heuristic.

- IMP environments demand cooperation among agents: 
    CTDE $>>$ Decentralised.

--

- Remaining challenges:
    - Scalability.
    - Correlated distribution. 
    - Campaign costs.


---
class: section

# How to train a team to compete against others?

---
# Competition

- In a two player competition:
    - Agents need to be good against many strategies.
    - Agents should adapt!

--

- How to play chess?

Agent learns by playing against itself with MCTS - AlphaGo

--

- How to play StarCraft 2?

A population of agents play against each other to face many strategies - AlphaStar

--

.bold[*Can we use these learning scenario to train teams with CTDE methods?*]

---

class: section

# Contribution


# Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition

---
# Two team Markov game

- Two symmetric teams, same agents face each other.

- 2 rewards, one per team $r^j\_t = R^j(s\_{t+1}, s\_t, \mathbf{u\_t})$.

- The goal of each agent $a\_i$ is to maximize its total expected sum of (discounted) rewards $\sum\_{t=0}^{T} \gamma^t r^{a\_i}\_t$.

New Competitive StarCraft Multi-Agent Challenge

.center.width-50[3M![](figures/3m.png)]

.center.width-50[3S5Z![](figures/3s5z.png)]


---
# The empirical study

Teams are trained with CTDE methods: QMIX, QVMix and MAVEN.

--

Teams are trained with three different learning scenarios:

--

1. Against a stationary strategy (.bold[heuristic]) like in a Dec-POMDP.

--

2. .bold[Self-play], against itself, using all played transitions to train.

--

3. Within a .bold[population] of five training teams.
    - Uniform chance to play against any of the five teams.

---
class: middle

# Which team is the best?

.bold[GOAL:] Be good against many strategies.

Create groups of teams evaluated together.

---
# Elo score

Assign a rating $R$ to compute the probability of winning.

--

- Let $R\_A$ and $R\_B$ be the ELO scores of player A and B.
- Probabilities:

$$
    E\_A=\frac{10^{R\_A/400}}{10^{R\_A/400} + 10^{R\_B/400}}
$$

$$
    E\_B=\frac{10^{R\_B/400}}{10^{R\_A/400} + 10^{R\_B/400}}
$$

--

Update

$$
    R'\_A = R\_A + cst \* (S\_A - E\_A)
$$
- $S\_A$ is equal to $1$ for a win, $0$ for a loss and $0.5$ for a draw.

???

- $400$ is a parameter. If the Elo score of player A is 400 points above that of B, it has a ten-times greater chance of defeating B.

- New score where $cst$ is a constant that defines the maximum possible update of the Elo score (10 in our paper, typically 32).

---

First, we group teams by training method.

- .bold[H] = trained against heuristic,
- .bold[S] = self-play,
- .bold[P] = within a population,
- .bold[BP] = the 10 bests of each population.

QVMix in the 3M map:

.center.width-70[![](figures/popu_qvmix.png)]


---
class: middle

.center.width-100[![](figures/figures_lesson/2team1.png)]

.footnote[Leroy, P., Pisane, J., Ernst, D. (2022). Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition.]

---
class: middle

All together: same conclucion!

.center.width-80[![](figures/figures_lesson/2teams3.png)]

.footnote[Leroy, P., Pisane, J., Ernst, D. (2022). Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition.]

---

# 3S5Z

What if the heuristic is the best?

.center.width-80[![](figures/3s5z_tiny_all_h_clean.png)]



.footnote[Leroy, P., Pisane, J., Ernst, D. (2022). Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition.]

---

# Conclusions

- Training teams within a population of learning teams is the best learning scenario.

- This is irrespective of whether or not the stationary strategy was better than all trained teams.

- A selection procedure is required in the same training population.


---
class: section

# Conclusions

---
# Conclusions

- What is (multi-agent) reinforcement learning.

- How to train a team of agents to cooperate.

- How to train a team of agents to compete against several strategies.

--

# Reinforcement Learning is not DEAD !

---
class: end-slide, center
count: false

Thank you.
