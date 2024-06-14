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
# Intuition

.center[
<video controls preload="auto" height="500" width="700">
  <source src="./figures/chicken1.mp4" type="video/mp4">
</video>]

.footnote[Video credits: [Megan Hayes](https://twitter.com/PigMegan), [@YAWScience](https://twitter.com/YAWScience/status/1304199719036444672), 2020. Taken from [Intro to IA](https://github.com/glouppe/info8006-introduction-to-ai/) by Gilles Louppe.]

---
# Intuition

.center[
<video controls preload="auto" height="500" width="700">
  <source src="./figures/chicken2.mp4" type="video/mp4">
</video>]

.footnote[Video credits: [Megan Hayes](https://twitter.com/PigMegan), [@YAWScience](https://twitter.com/YAWScience/status/1304199719036444672), 2020. Taken from [Intro to IA](https://github.com/glouppe/info8006-introduction-to-ai/) by Gilles Louppe.]

---

# Markov decision process

.italic[RL is learning by .bold[interacting] in its .bold[environment] to maximise a reward.]

.center[![](figures/MDP.pdf)]


---


# This is a slide

Some text goes here.

- or
- here
- as 
- a
- list

.footnote[This is a footnote.]

---

class: middle

## This is a sub-title

This slide is centered vertically with `class: middle`.

---

# Some maths 

$$y = mx + b$$

---

# Some code 

~~~
def fibonacci(n):
    if n <= 1:
        return n 
    else: 
        return fibonacci(n-1) + fibonacci(n-2)
~~~

---

class: end-slide, center
count: false

The end.
