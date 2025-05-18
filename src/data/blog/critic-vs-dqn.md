---
title: "Critic and DQN"
pubDatetime: 2025-05-18T09:00:00Z
description: "Critic 在 Actor-Critic 中是个打分机器，而在 DQN 里它直接拿 Q 值做决策。"
tags: [RL, Actor-Critic, DQN,]
---



---

## DQN：

DQN is a value-based method. Q neural nets  outoputs state value or state action value. 

Then we try to find the action(phi) to output the max value.
```math
a = \arg\max_a Q(s, a)
```



---

## Actor-Critic：


* Actor neural nets output action distributions $\pi(a|s)$，可能是sample()也可能是argmax.
* Critic neural nets output $V(s)$ or $Q(s, a)$，
* then we update the actor neural networks depends on the outputs (state values, q-values or advantages) from critic.






