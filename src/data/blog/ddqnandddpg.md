---

title: "DDQN vs. DDPG: Understanding Two Powerful Off-Policy RL Algorithms"
pubDatetime: 2025-05-19T14:36:00Z
description: "A clear side-by-side comparison between Double Deep Q-Networks (DDQN) and Deep Deterministic Policy Gradient (DDPG), highlighting the key differences in structure, use cases, and action spaces."
tags: [RL, DDQN, DDPG, Actor-Critic, Q-Learning, Deep RL]
----------------------------------------------------------

When working with off-policy reinforcement learning algorithms, two popular choices often arise: **Double DQN (DDQN)** and **Deep Deterministic Policy Gradient (DDPG)**. Although they share some core ideas like using target networks and replay buffers, they diverge significantly in how they handle actions and policy learning.

---

## 🔍 At a Glance

| Feature              | **DDQN**              | **DDPG**                               |
| -------------------- | --------------------- | -------------------------------------- |
| Full Name            | Double Deep Q-Network | Deep Deterministic Policy Gradient     |
| Action Space         | Discrete              | Continuous                             |
| Policy (Actor)?      | ❌ No (implicit)       | ✅ Yes (explicit)                       |
| Q-function (Critic)? | ✅ Yes                 | ✅ Yes                                  |
| Action Selection     | $\arg\max_a Q(s, a)$  | $a = \pi(s)$                           |
| Policy Gradient?     | ❌ No                  | ✅ Yes                                  |
| Exploration          | ε-greedy              | Noise added to actor output (e.g., OU) |
| Target Networks      | ✅ Q target            | ✅ Actor + Critic targets               |
| Based On             | Q-learning            | Deterministic Policy Gradient          |
| On/Off-policy        | Off-policy            | Off-policy                             |

---

## 🧠 Core Conceptual Differences

### 🔹 DDQN: Discrete Value-Based Learning

* Improves upon DQN by reducing overestimation via Double Q-learning:

```math
y = r + \gamma Q_{\phi'}(s', \arg\max_{a'} Q_\phi(s', a'))
```

* Learns only a Q-network: $Q(s, a)$
* No explicit policy; policy is derived from $\arg\max_a Q(s, a)$

---

### 🔸 DDPG: Continuous Actor-Critic Learning

* Learns two networks:

  * **Critic**: $Q(s, a)$
  * **Actor**: $\pi(s)$

* Actor is updated via policy gradient:

```math
\nabla_\theta J = \mathbb{E}[ \nabla_a Q(s, a) |_{a = \pi(s)} \cdot \nabla_\theta \pi(s) ]
```

* Can handle **continuous action spaces** efficiently
* Target networks for both actor and critic are softly updated via EMA:

```math
\phi' \leftarrow \tau \phi + (1 - \tau) \phi'
```

---

## 🤖 When to Use Which?

| Use Case                            | Recommended Algorithm |
| ----------------------------------- | --------------------- |
| Discrete action space (e.g., Atari) | DDQN                  |
| Continuous control (e.g., robotics) | DDPG                  |
| Simple greedy policies              | DDQN                  |
| Smooth, differentiable policies     | DDPG                  |

---

## 📌 Summary

| Concept             | DDQN           | DDPG                       |
| ------------------- | -------------- | -------------------------- |
| Architecture        | Value-only     | Actor-Critic               |
| Max Action Strategy | Hard-coded max | Learned by actor           |
| Exploration         | ε-greedy       | Stochastic noise           |
| Gradient Flow       | Critic only    | Actor from critic gradient |
| Action Type         | Discrete       | Continuous                 |

---

## 📖 References

* Mnih et al., *“Human-level control through deep reinforcement learning”*, Nature 2015
  [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)

* Hasselt et al., *“Deep Reinforcement Learning with Double Q-learning”*, AAAI 2016
  [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)

* Lillicrap et al., *“Continuous control with deep reinforcement learning”*, ICLR 2016
  [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)
