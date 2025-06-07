---
title: "Convex Combination in Machine Learning and Reinforcement Learning"
description: "An overview of convex combinations and their practical applications in optimization, deep learning, and RL."
pubDate: 2025-06-07T11:30:00Z
tags: ["ML", "RL", "convex optimization", "convex combination"]
---

## 1. What is a Convex Combination?

A **convex combination** of two elements $\(A\)$ and $\(B\)$ with a coefficient $\(r \in [0, 1]\)$ is defined as:

$$
x = (1 - r)A + rB
$$

- When $\(r = 0\)$, $\(x = A\)$
- When $\(r = 1\)$, $\(x = B\)$
- When $\(r = 0.5\)$, $\(x\)$ is the midpoint of $\(A\)$ and $\(B\)$

This formulation ensures $\(x\)$ lies on the line segment between $\(A\)$ and $\(B\)$, weighted by $\(r\)$.

---

## 2. Geometric and Mathematical Significance

Convex combinations appear in geometry, probability, optimization, and learning as a way to express interpolation between multiple points. They ensure:

- **Linearity**: The result lies within the convex hull.
- **Stability**: Values stay within bounds.

---

## 3. Applications in Machine Learning

### 3.1 Polyak Averaging

In optimization:

\[
\theta_{\text{target}} \leftarrow (1 - \tau)\theta_{\text{target}} + \tau\theta
\]

Used in DDPG, TD3 to smooth parameter updates and reduce variance.

### 3.2 Momentum in SGD

$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_t)
$$

Momentum uses convex averaging of gradients.

### 3.3 Attention Mechanisms

$$
\text{Attention}(Q, K, V) = \sum_i \alpha_i V_i,\quad \text{where } \sum_i \alpha_i = 1,\ \alpha_i \geq 0
$$

Attention weights are convex coefficients over value vectors.

---

## 4. Applications in Reinforcement Learning

### 4.1 Bellman Backup

$$
V(s) \leftarrow (1 - \alpha)V(s) + \alpha [r + \gamma V(s')]
$$

This updates the value estimate with a convex blend of old and new info.

### 4.2 Target Network Updates

$$
\theta_{\text{target}} \leftarrow (1 - \tau)\theta_{\text{target}} + \tau\theta
$$

Slows target changes to stabilize training (e.g., in DQN, TD3, SAC).

### 4.3 Mixture Policies

$$
\pi(a|s) = \sum_i w_i \pi_i(a|s),\quad \text{with } \sum w_i = 1
$$

Policy ensembles combine diverse behaviors using convex weighting.

### 4.4 Rollout Initialization

Model rollouts may begin from convex combinations of real and simulated states to control bias and variance in planning.

---

## 5. Intuition: Why Use Convex Combinations?

- Enforces **boundedness** and **interpolation**
- Enables **smooth updates** and **variance reduction**
- Aligns with **Bayesian averaging** and probabilistic mixtures
- Helps in **stabilizing** non-stationary training (e.g., target networks)

---

## 6. Visual Intuition

Interpolating between vectors $\(A\)$ and $\(B\)$ via a convex combination traces a line segment. This generalizes to:

$$
x = \sum_i \alpha_i x_i,\quad \alpha_i \geq 0,\ \sum_i \alpha_i = 1
$$

Which defines a point within the **convex hull** of the inputs.

---

## 7. References

- Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging.
- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (DDPG), 2016.
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (TD3), 2018.
- Vaswani et al., "Attention is All You Need", 2017.
- Mnih et al., "Human-level control through deep reinforcement learning" (DQN), 2015.
- Haarnoja et al., "Soft Actor-Critic", 2018.

---

*Written with espresso and entropy in mind.* ☕🌀
