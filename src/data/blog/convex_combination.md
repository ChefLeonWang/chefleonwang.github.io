---
title: "Smooth trasition: Convex Combination and other methods"
pubDatetime: 2025-06-07T11:30:00Z
description: "An overview of convex combinations and their practical applications in optimization, deep learning, and RL."

tags: [ML, RL, convex optimization, convex combination]
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

$$
\theta_{\text{target}} \leftarrow (1 - \tau)\theta_{\text{target}} + \tau\theta
$$

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

## 5. Other Smooth Transition Techniques

### 5.1 Exponential Moving Average (EMA)

$\theta_t^{\text{EMA}} = \beta \theta_{t-1}^{\text{EMA}} + (1 - \beta) \theta_t$
Used in parameter smoothing and stability, e.g., Mean Teacher, MoCo.

### 5.2 Soft vs. Hard Update

* **Soft Update**: Smooth transition using convex combination
* **Hard Update**: Replace entire target parameters at intervals

### 5.3 KL Penalty / Trust Region

Used in TRPO and PPO to prevent large policy shifts:

$$
\max_\theta \mathbb{E}_{s,a} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s,a) \right] - \lambda \text{KL}[\pi_\theta || \pi_{\text{old}}]
$$

### 5.4 Entropy Regularization

Adds stochasticity to prevent early convergence:
$\max_\theta \mathbb{E}[r + \alpha \mathcal{H}(\pi(\cdot|s))]$
Used in SAC and exploration strategies.

### 5.5 Gradient Clipping

Avoids exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5.6 Scheduled Annealing

Examples:

* Learning rate decay
* $\epsilon$-greedy decay
* Temperature scaling in softmax

### 5.7 Interpolation Between Networks

$f_{\text{interp}} = \alpha f_1 + (1 - \alpha) f_2$
Used in distillation, ensembling, or model fine-tuning.

### 5.8 Normalization Techniques

* BatchNorm
* LayerNorm
  They stabilize internal activations—functionally a smooth transition.

### 5.9 Warm Start / Pretraining

Start from pretrained or related initialization to ensure continuity in training.

## 6. Intuition: Why Use Smooth Transitions?

* Enforces **boundedness** and **gradual adaptation**
* Prevents **instabilities** in optimization or learning
* Aligns with **Bayesian updates** and **probabilistic mixtures**
* Helps with **variance reduction** and **convergence smoothing**

## 7. Visual Intuition

Imagine interpolating between two vectors $A$ and $B$. A convex combination draws a straight line from $A$ to $B$ and picks a point along it.

This generalizes to $n$ points:
$x = \sum_i \alpha_i x_i \quad \text{where } \alpha_i \geq 0, \sum_i \alpha_i = 1$

## 8. References

* Polyak and Juditsky (1992), *Acceleration of stochastic approximation by averaging*
* Schulman et al. (2015), *Trust Region Policy Optimization*
* Lillicrap et al. (2015), *Continuous control with deep reinforcement learning (DDPG)*
* Haarnoja et al. (2018), *Soft Actor-Critic Algorithms*
* Vaswani et al. (2017), *Attention is All You Need*





