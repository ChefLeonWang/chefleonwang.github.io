---

title: "Again into policy iteration and policy gradient: A Deep Learning Analogy"
pubDatetime: 2025-05-22T11:30:00Z
description: "How policy iteration relates to classical optimization and policy gradient to deep learning, with full mathematical detail."
tags: [RL, Policy Gradient, Policy Iteration, Deep Learning, Optimization, shallow machine learning]
---------------------------------------------------------------------------

## 🎯 Motivation

In reinforcement learning, policy iteration and policy gradient methods offer two distinct strategies for learning control policies. Interestingly, these two paradigms resemble the evolution in machine learning from shallow models with exact solvers to deep learning with gradient-based optimization.


* **Policy Iteration ↔ Classical Optimization (e.g., SVM, logistic regression)**
* **Policy Gradient ↔ Deep Learning (SGD)**

---

## 🔁 Policy Iteration

Policy iteration consists of two steps:

1. **Policy Evaluation**: Estimate the value function $V^{\pi}(s)$ for a given policy $\pi$:

```math
V^{\pi}(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s \right]
```

This typically involves solving the Bellman equation:

```math
V^{\pi}(s) = \sum_a \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^{\pi}(s') \right]
```

2. **Policy Improvement**: Compute a new policy by acting greedily w\.r.t. $Q^{\pi}(s,a)$:

```math
\pi'(s) = \arg\max_a Q^{\pi}(s,a)
```

Repeat until convergence.

---

## 🧠 Policy Gradient

Policy gradient methods **directly optimize** the parameters of a stochastic policy $\pi_\theta(a \mid s)$:

### Objective:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]
```

### Policy Gradient Theorem:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi}(s,a) \right]
```

This enables gradient ascent using SGD-like updates:

```math
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
```

This avoids hard $\arg\max$, allowing optimization over high-dimensional, continuous action spaces.

---

## 📈 Classical Optimization (Shallow ML)

Consider logistic regression or linear regression with a closed-form solution.

### Linear Regression:

Given $X \in \mathbb{R}^{n \times d}, y \in \mathbb{R}^n$, solve:

```math
\min_w \frac{1}{2} \|Xw - y\|^2
```

Closed-form solution:

```math
w^* = (X^T X)^{-1} X^T y
```

### Logistic Regression / SVM:

Convex optimization problems with unique optima:

```math
\min_w \sum_{i=1}^n \log(1 + \exp(-y_i w^T x_i)) + \lambda \|w\|^2
```

Solved with **batch gradient descent**, Newton methods, or dual solvers (for SVM).

---

## 🔁 Deep Learning and Stochastic Gradient Descent

In deep learning, we train a neural network parameterized by $\theta$ using stochastic gradient descent:

```math
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)
```

Where $\mathcal{L}(\theta)$ is typically:

```math
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)
```

* No closed-form solution
* Non-convex landscape
* Relies entirely on **gradient signals**

Just like **policy gradient**, this process improves the model **locally**, one step at a time.

---

## 🔁 Analogy Table

| ML Concept                      | Reinforcement Learning Analogue |
| ------------------------------- | ------------------------------- |
| Linear/logistic regression      | Policy iteration (tabular)      |
| Exact optimization ($\arg\max$) | Greedy policy improvement       |
| SGD in deep nets                | Policy gradient ascent          |
| Gradient of log likelihood      | Gradient of log policy          |
| Batch/closed-form solvers       | Dynamic programming (DP)        |

---

## 🧠 Summary

* Policy iteration is exact but limited to small, discrete spaces.
* Policy gradient is flexible, scalable, and essential for neural policies.
* This mirrors ML’s shift from closed-form optimization to scalable SGD.

> Just as deep learning replaced exact solvers in high-dimensional spaces,
> policy gradient replaced greedy iterations with continuous learning.

---

## 📖 References

* Sutton & Barto (2018). *Reinforcement Learning: An Introduction*
* Kakade (2002). *A Natural Policy Gradient*
* Goodfellow et al. (2016). *Deep Learning*
* Schulman et al. (2015). *Trust Region Policy Optimization*
