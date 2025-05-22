---

title: "First Delve into Natural Gradient"
pubDatetime: 2025-05-20T19:00:00Z
description: "A complete introduction to the natural policy gradient: motivation, mathematics, and intuitive meaning."
tags: [RL, Natural Gradient, Policy Gradient, Optimization, Fisher Information, TRPO, PPO]
--------------------------------------------------------------------------------
> It is a wonderful paper published by a single person Sham Kakade in 2001 NeurlIPS.
## 🎯 Motivation

Vanilla policy gradient methods optimize a stochastic policy $\pi_\theta(a|s)$ using gradient ascent in parameter space. However, this ignores the **geometry** of the underlying probability distribution, leading to inefficient and unstable learning.

**Natural Policy Gradient (NPG)** solves this by using a gradient that's aware of the policy's sensitivity — scaling steps using the **Fisher Information Matrix (FIM)**.

---

## 📐 Mathematical Foundations

We aim to maximize the expected return:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]
```

### 🔹 Standard Policy Gradient

Using the policy gradient theorem:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi}(s,a) \right]
```

We update parameters via:

```math
\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)
```

But this direction assumes **Euclidean geometry** — it doesn't respect the fact that small changes in $\theta$ can cause large or small changes in the distribution $\pi_\theta$.

---

## 🧠 Natural Policy Gradient

The **natural gradient** modifies the update to reflect the curvature of the probability distribution space:

```math
\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta)
```

Here, $F(\theta)$ is the **Fisher Information Matrix**.

### 🔸 Fisher Information Matrix (FIM)

```math
F(\theta) = \mathbb{E}_{s,a} \left[ \nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T \right]
```

* Captures local curvature of $\pi_\theta(a|s)$
* Approximates KL divergence: $D_{KL}(\pi_{\theta + \,\epsilon} \| \pi_\theta) \approx \frac{1}{2} \epsilon^T F \epsilon$

### 🔸 Natural Update Rule

```math
\theta_{k+1} = \theta_k + \alpha F(\theta_k)^{-1} \nabla_\theta J(\theta_k)
```

This is analogous to **Newton's method** or **second-order optimization**, but defined in **distribution space**.

---

## 🔍 Why It’s Called “Natural”

* In information geometry, the **natural metric** on the space of distributions is the Fisher information.
* The **natural gradient** is the steepest ascent under this metric.
* This ensures **invariance to reparameterization** — your gradient isn’t distorted by how you encode $\theta$.

---

## 🧭 Meaning and Insight

* Vanilla gradients are like walking in the dark with a flat map — they don’t adapt to the terrain.
* Natural gradients bring a **Riemannian compass**: step sizes adapt to how sensitive the distribution is to parameter change.

> 🎯 **NPG moves in the direction that improves the policy the most, with minimal disruption to its behavior.**

This leads to:

* More stable updates
* Faster convergence
* Better scalability to complex policies (like neural nets)

Natural gradients inspired the development of **TRPO** and **PPO**, which dominate modern policy optimization.

---

## 🧠 Summary

| Concept                           | Description                                           |
| --------------------------------- | ----------------------------------------------------- |
| $\nabla_\theta J(\theta)$         | Standard gradient in parameter space                  |
| $F(\theta)$                       | Fisher Information Matrix (local curvature of policy) |
| $\tilde{\nabla}_\theta J(\theta)$ | Natural gradient (preconditioned by FIM)              |
| Benefit                           | Invariance, stability, faster learning                |

---

## 📖 References

* Kakade, S. (2001). *A Natural Policy Gradient*. NeurIPS.
* Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*.
* Schulman et al. (2015). *Trust Region Policy Optimization*.
* Spinning Up (2019). *Natural Gradient Overview*.
