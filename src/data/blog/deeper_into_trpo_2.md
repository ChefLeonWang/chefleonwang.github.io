---

title: "Again Dive into TRPO: Understanding TRPO from Macro to Micro"
pubDatetime: 2025-05-22T19:30:00Z
description: "How Taylor expansion, KL divergence, and the Fisher matrix fit together in Trust Region Policy Optimization."
tags: [RL, TRPO, Optimization, KL Divergence, Fisher Information, Taylor Expansion, Natural Policy Gradient]
------------------------------------------------------------------------------------

All the math (Taylor expansions, KL divergence, Fisher Information Matrix) serve a unified goal in TRPO:

>Make policy updates safe by keeping the new policy close to the old one in a meaningful way.


## 🧭 MACRO LEVEL: The Philosophy of TRPO

The core idea of TRPO is:

> ✅ Improve the policy
> 🚫 But don't let it shift too much from the old one

Why? Because big changes to a policy can:

* Cause instability in learning
* Lead to distributional shift — where states visited by the new policy differ too much from the ones used to train it
* Waste data — if the new policy is too different, old trajectories are no longer useful

TRPO formalizes this with a trust region: only allow updates where the **KL divergence** between new and old policy is below a small threshold $\delta$.

---

## ⚙️ MID LEVEL: The Optimization Setup

We want to find a new policy $\pi_{\theta'}$ that maximizes improvement over $\pi_\theta$ but stays close to it.

### 🎯 Objective:

We use a **surrogate advantage objective**:

```math
\bar{A}(\theta') = \mathbb{E}_{s,a \sim \pi_\theta} \left[ \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)} A^{\pi_\theta}(s,a) \right]
```

This quantifies how much better the new policy performs **under old data**.

### 🚧 Constraint:

Keep the policy update small in terms of KL divergence:

```math
\mathbb{E}_{s \sim d^{\pi}} \left[ D_{\text{KL}}(\pi_{\theta'}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta
```

---

## 🔍 MICRO LEVEL: Each Math Tool’s Role

To solve this constrained optimization problem efficiently, we apply these mathematical tools:

### 1. First-order Taylor Expansion (for Objective)

Approximate the surrogate objective linearly:

```math
\bar{A}(\theta') \approx \bar{A}(\theta) + \nabla_\theta \bar{A}(\theta)^T (\theta' - \theta)
```

This simplifies the objective to something **linear in parameters**.

### 2. Second-order Taylor Expansion (for Constraint)

Approximate KL divergence quadratically:

```math
D_{\text{KL}}(\pi_{\theta'} \| \pi_\theta) \approx \frac{1}{2} (\theta' - \theta)^T F (\theta' - \theta)
```

### 3. Fisher Information Matrix (FIM)

This matrix approximates the curvature of the KL divergence landscape:

```math
F(\theta) = \mathbb{E}_{s,a} \left[ \nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T \right]
```

FIM measures **how much the policy output changes** with respect to small changes in parameters.

---

## ✅ SOLVING THE TRUST REGION PROBLEM

After applying Taylor approximations, we arrive at a constrained quadratic problem:

```math
\begin{aligned}
&\text{maximize:} && g^T x \\
&\text{subject to:} && \frac{1}{2} x^T F x \leq \delta
\end{aligned}
```

Where:

* $g = \nabla_\theta \bar{A}(\theta)$
* $F$ is the Fisher Information Matrix
* $x = \theta' - \theta$

This has a **closed-form solution**:

```math
x^* = \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g
```

So TRPO performs a **natural gradient step** scaled to lie inside the trust region.

---

## 📊 Visual Summary

![TRPO flowchart](/images/trpo.png)

---

## 🧠 WRAP-UP: Mental Model Summary

| Concept             | Role in TRPO                                      |
| ------------------- | ------------------------------------------------- |
| KL Divergence       | Defines how far the new policy is allowed to go   |
| First-Order Taylor  | Linearizes the surrogate objective                |
| Second-Order Taylor | Approximates KL divergence with curvature         |
| Fisher Matrix       | Defines meaningful distance in distribution space |
| Natural Gradient    | Optimal direction respecting trust region         |

> 🧠 TRPO is a method that:
>
> 1. Improves the policy reliably
> 2. Stays close to the current policy
> 3. Uses the geometry of probability space to guide learning

It is mathematically elegant and practically robust — laying the foundation for modern safe policy optimization.

---

## 📚 References

* Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). *Trust Region Policy Optimization*. ICML.
* Kakade, S. (2001). *A Natural Policy Gradient*. NeurIPS.
* Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*. Neural Computation.
* Spinning Up (2019). *TRPO Explained*. OpenAI.
