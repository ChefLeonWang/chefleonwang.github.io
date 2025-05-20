---

title: "Deeper dive into TRPO"
pubDatetime: 2025-05-20T16:00:00Z
description: "A detailed walkthrough of TRPO's constrained optimization strategy, natural gradient step, and KL trust region."
tags: [RL, TRPO, Trust Region, Policy Optimization, KL Constraint]
-------------------------------------------------------------------

## 🎯 Motivation

> It is so beautiful, and so hard. Really wish to meet this algorithm 5 years earlier.


Trust Region Policy Optimization (TRPO) is a reinforcement learning algorithm designed to ensure stable and monotonic policy improvement. 

It does so by solving a constrained optimization problem that **explicitly*** limits **HOW MUCH** the policy is allowed to change between updates.

---

## 📐 The TRPO Optimization Problem

We aim to maximize the expected advantage of a new policy $\pi_{\theta'}$ over the old policy $\pi_\theta$, while ensuring the KL divergence between them remains small:

```math
\theta' = \arg\max_{\theta'} \sum_t \mathbb{E}_{s_t \sim p_\theta, a_t \sim \pi_\theta} \left[ \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

Subject to:

```math
D_{\text{KL}}(\pi_{\theta'}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)) \leq \epsilon
```

This ensures that the new policy stays within a **trust region** of the old policy.

---

## 📚 Taylor Expansion: First-Order and Second-Order

Taylor expansion is a powerful mathematical tool to approximate complex functions around a reference point.

Given a smooth function $f(x)$, the Taylor expansion around point $x_0$ is:

```math
f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2} f''(x_0)(x - x_0)^2 + \cdots
```

This is generalized for multivariate functions using gradients and Hessians.

### 🔹 First-Order Taylor Expansion (Linear Approximation)

For a scalar-valued function $f(\theta)$, where $\theta \in \mathbb{R}^n$:

```math
f(\theta') \approx f(\theta) + \nabla f(\theta)^T (\theta' - \theta)
```

This gives a **linear approximation**, used in TRPO to simplify the policy objective:

```math
\bar{A}(\theta') \approx \bar{A}(\theta) + \nabla_\theta \bar{A}(\theta)^T (\theta' - \theta)
```

It is accurate when $\theta'$ is close to $\theta$, justifying the need for small updates.

### 🔸 Second-Order Taylor Expansion (Quadratic Approximation)

For more precision, especially for constraints like KL divergence, we include the second-order term:

```math
f(\theta') \approx f(\theta) + \nabla f(\theta)^T (\theta' - \theta) + \frac{1}{2} (\theta' - \theta)^T H (\theta' - \theta)
```

Where $H = \nabla^2 f(\theta)$ is the Hessian matrix.

In TRPO, we apply this to the KL divergence:

```math
D_{\text{KL}}(\pi_{\theta'} \| \pi_\theta) \approx \frac{1}{2} (\theta' - \theta)^T H (\theta' - \theta)
```

Where $H \approx \text{Fisher Information Matrix}$ under $\pi_\theta$.

---

## 🧠 The Hessian Matrix

The **Hessian matrix** is the matrix of second-order partial derivatives of a scalar-valued function.

Given a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, the Hessian $H \in \mathbb{R}^{n \times n}$ is defined as:

```math
H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}
```

It captures how the gradient changes — i.e., the curvature of the function surface.

* If $H$ is **positive definite**, the function is convex locally.
* If $H$ is **negative definite**, the function is concave locally.

In optimization problems like TRPO, the Hessian is approximated by the Fisher Information Matrix to reduce computational cost and improve stability.

---

## 🧮 Fisher Information Matrix (FIM)

The **Fisher Information Matrix** is a way to approximate the curvature of a likelihood or policy distribution.

Given a policy $\pi_\theta(a|s)$, the Fisher Information Matrix is:

```math
F(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T \right]
```

This matrix serves as a **natural metric** for measuring distances between probability distributions.

* It is symmetric and positive semi-definite.
* It arises as the second derivative of the KL divergence:

```math
\nabla^2 D_{\text{KL}}(\pi_{\theta + \epsilon} \| \pi_\theta) \approx F(\theta)
```

This is why TRPO uses FIM as a **second-order approximation of the KL constraint**.

FIM also leads to the **natural gradient**:

```math
\theta' = \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta)
```

---

## ✏️ First-Order Taylor Expansion Details

To make the objective tractable, we use a **first-order Taylor expansion** around the current parameters $\theta$. Define:

```math
\bar{A}(\theta') = \sum_t \mathbb{E}_{s_t, a_t} \left[ \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

Now expand $\bar{A}(\theta')$ around $\theta$:

```math
\bar{A}(\theta') \approx \bar{A}(\theta) + \nabla_\theta \bar{A}(\theta)^T (\theta' - \theta)
```

Since $\bar{A}(\theta)$ is constant w\.r.t. $\theta'$, we focus on maximizing:

```math
\theta' = \arg\max_{\theta'} \nabla_\theta \bar{A}(\theta)^T (\theta' - \theta)
```

This form makes the optimization problem linear in $\theta'$, and allows TRPO to proceed with efficient solvers like conjugate gradient to estimate the optimal step.

The advantage $A^{\pi_\theta}(s_t, a_t)$ is typically estimated using generalized advantage estimation (GAE) or Monte Carlo returns.

This approximation holds as long as the step $\theta' - \theta$ is small — hence the **trust region** enforced by KL divergence.

---

## 📉 Quadratic Approximation of the Constraint

We approximate the KL constraint using a second-order Taylor expansion:

```math
D_{KL}(\pi_{\theta'} \| \pi_\theta) \approx \frac{1}{2} (\theta' - \theta)^T H (\theta' - \theta)
```

Where $H$ is the **Fisher Information Matrix (FIM)** of $\pi_\theta$.

---

## 🧮 Trust Region Subproblem

This results in a quadratic programming problem:

```math
\text{maximize:} \quad g^T x \\
\text{subject to:} \quad \frac{1}{2} x^T H x \leq \delta
```

Where:

* $x = \theta' - \theta$
* $g = \nabla_\theta \bar{A}(\theta)$
* $H \approx \text{FIM}$

It has a closed-form solution:

```math
x^* = \sqrt{ \frac{2\delta}{g^T H^{-1} g} } H^{-1} g
```

This is a **natural gradient step**, scaled to remain inside the trust region.

---

## ✅ Guaranteed Improvement

TRPO guarantees that:

```math
J(\theta') - J(\theta) \geq \frac{1}{1 - \gamma} \mathbb{E}_{s,a}[A^{\pi_\theta}(s,a)] - \frac{2\epsilon \gamma}{(1 - \gamma)^2}
```

For small enough $\epsilon$, this is always **positive**.

---

## 📊 Summary

| Component                     | Purpose                                                               |
| ----------------------------- | --------------------------------------------------------------------- |
| First-order approximation     | Linearize the objective to make optimization tractable                |
| Second-order KL approximation | Quadratic form allows analytic constraint handling                    |
| Fisher Information Matrix     | Captures policy sensitivity, defines trust region geometry            |
| Natural gradient              | Efficiently computes update direction respecting information geometry |

TRPO is the gold standard for **theoretically sound** policy improvement, though more complex than PPO.

---

## 📖 References

* Schulman et al. (2015). *Trust Region Policy Optimization*. ICML.
* Kakade (2002). *A Natural Policy Gradient*. NeurIPS.
* Achiam (2019). *Spinning Up in Deep RL – TRPO notes*.
