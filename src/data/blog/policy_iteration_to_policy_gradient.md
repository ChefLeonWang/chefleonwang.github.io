---

title: "Policy Iteration → Policy Gradient: A Natural Evolution"
pubDatetime: 2025-05-19T16:30:00Z
description: "Why policy gradient methods emerged from the limitations of classical policy iteration, especially in continuous or high-dimensional action spaces. Includes key math and references."
tags: [RL, Policy Gradient, Policy Iteration, Continuous Control, Deep RL]
---------------------------------------------------------------------------

As reinforcement learning evolved from tabular environments to high-dimensional, continuous domains, classical methods like **Policy Iteration** faced major limitations. This naturally led to the emergence of **Policy Gradient** methods — a continuous, differentiable, and scalable generalization of the policy improvement idea.

---

## 🔹 Classical Policy Iteration

Policy Iteration is a two-step process:

1. **Policy Evaluation**:

   Estimate the value function:

   ```math
   V^{\pi}(s) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \mid s_0 = s \right]
   ```

2. **Policy Improvement**:

   Update policy greedily:

   ```math
   \pi'(s) = \arg\max_a Q^{\pi}(s, a)
   ```

Repeat these steps until convergence.

✅ This works well in discrete environments like Gridworld, Blackjack, or small MDPs.

❌ But it **fails in continuous action spaces**, where $\arg\max_a Q(s,a)$ is intractable.

---

## 🔸 Why It Breaks in Continuous Spaces

* The action space $\mathcal{A} \subseteq \mathbb{R}^n$ may be infinite or high-dimensional.
* $\arg\max_a Q(s,a)$ becomes a **non-trivial optimization problem**.
* Greedy policy improvement becomes either inexact, expensive, or undefined.

This is where **Policy Gradient** comes in.

---

## 🧠 The Natural Evolution: Policy Gradient

Instead of constructing a new greedy policy explicitly, we **learn a parameterized policy** $\pi_\theta(a \mid s)$ and optimize its parameters $\theta$ via gradient ascent:

```math
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
```

Where the objective is the **expected return**:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)\right]
```

---

## 🔍 Policy Gradient Theorem

The core result from \[Sutton et al., 2000]:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi}\left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi}(s, a) \right]
```

Where:

* $d^{\pi}(s)$: discounted state visitation distribution
* $Q^{\pi}(s, a)$: action-value under policy $\pi$

This avoids $\arg\max$, and works in **continuous, stochastic, and high-dimensional** environments.

---

## 🔁 Connection to Policy Iteration

Policy gradient implicitly mirrors policy iteration:

| Classic Step       | Policy Gradient Equivalent                                        |                                           |
| ------------------ | ----------------------------------------------------------------- | ----------------------------------------- |
| Policy Evaluation  | Estimate $Q^{\pi}(s,a)$ or $A^{\pi}(s,a)$ using critic or returns |                                           |
| Policy Improvement | Use gradient of ( \log \pi(a                                      | s) \cdot A^{\pi}(s,a) ) to improve policy |

Instead of computing $\arg\max$, we move in a **descent direction** where actions with positive advantage become more likely:

```math
\nabla_\theta J(\theta) \propto \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) A^{\pi}(s,a) \right]
```

---

## ✅ Summary

> Classical Policy Iteration breaks down when action spaces become continuous or large.

> Policy Gradient provides a **differentiable**, **local**, and **scalable** solution — performing the same logical operation: **favoring better actions**, but using soft, gradient-based updates.

This is why **modern RL shifted from table-based algorithms to neural, stochastic, differentiable pipelines.**

---

## 📖 References

* Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). *Policy gradient methods for reinforcement learning with function approximation.* NIPS.
* Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). *Deterministic policy gradient algorithms*. ICML.
* Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-dimensional continuous control using generalized advantage estimation*. arXiv:1506.02438.
