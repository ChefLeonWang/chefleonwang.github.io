---

title: "Importanace sampling and trust region: Distribution Mismatch and Why Small Policy Updates Work"
pubDatetime: 2025-05-19T14:00:00Z
description: "Combining practical tricks and theoretical bounds to justify ignoring distribution mismatch in policy gradient updates."
tags: [RL, Policy Gradient, Importance Sampling, TRPO, PPO, Distribution Shift]
--------------------------------------------------------------------------------

When optimizing policies in reinforcement learning using policy gradients, we often evaluate objectives using data from the current policy $\pi_\theta$, even though the objective involves the next policy $\pi_{\theta'}$. This introduces a **distribution mismatch** — and yet, it works well in practice.

In this post, we’ll:

* Explain how we approximate policy improvement objectives
* Show the role of **importance sampling**
* Derive a **theoretical bound** on how much the mismatch can matter

---

## 🔁 The Practical Approximation Trick

We aim to improve the policy by estimating the performance gap:

```math
J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

This expression tells us how much better the new policy $\pi_{\theta'}$ performs compared to the current policy $\pi_\theta$, using the advantage function defined under the current policy.

However, computing this requires us to sample trajectories from $\pi_{\theta'}$, which is **not available** yet. So we rewrite the expectation over $p_{\theta'}(\tau)$ using **importance sampling**:

Given two distributions $p$ and $q$, and a function $f$, we use the identity:

```math
\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]
```

Now break the trajectory distribution into per-step terms:

```math
\mathbb{E}_{\tau \sim p_{\theta'}} \left[ \sum_t \gamma^t A^{\pi_\theta}(s_t, a_t) \right] = \sum_t \mathbb{E}_{s_t \sim p_{\theta'}(s_t), a_t \sim \pi_{\theta'}(a_t | s_t)} \left[ \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

Using importance sampling with respect to $\pi_\theta$:

```math
= \sum_t \mathbb{E}_{s_t \sim p_{\theta'}(s_t), a_t \sim \pi_\theta(a_t | s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_\theta(a_t | s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

This form allows us to use data collected from $\pi_\theta$, even though we want to evaluate the new policy $\pi_{\theta'}$.

To simplify computation further, we approximate the state distribution $p_{\theta'}(s_t) \approx p_\theta(s_t)$, yielding:

```math
\bar{A}(\theta') := \sum_t \mathbb{E}_{s_t \sim p_\theta(s_t), a_t \sim \pi_\theta(a_t | s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_\theta(a_t | s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

This approximation allows the optimization:

```math
\theta' \leftarrow \arg\max_{\theta'} \bar{A}(\theta')
```

This is the objective used in algorithms like **TRPO**, **PPO**, and **REINFORCE with importance sampling**.

---

## ⚠️ The Approximation Assumption

We **replace**:

```math
\mathbb{E}_{s_t \sim p_{\theta'}(s_t)} \rightarrow \mathbb{E}_{s_t \sim p_\theta(s_t)}
```

This is **not strictly correct** — it introduces bias. But it’s justified when $\pi_{\theta'} \approx \pi_\theta$.

In practice:

* TRPO enforces $D_{\text{KL}}(\pi_\theta \| \pi_{\theta'}) \leq \delta$
* PPO uses clipping: $r_t = \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}$, bounded in $[1-\epsilon, 1+\epsilon]$

---

## ✅ Theoretical Justification: Bounding the Distribution Drift

We now prove that the **state visitation distributions** $p_{\theta}(s_t)$ and $p_{\theta'}(s_t)$ are close when policies are close.

### 🔹 Setup:

* Assume $\pi_\theta$ is a deterministic policy: $a_t = \pi_\theta(s_t)$
* Let $\epsilon$ be the probability that $\pi_\theta$ and $\pi_{\theta'}$ choose **different actions** at a state:

```math
\Pr[\pi_\theta(s) \neq \pi_{\theta'}(s)] \leq \epsilon
```

### 🔸 Claim:

The distribution over states at time $t$ under the new policy can be written as:

```math
p_{\theta'}(s_t) = (1 - \epsilon)^t p_\theta(s_t) + \left(1 - (1 - \epsilon)^t\right) p_{\text{mistake}}(s_t)
```

Here:

* $(1 - \epsilon)^t$: probability we made **no mistakes** over t steps
* $p_{\text{mistake}}$: some unknown distribution due to diverging behavior

### 🔹 Bounding the Distance

Use triangle inequality:

```math
|p_{\theta'}(s_t) - p_\theta(s_t)| \leq 2 \left(1 - (1 - \epsilon)^t\right)
```

Apply the identity:

```math
1 - (1 - \epsilon)^t \leq \epsilon t
```

Final bound:

```math
|p_{\theta'}(s_t) - p_\theta(s_t)| \leq 2 \epsilon t
```

---

## 🧠 Interpretation

> If the new policy is close to the old one (small $\epsilon$), and you’re not looking too far into the future (small $t$), then the mismatch in state distributions is **provably small**.

This justifies using $p_\theta$ in practice, and training via gradient ascent on approximated objectives like:

```math
J(\theta') - J(\theta) \approx \bar{A}(\theta')
```

---

## 📌 Summary

| What We Want                      | What We Approximate With       | Why It's OK                                        |
| --------------------------------- | ------------------------------ | -------------------------------------------------- |
| $\mathbb{E}_{p_{\theta'}}[\cdot]$ | $\mathbb{E}_{p_\theta}[\cdot]$ | Only valid when $\pi_{\theta'} \approx \pi_\theta$ |
| $p_{\theta'}(s_t)$                | $p_\theta(s_t)$                | Bound: $\leq 2\epsilon t$                          |

This is the bridge between **approximate gradient ascent** and **true policy improvement theory**.

---

## 📖 References

* Kakade & Langford (2002). *Approximately Optimal Approximate Reinforcement Learning*.
* Schulman et al. (2015). *Trust Region Policy Optimization (TRPO)*.
* Schulman et al. (2017). *Proximal Policy Optimization (PPO)*.
* Achiam (2019). *Spinning Up in Deep RL – Distribution Shift & Importance Sampling*.
