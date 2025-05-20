---

title: "Distribution Mismatch and Why Small Policy Updates Work"
pubDatetime: 2025-05-20T14:00:00Z
description: "Combining practical tricks and theoretical bounds to justify ignoring distribution mismatch in policy gradient updates."
tags: [RL, Policy Gradient, Importance Sampling, TRPO, PPO, Distribution Shift, KL-divergence, PPO-clipping]
--------------------------------------------------------------------------------

When optimizing policies in reinforcement learning using policy gradients, we often evaluate objectives using data from the current policy $\pi_\theta$, even though the objective involves the next policy $\pi_{\theta'}$. This introduces a **distribution mismatch** — and yet, it works well in practice.

Goal:
* Explain how we approximate policy improvement objectives
* Show the role of **importance sampling**
* Derive a **theoretical bound** on how much the mismatch can matter
* Explore how **KL divergence** and **clipping** stabilize updates in modern algorithms

---

## 🔁 The Practical Approximation Trick

We aim to improve the policy by estimating the performance gap:

```math
J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
```

This expression tells us how much better the new policy $\pi_{\theta'}$ performs compared to the current policy $\pi_\theta$, using the **advantage function** defined under the current policy.

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

## 📐 A More Convenient Bound Using KL Divergence

Rather than bounding per-step policy differences in terms of total variation (TV) distance, we can use **KL divergence**:

### 🔹 Pinsker’s Inequality

```math
|\pi_{\theta'}(a|s) - \pi_\theta(a|s)| \leq \sqrt{\frac{1}{2} D_{\text{KL}}(\pi_{\theta'}(\cdot|s) \| \pi_\theta(\cdot|s))}
```

This gives a differentiable and tractable surrogate for measuring **policy closeness**.

KL divergence is defined as:

```math
D_{\text{KL}}(p(x) \| q(x)) = \mathbb{E}_{x \sim p} \left[ \log \frac{p(x)}{q(x)} \right]
```

Hence, small KL implies that $p_{\theta'}(s_t) \approx p_\theta(s_t)$, and is used directly in:

* TRPO: enforce KL trust region constraint
* PPO: KL used implicitly via clipping surrogate

---

## 🔧 Clipping in PPO

Proximal Policy Optimization (PPO) avoids the complexities of second-order optimization (like TRPO) by using a clipped objective to restrict the size of policy updates.

### 🔹 The Motivation

Directly maximizing the surrogate objective using importance sampling:

```math
\mathbb{E}_t \left[ r_t(\theta') \hat{A}_t \right], \quad \text{where} \quad r_t(\theta') = \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}
```

can lead to large updates when $r_t(\theta')$ deviates too far from 1.

To prevent this, PPO introduces the **clipped surrogate objective**:

### 🔸 The PPO Objective

```math
L^{\text{CLIP}}(\theta') = \mathbb{E}_t \left[ \min \left( r_t(\theta') \hat{A}_t, \text{clip}(r_t(\theta'), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
```

This objective:

* Encourages improvement when $r_t \approx 1$
* Suppresses updates that would push $r_t$ outside of $[1 - \epsilon, 1 + \epsilon]$

### 🔹 Why It Works

* The $\min$ operator ensures that the improvement is **monotonic** and avoids overshooting
* Clipping creates a **flat region** in the loss around $r_t = 1$, leading to more stable gradients
* It **implicitly controls** KL divergence without computing it directly

### 🔹 Visual Intuition

* When $\hat{A}_t > 0$: we don’t want $r_t \gg 1$ to overly favor a good action — clipping caps the advantage
* When $\hat{A}_t < 0$: we don’t want $r_t \ll 1$ to overly punish — clipping limits how far you move away

In both cases, clipping avoids large policy changes — achieving the same stability goal as TRPO, but with simpler first-order methods.

### 🔹 Typical Values

* $\epsilon \in [0.1, 0.3]$ — common values in PPO implementations
* Used with **adaptive learning rate** and **early stopping by KL threshold** to further stabilize updates


---

## 🧠 Interpretation

> If the new policy is close to the old one (small $\epsilon$), and you’re not looking too far into the future (small $t$), then the mismatch in state distributions is **provably small**.

This justifies using $p_\theta$ in practice, and training via gradient ascent on approximated objectives like:

```math
J(\theta') - J(\theta) \approx \bar{A}(\theta')
```

Or using surrogate losses like:

```math
L^{\text{CLIP}}(\theta')
```

---

## 📌 Summary

| Concept             | Role                                                               |                                      |                    |
| ------------------- | ------------------------------------------------------------------ | ------------------------------------ | ------------------ |
| Importance Sampling | Use data from old policy to estimate new policy objective          |                                      |                    |
| TV distance bound   |                                             |
| KL divergence       | Convenient surrogate for TV: $D_{\text{KL}}$ bounds TV via Pinsker |                                      |                    |
| PPO Clipping        | Ensures ratio stays near 1, prevents instability                   |                                      |                    |

This is the bridge between **approximate gradient ascent** and **stable policy optimization**.

---

## 📖 References

* Kakade & Langford (2002). *Approximately Optimal Approximate Reinforcement Learning*.
* Schulman et al. (2015). *Trust Region Policy Optimization (TRPO)*.
* Schulman et al. (2017). *Proximal Policy Optimization (PPO)*.
* Achiam (2019). *Spinning Up in Deep RL – Distribution Shift & Importance Sampling*.




