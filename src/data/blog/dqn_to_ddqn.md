---

title: "From DQN to Double DQN: Fixing Overestimation Bias"
pubDatetime: 2025-05-18T11:00:00Z
description: "Understand the overestimation problem in Deep Q-Learning and how Double DQN (DDQN) provides a simple but powerful fix by decoupling action selection and evaluation."
tags: \[RL, DQN, DDQN, Q-learning, Overestimation, Deep RL]
-----------------------------------------------------------

Deep Q-Networks (DQN) marked a breakthrough in reinforcement learning. But they come with a hidden issue: **overestimation bias** in Q-values. This post explains where it comes from and how **Double DQN (DDQN)** solves it.

---

## 🎯 The Core Problem in DQN

In DQN, we update the Q-network using the Bellman target:

```math
y_j = r_j + \gamma \max_{a'} Q_{\phi'}(s_j', a')
```

This `max` term is problematic when Q-values are **noisy** (as they often are with neural nets). Because of the inequality:

```math
\mathbb{E}[\max(X_1, X_2)] \geq \max(\mathbb{E}[X_1], \mathbb{E}[X_2])
```

We tend to **overestimate** the value of the next state.

---

## 🤔 Why Does Overestimation Happen?

The Q-function is approximated using a neural network, and neural networks are **imperfect and noisy**. So when we do:

```math
\max_{a'} Q_{\phi'}(s', a')
```

we’re taking the maximum of several noisy estimates. This is statistically biased **upward**.

Even if each estimate is unbiased, the maximum of them is biased due to the statistical identity:

```math
\mathbb{E}[\max(X_1, X_2)] \geq \max(\mathbb{E}[X_1], \mathbb{E}[X_2])
```

---

## 📈 Why It’s Even Worse in Practice

In DQN, the target value is computed as:

```math
\max_{a'} Q_{\phi'}(s', a') = Q_{\phi'}(s', \arg\max_{a'} Q_{\phi'}(s', a'))
```

So the same noisy network $Q_{\phi'}$ is used to both:

* **Choose** the best action
* **Evaluate** its value

This compounds the noise — like letting a biased estimator grade its own work.

---

## 😬 Why This Matters

This **overestimation bias** leads to:

* Overly optimistic value estimates
* Unstable or divergent Q-values
* Risky policies that rely on hallucinated future rewards

In real-world environments (with sparse or delayed rewards), this can make learning much worse.

---

## 💡 Solution: Double Q-learning (DDQN)

**Double DQN** solves this by **decoupling** the two roles:

```math
\text{DQN: } y = r + \gamma \max_{a'} Q_{\phi'}(s', a')
\text{DDQN: } y = r + \gamma Q_{\phi'}(s', \arg\max_{a'} Q_{\phi}(s', a'))
```

Here:
We want to decorrelate the noise in value and action, both of which comes from the same network.

Idea: do no use the same network to choose the action and evaluate value.
* The **main network** $Q_{\phi}$ selects the best action, which is 

* The **target network** $Q_{\phi'}$ evaluates it

This way, noise from $Q_{\phi}$ doesn’t inflate the value estimate.

---

## 🧠 Why It Works

Using two networks avoids the bias of taking both the `argmax` and value from the same source. It’s conceptually like:

> “One student picks the answer, another one grades it.”

The evaluation is more **neutral**, making learning more stable.

---

## ✅ Summary Table

| Method         | Target Value Formula                                   | Overestimation Risk |
| -------------- | ------------------------------------------------------ | ------------------- |
| **DQN**        | $r + \gamma Q_{\phi'}(s', \arg\max_a Q_{\phi'}(s', a))$                   | High                |
| **Double DQN** | $r + \gamma Q_{\phi'}(s', \arg\max_a Q_{\phi}(s', a))$ | Reduced             |

---

## 📚 References

### 📘 DQN (Deep Q-Network)

* Mnih et al., *“Human-level control through deep reinforcement learning”*, Nature, 2015
  [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)

### 📙 DDQN (Double DQN)

* Hasselt et al., *“Deep Reinforcement Learning with Double Q-learning”*, AAAI, 2016
  [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)

---

> Double DQN is a minimal tweak to DQN that fixes a deep statistical flaw.

