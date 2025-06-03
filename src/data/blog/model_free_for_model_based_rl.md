---

title: "Why Backpropagation Fails in Multi-Model Systems"
pubDatetime: 2025-06-03T11:30:00Z
description: "Exploring the practical limitations of backpropagation in model-based reinforcement learning and the benefits of combining model-free techniques."
tags: [rl, backpropagation, multi-model, ppo, dynamics model]
------------------------------------------------------------------------

# Why Backpropagation Fails in Multi-Model Systems

Backpropagation is the workhorse of deep learning. But when we move from single-model architectures to multi-model systems—such as those used in model-based reinforcement learning (MBRL)—the optimization pipeline becomes brittle. Despite theoretical validity, backpropagation across multiple learned models introduces instability, poor convergence, and practical inefficiency.

This post dives into the motivation, challenges, and solutions around this issue, particularly highlighting the integration of model-free techniques like PPO into model-based pipelines.

---

## 🔍 Motivation: Why Backprop Across Models?

In model-based RL, we aim to optimize a policy not by interacting with the real world, but by using a **differentiable world model**. This allows gradients to propagate from reward signals all the way back to the policy parameters:

$$
r_t = r_\phi(f_\psi(\pi_\theta(s_t)))
$$

Here:

* $\pi_\theta$: policy model
* $f_\psi$: dynamics model
* $r_\phi$: reward model

Backpropagation lets us compute $\nabla_\theta r_t$, offering a direct learning signal to improve policy.

**In theory**, this is elegant and sample-efficient. **In practice**, it rarely works well.

---

## ⚠️ Challenge: Why It Fails in Practice

While math allows us to compute gradients through this pipeline, there are several practical issues:

### 1. Gradient Instability

Like deep RNNs, chained derivatives across models lead to exploding/vanishing gradients.

* Especially problematic for **long trajectories** (multi-step rollout)
* Compounded by **non-linearities** and **uncertainty** in each module

### 2. Objective Misalignment

Each model (policy, dynamics, reward) has different learning objectives:

* Policy wants to maximize return
* Dynamics wants to minimize prediction error
* Reward model often trained separately on human labels or sparse signals

### 3. Multi-Model Bottleneck

Backpropagation assumes **tight coupling**:

* Requires all models to be **differentiable and jointly trained**
* Small error in one model may break gradients for others

### 4. Compounding Errors

When the policy improves, it visits **new states** outside of the training distribution for the dynamics model. Prediction quality deteriorates, breaking the whole pipeline.

> ❗️ "The reward may be differentiable—but if the intermediate state is wrong, the gradient is wrong."

---

## 📐 What Is a Jacobian?

The **Jacobian** is a matrix of all partial derivatives for a vector-valued function. If a function maps a vector input to a vector output:

$$
\mathbf{y} = f(\mathbf{x}),\quad \mathbf{x} \in \mathbb{R}^n,\ \mathbf{y} \in \mathbb{R}^m
$$

Then the **Jacobian matrix** is:

$$
J_{ij} = \frac{\partial y_i}{\partial x_j},\quad \text{or } J \in \mathbb{R}^{m \times n}
$$

It tells us how each component of the output changes with respect to each input.

### 🧠 In Deep Learning

* Scalar output (e.g., loss): gradient vector
* Vector output: Jacobian matrix
* In neural nets, **backprop multiplies many Jacobians together** — which can cause instability in long chains

### 💥 In Model-Based RL

When backpropagating through multiple time steps and models, you multiply Jacobians across layers, time steps, and architectures:

$$
\nabla_\theta r_t = \frac{\partial r}{\partial s_T} \cdot \frac{\partial s_T}{\partial a_{T-1}} \cdot \frac{\partial a_{T-1}}{\partial \theta} \cdots
$$

If any Jacobian is ill-conditioned, gradients may vanish or explode.

### ✅ In Deep ML

Jacobian chains work because:

* Network is unified
* Residual/skip connections reduce depth
* Layer norms stabilize activations
* Gradient flow is smooth and local

> **Jacobian = how change in input causes change in output.**
> In deep learning: manageable. In model-based RL: chaotic.


---

## 🧠 Solution: Use Model-Free Updates for Policy

Instead of relying on backpropagation to update the policy model, we can use **model-free RL methods** like PPO or TRPO to optimize $\pi_\theta$, while still using a learned world model for rollouts.

### 🎯 Key Idea

Let the dynamics model and reward model generate synthetic trajectories:

$$
s_0 \rightarrow a_0 = \pi_\theta(s_0) \rightarrow s_1 = f_\psi(s_0, a_0) \rightarrow r_0 = r_\phi(s_0, a_0)
$$

Repeat for $T$ steps → get trajectory → use PPO to update policy.

### 🔁 No Backprop Across Models

* No need for $\nabla_\theta f_\psi$
* Use synthetic data to estimate return, advantage, value function, etc.

### ✅ Advantages

* **Stable optimization** from PPO/actor-critic methods
* **Modular training**: dynamics model and policy can be trained asynchronously
* **Compatible with long rollouts**

---

## 🔢 Mathematical View

Let $\tau = (s_0, a_0, r_0, ..., s_T)$ be a trajectory generated by models.

Use PPO objective:

$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_{\tau \sim f_\psi, \pi_\theta} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$, $\hat{A}_t$ is the advantage.

All gradients $\nabla_\theta$ are **contained within the policy network**.

No gradients backpropagate through $f_\psi$ or $r_\phi$.

---

## 🔬 Why Pathwise Gradient Works in Deep Learning but Not in RL

You might wonder: backpropagation works just fine in deep image models, transformers, and diffusion networks. Why does it fail in model-based RL?

The short answer:

> Because in deep ML, the entire computation graph belongs to **a single model**, with unified architecture, parameters, and objectives. In model-based RL, we must chain **multiple, mismatched models** across time and objectives.

### 📊 Comparison Table

| Aspect          | Deep Learning                | Model-Based RL Pathwise Gradient  |
| --------------- | ---------------------------- | --------------------------------- |
| Structure       | One monolithic model         | Multiple models: π, f, r          |
| Objective       | Single (e.g., cross-entropy) | Conflicting (predict vs. control) |
| Gradient Flow   | Within one network           | Across time and models            |
| Jacobian Chain  | Dozens of layers             | Tens of steps × model transitions |
| Stability Tools | Residuals, normalization     | No residuals, no layer sharing    |
| Data            | i.i.d. batches               | Autoregressive trajectories       |

### 🔥 The Chaos Problem

Parmas et al. (2018) describe this as the **"Curse of Chaos"**: when small inaccuracies in learned dynamics amplify as trajectories unroll, gradients become meaningless.

The math confirms it:

$$
\nabla_\theta J(\theta) = \sum_t \frac{\partial a_t}{\partial \theta} \cdot \frac{\partial s_{t+1}}{\partial a_t} \cdot \left( \sum_{t'} \frac{\partial r_{t'}}{\partial s_{t'}} \cdot \prod_{t''} \frac{\partial s_{t''}}{\partial s_{t''-1}} \right)
$$

If any Jacobian has high variance or is near-zero, this blows up or vanishes rapidly.

### ✅ Why Deep Learning Avoids It

* Residuals allow direct gradient flow across layers
* LayerNorm/BatchNorm stabilize activations
* Shared architecture ensures gradients reinforce each other
* No sequential rollout compounding model error

So in deep ML:

> **Pathwise gradient works because the network is stable, local, and unified.**

In model-based RL:

> **Pathwise gradient fails because you're gluing together unstable black boxes.**


---

## 🧩 Broader Implications: This Is Not Just RL

This isn't just a problem in reinforcement learning. Any system with:

* **multiple learned modules**
* **chained differentiable paths**
* **conflicting objectives**

will face the same optimization bottlenecks. Examples:

* Diffusion models (denoising vs. generation)
* GNN → Transformer → Classifier pipelines
* Multi-agent learning with shared environments

---

## ✅ Summary

| Strategy                     | Pros                    | Cons                         |
| ---------------------------- | ----------------------- | ---------------------------- |
| Backprop through all models  | Clean gradients         | Fragile, hard to optimize    |
| Model-free update (e.g. PPO) | Stable, robust, modular | Needs extra simulation infra |
| Hybrid (Dreamer, etc.)       | Efficient and scalable  | Still hard to balance        |

Backpropagation is powerful—but not omnipotent. In complex systems, **decoupling modules and using model-free updates** is often the most robust strategy.

---

## 📚 References

* Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination*, 2020
* Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2018
* Parmas et al., *PIPP: Flexible Model-Based Policy Search Robust to the Curse of Chaos*, 2018
* Yarin Gal et al., *Bayesian Deep Learning*, 2016





