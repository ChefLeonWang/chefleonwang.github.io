---

title: "Planning in Latent Space: Model-Based RL beyond Observations"
pubDatetime: 2025-06-01
Tags: [reinforcement learning, latent space, model-based RL, RL, planning]
draft: false
description: "An end-to-end explanation of why and how modern model-based reinforcement learning leverages latent state space modeling for stability, scalability, and differentiable planning."
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Planning in Latent Space: Model-Based RL beyond Observations

As model-based reinforcement learning (MBRL) scales to high-dimensional sensory inputs (like images), planning directly in observation space becomes computationally unstable and inefficient. To address this, a new class of MBRL methods propose to **learn a compact latent space** where dynamics and reward models can be efficiently trained and rolled out.

This document introduces the motivation, model structure, training objective, and planning strategy of latent-space MBRL, as used in methods like **PlaNet**, **Dreamer**, and **SimPLe**.

---

## 🧠 Why latent space?

In classic MBRL, we assume access to state transitions:

$$
s_{t+1} = f(s_t, a_t)
$$

But in reality, we often only observe high-dimensional data $o_t$, like images or sensor readings. These are not Markovian, nor tractable for model learning:

* Raw pixels $o_t$ are too large to model directly
* Observation noise causes instability
* The dynamics depend on latent variables we don’t directly observe

Instead, we postulate the existence of a compact **latent state** $s_t$, and assume a hidden Markov model:

* $p(o_t \mid s_t)$ — observation model
* $p(s_{t+1} \mid s_t, a_t)$ — dynamics model
* $p(r_t \mid s_t)$ — reward model

We’ll learn these jointly, using encoder networks $g_\psi(o_t)$ to map $o_t \to s_t$.

---

## 📦 Model Structure

### Latent Variables:

* $s_t$: latent state
* $o_t$: observation
* $a_t$: action
* $r_t$: reward

### Learned Models:

* $s_t = g_\psi(o_t)$ — encoder (deterministic or stochastic)
* $p_\phi(s_{t+1} \mid s_t, a_t)$ — dynamics model
* $p_\phi(o_t \mid s_t)$ — reconstruction model
* $p_\phi(r_t \mid s_t)$ — reward model

### Graphical Structure:

```text
o_1 → s_1 → a_1 → s_2 → a_2 → s_3
 |      ↓         ↓         ↓
 |      r_1       r_2       r_3
```

---

## 🧪 Training Objective

Assuming a dataset $\mathcal{D} = \{ (o_t, a_t, o_{t+1}) \}$, we aim to train the model by maximizing the log-likelihood of latent transitions and reconstructions.

### 📌 Deterministic Encoder:

If we use a deterministic encoder $s_t = g_\psi(o_t)$, then:

$$
\max_{\phi,\psi} \ \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T
\log p_\phi(g_\psi(o_{t+1,i}) \mid g_\psi(o_{t,i}), a_{t,i}) +
\log p_\phi(o_{t,i} \mid g_\psi(o_{t,i})) +
\log p_\phi(r_{t,i} \mid g_\psi(o_{t,i}))
$$

Everything is differentiable, and the entire pipeline is trained by backpropagation.

### 📌 Stochastic Encoder (Variational)

For better uncertainty modeling, many methods use a **variational posterior**:

$$
q_\psi(s_t \mid o_t) \quad \text{or} \quad q_\psi(s_t, s_{t+1} \mid o_{1:T}, a_{1:T})
$$

This allows modeling epistemic uncertainty in the latent belief.

---

## 🗺️ Planning in Latent Space

Once the latent space model is trained, we can do planning directly in that space using rollout + optimization.

### Latent MPC Procedure:

1. Observe $o_t$, encode $s_t = g_\psi(o_t)$
2. For candidate action sequence $a_{t:t+H}$, rollout in latent space:

$$
s_{t+1} = f(s_t, a_t), \quad s_{t+2} = f(s_{t+1}, a_{t+1}), \dots
$$

3. Predict total reward via $\sum r(s_t, a_t)$
4. Choose action sequence maximizing reward
5. Execute first action (MPC style), observe new $o_{t+1}$, repeat

This process allows fast and differentiable lookahead.

---

## 🔐 Handling Uncertainty

Latent models can be made probabilistic to represent **uncertainty**:

* **Stochastic encoder**: $q(s_t \mid o_t)$ is a distribution (e.g., Gaussian)
* **Ensemble dynamics**: Use multiple $f_i(s, a)$ to represent epistemic uncertainty
* **KL regularization**: Encourage smooth latent posterior priors

Uncertainty can be used to:

* Improve exploration (curiosity, information gain)
* Detect OOD states
* Estimate risk-sensitive value functions

---

## 🔁 Full Model-Based RL Loop

```text
Loop every N steps:
1. Collect data using current policy: (o, a, o')
2. Train models: p(s'|s,a), p(r|s), p(o|s), encoder g(o)
3. Plan action sequence in latent space
4. Execute first action
5. Add new (o, a, o') to dataset
```

This loop enables continuous data collection and model refinement.

---

## 🧠 Final Takeaways

* Latent-space MBRL abstracts away raw pixels, enabling efficient planning
* Everything is trained jointly via backprop
* Deterministic encoder is simpler, but stochastic encoder better captures uncertainty
* Used in: Dreamer, PlaNet, SimPLe, World Models

---

## 📚 References

* Hafner et al. *Learning Latent Dynamics for Planning from Pixels*. ICML 2019 (PlaNet)
* Hafner et al. *Dream to Control: Learning Behaviors by Latent Imagination*. ICLR 2020 (Dreamer)
* Kaiser et al. *Model-Based Reinforcement Learning for Atari*. ICLR 2020 (SimPLe)
* Ha and Schmidhuber. *World Models*. NeurIPS 2018
