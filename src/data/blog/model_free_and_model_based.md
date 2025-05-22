---

title: "Model-Free vs Model-Based RL: Black Boxes, Internal Models, and the Naming Irony"
pubDatetime: 2025-05-22T20:00:00Z
description: "A detailed, intuitive and technical comparison between model-free and model-based reinforcement learning."
tags: [RL, Model-Free, Model-Based, Policy Learning, Transition Model， Natural Policy Gradient, TRPO, PPO, SAC]
----------------------------------------------------------------------------------

## 🧭 Overview

Reinforcement learning (RL) methods are often categorized as **model-free** or **model-based**. These terms refer to whether the agent tries to **learn or use a model of the environment's dynamics**.

Yet, the names can be unintuitive:

> 🤔 "Model-free" agents still use neural networks (which are models!)
> 🤯 They still need the environment — so shouldn't it be "environment-free"?


---

## 🧱 Definitions

### 🔹 Model-Based RL

> **The agent uses or learns a model of the environment** — typically the transition function:

```math
P(s' \mid s, a)
```

This model can be:

* **Given** (if you have a simulator)
* **Learned** from data

The agent then uses the model to:

* Simulate future outcomes (rollouts)
* Plan ahead (e.g., via MCTS, MPC)
* Improve data efficiency

### 🔸 Model-Free RL

> **The agent learns directly from experience** — no internal model of how the environment works.

The agent only observes:

```text
(s_t, a_t, r_t, s_{t+1}) from env.step(a_t)
```

And uses this to:

* Learn a value function (e.g., Q-learning, actor-critic)
* Learn a policy (e.g., PPO, A2C)
* Estimate gradients, advantages, etc.

---

## 📦 Environment as a Black Box

In model-free RL:

* The **environment is a black box**
* The agent **does not attempt to understand how it works**
* The agent just collects samples and learns from them

This is why model-free methods are simple and flexible — they just need the ability to interact with the environment.

---

## 🧠 Model-Free ≠ No Neural Network

Let’s clear this up:

* You can use a neural network as a **policy** or **value function** in model-free RL
* But this is **not** the "model" we're talking about

> The "model" in "model-free" ≠ function approximator
> It means: ❌ no **transition model** of the environment

So PPO, TRPO, A2C, SAC — all model-free, even though they use deep networks.

---

## 🔄 Comparison Table

| Feature                                       | Model-Free RL              | Model-Based RL            |
| --------------------------------------------- | -------------------------- | ------------------------- |
| Learns/uses transition model $P(s' \mid s,a)$ | ❌ No                       | ✅ Yes                     |
| Uses reward samples from env                  | ✅ Yes                      | ✅ Yes                     |
| Can simulate rollouts without env             | ❌ No                       | ✅ Yes                     |
| Data efficient                                | ❌ Lower                    | ✅ Higher                  |
| Sample efficient                              | ❌ No                       | ✅ Yes                     |
| Planning                                      | ❌ None                     | ✅ Often used              |
| Complexity                                    | ✅ Simpler                  | ❌ More complex            |
| Examples                                      | Q-learning, PPO, TRPO, SAC | MuZero, PETS, MPC, Dyna-Q |

---

## 🤹 Naming Irony

> “Model-free” doesn’t mean the agent has no model —
> it just means the agent has **no model of the environment**.

So while your agent might be:

* A neural network (yes, a model!)
* Learning value functions
* Using gradients and SGD

…it’s still **model-free** in the RL sense because it doesn’t attempt to predict $s'$ from $s, a$.

Meanwhile, **model-based** methods learn exactly that prediction — and use it for planning, rollouts, or reward shaping.

---

## 🧠 Mental Model Summary

| Think of...             | Model-Free RL       | Model-Based RL               |
| ----------------------- | ------------------- | ---------------------------- |
| The agent as a...       | Blind explorer      | Map builder                  |
| The environment as a... | Black box to probe  | System to decode             |
| Learning style          | React to experience | Plan ahead using predictions |

---

## 📚 References

* Sutton & Barto (2018). *Reinforcement Learning: An Introduction*
* Deisenroth et al. (2013). *Survey of Model-Based Reinforcement Learning*
* Silver et al. (2017). *Mastering the Game of Go without Human Knowledge*
* Ha & Schmidhuber (2018). *World Models*
* OpenAI Spinning Up (2019). *Model-Free vs Model-Based*
