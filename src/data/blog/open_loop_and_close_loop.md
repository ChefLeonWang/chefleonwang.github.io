---

title: "Open-Loop vs Closed-Loop Planning"
pubDatetime: 2025-05-23T22:30:00Z
description: "This post explores the core differences between open-loop and closed-loop planning, their underlying assumptions, typical algorithms, and use cases."
tags: [Planning, Open-Loop, Closed-Loop, Trajectory Optimization, MPC, MCTS, Control]
--------------------------------------------------------------------------------------

## 🔍 Introduction

Planning is a fundamental problem in control, robotics, and reinforcement learning. Broadly, we can divide planning algorithms into two categories based on how they deal with feedback:

* **Open-loop planning**: Plans are computed once and executed without correction.
* **Closed-loop planning**: Plans are continuously updated based on state feedback.

This distinction affects how agents deal with uncertainty, disturbance, and long-term prediction.

---

## 🔁 Closed-Loop Planning

In **closed-loop (or feedback) planning**, the agent observes the current state $s_t$ and then selects an action $a_t \sim \pi(a_t | s_t)$. The environment transitions to a new state $s_{t+1}$, and the loop continues.

### 📌 Characteristics

* Reactive to unexpected events or noise
* Typically involves a **policy** $\pi$ or a value function
* Adaptation happens at every time step

### 🔧 Examples

* **Model Predictive Control (MPC)**: Plans a trajectory but only executes the first step and replans each time
* **Reinforcement Learning agents** (e.g. PPO, SAC): Policies are updated from data but evaluated closed-loop
* **MCTS (when rerun at every state)**: Performs open-loop search, but results in closed-loop behavior

---

## 📤 Open-Loop Planning

In **open-loop planning**, the agent computes a full sequence of actions $a_{1:T}$ ahead of time using a model $f(s, a)$, and executes them without modification.

### 📌 Characteristics

* No feedback during execution
* Requires a known or learned model of dynamics
* Assumes deterministic and predictable environments

### 🛠 Common Algorithms

#### 1. **Trajectory Optimization**

* Directly optimize $a_{1:T}$ to maximize return:

  ```math
  \max_{a_{1:T}} \sum_{t=1}^T r(s_t, a_t) \quad \text{s.t.} \quad s_{t+1} = f(s_t, a_t)
  ```
* Methods: gradient descent, iLQR, shooting methods

#### 2. **Cross Entropy Method (CEM)**

* Sample many action sequences, select the best, refit a Gaussian, repeat
* Great for non-differentiable or black-box models

#### 3. **CMA-ES, Random Shooting, MPPI**

* Evolutionary or path-integral methods for searching over $a_{1:T}$
* Used in robotics and model-based control

#### 4. **A\***, Dynamic Programming (in discrete spaces)

* When the state-action space is known, search over future trajectories using a graph

---

## 🧠 Mental Model: Loop vs No Loop

| Feature                  | Closed-Loop                   | Open-Loop                     |
| ------------------------ | ----------------------------- | ----------------------------- |
| Feedback used?           | ✅ Yes, at each step           | ❌ No, only initial state      |
| Action depends on state? | ✅ Yes                         | ❌ No, all actions pre-planned |
| Reactive                 | ✅ Yes                         | ❌ No                          |
| Example agent            | RL agent, MPC, AlphaGo (MCTS) | Trajectory optimizer, CEM     |
| Planning mode            | Receding horizon              | Full-horizon rollout          |

---

## 🔄 Hybrid: Receding Horizon = Open + Feedback

Methods like **MPC** and **AlphaGo’s MCTS** are hybrids:

* They solve an **open-loop planning problem** at each step
* But then replan from new state each time → **closed-loop system**

---

## 🧮 Summary

* **Open-loop planning** is simple and powerful in known, deterministic systems, but fragile to noise.
* **Closed-loop planning** is adaptive and robust, but often harder to optimize.
* Modern systems often blend both — performing open-loop rollout inside a closed-loop framework.

Use open-loop planning when:

* You have an accurate model
* You need fast rollout

Use closed-loop planning when:

* You expect uncertainty or disturbances
* You value robustness and flexibility

---

## 📚 References

* Tassa et al. (2012). *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*
* Schulman et al. (2015). *Trust Region Policy Optimization*
* Chua et al. (2018). *Deep Reinforcement Learning in a Handful of Trials with World Models*
* Sutton & Barto. *Reinforcement Learning: An Introduction*
* Levine et al. (2016). *End-to-End Training of Deep Visuomotor Policies*
