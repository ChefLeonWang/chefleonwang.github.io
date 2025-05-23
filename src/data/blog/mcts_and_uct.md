---

title: "MCTS and UCT: Monte Carlo Tree Search in Discrete Action Spaces"
pubDatetime: 2025-05-23T23:00:00Z
description: "A detailed walk-through of MCTS in discrete settings, including algorithmic steps, math formulations, and tree policy intuition."
tags: [MCTS, Tree Search, UCT, Planning, AlphaGo, Discrete RL]
---------------------------------------------------------------

## 🌲 What is Monte Carlo Tree Search (MCTS)?

Monte Carlo Tree Search (MCTS) is a powerful search algorithm for decision-making in large, stochastic, or partially observable domains. It has been notably used in board games (e.g., Go, Chess), reinforcement learning, and robotic planning.

MCTS builds a **partial search tree** and uses **simulations** (Monte Carlo rollouts) to estimate the value of actions and guide the tree expansion.

It balances **exploration** (trying new actions) and **exploitation** (focusing on promising actions) using the UCT (Upper Confidence Trees) strategy.

---

## 🧩 Problem Setting

Assume a discrete Markov Decision Process (MDP):

* Finite state space $\mathcal{S}$
* Discrete action space $\mathcal{A}$
* Transition model $P(s'|s, a)$
* Reward function $r(s, a)$

We want to find an action $a_t \in \mathcal{A}$ from current state $s_t$ that maximizes expected cumulative reward:

```math
\pi^*(s_t) = \arg\max_a \mathbb{E}\left[ \sum_{k=0}^T \gamma^k r(s_{t+k}, a_{t+k}) \right]
```

---

## 🧭 MCTS Core Loop

Each MCTS iteration involves four steps:

### 1. **Selection**

* Start at root $s_1$, select actions down the tree according to the **tree policy** until a leaf $s_l$ is reached

### 2. **Expansion**

* If $s_l$ is not terminal and not fully expanded, add one of its children $s_{l+1}$

### 3. **Simulation** (Rollout)

* From $s_{l+1}$, use a **default policy** (e.g., random) to simulate to the end of the episode or for a fixed depth

### 4. **Backpropagation**

* Propagate the result of the simulation up the tree to update $Q(s), N(s)$ for all nodes in the visited path

---

## 📐 UCT Tree Policy

To choose which child to expand, MCTS uses the UCT formula:

```math
\text{Score}(s) = \frac{Q(s)}{N(s)} + 2C \sqrt{\frac{2 \ln N(s_{\text{parent}})}{N(s)}}
```

Where:

* $Q(s)$: total value from rollouts
* $N(s)$: visit count
* $C$: exploration constant
* First term: exploitation (mean value)
* Second term: exploration (bonus for less-visited nodes)

This balances the **explore-exploit tradeoff**.

---

## 🧱 Example Tree Diagram (see illustration)

We start at root $s_1$, simulate down action branches $a_1 = 0, 1$, reach $s_2$, take $a_2 = 0, 1$, reach $s_3$, simulate and collect rewards.

Each leaf node $s_3$ is evaluated:

* Monte Carlo return is stored in $Q(s)$
* Visit counts $N(s)$ incremented
* Results are propagated up to $s_1$

---

## 🔁 Is MCTS Open-Loop or Closed-Loop?

* **MCTS itself is open-loop** in the sense that it simulates fixed action sequences from a root state
* But if **re-run at every decision step**, MCTS becomes **part of a closed-loop control system**

This is how it works in AlphaGo and AlphaZero:

* Each turn, MCTS is invoked using the **current board state**
* Resulting policy distribution is used to select the next move

---

## 💡 Advantages of MCTS

* No need for explicit value function
* Scales to very large state/action spaces
* Can be combined with neural networks (as in AlphaZero)
* Naturally handles stochasticity via sampling

---

## 📚 References

* Coulom (2006). *Efficient Selectivity and Backup in Monte-Carlo Tree Search*
* Kocsis & Szepesvári (2006). *Bandit-based Monte-Carlo Planning*
* Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search*
* Browne et al. (2012). *A Survey of Monte Carlo Tree Search Methods*
