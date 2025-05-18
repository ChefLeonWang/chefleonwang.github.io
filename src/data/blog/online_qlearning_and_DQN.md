---

title: "Is DQN an Online Q-learning Algorithm? Exploring the Boundary"
pubDatetime: 2025-05-18T09:00:00Z
description: "Is DQN an online Q-learning method? What if the minibatch size is 1? This post explores when DQN behaves like an online algorithm and clarifies how it connects to classical Q-learning."
tags: [RL, DQN, Q-learning, online q learning, deep RL]
---

All from GPT, it looks good for now:)

## Is Q-learning an online algorithm?

**Yes.** Classic **tabular Q-learning** is the textbook example of an **online reinforcement learning algorithm**:

* After each step, you **immediately update** the Q-value:

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]
```

* No replay buffer, no batching
* Learning happens right after interaction

✅ So: **Tabular Q-learning is truly online.**

---

## What about DQN?

DQN = Deep Q-Learning. It differs from classic Q-learning in key ways:

* It uses a **neural network** to approximate $Q(s,a)$
* It relies on a **replay buffer** to store past transitions
* It uses a **target network** for stable TD targets
* It optimizes a loss using **minibatch stochastic gradient descent**:

```math
\mathcal{L}(\phi) = \left( Q_\phi(s,a) - [r + \gamma \max_{a'} Q_{\phi_{\text{target}}}(s', a')] \right)^2
```

❌ So: **DQN is not an online method by default**. It is **off-policy, batch-based Q-learning**.

---

## What if the minibatch size is 1?

👉 If you configure DQN to:

* Disable the replay buffer
* Sample one transition at a time
* Update the Q-network immediately (minibatch size = 1)

Then you are essentially doing:

> ✅ **Online Q-learning with function approximation**

You're performing step-wise SGD on TD targets.

---

## ❗Caveats

1. DQN **still uses a target network**, which introduces a delay → still technically **off-policy**.
2. Even with batch size = 1, you're using a **nonlinear function approximator**, which makes learning less stable — this is part of the classic "deadly triad" in RL.

So while this setup is "online" in behavior, it differs from classical Q-learning in key ways. People usually call this:

> **Online Deep Q-learning** or **semi-gradient TD with function approximation**

---

## ✅ Summary Table

| Configuration                | Is it online? | Notes                           |
| ---------------------------- | ------------- | ------------------------------- |
| Tabular Q-learning           | ✅ Yes         | Fully online TD updates         |
| DQN (default)                | ❌ No          | Uses replay + batch updates     |
| DQN with no buffer + batch=1 | ✅ Yes         | Online SGD on TD error          |
| Fitted Q-Iteration           | ❌ No          | Fully offline, batch Q-learning |

 So: **DQN behaves like an online Q-learning algorithm only in the extreme case of batch size = 1 with no replay buffer.**

