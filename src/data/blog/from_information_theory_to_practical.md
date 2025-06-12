---
title: "From Information Theory to Practical RL"
pubDatetime: 2025-06-10T09:00:00Z
description: "This post introduces how core concepts from information theory like mutual information and information gain are approximated and implemented in real-world reinforcement learning algorithms."
tags: [rl, information-theory, exploration, heuristic, formalism]
---

**Mutual Information (MI)** quantifies the amount of shared information between two variables:

$$
\mathcal{I}(X; Y) = D_{\text{KL}}(p(x, y) \parallel p(x)p(y))
$$

This measures how different the joint distribution is from the product of marginals—i.e., how far X and Y are from being independent. In RL, MI is widely used for representation learning and exploration. Methods like InfoNCE and CPC approximate MI through contrastive objectives, bypassing the need to estimate explicit PDFs.

---

**Information Gain (IG)** measures how much observing a new transition changes the agent’s belief over environment dynamics:

$$
\text{IG}(\theta; s' \mid s, a) = D_{\text{KL}}(p(\theta \mid h, s, a, s') \parallel p(\theta \mid h))
$$

In practice, methods like VIME use variational approximations, replacing the true posterior with a tractable distribution \(q(\theta \mid \phi)\), and compute the KL between updated and previous beliefs as an intrinsic reward. This is theoretically elegant but computationally costly.

---

**Prediction Gain** is a lighter alternative. It rewards states that cause a large update in the density model, often implemented as:

$$
\log p_{\theta'}(s) - \log p_\theta(s)
$$

This signal appears in density-based methods and curiosity modules like NGU.

---

**Entropy** in policy distributions is used in MaxEnt RL to encourage stochasticity:

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi [r(s,a)] + \alpha \cdot \mathcal{H}(\pi)
$$

This is the core idea behind Soft Actor-Critic (SAC), which performs well in continuous control tasks by ensuring diverse policy behaviors.

---

**Surprise** quantifies how unexpected a transition is:

$$
\text{Surprise}(s') = -\log p(s' \mid s,a)
$$

When the true model is unavailable, surprise is approximated via prediction error. ICM uses a forward model loss, while RND compares outputs between a fixed random network and a trained predictor.

---

**KL Divergence** is not an exploration bonus but is crucial for stable policy updates. TRPO constrains KL explicitly, while PPO approximates it via clipping. These mechanisms keep policy shifts conservative and reliable.

---

**Latent Consistency** ensures that similar states are embedded close in latent space:

$$
D_{\text{KL}}(q(z \mid s)\|q(z \mid s'))
$$

Used in world model methods like Dreamer and DeepMDP, this regularization helps maintain smooth, meaningful latent dynamics.

---

### References

[1] van den Oord et al., "Representation Learning with Contrastive Predictive Coding", arXiv:1807.03748  
[2] Houthooft et al., "VIME: Variational Information Maximizing Exploration", NeurIPS 2016  
[3] Badia et al., "Never Give Up: Learning Directed Exploration Strategies", ICLR 2020  
[4] Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905  
[5] Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", ICML 2017  
[6] Burda et al., "Exploration by Random Network Distillation", ICLR 2019  
[7] Schulman et al., "Trust Region Policy Optimization", ICML 2015  
[8] Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347  
[9] Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination", ICLR 2020  
[10] Gelada et al., "DeepMDP: Learning Continuous Latent Space Models for Representation Learning", ICML 2019
