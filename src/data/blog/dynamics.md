---
title: "Dynamics: from Physics to Reinforcement Learning and Machine Learning"
description: "some random thinking"
pubDatetime: 2025-09-19T11:30:00Z
tags: [dynamics, reinforcement-learning, machine-learning, mathematics]

---

# Definition of Dynamics  

**Dynamics** (from Greek *dynamis*, meaning “power, force”) refers to:  

> **The study or model of how a system evolves over time under certain rules or forces.**  

General forms:  

- **Deterministic dynamics**:  
  $$
  x_{t+1} = f(x_t, a_t)
  $$ 
- **Stochastic dynamics**:  
  $$
  x_{t+1} \sim P(\cdot \mid x_t, a_t)
  $$

Here \(x_t\) is the state, \(a_t\) is an external input/force/action, and \(f\) or \(P\) describes the evolution rule.  

---

# Physics  

**Definition**: Describes the time evolution of objects, fields, or particles.  

- **Classical mechanics**:  
  $$
  m \frac{d^2 x}{dt^2} = F(x,t)
  $$ 
  **Applications**: orbital prediction, engineering systems.  

- **Quantum dynamics**:  
  $$
  i\hbar \frac{\partial}{\partial t}\Psi(x,t) = \hat{H}\Psi(x,t)
  $$ 
  **Applications**: electron motion, quantum computation, spectroscopy.  

---

# Chemistry  

**Definition**: Describes the time evolution of reactions and molecular systems.  

- **Chemical kinetics (rate equations)**:  
  $$
  \frac{d[C]}{dt} = k [A][B]
  $$ 
  **Applications**: reaction engineering, catalyst design, drug kinetics.  

- **Molecular dynamics (MD)**:  
  $$
  m_i \frac{d^2 r_i}{dt^2} = - \nabla V(r_1, r_2, \dots, r_N)
  $$
  **Applications**: materials science, protein folding, drug discovery.  

---

# Biology  

**Definition**: Time evolution of populations, cells, or ecosystems.  

- **Population dynamics (logistic growth)**:  
  $$
  \frac{dN}{dt} = r N \left(1 - \frac{N}{K}\right)
  $$
  **Applications**: ecology, conservation, epidemiology.  

- **Neuronal dynamics (Hodgkin–Huxley)**:  
  $$
  C_m \frac{dV}{dt} = - \sum I_{\text{ion}} + I_{\text{ext}}
  $$
  **Applications**: neuroscience, spiking models, brain simulation.  

---

# Finance  

**Definition**: Stochastic evolution of assets, interest rates, or markets.  

- **Geometric Brownian Motion (GBM)**:  
  $$
  dS_t = \mu S_t dt + \sigma S_t dW_t
  $$ 
  **Applications**: option pricing (Black–Scholes), risk management.  

- **Vasicek interest rate model**:  
  $$
  dr_t = a(b - r_t) dt + \sigma dW_t
  $$
  **Applications**: bond pricing, interest rate derivatives.  

---

# Time Series  

**Definition**: Dynamics expressed as dependence on past states.  

- **Autoregressive model (AR(p))**:  
  $$
  x_t = \sum_{i=1}^p \phi_i x_{t-i} + \epsilon_t
  $$
  **Applications**: forecasting, language modeling, econometrics.  

---

# Machine Learning  

**Definition**: Dynamics as the generative or latent process of data.  

- **Reparameterization trick (pathwise derivative)**:  
  $$
  z = \mu_\theta(x) + \sigma_\theta(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
  $$ 
  **Applications**: variational autoencoders, continuous policy gradients.  

- **Neural ODEs**:  
  $$
  \frac{dz}{dt} = f_\theta(z,t)
  $$ 
  **Applications**: continuous-time models, irregular time-series prediction.  

- **Diffusion models**:  
  Iterative dynamics of adding/removing noise.  
  **Applications**: generative modeling (image, audio, molecules).  

---

# Reinforcement Learning  

**Definition**: Environment dynamics — how the world evolves given actions.  

- **Markov Decision Process (MDP)**:  
  $$
  P(s_{t+1}, r_t \mid s_t, a_t)
  $$

- **Model-based RL**: explicitly learns the dynamics for planning.  
- **Model-free RL**: skips explicit dynamics, directly learns the policy.  

**Applications**: robotics, games, recommendation, control systems.  

---

# References  

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*.  
2. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *ICLR*.  
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.  
4. Feynman, R. P. (1948). Space-time approach to non-relativistic quantum mechanics. *Rev. Mod. Phys.*  
5. Allen, M. P., & Tildesley, D. J. (2017). *Computer Simulation of Liquids*.  
6. Björk, T. (2009). *Arbitrage Theory in Continuous Time*.  

---
