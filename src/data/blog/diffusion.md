---

title: "Go through Diffusion"
pubDatetime: 2025-08-17T11:30:00Z
description: "step by step"
tags: [rl, udl, diffusion]
-------------------------------------------------------------------------------


## 0. Introduction

Diffusion models have emerged as one of the most powerful generative modeling frameworks in recent years, producing state-of-the-art image, video, and audio samples.  
At first glance, their training objective looks deceptively simple: **predict Gaussian noise with MSE loss**.  

But how does this connect to **maximum likelihood estimation** (MLE), which is the standard principle behind autoregressive models, VAEs, and normalizing flows?  

We’ll walk through the **full derivation**: from log-likelihood → ELBO → Gaussian posteriors → noise-prediction loss.

---

## 1. Notation and Setup

- Data: $$(x_0 \sim p_{\text{data}}(x_0))v.  $$
- Noise schedule: $$({\beta_t\}_{t=1}^T)$$ , with $$ (\alpha_t = 1-\beta_t)$$ , and $$ (\bar\alpha_t = \prod_{s=1}^t \alpha_s). $$ 

---

## 2. Forward Diffusion (Noising Process)

We define a Markov chain that progressively adds Gaussian noise:
$$
q(x_t | x_{t-1}) = \mathcal N(\sqrt{\alpha_t}\,x_{t-1}, \, \beta_t I).

$$
Chained together:
$$

q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1}).

$$
Closed-form reparameterization:
$$

q(x_t \mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0, \, (1-\bar\alpha_t) I),

$$
$$

x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal N(0,I).

$$
---

## 3. Reverse Diffusion (Denoising Process)

We want to model the reverse chain:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t),
$$

with Gaussian transitions:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal N(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t)).
$$

---

## 4. Variational Lower Bound (ELBO)

We lower bound the log-likelihood:

$$
\log p_\theta(x_0) \ge 
\mathbb E_q \left[ \log p_\theta(x_{0:T}) - \log q(x_{1:T}\mid x_0) \right].
$$

Expanding yields:

$$
\mathcal L_{\text{VLB}} =
\underbrace{\mathrm{KL}(q(x_T \mid x_0) \,\|\, p(x_T))}_{L_T}
+ \sum_{t=2}^T \underbrace{\mathbb E_q \big[ \mathrm{KL}(q(x_{t-1}\mid x_t,x_0) \,\|\, p_\theta(x_{t-1}\mid x_t)) \big]}_{L_t}
- \underbrace{\mathbb E_q[\log p_\theta(x_0 \mid x_1)]}_{L_0}.
$$

---

## 5. The True Posterior

Since both forward and reverse are Gaussian, the posterior is closed-form:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal N(\tilde\mu_t(x_t, x_0), \, \tilde\beta_t I),
$$

where

$$
\tilde\mu_t(x_t,x_0) =
\frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1-\bar\alpha_t}\,x_0
+ \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
$$

$$
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$

---

## 6. Simplifying KL Terms

With fixed variance \(\Sigma_\theta(x_t,t)=\tilde\beta_t I\), the KL reduces to a squared error:

$$
L_t = \mathbb E_q \left[ \frac{1}{2\tilde\beta_t} \|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\|^2 \right] + \text{const}.
$$

---

## 7. Noise Prediction Parameterization

Recall:

$$
x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon.
$$

We rearrange to express \(x_0\):

$$
x_0 = \frac{1}{\sqrt{\bar\alpha_t}} \left(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon \right).
$$

Substituting into the posterior mean, we can define:

$$
\mu_\theta(x_t,t) = 
\frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t,t)\right).
$$

---

## 8. The Noise Prediction Loss

Now the KL term becomes:

$$
L_t = \mathbb E_{x_0,\epsilon} \left[ \frac{\alpha_t^2}{2\tilde\beta_t (1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta(x_t,t)\|^2 \right] + \text{const}.
$$

The prefactor depends only on the noise schedule.  
Thus the practical training loss is simply:

$$
\boxed{L_{\text{simple}}(\theta) = \mathbb E_{x_0, t, \epsilon} \big[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \big]}
$$

---

## 9. Sampling

After training, we generate by reversing the chain:

1. Start with \(x_T \sim \mathcal N(0,I)\).  
2. Iteratively compute:

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z\sim\mathcal N(0,I).
$$

3. After \(T\) steps, obtain \(x_0\).  

---

## 10. Summary

- Diffusion models are **likelihood-based generative models**.  
- Training uses a **noise-prediction objective**, but it is mathematically equivalent to maximizing a variational lower bound on \(\log p_\theta(x_0)\).  
- The trick: reformulate MLE into a **simple regression problem** of predicting Gaussian noise.  

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.  
2. Ho, J., Salimans, T. (2021). *Classifier-Free Diffusion Guidance*. arXiv:2207.12598.  
3. Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. arXiv:2010.02502.  
4. Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML.  
5. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR.  
6. Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS.  

