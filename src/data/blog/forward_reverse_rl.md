---
title: "Forward vs. Reverse KL Divergence: Why Machine Learning Chooses One Over the Other"
description: "Exploring the difference between forward and reverse KL divergence, and why maximum likelihood always corresponds to the forward direction."
pubDatetime: 2025-09-18T11:30:00Z
tags: [machine-learning, information-theory, KL-divergence, MLE, forward-KL, reverse-KL]

---

When we train probabilistic models, one of the most fundamental goals is to make the model distribution \(p_\theta(x)\) approximate the true (but unknown) data distribution \(p_{\text{data}}(x)\).  

A natural way to measure discrimination/closeness between distributions is the **Kullback–Leibler (KL) divergence**. But there are two directions:  

$$
D_{KL}(p_{\text{data}} \| p_\theta) 
\quad \text{and} \quad
D_{KL}(p_\theta \| p_{\text{data}}).
$$

At first glance, both look reasonable. Yet in practice, almost all maximum likelihood training corresponds to the **forward KL** \(D_{KL}(p_{\text{data}} \| p_\theta)\). 

Why not the reverse?   

---

##  Forward KL

$$
D_{KL}(p_{\text{data}} \| p_\theta) 
= \mathbb{E}_{x \sim p_{\text{data}}} \big[\log p_{\text{data}}(x) - \log p_\theta(x)\big].
$$

Expanding this:  

$$
= \underbrace{\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(x)]}_{\text{independent of } \theta}
- \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)].
$$

- The first term depends only on the true data distribution and is **constant w.r.t. \(\theta\)**.  
- The second term is what matters.  

So optimizing forward KL is equivalent to:  

$$
\max_\theta \;\; \mathbb{E}_{x\sim p_{\text{data}}}[\log p_\theta(x)].
$$

This is exactly **Maximum Likelihood Estimation (MLE)**.  

---

##  Reverse KL

$$
D_{KL}(p_\theta \| p_{\text{data}})
= \mathbb{E}_{x \sim p_\theta}\big[\log p_\theta(x) - \log p_{\text{data}}(x)\big].
$$

Expanding this:  

$$
= \mathbb{E}_{p_\theta}[\log p_\theta(x)] 
- \mathbb{E}_{p_\theta}[\log p_{\text{data}}(x)].
$$

Now here’s the subtlety:  

- Yes, \(\log p_{\text{data}}(x)\) doesn’t explicitly depend on \(\theta\).  
- But the expectation is taken **under \(p_\theta\)**, which *does* depend on \(\theta\).  
- So the gradient involves:
  $$
  \nabla_\theta \mathbb{E}_{p_\theta}[\log p_{\text{data}}(x)] 
  = \int \nabla_\theta p_\theta(x) \log p_{\text{data}}(x) \, dx.
  $$
- To evaluate this, we would need to know \(\log p_{\text{data}}(x)\) at arbitrary \(x \sim p_\theta\).  

But we don’t know \(p_{\text{data}}\)! We can sample from it (via real data), but we can’t compute its density at arbitrary model samples.  
👉 That’s why reverse KL is **intractable**.  

---

##  Why Forward KL Works but Reverse KL Doesn’t
- **Forward KL**: expectation is taken under the fixed real distribution \(p_{\text{data}}\). The \(\log p_{\text{data}}(x)\) term becomes a constant. Training only requires computing \(\log p_\theta(x)\) on data samples.  
- **Reverse KL**: expectation is taken under the moving model distribution \(p_\theta\). Now \(\log p_{\text{data}}(x)\) is inside the expectation, and we need to know it at model-generated points. That’s impossible unless we already know the true distribution.  

So the asymmetry is not philosophical—it’s purely about **computational feasibility**.  

---

##  Table Summary

| Divergence | Expectation taken under | Requires evaluating | Tractable? | Typical use |
|------------|-------------------------|---------------------|-------------|-------------|
| \(D_{KL}(p_{\text{data}} \| p_\theta)\) | Real data \(p_{\text{data}}\) | \(\log p_\theta(x)\) | ✅ Yes | MLE, supervised learning |
| \(D_{KL}(p_\theta \| p_{\text{data}})\) | Model \(p_\theta\) | \(\log p_{\text{data}}(x)\) | ❌ No (unless density known) | Approximated via adversarial methods (GANs, f-GAN) |

---

##  Intuition  
- Forward KL says: *“Given real data samples, how likely would the model have generated them?”*  
- Reverse KL says: *“Given model samples, how likely are they under the real data distribution?”*  

The second question cannot be answered unless you already know the real distribution.
> And if you did, there’d be **NO NEED**** to train a model.  

---

## References

- Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.  
- Kullback, S., & Leibler, R. A. (1951). *On Information and Sufficiency*. Annals of Mathematical Statistics.  
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.  
- Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NeurIPS.  
- Nowozin, S., Cseke, B., & Tomioka, R. (2016). *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*. NeurIPS.  
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.  

---

### Closing Thought

In machine learning, we don’t choose forward KL because it’s “better,” but because it’s the only one we can actually compute with real-world data.  


