---

title: "Beyond KL: Other Divergences"
description: "a rough looking through available other divergence methods."
pubDatetime: 2025-09-01T11:30:00Z
tags: [machine-learning, information-theory, kl-divergence, cross-entropy, variational-inference, supervised-learning, unsupervised-learning, reinforcement-learning, Total-Variation Distance,Jensen-Shannon Divergence, Wasserstein Distance, Chi-Squared Divergence, Hellinger Distance, f-Divergence Family, Maximum Mean Discrepancy, ]
-----------------------------------------------------------------------------------------------------------------------


Understanding the Kullback-Leibler (KL) divergence is foundational in machine learning, but it is not the only way to measure the distance between probability distributions. Depending on the scenario, several other divergence or distance metrics may be more appropriate. This blog post surveys major alternatives, their definitions, properties, and usage in machine learning.

---

## 1. Total Variation Distance (TV)

**Definition**:
$TV(P, Q) = \frac{1}{2} \sum_x |P(x) - Q(x)|$

**Properties**:

* Metric ✅
* Symmetric ✅
* Bounded between 0 and 1
* Measures maximum difference in assigned probabilities

**Applications**:

* Theoretical analysis in PAC learning
* Fairness, privacy, and robustness

---

## 2. Jensen-Shannon Divergence (JSD)

**Definition**:
$JS(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$

**Properties**:

* Symmetric ✅
* Bounded between 0 and $\log 2$
* Square root defines a metric

**Applications**:

* Used in GANs (e.g., original GAN by Goodfellow et al., 2014)
* Language modeling and diversity measures

---

## 3. Wasserstein Distance (Earth Mover's Distance)

**Definition (1D case)**:
$W(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$

Where $\Pi(P, Q)$ is the set of all joint distributions with marginals P and Q.

**Properties**:

* Metric ✅
* Defined even for disjoint supports ❗
* Captures geometry of distributions

**Applications**:

* Wasserstein GANs (WGAN)
* Domain adaptation
* Out-of-distribution detection

---

## 4. Hellinger Distance

**Definition**:
$H^2(P, Q) = \frac{1}{2} \sum_x \left(\sqrt{P(x)} - \sqrt{Q(x)}\right)^2$

**Properties**:

* Symmetric ✅
* Metric ✅
* Bounded between 0 and 1

**Applications**:

* Variational inference
* Theoretical guarantees

---

## 5. Chi-Squared Divergence

**Definition**:
$\chi^2(P \parallel Q) = \sum_x \frac{(P(x) - Q(x))^2}{Q(x)}$

**Properties**:

* Not symmetric ❌
* Very sensitive to small Q(x) values ❗

**Applications**:

* Second-order information analysis
* Gradient dynamics studies

---

## 6. f-Divergence Family

Many divergences, including KL, JS, TV, and Chi-Squared, are instances of the **f-divergence** family:

**General Form**:
$D_f(P \parallel Q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right) dx$

**Examples**:

* KL: $f(t) = t \log t$
* Total Variation: $f(t) = |t - 1|$
* Chi-Squared: $f(t) = (t - 1)^2$
* Hellinger: $f(t) = (\sqrt{t} - 1)^2$

---

## 7. Maximum Mean Discrepancy (MMD)

**Definition**:
$\text{MMD}(P, Q) = \sup_{f \in \mathcal{F}} \left( \mathbb{E}_{P}[f(x)] - \mathbb{E}_{Q}[f(x)] \right)$

Where $\mathcal{F}$ is a unit ball in a Reproducing Kernel Hilbert Space (RKHS).

**Properties**:

* Metric ✅
* Kernel-based
* No density estimation needed

**Applications**:

* Two-sample tests
* Representation learning
* MMD GANs

---

## Summary Comparison

| Name          | Symmetric | Metric | Requires Density? | Handles Disjoint? | Use Case                           |
| ------------- | --------- | ------ | ----------------- | ----------------- | ---------------------------------- |
| KL Divergence | ❌         | ❌      | Yes               | No                | Inference, supervised learning     |
| JS Divergence | ✅         | √      | Yes               | Yes               | GANs, model diversity              |
| Wasserstein   | ✅         | ✅      | No                | Yes               | WGANs, robustness                  |
| TV Distance   | ✅         | ✅      | Yes               | Yes               | Theoretical bounds                 |
| Hellinger     | ✅         | ✅      | Yes               | Yes               | Variational analysis               |
| Chi-Squared   | ❌         | ❌      | Yes               | No                | Gradient-sensitive objectives      |
| MMD           | ✅         | ✅      | No                | Yes               | Kernel methods, sample-based tests |

---

## References

* Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
* Nowozin, S., Cseke, B., & Tomioka, R. (2016). *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*. [arXiv:1606.00709](https://arxiv.org/abs/1606.00709)
* Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
* Gretton, A., et al. (2012). *A Kernel Two-Sample Test*. \[Journal of Machine Learning Research]
* Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. \[NeurIPS 2014]

---

If KL divergence opened the door to understanding model performance through the lens of information theory, these alternative distances reveal the rich terrain beyond that threshold.

