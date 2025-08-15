---
title: "Classic Interpolation Methods: Definitions · Formulas · Applications"
description: "From linear interpolation to spherical, from polynomial to exponential smoothing — a complete guide to classic interpolation techniques with formulas, definitions, and applications in ML, graphics, and signal processing."
pubDatetime: 2025-08-15T11:30:00Z
tags: [interpolation, machine learning]
---

## Introduction

**Interpolation** is the process of estimating unknown values between known data points.  
It appears not only in mathematics and computer graphics but also in machine learning, signal processing, animation, and game development.

Here summarizes **classic interpolation methods** with **definitions, formulas**, and **applications**, so that we can choose the right tool for our task.

---

## 1. Linear Interpolation (LERP)

**Definition**  
Given two points \(a\) and \(b\), interpolate proportionally with \(\lambda \in [0,1]\).

**Formula**
\[
\text{LERP}(a,b,\lambda) = (1-\lambda)\,a + \lambda\,b
\]

**Applications**
- ML: feature mixing (e.g., Mixup)
- Games/animation: smooth transitions of object positions
- Graphics: gradient color blending

**Reference**  
- Foley, J. D., et al. *Computer Graphics: Principles and Practice*. Addison-Wesley, 1996.

---

## 2. Spherical Linear Interpolation (SLERP)

**Definition**  
Interpolation on the surface of a unit sphere, keeping vector magnitude constant, often for smooth rotation.

**Formula**
\[
\text{SLERP}(a,b,\lambda) =
\frac{\sin((1-\lambda)\theta)}{\sin\theta}a +
\frac{\sin(\lambda\theta)}{\sin\theta}b
\]
where \(\theta = \cos^{-1}(a\cdot b)\).

**Applications**
- 3D rotation interpolation (quaternions)
- Smooth latent space transitions (e.g., VAE, StyleGAN)

**Reference**  
- Shoemake, Ken. "Animating rotation with quaternion curves." *ACM SIGGRAPH Computer Graphics* 19.3 (1985): 245–254.

---

## 3. Polynomial Interpolation

**Definition**  
Fit a single polynomial that exactly passes through all known points.

**Formula (Lagrange form)**
\[
P(x) = \sum_{i=0}^n y_i \prod_{\substack{0 \le j \le n \\ j \ne i}} \frac{x - x_j}{x_i - x_j}
\]

**Applications**
- Numerical computation
- High-precision curve fitting
- Reconstructing historical datasets

**Reference**  
- Stoer, J., and R. Bulirsch. *Introduction to Numerical Analysis*. Springer, 2002.

---

## 4. Spline Interpolation

**Definition**  
Divide data into segments and use low-degree polynomials (e.g., cubic splines) to interpolate smoothly across the whole range.

**Formula (Cubic spline example)**
\[
S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
\]
defined on each interval \([x_i, x_{i+1}]\).

**Applications**
- CAD and computer graphics curves
- Smooth time-series paths
- Video keyframe interpolation

**Reference**  
- De Boor, Carl. *A Practical Guide to Splines*. Springer, 1978.

---

## 5. Bilinear and Bicubic Interpolation

**Definition**  
For 2D grid data, interpolate linearly or cubically along both x and y axes.

**Formula (Bilinear)**  
First interpolate in the x-direction, then interpolate the results in the y-direction.

**Applications**
- Image scaling and rotation
- Data preprocessing in deep learning
- Downsampling/upsampling in convolutional layers

**Reference**  
- Keys, R. G. "Cubic convolution interpolation for digital image processing." *IEEE Transactions on Acoustics, Speech, and Signal Processing* 29.6 (1981): 1153–1160.

---

## 6. Kriging Interpolation

**Definition**  
A geostatistical interpolation method that uses spatial correlation for weighted averaging.

**Formula (general form)**
\[
\hat{Z}(x_0) = \sum_{i=1}^n \lambda_i Z(x_i)
\]
where weights \(\lambda_i\) come from variogram modeling.

**Applications**
- Geographic Information Systems (GIS)
- Air quality, meteorology prediction
- Underground resource estimation

**Reference**  
- Krige, D. G. "A statistical approach to some basic mine valuation problems on the Witwatersrand." *Journal of the Chemical, Metallurgical and Mining Society of South Africa* 52.6 (1951): 119–139.

---

## 7. Exponential Moving Average (EMA)

**Definition**  
A recursive interpolation method combining the smoothed past value and the current value, with historical weights decaying exponentially.

**Formula**
\[
\theta_{\text{EMA}}^{(t)} = \beta\,\theta_{\text{EMA}}^{(t-1)} + (1-\beta)\,\theta_{\text{current}}^{(t)}
\]
where \(\beta \in [0,1)\) controls smoothing.

**Applications**
- ML: parameter smoothing (GAN, Diffusion models)
- Training metric smoothing
- Finance: stock price smoothing

**Reference**  
- Brown, R. G. *Smoothing, Forecasting and Prediction of Discrete Time Series*. Prentice-Hall, 1963.

---

## 8. Piecewise Linear Interpolation

**Definition**  
Apply linear interpolation between each pair of adjacent points; the result is a set of connected straight lines.

**Formula**  
On interval \([x_i, x_{i+1}]\):
\[
y = y_i + \frac{y_{i+1}-y_i}{x_{i+1}-x_i}(x-x_i)
\]

**Applications**
- Simple and efficient real-time interpolation
- Keyframe animation
- Embedded systems

**Reference**  
- Burden, R. L., and J. D. Faires. *Numerical Analysis*. Cengage Learning, 2010.

---

## 9. Data Mixup Interpolation

**Definition**  
A data augmentation method in deep learning that mixes two samples and their labels proportionally.

**Formula**
\[
\tilde{x} = \lambda x_i + (1-\lambda) x_j,\quad
\tilde{y} = \lambda y_i + (1-\lambda) y_j
\]

**Applications**
- Image classification generalization
- Adversarial training stabilization
- Reducing overfitting

**Reference**  
- Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." *International Conference on Learning Representations* (ICLR), 2018.

---

## Summary Table

| Method | Core Idea | Formula | Applications | Reference |
|---|---|---|---|---|
| LERP | Proportional blending | Weighted sum | Animation, ML feature mixing | Foley et al., 1996 |
| SLERP | Constant-magnitude blending on sphere | Spherical trig | Rotation, latent space | Shoemake, 1985 |
| Polynomial | Global polynomial fit | Lagrange form | High-precision fitting | Stoer & Bulirsch, 2002 |
| Spline | Piecewise smooth polynomials | Cubic polynomials | CAD, curves | De Boor, 1978 |
| Bilinear/Bicubic | 2D grid interpolation | Linear/cubic in both axes | Image scaling | Keys, 1981 |
| Kriging | Spatially correlated weights | Variogram model | GIS, geostatistics | Krige, 1951 |
| EMA | Recursive exponential blending | Weighted history | ML, finance | Brown, 1963 |
| Piecewise Linear | Local linear segments | Per-interval LERP | Real-time interpolation | Burden & Faires, 2010 |
| Mixup | Sample & label blending | LERP on data | Deep learning | Zhang et al., 2018 |

---

## Closing Notes

Interpolation methods are not isolated techniques — they connect **numerical computation → signal processing → machine learning → computer graphics**.  
By understanding their **mathematical definitions, formulas, and applications**, you can choose the most suitable method and avoid misuse.
