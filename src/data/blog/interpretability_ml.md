---

title: "A  Overview of Interpretability in ML"
pubDatetime: 2025-08-31T11:30:00Z
description: "Understanding the dimensions of interpretability in ML, from global models to counterfactual reasoning."
tags: [machine-learning, interpretability, explainability, SHAP, LIME, model-explanation]
-----------------------------------------------------------------------------------------------

In machine learning, **interpretability** is crucial for trust, debugging, fairness, accountability, and real-world deployment. 

## 🔍 What is Interpretability?

Interpretability refers to the **degree to which a human can understand the cause of a decision** or prediction made by a model. It helps us answer:

* "**Why** did the model make this prediction?"
* "**Which features** mattered most?"
* "Can I **trust** this decision?"

## 📏 Global vs Local Interpretability

### Global Interpretability

Describes how the model behaves *overall*.

* **Simple models** (e.g., linear regression, decision trees) are naturally interpretable.
* Common tools: feature importance, model weights, decision paths.

### Local Interpretability

Focuses on **individual predictions**:

* **Why** did the model predict class A for instance X?
* Requires approximations or surrogate explanations.

#### 🔧 Common Local Methods

| Method                                                | Summary                                           |
| ----------------------------------------------------- | ------------------------------------------------- |
| **LIME** \[Ribeiro et al., 2016]                      | Fits interpretable model locally around instance. |
| **SHAP** \[Lundberg & Lee, 2017]                      | Uses Shapley values from game theory.             |
| **Integrated Gradients** \[Sundararajan et al., 2017] | Path-integrated gradient attribution.             |
| **Counterfactuals** \[Wachter et al., 2017]           | Asks what minimal change would flip the decision. |

## 🧱 Model Structural Transparency

Different ML models vary in inherent transparency:

| Model Type              | Transparency |
| ----------------------- | ------------ |
| Linear Regression       | High         |
| Decision Trees          | Medium–High  |
| Random Forest / XGBoost | Medium       |
| Deep Neural Networks    | Low          |

Structural transparency affects both global and local interpretability.

## 📊 Feature-level Interpretability

Focuses on feature *importance* and *influence* over outcomes:

* **Permutation Importance**: Shuffles values and observes change.
* **Partial Dependence Plots (PDP)** \[Friedman, 2001]: Visualizes marginal effect.
* **Accumulated Local Effects (ALE)**: Like PDP but avoids extrapolation issues.
* **SHAP Summary Plots**: Compact, intuitive, and powerful.

## 🧠 Semantic Concept-based Interpretability

Goes beyond raw features to understand **human-level concepts** within models, especially in deep learning:

* **TCAV** \[Kim et al., 2018]: Measures sensitivity to concepts like "stripes" or "happiness".
* **Neuron activation clustering**: Finds interpretable units in CNNs or transformers.
* **Feature visualization** \[Olah et al.]: Inverts learned filters to view what the model sees.

## 👁️ Visualization Techniques

* **Saliency maps**: Gradient-based heatmaps over input features.
* **Attention maps** (NLP, Vision): Visualize what the model "attends to".
* **UMAP/t-SNE**: Project embedding space into 2D for visual exploration.
* **Decision boundaries**: Helpful in low-dimensional problems.

## 🧪 Causal and Counterfactual Interpretability

Goes beyond correlation to model **what-if** scenarios:

* **Causal graphs**: Use known or learned structures to simulate interventions.
* **Counterfactual examples**: Identify the smallest change needed to alter a prediction.

Useful in fairness auditing, safety-critical systems, and recourse design.

## 🧭 Summary Table

| Aspect            | Description                                | Key Tools                             |
| ----------------- | ------------------------------------------ | ------------------------------------- |
| **Global**        | Entire model behavior                      | Coefficients, Tree Paths, SHAP Global |
| **Local**         | Individual prediction explanation          | LIME, SHAP, IG, Counterfactuals       |
| **Structural**    | Transparency of model type                 | Choice of model (linear vs NN)        |
| **Feature-level** | What features matter                       | SHAP, PDP, ALE, Permutation           |
| **Concept-based** | Relating model internals to human concepts | TCAV, Neuron Clustering               |
| **Visualization** | Graphical explanation of internal process  | Saliency, Attention, UMAP             |
| **Causal**        | Interventions and what-if reasoning        | Causal Graphs, Counterfactuals        |

## 📚 References

* Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD.*
* Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS.*
* Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML.*
* Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box. *Nature Machine Intelligence.*
* Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics.*
* Kim, B., Wattenberg, M., Gilmer, J., et al. (2018). Interpretability beyond feature attribution: Quantitative Testing with Concept Activation Vectors (TCAV). *ICML.*
* Olah, C., Satyanarayan, A., Johnson, I., et al. (2017). The building blocks of interpretability. *Distill.*

