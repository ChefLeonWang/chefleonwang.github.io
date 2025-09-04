---

title: "Embedding, Distribution, and Space in Machine Learning"
description: "embedding, probability distribution and vector space: data enters, gets mapped into a space, forms a distribution, and becomes the target of optimization."
pubDatetime: 2025-09-01T11:30:00Z
tags: [machine-learning, embedding, probability distribution, vector space, supervised-learning, unsupervised-learning, reinforcement-learning]
-----------------------------------------------------------------------------------------------------------------------


---

## Vector Space: The Stage for All Representations

> A structured set that supports distances, similarities, and other geometric operations.

**Examples**:

* $\mathbb{R}^d$: standard Euclidean space
* Unit hypersphere: L2-normalized space for cosine similarity
* Manifold: smooth, lower-dimensional surface embedded in high-dim space

**Role**:
Spaces define the geometry in which learning, embedding, and generation take place.

All embeddings live in some kind of **vector space**:

* The simplest case: \$\mathbb{R}^d\$, where \$d\$ is the embedding dimension.
* This space allows us to measure distances (e.g., L2 norm), angles (cosine similarity), and projections.
* Embedding spaces can also be non-Euclidean (e.g., hyperbolic geometry for tree-like structures).

> **Key Insight**: A **space** is just a **container**. It's the backdrop against which all learning occurs. It has no semantics by itself, but gives structure to learned semantics.

---

## Embedding: Learnable Coordinates in Space

An **embedding** is a vector that represents an input entity:

* Text: words, sentences (e.g., Word2Vec, BERT)
* Images: pixel arrays mapped to feature vectors (e.g., ResNet, ViT)
* Graph nodes, audio chunks, protein sequences...

The embedding function is often a deep neural network:

$$
\phi(x) = z \in \mathbb{R}^d
$$

These learned vectors are **dense**, **continuous**, and **semantic**: closer vectors often imply semantic similarity (though not always).

> **Example**: In contrastive learning, we train embeddings so that similar inputs have high cosine similarity.

---

## Distribution: Structure over the Space
> A function that assigns probabilities to values in a space, capturing frequency or uncertainty.

**Formal**:
$P: \mathcal{Z} \to [0,1], \quad \text{with} \int P(z) dz = 1$

**Types**:

* Marginal: $P(x), P(z)$
* Conditional: $P(y|x), P(z|x), P(x|z)$
* Learned model distributions: $Q_\theta \approx P$

The **distribution** describes the likelihood of encountering different embeddings in the space:

* \$P(x)\$: Marginal distribution over inputs
* \$P(z)\$: Distribution of learned representations
* \$P(y|x)\$ or \$P(y|z)\$: Conditional distribution over labels given input/embedding

Learning often means fitting \$Q\_\theta\$ to approximate \$P\$.

> **Example**: In classification, \$Q\_\theta(y|x)\$ is modeled by a softmax over a final linear layer.

---

## KL Divergence: Measuring Discrepancy

To measure how far our model's distribution \$Q\_\theta\$ is from the true distribution \$P\$, we often use **Kullback-Leibler (KL) divergence**:

$$
D_{KL}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

It has deep ties to **information theory**:

* \$H(P)\$: Entropy of true distribution (minimal bits needed to code P)
* \$H(P, Q)\$ or \$H(P, \hat{P})\$: Cross-entropy (bits needed if we use Q to code P)
* Thus: \$D\_{KL}(P || Q) = H(P, Q) - H(P)\$

---

## Learning as Compression
I love this part!!!
A surprising insight from information theory is:

> **Learning = Compression.**

We want to compress input data while preserving as much task-relevant information as possible.

* Cross-entropy loss is interpreted as **expected encoding length**
* Lower loss means our model represents the data more compactly
* Embedding layers = learned compression functions

---

## MSE as a Special Case of KL Divergence

When doing **regression**, we often use **mean squared error (MSE)**. But it can be derived from KL divergence:

* Assume the true target \$y\$ is sampled from a Gaussian with mean \$\mu = y^\*\$ and variance \$\sigma^2\$
* Then:

$$
D_{KL}(\mathcal{N}(y^*; \sigma^2) || \mathcal{N}(\hat{y}; \sigma^2)) \propto (y^* - \hat{y})^2
$$

Thus, MSE is a special case of KL divergence between two Gaussians with fixed variance.

---

## The Geometry of Representation Learning

These three concepts—**embedding**, **distribution**, and **space**—form a triangle:

* **Space** provides structure/container
* **Embedding** gives concrete positions in space
* **Distributions** shape how these points are organized

All learning problems can be seen as trying to **RESHAPE** distributions over representations in space.

---

## Extensions and Related Concepts

* **Latent space**: The space of compressed internal variables (e.g., in VAEs)
* **Manifold**: Embeddings often lie on a lower-dimensional manifold in high-dim space
* **Contrastive loss**: Aligns positive pairs in space, repels negatives
* **Normalization**: Keeps representations well-behaved (e.g., unit L2 norm for cosine similarity)
* **Projection heads**: Often applied to embeddings to manipulate how distributions look in space

---

##  Structures Formed by Embeddings: Trajectories, Clusters, and Manifolds

Although each embedding is a single point in space, groups of embeddings often form higher-order structures with meaningful interpretations:

### • Trajectories: Time-Varying or Sequential Embeddings

* In applications like natural language processing, reinforcement learning, or video analysis, inputs come with a time axis.
* Embeddings of these inputs form sequences (trajectories) in space.
* These trajectories can encode dynamics: e.g., in RNNs or Transformers, we can trace how the state evolves over time.
* They can also be used to model temporal coherence, attention flow, or decision processes.

### • Clusters: Semantic Categories

* Embeddings from the same semantic class tend to cluster together in space.
* Examples:

  * Word embeddings for colors: {"red", "blue", "green"} form a color cluster.
  * Sentence embeddings for questions vs. answers.
* These clusters often reflect the geometry of a **semantic manifold**, and are useful for classification, clustering, and retrieval tasks.

### • Manifolds: Nonlinear Subspaces

* Real-world data doesn't fill all of \$\mathbb{R}^d\$; instead, it lies on a much lower-dimensional, nonlinear surface called a **manifold**.
* Examples:

  * Faces under varying lighting form a low-dimensional face manifold
  * Shapes of handwritten digits span a digit manifold
* Many generative models (e.g. GANs, VAEs) aim to learn this manifold and sample from it.

> **Bottom line**: Embeddings are not just points—they are elements of larger semantic and geometric structures. Understanding these structures allows us to build better models, losses, and data priors.


---

## References

1. Bengio et al. "Representation Learning: A Review and New Perspectives." IEEE PAMI, 2013.
2. Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck Principle," 2015.
3. Cover & Thomas, "Elements of Information Theory."
4. Radford et al. "Learning Transferable Visual Models from Natural Language Supervision (CLIP)." 2021.
5. Oord et al. "Representation Learning with Contrastive Predictive Coding." 2018.
6. Goodfellow et al. "Deep Learning" textbook (MIT Press).
