# Hierarchical Token Merging

Hierarchical Token Merging (HTM) lies at the intersection of
- Representation Learning
- Hierarchical Compression & Generation
- Structured Latent Variable Models

It provides a general framework for learning hierarchical, bidirectional representations of arbitrary data modalities (e.g. images, speech, sequences, or sets) by iteratively merging and unmerging tokens.

Formally, for an input token set

$$\mathcal{T}_0=\{t_1,t_2,\dots,t_n\}, \quad t_i\in \mathbb{R}^d$$

the model constructs a binary merge tree reducing it to a single root token $t_{root}$.

