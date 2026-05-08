---
title: "Normalizing Flows: An Introduction and Review of Current Methods"
source: "https://arxiv.org/abs/1908.09257"
author: "Ivan Kobyzev et al."
published: "2019"
created: "2026-05-07"
description: null
tags: ["clippings"]
arxiv: "1908.09257"
url: "https://arxiv.org/abs/1908.09257"
---

# Normalizing Flows: An Introduction and Review of Current Methods

Ivan Kobyzev, Simon J.D. Prince, and Marcus A. Brubaker. Member, IEEE

Abstract—Normalizing Flows are generative models which produce tractable distributions where both sampling and density evaluation can be efficient and exact. The goal of this survey article is to give a coherent and comprehensive review of the literature around the construction and use of Normalizing Flows for distribution learning. We aim to provide context and explanation of the models, review current state-of-the-art literature, and identify open questions and promising future directions.

Index Terms—Generative models, Normalizing flows, Density estimation, Variational inference, Invertible neural networks.

# 1 INTRODUCTION

A MAJOR goal of statistics and machine learning hasbeen to model a probability distribution given samples MAJOR goal of statistics and machine learning has drawn from that distribution. This is an example of unsupervised learning and is sometimes called generative modelling. Its importance derives from the relative abundance of unlabelled data compared to labelled data. Applications include density estimation, outlier detection, prior construction, and dataset summarization.

Many methods for generative modeling have been proposed. Direct analytic approaches approximate observed data with a fixed family of distributions. Variational approaches and expectation maximization introduce latent variables to explain the observed data. They provide additional flexibility but can increase the complexity of learning and inference. Graphical models [Koller and Friedman, 2009] explicitly model the conditional dependence between random variables. Recently, generative neural approaches have been proposed including generative adversarial networks (GANs) [Goodfellow et al., 2014] and variational auto-encoders (VAEs) [Kingma and Welling, 2014].

GANs and VAEs have demonstrated impressive performance results on challenging tasks such as learning distributions of natural images. However, several issues limit their application in practice. Neither allows for exact evaluation of the probability density of new points. Furthermore, training can be challenging due to a variety of phenomena including mode collapse, posterior collapse, vanishing gradients and training instability [Bowman et al., 2015; Salimans et al., 2016].

Normalizing Flows (NF) are a family of generative models with tractable distributions where both sampling and density evaluation can be efficient and exact. Applications include image generation [Ho et al., 2019; Kingma and Dhariwal, 2018], noise modelling [Abdelhamed et al., 2019], video generation [Kumar et al., 2019], audio generation [Esling et al., 2019; Kim et al., 2018; Prenger et al., 2019], graph generation [Madhawa et al., 2019], reinforcement learning

[Mazoure et al., 2019; Nadeem Ward et al., 2019; Touati et al., 2019], computer graphics [Muller et al., 2018], and physics ¨ [Kanwar et al., 2020; Kohler et al., 2019; No ¨ e et al., 2019; ´ Wirnsberger et al., 2020; Wong et al., 2020].

There are several survey papers for VAEs [Kingma and Welling, 2019] and GANs [Creswell et al., 2018; Wang et al., 2017]. This article aims to provide a comprehensive review of the literature around Normalizing Flows for distribution learning. Our goals are to 1) provide context and explanation to enable a reader to become familiar with the basics, 2) review the current literature, and 3) identify open questions and promising future directions. Since this article was first made public, an excellent complementary treatment has been provided by Papamakarios et al. [2019]. Their article is more tutorial in nature and provides many details concerning implementation, whereas our treatment is more formal and focuses mainly on the families of flow models.

In Section 2, we introduce Normalizing Flows and describe how they are trained. In Section 3 we review constructions for Normalizing Flows. In Section 4 we describe datasets for testing Normalizing Flows and discuss the performance of different approaches. Finally, in Section 5 we discuss open problems and possible research directions.

# 2 BACKGROUND

Normalizing Flows were popularised by Rezende and Mohamed [2015] in the context of variational inference and by Dinh et al. [2015] for density estimation. However, the framework was previously defined in Tabak and Vanden-Eijnden [2010] and Tabak and Turner [2013], and explored for clustering and classification [Agnelli et al., 2010], and density estimation [Laurence et al., 2014; Rippel and Adams, 2013].

A Normalizing Flow is a transformation of a simple probability distribution (e.g., a standard normal) into a more complex distribution by a sequence of invertible and differentiable mappings. The density of a sample can be evaluated by transforming it back to the original simple distribution and then computing the product of i) the density of the inverse-transformed sample under this distribution and ii)

![[Normalizing Flows: An Introduction and Review of Current Methods/images/9905535bcb7b6ac1045d0acb51498a93654f9ae30b54bb8713f9c3d100bde855.jpg]]

![[Normalizing Flows: An Introduction and Review of Current Methods/images/55cebfa0892df398a58cb3522bc4ade0b1e993fbb5a7321b83b0bb9f9cf75210.jpg]]  
Fig. 1. Change of variables (Equation (1)). Top-left: the density of the source $p _ { \mathbf { Z } }$ . Top-right: the density function of the target distribution $p \mathbf { { v } } ( \mathbf { { y } } )$ . There exists a bijective function g, such that $p _ { \mathbf { Y } } = \mathbf { g } _ { \ast } p _ { \mathbf { Z } }$ , with inverse f . Bottom-left: the inverse function f . Bottom-right: the absolute Jacobian (derivative) of f .

the associated change in volume induced by the sequence of inverse transformations. The change in volume is the product of the absolute values of the determinants of the Jacobians for each transformation, as required by the change of variables formula.

The result of this approach is a mechanism to construct new families of distributions by choosing an initial density and then chaining together some number of parameterized, invertible and differentiable transformations. The new density can be sampled from (by sampling from the initial density and applying the transformations) and the density at a sample (i.e., the likelihood) can be computed as above.

# 2.1 Basics

Let $\textbf { Z } \in \ \mathbb { R } ^ { D }$ be a random variable with a known and tractable probability density function $p \mathbf { z } : \mathbb { R } ^ { D }  \mathbb { R }$ . Let g be an invertible function and $\mathbf { Y } = \mathbf { g } ( \mathbf { Z } )$ . Then using the change of variables formula, one can compute the probability density function of the random variable Y:

$$
\begin{array}{l} p _ {\mathbf {Y}} (\mathbf {y}) = p _ {\mathbf {Z}} (\mathbf {f} (\mathbf {y})) | \det  \operatorname {D f} (\mathbf {y}) | \\ = p _ {\mathbf {Z}} (\mathbf {f} (\mathbf {y})) \left| \det  \mathrm {D g} (\mathbf {f} (\mathbf {y})) \right| ^ {- 1}, \tag {1} \\ \end{array}
$$

where f is the inverse of $\begin{array} { r } { \mathbf { g } , \mathbf { D f } ( \mathbf { y } ) = { \frac { \partial \mathbf { f } } { \partial \mathbf { y } } } } \end{array}$ is the Jacobian of f and $\begin{array} { r } { \mathrm { D } \mathbf { g } ( \mathbf { z } ) = \frac { \partial \mathbf { g } } { \partial \mathbf { z } } } \end{array}$ ∂g∂z is the Jacobian of g. This new density function $p \mathbf { { v } } ( \mathbf { { y } } )$ is called a pushforward of the density $p \mathbf { z }$ by the function g and denoted by $\mathbf { g } _ { \ast p \mathbf { z } }$ (Figure 1).

In the context of generative models, the above function g (a generator) “pushes forward” the base density $p \mathbf { z }$ (sometimes referred to as the “noise”) to a more complex density. This movement from base density to final complicated density is the generative direction. Note that to generate a data point y, one can sample z from the base distribution, and then apply the generator: $\mathbf { y } = \mathbf { g } ( \mathbf { z } )$ .

The inverse function f moves (or “flows”) in the opposite, normalizing direction: from a complicated and irregular data distribution towards the simpler, more regular or “normal” form, of the base measure $p \mathbf { z }$ . This view is what gives rise to the name “normalizing flows” as f is “normalizing”

the data distribution. This term is doubly accurate if the base measure $p \mathbf { z }$ is chosen as a Normal distribution as it often is in practice.

Intuitively, if the transformation g can be arbitrarily complex, one can generate any distribution $p \mathbf { v }$ from any base distribution $p \mathbf { z }$ under reasonable assumptions on the two distributions. This has been proven formally [Bogachev et al., 2005; Medvedev, 2008; Villani, 2003]. See Section 3.4.3.

Constructing arbitrarily complicated non-linear invertible functions (bijections) can be difficult. By the term Normalizing Flows people mean bijections which are convenient to compute, invert, and calculate the determinant of their Jacobian. One approach to this is to note that the composition of invertible functions is itself invertible and the determinant of its Jacobian has a specific form. In particular, let $\mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { N }$ be a set of $N$ bijective functions and define $\textbf { g } = \textbf { g } _ { N } \circ \textbf { g } _ { N - 1 } \circ \cdot \cdot \cdot \circ \textbf { g } _ { 1 }$ to be the composition of the functions. Then it can be shown that g is also bijective, with inverse:

$$
\mathbf {f} = \mathbf {f} _ {1} \circ \dots \circ \mathbf {f} _ {N - 1} \circ \mathbf {f} _ {N}, \tag {2}
$$

and the determinant of the Jacobian is

$$
\det  \mathrm {D} \mathbf {f} (\mathbf {y}) = \prod_ {i = 1} ^ {N} \det  \mathrm {D} \mathbf {f} _ {i} (\mathbf {x} _ {i}), \tag {3}
$$

where $\begin{array} { r } { \mathbf { D } \mathbf { f } _ { i } ( \mathbf { y } ) = \frac { \partial \mathbf { f } _ { i } } { \partial \mathbf { x } } } \end{array}$ is the Jacobian of $\mathbf { f } _ { i }$ . We denote the value of the $i$ -th intermediate flow as $\mathbf { x } _ { i } = \mathbf { g } _ { i } \circ \cdot \cdot \cdot \circ \mathbf { g } _ { 1 } ( \mathbf { z } ) =$ $\mathbf { f } _ { i + 1 } \circ \cdots \circ \mathbf { f } _ { N } ( \mathbf { y } )$ and so $\mathbf { x } _ { N } = \mathbf { y }$ . Thus, a set of nonlinear bijective functions can be composed to construct successively more complicated functions.

# 2.1.1 More formal construction

In this section we explain normalizing flows from more formal perspective. Readers unfamiliar with measure theory can safely skip to Section 2.2. First, let us recall the general definition of a pushforward.

Definition 1. If $( \mathcal { Z } , \Sigma _ { \mathcal { Z } } ) .$ , $( \boldsymbol { \mathcal { V } } , \Sigma _ { \mathcal { V } } )$ are measurable spaces, g is a measurable mapping between them, and $\mu$ is a measure on $\mathcal { Z }$ , then one can define a measure on $\mathcal { V }$ (called the pushforward measure and denoted by $\mathbf { g } _ { \ast } \boldsymbol { \mu } _ { \cdot }$ ) by the formula

$$
\mathbf {g} _ {*} \mu (U) = \mu (\mathbf {g} ^ {- 1} (U)), \quad \text {f o r a l l} U \in \Sigma_ {\mathcal {Y}}. \tag {4}
$$

This notion gives a general formulation of a generative model. Data can be understood as a sample from a measured “data” space $( \boldsymbol { \mathcal { V } } , \Sigma _ { \mathcal { V } } , \nu )$ , which we want to learn. To do that one can introduce a simpler measured space $( \mathcal { Z } , \Sigma _ { \mathcal { Z } } , \mu )$ and find a function $\mathbf { g } : { \mathcal { Z } }  { \mathcal { Y } } ,$ such that $\boldsymbol { \nu } = \mathbf { g } _ { * } \mu$ . This function g can be interpreted as a “generator”, and $\mathcal { Z }$ as a latent space. This view puts generative models in the context of transportation theory [Villani, 2003].

In this survey we will assume that $\mathcal { Z } = \mathbb { R } ^ { D }$ , all sigmaalgebras are Borel, and all measures are absolutely continuous with respect to Lebesgue measure (i.e., $\begin{array} { r } { \mu = p _ { \mathbf { Z } } \mathbf { d } \mathbf { z } } \end{array}$ ).

Definition 2. A function $\mathbf { g } : \mathbb { R } ^ { D }  \mathbb { R } ^ { D }$ is called a diffeomorphism, if it is bijective, differentiable, and its inverse is differentiable as well.

The pushforward of an absolutely continuous measure $p \mathbf { z } \mathbf { d } \mathbf { z }$ by a diffeomorphism g is also absolutely continuous

with a density function given by Equation (1). Note that this more general approach is important for studying generative models on non-Euclidean spaces (see Section 5.2).

Remark 3. It is common in the normalizing flows literature to simply refer to diffeomorphisms as “bijections” even though this is formally incorrect. In general, it is not necessary that g is everywhere differentiable; rather it is sufficient that it is differentiable only almost everywhere with respect to the Lebesgue measure on $\mathbb { R } ^ { D }$ . This allows, for instance, piecewise differentiable functions to be used in the construction of g.

# 2.2 Applications

# 2.2.1 Density estimation and sampling

The natural and most obvious use of normalizing flows is to perform density estimation. For simplicity assume that only a single flow, g, is used and it is parameterized by the vector $\theta$ . Further, assume that the base measure, $p \mathbf { z }$ is given and is from some complicated distribution, parameterized by the vector $\phi$ . Given a set of data observed $\mathcal { D } = \{ \mathbf { y } ^ { ( i ) } \} _ { i = 1 } ^ { M } ,$ , we can then perform likelihood-based estimation of the parameters $\Theta = \bar { ( \theta , \phi ) }$ . The data likelihood in this case simply becomes

$$
\begin{array}{l} \log p (\mathcal {D} | \Theta) = \sum_ {i = 1} ^ {M} \log p _ {\mathbf {Y}} \left(\mathbf {y} ^ {(i)} | \Theta\right) \tag {5} \\ = \sum_ {i = 1} ^ {M} \log p _ {\mathbf {Z}} \left(\mathbf {f} \left(\mathbf {y} ^ {(i)} | \theta\right) | \phi\right) + \log \left| \det  D \mathbf {f} \left(\mathbf {y} ^ {(i)} | \theta\right) \right| \\ \end{array}
$$

where the first term is the log likelihood of the sample under the base measure and the second term, sometimes called the log-determinant or volume correction, accounts for the change of volume induced by the transformation of the normalizing flows (see Equation (1)). During training, the parameters of the flow (θ) and of the base distribution $( \phi )$ are adjusted to maximize the log-likelihood.

Note that evaluating the likelihood of a distribution modelled by a normalizing flow requires computing f (i.e., the normalizing direction), as well as its log determinant. The efficiency of these operations is particularly important during training where the likelihood is repeatedly computed. However, sampling from the distribution defined by the normalizing flow requires evaluating the inverse g (i.e., the generative direction). Thus sampling performance is determined by the cost of the generative direction. Even though a flow must be theoretically invertible, computation of the inverse may be difficult in practice; hence, for density estimation it is common to model a flow in the normalizing direction (i.e., f ). 1

Finally, while maximum likelihood estimation is often effective (and statistically efficient under certain conditions) other forms of estimation can and have been used with normalizing flows. In particular, adversarial losses can be used with normalizing flow models (e.g., in Flow-GAN [Grover et al., 2018]).

1. To ensure both efficient density estimation and sampling, van den Oord et al. [2017] proposed an approach called Probability Density Distillation which trains the flow f as normal and then uses this as a teacher network to train a tractable student network g.

# 2.2.2 Variational Inference

Consider a latent variable model $\begin{array} { r } { p ( \mathbf { x } ) = \int p ( \mathbf { x } , \mathbf { y } ) d \mathbf { y } } \end{array}$ where x is an observed variable and y the latent variable. The posterior distribution $p ( \mathbf { y } \vert \mathbf { x } )$ is used when estimating the parameters of the model, but its computation is usually intractable in practice. One approach is to use variational inference and introduce the approximate posterior $q ( \mathbf { y } | \mathbf { x } , \theta )$ where $\theta$ are parameters of the variational distribution. Ideally this distribution should be as close to the real posterior as possible. This is done by minimizing the KL divergence $\tilde { D _ { K L } } \big ( q ( \mathbf { y } | \mathbf { x } , \theta ) | | p ( \mathbf { y } | \mathbf { x } ) \big ) .$ , which is equivalent to maximizing the evidence lower bound $\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { \bar { E } } _ { q ( \mathbf { y } \mid \mathbf { x } , \boldsymbol { \theta } ) } [ \log ( p ( \mathbf { y } , \mathbf { x } ) ) -$ $\log ( q ( \mathbf { y } | \mathbf { x } , \theta ) ) ]$ . The latter optimization can be done with gradient descent; however for that one needs to compute gradients of the form $\nabla _ { \boldsymbol { \theta } } \mathbb { E } _ { \boldsymbol { q } ( \mathbf { y } | \mathbf { x } , \boldsymbol { \theta } ) } [ h ( \mathbf { y } ) ] ,$ , which is not straightforward.

As was observed by Rezende and Mohamed [2015], one can reparametrize $q ( \mathbf { y } | \mathbf { x } , \theta ) \ = \ p \mathbf { y } ( \mathbf { y } | \theta )$ with normalizing flows. Assume for simplicity, that only a single flow g with parameters $\theta$ is used, $\bar { \mathbf { y } } = \mathbf { g } ( \mathbf { z } | \boldsymbol { \theta } )$ and the base distribution $\bar { p } \mathbf { z } ( \mathbf { z } )$ does not depend on $\theta$ . Then

$$
\mathbb {E} _ {p _ {\mathbf {Y}} (\mathbf {y} | \theta)} [ h (\mathbf {y}) ] = \mathbb {E} _ {p _ {\mathbf {Z}} (\mathbf {z})} [ h (\mathbf {g} (\mathbf {z} | \theta)) ], \tag {6}
$$

and the gradient of the right hand side with respect to $\theta$ can be computed. This approach generally to computing gradients of an expectation is often called the “reparameterization trick”.

In this scenario evaluating the likelihood is only required at points which have been sampled. Here the sampling performance and evaluation of the log determinant are the only relevant metrics and computing the inverse of the mapping may not be necessary. Indeed, the planar and radial flows introduced in Rezende and Mohamed [2015] are not easily invertible (see Section 3.3).

# 3 METHODS

Normalizing Flows should satisfy several conditions in order to be practical. They should:

be invertible; for sampling we need g while for computing likelihood we need f,   
be sufficiently expressive to model the distribution of interest,   
be computationally efficient, both in terms of computing f and g (depending on the application) but also in terms of the calculation of the determinant of the Jacobian.

In the following section, we describe different types of flows and comment on the above properties. An overview of the methods discussed can be seen in figure 2.

# 3.1 Elementwise Flows

A basic form of bijective non-linearity can be constructed given any bijective scalar function. That is, let $h : \mathbb { R }  \mathbb { R }$ be a scalar valued bijection. Then, if $\mathbf { x } = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { D } ) ^ { T } .$ ,

$$
\mathbf {g} (\mathbf {x}) = \left(h \left(x _ {1}\right), h \left(x _ {2}\right), \dots , h \left(x _ {D}\right)\right) ^ {T} \tag {7}
$$

is also a bijection whose inverse simply requires computing $h ^ { - 1 }$ and whose Jacobian is the product of the absolute

![[Normalizing Flows: An Introduction and Review of Current Methods/images/9030525b7525e173fb68a578900b657a81dce970a504ba96300b3348b05afa34.jpg]]  
Fig. 2. Overview of flows discussed in this review. We start with elementwise bijections, linear flows, and planar and radial flows. All of these have drawbacks and are limited in utility. We then discuss two architectures (coupling flows and autoregressive flows) which support invertible non-linear transformations. These both use a coupling function, and we summarize the different coupling functions available. Finally, we discuss residual flows and their continuous extension infinitesimal flows.

values of the derivatives of $h$ . This can be generalized by allowing each element to have its own distinct bijective function which might be useful if we wish to only modify portions of our parameter vector. In deep learning terminology, $h ,$ could be viewed as an “activation function”. Note that the most commonly used activation function ReLU is not bijective and can not be directly applicable, however, the (Parametric) Leaky ReLU [He et al., 2015; Maas et al., 2013] can be used instead among others. Note that recently spline-based activation functions have also been considered [Durkan et al., 2019a,b] and will be discussed in Section 3.4.4.4.

# 3.2 Linear Flows

Elementwise operations alone are insufficient as they cannot express any form of correlation between dimensions. Linear mappings can express correlation between dimensions:

$$
\mathbf {g} (\mathbf {x}) = \mathbf {A x} + \mathbf {b} \tag {8}
$$

where $\mathbf { A } \in \mathbb { R } ^ { D \times D }$ and $\mathbf { b } \in \mathbb { R } ^ { D }$ are parameters. If A is an invertible matrix, the function is invertible.

Linear flows are limited in their expressiveness. Consider a Gaussian base distribution: $p _ { \mathbf { Z } } ( \mathbf { z } ) \bar { \mathbf { \Psi } } = \mathcal { N } ( \mathbf { z } , \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ . After transformation by a linear flow, the distribution remains Gaussian with distribution $p _ { \mathbf { Y } } = \mathcal { N } ( \mathbf { y } , \mathbf { A } \mu + \mathbf { b } , \mathbf { A } ^ { T } \Sigma \mathbf { A } )$ More generally, a linear flow of a distribution from the exponential family remains in the exponential family. However, linear flows are an important building block as they form the basis of affine coupling flows (Section 3.4.4.1).

Note that the determinant of the Jacobian is simply $\operatorname* { d e t } ( \mathbf { A } )$ , which can be computed in $\mathcal { O } ( D ^ { 3 } )$ , as can the inverse. Hence, using linear flows can become expensive for large $D$ . By restricting the form of A we can avoid these practical problems at the expense of expressive power. In the following sections we discuss different ways of limiting the form of linear transforms to make them more practical.

# 3.2.1 Diagonal

If A is diagonal with nonzero diagonal entries, then its inverse can be computed in linear time and its determinant

is the product of the diagonal entries. However, the result is an elementwise transformation and hence cannot express correlation between dimensions. Nonetheless, a diagonal linear flow can still be useful for representing normalization transformations [Dinh et al., 2017] which have become a ubiquitous part of modern neural networks [Ioffe and Szegedy, 2015].

# 3.2.2 Triangular

The triangular matrix is a more expressive form of linear transformation whose determinant is the product of its diagonal. It is non-singular so long as its diagonal entries are non-zero. Inversion is relatively inexpensive requiring a single pass of back-substitution costing $\mathcal { \hat { O } } ( D ^ { 2 } )$ operations.

Tomczak and Welling [2017] combined $K$ triangular matrices $\mathbf { T } _ { i } ,$ each with ones on the diagonal, and a $K \cdot$ - dimensional probability vector $\omega$ to define a more general linear flow $\begin{array} { r } { \dot { \textbf { y } } = ~ ( \sum _ { i = 1 } ^ { \bar { K } } \omega _ { i } \mathbf { T } _ { i } ) \mathbf { z } } \end{array}$ . The determinant of this bijection is one. However finding the inverse has $\mathcal { O } ( D ^ { 3 } )$ complexity, if some of the matrices are upper- and some are lower-triangular.

# 3.2.3 Permutation and Orthogonal

The expressiveness of triangular transformations is sensitive to the ordering of dimensions. Reordering the dimensions can be done easily using a permutation matrix which has an absolute determinant of 1. Different strategies have been tried, including reversing and a fixed random permutation [Dinh et al., 2017; Kingma and Dhariwal, 2018]. However, the permutations cannot be directly optimized and so remain fixed after initialization which may not be optimal.

A more general alternative is the use of orthogonal transformations. The inverse and absolute determinant of an orthogonal matrix are both trivial to compute which make them efficient. Tomczak and Welling [2016] used orthogonal matrices parameterized by the Householder transform. The idea is based on the fact from linear algebra that any orthogonal matrix can be written as a product of reflections. To parameterize a reflection matrix $H$ in $\mathbb { R } ^ { D }$ one fixes a nonzero vector $\mathbf { v } \in \mathbb { R } ^ { D }$ , and then defines $\begin{array} { r } { H = \mathbb { 1 } - \frac { 2 } { | | \mathbf { v } | | ^ { 2 } } \mathbf { v } \mathbf { v } ^ { T } } \end{array}$

# 3.2.4 Factorizations

Instead of limiting the form of A, Kingma and Dhariwal [2018] proposed using the $L U$ factorization:

$$
\mathbf {g} (\mathbf {x}) = \mathbf {P L U x} + \mathbf {b} \tag {9}
$$

where L is lower triangular with ones on the diagonal, U is upper triangular with non-zero diagonal entries, and $\mathbf { P }$ is a permutation matrix. The determinant is the product of the diagonal entries of U which can be computed in $\mathcal { O } ( D )$ . The inverse of the function g can be computed using two passes of backward substitution in $\mathcal { O } ( D ^ { 2 } )$ . However, the discrete permutation P cannot be easily optimized. To avoid this, P is randomly generated initially and then fixed. Hoogeboom et al. [2019a] noted that fixing the permutation matrix limits the flexibility of the transformation, and proposed using the $Q R$ decomposition instead where the orthogonal matrix $Q$ is described with Householder transforms.

# 3.2.5 Convolution

Another form of linear transformation is a convolution which has been a core component of modern deep learning architectures. While convolutions are easy to compute their inverse and determinant are non-obvious. Several approaches have been considered. Kingma and Dhariwal [2018] restricted themselves to $" 1 \ \times \ 1 "$ convolutions for flows which are simply a full linear transformation but applied only across channels. Zheng et al. [2018] used 1D convolutions (ConvFlow) and exploited the triangular structure of the resulting transform to efficiently compute the determinant. However Hoogeboom et al. [2019a] have provided a more general solution for modelling $d { \times } d$ convolutions, either by stacking together masked autoregressive convolutions (referred to as Emerging Convolutions) or by exploiting the Fourier domain representation of convolution to efficiently compute inverses and determinants (referred to as Periodic Convolutions).

# 3.3 Planar and Radial Flows

Rezende and Mohamed [2015] introduced planar and radial flows. They are relatively simple, but their inverses aren’t easily computed. These flows are not widely used in practice, yet they are reviewed here for completeness.

# 3.3.1 Planar Flows

Planar flows expand and contract the distribution along certain specific directions and take the form

$$
\mathbf {g} (\mathbf {x}) = \mathbf {x} + \mathbf {u} h \left(\mathbf {w} ^ {T} \mathbf {x} + b\right), \tag {10}
$$

where ${ \bf u } , { \bf w } \in \mathbb { R } ^ { D }$ and $b \in \mathbb { R }$ are parameters and $h : \mathbb { R }  \mathbb { R }$ is a smooth non-linearity. The Jacobian determinant for this transformation is

$$
\begin{array}{l} \det  \left(\frac {\partial \mathbf {g}}{\partial \mathbf {x}}\right) = \det  \left(\mathbb {1} _ {D} + \mathbf {u} h ^ {\prime} \left(\mathbf {w} ^ {T} \mathbf {x} + b\right) \mathbf {w} ^ {T}\right) \\ = 1 + h ^ {\prime} \left(\mathbf {w} ^ {T} \mathbf {x} + b\right) \mathbf {u} ^ {T} \mathbf {w}, \tag {11} \\ \end{array}
$$

where the last equality comes from the application of the matrix determinant lemma. This can be computed in $\mathcal { O } ( D )$ time. The inversion of this flow isn’t possible in closed form and may not exist for certain choices of $h ( \cdot )$ and certain parameter settings [Rezende and Mohamed, 2015].

The term $\mathbf { u } h ( \mathbf { \check { w } } ^ { T } \mathbf { x } + b )$ can be interpreted as a multilayer perceptron with a bottleneck hidden layer with a single unit [Kingma et al., 2016]. This bottleneck means that one needs to stack many planar flows to get high expressivity. Hasenclever et al. [2017] and van den Berg et al. [2018] introduced Sylvester flows to resolve this problem:

$$
\mathbf {g} (\mathbf {x}) = \mathbf {x} + \mathbf {U h} \left(\mathbf {W} ^ {T} \mathbf {x} + \mathbf {b}\right), \tag {12}
$$

where U and W are $D \times M$ matrices, $\mathbf { b _ { \lambda } } \in \mathbb { R } ^ { M }$ and h : $\mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ is an elementwise smooth nonlinearity, where $M \leq D$ is a hyperparameter to choose and which can be interpreted as the dimension of a hidden layer. In this case the Jacobian determinant is:

$$
\begin{array}{l} \det  \left(\frac {\partial \mathbf {g}}{\partial \mathbf {x}}\right) = \det  \left(\mathbb {1} _ {D} + \mathbf {U} \operatorname {d i a g} \left(\mathbf {h} ^ {\prime} \left(\mathbf {W} ^ {T} \mathbf {x} + b\right)\right) \mathbf {W} ^ {T}\right) \\ = \det  \left(\mathbb {1} _ {M} + \operatorname {d i a g} \left(\mathbf {h} ^ {\prime} \left(\mathbf {W} ^ {T} \mathbf {x} + b\right)\right) \mathbf {W} \mathbf {U} ^ {T}\right), \tag {13} \\ \end{array}
$$

where the last equality is Sylvester’s determinant identity (which gives these flows their name). This can be computationally efficient if $M$ is small. Some sufficient conditions for the invertibility of Sylvester transformations are discussed in Hasenclever et al. [2017] and van den Berg et al. [2018].

# 3.3.2 Radial Flows

Radial flows instead modify the distribution around a specific point so that

$$
\mathbf {g} (\mathbf {x}) = \mathbf {x} + \frac {\beta}{\alpha + \| \mathbf {x} - \mathbf {x} _ {0} \|} \left(\mathbf {x} - \mathbf {x} _ {0}\right) \tag {14}
$$

where $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { D }$ is the point around which the distribution is distorted, and $\alpha , \beta \in \mathbb { R }$ are parameters, $\alpha > 0$ . As for planar flows, the Jacobian determinant can be computed relatively efficiently. The inverse of radial flows cannot be given in closed form but does exist under suitable constraints on the parameters [Rezende and Mohamed, 2015].

# 3.4 Coupling and Autoregressive Flows

In this section we describe coupling and auto-regressive flows which are the two most widely used flow architectures. We first present them in the general form, and then in Section 3.4.4 we give specific examples.

# 3.4.1 Coupling Flows

Dinh et al. [2015] introduced a coupling method to enable highly expressive transformations for flows (Figure 3a). Consider a disjoint partition of the input $\textbf { x } \in { \overset { \sim } { \mathbb { R } } } ^ { D }$ into two subspaces: $( \mathbf { x } ^ { A } , \mathbf { \hat { x } } ^ { B } ) ~ \in ~ \mathbb { R } ^ { d } \times \mathbb { R } ^ { D \hat { - } d }$ and a bijection $\mathbf { h } ( \cdot ; \boldsymbol { \theta } ) : \dot { \mathbb { R } ^ { d } }  \mathbb { R } ^ { d } .$ , parameterized by $\theta$ . Then one can define a function $\mathbf { g } : \mathbb { R } ^ { D } \overset { \cdot } {  } \mathbb { R } ^ { D }$ by the formula:

$$
\begin{array}{l} \mathbf {y} ^ {A} = \mathbf {h} (\mathbf {x} ^ {A}; \Theta (\mathbf {x} ^ {B})) \\ \mathbf {y} ^ {B} = \mathbf {x} ^ {B}, \tag {15} \\ \end{array}
$$

where the parameters $\theta$ are defined by any arbitrary function $\Theta ( \mathbf { x } ^ { B } )$ which only uses $\mathbf { x } ^ { B }$ as input. This function is called a conditioner. The bijection $\mathbf { h }$ is called a coupling function, and the resulting function g is called a coupling flow. A coupling flow is invertible if and only if h is invertible and has inverse:

$$
\begin{array}{l} \mathbf {x} ^ {A} = \mathbf {h} ^ {- 1} (\mathbf {y} ^ {A}; \Theta (\mathbf {x} ^ {B})) \\ \mathbf {x} ^ {B} = \mathbf {y} ^ {B}. \tag {16} \\ \end{array}
$$

The Jacobian of g is a block triangular matrix where the diagonal blocks are Dh and the identity matrix respectively. Hence the determinant of the Jacobian of the coupling flow is simply the determinant of Dh.

Most coupling functions are applied to $\mathbf { x } ^ { A }$ element-wise:

$$
\mathbf {h} \left(\mathbf {x} ^ {A}; \theta\right) = \left(h _ {1} \left(x _ {1} ^ {A}; \theta_ {1}\right), \dots , h _ {d} \left(x _ {d} ^ {A}; \theta_ {d}\right)\right), \tag {17}
$$

where each $h _ { i } ( \cdot ; \theta _ { i } ) : \mathbb { R } \to \mathbb { R }$ is a scalar bijection. In this case a coupling flow is a triangular transformation (i.e., has a triangular Jacobian matrix). See Section 3.4.4 for examples.

The power of a coupling flow resides in the ability of a conditioner $\Theta ( \mathbf { x } ^ { B } )$ to be arbitrarily complex. In practice it is usually modelled as a neural network. For example, Kingma and Dhariwal [2018] used a shallow ResNet architecture.

![[Normalizing Flows: An Introduction and Review of Current Methods/images/55e3419c8cc8ddd914123c263d3ab8abee8b389505ee4c9119f18606a898a2d1.jpg]]

![[Normalizing Flows: An Introduction and Review of Current Methods/images/ec3413e46b4459f2892284e286b8712347eab7ef82ffdac000212026f5292b37.jpg]]  
Fig. 3. Coupling architecture. a) A single coupling flow described in Equation (15). A coupling function h is applied to one part of the space, while its parameters depend on the other part. b) Two subsequent multiscale flows in the generative direction. A flow is applied to a relatively low dimensional vector $\mathbf { z }$ ; its parameters no longer depend on the rest part $\mathbf { z } _ { a u x }$ . Then new dimensions are gradually introduced to the distribution.

Sometimes, however, the conditioner can be constant (i.e., not depend on $\mathbf { x } ^ { B }$ at all). This allows for the construction of a “multi-scale flow” Dinh et al. [2017] which gradually introduces dimensions to the distribution in the generative direction (Figure 3b). In the normalizing direction, the dimension reduces by half after each iteration step, such that most of semantic information is retained. This reduces the computational costs of transforming high dimensional distributions and can capture the multi-scale structure inherent in certain kinds of data like natural images.

The question remains of how to partition x. This is often done by splitting the dimensions in half [Dinh et al., 2015], potentially after a random permutation. However, more structured partitioning has also been explored and is common practice, particularly when modelling images. For instance, Dinh et al. [2017] used “masked” flows that take alternating pixels or blocks of channels in the case of an image in non-volume preserving flows (RealNVP). In place of permutation Kingma and Dhariwal [2018] used $1 \times 1$ convolution (Glow). For the partition for the multiscale flow in the normalizing direction, Das et al. [2019] suggested selecting features at which the Jacobian of the flow has higher values for the propagated part.

# 3.4.2 Autoregressive Flows

Kingma et al. [2016] used autoregressive models as a form of normalizing flow. These are non-linear generalizations of multiplication by a triangular matrix (Section 3.2.2).

Let $h ( \cdot ; \theta ) : \mathbb { R } \to \mathbb { R }$ be a bijection parameterized by $\theta$ . Then an autoregressive model is a function $\mathbf { g } : \mathbb { R } ^ { D }  \mathbf { \bar { \mathbb { R } } } ^ { D } .$ , which outputs each entry of $\mathbf { y } = \mathbf { g } ( \mathbf { x } )$ conditioned on the previous entries of the input:

$$
y _ {t} = h \left(x _ {t}; \Theta_ {t} \left(\mathbf {x} _ {1: t - 1}\right)\right), \tag {18}
$$

where $\mathbf { x } _ { 1 : t } ~ = ~ ( x _ { 1 } , \ldots , x _ { t } )$ . For $t = 2 , \ldots , D$ we choose arbitrary functions $\Theta _ { t } ( \cdot )$ mapping $\mathbb { R } ^ { t - 1 }$ to the set of all parameters, and $\Theta _ { 1 }$ is a constant. The functions $\Theta _ { t } ( \cdot )$ are called conditioners.

The Jacobian matrix of the autoregressive transformation g is triangular. Each output $y _ { t }$ only depends on $\mathbf { x } _ { 1 : t } ,$ and so the determinant is just a product of its diagonal entries:

$$
\det  \left(\mathrm {D} \mathbf {g}\right) = \prod_ {t = 1} ^ {D} \frac {\partial y _ {t}}{\partial x _ {t}}. \tag {19}
$$

In practice, it’s possible to efficiently compute all the entries of the direct flow (Equation (18)) in one pass using a single network with appropriate masks [Germain et al., 2015]. This idea was used by Papamakarios et al. [2017] to create masked autoregressive flows (MAF).

However, the computation of the inverse is more challenging. Given the inverse of $h _ { \iota }$ the inverse of g can be found with recursion: we have $x _ { 1 } ~ = ~ h ^ { - 1 } ( y _ { 1 } ; \Theta _ { 1 } )$ and for any t = $: 2 , \ldots , D , x _ { t } = h ^ { - 1 } ( y _ { t } ; \Theta _ { t } ( { \bf x } _ { 1 : t - 1 } ) )$ . This computation is inherently sequential which makes it difficult to implement efficiently on modern hardware as it cannot be parallelized.

Note that the functional form for the autoregressive model is very similar to that for the coupling flow. In both cases a bijection $h$ is used, which has as an input one part of the space and which is parameterized conditioned on the other part. We call this bijection a coupling function in both cases. Note that Huang et al. [2018] used the name “transformer” (which has nothing to do with transformers in NLP).

Alternatively, Kingma et al. [2016] introduced the “inverse autoregressive flow” (IAF), which outputs each entry of y conditioned the previous entries of y (with respect to the fixed ordering). Formally,

$$
y _ {t} = h \left(x _ {t}; \theta_ {t} \left(\mathbf {y} _ {1: t - 1}\right)\right). \tag {20}
$$

One can see that the functional form of the inverse autoregressive flow is the same as the form of the inverse of the flow in Equation (18), hence the name. Computation of the IAF is sequential and expensive, but the inverse of IAF (which is a direct autoregressive flow) can be computed relatively efficiently (Figure 4).

![[Normalizing Flows: An Introduction and Review of Current Methods/images/8821e80d358cfbd3e15ea0530abb8b3bf89aacae559014fd9c42b3f8875517c0.jpg]]

![[Normalizing Flows: An Introduction and Review of Current Methods/images/a965f1bcea89ce368682a399426cd17e7779f570df2a9ba328b799723f487f01.jpg]]  
Fig. 4. Autoregressive flows. On the left, is the direct autoregressive flow given in Equation (18). Each output depends on the current and previous inputs and so this operation can be easily parallelized. On the right, is the inverse autoregressive flow from Equation (20). Each output depends on the current input and the previous outputs and so computation is inherently sequential and cannot be parallelized.

In Section 2.2.1 we noted that papers typically model flows in the “normalizing flow” direction (i.e., in terms of f from data to the base density) to enable efficient evaluation of the log-likelihood during training. In this context one can think of IAF as a flow in the generative direction: i.e.in terms of g from base density to data. Hence Papamakarios et al.

[2017] noted that one should use IAFs if fast sampling is needed (e.g., for stochastic variational inference), and MAFs if fast density estimation is desirable. The two methods are closely related and, under certain circumstances, are theoretically equivalent [Papamakarios et al., 2017].

# 3.4.3 Universality

For several autoregressive flows the universality property has been proven [Huang et al., 2018; Jaini et al., 2019a]. Informally, universality means that the flow can learn any target density to any required precision given sufficient capacity and data. We will provide a formal proof of the universality theorem following Jaini et al. [2019a]. This section requires some knowledge of measure theory and functional analysis and can be safely skipped.

First, recall that a mapping $T = ( T _ { 1 } , \dots , T _ { D } ) : \mathbb { R } ^ { D } \to$ $\mathbb { R } ^ { D }$ is called triangular if $T _ { i }$ is a function of $\mathbf { x } _ { 1 : i }$ for each $i = 1 , \ldots , D$ . Such a triangular map $T$ is called increasing if $T _ { i }$ is an increasing function of $x _ { i }$ for each $i$ .

Proposition 4 ([Bogachev et al., 2005], Lemma 2.1). If $\mu$ and $\nu$ are absolutely continuous Borel probability measures on $\mathbb { R } ^ { D }$ , then there exists an increasing triangular transformation $T : \mathbb { R } ^ { D }  \mathbb { R } ^ { D } .$ , such that $\nu = T _ { \ast } \mu$ . This transformation is unique up to null sets of $\mu$ . A similar result holds for measures on $[ 0 , 1 ] ^ { D }$ .

Proposition 5. If $\mu$ is an absolutely continuous Borel probability measures on $\mathbb { R } ^ { D }$ and $\{ T _ { n } \}$ is a sequence of maps $\mathbb { R } ^ { D } \overset { \cdot } { \to } \mathbb { R } ^ { D }$ which converges pointwise to a map $T _ { \cdot }$ , then a sequence of measures $( \bar { T _ { n } } ) _ { * } \bar { \mu }$ weakly converges to $T _ { * } \mu$ .

Proof See Huang et al. [2018], Lemma 4. The result follows from the dominated convergence theorem.

As a corollary, to claim that a class of autoregressive flows $\mathbf { g } ( \cdot , \boldsymbol { \theta } ) : \mathbb { R } ^ { \bar { D } } \to \mathbb { R } ^ { D }$ is universal, it is enough to demonstrate that a family of coupling functions $h$ used in the class is dense in the set of all monotone functions in the pointwise convergence topology. In particular, Huang et al. [2018] used neural monotone networks for coupling functions, and Jaini et al. [2019a] used monotone polynomials. Using the theory outlined in this section, universality could also be proved for spline flows [Durkan et al., 2019a,b] with splines for coupling functions (see Section 3.4.4.4).

# 3.4.4 Coupling Functions

As described in the previous sections, coupling flows and autoregressive flows have a similar functional form and both have coupling functions as building blocks. A coupling function is a bijective differentiable function $\mathbf { h } ( \cdot , \theta ) : \bar { \mathbb { R } } ^ { d } \overset { \smile } {  }$ $\mathbb { R } ^ { d } .$ , parameterized by $\theta$ . In coupling flows, these functions are typically constructed by applying a scalar coupling function $\bar { h } ( \cdot , \theta ) : \mathbb { R } \to \mathbb { R }$ elementwise. In autoregressive flows, $d = 1$ and hence they are also scalar valued. Note that scalar coupling functions are necessarily (strictly) monotone. In this section we describe the scalar coupling functions commonly used in the literature.

3.4.4.1 Affine coupling: Two simple forms of coupling functions $h : \mathbb { R }  \mathbb { R }$ were proposed by [Dinh et al., 2015] in NICE (nonlinear independent component estimation). These were the additive coupling function:

$$
h (x; \theta) = x + \theta , \quad \theta \in \mathbb {R}, \tag {21}
$$

and the affine coupling function:

$$
h (x; \theta) = \theta_ {1} x + \theta_ {2}, \quad \theta_ {1} \neq 0, \theta_ {2} \in \mathbb {R}. \tag {22}
$$

Affine coupling functions are used for coupling flows in NICE [Dinh et al., 2015], RealNVP [Dinh et al., 2017], Glow [Kingma and Dhariwal, 2018] and for autoregressive architectures in IAF [Kingma et al., 2016] and MAF [Papamakarios et al., 2017]. They are simple and computation is efficient. However, they are limited in expressiveness and many flows must be stacked to represent complicated distributions.

3.4.4.2 Nonlinear squared flow: Ziegler and Rush [2019] proposed an invertible non-linear squared transformation defined by:

$$
h (x; \theta) = a x + b + \frac {c}{1 + (d x + h) ^ {2}}. \tag {23}
$$

Under some constraints on parameters $\theta = [ a , b , c , d , h ] \in$ $\mathbb { R } ^ { 5 }$ , the coupling function is invertible and its inverse is analytically computable as a root of a cubic polynomial (with only one real root). Experiments showed that these coupling functions facilitate learning multimodal distributions.

3.4.4.3 Continuous mixture CDFs: Ho et al. [2019] proposed the Flow $^ { + + }$ model, which contained several improvements, including a more expressive coupling function. The layer is almost like a linear transformation, but one also applies a monotone function to $x$ :

$$
h (x; \theta) = \theta_ {1} F (x, \theta_ {3}) + \theta_ {2}, \tag {24}
$$

where $\theta _ { 1 } \neq 0 .$ , $\theta _ { 2 } \in \mathbb { R }$ and $\theta _ { 3 } = [ \pmb { \pi } , \pmb { \mu } , \mathbf { s } ] \in \mathbb { R } ^ { K } \times \mathbb { R } ^ { K } \times \mathbb { R } _ { + } ^ { K }$ . The function $F ( x , \pi , \mu , \mathbf { s } )$ is the CDF of a mixture of $K$ logistics, postcomposed with an inverse sigmoid:

$$
F (x, \boldsymbol {\pi}, \boldsymbol {\mu}, \mathbf {s}) = \sigma^ {- 1} \left(\sum_ {j = 1} ^ {K} \pi_ {j} \sigma \left(\frac {x - \mu_ {j}}{s _ {j}}\right)\right). \tag {25}
$$

Note, that the post-composition with $\sigma ^ { - 1 } : [ 0 , 1 ] \to \mathbb { R }$ is used to ensure the right range for $h$ . Computation of the inverse is done numerically with the bisection algorithm. The derivative of the transformation with respect to $x$ is expressed in terms of PDF of logistic mixture (i.e., a linear combination of hyperbolic secant functions), and its computation is not expensive. An ablation study demonstrated that switching from an affine coupling function to a logistic mixture improved performance slightly.

3.4.4.4 Splines: A spline is a piecewise-polynomial or a piecewise-rational function which is specified by $K + 1$ points $( x _ { i } , y _ { i } ) _ { i = 0 } ^ { K } ,$ called knots, through which the spline passes. To make a useful coupling function, the spline should be monotone which will be the case if $x _ { i } ~ < ~ x _ { i + 1 }$ and $y _ { i } < y _ { i + 1 }$ . Usually splines are considered on a compact interval.

# Piecewise-linear and piecewise-quadratic:

Muller et al. [2018] used linear splines for coupling ¨ functions $h : [ 0 , 1 ]  [ 0 , 1 ]$ . They divided the domain into $K$ equal bins. Instead of defining increasing values for $y _ { i } ,$ they modeled $h$ as the integral of a positive piecewise-constant function:

$$
h (x; \theta) = \alpha \theta_ {b} + \sum_ {k = 1} ^ {b - 1} \theta_ {k}, \tag {26}
$$

where $\boldsymbol { \theta } \in \mathbb { R } ^ { K }$ is a probability vector, $b = \lfloor K x \rfloor$ (the bin that contains $x$ ), and $\alpha = K x - b$ (the position of $x$ in bin b). This map is invertible, if all $\theta _ { k } > 0$ , with derivative: $\frac { \partial h } { \partial x } = \dot { \theta } _ { b } K$ ∂x .

Muller et al. [2018] also used a monotone quadratic ¨ spline on the unit interval for a coupling function and modeled this as the integral of a positive piecewise-linear function. A monotone quadratic spline is invertible; finding its inverse map requires solving a quadratic equation.

Cubic Splines: Durkan et al. [2019a] proposed using monotone cubic splines for a coupling function. They do not restrict the domain to the unit interval, but instead use the form: $h ( \cdot ; \theta ) = \sigma ^ { - 1 } ( \hat { h } ( \sigma ( \cdot ) ; \theta ) )$ , where $\hat { h } ( \cdot ; \theta ) : [ 0 , 1 ]  [ 0 , 1 ]$ is a monotone cubic spline and $\sigma$ is a sigmoid. Steffen’s method is used to construct the spline. Here, one specifies $K + 1$ knots of the spline and boundary derivatives $\hat { h } ^ { \prime } ( 0 )$ and $\hat { h } ^ { \prime } ( 1 )$ . These quantities are modelled as the output of a neural network.

Computation of the derivative is easy as it is piecewisequadratic. A monotone cubic polynomial has only one real root and for inversion, one can find this either analytically or numerically. However, the procedure is numerically unstable if not treated carefully. The flow can be trained by gradient descent by differentiating through the numerical root finding method. However, Durkan et al. [2019b], noted numerical difficulties when the sigmoid saturates for values far from zero.

Rational quadratic splines: Durkan et al. [2019b] model a coupling function $h ( x ; \theta )$ as a monotone rationalquadratic spline on an interval as the identity function otherwise. They define the spline using the method of Gregory and Delbourgo [1982], by specifying $K + 1$ knots $\{ h ( x _ { i } ) \} _ { i = 0 } ^ { K }$ and the derivatives at the inner points: $\{ h ^ { \prime } ( x _ { i } ) \} _ { i = 1 } ^ { K - 1 }$ . These locations of the knots and their derivatives are modelled as the output of a neural network.

The derivative with respect to $x$ is a quotient derivative and the function can be inverted by solving a quadratic equation. Durkan et al. [2019b] used this coupling function with both a coupling architecture RQ-NSF(C) and an auto-regressive architecture RQ-NSF(AR).

3.4.4.5 Neural autoregressive flow: Huang et al. [2018] introduced Neural Autoregressive Flows (NAF) where a coupling function $h ( \cdot ; \theta )$ is modelled with a deep neural network. Typically such a network is not invertible, but they proved a sufficient condition for it to be bijective:

Proposition 6. If $\mathrm { N N } ( \cdot ) : \mathbb { R } \to \mathbb { R }$ is a multilayer percepton, such that all weights are positive and all activation functions are strictly monotone, then $\operatorname { N N } ( \cdot )$ is a strictly monotone function.

They proposed two forms of neural networks: the deep sigmoidal coupling function (NAF-DSF) and deep dense sigmoidal coupling function (NAF-DDSF). Both are MLPs with layers of sigmoid and logit units and non-negative weights; the former has a single hidden layer of sigmoid units, whereas the latter is more general and does not have this bottleneck. By Proposition 6, the resulting $h ( \cdot ; \theta )$ is a strictly monotone function. They also proved that a DSF network can approximate any strictly monotone univariate function and so NAF-DSF is a universal flow.

Wehenkel and Louppe [2019] noted that imposing positivity of weights on a flow makes training harder and requires more complex conditioners. To mitigate this, they introduced unconstrained monotonic neural networks (UMNN). The idea is in order to model a strictly monotone function, one can describe a strictly positive (or negative) function with a neural network and then integrate it numerically. They demonstrated that UMNN requires less parameters than NAF to reach similar performance, and so is more scalable for high-dimensional datasets.

3.4.4.6 Sum-of-Squares polynomial flow: Jaini et al. [2019a] modeled $h ( \cdot ; \theta )$ as a strictly increasing polynomial. They proved such polynomials can approximate any strictly monotonic univariate continuous function. Hence, the resulting flow (SOS - sum of squares polynomial flow) is a universal flow.

The authors observed that the derivative of an increasing single-variable polynomial is a positive polynomial. Then they used a classical result from algebra: all positive singlevariable polynomials are the sum of squares of polynomials. To get the coupling function, one needs to integrate the sum of squares:

$$
h (x; \theta) = c + \int_ {0} ^ {x} \sum_ {k = 1} ^ {K} \left(\sum_ {l = 0} ^ {L} a _ {k l} u ^ {l}\right) ^ {2} d u, \tag {27}
$$

where $L$ and $K$ are hyperparameters (and, as noted in the paper, can be chosen to be 2).

SOS is easier to train than NAF, because there are no restrictions on the parameters (like positivity of weights). For ${ \mathrm { L } } { = } 0$ , SOS reduces to the affine coupling function and so it is a generalization of the basic affine flow.

3.4.4.7 Piecewise-bijective coupling: Dinh et al. [2019] explore the idea that a coupling function does not need to be bijective, but just piecewise-bijective (Figure 5). Formally, they consider a function $h ( \cdot ; \theta ) : \mathbb { R } \ \xrightarrow { } \ \mathbb { R }$ and a covering of the domain into $K$ disjoint subsets: $\begin{array} { r } { \mathbb { R } = \bigcup _ { i = 1 } ^ { K } A _ { i } , } \end{array}$ such that the restriction of the function onto each subset $h ( \cdot ; \theta ) | _ { A _ { i } }$ is injective.

Dinh et al. [2019] constructed a flow f : $\mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ with a coupling architecture and piecewise-bijective coupling function in the normalizing direction - from data distribution to (simpler) base distribution. There is a covering of the data domain, and each subset of this covering is separately

![[Normalizing Flows: An Introduction and Review of Current Methods/images/33475213af0695304527bffb861c5c081596b105c197bbf6045cc17a83fc7360.jpg]]  
Fig. 5. Piecewise bijective coupling. The target domain (right) is divided into disjoint sections (colors) and each mapped by a monotone function (center) to the base distribution (left). For inverting the function, one samples a component of the base distribution using a gating network.

mapped to the base distribution. Each part of the base distribution now receives contributions from each subset of the data domain. For sampling, Dinh et al. [2019] proposed a probabilistic mapping from the base to data domain.

More formally, denote the target $y$ and base $z ,$ and consider a lookup function $\phi : \mathbb { R }  [ K ] = \{ 1 , \ldots , K \} ,$ such that $\begin{array} { r } { \phi ( y ) = k , } \end{array}$ if $y \in A _ { k }$ . One can define a new map $\mathbb { R }  \mathbb { R } \times [ K ] ,$ given by the rule $y \mapsto ( h ( y ) , \phi ( y ) ) .$ , and a density on a target space $p _ { Z , [ K ] } ( z , k ) = p _ { [ K ] | Z } ( k | z ) p _ { Z } ( z )$ One can think of this as an unfolding of the non-injective map $h$ . In particular, for each point $z$ one can find its preimage by sampling from $p _ { [ K ] \mid Z } ,$ which is called a gating network. Pushing forward along this unfolded map is now well-defined and one gets the formula for the density $p _ { Y }$ :

$$
p _ {Y} (y) = p _ {Z, [ K ]} (h (y), \phi (y)) | \mathrm {D} h (y) |. \tag {28}
$$

This real and discrete (RAD) flow efficiently learns distributions with discrete structures (multimodal distributions, distributions with holes, discrete symmetries etc).

# 3.5 Residual Flows

Residual networks [He et al., 2016] are compositions of the function of the form

$$
\mathbf {g} (\mathbf {x}) = \mathbf {x} + F (\mathbf {x}). \tag {29}
$$

Such a function is called a residual connection, and here the residual block $F ( \cdot )$ is a feed-forward neural network of any kind (a CNN in the original paper).

The first attempts to build a reversible network architecture based on residual connections were made in RevNets [Gomez et al., 2017] and iRevNets [Jacobsen et al., 2018]. Their main motivation was to save memory during training and to stabilize computation. The central idea is a variation of additive coupling functions: consider a disjoint partition of $\mathbb { R } ^ { D } = \mathbb { R } ^ { d } \times \dot { \mathbb { R } } ^ { D - \check { d } }$ denoted by $\mathbf { x } = ( \mathbf { x } ^ { A } , \mathbf { x } ^ { B } )$ for the input and $\mathbf { y } = ( \mathbf { y } ^ { A } , \mathbf { y } ^ { B } )$ for the output, and define a function:

$$
\mathbf {y} ^ {A} = \mathbf {x} ^ {A} + F (\mathbf {x} ^ {B})
$$

$$
\mathbf {y} ^ {B} = \mathbf {x} ^ {B} + G (\mathbf {y} ^ {A}), \tag {30}
$$

where $F : \mathbb { R } ^ { D - d }  \mathbb { R } ^ { d }$ and $G : \mathbb { R } ^ { d }  \mathbb { R } ^ { D - d }$ are residual blocks. This network is invertible (by re-arranging the equations in terms of $\mathbf { x } ^ { A }$ and $\mathbf { x } ^ { B }$ and reversing their order) but computation of the Jacobian is inefficient.

A different point of view on reversible networks comes from a dynamical systems perspective via the observation

that a residual connection is a discretization of a first order ordinary differential equation (see Section 3.6 for more details). Chang et al. [2018, 2019] proposed several architectures, some of these networks were demonstrated to be invertible. However, the Jacobian determinants of these networks cannot be computed efficiently.

Other research has focused on making the residual connection $\mathbf { g } ( \cdot )$ invertible. A sufficient condition for the invertibility was found in [Behrmann et al., 2019]. They proved the following statement:

Proposition 7. A residual connection (29) is invertible, if the Lipschitz constant of the residual block is $\mathrm { L i p } ( F ) < 1$ .

There is no analytically closed form for the inverse, but it can be found numerically using fixed-point iterations (which, by the Banach theorem, converge if we assume $\mathrm { L i p } ( F ) < 1 )$ .

Controlling the Lipschitz constant of a neural network is not simple. The specific architecture proposed by Behrmann et al. [2019], called iResNet, uses a convolutional network for the residual block. It constrains the spectral radius of each convolutional layer in this network to be less than one.

The Jacobian determinant of the iResNet cannot be computed directly, so the authors propose to use a (biased) stochastic estimate. The Jacobian of the residual connection $\mathbf { g }$ in Equation (29) is: $\mathrm { D } \mathbf { g } = I + \mathrm { D } F$ . Because the function $F$ is assumed to be Lipschitz with $\mathrm { L i p } ( F ) < 1 ,$ , one has: $| \mathrm { d e t } ( I + \mathrm { D } F ) | ~ = ~ \mathrm { d e t } ( \hat { I } + \mathrm { D } F )$ . Using the linear algebra identity, ln det $\mathbf { A } = \operatorname { T r } \ln \mathbf { A }$ we have:

$$
\ln | \det  D g | = \ln \det  (I + D F) = \operatorname {T r} (\ln (I + D F)), \tag {31}
$$

Then one considers a power series for the trace of the matrix logarithm:

$$
\operatorname {T r} (\ln (I + D F)) = \sum_ {k = 1} ^ {\infty} (- 1) ^ {k + 1} \frac {\operatorname {T r} (\mathrm {D} F) ^ {k}}{k}. \tag {32}
$$

By truncating this series one can calculate an approximation to the log Jacobian determinant of g. To efficiently compute each member of the truncated series, the Hutchinson trick was used. This trick provides a stochastic estimation of of a matrix trace $\begin{array} { r } { \mathbf { A } \ \in \ \mathbb { R } ^ { D \times D } . } \end{array}$ , using the relation: $\mathbf { T r A } \ =$ $\mathbb { E } _ { p ( \mathbf { v } ) } [ \mathbf { v } ^ { T } \mathbf { A } \mathbf { v } ] ,$ , where $\mathbf { v } \in \mathbb { R } ^ { D } , \mathbb { E } [ \mathbf { v } ] = \mathbf { \check { 0 } } _ { \mathrm { \ell } }$ , and $\mathbf { c o v ( v ) } = I$ .

Truncating the power series gives a biased estimate of the log Jacobian determinant (the bias depends on the truncation error). An unbiased stochastic estimator was proposed by Chen et al. [2019] in a model they called a Residual flow. The authors used a Russian roulette estimator next term instead of truncation. Informally, every time one adds the $a _ { n + 1 }$ to the partial sum $\textstyle \sum _ { i = 1 } ^ { n } a _ { i }$ while calculating the series $\textstyle \sum _ { i = 1 } ^ { \infty } a _ { i } ,$ one flips a coin to decide if the calculation should be continued or stopped. During this process one needs to re-weight terms for an unbiased estimate.

# 3.6 Infinitesimal (Continuous) Flows

The residual connections discussed in the previous section can be viewed as discretizations of a first order ordinary differential equation (ODE) [E, 2017; Haber et al., 2018]:

$$
\frac {d}{d t} \mathbf {x} (t) = F (\mathbf {x} (t), \theta (t)), \tag {33}
$$

where $F : \mathbb { R } ^ { D } \times \Theta  \mathbb { R } ^ { D }$ is a function which determines the dynamic (the evolution function), $\Theta$ is a set of parameters and $\Dot { \theta } : \mathbb { R } \to \Theta$ is a parameterization. The discretization of this equation (Euler’s method) is

$$
\mathbf {x} _ {n + 1} - \mathbf {x} _ {n} = \varepsilon F (\mathbf {x} _ {n}, \theta_ {n}), \tag {34}
$$

and this is equivalent to a residual connection with a residual block $\varepsilon F { \bar { ( } } \cdot , \theta _ { n } )$ .

In this section we consider the case where we do not discretize but try to learn the continuous dynamical system instead. Such flows are called infinitesimal or continuous. We consider two distinct types. The formulation of the first type comes from ordinary differential equations, and of the second type from stochastic differential equations.

# 3.6.1 ODE-based methods

Consider an ODE as in Equation (33), where $t ~ \in ~ [ 0 , 1 ]$ . Assuming uniform Lipschitz continuity in x and continuity in $t ,$ the solution exists (at least, locally) and, given an initial condition ${ \bf x } ( 0 ) = { \bf z } ,$ is unique (Picard-Lindelof-Lipschitz- ¨ Cauchy theorem [Arnold, 1978]). We denote the solution at each time $t$ as $\Phi ^ { t } ( \mathbf { z } )$ .

Remark 8. At each time $t , \Phi ^ { t } ( \cdot ) : \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ is a diffeomorphism and satisfies the group law: $\Phi ^ { t } \circ \Phi ^ { s } = \Phi ^ { t + s }$ . Mathematically speaking, an ODE (33) defines a oneparameter group of diffeomorphisms on $\mathbb { R } ^ { D }$ . Such a group is called a smooth flow in dynamical systems theory and differential geometry [Katok and Hasselblatt, 1995].

When $t = 1$ , the diffeomorphism $\Phi ^ { 1 } ( \cdot )$ is called a time one map. The idea to model a normalizing flow as a time one map $\mathbf { y } = \mathbf { g } ( \mathbf { z } ) = \Phi ^ { 1 } ( \mathbf { z } )$ was presented by [Chen et al., 2018a] under the name Neural ODE (NODE). From a deep learning perspective this can be seen as an “infinitely deep” neural network with input z, output $\mathbf { y }$ and continuous weights $\theta ( t )$ . The invertibility of such networks naturally comes from the theorem of the existence and uniqueness of the solution of the ODE.

Training these networks for a supervised downstream task can be done by the adjoint sensitivity method which is the continuous analog of backpropagation. It computes the gradients of the loss function by solving a second (augmented) ODE backwards in time. For loss $L ( { \dot { \mathbf { x } } } ( t ) )$ , where ${ \bf x } ( t )$ is a solution of ODE (33), its sensitivity or adjoint is $\begin{array} { r } { \mathbf { \dot { a } } ( t ) ~ = ~ \frac { d L } { d \mathbf { x } ( t ) } } \end{array}$ . This is the analog of the derivative of the loss with respect to the hidden layer. In a standard neural network, the backpropagation formula computes this derivative: $\begin{array} { r } { \frac { d L } { d { \bf h } _ { n } } = \frac { d L } { d { \bf h } _ { n + 1 } } \frac { { \bf { \dot { d } } h } _ { n + 1 } } { d { \bf h } _ { n } } } \end{array}$ dhn dhn+1 dhn+1 . For “infinitely deep” neural network, this formula changes into an ODE:

$$
\frac {d \mathbf {a} (t)}{d t} = - \mathbf {a} (t) \frac {d F (\mathbf {x} (t) , \theta (t))}{d \mathbf {x} (t)}. \tag {35}
$$

For density estimation learning, we do not have a loss, but instead seek to maximize the log likelihood. For normalizing flows, the change of variables formula is given by another ODE:

$$
\frac {d}{d t} \log (p (\mathbf {x} (t))) = - \operatorname {T r} \left(\frac {d F (\mathbf {x} (t))}{d \mathbf {x} (t)}\right). \tag {36}
$$

Note that we no longer need to compute the determinant. To train the model and sample from $p \mathbf { r }$ we solve these ODEs, which can be done with any numerical ODE solver.

Grathwohl et al. [2019] used the Hutchinson estimator to calculate an unbiased stochastic estimate of the traceterm. This approach which they termed FFJORD reduces the complexity even further. Finlay et al. [2020] added two regularization terms into the loss function of FFJORD: the first term forces solution trajectories to follow straight lines with constant speed, and the second term is the Frobenius norm of the Jacobian. This regularization decreased the training time significantly and reduced the need for multiple GPUs. An interesting side-effect of using continuous ODEtype flows is that one needs fewer parameters to achieve the similar performance. For example, Grathwohl et al. [2019] show that for the comparable performance on CIFAR10, FFJORD uses less than $2 \%$ as many parameters as Glow.

Not all diffeomorphisms can be presented as a time one map of an ODE (see [Arango and Gomez, 2002; Katok and ´ Hasselblatt, 1995]). For example, one necessary condition is that the map is orientation preserving which means that the Jacobian determinant must be positive. This can be seen because the solution $\Phi ^ { t }$ is a (continuous) path in the space of diffeomorphisms from the identity map $\Phi ^ { 0 } = { \dot { I } } { \dot { d } }$ to the time one map $\Phi ^ { 1 }$ . Since the Jacobian determinant of a diffeomorphism is nonzero, its sign cannot change along the path. Hence, a time one map must have a positive Jacobian determinant. For example, consider a map $f : \mathbb { R } \to \mathbb { R } ,$ such that $f ( x ) = - x$ . It is obviously a diffeomorphism, but it can not be presented as a time one map of any ODE, because it is not orientation preserving.

Dupont et al. [2019] suggested how one can improve Neural ODE in order to be able to represent a broader class of diffeomorphisms. Their model is called Augmented Neural ODE (ANODE). They add variables $\hat { \mathbf { x } } ( t ) \in \mathbb { R } ^ { p }$ and consider a new ODE:

$$
\frac {d}{d t} \left[ \begin{array}{l} \mathbf {x} (t) \\ \hat {\mathbf {x}} (t) \end{array} \right] = \hat {F} \left(\left[ \begin{array}{l} \mathbf {x} (t) \\ \hat {\mathbf {x}} (t) \end{array} \right], \theta (t)\right) \tag {37}
$$

with initial conditions $\mathbf { x ( 0 ) } \ = \ \mathbf { z }$ and $\hat { \mathbf { x } } ( 0 ) ~ = ~ 0$ . The addition of $\hat { { \mathbf x } } ( t )$ in particular gives freedom for the Jacobian determinant to remain positive. As was demonstrated in the experiments, ANODE is capable of learning distributions that the Neural ODE cannot, and the training time is shorter. Zhang et al. [2019] proved that any diffeomorphism can be represented as a time one map of ANODE and so this is a universal flow.

A similar ODE-base approach was taken by Salman et al. [2018] in Deep Diffeomorphic Flows. In addition to modelling a path $\bar { \Phi } ^ { t } ( \cdot )$ in the space of all diffeomorphic transformations, for $t ~ \in ~ [ 0 , 1 ]$ , they proposed geodesic regularisation in which longer paths are punished.

# 3.6.2 SDE-based methods (Langevin flows)

The idea of the Langevin flow is simple; we start with a complicated and irregular data distribution $p \mathbf { { v } } ( \mathbf { { y } } )$ on $\mathbb { R } ^ { D }$ , and then mix it to produce the simple base distribution $p \mathbf { z } ( \mathbf { z } )$ . If this mixing obeys certain rules, then this procedure can be invertible. This idea was explored by Chen et al. [2018b]; Jankowiak and Obermeyer [2018]; Rezende and Mohamed [2015]; Salimans et al. [2015]; Sohl-Dickstein et al.

[2015]; Suykens et al. [1998]; Welling and Teh [2011]. We provide a high-level overview of the method, including the necessary mathematical background.

A stochastic differential equation (SDE) or Ito process ˆ describes a change of a random variable $\textbf { x } \in \ \mathbb { R } ^ { \mathbf { \hat { D } } }$ as a function of time $t \in \mathbb { R } _ { + }$ :

$$
d \mathbf {x} (t) = b (\mathbf {x} (t), t) d t + \sigma (\mathbf {x} (t), t) d B _ {t}, \tag {38}
$$

where $b ( \mathbf { x } , t ) \in \mathbb { R } ^ { D }$ is the drift coefficient, $\sigma ( \mathbf { x } , t ) \in \mathbb { R } ^ { D \times D }$ is the diffusion coefficient, and $B _ { t }$ is $D$ -dimensional Brownian motion. One can interpret the drift term as a deterministic change and the diffusion term as providing the stochasticity and mixing. Given some assumptions about these functions, the solution exists and is unique [Oksendal, 1992].

Given a time-dependent random variable ${ \bf x } ( t )$ we can consider its density function $\boldsymbol { p } ( \mathbf { x } , t )$ and this is also time dependent. If ${ \bf x } ( t )$ is a solution of Equation (38), its density function satisfies two partial differential equations describing the forward and backward evolution [Oksendal, 1992]. The forward evolution is given by Fokker-Plank equation or Kolmogorov’s forward equation:

$$
\frac {\partial}{\partial t} p (\mathbf {x}, t) = - \nabla_ {\mathbf {x}} \cdot (b (\mathbf {x}, t) p (\mathbf {x}, t)) + \sum_ {i, j} \frac {\partial^ {2}}{\partial x _ {i} \partial x _ {j}} D _ {i j} (\mathbf {x}, t) p (\mathbf {x}, t), \tag {39}
$$

where $\begin{array} { r } { D = \frac { 1 } { 2 } \sigma \sigma ^ { T } . } \end{array}$ , with the initial condition $p ( \cdot , 0 ) = p _ { \mathbf { Y } } ( \cdot )$ The reverse is given by Kolmogorov’s backward equation:

$$
- \frac {\partial}{\partial t} p (\mathbf {x}, t) = b (\mathbf {x}, t) \cdot \nabla_ {\mathbf {x}} (p (\mathbf {x}, t)) + \sum_ {i, j} D _ {i j} (\mathbf {x}, t) \frac {\partial^ {2}}{\partial x _ {i} \partial x _ {j}} p (\mathbf {x}, t), \tag {40}
$$

where $0 < t < T ,$ , and the initial condition is $p ( \cdot , T ) = p \mathbf { z } ( \cdot )$

Asymptotically the Langevin flow can learn any distribution if one picks the drift and diffusion coefficients appropriately [Suykens et al., 1998]. However this result is not very practical, because one needs to know the (unnormalized) density function of the data distribution.

One can see that if the diffusion coefficient is zero, the Itoˆ process reduces to the ODE (33), and the Fokker-Plank equation becomes a Liouville’s equation, which is connected to Equation (36) (see Chen et al. [2018a]). It is also equivalent to the form of the transport equation considered in Jankowiak and Obermeyer [2018] for stochastic optimization.

Sohl-Dickstein et al. [2015] and Salimans et al. [2015] suggested using MCMC methods to model the diffusion. They considered discrete time $t = 0 , \ldots , T$ . For each time $t ,$ $\mathbf { x } ^ { t }$ is a random variable where $\mathbf { x } ^ { 0 } = \mathbf { y }$ is the data point, and $\textbf { x } ^ { T } = \textbf { z }$ is the base point. The forward transition probability $q ( \mathbf { x } ^ { t } | \mathbf { x } ^ { t - 1 } )$ is taken to be either normal or binomial distribution with trainable parameters. Kolmogorov’s backward equation implies that the backward transition $p ( \mathbf { x } ^ { t - 1 } | \mathbf { x } ^ { t } )$ must have the same functional form as the forward transition (i.e., be either normal or binomial). Denote: $q ( \mathbf { x } ^ { 0 } ) = p \mathbf { v } ( \mathbf { y } )$ , the data distribution, and $p ( \mathbf { x } ^ { T } ) = p \mathbf { z } ( \mathbf { z } ) .$ , the base distribution. Applying the backward transition to the base distribution, one obtains a new density $p ( \mathbf { x } ^ { 0 } )$ , which one wants to match with $q ( \mathbf { x } ^ { 0 } )$ . Hence, the optimization objective is the log likelihood $\begin{array} { r } { { \cal L } = \int d { \bf x } ^ { 0 } q ( { \bf x } ^ { 0 } ) \log p ( { \bf x } ^ { 0 } ) } \end{array}$ . This is intractable, but one can find a lower bound as in variational inference.

Several papers have worked explicitly with the SDE [Chen et al., 2018b; Li et al., 2020; Liutkus et al., 2019;

TABLE 1 List of Normalizing Flows for which we show performance results.   

<table><tr><td>Architecture</td><td>Coupling function</td><td>Flow name</td></tr><tr><td rowspan="4">Coupling, 3.4.1</td><td>Affine, 3.4.4.1</td><td>RealNVP 
Glow</td></tr><tr><td>Mixture CDF, 3.4.4.3</td><td>Flow++</td></tr><tr><td>Splines, 3.4.4.4</td><td>quadratic (C) 
cubic 
RQ-NSF(C)</td></tr><tr><td>Piecewise Bijective, 3.4.4.7</td><td>RAD</td></tr><tr><td rowspan="4">Autoregressive, 3.4.2</td><td>Affine</td><td>MAF</td></tr><tr><td>Polynomial, 3.4.4.6</td><td>SOS</td></tr><tr><td>Neural Network, 3.4.4.5</td><td>NAF 
UMNN</td></tr><tr><td>Splines</td><td>quadratic (AR) 
RQ-NSF(AR)</td></tr><tr><td rowspan="2">Residual, 3.5</td><td></td><td>iResNet</td></tr><tr><td></td><td>Residual flow</td></tr><tr><td>ODE, 3.6.1</td><td></td><td>FFJORD</td></tr></table>

Peluchetti and Favaro, 2019; Tzen and Raginsky, 2019]. Chen et al. [2018b] use SDEs to create an interesting posterior for variational inference. They sample a latent variable $\mathbf { z } _ { 0 }$ conditioned on the input x, and then evolve $\mathbf { z } _ { 0 }$ with SDE. In practice this evolution is computed by discretization. By analogy to Neural ODEs, Neural Stochastic Differential Equations were proposed [Peluchetti and Favaro, 2019; Tzen and Raginsky, 2019]. In this approach coefficients of the SDE are modelled as neural networks, and black box SDE solvers are used for inference. To train Neural SDE one needs an analog of backpropagation, Tzen and Raginsky [2019] proposed the use of Kunita’s theory of stochastic flows. Following this, Li et al. [2020] derived the adjoint SDE whose solution gives the gradient of the original Neural SDE.

Note, that even though Langevin flows manifest nice mathematical properties, they have not found practical applications. In particular, none of the methods has been tested on baseline datasets for flows.

# 4 DATASETS AND PERFORMANCE

In this section we discuss datasets commonly used for training and testing normalizing flows. We provide comparison tables of the results as they were presented in the corresponding papers. The list of the flows for which we post the performance results is given in Table 1.

# 4.1 Tabular datasets

We describe datasets as they were preprocessed in Papamakarios et al. [2017] (Table 2)2. These datasets are relatively small and so are a reasonable first test of unconditional density estimation models. All datasets were cleaned and de-quantized by adding uniform noise, so they can be considered samples from an absolutely continuous distribution.

We use a collection of datasets from the UC Irvine machine learning repository [Dua and Graff, 2017].

1) POWER: a collection of electric power consumption measurements in one house over 47 months.

2. See https://github.com/gpapamak/maf

2) GAS: a collection of measurements from chemical sensors in several gas mixtures.   
3) HEPMASS: measurements from high-energy physics experiments aiming to detect particles with unknown mass.   
4) MINIBOONE: measurements from MiniBooNE experiment for observing neutrino oscillations.

In addition we consider the Berkeley segmentation dataset [Martin et al., 2001] which contains segmentations of natural images. Papamakarios et al. [2017] extracted $8 \times 8$ random monochrome patches from it.

In Table 3 we compare performance of flows for these tabular datasets. For experimental details, see the following papers: RealNVP [Dinh et al., 2017] and MAF [Papamakarios et al., 2017], Glow [Kingma and Dhariwal, 2018] and FFJORD [Grathwohl et al., 2019], NAF [Huang et al., 2018], UMNN [Wehenkel and Louppe, 2019], SOS [Jaini et al., 2019a], Quadratic Spline flow and RQ-NSF [Durkan et al., 2019b], Cubic Spline Flow [Durkan et al., 2019a].

Table 3 shows that universal flows (NAF, SOS, Splines) demonstrate relatively better performance.

# 4.2 Image datasets

These datasets summarized in Table 4. They are of increasing complexity and are preprocessed as in Dinh et al. [2017] by dequantizing with uniform noise (except for Flow $^ { + + }$ ).

Table 5 compares performance on the image datasets for unconditional density estimation. For experimental details, see: RealNVP for CIFAR-10 and ImageNet [Dinh et al., 2017], Glow for CIFAR-10 and ImageNet [Kingma and Dhariwal, 2018], RealNVP and Glow for MNIST, MAF and FFJORD [Grathwohl et al., 2019], SOS [Jaini et al., 2019a], RQ-NSF [Durkan et al., 2019b], UMNN [Wehenkel and Louppe, 2019], iResNet [Behrmann et al., 2019], Residual Flow [Chen et al., 2019], Flow $^ { \cdot + + }$ [Ho et al., 2019].

As of this writing Flow $^ { + + }$ [Ho et al., 2019] is the best performing approach. Besides using more expressive coupling layers (see Section 3.4.4.3) and a different architecture for the conditioner, variational dequantization was used instead of uniform. An ablation study shows that the change in dequantization approach gave the most significant improvement.

# 5 DISCUSSION AND OPEN PROBLEMS

# 5.1 Inductive biases

# 5.1.1 Role of the base measure

The base measure of a normalizing flow is generally assumed to be a simple distribution (e.g., uniform or Gaussian). However this doesn’t need to be the case. Any distribution where we can easily draw samples and compute the log probability density function is possible and the parameters of this distribution can be learned during training.

Theoretically the base measure shouldn’t matter: any distribution for which a CDF can be computed, can be simulated by applying the inverse CDF to draw from the uniform distribution. However in practice if structure is provided in the base measure, the resulting transformations may become easier to learn. In other words, the choice of base measure can be viewed as a form of prior or inductive

bias on the distribution and may be useful in its own right. For example, a trade-off between the complexity of the generative transformation and the form of base measure was explored in [Jaini et al., 2019b] in the context of modelling tail behaviour.

# 5.1.2 Form of diffeomorphisms

The majority of the flows explored are triangular flows (either coupling or autoregressive architectures). Residual networks and Neural ODEs are also being actively investigated and applied. A natural question to ask is: are there other ways to model diffeomorphisms which are efficient for computation? What inductive bias does the architecture impose? For instance, Spantini et al. [2017] investigate the relation between the sparsity of the triangular flow and Markov property of the target distribution.

A related question concerns the best way to model conditional normalizing flows when one needs to learn a conditional probability distribution. Trippe and Turner [2017] suggested using different flows for each condition, but this approach doesn’t leverage weight sharing, and so is inefficient in terms of memory and data usage. Atanov et al. [2019] proposed using affine coupling layers where the parameters $\theta$ depend on the condition. Conditional distributions are useful in particular for time series modelling, where one needs to find $p ( y _ { t } | \mathbf { y } _ { < t } )$ [Kumar et al., 2019].

# 5.1.3 Loss function

The majority of the existing flows are trained by minimization of KL-divergence between source and the target distributions (or, equivalently, with log-likelihood maximization). However, other losses could be used which would put normalizing flows in a broader context of optimal transport theory [Villani, 2003]. Interesting work has been done in this direction including Flow-GAN [Grover et al., 2018] and the minimization of the Wasserstein distance as suggested by [Arjovsky et al., 2017; Tolstikhin et al., 2018].

# 5.2 Generalisation to non-Euclidean spaces

# 5.2.1 Flows on manifolds.

Modelling probability distributions on manifolds has applications in many fields including robotics, molecular biology, optics, fluid mechanics, and plasma physics [Gemici et al., 2016; Rezende et al., 2020]. How best to construct a normalizing flow on a general differentiable manifold remains an open question. One approach to applying the normalizing flow framework on manifolds, is to find a base distribution on the Euclidean space and transfer it to the manifold of interest. There are two main approaches: 1) embed the manifold in the Euclidean space and “restrict” the measure, or 2) induce the measure from the tangent space to the manifold. We will briefly discuss each in turn.

One can also use differential structure to define measures on manifolds [Spivak, 1965]. Every differentiable and orientable manifold $M$ has a volume form $\omega$ , then for a Borel subset $U \subset M$ one can define its measure as $\textstyle \mu _ { \omega } ( U ) = \int _ { U } \omega$ . A Riemannian manifold has a natural volume form given by its metric tensor: $\omega = \sqrt { | g | } d x _ { 1 } \wedge \cdot \cdot \cdot \wedge d x _ { D }$ . Gemici et al. [2016] explore this approach considering an immersion of an $D$ -dimensional manifold $M$ into a Euclidean space:

TABLE 2 Tabular datasets: data dimensionality and number of training examples.   

<table><tr><td></td><td>POWER</td><td>GAS</td><td>HEPMASS</td><td>MINIBOONE</td><td>BSDS300</td></tr><tr><td>Dims</td><td>6</td><td>8</td><td>21</td><td>43</td><td>63</td></tr><tr><td>#Train</td><td>≈ 1.7M</td><td>≈ 800K</td><td>≈ 300K</td><td>≈ 30K</td><td>≈ 1M</td></tr></table>

TABLE 3 Average test log-likelihood (in nats) for density estimation on tabular datasets (higher the better). A number in parenthesis next to a flow indicates number of layers. MAF MoG is MAF with mixture of Gaussians as a base density.   

<table><tr><td></td><td>POWER</td><td>GAS</td><td>HEPMASS</td><td>MINIBOONE</td><td>BSDS300</td></tr><tr><td>MAF(5)</td><td>0.14±0.01</td><td>9.07±0.02</td><td>-17.70±0.02</td><td>-11.75±0.44</td><td>155.69±0.28</td></tr><tr><td>MAF(10)</td><td>0.24±0.01</td><td>10.08±0.02</td><td>-17.73±0.02</td><td>-12.24±0.45</td><td>154.93±0.28</td></tr><tr><td>MAF MoG</td><td>0.30±0.01</td><td>9.59±0.02</td><td>-17.39±0.02</td><td>-11.68±0.44</td><td>156.36±0.28</td></tr><tr><td>RealNVP(5)</td><td>-0.02±0.01</td><td>4.78±1.8</td><td>-19.62±0.02</td><td>-13.55±0.49</td><td>152.97±0.28</td></tr><tr><td>RealNVP(10)</td><td>0.17±0.01</td><td>8.33±0.14</td><td>-18.71±0.02</td><td>-13.84±0.52</td><td>153.28±1.78</td></tr><tr><td>Glow</td><td>0.17</td><td>8.15</td><td>-18.92</td><td>-11.35</td><td>155.07</td></tr><tr><td>FFJORD</td><td>0.46</td><td>8.59</td><td>-14.92</td><td>-10.43</td><td>157.40</td></tr><tr><td>NAF(5)</td><td>0.62±0.01</td><td>11.91±0.13</td><td>-15.09±0.40</td><td>-8.86±0.15</td><td>157.73±0.04</td></tr><tr><td>NAF(10)</td><td>0.60±0.02</td><td>11.96±0.33</td><td>-15.32±0.23</td><td>-9.01±0.01</td><td>157.43±0.30</td></tr><tr><td>UMNN</td><td>0.63±0.01</td><td>10.89±0.70</td><td>-13.99±0.21</td><td>-9.67±0.13</td><td>157.98±0.01</td></tr><tr><td>SOS(7)</td><td>0.60±0.01</td><td>11.99±0.41</td><td>-15.15±0.10</td><td>-8.90±0.11</td><td>157.48±0.41</td></tr><tr><td>Quadratic Spline (C)</td><td>0.64±0.01</td><td>12.80±0.02</td><td>-15.35±0.02</td><td>-9.35±0.44</td><td>157.65±0.28</td></tr><tr><td>Quadratic Spline (AR)</td><td>0.66±0.01</td><td>12.91±0.02</td><td>-14.67±0.03</td><td>-9.72±0.47</td><td>157.42±0.28</td></tr><tr><td>Cubic Spline</td><td>0.65±0.01</td><td>13.14±0.02</td><td>-14.59±0.02</td><td>-9.06±0.48</td><td>157.24±0.07</td></tr><tr><td>RQ-NSF(C)</td><td>0.64±0.01</td><td>13.09±0.02</td><td>-14.75±0.03</td><td>-9.67±0.47</td><td>157.54±0.28</td></tr><tr><td>RQ-NSF(AR)</td><td>0.66±0.01</td><td>13.09±0.02</td><td>-14.01±0.03</td><td>-9.22±0.48</td><td>157.31±0.28</td></tr></table>

TABLE 4 Image datasets: data dimensionality and number of training examples for MNIST, CIFAR-10, ImageNet32 and ImageNet64 datasets.   

<table><tr><td></td><td>MNIST</td><td>CIFAR-10</td><td>ImNet32</td><td>ImNet64</td></tr><tr><td>Dims</td><td>784</td><td>3072</td><td>3072</td><td>12288</td></tr><tr><td>#Train</td><td>50K</td><td>90K</td><td>≈1.3M</td><td>≈1.3M</td></tr></table>

TABLE 5 Average test negative log-likelihood (in bits per dimension) for density estimation on image datasets (lower is better).   

<table><tr><td></td><td>MNIST</td><td>CIFAR-10</td><td>ImNet32</td><td>ImNet64</td></tr><tr><td>RealNVP</td><td>1.06</td><td>3.49</td><td>4.28</td><td>3.98</td></tr><tr><td>Glow</td><td>1.05</td><td>3.35</td><td>4.09</td><td>3.81</td></tr><tr><td>MAF</td><td>1.89</td><td>4.31</td><td></td><td></td></tr><tr><td>FFJORD</td><td>0.99</td><td>3.40</td><td></td><td></td></tr><tr><td>SOS</td><td>1.81</td><td>4.18</td><td></td><td></td></tr><tr><td>RQ-NSF(C)</td><td></td><td>3.38</td><td></td><td>3.82</td></tr><tr><td>UMNN</td><td>1.13</td><td></td><td></td><td></td></tr><tr><td>iResNet</td><td>1.06</td><td>3.45</td><td></td><td></td></tr><tr><td>Residual Flow</td><td>0.97</td><td>3.28</td><td>4.01</td><td>3.76</td></tr><tr><td>Flow++</td><td></td><td>3.08</td><td>3.86</td><td>3.69</td></tr></table>

$\phi : { \cal M }  \mathbb { R } ^ { N } .$ , where $N \geq D$ . In this case, one pullsback a Euclidean metric, and locally a volume form on $M$ is $\omega = \sqrt { \operatorname * { d e t } ( ( \mathrm { D } \phi ) ^ { T } \mathrm { D } \phi ) } d x _ { 1 } \wedge \cdot \cdot \cdot \wedge d x _ { D } ,$ where $\ Ḋ \mathrm Ḋ \phi Ḍ Ḍ$ is the Jacobian matrix of $\phi$ . Rezende et al. [2020] pointed out that the realization of this method is computationally hard, and proposed an alternative construction of flows on tori and spheres using diffeomorphisms of the one-dimensional circle as building blocks.

As another option, one can consider exponential maps $\exp _ { x } \ : \ T _ { x } M \  \ M ,$ , mapping a tangent space of a Rie-

mannian manifold (at some point $x$ ) to the manifold itself. If the manifold is geodesic complete, this map is globally defined, and locally is a diffeomorphism. A tangent space has a structure of a vector space, so one can choose an isomorphism $T _ { x } M \cong \mathbb { R } ^ { D }$ . Then for a base distribution with the density $p \mathbf { z }$ on $\mathbb { R } ^ { D }$ , one can push it forward on $M$ via the exponential map. Additionally, applying a normalizing flow to a base measure before pushing it to $M$ helps to construct multimodal distributions on $M$ . If the manifold $M$ is a hyberbolic space, the exponential map is a global diffeomorphism and all the formulas could be written explicitly. Using this method, Ovinnikov [2018] introduced the Gaussian reparameterization trick in a hyperbolic space and Bose et al. [2020] constructed hyperbolic normalizing flows.

Instead of a Riemannian structure, one can impose a Lie group structure on a manifold $G$ . In this case there also exists an exponential map $\exp { \textsf { : } } { \mathfrak { g } } \to G$ mapping a Lie algebra to the Lie group and one can use it to construct a normalizing flow on $G$ . Falorsi et al. [2019] introduced an analog of the Gaussian reparameterization trick for a Lie group.

# 5.2.2 Discrete distributions

Modelling distributions over discrete spaces is important in a range of problems, however the generalization of normalizing flows to discrete distributions remains an open problem in practice. Discrete latent variables were used by Dinh et al. [2019] as an auxiliary tool to pushforward continuous random variables along piecewise-bijective maps (see Section 3.4.4.7). However, can we define normalizing flows if one or both of our distributions are discrete? This could be useful for many applications including natural language modelling, graph generation and others.

To this end Tran et al. [2019] model bijective functions on a finite set and show that, in this case, the change of variables is given by the formula: $p _ { \mathbf { Y } } ( \mathbf { y } ) \ = \ p _ { \mathbf { Z } } ( \mathbf { g } ^ { - 1 } ( \mathbf { y } ) )$ , i.e., with no Jacobian term (compare with Definition 1). For backpropagation of functions with discrete variables they use the straight-through gradient estimator [Bengio et al., 2013]. However this method is not scalable to distributions with large numbers of elements.

Alternatively Hoogeboom et al. [2019b] models bijections on $\mathbb { Z } ^ { D }$ directly with additive coupling layers. Other approaches transform a discrete variable into a continuous latent variable with a variational autoencoder, and then apply normalizing flows in the continuous latent space [Wang and Wang, 2019; Ziegler and Rush, 2019].

A different approach is dequantization, (i.e., adding noise to discrete data to make it continuous) which can be used with ordinal variables, e.g., discretized pixel intensities. The noise can be uniform but other forms are possible and this dequantization can even be learned as a latent variable model [Ho et al., 2019; Hoogeboom et al., 2020]. Hoogeboom et al. [2020] analyzed how different choices of dequantization objectives and dequantization distributions affect the performance.

# ACKNOWLEDGMENTS

The authors would like to thank Matt Taylor and Kry Yik-Chau Lui for their insightful comments.

# REFERENCES

A. Abdelhamed, M. A. Brubaker, and M. S. Brown, “Noise flow: Noise modeling with conditional normalizing flows,” in Proceedings of the IEEE International Conference on Computer Vision, 2019, pp. 3165–3173.   
J. Agnelli, M. Cadeiras, E. Tabak, T. Cristina, and E. Vanden-Eijnden, “Clustering and classification through normalizing flows in feature space,” Multiscale Modeling and Simulation, vol. 8, pp. 1784–1802, 2010.   
J. Arango and A. Gomez, “Diffeomorphisms as time one ´ maps,” Aequationes Math., vol. 64, pp. 304–314, 2002.   
M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein Generative Adversarial Networks,” in ICML, 2017.   
V. Arnold, Ordinary Differential Equations. The MIT Press, 1978.   
A. Atanov, A. Volokhova, A. Ashukha, I. Sosnovik, and D. Vetrov, “Semi-Conditional Normalizing Flows for Semi-Supervised Learning,” in Workshop on Invertible Neural Nets and Normalizing Flows, ICML, 2019.   
J. Behrmann, D. Duvenaud, and J.-H. Jacobsen, “Invertible residual networks,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.   
Y. Bengio, N. Leonard, and A. Courville, “Estimating or ´ propagating gradients through stochastic neurons for conditional computation,” arXiv preprint, arXiv:1308.3432, 2013.   
V. Bogachev, A. Kolesnikov, and K. Medvedev, “Triangular transformations of measures,” Sbornik Math., vol. 196, no. 3-4, pp. 309–335, 2005.   
A. J. Bose, A. Smofsky, R. Liao, P. Panangaden, and W. L. Hamilton, “Latent Variable Modelling with Hyperbolic

Normalizing Flows,” arXiv preprint, arXiv:2002.06336, 2020.   
S. R. Bowman, L. Vilnis, O. Vinyals, A. M. Dai, R. Jozefowicz, ´ and S. Bengio, “Generating sentences from a continuous space,” in CoNLL, 2015.   
B. Chang, L. Meng, E. Haber, L. Ruthotto, D. Begert, and E. Holtham, “Reversible Architectures for Arbitrarily Deep Residual Neural Networks,” in AAAI, 2018.   
B. Chang, M. Chen, E. Haber, and E. H. Chi, “AntisymmetricRNN: A dynamical system view on recurrent neural networks,” in ICLR, 2019.   
C. Chen, C. Li, L. Chen, W. Wang, Y. Pu, and L. Carin, “Continuous-Time Flows for Efficient Inference and Density Estimation,” in ICML, 2018.   
R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, “Neural ordinary differential equations,” Advances in Neural Information Processing Systems, 2018.   
R. T. Q. Chen, J. Behrmann, D. Duvenaud, and J.-H. Jacobsen, “Residual Flows for Invertible Generative Modeling,” Advances in Neural Information Processing Systems, 2019.   
A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sengupta, and A. A. Bharath, “Generative adversarial networks: An overview,” IEEE Signal Processing Magazine, vol. 35, pp. 53–65, 2018.   
H. P. Das, P. Abbeel, and C. J. Spanos, “Dimensionality Reduction Flows,” arXiv preprint, arXiv:1908.01686, 2019.   
L. Dinh, D. Krueger, and Y. Bengio, “NICE: Non-linear Independent Components Estimation,” in ICLR Workshop, 2015.   
L. Dinh, J. Sohl-Dickstein, and S. Bengio, “Density Estimation using Real NVP,” in ICLR, 2017.   
L. Dinh, J. Sohl-Dickstein, R. Pascanu, and H. Larochelle, “A RAD approach to deep mixture models,” in ICLR Workshop, 2019.   
D. Dua and C. Graff, “UCI Machine Learning Repository,” 2017.   
E. Dupont, A. Doucet, and Y. W. Teh, “Augmented Neural ODEs,” Advances in Neural Information Processing Systems, 2019.   
C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, “Cubic-spline flows,” in Workshop on Invertible Neural Networks and Normalizing Flows, ICML, 2019.   
, “Neural Spline Flows,” Advances in Neural Information Processing Systems, 2019.   
W. E, “A proposal on machine learning via dynamical systems,” Communications in Mathematics and Statistics, vol. 5, pp. 1–11, 2017.   
P. Esling, N. Masuda, A. Bardet, R. Despres, and A. Chemla-Romeu-Santos, “Universal audio synthesizer control with normalizing flows,” arXiv preprint, arXiv:1907.00971, 2019.   
L. Falorsi, P. de Haan, T. R. Davidson, and P. Forre, “Repa- ´ rameterizing Distributions on Lie Groups,” arXiv preprint, arXiv:1903.02958, 2019.   
C. Finlay, J.-H. Jacobsen, L. Nurbekyan, and A. M. Oberman, “How to train your neural ODE,” arXiv preprint, arXiv:2002.02798, 2020.   
M. C. Gemici, D. Rezende, and S. Mohamed, “Normalizing Flows on Riemannian Manifolds,” arXiv preprint, arXiv:1611.02304, 2016.   
M. Germain, K. Gregor, I. Murray, and H. Larochelle, “MADE: Masked Autoencoder for Distribution Estima-

tion,” in ICML, 2015.   
A. N. Gomez, M. Ren, R. Urtasun, and R. B. Grosse, “The Reversible Residual Network: Backpropagation Without Storing Activations,” Advances in Neural Information Processing Systems, 2017.   
I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. C. Courville, and Y. Bengio, “Generative Adversarial Nets,” Advances in Neural Information Processing Systems, 2014.   
W. Grathwohl, R. T. Q Chen, J. Bettencourt, I. Sutskever, and D. Duvenaud, “FFJORD: Free-form continuous dynamics for scalable reversible generative models,” in ICLR, 2019.   
J. Gregory and R. Delbourgo, “Piecewise rational quadratic interpolation to monotonic data,” IMA Journal of Numerical Analysis, vol. 2, no. 2, pp. 123–130, 1982.   
A. Grover, M. Dhar, and S. Ermon, “Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in Generative Models,” in AAAI, 2018.   
E. Haber, L. Ruthotto, and E. Holtham, “Learning across scales - a multiscale method for convolution neural networks,” in AAAI, 2018.   
L. Hasenclever, J. M. Tomczak, R. Van Den Berg, and M. Welling, “Variational Inference with Orthogonal Normalizing Flows,” in Workshop on Bayesian Deep Learning, NIPS, 2017.   
K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” in ICCV, 2015.   
—, “Deep Residual Learning for Image Recognition,” in CVPR, 2016.   
J. Ho, X. Chen, A. Srinivas, Y. Duan, and P. Abbeel, “Flow $^ { \cdot + + }$ : Improving flow-based generative models with variational dequantization and architecture design,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.   
E. Hoogeboom, R. V. D. Berg, and M. Welling, “Emerging Convolutions for Generative Normalizing Flows,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.   
E. Hoogeboom, J. W. Peters, R. van den Berg, and M. Welling, “Integer discrete flows and lossless compression,” in NeurIPS, 2019.   
E. Hoogeboom, T. S. Cohen, and J. M. Tomczak, “Learning discrete distributions by dequantization,” arXiv preprint, arXiv:2001.11235, 2020.   
C.-W. Huang, D. Krueger, A. Lacoste, and A. Courville, “Neural Autoregressive Flows,” in ICML, 2018.   
S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” in ICML, 2015.   
J.-H. Jacobsen, A. W. Smeulders, and E. Oyallon, “i-RevNet: Deep Invertible Networks,” in ICLR, 2018.   
P. Jaini, I. Kobyzev, M. Brubaker, and Y. Yu, “Tails of Triangular Flows,” arXiv preprint, arXiv:1907.04481, 2019.   
P. Jaini, K. A. Selby, and Y. Yu, “Sum-of-squares polynomial flow,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 5 2019.   
M. Jankowiak and F. Obermeyer, “Pathwise derivatives beyond the reparameterization trick,” in Proceedings of the 35th International Conference on Machine Learning, ICML, 2018.

G. Kanwar, M. S. Albergo, D. Boyda, K. Cranmer, D. C. Hackett, S. Racaniere, D. J. Rezende, and P. E. Shanahan, ` “Equivariant flow-based sampling for lattice gauge theory,” arXiv preprint, arXiv:2003.06413, 2020.   
A. Katok and B. Hasselblatt, Introduction to the modern theory of dynamical systems. Cambridge University Press, New York, 1995.   
S. Kim, S. gil Lee, J. Song, J. Kim, and S. Yoon, “FloWaveNet: A Generative Flow for Raw Audio,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2018.   
D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in Proceedings of the 2nd International Conference on Learning Representations, ICLR, 2014.   
, “An Introduction to Variational Autoencoders,” arXiv preprint, arXiv:1906.02691, 2019.   
D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling, “Improved Variational Inference with Inverse Autoregressive Flow,” in NIPS, 2016.   
D. P. Kingma and P. Dhariwal, “Glow: Generative flow with invertible 1x1 convolutions,” in Advances in Neural Information Processing Systems, 2018, pp. 10 215–10 224.   
J. Kohler, L. Klein, and F. No ¨ e, “Equivariant flows: sampling ´ configurations for multi-body systems with symmetric energies,” in Workshop on Machine Learning and the Physical Sciences, NeurIPS, 2019.   
D. Koller and N. Friedman, Probabilistic Graphical Models. Massachusetts: MIT Press, 2009.   
M. Kumar, M. Babaeizadeh, D. Erhan, C. Finn, S. Levine, L. Dinh, and D. Kingma, “VideoFlow: A Flow-Based Generative Model for Video,” in Workshop on Invertible Neural Nets and Normalizing Flows, ICML, 2019.   
P. M. Laurence, R. J. Pignol, and E. G. Tabak, “Constrained density estimation,” Proceedings of the 2011 Wolfgang Pauli Institute conference on energy and commodity trading, Springer Verlag, pp. 259–284, 2014.   
X. Li, T.-K. L. Wong, R. T. Q. Chen, and D. Duvenaud, “Scalable Gradients for Stochastic Differential Equations,” arXiv preprint, arXiv:2001.01328, 2020.   
A. Liutkus, U. Simsekli, S. Majewski, A. Durmus, and F.-R. Stoter, “Sliced-Wasserstein Flows: Nonparametric Gener- ¨ ative Modeling via Optimal Transport and Diffusions,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.   
A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectifier Nonlinearities Improve Neural Network Acoustic Models,” in ICML, 2013.   
K. Madhawa, K. Ishiguro, K. Nakago, and M. Abe, “Graph-NVP: An Invertible Flow Model for Generating Molecular Graphs,” arXiv preprint, arXiv:1905.11600, 2019.   
D. Martin, C. Fowlkes, D. Tal, and J. Malik, “A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics,” in Proceedings of the 8th International Conference on Computer Vision, ICCV, 2001.   
B. Mazoure, T. Doan, A. Durand, J. Pineau, and R. D. Hjelm, “Leveraging exploration in off-policy algorithms via normalizing flows,” in 3rd Conference on Robot Learning (CoRL 2019), 2019.   
K. V. Medvedev, “Certain properties of triangular transformations of measures,” Theory Stoch. Process., vol. 14(30),

pp. 95–99, 2008.   
T. Muller, B. McWilliams, F. Rousselle, M. Gross, and J. No-¨ vak, “Neural Importance Sampling,” ACM Transactions on Graphics (TOG), vol. 38, 2018.   
P. Nadeem Ward, A. Smofsky, and A. Joey Bose, “Improving exploration in soft-actor-critic with normalizing flows policies,” in Workshop on Invertible Neural Networks and Normalizing Flows, ICML, 2019.   
F. Noe, S. Olsson, J. K ´ ohler, and H. Wu, “Boltzmann gener- ¨ ators: Sampling equilibrium states of many-body systems with deep learning,” Science, vol. 365, 2019.   
B. Oksendal, Stochastic Differential Equations (3rd Ed.): An Introduction with Applications. Berlin, Heidelberg: Springer-Verlag, 1992.   
I. Ovinnikov, “Poincare Wasserstein Autoencoder,” in ´ Bayesian Deep Learning Workshop, NeurIPS, 2018.   
G. Papamakarios, T. Pavlakou, and I. Murray, “Masked Autoregressive Flow for Density Estimation,” in NIPS, 2017.   
G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan, “Normalizing Flows for Probabilistic Modeling and Inference,” arXiv preprint, arXiv:1912.02762, 2019.   
S. Peluchetti and S. Favaro, “Neural Stochastic Differential Equations,” arXiv preprint, arXiv:1905.11065, 2019.   
R. Prenger, R. Valle, and B. Catanzaro, “Waveglow: A flow-based generative network for speech synthesis,” in ICASSP, 2019.   
D. J. Rezende and S. Mohamed, “Variational Inference with Normalizing Flows,” in ICML, 2015.   
D. J. Rezende, G. Papamakarios, S. Racaniere, M. S. Al-` bergo, G. Kanwar, P. E. Shanahan, and K. Cranmer, “Normalizing Flows on Tori and Spheres,” arXiv preprint, arXiv:2002.02428, 2020.   
O. Rippel and R. P. Adams, “High-dimensional probability estimation with deep density models,” arXiv preprint arXiv:1302.5125, 2013.   
T. Salimans, A. Diederik, D. P. Kingma, and M. Welling, “Markov Chain Monte Carlo and Variational Inference: Bridging the Gap,” in ICML, 2015.   
T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved Techniques for Training GANs,” in NIPS, 2016.   
H. Salman, P. Yadollahpour, T. Fletcher, and N. Batmanghelich, “Deep diffeomorphic normalizing flows,” arXiv preprint, arXiv:1810.03256, 2018.   
J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in Proceedings of the 32nd International Conference on Machine Learning, ICML, 2015.   
A. Spantini, D. Bigoni, and Y. Marzouk, “Inference via low-dimensional couplings,” Journal of Machine Learning Research, vol. 19, 03 2017.   
M. Spivak, Calculus on Manifolds: A Modern Approach to Classical Theorems of Advanced Calculus. Print, 1965.   
J. Suykens, H. Verrelst, and J. Vandewalle, “On-Line Learning Fokker-Planck Machine,” Neural Processing Letters, vol. 7, pp. 81–89, 1998.   
E. G. Tabak and C. V. Turner, “A Family of Nonparametric Density Estimation Algorithms,” Communications on Pure and Applied Mathematics, vol. 66, no. 2, pp. 145–164, 2013.

E. G. Tabak and E. Vanden-Eijnden, “Density Estimation by Dual Ascent of the Log-Likelihood,” Communications in Mathematical Sciences, vol. 8, no. 1, pp. 217–233, 2010.   
I. O. Tolstikhin, O. Bousquet, S. Gelly, and B. Scholkopf, ¨ “Wasserstein Auto-Encoders,” in ICLR, 2018.   
J. Tomczak and M. Welling, “Improving Variational Auto-Encoders using convex combination linear Inverse Autoregressive Flow,” Benelearn, 2017.   
J. M. Tomczak and M. Welling, “Improving variational auto-encoders using householder flow,” arXiv preprint arXiv:1611.09630, 2016.   
A. Touati, H. Satija, J. Romoff, J. Pineau, and P. Vincent, “Randomized value functions via multiplicative normalizing flows,” in UAI2019: Conference on Uncertainty in Artificial Intelligence, 2019.   
D. Tran, K. Vafa, K. Agrawal, L. Dinh, and B. Poole, “Discrete Flows: Invertible Generative Models of Discrete Data,” in ICLR Workshop, 2019.   
B. L. Trippe and R. E. Turner, “Conditional Density Estimation with Bayesian Normalising Flows,” in Workshop on Bayesian Deep Learning, NIPS, 2017.   
B. Tzen and M. Raginsky, “Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit,” arXiv preprint, arXiv:1905.09883, 2019.   
R. van den Berg, L. Hasenclever, J. M. Tomczak, and M. Welling, “Sylvester normalizing flows for variational inference,” in Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence, UAI, 2018.   
A. van den Oord, Y. Li, I. Babuschkin, K. Simonyan, O. Vinyals, K. Kavukcuoglu, G. van den Driessche, E. Lockhart, L. C. Cobo, F. Stimberg, N. Casagrande, D. Grewe, S. Noury, S. Dieleman, E. Elsen, N. Kalchbrenner, H. Zen, A. Graves, H. King, T. Walters, D. Belov, and D. Hassabis, “Parallel wavenet: Fast high-fidelity speech synthesis,” in ICML, 2017.   
C. Villani, Topics in optimal transportation (Graduate Studies in Mathematics 58). American Mathematical Society, Providence, RI, 2003.   
K. Wang, C. Gou, Y. Duan, Y. Lin, X. Zheng, and F. yue Wang, “Generative adversarial networks: introduction and outlook,” IEEE/CAA Journal of Automatica Sinica, vol. 4, pp. 588–598, 2017.   
P. Z. Wang and W. Y. Wang, “Riemannian Normalizing Flow on Variational Wasserstein Autoencoder for Text Modeling,” arXiv preprint, arXiv:1904.02399, 2019.   
A. Wehenkel and G. Louppe, “Unconstrained Monotonic Neural Networks,” arXiv preprint, arXiv:1908.05164, 2019.   
M. Welling and Y. W. Teh, “Bayesian Learning via Stochastic Gradient Langevin Dynamics,” in ICML, 2011.   
P. Wirnsberger, A. Ballard, G. Papamakarios, S. Abercrombie, S. Racaniere, A. Pritzel, D. Jimenez Rezende, and ` C. Blundell, “Targeted free energy estimation via learned mappings,” arXiv preprint, arXiv:2002.04913, 2020.   
K. W. K. Wong, G. Contardo, and S. Ho, “Gravitational wave population inference with deep flow-based generative network,” arXiv preprint, arXiv:2002.09491, 2020.   
H. Zhang, X. Gao, J. Unterman, and T. Arodz, “Approximation Capabilities of Neural Ordinary Differential Equations,” arXiv preprint, arXiv:1907.12998, 2019.   
G. Zheng, Y. Yang, and J. Carbonell, “Convolutional Normalizing Flows,” in Workshop on Theoretical Foundations

and Applications of Deep Generative Models, ICML, 2018. Z. M. Ziegler and A. M. Rush, “Latent Normalizing Flows for Discrete Sequences,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.

![[Normalizing Flows: An Introduction and Review of Current Methods/images/8cdc832aa0b7a5320fa12d0c2085b6f2ca1dfee129e1305c1e7f6ff85bf3b8e1.jpg]]

Ivan Kobyzev Ivan Kobyzev received his Masters degree in Mathematical Physics from St Petersburg State University, Russia, in 2011, and his PhD in Mathematics from Western University, Canada, in 2016. He did two postdocs in Mathematics and in Computer Science at the University of Waterloo. Currently he is a researcher at Borealis AI. His research interests include Algebra, Generative Models, Cognitive Computing, Natural Language Processing.

![[Normalizing Flows: An Introduction and Review of Current Methods/images/eb70b0cd82afa26df1d4a9c1a3b717ec10543961124df2fded659bc339fff458.jpg]]

Simon J.D. Prince Simon Prince holds a Masters by Research from University College London and a doctorate from the University of Oxford. He has a diverse research background and has published in wide-ranging areas including Computer Vision, Neuroscience, HCI, Computer Graphics, Medical Imaging, and Augmented Reality. He is also the author of a popular textbook on Computer Vision. From 2005-2012 Dr. Prince was a tenured faculty member in the Department of Computer Science at University College Lon-

don, where he taught courses in Computer Vision, Image Processing and Advanced Statistical Methods. During this time, he was Director of the M.Sc. in Computer Vision, Graphics and Imaging. Dr. Prince worked in industry applying AI to computer graphics software. Currently he is a Research Director of Borealis AI’s Montreal office.

![[Normalizing Flows: An Introduction and Review of Current Methods/images/5544b45169a42e96b55c0e42c16a2ff0ef0b954e67c7ffa6a732cd8df956dd27.jpg]]

Marcus A. Brubaker Marcus A. Brubaker received his PhD in 2011 at the University of Toronto. He did postdocs at the Toyota Technological Institute at Chicago, Toronto Rehabilitation Hospital and the University of Toronto. His research interests span computer vision, machine learning and statistics. Dr. Brubaker is an Assistant Professor at York University (Toronto, Canada), an Adjunct Professor at the University of Toronto and a Faculty Affiliate of the Vector Institute. He is also an Academic Advisor to Bo-

realis AI where he previously worked as the Research Director of the Toronto office. He is currently an Associate Editor for the journal IET Computer Vision and has served as a reviewer and area chair for many computer vision and machine learning conferences.