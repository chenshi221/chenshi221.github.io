---
title: "Flow Matching for Generative Modeling"
source: "https://ar5iv.labs.arxiv.org/html/2210.02747"
author:
published:
created: 2026-04-30
description: "We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulatio…"
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/05_%E5%85%B6%E4%BB%96/Flow%20Matching%20for%20Generative%20Modeling%2C%20Yaron%20Lipman%20et%20al.%2C%202022.no_watermark.zh-CN.dual.pdf"
---
Yaron Lipman <sup>1,2</sup> Ricky T. Q. Chen <sup>1</sup> Heli Ben-Hamu <sup>2</sup> Maximilian Nickel <sup>1</sup> Matt Le <sup>1</sup>  
<sup>1</sup> Meta AI (FAIR) <sup>2</sup> Weizmann Institute of Science

###### Abstract

We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples—which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.

## 1 Introduction

Deep generative models are a class of deep learning algorithms aimed at estimating and sampling from an unknown data distribution. The recent influx of amazing advances in generative modeling, e.g., for image generation [^36] [^37], is mostly facilitated by the scalable and relatively stable training of diffusion-based models [^18] [^44]. However, the restriction to simple diffusion processes leads to a rather confined space of sampling probability paths, resulting in very long training times and the need to adopt specialized methods (e.g., [^42] [^52]) for efficient sampling.

In this work we consider the general and deterministic framework of Continuous Normalizing Flows (CNFs; [^7]). CNFs are capable of modeling arbitrary probability path

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet128/imagenet128_curated_.png)

Figure 1: Unconditional ImageNet-128 samples of a CNF trained using Flow Matching with Optimal Transport probability paths.

and are in particular known to encompass the probability paths modeled by diffusion processes [^45]. However, aside from diffusion that can be trained efficiently via, e.g., denoising score matching [^49], no scalable CNF training algorithms are known. Indeed, maximum likelihood training (e.g., [^16]) require expensive numerical ODE simulations, while existing simulation-free methods either involve intractable integrals [^38] or biased gradients [^4].

The goal of this work is to propose Flow Matching (FM), an efficient simulation-free approach to training CNF models, allowing the adoption of general probability paths to supervise CNF training. Importantly, FM breaks the barriers for scalable CNF training beyond diffusion, and sidesteps the need to reason about diffusion processes to directly work with probability paths.

In particular, we propose the Flow Matching objective (Section 3), a simple and intuitive training objective to regress onto a target vector field that generates a desired probability path. We first show that we can construct such target vector fields through per-example (i.e., conditional) formulations. Then, inspired by denoising score matching, we show that a per-example training objective, termed Conditional Flow Matching (CFM), provides equivalent gradients and does not require explicit knowledge of the intractable target vector field. Furthermore, we discuss a general family of per-example probability paths (Section 4) that can be used for Flow Matching, which subsumes existing diffusion paths as special instances. Even on diffusion paths, we find that using FM provides more robust and stable training, and achieves superior performance compared to score matching. Furthermore, this family of probability paths also includes a particularly interesting case: the vector field that corresponds to an Optimal Transport (OT) displacement interpolant [^30]. We find that conditional OT paths are simpler than diffusion paths, forming straight line trajectories whereas diffusion paths result in curved paths. These properties seem to empirically translate to faster training, faster generation, and better performance.

We empirically validate Flow Matching and the construction via Optimal Transport paths on ImageNet, a large and highly diverse image dataset. We find that we can easily train models to achieve favorable performance in both likelihood estimation and sample quality amongst competing diffusion-based methods. Furthermore, we find that our models produce better trade-offs between computational cost and sample quality compared to prior methods. Figure 1 depicts selected unconditional ImageNet 128 $\times$ 128 samples from our model.

## 2 Preliminaries: Continuous Normalizing Flows

Let $\mathbb{R}^{d}$ denote the data space with data points $x=(x^{1},\ldots,x^{d})\in\mathbb{R}^{d}$. Two important objects we use in this paper are: the *probability density path* $p:[0,1]\times\mathbb{R}^{d}\rightarrow\mathbb{R}_{>0}$, which is a time dependent <sup>1</sup> probability density function, i.e., $\int p_{t}(x)dx=1$, and a *time-dependent vector field*, $v:[0,1]\times\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$. A vector field $v_{t}$ can be used to construct a time-dependent diffeomorphic map, called a *flow*, $\phi:[0,1]\times\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$, defined via the ordinary differential equation (ODE):

$$
\displaystyle\frac{d}{dt}\phi_{t}(x)
$$
 
$$
\displaystyle=v_{t}(\phi_{t}(x))
$$
 
$$
\displaystyle\phi_{0}(x)
$$
 
$$
\displaystyle=x
$$

Previously, [^7] suggested modeling the vector field $v_{t}$ with a neural network, $v_{t}(x;\theta)$, where $\theta\in\mathbb{R}^{p}$ are its learnable parameters, which in turn leads to a deep parametric model of the flow $\phi_{t}$, called a *Continuous Normalizing Flow* (CNF). A CNF is used to reshape a simple prior density $p_{0}$ (e.g., pure noise) to a more complicated one, $p_{1}$, via the push-forward equation

$$
p_{t}=[\phi_{t}]_{*}p_{0}
$$

where the push-forward (or change of variables) operator $*$ is defined by

$$
[\phi_{t}]_{*}p_{0}(x)=p_{0}(\phi_{t}^{-1}(x))\det\left[\frac{\partial\phi_{t}^{-1}}{\partial x}(x)\right].
$$

A vector field $v_{t}$ is said to *generate* a probability density path $p_{t}$ if its flow $\phi_{t}$ satisfies equation 3. One practical way to test if a vector field generates a probability path is using the continuity equation, which is a key component in our proofs, see Appendix B. We recap more information on CNFs, in particular how to compute the probability $p_{1}(x)$ at an arbitrary point $x\in\mathbb{R}^{d}$ in Appendix C.

## 3 Flow Matching

Let $x_{1}$ denote a random variable distributed according to some unknown data distribution $q(x_{1})$. We assume we only have access to data samples from $q(x_{1})$ but have no access to the density function itself. Furthermore, we let $p_{t}$ be a probability path such that $p_{0}=p$ is a simple distribution, e.g., the standard normal distribution $p(x)={\mathcal{N}}(x|0,I)$, and let $p_{1}$ be approximately equal in distribution to $q$. We will later discuss how to construct such a path. The Flow Matching objective is then designed to match this target probability path, which will allow us to flow from $p_{0}$ to $p_{1}$.

Given a target probability density path $p_{t}(x)$ and a corresponding vector field $u_{t}(x)$, which generates $p_{t}(x)$, we define the Flow Matching (FM) objective as

$$
\mathcal{L}_{\scriptscriptstyle\text{FM}}(\theta)=\mathbb{E}_{t,p_{t}(x)}\|v_{t}(x)-u_{t}(x)\|^{2},
$$

where $\theta$ denotes the learnable parameters of the CNF vector field $v_{t}$ (as defined in Section 2), $t\sim{\mathcal{U}}[0,1]$ (uniform distribution), and $x\sim p_{t}(x)$. Simply put, the FM loss regresses the vector field $u_{t}$ with a neural network $v_{t}$. Upon reaching zero loss, the learned CNF model will generate $p_{t}(x)$.

Flow Matching is a simple and attractive objective, but naïvely on its own, it is intractable to use in practice since we have no prior knowledge for what an appropriate $p_{t}$ and $u_{t}$ are. There are many choices of probability paths that can satisfy $p_{1}(x)\approx q(x)$, and more importantly, we generally don’t have access to a closed form $u_{t}$ that generates the desired $p_{t}$. In this section, we show that we can construct both $p_{t}$ and $u_{t}$ using probability paths and vector fields that are only defined *per sample*, and an appropriate method of aggregation provides the desired $p_{t}$ and $u_{t}$. Furthermore, this construction allows us to create a much more tractable objective for Flow Matching.

### 3.1 Constructing pt,utsubscript𝑝𝑡subscript𝑢𝑡p\_{t},u\_{t} from conditional probability paths and vector fields

A simple way to construct a target probability path is via a mixture of simpler probability paths: Given a particular data sample $x_{1}$ we denote by $p_{t}(x|x_{1})$ a *conditional probability path* such that it satisfies $p_{0}(x|x_{1})=p(x)$ at time $t=0$, and we design $p_{1}(x|x_{1})$ at $t=1$ to be a distribution concentrated around $x=x_{1}$, e.g., $p_{1}(x|x_{1})={\mathcal{N}}(x|x_{1},\sigma^{2}I)$, a normal distribution with $x_{1}$ mean and a sufficiently small standard deviation $\sigma>0$. Marginalizing the conditional probability paths over $q(x_{1})$ give rise to *the marginal probability path*

$$
p_{t}(x)=\int p_{t}(x|x_{1})q(x_{1})dx_{1},
$$

where in particular at time $t=1$, the marginal probability $p_{1}$ is a mixture distribution that closely approximates the data distribution $q$,

$$
p_{1}(x)=\int p_{1}(x|x_{1})q(x_{1})dx_{1}\approx q(x).
$$

Interestingly, we can also define a *marginal vector field*, by “marginalizing” over the conditional vector fields in the following sense (we assume $p_{t}(x)>0$ for all $t$ and $x$):

$$
u_{t}(x)=\int u_{t}(x|x_{1})\frac{p_{t}(x|x_{1})q(x_{1})}{p_{t}(x)}dx_{1},
$$

where $u_{t}(\cdot|x_{1}):\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ is a conditional vector field that generates $p_{t}(\cdot|x_{1})$. It may not seem apparent, but this way of aggregating the conditional vector fields actually results in the correct vector field for modeling the marginal probability path.

Our first key observation is this:

*The marginal vector field (equation 8) generates the marginal probability path (equation 6).*

This provides a surprising connection between the conditional VFs (those that generate conditional probability paths) and the marginal VF (those that generate the marginal probability path). This connection allows us to break down the unknown and intractable marginal VF into simpler conditional VFs, which are much simpler to define as these only depend on a single data sample. We formalize this in the following theorem.

###### Theorem 1.

Given vector fields $u_{t}(x|x_{1})$ that generate conditional probability paths $p_{t}(x|x_{1})$, for any distribution $q(x_{1})$, the marginal vector field $u_{t}$ in equation 8 generates the marginal probability path $p_{t}$ in equation 6, i.e., $u_{t}$ and $p_{t}$ satisfy the continuity equation (equation 26).

The full proofs for our theorems are all provided in Appendix A. Theorem 1 can also be derived from the Diffusion Mixture Representation Theorem in [^35] that provides a formula for the marginal drift and diffusion coefficients in diffusion SDEs.

### 3.2 Conditional Flow Matching

Unfortunately, due to the intractable integrals in the definitions of the marginal probability path and VF (equations 6 and 8), it is still intractable to compute $u_{t}$, and consequently, intractable to naïvely compute an unbiased estimator of the original Flow Matching objective. Instead, we propose a simpler objective, which surprisingly will result in the same optima as the original objective. Specifically, we consider the *Conditional Flow Matching* (CFM) objective,

$$
\mathcal{L}_{\scriptscriptstyle\text{CFM}}(\theta)=\mathbb{E}_{t,q(x_{1}),p_{t}(x|x_{1})}\big{\|}v_{t}(x)-u_{t}(x|x_{1})\big{\|}^{2},
$$

where $t\sim{\mathcal{U}}[0,1]$, $x_{1}\sim q(x_{1})$, and now $x\sim p_{t}(x|x_{1})$. Unlike the FM objective, the CFM objective allows us to easily sample unbiased estimates as long as we can efficiently sample from $p_{t}(x|x_{1})$ and compute $u_{t}(x|x_{1})$, both of which can be easily done as they are defined on a per-sample basis.  
Our second key observation is therefore:

*The FM (equation 5) and CFM (equation 9) objectives have identical gradients w.r.t. $\theta$.*

That is, optimizing the CFM objective is equivalent (in expectation) to optimizing the FM objective. Consequently, this allows us to train a CNF to generate the marginal probability path $p_{t}$ —which in particular, approximates the unknown data distribution $q$ at $t$ = $1$ — without ever needing access to either the marginal probability path or the marginal vector field. We simply need to design suitable *conditional* probability paths and vector fields. We formalize this property in the following theorem.

###### Theorem 2.

Assuming that $p_{t}(x)>0$ for all $x\in\mathbb{R}^{d}$ and $t\in[0,1]$, then, up to a constant independent of $\theta$, ${\mathcal{L}}_{\scriptscriptstyle\text{CFM}}$ and ${\mathcal{L}}_{\scriptscriptstyle\text{FM}}$ are equal. Hence, $\nabla_{\theta}\mathcal{L}_{\scriptscriptstyle\text{FM}}(\theta)=\nabla_{\theta}\mathcal{L}_{\scriptscriptstyle\text{CFM}}(\theta)$.

## 4 Conditional Probability Paths and Vector Fields

The Conditional Flow Matching objective works with any choice of conditional probability path and conditional vector fields. In this section, we discuss the construction of $p_{t}(x|x_{1})$ and $u_{t}(x|x_{1})$ for a general family of Gaussian conditional probability paths. Namely, we consider conditional probability paths of the form

$$
p_{t}(x|x_{1})={\mathcal{N}}(x\,|\,\mu_{t}(x_{1}),\sigma_{t}(x_{1})^{2}I),
$$

where $\mu:[0,1]\times\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ is the time-dependent mean of the Gaussian distribution, while $\sigma:[0,1]\times\mathbb{R}\rightarrow\mathbb{R}_{>0}$ describes a time-dependent scalar standard deviation (std). We set $\mu_{0}(x_{1})=0$ and $\sigma_{0}(x_{1})=1$, so that all conditional probability paths converge to the same standard Gaussian noise distribution at $t=0$, $p(x)={\mathcal{N}}(x|0,I)$. We then set $\mu_{1}(x_{1})=x_{1}$ and $\sigma_{1}(x_{1})=\sigma_{\text{min}}$, which is set sufficiently small so that $p_{1}(x|x_{1})$ is a concentrated Gaussian distribution centered at $x_{1}$.

There is an infinite number of vector fields that generate any particular probability path (e.g., by adding a divergence free component to the continuity equation, see equation 26), but the vast majority of these is due to the presence of components that leave the underlying distribution invariant—for instance, rotational components when the distribution is rotation-invariant—leading to unnecessary extra compute. We decide to use the simplest vector field corresponding to a canonical transformation for Gaussian distributions. Specifically, consider the flow (conditioned on $x_{1}$)

$$
\psi_{t}(x)=\sigma_{t}(x_{1})x+\mu_{t}(x_{1}).
$$

When $x$ is distributed as a standard Gaussian, $\psi_{t}(x)$ is the affine transformation that maps to a normally-distributed random variable with mean $\mu_{t}(x_{1})$ and std $\sigma_{t}(x_{1})$. That is to say, according to equation 4, $\psi_{t}$ pushes the noise distribution $p_{0}(x|x_{1})=p(x)$ to $p_{t}(x|x_{1})$, i.e.,

$$
\left[\psi_{t}\right]_{*}p(x)=p_{t}(x|x_{1}).
$$

This flow then provides a vector field that generates the conditional probability path:

$$
\frac{d}{dt}\psi_{t}(x)=u_{t}(\psi_{t}(x)|x_{1}).
$$

Reparameterizing $p_{t}(x|x_{1})$ in terms of just $x_{0}$ and plugging equation 13 in the CFM loss we get

$$
{\mathcal{L}}_{\scriptscriptstyle\text{CFM}}(\theta)=\mathbb{E}_{t,q(x_{1}),p(x_{0})}\Big{\|}v_{t}(\psi_{t}(x_{0}))-\frac{d}{dt}\psi_{t}(x_{0})\Big{\|}^{2}.
$$

Since $\psi_{t}$ is a simple (invertible) affine map we can use equation 13 to solve for $u_{t}$ in a closed form. Let $f^{\prime}$ denote the derivative with respect to time, i.e., $f^{\prime}=\frac{d}{dt}f$, for a time-dependent function $f$.

###### Theorem 3.

Let $p_{t}(x|x_{1})$ be a Gaussian probability path as in equation 10, and $\psi_{t}$ its corresponding flow map as in equation 11. Then, the unique vector field that defines $\psi_{t}$ has the form:

$$
u_{t}(x|x_{1})=\frac{\sigma^{\prime}_{t}(x_{1})}{\sigma_{t}(x_{1})}\left(x-\mu_{t}(x_{1})\right)+\mu^{\prime}_{t}(x_{1}).
$$

Consequently, $u_{t}(x|x_{1})$ generates the Gaussian path $p_{t}(x|x_{1})$.

### 4.1 Special instances of Gaussian conditional probability paths

Our formulation is fully general for arbitrary functions $\mu_{t}(x_{1})$ and $\sigma_{t}(x_{1})$, and we can set them to any differentiable function satisfying the desired boundary conditions. We first discuss the special cases that recover probability paths corresponding to previously-used diffusion processes. Since we directly work with probability paths, we can simply depart from reasoning about diffusion processes altogether. Therefore, in the second example below, we directly formulate a probability path based on the Wasserstein-2 optimal transport solution as an interesting instance.

#### Example I: Diffusion conditional VFs.

Diffusion models start with data points and gradually add noise until it approximates pure noise. These can be formulated as stochastic processes, which have strict requirements in order to obtain closed form representation at arbitrary times $t$, resulting in Gaussian conditional probability paths $p_{t}(x|x_{1})$ with specific choices of mean $\mu_{t}(x_{1})$ and std $\sigma_{t}(x_{1})$ [^41] [^18] [^44]. For example, the reversed (noise $\rightarrow$ data) Variance Exploding (VE) path has the form

$$
p_{t}(x)={\mathcal{N}}(x|x_{1},\sigma_{1-t}^{2}I),
$$

where $\sigma_{t}$ is an increasing function, $\sigma_{0}=0$, and $\sigma_{1}\gg 1$. Next, equation 16 provides the choices of $\mu_{t}(x_{1})=x_{1}$ and $\sigma_{t}(x_{1})=\sigma_{1-t}$. Plugging these into equation 15 of Theorem 3 we get

$$
u_{t}(x|x_{1})=-\frac{\sigma^{\prime}_{1-t}}{\sigma_{1-t}}(x-x_{1}).
$$

The reversed (noise $\rightarrow$ data) Variance Preserving (VP) diffusion path has the form

$$
p_{t}(x|x_{1})={\mathcal{N}}(x\,|\,\alpha_{1-t}x_{1},\left(1-\alpha_{1-t}^{2}\right)I),\text{where }\alpha_{t}=e^{-\frac{1}{2}T(t)},T(t)=\int_{0}^{t}\beta(s)ds,
$$

and $\beta$ is the noise scale function. Equation 18 provides the choices of $\mu_{t}(x_{1})=\alpha_{1-t}x_{1}$ and $\sigma_{t}(x_{1})=\sqrt{1-\alpha_{1-t}^{2}}$. Plugging these into equation 15 of Theorem 3 we get

$$
u_{t}(x|x_{1})=\frac{\alpha^{\prime}_{1-t}}{1-\alpha^{2}_{1-t}}\left(\alpha_{1-t}x-x_{1}\right)=-\frac{T^{\prime}(1-t)}{2}\left[\frac{e^{-T(1-t)}x-e^{-\frac{1}{2}T(1-t)}x_{1}}{1-e^{-T(1-t)}}\right].
$$

Our construction of the conditional VF $u_{t}(x|x_{1})$ does in fact coincide with the vector field previously used in the deterministic probability flow ([^44], equation 13) when restricted to these conditional diffusion processes; see details in Appendix D. Nevertheless, combining the diffusion conditional VF with the Flow Matching objective offers an attractive training alternative—which we find to be more stable and robust in our experiments—to existing score matching approaches.

Another important observation is that, as these probability paths were previously derived as solutions of diffusion processes, they do not actually reach a true noise distribution in finite time. In practice, $p_{0}(x)$ is simply approximated by a suitable Gaussian distribution for sampling and likelihood evaluation. Instead, our construction provides full control over the probability path, and we can just directly set $\mu_{t}$ and $\sigma_{t}$, as we will do next.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/plots/2d_vf_reference.png)

t = 0.0 𝑡 t=0.0

#### Example II: Optimal Transport conditional VFs.

An arguably more natural choice for conditional probability paths is to define the mean and the std to simply change linearly in time, i.e.,

$$
\mu_{t}(x)=tx_{1},\text{ and }\sigma_{t}(x)=1-(1-\sigma_{\text{min}})t.
$$

According to Theorem 3 this path is generated by the VF

$$
u_{t}(x|x_{1})=\frac{x_{1}-(1-\sigma_{\text{min}})x}{1-(1-\sigma_{\text{min}})t},
$$

which, in contrast to the diffusion conditional VF (equation 19), is defined for all $t\in[0,1]$. The conditional flow that corresponds to $u_{t}(x|x_{1})$ is

$$
\psi_{t}(x)=(1-(1-\sigma_{\text{min}})t)x+tx_{1},
$$

and in this case, the CFM loss (see equations 9, 14) takes the form:

$$
{\mathcal{L}}_{\scriptscriptstyle\text{CFM}}(\theta)=\mathbb{E}_{t,q(x_{1}),p(x_{0})}\Big{\|}v_{t}(\psi_{t}(x_{0}))-\Big{(}x_{1}-(1-\sigma_{\min})x_{0}\Big{)}\Big{\|}^{2}.
$$

Allowing the mean and std to change linearly not only leads to simple and intuitive paths, but it is actually also optimal in the following sense. The conditional flow $\psi_{t}(x)$ is in fact the Optimal Transport (OT) *displacement map* between the two Gaussians $p_{0}(x|x_{1})$ and $p_{1}(x|x_{1})$. The OT *interpolant*, which is a probability path, is defined to be (see Definition 1.1 in [^30]):

$$
p_{t}=[(1-t)\mathrm{id}+t\psi]_{\star}p_{0}
$$

where $\psi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ is the OT map pushing $p_{0}$ to $p_{1}$, $\mathrm{id}$ denotes the identity map, i.e., $\mathrm{id}(x)=x$, and $(1-t)\mathrm{id}+t\psi$ is called the OT displacement map. Example 1.7 in [^30] shows, that in our case of two Gaussians where the first is a standard one, the OT displacement map takes the form of equation 22.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/2d_traj/2d_traj_diff.png)

Figure 3: Diffusion and OT trajectories.

Intuitively, particles under the OT displacement map always move in straight line trajectories and with constant speed. Figure 3 depicts sampling paths for the diffusion and OT conditional VFs. Interestingly, we find that sampling trajectory from diffusion paths can “overshoot” the final sample, resulting in unnecessary backtracking, whilst the OT paths are guaranteed to stay straight.

Figure 2 compares the diffusion conditional score function (the regression target in a typical diffusion methods), i.e., $\nabla\log p_{t}(x|x_{1})$ with $p_{t}$ defined as in equation 18, with the OT conditional VF (equation 21). The start ($p_{0}$) and end ($p_{1}$) Gaussians are identical in both examples. An interesting observation is that the OT VF has a constant direction in time, which arguably leads to a simpler regression task. This property can also be verified directly from equation 21 as the VF can be written in the form $u_{t}(x|x_{1})=g(t)h(x|x_{1})$. Figure 8 in the Appendix shows a visualization of the Diffusion VF. Lastly, we note that although the conditional flow is optimal, this by no means imply that the marginal VF is an optimal transport solution. Nevertheless, we expect the marginal vector field to remain relatively simple.

## 5 Related Work

Continuous Normalizing Flows were introduced in [^7] as a continuous-time version of Normalizing Flows (see e.g., [^23] [^34] for an overview). Originally, CNFs are trained with the maximum likelihood objective, but this involves expensive ODE simulations for the forward and backward propagation, resulting in high time complexity due to the sequential nature of ODE simulations. Although some works demonstrated the capability of CNF generative models for image synthesis [^16], scaling up to very high dimensional images is inherently difficult. A number of works attempted to regularize the ODE to be easier to solve, e.g., using augmentation [^14], adding regularization terms [^51] [^15] [^33] [^47] [^20], or stochastically sampling the integration interval [^13]. These works merely aim to regularize the ODE but do not change the fundamental training algorithm.

In order to speed up CNF training, some works have developed simulation-free CNF training frameworks by explicitly designing the target probability path and the dynamics. For instance, [^38] consider a linear interpolation between the prior and the target density but involves integrals that were difficult to estimate in high dimensions, while [^4] consider general probability paths similar to this work but suffers from biased gradients in the stochastic minibatch regime. In contrast, the Flow Matching framework allows simulation-free training with unbiased gradients and readily scales to very high dimensions.

Another approach to simulation-free training relies on the construction of a diffusion process to indirectly define the target probability path [^41] [^18] [^43]. [^44] shows that diffusion models are trained using denoising score matching [^49], a conditional objective that provides unbiased gradients with respect to the score matching objective. Conditional Flow Matching draws inspiration from this result, but generalizes to matching vector fields directly. Due to the ease of scalability, diffusion models have received increased attention, producing a variety of improvements such as loss-rescaling [^45], adding classifier guidance along with architectural improvements [^11], and learning the noise schedule [^32] [^21]. However, [^32] and [^21] only consider a restricted setting of Gaussian conditional paths defined by simple diffusion processes with a single parameter—in particular, it does not include our conditional OT path. In an another line of works, [^9] [^50] [^35] proposed finite time diffusion constructions via diffusion bridges theory resolving the approximation error incurred by infinite time denoising constructions. While existing works make use of a connection between diffusion processes and continuous normalizing flows with the same probability path [^29] [^44] [^45], our work allows us to generalize beyond the class of probability paths modeled by simple diffusion. With our work, it is possible to completely sidestep the diffusion process construction and reason directly with probability paths, while still retaining efficient training and log-likelihood evaluations. Lastly, concurrently to our work [^26] [^1] arrived at similar conditional objectives for simulation-free training of CNFs, while [^31] derived an implicit objective when $u_{t}$ is assumed to be a gradient field.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/2d_checker/2d_checkerboard_score_dif.png)

Figure 4: ( left ) Trajectories of CNFs trained with different objectives on 2D checkerboard data. The OT path introduces the checkerboard pattern much earlier, while FM results in more stable training. ( right ) FM with OT results in more efficient sampling, solved using the midpoint scheme.

## 6 Experiments

We explore the empirical benefits of using Flow Matching on the image datasets of CIFAR-10 [^24] and ImageNet at resolutions 32, 64, and 128 [^8] [^10]. We also ablate the choice of diffusion path in Flow Matching, particularly between the standard variance preserving diffusion path and the optimal transport path. We discuss how sample generation is improved by directly parameterizing the generating vector field and using the Flow Matching objective. Lastly we show Flow Matching can also be used in the conditional generation setting. Unless otherwise specified, we evaluate likelihood and samples from the model using dopri5 [^12] at absolute and relative tolerances of 1e-5. Generated samples can be found in the Appendix, and all implementation details are in Appendix E.

<table><tbody><tr><th></th><td colspan="3">CIFAR-10</td><td></td><td colspan="3">ImageNet 32 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 32</td><td></td><td colspan="3">ImageNet 64 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 64</td></tr><tr><th>Model</th><td>NLL <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>FID <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>NFE <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td></td><td>NLL <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>FID <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>NFE <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td></td><td>NLL <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>FID <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>NFE <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><th>Ablations</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>DDPM</th><td>3.12</td><td>7.48</td><td>274</td><td></td><td>3.54</td><td>6.99</td><td>262</td><td></td><td>3.32</td><td>17.36</td><td>264</td></tr><tr><th>Score Matching</th><td>3.16</td><td>19.94</td><td>242</td><td></td><td>3.56</td><td>5.68</td><td>178</td><td></td><td>3.40</td><td>19.74</td><td>441</td></tr><tr><th>ScoreFlow</th><td>3.09</td><td>20.78</td><td>428</td><td></td><td>3.55</td><td>14.14</td><td>195</td><td></td><td>3.36</td><td>24.95</td><td>601</td></tr><tr><th>Ours</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>FM <sup>w</sup> / Diffusion</th><td>3.10</td><td>8.06</td><td>183</td><td></td><td>3.54</td><td>6.37</td><td>193</td><td></td><td>3.33</td><td>16.88</td><td>187</td></tr><tr><th>FM <sup>w</sup> / OT</th><td>2.99</td><td>6.35</td><td>142</td><td></td><td>3.53</td><td>5.02</td><td>122</td><td></td><td>3.31</td><td>14.45</td><td>138</td></tr></tbody></table>

<table><thead><tr><th></th><th colspan="2">ImageNet 128 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 128</th></tr><tr><th>Model</th><th>NLL <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th><th>FID <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>MGAN <sup><a href="#fn:19">19</a></sup></th><td>–</td><td>58.9</td></tr><tr><th>PacGAN2 <sup><a href="#fn:25">25</a></sup></th><td>–</td><td>57.5</td></tr><tr><th>Logo-GAN-AE <sup><a href="#fn:39">39</a></sup></th><td>–</td><td>50.9</td></tr><tr><th>Self-cond. GAN <sup><a href="#fn:27">27</a></sup></th><td>–</td><td>41.7</td></tr><tr><th>Uncond. BigGAN <sup><a href="#fn:27">27</a></sup></th><td>–</td><td>25.3</td></tr><tr><th>PGMGAN <sup><a href="#fn:3">3</a></sup></th><td>–</td><td>21.7</td></tr><tr><th>FM <sup>w</sup> / OT</th><td>2.90</td><td>20.9</td></tr></tbody></table>

Table 1: Likelihood (BPD), quality of generated samples (FID), and evaluation time (NFE) for the same model trained with different methods.

### 6.1 Density Modeling and Sample Quality on ImageNet

We start by comparing the same model architecture, i.e., the U-Net architecture from [^11] with minimal changes, trained on CIFAR-10, and ImageNet 32/64 with different popular diffusion-based losses: DDPM from [^18], Score Matching (SM) [^44], and Score Flow (SF) [^45]; see Appendix E.1 for exact details. Table 1 (left) summarizes our results alongside these baselines reporting negative log-likelihood (NLL) in units of bits per dimension (BPD), sample quality as measured by the Frechet Inception Distance (FID; [^17]), and averaged number of function evaluations (NFE) required for the adaptive solver to reach its a prespecified numerical tolerance, averaged over 50k samples. All models are trained using the same architecture, hyperparameter values and number of training iterations, where baselines are allowed more iterations for better convergence. Note that these are *unconditional* models. On both CIFAR-10 and ImageNet, FM-OT consistently obtains best results across all our quantitative measures compared to competing methods. We are noticing a higher that usual FID performance in CIFAR-10 compared to previous works [^18] [^44] [^45] that can possibly be explained by the fact that our used architecture was not optimized for CIFAR-10.

Secondly, Table 1 (right) compares a model trained using Flow Matching with the OT path on ImageNet at resolution 128 $\times$ 128. Our FID is state-of-the-art with the exception of IC-GAN [^5] which uses conditioning with a self-supervised ResNet50 model, and therefore is left out of this table. Figures 11, 12, 13 in the Appendix show non-curated samples from these models.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/x9.png)

Figure 5: Image quality during training, ImageNet 64 × \\times 64.

Faster training. While existing works train diffusion models with a very high number of iterations (e.g., 1.3m and 10m iterations are reported by Score Flow and VDM, respectively), we find that Flow Matching generally converges much faster. Figure 5 shows FID curves during training of Flow Matching and all baselines for ImageNet 64 $\times$ 64; FM-OT is able to lower the FID faster and to a greater extent than the alternatives. For ImageNet-128 [^11] train for 4.36m iterations with batch size 256, while FM (with 25% larger model) used 500k iterations with batch size 1.5k, i.e., 33% less image throughput; see Table 3 for exact details. Furthermore, the cost of sampling from a model can drastically change during training for score matching, whereas the sampling cost stays constant when training with Flow Matching (Figure 10 in Appendix).

### 6.2 Sampling Efficiency

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet64/imagenet64_sm_difv2.png)

Score Matching w/ Diffusion

For sampling, we first draw a random noise sample $x_{0}\sim{\mathcal{N}}(0,I)$ then compute $\phi_{1}(x_{0})$ by solving equation 1 with the trained VF, $v_{t}$, on the interval $t\in[0,1]$ using an ODE solver. While diffusion models can also be sampled through an SDE formulation, this can be highly inefficient and many methods that propose fast samplers (e.g., [^42] [^52]) directly make use of the ODE perspective (see Appendix D). In part, this is due to ODE solvers being much more efficient—yielding lower error at similar computational costs [^22] —and the multitude of available ODE solver schemes. When compared to our ablation models, we find that models trained using Flow Matching with the OT path always result in the most efficient sampler, regardless of ODE solver, as demonstrated next.

Sample paths. We first qualitatively visualize the difference in sampling paths between diffusion and OT. Figure 6 shows samples from ImageNet-64 models using identical random seeds, where we find that the OT path model starts generating images sooner than the diffusion path models, where noise dominates the image until the very last time point. We additionally depict the probability density paths in 2D generation of a checkerboard pattern, Figure 4 (left), noticing a similar trend.

Low-cost samples. We next switch to fixed-step solvers and compare low ($\leq$ 100) NFE samples computed with the ImageNet-32 models from Table 1. In Figure 7 (left), we compare the per-pixel MSE of low NFE solutions compared with 1000 NFE solutions (we use 256 random noise seeds), and notice that the FM with OT model produces the best numerical error, in terms of computational cost, requiring roughly only 60% of the NFEs to reach the same error threshold as diffusion models. Secondly, Figure 7 (right) shows how FID changes as a result of the computational cost, where we find FM with OT is able to achieve decent FID even at very low NFE values, producing better trade-off between sample quality and cost compared to ablated models. Figure 4 (right) shows low-cost sampling effects for the 2D checkerboard experiment.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/x10.png)

Figure 7: Flow Matching, especially when using OT paths, allows us to use fewer evaluations for sampling while retaining similar numerical error (left) and sample quality (right). Results are shown for models trained on ImageNet 32 × \\times 32, and numerical errors are for the midpoint scheme.

### 6.3 Conditional sampling from low-resolution images

| Model | FID $\downarrow$ | IS $\uparrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ |
| --- | --- | --- | --- | --- |
| Reference | 1.9 | 240.8 | – | – |
| Regression | 15.2 | 121.1 | 27.9 | 0.801 |
| SR3 [^40] | 5.2 | 180.1 | 26.4 | 0.762 |
| FM <sup>w</sup> / OT | 3.4 | 200.8 | 24.7 | 0.747 |

Table 2: Image super-resolution on the ImageNet validation set.

Lastly, we experimented with Flow Matching for conditional image generation. In particular, upsampling images from 64 $\times$ 64 to 256 $\times$ 256. We follow the evaluation procedure in [^40] and compute the FID of the upsampled validation images; baselines include reference (FID of original validation set), and regression. Results are in Table 2. Upsampled image samples are shown in Figures 14, 15 in the Appendix. FM-OT achieves similar PSNR and SSIM values to [^40] while considerably improving on FID and IS, which as argued by [^40] is a better indication of generation quality.

## 7 Conclusion

We introduced Flow Matching, a new simulation-free framework for training Continuous Normalizing Flow models, relying on conditional constructions to effortlessly scale to very high dimensions. Furthermore, the FM framework provides an alternative view on diffusion models, and suggests forsaking the stochastic/diffusion construction in favor of more directly specifying the probability path, allowing us to, e.g., construct paths that allow faster sampling and/or improve generation. We experimentally showed the ease of training and sampling when using the Flow Matching framework, and in the future, we expect FM to open the door to allowing a multitude of probability paths (e.g., non-isotropic Gaussians or more general kernels altogether).

## References

## Appendix A Theorem Proofs

See 1

###### Proof.

To verify this, we check that $p_{t}$ and $u_{t}$ satisfy the continuity equation (equation 26):

$$
\displaystyle\frac{d}{dt}p_{t}(x)
$$
 
$$
\displaystyle=\int\Big{(}\frac{d}{dt}p_{t}(x|x_{1})\Big{)}q(x_{1})dx_{1}=-\int\mathrm{div}\Big{(}u_{t}(x|x_{1})p_{t}(x|x_{1})\Big{)}q(x_{1})dx_{1}
$$
 
$$
\displaystyle=-\mathrm{div}\Big{(}\int u_{t}(x|x_{1})p_{t}(x|x_{1})q(x_{1})dx_{1}\Big{)}=-\mathrm{div}\Big{(}u_{t}(x)p_{t}(x)\Big{)},
$$

where in the second equality we used the fact that $u_{t}(\cdot|x_{1})$ generates $p_{t}(\cdot|x_{1})$, in the last equality we used equation 8. Furthermore, the first and third equalities are justified by assuming the integrands satisfy the regularity conditions of the Leibniz Rule (for exchanging integration and differentiation). ∎

See 2

###### Proof.

To ensure existence of all integrals and to allow the changing of integration order (by Fubini’s Theorem) in the following we assume that $q(x)$ and $p_{t}(x|x_{1})$ are decreasing to zero at a sufficient speed as $\left\|x\right\|\rightarrow\infty$, and that $u_{t},v_{t},\nabla_{\theta}v_{t}$ are bounded.

First, using the standard bilinearity of the $2$ -norm we have that

$$
\displaystyle\|v_{t}(x)-u_{t}(x)\|^{2}
$$
 
$$
\displaystyle=\left\|v_{t}(x)\right\|^{2}-2\left\langle v_{t}(x),u_{t}(x)\right\rangle+\left\|u_{t}(x)\right\|^{2}
$$
 
$$
\displaystyle\|v_{t}(x)-u_{t}(x|x_{1})\|^{2}
$$
 
$$
\displaystyle=\left\|v_{t}(x)\right\|^{2}-2\left\langle v_{t}(x),u_{t}(x|x_{1})\right\rangle+\left\|u_{t}(x|x_{1})\right\|^{2}
$$

Next, remember that $u_{t}$ is independent of $\theta$ and note that

$$
\displaystyle\mathbb{E}_{p_{t}(x)}\|v_{t}(x)\|^{2}
$$
 
$$
\displaystyle=\int\|v_{t}(x)\|^{2}p_{t}(x)dx=\int\|v_{t}(x)\|^{2}p_{t}(x|x_{1})q(x_{1})dx_{1}dx
$$
 
$$
\displaystyle=\mathbb{E}_{q(x_{1}),p_{t}(x|x_{1})}\|v_{t}(x)\|^{2},
$$

where in the second equality we use equation 6, and in the third equality we change the order of integration. Next,

$$
\displaystyle\mathbb{E}_{p_{t}(x)}\left\langle v_{t}(x),u_{t}(x)\right\rangle
$$
 
$$
\displaystyle=\int\left\langle v_{t}(x),\frac{\int u_{t}(x|x_{1})p_{t}(x|x_{1})q(x_{1})dx_{1}}{p_{t}(x)}\right\rangle p_{t}(x)dx
$$
 
$$
\displaystyle=\int\left\langle v_{t}(x),\int u_{t}(x|x_{1})p_{t}(x|x_{1})q(x_{1})dx_{1}\right\rangle dx
$$
 
$$
\displaystyle=\int\left\langle v_{t}(x),u_{t}(x|x_{1})\right\rangle p_{t}(x|x_{1})q(x_{1})dx_{1}dx
$$
 
$$
\displaystyle=\mathbb{E}_{q(x_{1}),p_{t}(x|x_{1})}\left\langle v_{t}(x),u_{t}(x|x_{1})\right\rangle,
$$

where in the last equality we change again the order of integration. ∎

See 3

###### Proof.

For notational simplicity let $w_{t}(x)=u_{t}(x|x_{1})$. Now consider equation 1:

$$
\frac{d}{dt}\psi_{t}(x)=w_{t}(\psi_{t}(x)).
$$

Since $\psi_{t}$ is invertible (as $\sigma_{t}(x_{1})>0$) we let $x=\psi^{-1}(y)$ and get

$$
\psi^{\prime}_{t}(\psi^{-1}(y))=w_{t}(y),
$$

where we used the apostrophe notation for the derivative to emphasis that $\psi^{\prime}_{t}$ is evaluated at $\psi^{-1}(y)$. Now, inverting $\psi_{t}(x)$ provides

$$
\psi_{t}^{-1}(y)=\frac{y-\mu_{t}(x_{1})}{\sigma_{t}(x_{1})}.
$$

Differentiating $\psi_{t}$ with respect to $t$ gives

$$
\psi_{t}^{\prime}(x)=\sigma_{t}^{\prime}(x_{1})x+\mu_{t}^{\prime}(x_{1}).
$$

Plugging these last two equations in equation 25 we get

$$
w_{t}(y)=\frac{\sigma^{\prime}_{t}(x_{1})}{\sigma_{t}(x_{1})}\left(y-\mu_{t}(x_{1})\right)+\mu_{t}^{\prime}(x_{1})
$$

as required. ∎

## Appendix B The continuity equation

One method of testing if a vector field $v_{t}$ generates a probability path $p_{t}$ is the continuity equation [^48]. It is a Partial Differential Equation (PDE) providing a necessary and sufficient condition to ensuring that a vector field $v_{t}$ generates $p_{t}$,

$$
\frac{d}{dt}p_{t}(x)+\mathrm{div}(p_{t}(x)v_{t}(x))=0,
$$

where the divergence operator, $\mathrm{div}$, is defined with respect to the spatial variable $x=(x^{1},\ldots,x^{d})$, i.e., $\mathrm{div}=\sum_{i=1}^{d}\frac{\partial}{\partial x^{i}}$.

## Appendix C Computing probabilities of the CNF model

We are given an arbitrary data point $x_{1}\in\mathbb{R}^{d}$ and need to compute the model probability at that point, i.e., $p_{1}(x_{1})$. Below we recap how this can be done covering the basic relevant ODEs, the scaling of the divergence computation, taking into account data transformations (e.g., centering of data), and Bits-Per-Dimension computation.

#### ODE for computing p1(x1)subscript𝑝1subscript𝑥1p\_{1}(x\_{1})

The continuity equation with equation 1 lead to the instantaneous change of variable [^7] [^4]:

$$
\displaystyle\frac{d}{dt}\log p_{t}(\phi_{t}(x))+\mathrm{div}(v_{t}(\phi_{t}(x))=0.
$$

Integrating $t\in[0,1]$ gives:

$$
\displaystyle\log p_{1}(\phi_{1}(x))-\log p_{0}(\phi_{0}(x))=-\int_{0}^{1}\mathrm{div}(v_{t}(\phi_{t}(x)))dt
$$

Therefore, the log probability can be computed together with the flow trajectory by solving the ODE:

$$
\displaystyle\frac{d}{dt}\begin{bmatrix}\phi_{t}(x)\\
f(t)\end{bmatrix}=\begin{bmatrix}v_{t}(\phi_{t}(x))\\
-\mathrm{div}(v_{t}(\phi_{t}(x)))\end{bmatrix}
$$

Given initial conditions

$$
\displaystyle\begin{bmatrix}\phi_{0}(x)\\
f(0)\end{bmatrix}=\begin{bmatrix}x_{0}\\
c\end{bmatrix}.
$$

the solution $\left[\phi_{t}(x),f(t)\right]^{T}$ is uniquely defined (up to some mild conditions on the VF $v_{t}$). Denote $x_{1}=\phi_{1}(x)$, and according to equation 27,

$$
f(1)=c+\log p_{1}(x_{1})-\log p_{0}(x_{0}).
$$

Now, we are given an arbitrary $x_{1}$ and want to compute $p_{1}(x_{1})$. For this end, we will need to solve equation 28 in reverse. That is,

$$
\displaystyle\frac{d}{ds}\begin{bmatrix}\phi_{1-s}(x)\\
f(1-s)\end{bmatrix}=\begin{bmatrix}-v_{1-s}(\phi_{1-s}(x))\\
\mathrm{div}(v_{1-s}(\phi_{1-s}(x)))\end{bmatrix}
$$

and we solve this equation for $s\in[0,1]$ with the initial conditions at $s=0$:

$$
\displaystyle\begin{bmatrix}\phi_{1}(x)\\
f(1)\end{bmatrix}=\begin{bmatrix}x_{1}\\
0\end{bmatrix}.
$$

From uniqueness of ODEs, the solution will be identical to the solution of equation 28 with initial conditions in equation 29 where $c=\log p_{0}(x_{0})-\log p_{1}(x_{1})$. This can be seen from equation 30 and setting $f(1)=0$. Therefore we get that

$$
\displaystyle f(0)
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\log p_{1}(x_{1})
$$

and consequently

$$
\displaystyle\log p_{1}(x_{1})=\log p_{0}(x_{0})-f(0).
$$

To summarize, to compute $p_{1}(x_{1})$ we first solve the ODE in equation 31 with initial conditions in equation 32, and the compute equation 33.

#### Unbiased estimator to p1(x1)subscript𝑝1subscript𝑥1p\_{1}(x\_{1})

Solving equation 31 requires computation of $\mathrm{div}$ of VFs in $\mathbb{R}^{d}$ which is costly. [^16] suggest to replace the divergence by the (unbiased) Hutchinson trace estimator,

$$
\displaystyle\frac{d}{ds}\begin{bmatrix}\phi_{1-s}(x)\\
\tilde{f}(1-s)\end{bmatrix}=\begin{bmatrix}-v_{1-s}(\phi_{1-s}(x))\\
z^{T}Dv_{1-s}(\phi_{1-s}(x))z\end{bmatrix},
$$

where $z\in\mathbb{R}^{d}$ is a sample from a random variable such that $\mathbb{E}zz^{T}=I$. Solving the ODE in equation 34 exactly (in practice, with a small controlled error) with initial conditions in equation 32 leads to

$$
\displaystyle\mathbb{E}_{z}\left[\log p_{0}(x_{0})-\tilde{f}(0)\right]
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\mathbb{E}_{z}\left[\tilde{f}(0)-\tilde{f}(1)\right]
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\mathbb{E}_{z}\left[\int_{0}^{1}z^{T}Dv_{1-s}(\phi_{1-s}(x))z\,ds\right]
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\int_{0}^{1}\mathbb{E}_{z}\left[z^{T}Dv_{1-s}(\phi_{1-s}(x))z\right]ds
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\int_{0}^{1}\mathrm{div}(v_{1-s}(\phi_{1-s}(x)))ds
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\left(f(0)-f(1)\right)
$$
 
$$
\displaystyle=\log p_{0}(x_{0})-\left(\log p_{0}(x_{0})-\log p_{1}(x_{1})\right)
$$
 
$$
\displaystyle=\log p_{1}(x_{1}),
$$

where in the third equality we switched order of integration assuming the sufficient condition of Fubini’s theorem hold, and in the previous to last equality we used equation 30. Therefore the random variable

$$
\displaystyle\log p_{0}(x_{0})-\tilde{f}(0)
$$

is an unbiased estimator for $\log p_{1}(x_{1})$. To summarize, for a scalable unbiased estimation of $p_{1}(x_{1})$ we first solve the ODE in equation 34 with initial conditions in equation 32, and then output equation 35.

#### Transformed data

Often, before training our generative model we transform the data, e.g., we scale and/or translate the data. Such a transformation is denoted by $\varphi^{-1}:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ and our generative model becomes a composition

$$
\displaystyle\psi(x)=\varphi\circ\phi(x)
$$

where $\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ is the model we train. Given a prior probability $p_{0}$ we have that the push forward of this probability under $\psi$ (equation 3 and equation 4) takes the form

$$
\displaystyle p_{1}(x)=\psi_{*}p_{0}(x)
$$
 
$$
\displaystyle=p_{0}(\phi^{-1}(\varphi^{-1}(x)))\det\left[D\phi^{-1}(\varphi^{-1}(x))\right]\det\left[D\varphi^{-1}(x)\right]
$$
 
$$
\displaystyle=\left(\phi_{*}p_{0}(\varphi^{-1}(x))\right)\det\left[D\varphi^{-1}(x)\right]
$$

and therefore

$$
\displaystyle\log p_{1}(x)=\log\phi_{*}p_{0}(\varphi^{-1}(x))+\log\det\left[D\varphi^{-1}(x)\right].
$$

For images $d=H\times W\times 3$ and we consider a transform $\phi$ that maps each pixel value from $[-1,1]$ to $[0,256]$. Therefore,

$$
\varphi(y)=2^{7}(y+1)
$$

and

$$
\varphi^{-1}(x)=2^{-7}x-1
$$

For this case we have

$$
\log p_{1}(x)=\log\phi_{*}p_{0}(\varphi^{-1}(x))-7d\log 2.
$$

#### Bits-Per-Dimension (BPD) computation

BPD is defined by

$$
\mathrm{BPD}=\mathbb{E}_{x_{1}}\left[-\frac{\log_{2}p_{1}(x_{1})}{d}\right]=\mathbb{E}_{x_{1}}\left[-\frac{\log p_{1}(x_{1})}{d\log 2}\right]
$$

Following equation 36 we get

$$
\displaystyle\mathrm{BPD}=-\frac{\log\phi_{*}p_{0}(\varphi^{-1}(x))}{d\log 2}+7
$$

and $\log\phi_{*}p_{0}(\varphi^{-1}(x))$ is approximated using the unbiased estimator in equation 35 over the transformed data $\varphi^{-1}(x_{1})$. Averaging the unbiased estimator on a large test test $x_{1}$ provides a good approximation to the test set BPD.

## Appendix D Diffusion conditional vector fields

We derive the vector field governing the Probability Flow ODE (equation 13 in [^44]) for the VE and VP diffusion paths (equation 18) and note that it coincides with the conditional vector fields we derive using Theorem 3, namely the vector fields defined in equations 16 and 19.

We start with a short primer on how to find a conditional vector field for the probability path described by the Fokker-Planck equation, then instantiate it for the VE and VP probability paths.

Since in the diffusion literature the diffusion process runs from data at time $t=0$ to noise at time $t=1$, we will need the following lemma to translate the diffusion VFs to our convention of $t=0$ corresponds to noise and $t=1$ corresponds to data:

###### Lemma 1.

Consider a flow defined by a vector field $u_{t}(x)$ generating probability density path $p_{t}(x)$. Then, the vector field $\tilde{u}_{t}(x)=-u_{1-t}(x)$ generates the path $\tilde{p}_{t}(x)=p_{1-t}(x)$ when initiated from $\tilde{p}_{0}(x)=p_{1}(x)$.

###### Proof.

We use the continuity equation (equation 26):

$$
\displaystyle\frac{d}{dt}\tilde{p}_{t}(x)=\frac{d}{dt}p_{1-t}(x)
$$
 
$$
\displaystyle=-p^{\prime}_{1-t}(x)
$$
 
$$
\displaystyle=\mathrm{div}(p_{1-t}(x)u_{1-t}(x))
$$
 
$$
\displaystyle=-\mathrm{div}(\tilde{p}_{t}(x)(-u_{1-t}(x)))
$$

and therefore $\tilde{u}_{t}(x)=-u_{1-t}(x)$ generates $\tilde{p}_{t}(x)$. ∎

#### Conditional VFs for Fokker-Planck probability paths

Consider a Stochastic Differential Equation (SDE) of the standard form

$$
dy=f_{t}dt+g_{t}dw
$$

with time parameter $t$, drift $f_{t}$, diffusion coefficient $g_{t}$, and $dw$ is the Wiener process. The solution $y_{t}$ to the SDE is a stochastic process, i.e., a continuous time-dependent random variable, the probability density of which, $p_{t}(y_{t})$, is characterized by the Fokker-Planck equation:

$$
\frac{dp_{t}}{dt}=-\mathrm{div}(f_{t}p_{t})+\frac{g_{t}^{2}}{2}\Delta p_{t}
$$

where $\Delta$ represents the Laplace operator (in $y$), namely $\mathrm{div}\nabla$, where $\nabla$ is the gradient operator (also in $y$). Rewriting this equation in the form of the continuity equation can be done as follows [^28]:

$$
\displaystyle\frac{dp_{t}}{dt}=-\mathrm{div}\Big{(}f_{t}p_{t}-\frac{g^{2}}{2}\frac{\nabla p_{t}}{p_{t}}p_{t}\Big{)}=-\mathrm{div}\Big{(}\big{(}f_{t}-\frac{g_{t}^{2}}{2}\nabla\log p_{t}\big{)}p_{t}\Big{)}=-\mathrm{div}\Big{(}w_{t}p_{t}\Big{)}
$$

where the vector field

$$
w_{t}=f_{t}-\frac{g_{t}^{2}}{2}\nabla\log p_{t}
$$

satisfies the continuity equation with the probability path $p_{t}$, and therefore generates $p_{t}$.

#### Variance Exploding (VE) path

The SDE for the VE path is

$$
dy=\sqrt{\frac{d}{dt}\sigma_{t}^{2}}dw,
$$

where $\sigma_{0}=0$ and increasing to infinity as $t\rightarrow 1$. The SDE is moving from data, $y_{0}$, at $t=0$ to noise, $y_{1}$, at $t=1$ with the probability path

$$
p_{t}(y|y_{0})={\mathcal{N}}(y|y_{0},\sigma^{2}_{t}I).
$$

The conditional VF according to equation 40 is:

$$
w_{t}(y|y_{0})=\frac{\sigma_{t}^{\prime}}{\sigma_{t}}(y-y_{0})
$$

Using Lemma 1 we get that the probability path

$$
\tilde{p}_{t}(y|y_{0})={\mathcal{N}}(y|y_{0},\sigma_{1-t}^{2}I)
$$

is generated by

$$
\displaystyle\tilde{w}_{t}(y|y_{0})=-\frac{\sigma_{1-t}^{\prime}}{\sigma_{1-t}}(y-y_{0}),
$$

which coincides with equation 17.

#### Variance Preserving (VP) path

The SDE for the VP path is

$$
dy=-\frac{T^{\prime}(t)}{2}y+\sqrt{T^{\prime}(t)}dw,
$$

where $T(t)=\int_{0}^{t}\beta(s)ds$, $t\in[0,1]$. The SDE coefficients are therefore

$$
\displaystyle f_{s}(y)=-\frac{T^{\prime}(s)}{2}y,\quad g_{s}=\sqrt{T^{\prime}(s)}
$$

and

$$
\displaystyle p_{t}(y|y_{0})={\mathcal{N}}(y|e^{-\frac{1}{2}T(t)}y_{0},(1-e^{-T(t)})I).
$$

Plugging these choices in equation 40 we get the conditional VF

$$
w_{t}(y|y_{0})=\frac{T^{\prime}(t)}{2}\left(\frac{y-e^{-\frac{1}{2}T(t)}y_{0}}{1-e^{-T(t)}}-y\right)
$$

Using Lemma 1 to reverse the time we get the conditional VF for the reverse probability path:

$$
\displaystyle\tilde{w}_{t}(y|y_{0})
$$
 
$$
\displaystyle=-\frac{T^{\prime}(1-t)}{2}\left(\frac{y-e^{-\frac{1}{2}T(1-t)}y_{0}}{1-e^{-T(1-t)}}-y\right)
$$
 
$$
\displaystyle=-\frac{T^{\prime}(1-t)}{2}\left[\frac{e^{-T(1-t)}y-e^{-\frac{1}{2}T(1-t)}y_{0}}{1-e^{-T(1-t)}}\right],
$$

which coincides with equation 19.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/plots/2d_vf_reference.png)

t = 0.0 𝑡 t=0.0

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/2d_checker/2d_checkerboard_scoreflow_dif.png)

Figure 9: Trajectories of CNFs trained with ScoreFlow 45 and DDPM 18 losses on 2D checkerboard data, using the same learning rate and other hyperparameters as Figure 4.

## Appendix E Implementation details

For the 2D example we used an MLP with 5-layers of 512 neurons each, while for images we used the UNet architecture from [^11]. For images, we center crop images and resize to the appropriate dimension, whereas for the 32 $\times$ 32 and 64 $\times$ 64 resolutions we use the same pre-processing as [^8]. The three methods (FM-OT, FM-Diffusion, and SM-Diffusion) are always trained on the same architecture, same hyper-parameters, and for the same number of epochs.

### E.1 Diffusion baselines

#### Losses.

We consider three options as diffusion baselines that correspond to the most popular diffusion loss parametrizations [^43] [^45] [^18] [^21]. We will assume general Gaussian path form of equation 10, i.e.,

$$
p_{t}(x|x_{1})={\mathcal{N}}(x|\mu_{t}(x_{1}),\sigma_{t}^{2}(x_{1})I).
$$

Score Matching loss is

$$
\displaystyle{\mathcal{L}}_{\scriptscriptstyle\text{SM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t,q(x_{1}),p_{t}(x|x_{1})}\lambda(t)\left\|s_{t}(x)-\nabla\log p_{t}(x|x_{1})\right\|^{2}
$$
 
$$
\displaystyle=\mathbb{E}_{t,q(x_{1}),p_{t}(x|x_{1})}\lambda(t)\left\|s_{t}(x)-\frac{x-\mu_{t}(x_{1})}{\sigma_{t}^{2}(x_{1})}\right\|^{2}.
$$

Taking $\lambda(t)=\sigma_{t}^{2}(x_{1})$ corresponds to the original Score Matching (SM) loss from [^43], while considering $\lambda(t)=\beta(1-t)$ ($\beta$ is defined below) corresponds to the Score Flow (SF) loss motivated by an NLL upper bound [^45]; $s_{t}$ is the learnable score function. DDPM (Noise Matching) loss from [^18] (equation 14) is

$$
\displaystyle{\mathcal{L}}_{\scriptscriptstyle\text{NM}}(\theta)
$$
 
$$
\displaystyle=\mathbb{E}_{t,q(x_{1}),p_{t}(x|x_{1})}\left\|\epsilon_{t}(x)-\frac{x-\mu_{t}(x_{1})}{\sigma_{t}(x_{1})}\right\|^{2}
$$
 
$$
\displaystyle=\mathbb{E}_{t,q(x_{1}),p_{0}(x_{0})}\Big{\|}{\epsilon_{t}(\sigma_{t}(x_{1})x_{0}+\mu_{t}(x_{1}))-x_{0}}\Big{\|}^{2}
$$

where $p_{0}(x)={\mathcal{N}}(x|0,I)$ is the standard Gaussian, and $\epsilon_{t}$ is the learnable noise function.

#### Diffusion path.

For the diffusion path we use the standard VP diffusion (equation 19), namely,

$$
\mu_{t}(x_{1})=\alpha_{1-t}x_{1},\quad\sigma_{t}(x_{1})=\sqrt{1-\alpha_{1-t}^{2}},\quad\text{where }\alpha_{t}=e^{-\frac{1}{2}T(t)},\quad T(t)=\int_{0}^{t}\beta(s)ds,
$$

with, as suggested in [^44], $\beta(s)=\beta_{\min}+s(\beta_{\max}-\beta_{\min})$ and consequently

$$
T(s)=\int_{0}^{s}\beta(r)dr=s\beta_{\min}+\frac{1}{2}s^{2}(\beta_{\max}-\beta_{\min}),
$$

where $\beta_{\min}=0.1$, $\beta_{\max}=20$ and time is sampled in $[0,1-{\epsilon}]$, ${\epsilon}=10^{-5}$ for training and likelihood and ${\epsilon}=10^{-5}$ for sampling.

#### Sampling.

Score matching samples are produced by solving the ODE (equation 1) with the vector field

$$
u_{t}(x)=-\frac{T^{\prime}(1-t)}{2}\left[s_{t}(x)-x\right].
$$

DDPM samples are computed with equation 46 after setting $s_{t}(x)=\epsilon_{t}(x)/\sigma_{t}$, where $\sigma_{t}=\sqrt{1-\alpha_{1-t}^{2}}$.

### E.2 Training & evaluation details

|  | CIFAR10 | ImageNet-32 | ImageNet-64 | ImageNet-128 |
| --- | --- | --- | --- | --- |
| Channels | 256 | 256 | 192 | 256 |
| Depth | 2 | 3 | 3 | 3 |
| Channels multiple | 1,2,2,2 | 1,2,2,2 | 1,2,3,4 | 1,1,2,3,4 |
| Heads | 4 | 4 | 4 | 4 |
| Heads Channels | 64 | 64 | 64 | 64 |
| Attention resolution | 16 | 16,8 | 32,16,8 | 32,16,8 |
| Dropout | 0.0 | 0.0 | 0.0 | 0.0 |
| Effective Batch size | 256 | 1024 | 2048 | 1536 |
| GPUs | 2 | 4 | 16 | 32 |
| Epochs | 1000 | 200 | 250 | 571 |
| Iterations | 391k | 250k | 157k | 500k |
| Learning Rate | 5e-4 | 1e-4 | 1e-4 | 1e-4 |
| Learning Rate Scheduler | Polynomial Decay | Polynomial Decay | Constant | Polynomial Decay |
| Warmup Steps | 45k | 20k | \- | 20k |

Table 3: Hyper-parameters used for training each model

We report the hyper-parameters used in Table 3. We use full 32 bit-precision for training CIFAR10 and ImageNet-32 and 16-bit mixed precision for training ImageNet-64/128/256. All models are trained using the Adam optimizer with the following parameters: $\beta_{1}=0.9$, $\beta_{2}=0.999$, weight decay = 0.0, and $\epsilon=1e{-8}$. All methods we trained (i.e., FM-OT, FM-Diffusion, SM-Diffusion) using identical architectures, with the same parameters for the the same number of Epochs (see Table 3 for details). We use either a constant learning rate schedule or a polynomial decay schedule (see Table 3). The polynomial decay learning rate schedule includes a warm-up phase for a specified number of training steps. In the warm-up phase, the learning rate is linearly increased from $1e{-8}$ to the peak learning rate (specified in Table 3). Once the peak learning rate is achieved, it linearly decays the learning rate down to $1e{-8}$ until the final training step.

When reporting negative log-likelihood, we dequantize using the standard uniform dequantization. We report an importance-weighted estimate using

$$
\log\frac{1}{K}\sum_{k=1}^{K}p_{t}(x+u_{k}),\text{ where }u_{k}\sim{\mathcal{U}}(0,1),
$$

with $x$ is in {0, …, 255} and solved at $t=1$ with an adaptive step size solver dopri5 with atol=rtol=1e-5 using the torchdiffeq [^6] library. Estimated values for different values of $K$ are in Table 4.

When computing FID/Inception scores for CIFAR10, ImageNet-32/64 we use the TensorFlow GAN library <sup>2</sup>. To remain comparable to [^11] for ImageNet-128 we use the evaluation script they include in their publicly available code repository <sup>3</sup>.

## Appendix F Additional tables and figures

<table><tbody><tr><th></th><td colspan="3">CIFAR-10</td><td></td><td colspan="3">ImageNet 32 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 32</td><td></td><td colspan="3">ImageNet 64 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 64</td></tr><tr><th>Model</th><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =1</td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =20</td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =50</td><td></td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =1</td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =5</td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =15</td><td></td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =1</td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =5</td><td><math><semantics><mi>K</mi> <ci>𝐾</ci> <annotation>K</annotation></semantics></math> =10</td></tr><tr><th>Ablation</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>DDPM</th><td>3.24</td><td>3.14</td><td>3.12</td><td></td><td>3.62</td><td>3.57</td><td>3.54</td><td></td><td>3.36</td><td>3.33</td><td>3.32</td></tr><tr><th>Score Matching</th><td>3.28</td><td>3.18</td><td>3.16</td><td></td><td>3.65</td><td>3.59</td><td>3.57</td><td></td><td>3.43</td><td>3.41</td><td>3.40</td></tr><tr><th>ScoreFlow</th><td>3.21</td><td>3.11</td><td>3.09</td><td></td><td>3.63</td><td>3.57</td><td>3.55</td><td></td><td>3.39</td><td>3.37</td><td>3.36</td></tr><tr><th>Ours</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>FM <sup>w</sup> / Diffusion</th><td>3.23</td><td>3.13</td><td>3.10</td><td></td><td>3.64</td><td>3.58</td><td>3.56</td><td></td><td>3.37</td><td>3.34</td><td>3.33</td></tr><tr><th>FM <sup>w</sup> / OT</th><td>3.11</td><td>3.01</td><td>2.99</td><td></td><td>3.62</td><td>3.56</td><td>3.53</td><td></td><td>3.35</td><td>3.33</td><td>3.31</td></tr></tbody></table>

Table 4: Negative log-likelihood (in bits per dimension) on the test set with different values of $K$ using uniform dequantization.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/x18.png)

Figure 10: Function evaluations for sampling during training, for models trained on CIFAR-10 using dopri5 solver with tolerance 1 e − 5 superscript 𝑒 1e^{-5}.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet32/imagenet32_samples.png)

Figure 11: Non-curated unconditional ImageNet-32 generated images of a CNF trained with FM-OT.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet64/imagenet64_samples.png)

Figure 12: Non-curated unconditional ImageNet-64 generated images of a CNF trained with FM-OT.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet128/imagenet128_samples_.png)

Figure 13: Non-curated unconditional ImageNet-128 generated images of a CNF trained with FM-OT.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/upsampled/upsample_8.png)

Figure 14: Conditional generation 64 × \\times 64 → \\rightarrow 256 256. Flow Matching OT upsampled images from validation set.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/upsampled/upsampled2/upsample_47.png)

Figure 15: Conditional generation 64 × \\times 64 → \\rightarrow 256 256. Flow Matching OT upsampled images from validation set.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet128/solver_samples/imagenet128_fm_ot_17_05.png)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2210.02747/assets/figures/imagenet256/solver_samples/imagenet256_fm_ot_11_05.png)

Refer to caption

[^1]: Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. *arXiv preprint arXiv:2209.15571*, 2022.

[^2]: Dario Amodei, Danny Hernandez, Girish SastryJack, Jack Clark, Greg Brockman, and Ilya Sutskever. Ai and compute. [https://openai.com/blog/ai-and-compute/](https://openai.com/blog/ai-and-compute/), 2018.

[^3]: Mohammadreza Armandpour, Ali Sadeghian, Chunyuan Li, and Mingyuan Zhou. Partition-guided gans. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 5099–5109, 2021.

[^4]: Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Aditya Grover, Maximilian Nickel, Ricky T. Q. Chen, and Yaron Lipman. Matching normalizing flows and probability paths on manifolds. *arXiv preprint arXiv:2207.04711*, 2022.

[^5]: Arantxa Casanova, Marlene Careil, Jakob Verbeek, Michal Drozdzal, and Adriana Romero Soriano. Instance-conditioned gan. *Advances in Neural Information Processing Systems*, 34:27517–27529, 2021.

[^6]: Ricky T. Q. Chen. torchdiffeq, 2018. URL [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq).

[^7]: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. *Advances in neural information processing systems*, 31, 2018.

[^8]: Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an alternative to the cifar datasets. *arXiv preprint arXiv:1707.08819*, 2017.

[^9]: Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. (arXiv:2106.01357), Dec 2021. doi: 10.48550/arXiv.2106.01357. URL [http://arxiv.org/abs/2106.01357](http://arxiv.org/abs/2106.01357). arXiv:2106.01357 \[cs, math, stat\].

[^10]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

[^11]: Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, 2021. URL [https://openreview.net/forum?id=AAWuCvzaVt](https://openreview.net/forum?id=AAWuCvzaVt).

[^12]: John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. *Journal of computational and applied mathematics*, 6(1):19–26, 1980.

[^13]: Shian Du, Yihong Luo, Wei Chen, Jian Xu, and Delu Zeng. To-flow: Efficient continuous normalizing flows with temporal optimization adjoint with moving speed, 2022. URL [https://arxiv.org/abs/2203.10335](https://arxiv.org/abs/2203.10335).

[^14]: Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. Augmented neural odes. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d´ Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper/2019/file/21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf](https://proceedings.neurips.cc/paper/2019/file/21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf).

[^15]: Chris Finlay, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. How to train your neural ode: the world of jacobian and kinetic regularization. In *ICML*, pp. 3154–3164, 2020. URL [http://proceedings.mlr.press/v119/finlay20a.html](http://proceedings.mlr.press/v119/finlay20a.html).

[^16]: Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models, 2018. URL [https://arxiv.org/abs/1810.01367](https://arxiv.org/abs/1810.01367).

[^17]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017.

[^18]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33:6840–6851, 2020.

[^19]: Quan Hoang, Tu Dinh Nguyen, Trung Le, and Dinh Phung. Mgan: Training generative adversarial nets with multiple generators. In *International conference on learning representations*, 2018.

[^20]: Jacob Kelly, Jesse Bettencourt, Matthew J Johnson, and David K Duvenaud. Learning differential equations that are easy to solve. *Advances in Neural Information Processing Systems*, 33:4370–4380, 2020.

[^21]: Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, 2021. URL [https://openreview.net/forum?id=2LdBqxc1Yv](https://openreview.net/forum?id=2LdBqxc1Yv).

[^22]: Peter Eris Kloeden, Eckhard Platen, and Henri Schurz. *Numerical solution of SDE through computer experiments*. Springer Science & Business Media, 2012.

[^23]: Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. *IEEE transactions on pattern analysis and machine intelligence*, 43(11):3964–3979, 2020.

[^24]: Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[^25]: Zinan Lin, Ashish Khetan, Giulia Fanti, and Sewoong Oh. Pacgan: The power of two samples in generative adversarial networks. *Advances in neural information processing systems*, 31, 2018.

[^26]: Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. *arXiv preprint arXiv:2209.03003*, 2022.

[^27]: Mario Lučić, Michael Tschannen, Marvin Ritter, Xiaohua Zhai, Olivier Bachem, and Sylvain Gelly. High-fidelity image generation with fewer labels. In *International conference on machine learning*, pp. 4183–4192. PMLR, 2019.

[^28]: Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–planck equations through gradient–log–density estimation. *Entropy*, 22(8):802, jul 2020a. doi: 10.3390/e22080802. URL [https://doi.org/10.3390%2Fe22080802](https://doi.org/10.3390%2Fe22080802).

[^29]: Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–planck equations through gradient–log–density estimation. *Entropy*, 22(8):802, 2020b.

[^30]: Robert J McCann. A convexity principle for interacting gases. *Advances in mathematics*, 128(1):153–179, 1997.

[^31]: Kirill Neklyudov, Daniel Severo, and Alireza Makhzani. Action matching: A variational method for learning stochastic dynamics from samples, 2023. URL [https://openreview.net/forum?id=T6HPzkhaKeS](https://openreview.net/forum?id=T6HPzkhaKeS).

[^32]: Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In *International Conference on Machine Learning*, pp. 8162–8171. PMLR, 2021.

[^33]: Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. Ot-flow: Fast and accurate continuous normalizing flows via optimal transport. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(10):9223–9232, May 2021. URL [https://ojs.aaai.org/index.php/AAAI/article/view/17113](https://ojs.aaai.org/index.php/AAAI/article/view/17113).

[^34]: George Papamakarios, Eric T Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. *J. Mach. Learn. Res.*, 22(57):1–64, 2021.

[^35]: Stefano Peluchetti. Non-denoising forward-time diffusions. 2021.

[^36]: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 2022.

[^37]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 10684–10695, 2022.

[^38]: Noam Rozen, Aditya Grover, Maximilian Nickel, and Yaron Lipman. Moser flow: Divergence-based generative modeling on manifolds. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, 2021. URL [https://openreview.net/forum?id=qGvMv3undNJ](https://openreview.net/forum?id=qGvMv3undNJ).

[^39]: Alexander Sage, Eirikur Agustsson, Radu Timofte, and Luc Van Gool. Logo synthesis and manipulation with clustered generative adversarial networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 5879–5888, 2018.

[^40]: Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.

[^41]: Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International Conference on Machine Learning*, pp. 2256–2265. PMLR, 2015.

[^42]: Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *arXiv preprint arXiv:2010.02502*, 2020a.

[^43]: Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d´ Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf).

[^44]: Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*, 2020b.

[^45]: Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. In *Thirty-Fifth Conference on Neural Information Processing Systems*, 2021.

[^46]: Neil C Thompson, Kristjan Greenewald, Keeheon Lee, and Gabriel F Manso. The computational limits of deep learning. *arXiv preprint arXiv:2007.05558*, 2020.

[^47]: Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajectorynet: A dynamic optimal transport network for modeling cellular dynamics. In *International conference on machine learning*, pp. 9526–9536. PMLR, 2020.

[^48]: Cédric Villani. *Optimal transport: old and new*, volume 338. Springer, 2009.

[^49]: Pascal Vincent. A connection between score matching and denoising autoencoders. *Neural computation*, 23(7):1661–1674, 2011.

[^50]: Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via schrödinger bridge. (arXiv:2106.10410), Jul 2021. doi: 10.48550/arXiv.2106.10410. URL [http://arxiv.org/abs/2106.10410](http://arxiv.org/abs/2106.10410). arXiv:2106.10410 \[cs\].

[^51]: Liu Yang and George E. Karniadakis. Potential flow generator with $l\_2$ optimal transport regularity for generative models. *CoRR*, abs/1908.11462, 2019. URL [http://arxiv.org/abs/1908.11462](http://arxiv.org/abs/1908.11462).

[^52]: Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. *arXiv preprint arXiv:2204.13902*, 2022.