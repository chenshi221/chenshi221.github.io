---
title: "InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction"
source: "https://arxiv.org/html/2411.09852v4"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Zhichen Zeng <sup>∗1</sup>   Xiaolong Liu <sup>∗3</sup>   Mengyue Hang <sup>∗2</sup>   Xiaoyi Liu <sup>∗2</sup>   Qinghai Zhou <sup>2</sup>   Chaofei Yang <sup>2</sup>   Yiqun Liu <sup>2</sup>   Yichen Ruan <sup>2</sup>  Laming Chen <sup>2</sup>   Yuxin Chen <sup>2</sup>   Yujia Hao <sup>2</sup>   Jiaqi Xu <sup>2</sup>   Jade Nie <sup>2</sup>   Xi Liu <sup>2</sup>   Buyun Zhang <sup>2</sup>   Wei Wen <sup>2</sup>   Siyang Yuan <sup>2</sup>   Hang Yin <sup>2</sup>   Xin Zhang <sup>2</sup>   Kai Wang <sup>2</sup>   Wen-Yen Chen <sup>2</sup>   Yiping Han <sup>2</sup>   Huayu Li <sup>2</sup>   Chunzhi Yang <sup>2</sup>   Bo Long <sup>2</sup>   Philip S. Yu <sup>3</sup>   Hanghang Tong [^1]   Jiyan Yang <sup>2</sup>  
<sup>1</sup> University of Illinois Urbana-Champaign, <sup>2</sup> Meta AI, <sup>3</sup> University of Illinois Chicago  
{zhichenz, htong}@illinois.edu, {xliu262, psyu}@uic.edu, {hangm, xiaoyliu, chocjy}@meta.com

###### Abstract.

A clear and well-documented document is presented as an article formatted for publication by ACM in a conference proceedings or journal publication. Based on the “acmart” document class, this article presents and explains many of the common variations, as well as many of the formatting elements an author may use in the preparation of the documentation of their work.

###### Abstract.

Click-through rate (CTR) prediction, which predicts the probability of a user clicking an ad, is a fundamental task in recommender systems. The emergence of heterogeneous information, such as user profile and behavior sequences, depicts user interests from different aspects. A mutually beneficial integration of heterogeneous information is the cornerstone towards the success of CTR prediction. However, most of the existing methods suffer from two fundamental limitations, including (1) insufficient inter-mode interaction due to the unidirectional information flow between modes, and (2) aggressive information aggregation caused by early summarization, resulting in excessive information loss. To address these limitations, we propose a novel module named InterFormer to learn heterogeneous information interaction in an interleaving style. To achieve better interaction learning, InterFormer enables bidirectional information flow for mutually beneficial learning across different modes. To avoid aggressive information aggregation, we retain complete information in each data mode and use a separate Cross Arch for effective information selection and summarization. InterFormer has been deployed across multiple platforms at Meta Ads, achieving 0.15% performance gain and 24% QPS gain compared to prior state-of-the-art models and yielding sizable online impact.

Recommendation, CTR Prediction, Heterogeneous Information

## 1\. Introduction

Click-through rate (CTR) prediction, which predicts the probability of a user clicking an item, is the fundamental task for various applications such as online advertising and recommender systems [^80] [^79]. The quality of CTR prediction significantly influences company revenue and user experience, drawing extensive attention from academia and industry [^86] [^85] [^28] [^23] [^70] [^29] [^30] [^31]. For example, in ad bidding, CTR prediction helps advertisers optimize the bids and target the most receptive audiences. In content recommendation, it enables platforms to suggest relevant content to users.

To achieve better CTR prediction, it is crucial to capture the user interests in the evolving environment [^86] [^33] [^50]. The abundance of heterogeneous information presents both opportunities and challenges. On the one hand, heterogeneous information depicts user interests from different aspects, providing diverse context [^82]. For instance, the non-sequence features, e.g., user profile and context features, offer a static view on general user interests, while behavior sequences provide substantial information for modeling dynamic user interests [^50]. On the other hand, the heterogeneous nature of the data requires different modeling approaches and careful integration across different information modes [^82]. For example, while modeling interactions among non-sequential information is critical to personalized recommendation [^36] [^22] [^48], capturing sequential dependencies is the major focus for user behavior modeling [^40] [^6].

Most of the existing CTR prediction models fall into two categories, including non-sequential and sequential models. Non-sequential models focus on learning informative embeddings through feature interaction via inner-product [^22] [^41], MLP [^46] [^48] and deep structured semantic model [^14] [^9], but ignore the sequential information in user behaviors. Sequential models, in contrast, employ additional modules like CNN [^42], RNN [^40] [^86] and Attention modules [^33] [^85] [^78], to capture the sequential dependencies in user behaviors. Promising as it might be, existing sequential methods mostly employ a unidirectional information flow, where the non-sequential information is used to guide sequence learning, while the reverse information flow from sequence to non-sequence is largely ignored, resulting in insufficient inter-mode interaction. For example, the non-sequential information captures long-term interests, while the sequential information reveals momentary interests, such as a sudden focus on a specific category of products, which can enhance the non-sequential context with immediate preference. Besides, due to the computational challenges of performing interaction learning among numerous non-sequence features and lengthy sequences, aggressive feature aggregation, e.g., sequence summation [^86], pooling [^54], and concatenation [^85], is often performed in the early stages, inevitably leading to excessive information loss.

In light of the above limitations, we propose a novel heterogeneous interaction learning module named InterFormer, whose ideas are two-fold. To avoid insufficient inter-mode interaction, we enable bidirectional information flows between different modes, such that non-sequence and sequence learning are performed in an interleaving style. Specifically, to learn context-aware sequence embeddings, non-sequence summarization guides sequence learning via Personalized FeedForward Network (PFFN) and Multihead Attention (MHA) [^44]. To learn behavior-aware non-sequence embeddings, sequence summarization instructs non-sequence learning via an interaction module. To mitigate aggressive information aggregation, we adopt MHA for effective information selection such that the one-to-one mappings between input and output tokens are retained. Note that our framework is compatible with various interaction learning models like DCNv2 [^48], DHEN [^80], etc.

The main contributions of this paper are summarized as follows:

- Challenges. We identify two key bottlenecks of heterogeneous interaction learning, namely insufficient inter-mode interaction and aggressive information aggregation.
- Model Design. A novel heterogeneous interaction learning framework named InterFormer is proposed for effective feature interaction and selective information aggregation. To our best knowledge, we are the first to address the mutual benefits in heterogeneous interaction learning.
- Experiments and Analysis. Extensive experiments suggest that InterFormer achieves up to 0.14% AUC improvement on benchmark datasets and 0.15% Normalized Entropy (NE) gain on industrial dataset. In addition, the internal deployment at Meta suggests that InterFormer exhibits promising scaling results, achieving 0.15% performance gain and 24% QPS gain compared to SOTA CTR models, and yielding sizable online impact.

The rest of the paper is organized as follows. Section 2 briefly reviews the recent works on interaction learning. Section 3 summarizes the preliminaries, and section 4 introduces our proposed InterFormer. Extensive experiments and analyses are carried out in Section 5. We conclude our paper in Section 6.

## 2\. Related Works

In the era of big data and AI [^64] [^65] [^59] [^63] [^66] [^61] [^62] [^58] [^60] [^56] [^72] [^71] [^75] [^74] [^76] [^77] [^73] [^3] [^57] [^25] [^26] [^24], recommender systems have drawn significant attention, exerting profound influence on company revenue and user experience [^49] [^45] [^46] [^48] [^20] [^19] [^18] [^15] [^16] [^68] [^47]. In this section, we review the related works on CTR prediction, including non-sequential and sequential methods.

### 2.1. Non-Sequential Methods

The vast majority of non-sequential models is built upon the idea of Factorization Machine (FM) [^36] [^22] [^41], which models the user-item interaction by linearly combining their low-dimensional embeddings [^81]. [^36] is the very first FM model to capture pairwise interactions. To model high-order interactions, different methods combine FMs with deep neural networks, where FMs learn low-order interaction via pairwise operation and neural networks learn high-order interactions via deep architectures. For example, MLP [^22] [^67] [^48] [^41] [^84] captures high-order interactions via dense connections between features, and Attention mechanism [^38] [^53] [^55] learns more complex embeddings through linear combination. These deep learning-based approaches enable end-to-end training without hand-craft feature designs, and are capable of handling heterogeneous information like text, image and video [^81]. Besides, recent works address the scaling law in recommendation, where DHEN [^80] ensembles multiple interaction modules and Wukong [^79] stacks FMs to learn a hierarchy of interactions. Promising as they are, non-sequential models fail to capture the sequential dependencies in user behaviors, resulting in suboptimal solutions.

### 2.2. Sequential Methods

With the recent emergence of sequential information, e.g., user interaction history, in recommender systems, extensive efforts have been made to capture the evolving user interest. A key challenge behind sequential methods is to combine the sequential information with non-sequential information in a mutually beneficial manner. Markov models [^37] [^11] [^69] consider sequential data as a stochastic process over discrete random variables, but the oversimplified Markovian property limits the model capability in capturing long-term sequential dependencies [^34]. To model the long-term dependencies, RNN and Attention mechanism are employed as the backbone module for many sequential methods. For example, various attention-based networks are designed [^85] [^33], and Transformer [^8] [^44] [^27] is adopted for sequential modeling [^40] [^6]. Besides, to model multi-faceted user interests, [^54] [^10] propose to capture multiple user interests from multi-behavior sequences. More recently, TransAct [^52] adopt a hybrid ranking model to combine real-time user actions for immediate preference and batch user representations for long-term interests. LiRank [^4] improves CTR prediction at LinkedIn by ensembling multiple interaction modules, accelerated by quantization and vocabulary compression. CARL [^7] achieves fast inference on large-scale recommendation in Kuaishou by utilizing cached results, monitored by a reinforcement learning-based framework. While most of the existing sequential methods leverage non-sequential information for personalized sequence modeling, sequential information is rarely explored for non-sequence learning. We believe such unidirectional design limits the expressiveness of the learned embeddings, and a bidirectional information flow between different modes is the key towards better heterogeneous interaction learning.

## 3\. Preliminaries

Table 1 summarizes the main symbols and notations used throughout the paper. We use bold uppercase letters for matrices (e.g., $\mathbf{X}$), bold lowercase letters for vectors (e.g., $\mathbf{x}$), and lowercase letters for scalars (e.g., $n$). The element at the $i$ -th row and $j$ -th column of a matrix $\mathbf{X}$ is denoted as $\mathbf{X}(i,j)$. The transpose of $\mathbf{X}$ is denoted by the superscript ${\scriptscriptstyle\mathsf{T}}$ (e.g., $\mathbf{X}^{\scriptscriptstyle\mathsf{T}}$). We use superscript $u$ to denote users, and subscripts $i$ and $t$ to denote item and timestamp, respectively (e.g., $y^{u}_{i_{t}}$). We use $\mathbf{x}_{j}^{(l)}$ to denote the $j$ -th non-sequence feature of the $l$ -th layer, and $\mathbf{s}_{t}^{(l)}$ to denote the sequence feature of the $l$ -th layer at timestamp $t$. We consider the scenario with $m$ dense features, $n$ sparse features, and $k$ sequence features of length $T$. We use $d$ to denote the embedding dimension of the CTR model.

Table 1. Symbols and notations.

| Symbol | Definition |
| --- | --- |
| $\mathcal{U},\mathcal{I}$ | user set and item set |
| $\mathbf{x}_{j}^{(l)}$ | $j$ -th non-sequence feature of layer $l$ |
| $\mathbf{s}_{t}^{(l)}$ | sequence feature of layer $l$ at time step $t$ |
| $m,n,k$ | number of dense, sparse and sequence features |
| $T,d$ | sequence length and embedding dimension |
| $\odot$ | Hadamard product |
| $\langle\cdot,\cdot\rangle$ | inner product |
| $[\cdot\\|\cdot]$ | horizontal concatenation of vectors |

### 3.1. Click-Through Rate (CTR) Prediction

CTR prediction estimates the probability of a user clicking on an item given heterogeneous information such as static context and behavior sequences. Formally, given a user set $\mathcal{U}$ and an item set $\mathcal{I}$, the interaction sequence of a user $u\in\mathcal{U}$ is defined as $S^{u}=[i^{u}_{1},i^{u}_{2},\dots,i^{u}_{T}]$, where $i^{u}_{t}\in\mathcal{I}$ is the item interacted at time step $t$ by user $u$. We aim to estimate the probability of a user clicking on a new item $i^{u}_{T+1}$, denoted as $y^{u}_{i_{T+1}}$, given the historical interaction sequence $S^{u}$. Formally, we seek to learn a function $f:\mathcal{U}\times\mathcal{I}\times\mathcal{S}\rightarrow[0,1]$, where $\mathcal{S}$ is the set of all possible sequences, such that:

$$
P(y^{u}_{i_{T+1}}=1|u,i_{T+1},S^{u};\theta)=f(u,i_{T+1},S^{u};\theta).
$$

To optimize the model, the cross-entropy loss, which measures the difference between predicted and groundtruth click probabilities, is commonly employed as the objective function. Since the formulation in Eq. (1) incorporates temporal dynamics in user behaviors, it potentially leads to better CTR predictions than non-sequential methods that ignore the sequential nature of user interactions.

### 3.2. Feature Interaction

Interaction learning, which aims to capture the complex relationships between different features, is the key towards the success of CTR prediction. We briefly introduce three prominent feature interaction modules: inner product, DCNv2 [^48] and DHEN [^80].

Inner Product-based Interaction. Given input feature vector $\mathbf{x}\in\mathbb{R}^{d}$, the inner product-based interaction, exemplified by Factorization Machines (FM) [^36], learns a latent vector $\mathbf{v}_{j}\in\mathbb{R}^{r}$ for each feature $j$, whose inner product $\langle\mathbf{v}_{j},\mathbf{v}_{k}\rangle$ describes the interaction strength between features $\mathbf{x}(j)$ and $\mathbf{x}(k)$. Hence, the second-order feature interactions can be modeled as:

$$
f_{FM}(\mathbf{x})=\sum_{j=1}^{d}\sum_{k=j+1}^{d}\langle\mathbf{v}_{j},\mathbf{v}_{k}\rangle\mathbf{x}(j)\mathbf{x}(k)+\sum_{j=1}^{d}w_{j}\mathbf{x}(j)+w_{0}.
$$

Deep & Cross Network (DCNv2) [^48]. The DCNv2 model combines a cross network for explicit feature interactions with a deep neural network for implicit feature interactions. Given input feature vector $\mathbf{x}^{(0)}\in\mathbb{R}^{d}$, the $l$ -th layer of the cross network models the second order interaction as follows:

$$
\mathbf{x}^{(l+1)}=\mathbf{x}^{(0)}\odot\left(\mathbf{w}^{(l)}\mathbf{x}^{(l)}+\mathbf{b}^{(l)}\right)+\mathbf{x}^{(l)}.
$$

DCNv2 further combines the stacked cross layers for explicit interactions with a deep layer for implicit interactions, enabling an output enriched with comprehensive information.

Deep Hierarchical Ensemble Network (DHEN) [^80]. DHEN learns a hierarchy of interactions based on a layered structure where each layer is an ensemble of multiple heterogeneous interaction modules. Given the concatenation of $m$ input features $\mathbf{X}^{(l)}\in\mathbb{R}^{d\times m}$, the output of the $l$ -th DHEN layer is computed as:

$$
\mathbf{X}^{(l+1)}=\text{Norm}\left(\text{Ensemble}_{i=1}^{k}\text{Interaction}_{i}(\mathbf{X}^{(l)})+\text{ShortCut}(\mathbf{X}^{(l)})\right),
$$

where $\text{Norm}(\cdot)$ is a normalization function, $\text{Interaction}_{i}$ represents different interaction modules that are further assembled by $\text{Ensemble}_{i=1}^{k}(\cdot)$ such as summation and concatenation, and $\text{ShortCut}(\cdot)$ is an MLP serving as the residual connection for deep layer stacking.

### 3.3. Attention Mechanism

Attention mechanisms [^2] have become an integral part of sequence modeling, serving as the core component behind various model designs [^44] [^8]. We briefly review attention mechanisms, including Multi-Head Attention and Pooling by Multihead Attention (PMA).

Multi-Head Attention (MHA) [^44] allows sequence modeling without regard to distance between tokens [^44]. Given an input sequence $\mathbf{S}=[\mathbf{s}_{1},\dots,\mathbf{s}_{T}]$, where $\mathbf{s}_{t}\in\mathbb{R}^{d}$ is the embedding vector at timestamp $t$, the self-attention operation is defined as:

$$
\text{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})=\text{softmax}\left(\mathbf{QK}^{\scriptscriptstyle\mathsf{T}}/\sqrt{d_{k}}\right)\mathbf{V},
$$

where $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{n\times d_{k}}$ are query, key and value representations computed by linear projections $\mathbf{Q}=\mathbf{S}^{\scriptscriptstyle\mathsf{T}}\mathbf{W}^{Q},\mathbf{K}=\mathbf{S}^{\scriptscriptstyle\mathsf{T}}\mathbf{W}^{K},\mathbf{V}=\mathbf{S}^{\scriptscriptstyle\mathsf{T}}\mathbf{W}^{V}$.

To enable joint attention to information from different embedding subspaces, MHA aggregates $h$ parallel attention as follows:

$$
\text{MHA}(\mathbf{Q},\mathbf{K},\mathbf{V})=\left[\textbf{head}_{1}\|\ldots\|\textbf{head}_{h}\right]\mathbf{W}^{O},
$$

where heads are computed by self-attention $\textbf{head}_{i}=\text{Attn}(\mathbf{QW}_{i}^{Q},\allowbreak\mathbf{KW}_{i}^{K},\mathbf{VW}_{i}^{V})$ and aggregated via an output projector $\mathbf{W}^{O}\in\mathbb{R}^{hd_{k}\times d}$.

Pooling by Multi-Head Attention (PMA) [^18]. Rather than obtaining query, key and values from the same input sequence $\mathbf{S}$, PMA utilizes a learnable query $\mathbf{Q}_{\text{PMA}}\in\mathbb{R}^{k\times d_{k}}$ to summarize the sequence from $k$ different aspects, which can be defined as follows

$$
\text{PMA}(\mathbf{Q}_{\text{PMA}},\mathbf{S})=\text{MHA}(\mathbf{Q}_{\text{PMA}},\mathbf{K},\mathbf{V}).
$$

Intuitively, each column in $\mathbf{Q}_{\text{PMA}}$ is a seed vector summarizing $\mathbf{S}$ as a $d$ -dimensional vector, and the output of PMA is the concatenation of $k$ summarization depicting sequence from different aspects.

## 4\. Methodology

![Refer to caption](https://arxiv.org/html/2411.09852v4/figures/arch.png)

Figure 1. An overview of the InterFormer model architecture. Each block consists of three parts, including (1) Interaction Arch (orange) to learn behavior-aware non-sequence embeddings given sequence queries; (2) Sequence Arch (blue) to learn context-aware sequence embeddings given non-sequence queries; (3) Cross Arch (green) to exchange information between Interaction and Sequence Arch. Note that the dashed green line for CLS token appending only happens at the first layer.

In this section, we present our proposed InterFormer. We first introduce the preprocessing module in Section 4.1, followed by three major modules, including Interaction Arch, Sequence Arch and Cross Arch. To learn behavior-aware non-sequence embeddings, the Interaction Arch models feature interactions (Section 4.2). To learn context-aware sequence embeddings, the Sequence Arch is proposed to optimize sequence (Section 4.3). The Cross Arch connects Interaction and Sequence Arches, enabling effective information summarization and exchange between different modes (Section 4.4). An overview of InterFormer is shown in Figure 1.

### 4.1. Feature Preprocessing

Non-Sequence Feature Preproessing. Two types of non-sequence features are considered, including dense features like user age and item price, and sparse features like user id and item category. To unify the heterogeneity in non-sequence features, different features are transformed into embeddings of the same dimensionality.

Specifically, raw dense features $x_{\text{dense}_{i}}^{(0)}$ are first concatenated to form a dense vector $\mathbf{x}_{\text{dense}}^{(0)}=\left[x_{\text{dense}_{1}}^{(0)},\dots,x_{\text{dense}_{m}}^{(0)}\right]^{\scriptscriptstyle\mathsf{T}}$, which is further transformed into a $d$ -dimensional dense embedding vector via linear transformation, i.e., $\mathbf{x}_{\text{dense}}^{(1)}=\mathbf{W}_{\text{dense}}\mathbf{x}_{\text{dense}}^{(0)}$. Similarly, each raw sparse feature is first encoded as an one-hot vector $\mathbf{x}_{\text{sparse}_{i}}^{(0)}\in\mathbb{R}^{n_{v_{i}}}$, where $n_{v_{i}}$ is the vocabulary size of the $i$ -th sparse feature, and further transformed into a $d$ -dimensional embedding vector by $\mathbf{x}_{\text{sparse}}^{(1)}=\mathbf{W}_{\text{sparse}}\mathbf{x}_{\text{sparse}}^{(0)}$.

By concatenating the dense and sparse embedding vectors, the input non-sequence embedding matrix can be obtained as follows

$$
\mathbf{X}^{(1)}=\left[\mathbf{x}_{\text{dense}}^{(1)}\|\mathbf{x}_{\text{sparse}_{1}}^{(1)}\|\dots\|\mathbf{x}_{\text{sparse}_{n}}^{(1)}\right].
$$

Sequence Feature Preprocessing. Similar to the non-sequence feature preprocessing, an embedding layer is employed such that each interacted item in the sequence is mapped to a $d$ -dimensional vector $\mathbf{s}_{t}$, and the behavior sequences is represented as the concatenation of item embeddings, i.e., $\mathbf{S}^{(0)}=\left[\mathbf{s}_{1}^{(0)}\|\dots\|\mathbf{s}_{T}^{(0)}\right]\in\mathbb{R}^{d\times T}$.

Real-world scenarios often encounter multiple sequences from different user actions (e.g., click and conversion)or various platforms. Moreover, due to the uncertainty of user behaviors, these sequences often contain noisy and irrelevant user-item interactions. To address this, a MaskNet [^51] is employed to unify multiple sequences and filter out the internal noises via self-masking.

Specifically, given $k$ sequences $\mathbf{S}_{1},\cdots,\mathbf{S}_{k}$, they are first concatenated along the embedding dimension, i.e., $\left[\mathbf{S}_{1}^{(0)^{\scriptscriptstyle\mathsf{T}}}\|\cdots\|\mathbf{S}_{k}^{(0)^{\scriptscriptstyle\mathsf{T}}}\right]^{\scriptscriptstyle\mathsf{T}}\in\mathbb{R}^{kd\times T}$, and further processed by the MaskNet operation as follows

$$
\text{MaskNet}(\mathbf{S})=\text{MLP}_{\text{lce}}\left(\mathbf{S}\odot\text{MLP}_{\text{mask}}(\mathbf{S})\right),
$$

where $\text{MLP}_{\text{mask}}\!:\mathbb{R}^{kd\times T}\!\!\to\!\mathbb{R}^{kd\times T}$ computes self-masking to select relevant information from input sequences, and $\text{MLP}_{\text{lce}}:\mathbb{R}^{kd\times T}\!\to\!\mathbb{R}^{d\times T}$ linearly combines multiple sequences into one, matching the dimensionalities of non-sequence and sequence features.

### 4.2. Interaction Arch: Towards Behavior-Aware Interaction Learning

Non-sequence features, such as user profile and ad content, provide substantial information in understanding user preference over a specific item. Modeling the interaction among non-sequence features is the key towards the success of CTR prediction [^36] [^48]. While non-sequence features reflect static user interests [^22] [^80], the behavior sequence provides complementary information depicting user interests from a dynamic view [^85]. For instance, while user profile exhibits a general interest in electronics, the recent browsing history on smartphones offers more specific and timely information on current needs. Therefore, it is crucial to learn behavior-aware non-sequence interactions to adapt to the evolving environment.

To learn behavior-aware non-sequence interaction, we model the interactions among non-sequence features, as well as sequence summarization, by an interaction module. Formally, given the non-sequence input $\mathbf{X}^{(l)}$ and sequence summarization $\mathbf{S}_{\text{sum}}^{(l)}$ at the $l$ -th layer, the output of $l$ -th Interaction Arch is defined as:

$$
\mathbf{X}^{(l+1)}=\text{MLP}^{(l)}\left(\text{Interaction}^{(l)}\left(\left[\mathbf{X}^{(l)}\|\mathbf{S}_{\text{sum}}^{(l)}\right]\right)\right),
$$

where $\text{Interaction}^{(l)}(\cdot)$ is the interaction module. Note that we do not adhere to a specific interaction module, but rather, various backbone models, such as inner product, DCNv2 [^48] and DHEN [^80], can be adopted. Additionally, an MLP is used to transform the output $\mathbf{X}^{(l+1)}$ to match the shape of the input $\mathbf{X}^{(l)}$ enabling selective information aggregation and facilitating easy layer stacking.

By performing interaction learning on the concatenation of $\mathbf{X}^{(l)}$ and $\mathbf{S}^{(l)}_{\text{sum}}$, Eq. (7) incorporates non-sequence and sequence features. The interactions among non-sequence features capture explicit user interests by computing the relevance between user profile and target item content, while the interactions between non-sequence and sequence capture implicit user interests by computing the relevance between target item and user’s current need within the interaction sequence. Additionally, by computing the interactions among sequence features, the most representative information is promoted with high interaction scores, while inactive items with low scores are filtered out. By stacking multiple layers, the Interaction Arch captures rich behavior-aware interactions at different orders.

### 4.3. Sequence Arch: Towards Context-Aware Sequence Modeling

In addition to the explicit static user preference in non-sequence features, the implicit dynamic user interests behind behavior sequences provide complementary information [^85] [^40]. However, given that behavior sequences are highly random and noisy, solely relying on sequential information is ineffective. It is important to incorporate non-sequential context into sequence modeling. For example, a user may randomly browse items on shopping platforms, but given the static information that the user is an electronic fan, electronic items, e.g., smartphones and laptops, in the browsing history provide key information demanding extra attention.

To learn context-aware sequence embeddings, a Sequence Arch is designed based on two key ideas: Personalized FFN (PFFN) and Multi-Head Attention (MHA). To enable interactions between non-sequential and sequential information, PFFN is employed to transform sequence embeddings given non-sequence summarization as query. Given non-sequence summarization $\mathbf{X}_{\text{sum}}^{(l)}$ and sequence embedding $\mathbf{S}^{(l)}$ at the $l$ -th layer, the PFFN operation is defined as:

$$
\text{PFFN}\left(\mathbf{X}_{\text{sum}}^{(l)},\mathbf{S}^{(l)}\right)=f\left(\mathbf{X}_{\text{sum}}^{(l)}\right)\mathbf{S}^{(l)},\vskip-5.0pt
$$

where $f\left(\mathbf{X}_{\text{sum}}^{(l)}\right)$ is an MLP that aims to learn the linear projection on sequence based on non-sequence summarization.

Besides, to model the relationship among events in a sequence, MHA is applied to enable the model to attend to different parts of the sequence, capturing long-range dependencies and contextual information. To incorporate non-sequential context, before feeding into the first InterFormer layer, the non-sequence summarization $\mathbf{X}^{(1)}_{\text{sum}}$ is prepended to the sequence as the CLS token, i.e., $\mathbf{S}^{(1)}=\left[\mathbf{X}^{(1)}_{\text{sum}}\|\mathbf{S}^{(1)}\right]$, hence the following MHA on the CLS token can aggregate sequential information using non-sequential information as the query [^44] [^8]. This is similar to the early fusion idea described in Transact [^52] but we primarily consider ’append’ instead of ’concat’ as the fusion method. Additionally, rotary position embeddings [^39] are applied on tokens so that the positional information in sequences can be effectively leveraged. In general, the Sequence Arch can be written as follow:

$$
\mathbf{S}^{(l+1)}=\text{MHA}^{(l)}\left(\text{PFFN}\left(\mathbf{X}_{\text{sum}}^{(l)},\mathbf{S}^{(l)}\right)\right).\vskip-5.0pt
$$

As the output $\mathbf{S}^{(l+1)}$ is of the same shape as the input $\mathbf{S}^{(l)}$, aggressive aggregation can be avoided and layers can be easily stacked.

Through layer stacking, for one thing, the sequence embeddings is aware of non-sequence interactions at different orders via PFFN; for another, MHA at different layers focuses on different parts of the sequence, capturing multi-scale sequential information. Therefore, the model can learn a rich context-aware encoding of the sequential data capturing both local and global patterns within the sequence.

### 4.4. Cross Arch: Towards Effective Information Selection and Summarization

Though selective information aggregation is achieved in both the Interaction and Sequence Arch, as the dimensionalities of input and output features are retained till the final layer, it is infeasible to directly exchange such information due to (1) ineffectiveness given the noisy information and (2) inefficiency given the high-dimensionality. To solve this dilemma, we propose a cross Arch to select and summarize information before exchanging information.

To start with, given the large scale of non-sequence embeddings, it is important to selectively summarize them to guide sequence learning. To this end, we highlight the most useful information through a personalized gated selection mechanism as follows:

$$
\displaystyle\mathbf{X}_{\text{sum}}^{(l)}=\text{Gating}(\text{MLP}(\mathbf{X}^{(l)})),
$$
$$
\displaystyle\text{where }\text{Gating}(\mathbf{X})=\sigma\left(\mathbf{X}\odot\text{MLP}(\mathbf{X})\right).
$$

where MLP: $\mathbb{R}^{d\times n}\to\mathbb{R}^{d\times n_{\text{sum}}}$ with $n_{\text{sum}}\ll n$, and $\sigma$ is the activation function, e.g., sigmoid, tanh or even identity functions. Self-gating [^5] provides sparse masking on the embeddings such that relevant information is retained while irrelevant noises are filtered out, providing high-quality context for sequence learning.

For sequential information, three types of summarization are neatly designed, including CLS tokens, PMA tokens and recent interacted tokens. The CLS tokens $\mathbf{S}_{\text{CLS}}$ learned by MHA are selected as context-aware sequence summarization. However, the quality of CLS tokens largely depend on the learned non-sequential context. To compensate such heavy reliance and enable more flexibility, the PMA tokens $\mathbf{S}_{\text{PMA}}$ [^18], which are essentially sequence summarization based on learnable queries, are employed. Besides, the $K$ most recent interacted tokens ($\mathbf{S}_{\text{recent}}$) have been proven to be effective in capturing the user’s recent interests [^4] [^52]. The combination of the above information is further gated by a self-gating layer, serving as the behavior summarization for non-sequence interaction:

$$
\mathbf{S}^{(l)}_{\text{sum}}=\text{Gating}\left(\left[\mathbf{S}^{(l)}_{\text{CLS}}\|\mathbf{S}^{(l)}_{\text{PMA}}\|\mathbf{S}^{(l)}_{\text{recent}}\right]\right).
$$

In general, the benefits of the cross Arch are two-fold. On the one hand, by separating information summarization from Interaction and Sequence Arch, information can be retained in both arches, avoiding aggressive information aggregation. On the other hand, effective information exchange can be achieved as high-dimensional non-sequence/sequence features are selected and summarized into low-dimensional embeddings. Therefore, the Cross Arch plays a pivotal role in enabling the model to capture complex interactions between non-sequence behaviors and sequential patterns, leading to more comprehensive representations of the input data.

## 5\. Experiment

In this section, we carry out extensive experiments to evaluate the proposed InterFormer. We first introduce the experiment setup in Section 5.1. Afterwards, we evaluate InterFormer on public benchmark datasets are carried out in Section 5.2. The proposed InterFormer is deployed at Meta, and the post-launch results on internal industrial-scale dataset are provided in Section 5.3.

### 5.1. Experiment Setup

Datasets. We adopt three benchmark datasets, including AmazonElectronics [^12], TaobaoAd [^43], KuaiVideo [^21], and a large-scale internal dataset for evaluation. Dataset statistics are summarized in Table 2 with more details in Appendix B.1.

Table 2. Dataset Summary.

| Dataset | #Samples | #Feat. (Seq/Non-Seq) | Seq Length |
| --- | --- | --- | --- |
| Amazon | 3.0M | 6 (2/4) | 100 |
| TaobaoAd | 25.0M | 22 (3/19) | 50 |
| KuaiVideo | 13.7M | 9 (4/5) | 100 |

Baseline Methods. We compare the proposed InterFormer with 11 state-of-the-art models, including (1) non-sequential methods: FM [^36], xDeepFM [^22], AutoInt+ [^38], DCNv2 [^48], FmFM [^41], DOT product, DHEN [^80], Wukong [^79], and (2) sequential methods: DIN [^86], DIEN [^85], BST [^6], DMIN [^54], DMR [^33], TransAct [^52]. We adopt DHEN as the Interaction Arch in the experiments. Detailed model configurations are provided in Appendix B.2.

Metrics. We adopt four widely-used metrics to evaluate the models from different aspects, including:

- AUC provides an aggregated measure of model capacity in correctly classifying positive and negative samples across all thresholds. Higher the better.
- gAUC provides personalized AUC evaluation, where users are weighted by their click account. Higher the better.
- LogLoss (cross-entropy loss) measures the distance between the prediction $\hat{y}$ and the label $y$, and is computed as $L(y,\hat{y})\!=\!-\left(y\log\left(\hat{y}\right)+(1\!-\!y)\log\left(1\!-\!\hat{y}\right)\right)$. Lower the better.
- NE (normalized entropy) [^13], is the LogLoss normalized by the entropy of the average empirical CTR of the training set. NE provides a data-insensitive evaluation on model performance as the normalization alleviates the effect of background CTR on model evaluation. Lower the better.

### 5.2. Evaluation on Benchmark Datasets

#### 5.2.1. Results.

The experiment results are shown in Table 3. We first observe that sequential methods consistently outperforms non-sequential methods on all datasets. For non-sequential methods, sequence information is naively aggregated in early stages and further processed together with non-sequence information. While in sequential methods, neatly designed sequence processing modules, e.g., RNN [^85] [^40] and attention mechanism [^86] [^6] [^53] [^33], are employed, so that sequential information can be processed in aware of non-sequence context. The universal outperformance of sequential methods validates that different data modes should be processed differently. Besides, results show that aggressive sequence summarization in the early stages, without considering non-sequence context, will impair model performance.

Comparing InterFormer with other sequential methods, InterFormer achieves state-of-the-art performance. Specifically, InterFormer outperforms the best competitor by up to 0.9% in gAUC, 0.14% in AUC and 0.54% in LogLoss. These results demonstrate InterFormer’s effectiveness on diverse datasets and generalization across different recommendation tasks.

Table 3. Experiments on benchmark datasets. Methods with high gAUC and AUC, and low LogLoss and #Params are preferred.

<table><tbody><tr><td>Dataset</td><td colspan="4">AmazonElectronics</td><td colspan="4">TaobaoAds</td><td colspan="4">KuaiVideo</td></tr><tr><td>Metric</td><td>gAUC</td><td>AUC</td><td>LogLoss</td><td>#Params</td><td>gAUC</td><td>AUC</td><td>LogLoss</td><td>#Params</td><td>gAUC</td><td>AUC</td><td>LogLoss</td><td>#Params</td></tr><tr><td>FM</td><td>0.8494</td><td>0.8485</td><td>0.5060</td><td>4.23M</td><td>0.5628</td><td>0.6231</td><td>0.1973</td><td>43.08M</td><td>0.6567</td><td>0.7389</td><td>0.4445</td><td>52.76M</td></tr><tr><td>xDeepFM</td><td>0.8763</td><td>0.8791</td><td>0.4394</td><td>5.46M</td><td>0.5675</td><td>0.6378</td><td>0.1960</td><td>53.79M</td><td>0.6621</td><td>0.7423</td><td>0.4382</td><td>43.56M</td></tr><tr><td>AutoInt+</td><td>0.8786</td><td>0.8804</td><td>0.4441</td><td>5.87M</td><td>0.5701</td><td>0.6467</td><td>0.1941</td><td>42.05M</td><td>0.6619</td><td>0.7420</td><td>0.4369</td><td>43.95M</td></tr><tr><td>DCNv2</td><td>0.8783</td><td>0.8807</td><td>0.4447</td><td>5.23M</td><td>0.5704</td><td>0.6472</td><td>0.1933</td><td>43.71M</td><td>0.6627</td><td>0.7426</td><td>0.4378</td><td>42.48M</td></tr><tr><td>FmFM</td><td>0.8521</td><td>0.8537</td><td>0.4796</td><td>4.21M</td><td>0.5698</td><td>0.6330</td><td>0.1963</td><td>43.06M</td><td>0.6552</td><td>0.7389</td><td>0.4429</td><td>51.97M</td></tr><tr><td>DOT</td><td>0.8697</td><td>0.8703</td><td>0.4485</td><td>4.23M</td><td>0.5701</td><td>0.6482</td><td>0.1941</td><td>41.54M</td><td>0.6605</td><td>0.7435</td><td>0.4361</td><td>41.29M</td></tr><tr><td>DHEN</td><td>0.8759</td><td>0.8790</td><td>0.4398</td><td>4.99M</td><td>0.5708</td><td>0.6509</td><td>0.1929</td><td>43.89M</td><td>0.6589</td><td>0.7424</td><td>0.4365</td><td>42.06M</td></tr><tr><td>Wukong</td><td>0.8747</td><td>0.8765</td><td>0.4455</td><td>4.28M</td><td>0.5693</td><td>0.6478</td><td>0.1932</td><td>41.72M</td><td>0.6587</td><td>0.7423</td><td>0.4372</td><td>41.37M</td></tr><tr><td>DIN</td><td>0.8817</td><td>0.8848</td><td>0.4324</td><td>5.40M</td><td>0.5719</td><td>0.6507</td><td>0.1931</td><td>42.26M</td><td>0.6621</td><td>0.7437</td><td>0.4353</td><td>43.03M</td></tr><tr><td>DIEN</td><td>0.8825</td><td>0.8856</td><td>0.4276</td><td>5.37M</td><td>0.5721</td><td>0.6519</td><td>0.1929</td><td>42.38M</td><td>0.6651</td><td>0.7451</td><td>0.4343</td><td>43.43M</td></tr><tr><td>BST</td><td>0.8823</td><td>0.8847</td><td>0.4305</td><td>5.30M</td><td>0.5698</td><td>0.6489</td><td>0.1935</td><td>42.05M</td><td>0.6617</td><td>0.7446</td><td>0.4352</td><td>42.83M</td></tr><tr><td>DMIN</td><td>0.8831</td><td>0.8852</td><td>0.4298</td><td>5.94M</td><td>0.5723</td><td>0.6498</td><td>0.1933</td><td>42.17M</td><td>0.6623</td><td>0.7449</td><td>0.4356</td><td>41.61M</td></tr><tr><td>DMR</td><td>0.8827</td><td>0.8848</td><td>0.4309</td><td>6.47M</td><td>0.5711</td><td>0.6504</td><td>0.1932</td><td>45.82M</td><td>0.6642</td><td>0.7449</td><td>0.4355</td><td>44.15M</td></tr><tr><td>TransAct</td><td>0.8835</td><td>0.8851</td><td>0.4285</td><td>7.56M</td><td>0.5715</td><td>0.6498</td><td>0.1933</td><td>44.39M</td><td>0.6632</td><td>0.7448</td><td>0.4352</td><td>42.97M</td></tr><tr><td>InterFormer</td><td>0.8843</td><td>0.8865</td><td>0.4253</td><td>7.18M</td><td>0.5728</td><td>0.6528</td><td>0.1926</td><td>44.73M</td><td>0.6637</td><td>0.7453</td><td>0.4340</td><td>43.61M</td></tr></tbody></table>

#### 5.2.2. Analysis of the Model.

![Refer to caption](https://arxiv.org/html/2411.09852v4/x1.png)

(a) DOT

First (Interleaving Learning Style), we analyze how the interleaving learning style benefits heterogeneous interaction learning. To validate the universality of InterFormer, we consider three backbone Interaction Arch, including dot product (DOT), DCNv2, and DHEN, and five different scenarios, including

- sole where only Interaction Arch is adopted, while sequence information is naively aggregated in early stage.
- sep where inter-mode interaction is disabled and Sequence and Interaction Arch are learnt separately.
- s2n where only sequence-to-non-sequence information flow is enabled, while the reverse direction is disabled.
- n2s where only non-sequence-to-sequence information flow is enabled, while the reverse direction is disabled.
- int where the bidirectional information flows are activated.

As shown in Figure 2, int consistently outperforms other scenarios regardless of the backbone Interaction Arch, with an up to 1.46% outperformance in AUC, validating the universal benefits brought by our proposed interleaving learning style. We also observe an ascending order in model performance when more information is exchanged between different data modes, i.e., sole¡sep¡n2s $\approx$ s2n¡int. This validates our claim that the insufficient inter-mode interaction is a key bottleneck of heterogeneous interaction learning, and bidirectional information flow enables different data modes to be learnt in a mutually beneficial manner. Specifically, when equipped with an additional Sequence Arch (sep), the model performance consistently outperforms sole w/o sequence modeling. This not only shows the necessity of employing different arch for different data modes, but also implies the possible performance degradation when heterogeneous data is integrated naively. Furthermore, given that the bidirectional information flow (int) consistently outperforms the unidirectional setting (n2s and s2n), we attribute the outperformance of InterFormer on other state-of-the-art sequential methods [^85] [^54] [^6] to the bidirectional information flow.

Second (Selective Information Aggregation), we evaluate the effect of selective information aggregation by comparing with three aggressive aggregation variants, where sequence information is first compressed by (1) average pooling, (2) MLP, and (3) MHA, and further combined with non-sequence features by the interaction module DHEN. The experimental results are shown in Figure 3.

![Refer to caption](https://arxiv.org/html/2411.09852v4/x4.png)

Figure 3. Study on information aggregation. We compare the performance with early summarization via average pooling, MLP, and MHA, and InterFormer ’s selective aggregation.

As the results suggest, selective aggregation (InterFormer) consistently achieves the best performance compared with other aggressive variants. In general, we can regard MLP as a more selective version of average pooling as MLP gradually compresses the sequence information, and similarly, MHA process sequence less aggressive than MLP. Therefore, we may conclude that with more selective aggregation, the CTR prediction quality improves.

Third (Sequence Modeling), we visualize the learned attention map in Figure 4 to understand sequence modeling. We observe that the attention map exhibits a cluster structure, i.e., surrounding tokens are likely to interact, which can be regarded as a weighted pooling within selected surrounding tokens. Such selective pooling process alleviates random noises in the behavior history to extract better temporal user interests. Besides, MHAs at different layers focus on different tokens at different scales. For example, the attention map is quite uniform in the first layer, indicating pooling in a broad scale for long-term interests. In contrast, the third layer exhibits stronger interactions within smaller clusters, resulting in fine-grained aggregation for short-term interests. Furthermore, the second layer focuses on specific tokens, e.g., recent tokens (tokens in right columns) and the 24-th token, for item-specific information.

![Refer to caption](https://arxiv.org/html/2411.09852v4/x5.png)

Figure 4. Attention map on TaobaoAds. The first 4 tokens are CLS tokens, followed by the behavior sequence of length 50.

#### 5.2.3. Ablation Study

![Refer to caption](https://arxiv.org/html/2411.09852v4/x6.png)

Figure 5. Ablation study on InterFormer, where ’-’ indicates ablating different information or modules.

To better understand how different model designs contribute to the final performance, we ablate different exchanged information, including PMA tokens and recent interacted tokens, as well as different modules like the gating module, PFFN and MHA. As the results shown in Figure 5, all the exchanged information and modules contribute to the model performance. Specifically, PMA tokens provide effective sequence summarization given non-sequence context and are of great importance, ablating which decreases AUC by up to 0.004. The gating modules help selective summarize information from different data modes contribute up to 0.003 AUC improvement. The sequence modeling modules (PFFN and MHA) also improves the performance to some extent.

### 5.3. Evaluation on Industrial Datasets

To evaluate InterFormer’s performance at industrial-scale data, we carry out experiments on a large-scale internal dataset from Meta, containing 70B samples in total, hundreds of non-sequence features and 10 sequences of length 200 to 1,000.

#### 5.3.1. Results.

In general, a 3-layer InterFormer achieves a 0.15% NE gain compared to the internal SOTA model with similar FLOPs and 24% Queries Per Second (QPS) gain. Together with feature scaling, the improvement on NE can be further enlarged with a 10% of Model FLOPs Utilization (MFU) on 512 GPUs attained. InterFormer shows great generalizability on a wide range of models showing promising ROI.

Besides (Sequence Feature Scaling), we evaluate how the performance of InterFormer changes when the sequence feature scale increases in internal dataset. In addition to six sequences of length 100, we include two additional long sequences of length 1000, and we observe a 0.14% improvement in NE. As shown in Figure 6(a), InterFormer exhibits better scalability compared to the strong internal baseline (Internal) that leverages cross-attention to capture sequence and non-sequence information, as NE curve of InterFormer continues to decrease when more training samples are involved, outperforming Internal by 0.06% in NE. Besides, we also tried to merge six sequences to generate one long sequence of length 600, and we observe promising efficiency improvements in QPS (+20%) and MFU (+17%) with a 0.02% NE tradeoff. From the modeling perspective, the results validate that InterFormer is able to enlarge the NE gain brought by the sequence feature scaling.

In addition (Model Scaling), we evaluate how InterFormer performs when the model scale increases. As Figure 6(b) shows, scaling InterFormer from 1 to 4 layers achieves consistent NE gains, exhibiting good scaling properties. Specifically, compared to a single layer InterFormer, a two-layer InterFormer achieves a significant 0.13% NE gain, with an additional 0.05% and 0.04% NE gain achieved by stacking 3 and 4 layers, respectively.

![Refer to caption](https://arxiv.org/html/2411.09852v4/figures/feature_scaling.png)

(a) Feature scaling

#### 5.3.2. Model-System Co-Design.

Model architecture design plays a crucial role in training efficiency with its implication on GPU FLOPs utilization and inter-GPU communication. We highlight two optimizations based on InterFormer architecture that in total boost training efficiency by more than 30%, including communication overhead reduction and computation efficiency.

For communication overhead reduction, the Interaction Arch DHEN with heavy parameters and relatively light computation tends to be FSDP [^83] communication-bound in distributed training, while the Sequence Arch Transformer with much higher computation (FLOPs) to parameter ratio is normally computation-bound. To alleviate such inefficiency, our parallel design of Interaction and Sequence Arch allows the exposed communication from DHEN modules to overlap with sequence computation effectively, resulting in a 20% QPS improvement compared to the performing two modules sequentially.

For computation efficiency, we perform a series of optimizations, including 1) reallocating FLOPs from small, low-return modules to larger, high ROI modules and 2) combining smaller kernels to better utilize GPU resources. These optimizations improve MFU for interaction modules from 11% to 16%, and DHEN from 38% to 45%, with a 19% MFU improvement for the overall InterFormer layer.

#### 5.3.3. Online Impact

Interformer has been deployed in key Ads models, including the largest ones through [^32], showing superior ROI in scaling compared to other state-of-the-art architectures. Pilot launches in 2024 have yielded a 0.6% improvement in topline metrics. We expect great impact to be delivered in subsequent deployment.

## 6\. Conclusion

In this paper, we propose a novel building block for heterogeneous interaction learning in CTR prediction named InterFormer. The key idea is an interleaving learning style between different data modes to achieve effective inter-mode interaction, as well as selective information aggregation. In general, An Interaction Arch and a Sequence Arch are adopted to achieve behavior-aware interaction learning and context-aware sequence modeling, respectively. A Cross Arch is further proposed to for effective information selection and summarization. InterFormer achieves consistent outperformance on benchmark datasets, and 0.15% improvement on the NE of CTR prediction on internal large-scale dataset. Pilot launches in key Ads models have yielded a 0.6% improvement in topline metrics and we expect great impact to be delivered.

## References

## Appendix

## Appendix A Model Design

### A.1. Algorithm

We present the overall algorithm of InterFormer in Algorithm 1.

Algorithm 1 InterFormer

 Non-sequence feature $\mathbf{X}^{(0)}$, sequence feature $\mathbf{S}^{(0)}$, number of layers $L$;

 CTR prediction $\hat{y}$;

 Preprocess $\mathbf{X}^{(0)},\mathbf{S}^{(0)}$ via Eqs. (5) and (6) to obtain $\mathbf{X}^{(1)},\mathbf{S}^{(1)}$;

 Compute non-sequence summarization $\mathbf{X}_{\text{sum}}^{(1)}$ via Eq. (10);

 Prepend $\mathbf{X}_{\text{sum}}^{(1)}$ before $\mathbf{S}^{(1)}$ as CLS tokens;

 for $l=1,2,\dots,L$ do

  Cross Arch: compute non-sequence summarization $\mathbf{X}^{(l)}_{\text{sum}}$ and sequence summarization $\mathbf{S}^{(l)}_{\text{sum}}$ by Eqs. (10) and (11);

  Interaction Arch: compute non-sequence embeddings $\mathbf{X}^{(l+1)}$ given $\mathbf{X}^{(l)}$ and $\mathbf{S}^{(l)}_{\text{sum}}$ by Eq. (7);

  Sequence Arch: compute sequence embeddings $\mathbf{S}^{(l+1)}$ given $\mathbf{S}^{(l)}$ and $\mathbf{X}^{(l)}_{\text{sum}}$ by Eq. (9);

 end for

 Compute CTR prediction by $\hat{y}=\text{MLP}\left(\left[\mathbf{X}^{(L)}_{\text{sum}}\|\mathbf{S}^{(L)}_{\text{sum}}\right]\right)$;

 return CTR prediction $\hat{y}$.

### A.2. More Details on Module Blocks

We provide more details on module blocks involved in InterFormer architecture, including Linear Compressed Embedding (LCE), and personalized FeedForward Network (PFFN).

#### A.2.1. Linear Compressed Embedding

Given numerous non-sequence features, it is essential to compress embeddings to a manageable size. In general, given $N$ $d$ -dimensional features $\mathbf{X}\in\mathbb{R}^{d\times N}$, LCE is a linear transformation on the sample dimension $\mathbf{W}\in\mathbb{R}^{N\times M}$, such that the linear transformation $\mathbf{XW}\in\mathbb{R}^{d\times M}$ serves as the compressed embedding with $M$ features. Together with self-gating modules, LCE selectively compress and denoise numerous embeddings, benefiting the model in both efficiency and effectiveness.

#### A.2.2. Personalized FFN

PFFN learns interactions between non-sequence and sequence embeddings, whose key idea is to apply an FFN on the sequence embeddings with weight learned based on the summarized non-sequence embeddings. Given non-sequence summarization $\mathbf{X}_{\text{sum}}\in\mathbb{R}^{d\times n_{\text{sum}}}$ and sequence embedding $\mathbf{S}\in\mathbb{R}^{d\times T}$, PFFN first learns the transformation weight $\mathbf{W}_{\text{PFFN}}=\mathbf{X}_{\text{sum}}\mathbf{W}\in\mathbb{R}^{d\times d}$, which is further applied on the feature dimension of sequence embeddings, i.e., $\mathbf{W}_{\text{PFFN}}\mathbf{S}$. In contrast to the vanilla FFN in Transformer that may not add too much value to subsequent sequence modeling, PFFN integrate side information from other data modes. In fact, PFFN is essentially computing enhanced dot-products based interaction between sequence embeddings and summarized non-sequence embeddings that brings much more value for the following sequence modeling.

## Appendix B Experiment Pipeline

We adopt the public BARS evaluation framework [^87] for benchmark experiments. In general, an Adam [^17] optimizer with a learning rate scheduler is adopted for optimization, where initial learning rate is tuned in {1e-1, 1e-2, 1e-3}. We use a batch size of 2048, and train up to 100 epochs with early stop. We adopt the Swish function [^35] for activation. We use NVIDIA A100 for benchmark experiments and NVIDIA H100 for internal experiments. More details on datasets and model configurations are provided as follows.

### B.1. Datasets

We provide a brief description of datasets used in the paper.

- AmazonElectronics [^12] contains product reviews and metadata from Amazon with 192,403 users, 63,001 goods, 801 categories and 1,689,188 samples. Non-sequence features include user id, item id and item category. Sequence features include interacted items and corresponding categories with length of 100. We follow the public split with 2.60M samples for training and 0.38M samples for testing.
- TaobaoAds [^43] contains 8 days of ad click-through data on Taobao (26 million records) randomly sampled from 1,140,000 users. Non-sequence features include item-related features, e.g., ad id, category, and price, and user-related features, e.g., user id, gender, and age. Sequence features include interacted items’ brands and categories, and user behaviors with length of 50. We follow the public split with 22.0M samples for training and 3.1M samples for testing.
- KuaiVideo [^21] contains 10,000 users and their 3,239,534 interacted micro-videos. Non-sequence features include user id, video id, and visual embeddings of videos. Sequence features include different behaviors, e.g., click, like, and not click, with length of 100. We follow the public split with 10.9M samples for training and 2.7M samples for testing.
- Internal contains 70B entries in total, with hundreds of non-sequence features describing users and items, and 10 sequence features of length 200 to 1,000.

### B.2. Model Configuration

We provide the model configurations in the paper. We use the best searched model configurations on BARS whenever possible, and use the provided model default hyperparameters for the rest. In general, the attention MLP has sizes in {512,256,128,64} with number of heads ranged in {1,2,4,8}. An MLP with sizes in {1024,512,256,128} is adopted as the classifier head. Other model-specific configurations are provided as follows

- xDeepFM [^22]: The compressed interaction network (CIN) is an MLP with size of 32.
- DCNv2 [^48]: The parallel structure is an MLP of size 512, and the stacked structure is an MLP with size of 500. We set the low-rank of the cross layer as 32.
- DHEN [^80]: We adopt DOT product and DCN as the ensembled modules. The number of layers ranged in {1,2,3}.
- InterFormer: The number of cls tokens is set to be 4, the number of PMA tokens ranged is set to be 2, and the number of recent tokens ranged is set to be 2.

[^1]: 

[^2]: Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473* (2014).

[^3]: Wenxuan Bao, Zhichen Zeng, Zhining Liu, Hanghang Tong, and Jingrui He. 2024. Matcha: Mitigating Graph Structure Shifts with Test-Time Adaptation. *arXiv preprint arXiv:2410.06976* (2024).

[^4]: Fedor Borisyuk, Mingzhou Zhou, Qingquan Song, Siyu Zhu, Birjodh Tiwana, Ganesh Parameswaran, Siddharth Dangi, Lars Hertel, Qiang Xiao, Xiaochen Hou, et al. 2024. LiRank: Industrial Large Scale Ranking Models at LinkedIn. *arXiv preprint arXiv:2402.06859* (2024).

[^5]: Yekun Chai, Shuo Jin, and Xinwen Hou. 2020. Highway transformer: Self-gating enhanced self-attentive networks. *arXiv preprint arXiv:2004.08178* (2020).

[^6]: Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior sequence transformer for e-commerce recommendation in alibaba. In *Proceedings of the 1st international workshop on deep learning practice for high-dimensional sparse data*. 1–4.

[^7]: Xiaoshuang Chen, Gengrui Zhang, Yao Wang, Yulin Wu, Shuo Su, Kaiqiao Zhan, and Ben Wang. 2024. Cache-Aware Reinforcement Learning in Large-Scale Recommender Systems. In *Companion Proceedings of the ACM on Web Conference 2024*. 284–291.

[^8]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805* (2018).

[^9]: Ali Mamdouh Elkahky, Yang Song, and Xiaodong He. 2015. A multi-view deep learning approach for cross domain user modeling in recommendation systems. In *Proceedings of the 24th international conference on world wide web*. 278–288.

[^10]: Yongqiang Han, Hao Wang, Kefan Wang, Likang Wu, Zhi Li, Wei Guo, Yong Liu, Defu Lian, and Enhong Chen. 2024. Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation. In *Proceedings of the ACM on Web Conference 2024*. 3297–3306.

[^11]: Ruining He and Julian McAuley. 2016a. Fusing similarity models with markov chains for sparse sequential recommendation. In *2016 IEEE 16th international conference on data mining (ICDM)*. IEEE, 191–200.

[^12]: Ruining He and Julian McAuley. 2016b. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In *proceedings of the 25th international conference on world wide web*. 507–517.

[^13]: Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In *Proceedings of the eighth international workshop on data mining for online advertising*. 1–9.

[^14]: Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In *Proceedings of the 22nd ACM international conference on Information & Knowledge Management*. 2333–2338.

[^15]: Baoyu Jing, Yuchen Yan, Kaize Ding, Chanyoung Park, Yada Zhu, Huan Liu, and Hanghang Tong. 2024. Sterling: Synergistic representation learning on bipartite graphs. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 38. 12976–12984.

[^16]: Baoyu Jing, Yuchen Yan, Yada Zhu, and Hanghang Tong. 2022. Coin: Co-cluster infomax for bipartite graphs. *arXiv preprint arXiv:2206.00006* (2022).

[^17]: Diederik P Kingma. 2014. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980* (2014).

[^18]: Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set transformer: A framework for attention-based permutation-invariant neural networks. In *International conference on machine learning*. PMLR, 3744–3753.

[^19]: Jinning Li, Ruipeng Han, Chenkai Sun, Dachun Sun, Ruijie Wang, Jingying Zeng, Yuchen Yan, Hanghang Tong, and Tarek Abdelzaher. 2024. Large language model-guided disentangled belief representation learning on polarized social graphs. In *2024 33rd International Conference on Computer Communications and Networks (ICCCN)*. IEEE, 1–9.

[^20]: Jinning Li, Huajie Shao, Dachun Sun, Ruijie Wang, Yuchen Yan, Jinyang Li, Shengzhong Liu, Hanghang Tong, and Tarek Abdelzaher. 2022. Unsupervised belief representation learning with information-theoretic variational graph auto-encoders. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 1728–1738.

[^21]: Yongqi Li, Meng Liu, Jianhua Yin, Chaoran Cui, Xin-Shun Xu, and Liqiang Nie. 2019. Routing micro-videos via a temporal graph-guided recommendation system. In *Proceedings of the 27th ACM international conference on multimedia*. 1464–1472.

[^22]: Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1754–1763.

[^23]: Mingfu Liang, Xi Liu, Rong Jin, Boyang Liu, Qiuling Suo, Qinghai Zhou, Song Zhou, Laming Chen, Hua Zheng, Zhiyuan Li, Shali Jiang, Jiyan Yang, Xiaozhen Xia, Fan Yang, Yasmine Badr, Ellie Wen, Shuyu Xu, Hansey Chen, Zhengyu Zhang, Jade Nie, Chunzhi Yang, Zhichen Zeng, Weilin Zhang, Xingliang Huang, Qianru Li, Shiquan Wang, Evelyn Lyu, Wenjing Lu, Rui Zhang, Wenjun Wang, Jason Rudy, Mengyue Hang, Kai Wang, Bo Long, Wenlin Chen, Santanu Kolay, and Huayu Li. 2025. External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation. In *Companion Proceedings of the ACM on Web Conference 2025* (Sydney NSW, Australia) *(WWW ’25)*. Association for Computing Machinery, New York, NY, USA, 344–353. [https://doi.org/10.1145/3701716.3715223](https://doi.org/10.1145/3701716.3715223)

[^24]: Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. 2025a. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. *arXiv preprint arXiv:2505.05422* (2025).

[^25]: Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. 2024. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. *Advances in Neural Information Processing Systems* 37 (2024), 87766–87800.

[^26]: Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, and Zhenan Sun. 2025b. Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs. *arXiv preprint arXiv:2508.14896* (2025).

[^27]: Xiao Lin, Zhichen Zeng, Tianxin Wei, Zhining Liu, Hanghang Tong, et al. 2025c. Cats: Mitigating correlation shift for multivariate time series classification. *arXiv preprint arXiv:2504.04283* (2025).

[^28]: Xiaolong Liu, Zhichen Zeng, Xiaoyi Liu, Siyang Yuan, Weinan Song, Mengyue Hang, Yiqun Liu, Chaofei Yang, Donghyun Kim, Wen-Yen Chen, et al. 2024b. A collaborative ensemble framework for ctr prediction. *arXiv preprint arXiv:2411.13700* (2024).

[^29]: Zhining Liu, Ruizhong Qiu, Zhichen Zeng, Hyunsik Yoo, David Zhou, Zhe Xu, Yada Zhu, Kommy Weldemariam, Jingrui He, and Hanghang Tong. 2023. Class-imbalanced graph learning without class rebalancing. *arXiv preprint arXiv:2308.14181* (2023).

[^30]: Zhining Liu, Ruizhong Qiu, Zhichen Zeng, Yada Zhu, Hendrik Hamann, and Hanghang Tong. 2024a. AIM: Attributing, interpreting, mitigating data unfairness. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 2014–2025.

[^31]: Zhining Liu, Ze Yang, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Yada Zhu, Hendrik Hamann, Jingrui He, and Hanghang Tong. 2025. Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting. *arXiv preprint arXiv:2505.18442* (2025).

[^32]: Liang Luo, Mengyue Hang, Zhengyu Zhang, Andrew Gu, Buyun Zhang, Boyang Liu, Chen Chen, Fan Yang, Huayu Li, Jade Nie, et al. \[n. d.\]. M3C: a Multi-Domain Multi-Objective, Mixed-Modality Framework for Cost-Effective, Industry Scale Recommendation. (\[n. d.\]).

[^33]: Ze Lyu, Yu Dong, Chengfu Huo, and Weijun Ren. 2020. Deep match to rank model for personalized click-through rate prediction. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 34. 156–163.

[^34]: Massimo Quadrana, Paolo Cremonesi, and Dietmar Jannach. 2018. Sequence-aware recommender systems. *ACM computing surveys (CSUR)* 51, 4 (2018), 1–36.

[^35]: Prajit Ramachandran, Barret Zoph, and Quoc V Le. 2017. Searching for activation functions. *arXiv preprint arXiv:1710.05941* (2017).

[^36]: Steffen Rendle. 2010. Factorization machines. In *2010 IEEE International conference on data mining*. IEEE, 995–1000.

[^37]: Guy Shani, David Heckerman, Ronen I Brafman, and Craig Boutilier. 2005. An MDP-based recommender system. *Journal of machine Learning research* 6, 9 (2005).

[^38]: Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 2019. Autoint: Automatic feature interaction learning via self-attentive neural networks. In *Proceedings of the 28th ACM international conference on information and knowledge management*. 1161–1170.

[^39]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing* 568 (2024), 127063.

[^40]: Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM international conference on information and knowledge management*. 1441–1450.

[^41]: Yang Sun, Junwei Pan, Alex Zhang, and Aaron Flores. 2021. FM2: Field-matrixed factorization machines for recommender systems. In *Proceedings of the web conference 2021*. 2828–2837.

[^42]: Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In *Proceedings of the eleventh ACM international conference on web search and data mining*. 565–573.

[^43]: Tianchi. \[n. d.\]. Ad display/click data on taobao.com, 2018. [https://tianchi.aliyun.com/dataset/56](https://tianchi.aliyun.com/dataset/56).

[^44]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. *Advances in neural information processing systems* 30 (2017).

[^45]: Dingsu Wang, Yuchen Yan, Ruizhong Qiu, Yada Zhu, Kaiyu Guan, Andrew Margenot, and Hanghang Tong. 2023b. Networked time series imputation via position-aware graph enhanced variational autoencoders. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 2256–2268.

[^46]: Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD’17*. 1–7.

[^47]: Ruijie Wang, Baoyu Li, Yichen Lu, Dachun Sun, Jinning Li, Yuchen Yan, Shengzhong Liu, Hanghang Tong, and Tarek F Abdelzaher. 2023a. Noisy positive-unlabeled learning with self-training for speculative knowledge graph reasoning. *arXiv preprint arXiv:2306.07512* (2023).

[^48]: Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021b. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In *Proceedings of the web conference 2021*. 1785–1797.

[^49]: Ruijie Wang, Yuchen Yan, Jialu Wang, Yuting Jia, Ye Zhang, Weinan Zhang, and Xinbing Wang. 2018. Acekg: A large-scale knowledge graph for academic data mining. In *Proceedings of the 27th ACM international conference on information and knowledge management*. 1487–1490.

[^50]: Shoujin Wang, Liang Hu, Yan Wang, Longbing Cao, Quan Z Sheng, and Mehmet Orgun. 2019. Sequential recommender systems: challenges, progress and prospects. *arXiv preprint arXiv:2001.04830* (2019).

[^51]: Zhiqiang Wang, Qingyun She, and Junlin Zhang. 2021a. Masknet: Introducing feature-wise multiplication to CTR ranking models by instance-guided mask. *arXiv preprint arXiv:2102.07619* (2021).

[^52]: Xue Xia, Pong Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-Wei Wang, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan Zhang, and Andrew Zhai. 2023. Transact: Transformer-based realtime user action model for recommendation at pinterest. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 5249–5259.

[^53]: Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional factorization machines: learning the weight of feature interactions via attention networks. In *Proceedings of the 26th International Joint Conference on Artificial Intelligence*. 3119–3125.

[^54]: Zhibo Xiao, Luwei Yang, Wen Jiang, Yi Wei, Yi Hu, and Hao Wang. 2020. Deep multi-interest network for click-through rate prediction. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*. 2265–2268.

[^55]: Xin Xin, Bo Chen, Xiangnan He, Dong Wang, Yue Ding, and Joemon M Jose. 2019. CFM: Convolutional factorization machines for context-aware recommendation.. In *IJCAI*, Vol. 19. 3926–3932.

[^56]: Haobo Xu, Yuchen Yan, Dingsu Wang, Zhe Xu, Zhichen Zeng, Tarek F Abdelzaher, Jiawei Han, and Hanghang Tong. \[n. d.\]. Slog: An inductive spectral graph neural network beyond polynomial filter. In *Forty-first International Conference on Machine Learning*.

[^57]: Zhe Xu, Ruizhong Qiu, Yuzhong Chen, Huiyuan Chen, Xiran Fan, Menghai Pan, Zhichen Zeng, Mahashweta Das, and Hanghang Tong. 2024. Discrete-state continuous-time diffusion for graph generation. *Advances in Neural Information Processing Systems* 37 (2024), 79704–79740.

[^58]: Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Xiaoting Li, Zhe Xu, Zhichen Zeng, Lihui Liu, Zhining Liu, and Hanghang Tong. 2024a. Thegcn: Temporal heterophilic graph convolutional network. *arXiv preprint arXiv:2412.16435* (2024).

[^59]: Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Minghua Xu, Mahashweta Das, Hao Yang, and Hanghang Tong. 2023a. From trainable negative depth to edge heterophily in graphs. *Advances in Neural Information Processing Systems* 36 (2023), 70162–70178.

[^60]: Yuchen Yan, Yuzhong Chen, Mahashweta Das, Hao Yang, and Hanghang Tong. \[n. d.\]. ReD-GCN: Revisit the Depth of Graph Convolutional Network. (\[n. d.\]).

[^61]: Yuchen Yan, Yongyi Hu, Qinghai Zhou, Lihui Liu, Zhichen Zeng, Yuzhong Chen, Menghai Pan, Huiyuan Chen, Mahashweta Das, and Hanghang Tong. 2024b. Pacer: Network embedding from positional to structural. In *Proceedings of the ACM Web Conference 2024*. 2485–2496.

[^62]: Yuchen Yan, Yongyi Hu, Qinghai Zhou, Shurang Wu, Dingsu Wang, and Hanghang Tong. 2024c. Topological anonymous walk embedding: A new structural node embedding approach. In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management*. 2796–2806.

[^63]: Yuchen Yan, Baoyu Jing, Lihui Liu, Ruijie Wang, Jinning Li, Tarek Abdelzaher, and Hanghang Tong. 2023b. Reconciling competing sampling strategies of network embedding. *Advances in Neural Information Processing Systems* 36 (2023), 6844–6861.

[^64]: Yuchen Yan, Lihui Liu, Yikun Ban, Baoyu Jing, and Hanghang Tong. 2021a. Dynamic knowledge graph alignment. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 35. 4564–4572.

[^65]: Yuchen Yan, Si Zhang, and Hanghang Tong. 2021b. Bright: A bridging algorithm for network alignment. In *Proceedings of the web conference 2021*. 3907–3917.

[^66]: Yuchen Yan, Qinghai Zhou, Jinning Li, Tarek Abdelzaher, and Hanghang Tong. 2022. Dissecting cross-layer dependency inference on multi-layered inter-dependent networks. In *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*. 2341–2351.

[^67]: Carl Yang, Lanxiao Bai, Chao Zhang, Quan Yuan, and Jiawei Han. 2017. Bridging collaborative filtering and semi-supervised learning: a neural approach for poi recommendation. In *Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining*. 1245–1254.

[^68]: Xiaodong Yang, Huiyuan Chen, Yuchen Yan, Yuxin Tang, Yuying Zhao, Eric Xu, Yiwei Cai, and Hanghang Tong. 2024. SimCE: Simplifying Cross-Entropy Loss for Collaborative Filtering. *arXiv preprint arXiv:2406.16170* (2024).

[^69]: Yeongwook Yang, Hong-Jun Jang, and Byoungwook Kim. 2020. A hybrid recommender system for sequential recommendation: combining similarity models with markov chains. *IEEE Access* 8 (2020), 190136–190146.

[^70]: Hyunsik Yoo, Zhichen Zeng, Jian Kang, Ruizhong Qiu, David Zhou, Zhining Liu, Fei Wang, Charlie Xu, Eunice Chan, and Hanghang Tong. 2024. Ensuring user-side fairness in dynamic recommender systems. In *Proceedings of the ACM Web Conference 2024*. 3667–3678.

[^71]: Qi Yu, Zhichen Zeng, Yuchen Yan, Zhining Liu, Baoyu Jing, Ruizhong Qiu, Ariful Azad, and Hanghang Tong. 2025a. PLANETALIGN: A Comprehensive Python Library for Benchmarking Network Alignment. *arXiv preprint arXiv:2505.21366* (2025).

[^72]: Qi Yu, Zhichen Zeng, Yuchen Yan, Lei Ying, R Srikant, and Hanghang Tong. 2025b. Joint optimal transport and embedding for network alignment. In *Proceedings of the ACM on Web Conference 2025*. 2064–2075.

[^73]: Zhichen Zeng, Boxin Du, Si Zhang, Yinglong Xia, Zhining Liu, and Hanghang Tong. 2024. Hierarchical multi-marginal optimal transport for network alignment. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 38. 16660–16668.

[^74]: Zhichen Zeng, Ruizhong Qiu, Wenxuan Bao, Tianxin Wei, Xiao Lin, Yuchen Yan, Tarek F Abdelzaher, Jiawei Han, and Hanghang Tong. 2025. Pave Your Own Path: Graph Gradual Domain Adaptation on Fused Gromov-Wasserstein Geodesics. *arXiv preprint arXiv:2505.12709* (2025).

[^75]: Zhichen Zeng, Ruizhong Qiu, Zhe Xu, Zhining Liu, Yuchen Yan, Tianxin Wei, Lei Ying, Jingrui He, and Hanghang Tong. \[n. d.\]. Graph mixup on approximate gromov–wasserstein geodesics. In *Forty-first International Conference on Machine Learning*.

[^76]: Zhichen Zeng, Si Zhang, Yinglong Xia, and Hanghang Tong. 2023a. Parrot: Position-aware regularized optimal transport for network alignment. In *Proceedings of the ACM web conference 2023*. 372–382.

[^77]: Zhichen Zeng, Ruike Zhu, Yinglong Xia, Hanqing Zeng, and Hanghang Tong. 2023b. Generative graph dictionary learning. In *International Conference on Machine Learning*. PMLR, 40749–40769.

[^78]: Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152* (2024).

[^79]: Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, et al. 2024. Wukong: Towards a Scaling Law for Large-Scale Recommendation. *arXiv preprint arXiv:2403.02545* (2024).

[^80]: Buyun Zhang, Liang Luo, Xi Liu, Jay Li, Zeliang Chen, Weilin Zhang, Xiaohan Wei, Yuchen Hao, Michael Tsang, Wenjun Wang, et al. 2022. DHEN: A deep and hierarchical ensemble network for large-scale click-through rate prediction. *arXiv preprint arXiv:2203.11014* (2022).

[^81]: Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. *ACM computing surveys (CSUR)* 52, 1 (2019), 1–38.

[^82]: Yongfeng Zhang, Qingyao Ai, Xu Chen, and W Bruce Croft. 2017. Joint representation learning for top-n recommendation with heterogeneous information sources. In *Proceedings of the 2017 ACM on Conference on Information and Knowledge Management*. 1449–1458.

[^83]: Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. 2023. Pytorch fsdp: experiences on scaling fully sharded data parallel. *arXiv preprint arXiv:2304.11277* (2023).

[^84]: Guorui Zhou, Weijie Bian, Kailun Wu, Lejian Ren, Qi Pi, Yujing Zhang, Can Xiao, Xiang-Rong Sheng, Na Mou, Xinchen Luo, et al. 2020. CAN: revisiting feature co-action for click-through rate prediction. *arXiv preprint arXiv:2011.05625* (2020).

[^85]: Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 33. 5941–5948.

[^86]: Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1059–1068.

[^87]: Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, and Rui Zhang. 2022. Bars: Towards open benchmarking for recommender systems. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2912–2923.