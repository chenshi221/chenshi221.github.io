---
title: "OneRec Technical Report"
source: "https://arxiv.org/html/2506.13695v4"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
OneRec Team

###### Abstract

Recommender systems have been widely used in various large-scale user-oriented platforms for many years. Over the past decade, recommendation technology has evolved from traditional heuristic-based rules to deep learning models, significantly improving recommendation accuracy. However, compared to the rapid changes and developments in the AI community, recommendation systems have not achieved a breakthrough in recent years. For instance, they still rely on a multi-stage cascaded architecture rather than an end-to-end approach, leading to computational fragmentation and optimization inconsistencies. Additionally, the cascading structure has hindered the effective application of key breakthrough technologies from the AI community in recommendation scenarios.

To address these issues, we propose OneRec, which reshapes the recommendation system through an end-to-end generative approach. Under this new architecture, we have achieved promising results. Firstly, we have enhanced the computational FLOPs of the current recommendation model by 10 $\times$ and have identified the scaling laws for recommendations within certain boundaries. Secondly, reinforcement learning (RL) techniques, previously difficult to apply for optimizing recommendations, show significant potential in this framework. Lastly, through infrastructure optimizations, we have achieved 23.7% and 28.8% Model FLOPs Utilization (MFU) on flagship GPUs during training and inference, respectively, aligning closely with the LLM community. This architecture significantly reduces communication and storage overhead, resulting in operating expense (OPEX) that is only 10.6% of traditional recommendation pipelines. Deployed in Kuaishou/Kuaishou Lite APP, it handles 25% of total queries per second (QPS), enhancing overall App Stay Time by 0.54% and 1.24%, respectively. Additionally, we have observed significant increases in metrics such as 7-day Lifetime (LT7), which is a crucial indicator of recommendation experience. We also provide practical lessons and insights derived from developing, optimizing, and maintaining a production-scale recommendation system with significant real-world impact.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x1.png)

Figure 1: Online performance, FLOPs, OPEX, and MFU comparison.

## 1 Introduction

With the rapid advancement of online services, recommender systems (RS) have become essential infrastructure for mitigating information overload and delivering personalized content at scale [^25]. During the past decades, recommender systems have achieved several breakthrough advancements - from early Factorization Machines [^24] to modern deep learning architectures [^8] [^4] [^34] [^19]. Despite the substantial progress made by the RS research community, traditional recommendation models still rely on multi-stage cascaded architectures (see the top part of Figure 2) rather than end-to-end approaches, which face several limitations that hinder their optimal performance:

Fragmented Compute. The cascaded architecture suffers from low computational efficiency. Our comprehensive analysis of resource distribution, using Kuaishou as a case study, reveals that over 50% of resources during serving are allocated to communication and storage rather than high-precision computation. This significant allocation to non-computational tasks highlights a fundamental inefficiency in the current architecture. Moreover, the resources dedicated to computation, particularly for the most computation-intensive ranking models, demonstrate markedly low utilization. Specifically, the model’s training and inference MFU is only 4.6% and 11.2% on flagship GPUs, respectively, which is substantially lower than the efficiency observed in large language models (LLMs), where the MFU is approximately 40% on H100 [^28] [^7]. This discrepancy underscores the inefficiency in resource utilization for computational tasks in recommender systems. Additionally, due to the high QPS requirements (greater than 400k) and low latency demands (less than 500ms), recommender models are often constrained to operate at a low scale and are not computation-intensive. This operational constraint further limits the potential for high-precision computation, thereby affecting the overall performance and scalability of the recommender system.

Objective Collision. What optimization objectives correspond to “good” recommendation results are not well-defined, which leads to the following conflict:

1) Conflicts from Diverse Objectives: Beyond common optimization goals like click-through rate and watch time, there are competing goals (hundreds of goals in Kuaishou) from users, creators, and platform ecosystems. These objectives intervene at various stages of the system, gradually undermining system consistency and increasing complexity and operational inefficiency.

2) Cross-Stage Modeling Conflicts: Even when modeling similar objectives, conflicts can arise due to different structures and sizes of models at various stages. For instance, the effectiveness of the retrieval stage might be constrained by the limitations of the ranking model, which, in turn, could be affected by suboptimal upstream results. This highlights the need for a more unified optimization goal and model structure across the recommendation system to ensure coherence and efficiency.

Lag Behind AI Evolution. While remarkable progress has been made in LLM and visual language model (VLM) domains (e.g., scaling laws [^13] [^9] [^10], reinforcement learning [^36] [^18] [^20] [^26]), the existing cascaded recommendation framework presents fundamental architectural barriers to adopting these proven techniques. This structural misalignment creates a widening gap between recommendation systems and mainstream AI advancements, limiting potential performance gains from state-of-the-art approaches.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x2.png)

Figure 2: Comparison between a cascaded recommender system and the OneRec. The cascaded approach system typically involves stages such as retrieval, pre-ranking, and ranking, each potentially employing multiple strategies or models. In contrast, OneRec adopts an encoder-decoder architecture to generate user-preferred videos in an end-to-end manner under the guidance of a reward model.

To address the challenges faced by traditional cascaded recommendation architectures, we propose OneRec (See the bottom part of Figure 2), a novel recommendation system designed to overcome the limitations of cascade ranking systems by integrating retrieval and ranking processes into a single-stage encoder-decoder based generative framework. This approach exhibits the following characteristics:

$\vdash$ End-to-End Optimization: The system is designed to be both end-to-end and sufficiently simple to enable direct optimization for the final objective.

$\vdash$ Computational Efficiency: With a focus on computational intensity, the method rigorously optimizes computational utilization efficiency during both training and inference phases, thereby fully leveraging the benefits brought by computing power advancements.

Our new framework yields several significant findings:

- Through extensive infrastructure optimizations, we have achieved 23.7% and 28.8% MFU on flagship GPUs during training and inference, respectively — representing 5.2 $\times$ and 2.6 $\times$ improvements over the original ranking model — significantly narrowing the gap with the LLM community. More importantly, this end-to-end architecture dramatically reduces unnecessary communication and storage overhead, resulting in OPEX that is merely 10.6% of that associated with traditional complex recommendation pipelines. Currently, its deployment in the main scenarios of the Kuaishou/Kuaishou Lite APP manages approximately 25% of total QPS, delivering improvements of 0.54% and 1.24% in App Stay Time, while simultaneously improving all core metrics—including user engagement, video cold start, and distribution balance — demonstrating comprehensive performance gains.
- We have enhanced the computational FLOPs of the current recommendation model by 10 $\times$. Through this process, we have identified the scaling laws for recommendation systems. This discovery provides valuable insights into how recommendation system performance can be optimized as model size and computational resources are scaled, ensuring efficient and effective deployment in various operational contexts.
- Reinforcement learning (RL) techniques, which previously had shown limited impact in traditional architectures, now demonstrate substantial potential within our framework. We have conducted extensive experiments with both offline and online performance comparisons and have developed specific application practices tailored to meet real-world industrial iteration requirements. These implementations enable the system to leverage RL, resulting in improved adaptability and performance.

In the remainder of this paper, we first elaborate on the OneRec architecture (Section 2), detailing our tokenization pipeline for short videos, the encoder’s design for user interest modeling and compression, and scalable decoder optimization for precise output generation; we also introduce our reinforcement learning framework for recommendation optimization, discussing the impact of sampling space design, policy, and reward function on recommendation outcomes, along with empirical insights from production deployment. Next, we present the pre-training and post-training pipeline (Section 3), covering training data construction, hyperparameter configurations, and critical implementation discussions, followed by a description of the evaluation framework (Section 4), including offline metric systems and online performance/efficiency optimizations. Lastly, we conclude this work, discuss the existing limitations of OneRec, and propose potential directions for future research (Section 5).

## 2 Architecture

In this section, we present the OneRec architecture (as illustrated in the bottom part of Figure 2). The architecture first employs a tokenizer (Section 2.1) to convert videos into semantic IDs which serve as the prediction targets for the model. During the training phase, the encoder-decoder structure (Section 2.2 and Section 2.3) performs next token prediction to forecast target items, while simultaneously undergoing reinforcement learning alignment through the reward system (Section 2.4). In the inference phase, the model first generates semantic IDs and then maps these tokens back to video recommendations, with an optional reward-based selection step for further refinement.

### 2.1 Tokenizer

OneRec is a generative recommendation system at Kuaishou, while its billion-scale, ever-growing item space prevents generating atomic identifiers due to computational and architectural constraints. To resolve these, OneRec tokenizes items into coarse-to-fine semantic IDs using a reduced and fixed vocabulary, enabling knowledge transfer among similar items and better generalization to new items [^22]. However, prior solutions [^22] [^33] generate semantic IDs exclusively from context features, neglecting collaborative signals and yielding suboptimal reconstruction quality, as demonstrated in Section 4.4. Consequently, our solution integrates collaborative signals with multimodal features and then leverages RQ-Kmeans [^17] to generate higher-quality hierarchical semantic IDs.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x3.png)

Figure 3: Illustration of our tokenizer implementation. We first align multimodal representations of item pairs with high collaborative similarity to obtain collaborative multimodal representations, then tokenize these representations into discrete semantic IDs using RQ-Kmeans.

#### 2.1.1 Aligned Collaborative-Aware Multimodal Representation

We integrate multimodal content with collaborative signals by aligning multimodal representations of collaboratively similar item pairs, as shown in Figure 3 (left). Therefore, we require the preparation of multimodal representations, item pairs, and an alignment strategy:

- Multimodal Representations. We incorporate multimodal inputs for each video: the caption, tag, ASR (speech-to-text), OCR (image-to-text), the cover image, and 5 uniformly sampled frames. These inputs are processed using miniCPM-V-8B [^11], generating $N_{M}=1280$ token vectors $\mathbf{M}\in\mathbb{R}^{N_{M}\times d_{t}}$ ($d_{t}=512$). A Querying Transformer (QFormer) [^15] then compresses these tokens with $N_{\tilde{M}}=4$ learnable query tokens $\mathbf{Q}^{(1)}\in\mathbb{R}^{N_{\tilde{M}}\times d_{t}}$:
	$$
	\displaystyle\mathbf{Q}^{(i+1)}
	$$
	 
	$$
	\displaystyle=\text{CrossAttn}\left(\mathbf{Q}^{(i)},\mathbf{M},\mathbf{M}\right),
	$$
	$$
	\displaystyle\mathbf{Q}^{(i+1)}
	$$
	 
	$$
	\displaystyle=\text{FFN}(\text{RMSNorm}(\mathbf{Q}^{(i+1)}),\quad\text{for }i\in\{1,2,\ldots,N_{c}\},
	$$
	where $\tilde{\mathbf{M}}=\mathbf{Q}^{(N_{c}+1)}\in\mathbb{R}^{N_{\tilde{M}}\times d_{t}}$ denotes the compressed version of $\mathbf{M}$, and $N_{c}=4$ denotes the number of QFormer layers.
- Item Pairs. We construct high-quality item-pair dataset $\mathcal{D}_{pair}$ via: 1) User-to-Item Retrieval: For each user, we take a positively clicked target item and pair it with the most collaboratively similar item from the user’s latest historical positive clicks, and 2) Item-to-Item Retrieval: We pair items exhibiting high similarity scores (e.g., the Swing similarity) [^32].
- Item-to-Item Loss and Caption Loss. We introduce dual training objectives: 1) An item-to-item contrastive loss aligns representations of collaboratively similar video pairs $(i,j)\in\mathcal{D}_{pair}$, capturing behavioral patterns, and 2) a caption loss prevents hallucination by performing next-token prediction on video captions using LLaMA3 [^5] as the decoder, thereby preserving content understanding capabilities.
	$$
	\mathcal{L}_{I2I}=-\frac{1}{\left|\mathcal{B}\right|}\sum_{(i,j)\in\mathcal{B}}\log\frac{\exp\left(\text{sim}\left(\tilde{\mathbf{M}}_{i},\tilde{\mathbf{M}}_{j}\right)/\tau\right)}{\sum_{(i^{\prime},j^{\prime})\in\mathcal{B}}\exp\left(\text{sim}\left(\tilde{\mathbf{M}}_{i},\tilde{\mathbf{M}}_{j^{\prime}}\right)/\tau\right)},
	$$
	 
	$$
	\mathcal{L}_{caption\_gen}=-\sum_{k}\log P\left(t^{k+1}|\left[t^{1},t^{2},\cdots,t^{k}\right]\right),
	$$
	where $\tau$ denotes the temperature coefficient, $\text{sim}(\cdot,\cdot)$ denotes the similarity function, $\mathcal{B}$ denotes a batch of $\mathcal{D}_{pair}$, $t^{k}$ denotes the $k$ -th caption token.

#### 2.1.2 Tokenization

We utilize RQ-Kmeans [^17] for tokenization, which employs residual quantization to generate semantic IDs in a coarse-to-fine manner. This method constructs codebooks by applying K-means clustering directly on the residuals. An illustration of the RQ-Kmeans process is provided in Figure 3 (right).

Formally, the initial residual at layer $l=1$ is defined as:

$$
\mathcal{R}^{(1)}=\left\{\tilde{\mathbf{M}}_{i}\in\mathbb{R}^{N_{\tilde{M}}\times d_{t}}\mid\forall\text{ video }i\right\}.
$$

For each layer $l$, the codebook $\mathcal{C}^{(l)}$ is derived from K-means centroids of $\mathcal{R}^{(l)}$:

$$
\mathcal{C}^{(l)}=\text{K-means}\left(\mathcal{R}^{(l)},N_{t}\right),
$$

where $\mathcal{C}^{(l)}=\left\{\bm{c}^{(l)}_{k}\in\mathbb{R}^{N_{\tilde{M}}\times d_{t}}\mid k=1,\dots,N_{t}\right\}$ and $N_{t}$ is the codebook size. The nearest centroid index for item $i$ is computed as:

$$
s^{l}_{i}=\arg\min_{k}\left\|\mathcal{R}^{(l)}_{i}-\bm{c}^{(l)}_{k}\right\|,
$$

where $\|\cdot\|$ denotes the Euclidean norm. The residual of video $i$ for layer $l+1$ is then updated:

$$
\mathcal{R}^{(l+1)}_{i}=\mathcal{R}^{(l)}_{i}-\bm{c}^{(l)}_{s^{l}_{i}}.
$$

This quantization iterates across $L_{t}=3$ layers.

As demonstrated in Section 4.4, RQ-Kmeans offers enhanced reconstruction quality, better codebook utilization, and improved balance compared to the widely used RQ-VAE [^22] [^14]. At this stage, each video $m$ can be represented by $L_{t}$ coarse-to-fine semantic identifiers: $\{s^{1}_{m},s^{2}_{m},\ldots,s^{L_{t}}_{m}\}$, which will serve as the output of the OneRec recommendation system, enabling progressive item generation.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x4.png)

Figure 4: Illustration of our encoder-decoder architecture.

### 2.2 Encoder

#### 2.2.1 Multi-Scale Feature Engineering

This section presents the feature engineering component of OneRec. We process user behavior data through four specialized embedding pathways, each designed to capture distinct scales of user interaction patterns: user static pathway, short-term pathway, positive-feedback pathway, and lifelong pathway.

##### User Static Pathway

The user static pathway generates a compact representation of core user characteristics, incorporating user identifier (uid), age (age), gender (gender), etc., which is then transformed into the model’s hidden dimension:

$$
\displaystyle\mathbf{f}_{u}
$$
 
$$
\displaystyle=[\mathbf{e}_{\text{uid}};\mathbf{e}_{\text{gender}};\mathbf{e}_{\text{age}};\cdots],
$$
$$
\displaystyle\mathbf{h}_{u}
$$
 
$$
\displaystyle=\text{Dense}(\text{LeakyReLU}(\text{Dense}(\mathbf{f}_{u}))).
$$

where $\mathbf{e}_{\text{uid}},\mathbf{e}_{\text{gender}},\mathbf{e}_{\text{age}}\in\mathbb{R}^{64}$ and $\mathbf{h}_{u}\in\mathbb{R}^{1\times d_{\text{model}}}$.

##### Short-term Pathway

The short-term behavior pathway processes the most recent ($L_{s}=20$) user interactions, incorporating video identifier (which can be represented as video identifiers vid or semantic identifiers sid as described in Section 2.1.2, we will discuss these two representation approaches in Section 4.2.2.), author identifiers (aid), tags (tag), timestamps (ts), playtime (playtime), duration (dur), labels (label, user interactions with each video, including like, follow, forward, dislike, comment, profile entry, etc.) This pathway produces representations that capture immediate user preferences and contextual factors influencing current behavior patterns:

$$
\displaystyle\mathbf{f}_{s}
$$
 
$$
\displaystyle=[\mathbf{e}_{\text{vid}}^{s};\mathbf{e}_{\text{aid}}^{s};\mathbf{e}_{\text{tag}}^{s};\mathbf{e}_{\text{ts}}^{s};\mathbf{e}_{\text{playtime}}^{s};\mathbf{e}_{\text{dur}}^{s};\mathbf{e}_{\text{label}}^{s}],
$$
$$
\displaystyle\mathbf{h}_{s}
$$
 
$$
\displaystyle=\text{Dense}(\text{LeakyReLU}(\text{Dense}(\mathbf{f}_{s}))),
$$

The feature dimensions are organized as follows: video embeddings $\mathbf{e}_{\text{vid}}^{s}$ match the model dimension $d_{\text{model}}$, author embeddings $\mathbf{e}_{\text{aid}}^{s}$ use 512 dimensions, while all remaining features employ 128 dimensions. All features span $L_{s}$ sequence positions, yielding the final representation $\mathbf{h}_{s}\in\mathbb{R}^{L_{s}\times d_{\text{model}}}$.

##### Positive-feedback Pathway

The positive-feedback behavior pathway operates on a sequence of high-engagement interactions ($L_{p}=256$). The pathway maintains the established dimensional structure:

$$
\displaystyle\mathbf{f}_{p}
$$
 
$$
\displaystyle=[\mathbf{e}_{\text{vid}}^{p};\mathbf{e}_{\text{aid}}^{p};\mathbf{e}_{\text{tag}}^{p};\mathbf{e}_{\text{ts}}^{p};\mathbf{e}_{\text{playtime}}^{p};\mathbf{e}_{\text{dur}}^{p};\mathbf{e}_{\text{label}}^{p}],
$$
$$
\displaystyle\mathbf{h}_{p}
$$
 
$$
\displaystyle=\text{Dense}(\text{LeakyReLU}(\text{Dense}(\mathbf{f}_{p}))).
$$

All features span $L_{p}$ sequence positions, yielding the final representation $\mathbf{h}_{p}\in\mathbb{R}^{L_{p}\times d_{\text{model}}}$.

##### Lifelong Pathway

The lifelong behavior pathway is designed to process ultra-long user interaction histories with sequences of up to 100,000 videos. Directly applying attention mechanisms to such sequences is computationally prohibitive. This pathway employs a two-stage hierarchical compression strategy inspired by our previous work [^29].

Behavior Compression Using the multimodal content representations described in Section 2.1.1, we perform hierarchical K-means clustering on each user’s interaction sequence. To balance computational efficiency and model effectiveness, we dynamically adjust the number of clusters by setting the cluster count for each step to $\lfloor\sqrt[3]{|D|}\rfloor$, where $|D|$ is the number of items in the current data. This is an empirically determined setting. The clustering process terminates when the number of items in the current cluster does not exceed a preset threshold $M$. Upon termination, we select the item closest to each cluster center as the representative of that cluster.

Feature Aggregation For each cluster, we construct representative features by handling discrete and continuous attributes differently. For sparse categorical features such as vid, aid, and label, we directly inherit the features from the representative video (i.e., the video closest to the cluster center). For continuous features such as tag, ts, playtime, and duration, we compute the average values across all videos within the cluster to capture collective behavioral patterns.

For the user’s long-term historical sequence ($L_{l}=2000$), each video is replaced by the features of its corresponding cluster representative:

$$
\displaystyle\mathbf{f}_{l}
$$
 
$$
\displaystyle=[\mathbf{e}_{\text{vid}}^{l};\mathbf{e}_{\text{aid}}^{l};\mathbf{e}_{\text{tag}}^{l};\mathbf{e}_{\text{ts}}^{l};\mathbf{e}_{\text{playtime}}^{l};\mathbf{e}_{\text{dur}}^{l};\mathbf{e}_{\text{label}}^{l}],
$$
$$
\displaystyle\mathbf{v}_{l}
$$
 
$$
\displaystyle=\text{Dense}(\text{LeakyReLU}(\text{Dense}(\mathbf{f}_{l}))).
$$

The final representation $\mathbf{v}_{l}\in\mathbb{R}^{L_{l}\times d_{\text{model}}}$. The lifelong pathway compresses historical sequences through QFormer, where learnable query vectors $\mathbf{h}_{l}^{(0)}\in\mathbb{R}^{N_{q}\times d_{\text{model}}}$ ($N_{q}=128$) attend to the processed historical features:

$$
\displaystyle\mathbf{h}_{l}^{(i+1)}
$$
 
$$
\displaystyle=\text{CrossAttn}(\mathbf{h}_{l}^{(i)},\mathbf{v}_{l},\mathbf{v}_{l}),
$$
$$
\displaystyle\mathbf{h}_{l}^{(i+1)}
$$
 
$$
\displaystyle=\text{FFN}(\text{RMSNorm}(\mathbf{h}_{l}^{(i+1)})).
$$

Followed by $N_{l}=2$ blocks, we obtain the compressed lifelong feature representation $\mathbf{h}_{l}=\mathbf{h}_{l}^{(N_{l})}\in\mathbb{R}^{N_{q}\times d_{\text{model}}}$.

#### 2.2.2 Encoder Architecture

As illustrated in Figure 4, the encoder architecture of OneRec integrates multi-scale user behavior representations through a unified transformer-based framework. The encoder concatenates the outputs from the four multi-scale pathways to form a comprehensive input sequence:

$$
\mathbf{z}^{(1)}=[\mathbf{h}_{u};\mathbf{h}_{s};\mathbf{h}_{p};\mathbf{h}_{l}]+\mathbf{e}_{\text{pos}}
$$

where $\mathbf{e}_{\text{pos}}\in\mathbb{R}^{(1+L_{s}+L_{p}+N_{q})\times d_{\text{model}}}$ represents learnable positional embeddings. The integrated representation is processed through $L_{\text{enc}}$ transformer encoder layers, each consisting of fully visible self-attention mechanisms followed by feed-forward networks with RMS normalization:

$$
\displaystyle\mathbf{z}^{(i+1)}
$$
 
$$
\displaystyle=\mathbf{z}^{(i)}+\text{SelfAttn}(\text{RMSNorm}(\mathbf{z}^{(i)})),
$$
$$
\displaystyle\mathbf{z}^{(i+1)}
$$
 
$$
\displaystyle=\mathbf{z}^{(i+1)}+\text{FFN}(\text{RMSNorm}(\mathbf{z}^{(i+1)})).
$$

The final encoder output $\mathbf{z}_{\text{enc}}=\mathbf{z}^{(L_{enc}+1)}\in\mathbb{R}^{(1+L_{s}+L_{p}+N_{q})\times d_{\text{model}}}$ provides a holistic multi-scale user behavior representation, serving as the foundation for subsequent recommendation generation.

### 2.3 Decoder

OneRec adopts a point-wise generation paradigm during the decoding phase. For each target video $m$, the decoder input sequence is constructed by concatenating a learnable beginning-of-sequence token with the video’s semantic identifiers:

$$
\displaystyle\mathcal{S}_{m}
$$
 
$$
\displaystyle=\left\{s_{[\mathrm{BOS}]},s^{1}_{m},s^{2}_{m},\cdots,s^{L_{t}}_{m}\right\},
$$
$$
\displaystyle\mathbf{d}^{(0)}_{m}
$$
 
$$
\displaystyle=\text{Emb\_lookup}(\mathcal{S}_{m}).
$$

The decoder processes this sequence through $L_{\text{dec}}$ transformer layers. Each layer performs sequential operations:

$$
\displaystyle\mathbf{d}^{(i+1)}_{m}
$$
 
$$
\displaystyle=\mathbf{d}^{(i)}_{m}+\text{CausalSelfAttn}(\mathbf{d}^{(i)}_{m}),
$$
$$
\displaystyle\mathbf{d}^{(i+1)}_{m}
$$
 
$$
\displaystyle=\mathbf{d}^{(i+1)}_{m}+\text{CrossAttn}(\mathbf{d}^{(i+1)}_{m},\mathbf{Z}_{\text{enc}},\mathbf{Z}_{\text{enc}}),
$$
$$
\displaystyle\mathbf{d}^{(i+1)}_{m}
$$
 
$$
\displaystyle=\mathbf{d}^{(i+1)}_{m}+\text{MoE}(\text{RMSNorm}(\mathbf{d}^{(i+1)}_{m})).
$$

Each decoder layer incorporates a Mixture of Experts (MoE) feed-forward network to enhance model capacity while maintaining computational efficiency. The MoE layer employs $N_{\text{experts}}$ expert networks with a top- $k$ routing strategy:

$$
\displaystyle\text{MoE}(\mathbf{x})=\sum_{j=1}^{k}\text{Gate}_{j}(\mathbf{x})\cdot\text{Expert}_{j}(\mathbf{x}),
$$

where $\text{Gate}_{j}(\mathbf{x})$ represents the gating weights determined by the routing mechanism, and $\text{Expert}_{j}(\mathbf{x})$ denotes the output of the $j$ -th selected expert network. To ensure balanced expert utilization without introducing interference gradients, we implement a loss-free load balancing strategy following [^16].

The model is trained using cross-entropy loss for next-token prediction on the semantic identifiers of target video $m$:

$$
\mathcal{L}_{\mathrm{NTP}}=-\sum_{j=0}^{L_{t}-1}\log P\left(s_{m}^{j+1}\mid\left[s_{[\mathrm{BOS}]},s^{1}_{m},s^{2}_{m},\cdots,s^{j}_{m}\right]\right)
$$

### 2.4 Reward System

The pre-trained model only fits the distribution of the exposed item space through next token prediction, and the exposed items are obtained from the past traditional recommendation system. This results in the model being unable to break through the ceiling of traditional recommendations. To address this issue, we introduce preference alignment based on a reward system, using on-policy reinforcement learning to train the model in the generated item space. Through rewards, the model perceives more fine-grained preference information. We introduce the preference reward to align user preferences, the format reward to ensure the generation format is as legal as possible, and the specific industrial reward to align with some special industrial scenario needs.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x5.png)

Figure 5: Overall Framework of the Reward System. The Reward System is composed of three parts. They assign Preference Reward (P-Score), Format Reward, and specific Industrial Reward to the videos generated by the model, respectively.

#### 2.4.1 User Preference Alignment

In recommendation systems, defining a "good recommendation" is much more challenging than determining the correctness of a mathematical solution. Traditional approaches [^31] [^3] often define multiple objectives, such as clicks, likes, comments, and watch time, which are then combined into a score through a weighted fusion of the predicted values (xtr) for each objective. However, manually tuning these fusion weights is challenging, not only lacking accuracy but also lacking personalization, and often results in optimization conflicts between objectives.

To address these limitations, we propose using a neural network to learn a personalized fusion score, referred to as P-Score (Preference Score) [^2]. The overall framework of this model is illustrated in Figure 5 (middle). The model’s underlying architecture is based on the Search-based Interest Model (SIM) [^19]. It includes multiple towers, each dedicated to learning specific objectives. During training, these towers compute binary cross-entropy (BCE) loss using the corresponding objective labels as auxiliary tasks. The hidden states of each tower, along with user and item representations, are fed into the final layer’s Multi-Layer Perceptron (MLP). This MLP is followed by a single tower outputting the P-Score, which computes binary cross-entropy loss using the labels of all objectives.The loss can be formally represented as follows:

$$
\mathcal{L}_{\mathrm{P\text{-}Score}}=\sum_{xtr\in S_{o}}w^{\mathrm{xtr}}\mathcal{L}_{\mathrm{P\text{-}Score}}^{\mathrm{xtr}}
$$
 
$$
\mathcal{L}_{\mathrm{P\text{-}Score}}^{\mathrm{xtr}}=-(y^{\mathrm{xtr}}\log{p}+(1-y^{\mathrm{xtr}})\log{(1-p)}),\\
$$
 
$$
S_{o}=\{\mathrm{ctr,lvtr,ltr,vtr,...\}}
$$

We adjust the value of $w^{\mathrm{xtr}}$ to bias the P-Score towards each objective, ultimately achieving an improvement in AUC across all objectives. This method allows the model to receive specific user information and adjust the Preference Score for that user appropriately, without compromising the experience of other users. Compared to the previous approach of indiscriminate weighted summation, this method is more likely to achieve Pareto optimization. Therefore, we use the P-Score obtained by this method as the reward for preference alignment.

##### Early Clipped GRPO

In this section, we introduce how to use the Preference Score to align user preferences. We use ECPO (Early Clipped GRPO) for optimization. Specifically, for a user $u$, we generate $G$ items using the old policy model. Each item, along with the user, is input into the Preference Reward Model to obtain the P-Score as reward $r_{i}$. The optimization objective is as follows:

$$
\mathcal{J}_{ECPO}(\theta)=\mathbb{E}_{u\sim P(U),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}}\left[\frac{1}{G}\sum_{i=1}^{G}\text{min}\left(\frac{\pi_{\theta}(o_{i}|u)}{\pi_{\theta_{old}}^{\prime}(o_{i}|u)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i}|u)}{\pi_{\theta_{old}}^{\prime}(o_{i}|u)},1-\epsilon,1+\epsilon\right)A_{i}\right)\right],
$$
 
$$
A_{i}=\frac{r_{i}-\text{mean}(\{r_{1},r_{2},...,r_{G}\})}{\text{std}(\{r_{1},r_{2},...,r_{G}\})},
$$
 
$$
\pi_{\theta_{old}}^{\prime}(o_{i}|u)=\text{max}\left(\frac{\text{sg}(\pi_{\theta}(o_{i}|u))}{1+\epsilon+\delta},\pi_{\theta_{old}}(o_{i}|u)\right),\quad\delta>0,
$$

where sg represents the stop gradient operation and $\delta$ is a hyperparameter greater than 0.

We make a modification to GRPO (Group Policy Relative Optimization) [^16] to make its training process more stable. The illustration is presented in Figure 6. In the original GRPO, a large policy ratio ($\pi_{\theta}/\pi_{\theta_{old}}$) is allowed for negative advantages, which can easily lead to gradient explosion. Therefore, we preemptively clip policies with large ratios to ensure training stability while still allowing corresponding negative advantages to take effect. The larger the $\delta$, the larger the tolerable policy ratio, which means the larger the tolerable gradient. This can be determined based on actual needs. In OneRec, we set $\delta$ to 0.1, which indicates that the ratio of policies with negative advantages is allowed to slightly exceed $1+\epsilon$. We remove the KL divergence loss because the Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) are trained together in OneRec, and the SFT loss ensures the model remains stable.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x6.png)

Figure 6: Illustration of ECPO. The x -axis is π θ / o l d \\pi\_{\\theta}/\\pi\_{\\theta\_{old}} and the y -axis is the clipped. Items with A > 0 A>0 are processed in the same way as the original GRPO, while items with < A<0 are constrained by early-clipping to limit the maximum ratio.

#### 2.4.2 Generation Format Regularization

In generative recommendation, the legality ratio refers to the proportion of generated semantic ID sequences that can be mapped to actual item IDs. This metric is crucial for assessing the stability of generation. In practice, the cardinality of semantic ID sequences $N_{t}^{L_{t}}$ is much larger than that of videos. This ensures that all items are covered, and a larger vocabulary introduces more parameters, leading to better performance. However, this may also result in generating semantic ID sequences without corresponding item IDs during inference, i.e., illegal generation.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x7.png)

Figure 7: Illustration of squeezing effect. π θ P T \\pi\_{\\theta\_{PT}} represents the pre-trained model, while R L \\pi\_{\\theta\_{RL}} represents the model trained with ECPO. o + o^{+} refers to videos with positive advantages, while − o^{-} refers to those with negative advantages.

Introducing reinforcement learning with ECPO significantly increases the generation of illegal outputs. Recent work [^23] suggests that this is due to the squeezing effect caused by negative advantages. As shown in Figure 7, the pre-trained model has learned to generate most of the legal tokens. After incorporating RL, items with $A>0$ only slightly adjust the distribution. When an item with $A<0$ is applied, the model’s probability distribution compresses most of the probability mass into what it currently considers the optimal output $o^{*}$. This results in the probabilities of some legal tokens being squeezed to levels comparable to those of illegal tokens, making it difficult for the model to distinguish legal tokens.

To address this issue, we propose incorporating a format reward in reinforcement learning to encourage the model’s legal generation. Specifically, we randomly select $K$ samples from the $G$ samples for legality reinforcement learning. For legal samples, we set the advantage to 1, and for illegal samples, we discard them directly to avoid the squeezing effect.

$$
A_{i}=\begin{cases}1&\text{if }o_{i}\in I_{\text{legal}}\\
0&\text{if }o_{i}\notin I_{\text{legal}}\end{cases}
$$

The optimization objective formulation is the same as the ECPO (Equation 32) and we directly use $A_{i}$ as advantages.

#### 2.4.3 Industrial Scenario Alignment

In industrial scenarios, the recommendation system needs to consider not only user preferences but also various other aspects. For example, at Kuaishou, the ecosystem of the video community, commercialization needs, and the delivery of cold-start and long-tail videos. Traditional recommendation systems attempt to address these issues by applying algorithms or strategies at one stage of the recommendation pipeline. Due to inconsistencies across different stages, this can easily lead to a recurring cycle of unexpected problems emerging alternately. Engineers are forced to constantly make adjustments through patching, resulting in a bloated system over time that hinders iteration. In OneRec, we only need to incorporate optimization objectives into the reward system and adopt reinforcement learning to perform targeted optimization. This approach is not only convenient but also allows for end-to-end implementation, maintaining system consistency. We will provide an example of optimization practice in Section 4.3.3.

## 3 Training Framework

### 3.1 Training Infrastructure

In this section, we describe our hardware and infrastructure that facilitated the large-scale pre-training of OneRec and introduce several optimizations that enhance training efficiency.

Compute. We utilize 90 servers for training, each equipped with 8 flagship GPUs and 2 CPUs interconnected via 400Gbps NVLink to ensure high-speed intra-node bandwidth.

Networking. Intra-node communication is managed by the efficient NVLink network, while inter-node communication is supported by 400Gbps RDMA for training traffic and 100Gbps TCP for training data and embedding prefetching operations.

Storage. Each server is equipped with 4 NVMe SSDs to expedite checkpoint writes, allowing for the storage of large-scale embedding parameters and dense parameters in HDFS with minimal downtime for fault tolerance.

Training Acceleration. For training acceleration, several core optimizations are implemented:

1) Embedding Acceleration: To manage the extensive embedding workload beyond CPU capacity, we utilize Kuaishou’s SKAI framework for GPU-based parameter servers. This framework leverages cross-GPU unified embedding tables, GPU caching paradigms, and prefetching pipelines to enhance training efficiency and reduce management overhead.

2) Training Parallelism: A combination of data parallelism, ZERO1 [^21], and gradient accumulation is employed for model training. ZERO1 is selected because the current model’s dense parameters can be loaded on a single GPU, minimizing synchronization overhead in data parallel groups when interleaving multiple macro batches.

3) Mixed Precision Training: BFloat16 is used for computations in certain MLP networks to optimize performance.

4) Compilation Optimization: For attention networks, compilation optimizations are applied to reduce computational overhead.

Thanks to advancements in highly optimized training infrastructure, the model’s training MFU has improved to 23.7%, significantly narrowing the gap with the LLM training efficiency.

### 3.2 Pre-training

##### Pre-Training Data

As illustrated in Section 2.2.1, our model takes multi-scale user behavior representations as input. The pre-training objective involves predicting sequences of target items for users. Each training sample comprises a target item which is tokenized into 3 semantic identifiers. This tokenization scheme results in 3 target tokens per training sample for the generative model’s next-token prediction task. Our training pipeline processes approximately 18 billion samples daily, yielding a throughput of 54 billion tokens per day. The OneRec-0.935B model (detailed in Table 1) achieves convergence after training on approximately 100 billion samples, corresponding to a total exposure of 300 billion tokens during pre-training.

##### Key Hyperparameters

The OneRec series comprises four models (two dense and two MoE variants) designed for recommendation tasks. Key architectural hyperparameters such as layer counts, hidden dimensions, and attention head numbers are detailed in Table 1. In these models, encoders and decoders have the same number of layers. For dense variants, the standard Feed-Forward Networks (FFNs) typically expand the hidden dimension $d_{\text{ff}}$ to $2\times d_{\text{model}}$. For the MoE variants, we replace standard FFNs with MoE layers in designated blocks, and employ SwiGLU FFNs [^30] [^27] as experts. Consistent with open-source MoE LLM settings [^12] [^6], the hidden dimension for each SwiGLU expert is calculated as $\frac{2}{3}\times 4\times d_{\text{model}}$, ensuring it is a multiple of 128.

The convergence curves for each model can be found in Section 4.2.1.

Table 1: OneRec model architectures. "Layers" = #Encoder + #Decoder. "FFN Hid. Dim" is FFNs’ intermediate size or MoEs’ intermediate expert size.

| Model | Layers | Hid. Dim | FFN Hid. Dim | Attn. Heads | Experts (Tot/Act) | MoE Loc. |
| --- | --- | --- | --- | --- | --- | --- |
| OneRec-0.015B (Dense) | 4 | 128 | 256 | 4 | N/A | N/A |
| OneRec-0.121B (Dense) | 8 | 1024 | 2048 | 8 | N/A | N/A |
| OneRec-0.935B (MoE) | 8 | 1024 | 2048 | 8 | 24 / 2 | Decoder |
| OneRec-2.633B (MoE) | 24 | 1024 | 2048 | 8 | 24 / 4 | Enc & Dec |

### 3.3 Post-training

In the post-training phase, we perform online training using real-time data streams. We simultaneously perform Reject Sampling Fine-Tuning (RSFT) and Reinforcement Learning (RL). For RSFT, we filter out the bottom 50% of exposure sessions based on play duration. The training loss is the same as the $\mathcal{L}_{\rm NTP}$ loss in the pre-training process, but we apply annealing by reducing the learning rate of sparse parameters to $1\times 10^{-4}$ and dense parameters to $8\times 10^{-5}$. For RL, we randomly select 1% of users from the RSFT data to generate RL samples.

To maximize computational resource utilization, we decouple the generation of RL samples from the training process by using an external inference service. During training, 1% of users access the external service to generate 512 items, request rewards for each item from the reward model, and then return the data to the training task. The training task sends updated parameters to the external inference service via a Message Queue (MQ) every 1000 steps. The overall post-training process is summarized in Figure 8.

Figure 8: The overall process of OneRec’s post-training, including continual pre-training and reinforcement learning.

## 4 Evaluation

### 4.1 Evaluation Metric

We assess model performance through the following metrics:

- Cross-entropy loss: Next-token prediction loss $\mathcal{L}_{\mathrm{NTP}}$ curves.
- P (preference)-Score: Learned comprehensive evaluation metric, as detailed in Section 2.4.1.
- xtr metrics: A set of user engagement indicators derived from a pre-trained ranking model [^31] [^3] currently deployed in our system, including:
	- lvtr (Long View Through Rate): Predicted probability of significant video viewing
	- vtr (View Through Rate): Predicted probability of video viewing
	- ltr (Like Through Rate): Predicted probability of video liking
	- wtr (Follow Through Rate): Predicted probability of the creator following
	- cmtr (Comment Through Rate): Predicted probability of video commenting

For P-Score and xtr reward metrics, our evaluation system operates on streaming data where values may vary across different periods. Consequently, identical metrics may show different absolute values across experiments due to temporal variations in the data stream. However, we ensure reliable evaluation by conducting comparative experiments within the same periods and averaging results over sufficiently long observation windows, making our findings statistically confident.

### 4.2 Scaling

#### 4.2.1 Training Scaling

##### Parameters Scaling

The OneRec series includes models of varying sizes: OneRec-0.015B, OneRec-0.121B, OneRec-0.935B, and OneRec-2.633B, as detailed in Table 1. We investigated the impact of model parameter count on performance. Figure 9 illustrates the loss curves for these models, demonstrating a clear scaling trend where larger models achieve lower loss as training progresses. This indicates a strong capability for performance improvement with increased model size.

Regarding the influence of training data size, our experiments show that performance converges rapidly within the initial approximately 10 billion samples. While the rate of improvement diminishes significantly beyond this point, performance does not completely plateau and continues to benefit, albeit more slowly, from additional data (i.e., beyond 100 billion samples). This suggests that while substantial gains are achieved early in training, further, more gradual improvements are possible with larger datasets.

![Refer to caption](https://arxiv.org/html/2506.13695v4/Figures/loss_curve_comparison4.png)

Figure 9: Comparison of loss curves for different OneRec model sizes, showing loss scaling with training samples.

As model parameters scale up, load balancing among experts becomes a critical issue. Uneven expert utilization can lead to training inefficiency and suboptimal performance. We adopt DeepSeek’s loss-free load balancing strategy [^16], which maintains expert utilization balance without introducing additional loss terms. With this strategy, we observe a loss reduction of 0.2, demonstrating its effectiveness in improving convergence for scaled OneRec models.

Beyond parameter scaling, we conduct additional experiments to validate the effectiveness of scaling across other key dimensions using our 0.935B model. These experiments encompass feature scaling (examining the impact of comprehensive feature engineering), codebook scaling (investigating the effect of vocabulary size expansion), and inference scaling (analyzing the influence of beam search parameters). Each dimension demonstrates distinct scaling behaviors and provides valuable insights for future model optimization.

##### Feature Scaling

To investigate the impact of feature engineering on model performance, we compare the model with two input configurations: a baseline using only item ID vid embeddings from 256 positive-feedback items, and an enhanced version incorporating the comprehensive feature set described in our methodology. As shown in Figure 10 and Table 10, the enhanced model with additional features achieves lower training loss and substantial improvements across multiple dimensions of recommendation quality.

![Refer to caption](https://arxiv.org/html/2506.13695v4/Figures/feature_scaling.png)

Figure 10: Training loss comparison with and without additional features.

##### Codebook Scaling

To investigate the impact of codebook size on model performance, we experiment by expanding the codebook from 8,192 to 32,768. It is important to note that NTP loss, as defined in our parameter scaling experiments, cannot be directly used for comparison here. This is because an increase in codebook size inherently expands the candidate set for the cross-entropy loss calculation, rendering direct loss comparisons misleading. Consequently, we evaluate performance using reward-based metrics. The performance improvements across various metrics are presented in Table 4.2.1. As shown in the result, increasing the codebook size yields significant improvements in playtime metrics and a slight gain in interaction metrics.

##### Infer Scaling

We investigate the impact of different numbers of generated items in inference (Pass@K) on model performance. As detailed in Table 4.2.1, increasing K of Pass@K from 8 to 512 results in consistent performance improvements across all evaluated metrics. However, further increasing K from 512 to 1,024 yields only marginal gains. Considering the trade-off between performance improvements and the associated computational resource consumption, we select K=512 for deployment in our production environment.

| Metric | Size=8K | Size=32K | Impr. |
| --- | --- | --- | --- |
| lvtr | 0.5118 | 0.5245 | 2.48% |
| vtr | 0.9384 | 0.9491 | 1.14% |
| ltr | 0.0298 | 0.0299 | 0.34% |
| wtr | 0.0153 | 0.0154 | 0.65% |
| cmtr | 0.0650 | 0.0664 | 2.15% |
| P-score | 0.2516 | 0.2635 | 4.75% |

Table 3: Codebook Scaling.

| Metric | Pass@8 | Pass@64 | Pass@512 | Pass@1024 | Impr. |
| --- | --- | --- | --- | --- | --- |
| lvtr | 0.3675 | 0.4927 | 0.5351 | 0.5443 | 48.11% |
| vtr | 0.9444 | 0.9462 | 0.9513 | 0.9530 | 0.91% |
| ltr | 0.0278 | 0.0346 | 0.0425 | 0.0452 | 62.59% |
| wtr | 0.0114 | 0.0138 | 0.0182 | 0.0197 | 72.81% |
| cmtr | 0.0350 | 0.0566 | 0.0809 | 0.0891 | 154.57% |
| P-score | 0.0811 | 0.2051 | 0.3375 | 0.3859 | 376.10% |

Table 4: Inference Pass@K Scaling.

#### 4.2.2 Semantic Identifier Input Representation

As model sizes scale to billions of parameters, we explore an alternative input representation strategy that leverages video semantic identifiers for user interaction histories instead of constructing separate sparse embeddings for video identifiers (vid). This semantic identifier input achieves performance comparable to traditional sparse embedding methods, while offering significant advantages in parameter efficiency, communication overhead, and sequence processing capacity that make it particularly promising for further scaling exploration.

##### Scaling Performance Analysis

As shown in Figure 11, our empirical analysis reveals that at scale (2.6B parameters), the semantic identifier input approach achieves performance comparable to or exceeding traditional sparse embedding methods.

![Refer to caption](https://arxiv.org/html/2506.13695v4/Figures/sid_vs_pid.png)

Figure 11: Training loss comparison between OneRec-2.633B with semantic identifier input and sparse embedding input.

##### Advantages and Future Scaling

The semantic identifier approach provides several key advantages over traditional sparse embedding methods, making it particularly attractive for further scaling exploration:

- Parameter Efficiency: By sharing embeddings between input and output representations, the model eliminates the need for separate sparse embedding tables for vid. This dramatically reduces the total parameter count, particularly for Kuaishou with billions of items.
- Communication Efficiency: In distributed training environments, sparse embedding operations require extensive parameter server communication for embedding lookup and gradient updates. The semantic identifier approach reduces communication overhead by leveraging dense operations and shared vocabulary, leading to faster training throughput and reduced communication bottlenecks.
- Extended Sequence Capacity: The elimination of large sparse embedding tables enables the allocation of computational resources toward processing longer user interaction sequences. This allows the model to capture more comprehensive user preference evolution patterns, potentially extending sequence lengths from thousands to tens of thousands of interactions.
- Representation Consistency: Sharing the same semantic space between input and output ensures representational consistency and enables the model to learn more coherent item-to-item relationships. This unified representation has the potential to facilitate better generalization across different recommendation scenarios.

Given these compelling advantages and the competitive performance demonstrated at the 2.6B parameter scale, we are actively pursuing further scaling exploration based on semantic identifier input representation. This approach promises to unlock new possibilities for large-scale recommendation systems while maintaining computational efficiency and architectural elegance.

### 4.3 Reinforcement Learning

#### 4.3.1 User Preference Alignment

Defining what constitutes a "good" recommendation has always been a challenging task. To rigorously verify RL’s impact, we use the single-objective vtr (view-through rate) as the reward, which corresponds to online metrics such as Watch Time and App Stay Time. The reported online results are relative improvements compared to Kuaishou’s traditional recommendation system, referred to as the overall baseline. Relative Impr. in the table indicates the relative enhancement of the latter group over the former group.

Notably, while using vtr as the reward can significantly improve duration metrics, it does not necessarily indicate a high-quality recommendation, as other metrics, such as Video View, which represent the number of videos viewed, may decrease significantly. We primarily focus on Watch Time and App Stay Time to find the optimal RL setting, and ultimately use it to validate the benefits of the P-Score reward.

##### Sampling Efficiency

Reinforcement learning optimizes the probability distribution of sampled items to increase the likelihood of selecting high-reward items, thereby significantly enhancing sampling efficiency. To quantify this effect, we conduct multi-point sampling experiments at pass@32, pass@128, and pass@512, with results summarized in Table 6. Treating the model without RL as the baseline, we define the improvement in app stay time as the sampling efficiency gap. Notably, RL shows the most substantial improvement gap at pass@32, indicating that the accuracy of top-ranked items is significantly enhanced. This improvement is crucial for reducing sampling overhead, as it ensures high precision when sampling a small number of items. In recommendation systems, balancing cost and benefit is essential, and the enhanced accuracy at lower sample numbers $K$ provides a solid foundation for achieving this balance.

<table><tbody><tr><td></td><td>Method</td><td>vtr</td><td>Watch time</td><td>App Stay Time</td><td>Video View <sup>1</sup></td></tr><tr><td rowspan="3">Pass@32</td><td>OneRec w/o RL</td><td>0.1978</td><td>+1.62%</td><td>-0.10%</td><td>-4.18%</td></tr><tr><td>OneRec w/ RL</td><td>0.2138</td><td>+3.17%</td><td>+0.39%</td><td>-9.87%</td></tr><tr><td>Relative Impr.</td><td>+8.08%</td><td>+1.55%</td><td>+0.49% <math><semantics><mrow><mo>↑</mo> <mo>⁣</mo> <mo>↑</mo> <mo>⁣</mo> <mo>↑</mo></mrow> <annotation>\uparrow\uparrow\uparrow</annotation></semantics></math></td><td>-3.69%</td></tr><tr><td rowspan="3">Pass@128</td><td>OneRec w/o RL</td><td>0.2239</td><td>+4.61%</td><td>+1.11%</td><td>-12.75%</td></tr><tr><td>OneRec w/ RL</td><td>0.2387</td><td>+5.22%</td><td>+1.49%</td><td>-15.06%</td></tr><tr><td>Relative Impr.</td><td>+6.61%</td><td>+1.53%</td><td>+0.38% <math><semantics><mrow><mo>↑</mo> <mo>↑</mo></mrow> <annotation>\uparrow\uparrow</annotation></semantics></math></td><td>-2.65%</td></tr><tr><td rowspan="3">Pass@512</td><td>OneRec w/o RL</td><td>0.2444</td><td>+6.32%</td><td>+1.66%</td><td>-15.54%</td></tr><tr><td>OneRec w/ RL</td><td>0.2494</td><td>+5.88%</td><td>+1.75%</td><td>-13.88%</td></tr><tr><td>Relative Impr.</td><td>+2.05%</td><td>-0.41%</td><td>+0.09% <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>+1.97%</td></tr></tbody></table>

Table 6: The impact of reinforcement learning under different numbers of generated items (Pass@K) during inference.

<sup>1</sup>

##### Search Space

In ECPO training, expanding the action search space increases the likelihood of discovering the optimal item with maximum reward, albeit at higher computational costs. To investigate this trade-off, we examine how the search space size (i.e., group size) affects performance. The results for pass@128 are summarized in Table 7. From Table 7, we observe a significant improvement in performance when the group size is increased from 128 to 512. This clearly demonstrates the positive impact of expanding the search space. It is somewhat disappointing that increasing the search space to 2048 does not yield much additional benefit, which might be due to the current reference model’s diversity not being sufficient to discover more and better items. Nonetheless, this finding is promising, and we empirically suggest setting the ECPO training group size to approximately four times the inference output quantity for optimal results.

| Group Size | vtr | Watch time | App Stay Time | Video View <sup>1</sup> |
| --- | --- | --- | --- | --- |
| 0(w/o RL) | 0.2198 | +4.61% | +1.11% | \-12.75% |
| 128 | 0.2303 | +5.22% | +1.49% | \-15.06% |
| 512 | 0.2350 | +5.73% | +1.82% | \-15.49% |
| 2048 | 0.2352 | +5.84% | +1.78% | \-15.49% |

Table 7: Performance of different group sizes when calculating ECPO loss on Pass@128.

##### Search Strategy

Reinforcement learning for large language models typically employs top- $k$ and top- $p$ sampling for sample generation. In OneRec, we also explore beam search as an alternative strategy. Table 8 compares the results of these two approaches, revealing that beam search significantly outperforms top- $k$ and top- $p$ sampling in OneRec’s reinforcement learning framework. This improvement stems from the inherent regularity of semantic ID structures, which follow a prefix tree encoding scheme and thus align well with the systematic exploration of beam search.

|  | vtr | Watch time | App Stay Time | Video View <sup>1</sup> |
| --- | --- | --- | --- | --- |
| Top- $k$ +Top- $p$ | 0.2131 | +4.45% | +1.16% | \-13.61% |
| Beam Search | 0.2162 | +5.35% | +1.76% | \-13.30% |
| Relative Impr. | +1.45% | +0.87% | +0.60% | +0.36% |

Table 8: Performance of reinforcement learning with different search strategies.

##### Reference Model

In this section, we compare two reference models for strategy generation in ECPO: (1) the pre-trained model (off-policy) and (2) the current policy model (on-policy). The experimental results are summarized in Table 9. From the table, it is evident that using the current policy model yields better results, especially in offline reward evaluation. This indicates that the on-policy approach allows the model to continuously teach itself, breaking through the limitations of the reference model and achieving a higher upper limit. However, in terms of online performance, the improvement with the on-policy approach is not very significant. This is due to the suboptimal definition of the reward, leading to slight reward hacking. We will focus on this aspect as a key direction for future work.

| Reference Model | vtr | Watch time | App Stay Time | Video View <sup>1</sup> |
| --- | --- | --- | --- | --- |
| Pre-trained Model | 0.2262 | +5.35% | +1.51% | \-13.51% |
| Current Policy Model | 0.2389 | +6.19% | +1.56% | \-13.89% |
| Relative Impr. | +5.61% | +0.79% | +0.04% | \-13.89% |

Table 9: Performance of reinforcement learning with different reference models.

##### P-Score Reward

In this section, we observe the comprehensive improvements achieved through reinforcement learning when using P-Score as the reward. Based on the conclusions from the above ablation experiments, we select the optimal RL setting, which involves using beam search for RL sample generation and employing the current policy model as the reference model. We examine the impact of RL in two scenarios, including Kuaishou and Kuaishou Lite, with the results summarized in Table 1. From the table, we can conclude that in both scenarios, P-Score significantly improves App Stay Time and Watch Time while also increasing Video View, indicating an enhancement in the overall user recommendation experience.

| Scenario | Watch time | App Stay Time | Video View |
| --- | --- | --- | --- |
| Kuaishou | +0.21% | +0.26% | +0.17% |
| Kuaishou Lite | +0.71% | +0.22% | +0.35% |

Table 10: The relative improvement of OneRec with P-Score Reward compared to without it in the Kuaishou and Kuaishou Lite scenarios.

#### 4.3.2 Generation Format Regularization

In this section, we conduct experiments to verify the effectiveness of format reward. As mentioned in Section 2.4.2, after incorporating reinforcement learning into the pre-trained model, the legality of the model’s output significantly drops to below 50% due to the squeezing effect. This means that more than half of the generated semantic IDs do not correspond to actual video IDs, which is detrimental to the stability of recommendations and the scalability of inference. We evaluate the impact of format reward by comparing two sample selection methods for computing format loss: (1) selecting the top-5 highest-probability samples from 128 generated candidates, and (2) randomly selecting 5 samples.

Figure 12 illustrates their effects on output legality. The left figure shows legality rates across all 128 generated samples, while the right panel focuses on the selected samples. Without format rewards, baseline legality remains below 50%. The Top-k Selection approach produces an interesting pattern: while overall legality initially rises then falls, the selected samples rapidly achieve 100% legality, suggesting the model learns to generate legal outputs only within the top-ranked subset. In contrast, Random Selection presents a more challenging learning objective, yet drives steady improvement - ultimately reaching 95% legality without showing a decline.

Notably, format reward integration yields benefits beyond legality alone. Online metrics demonstrate substantial gains: +0.13% in APP Stay Time and +0.30% in Watch Time. This experimental case not only validates the format reward mechanism but also highlights the critical role of careful reward design in reinforcement learning systems.

![Refer to caption](https://arxiv.org/html/2506.13695v4/x9.png)

Figure 12: The impact of training with format reward with samples obtained through different sampling strategies on the model’s legality.

#### 4.3.3 Industrial Scenario Alignment

In this section, we present a practical example of using reinforcement learning to address industrial challenges. On the Kuaishou platform, viral content farms represent a significant portion of content creators, primarily producing repurposed and clipped videos with inconsistent quality. While OneRec demonstrates superior performance over traditional recommendation systems across multiple business metrics, we observe that without proper post-filtering strategies, the exposure ratio of viral content increases significantly, which may negatively impact the platform’s ecosystem.

The optimal proportion of viral content videos can be set to $f$. When the proportion exceeds $f$, we down-weight their P-score reward to suppress them while maintaining the system’s perception of the quality of these contents.

$$
r_{i}^{\prime}=\begin{cases}r_{i}&\text{if }o_{i}\notin I_{\text{viral}}\\
\alpha r_{i}&\text{if }o_{i}\in I_{\text{viral}}\end{cases},
$$

where $\alpha\in(0,1)$ is the suppression factor.

We term this approach Specific Industrial Reward (SIR). Experimental results show that SIR effectively reduces viral content exposure by 9.59% while maintaining stable performance on core metrics (Watch time and APP Stay Time). This experiment highlights OneRec’s key advantage: the ability to achieve precise and consistent optimization through reinforcement learning’s reward-shaping capability, a feature fundamentally unavailable in traditional recommendation systems.

### 4.4 Tokenizer

We employ three metrics to comprehensively evaluate our tokenization method, encompassing aspects of accuracy, resource utilization, and distribution uniformity:

- Reconstruction Loss: This metric assesses the accuracy with which discrete tokens reconstruct the original input, serving as an indicator of the model’s fidelity in preserving the input data.
- Codebook Utilization [^35]: This metric evaluates the efficiency of vector usage within the codebook, reflecting how effectively the model leverages available resources to represent data.
- Token Distribution Entropy [^1]: Utilizing Shannon entropy, this metric quantifies the uniformity of token distribution, providing insight into the diversity and balance of token allocation across the model.

Table 11: Performance comparison of tokenization algorithms with a three-layer 8,192 codebook.

<table><tbody><tr><td colspan="2"></td><td>RQ-VAE</td><td>RQ-Kmeans</td></tr><tr><td colspan="2">Reconstruction Loss <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>0.0548</td><td>0.0410</td></tr><tr><td rowspan="3">Codebook Utilization <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>layer 1</td><td>1.0000</td><td>1.0000</td></tr><tr><td>layer 2</td><td>0.9963</td><td>1.0000</td></tr><tr><td>layer 3</td><td>0.9958</td><td>1.0000</td></tr><tr><td rowspan="3">Token Distribution Entropy <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>layer 1</td><td>8.3892</td><td>8.9191</td></tr><tr><td>layer 2</td><td>8.4805</td><td>8.7770</td></tr><tr><td>layer 3</td><td>8.6037</td><td>8.7276</td></tr></tbody></table>

As shown in Table 11, compared to RQ-VAE, RQ-Kmeans’s reconstruction loss is reduced by 25.18%, demonstrating superior accuracy in preserving input information. Simultaneously, RQ-Kmeans achieves perfect utilization (1.0000) in all three layers, indicating optimal resource efficiency in the codebook, while RQ-VAE shows slightly lower utilization rates in layers 2 and 3. Furthermore, RQ-Kmeans exhibits higher entropy values in all three layers compared to RQ-VAE, with significant improvements of 6.31%, 3.50%, and 1.44% in layers 1, 2, and 3, respectively, suggesting that RQ-Kmeans produces a more uniform and balanced token distribution, which is beneficial for model stability and generalization capability. These comprehensive results demonstrate that RQ-Kmeans outperforms RQ-VAE across all three evaluation metrics, making it a more effective choice for tokenization.

Further qualitative analyses of item representation and tokenization quality are provided in Appendix C.

### 4.5 Online A/B Test

We deployed OneRec in two major short-video scenarios on Kuaishou: the main Kuaishou feed and Kuaishou Lite feed - the platform’s highest-traffic scenarios with daily active users of 400 million. Using a 5% traffic experimental group observed over one week, our primary metrics were APP Stay Time (reflecting total user engagement time) and LT7 (7-day Lifetime). Two experimental groups were established: one employing a pure generative model (OneRec) and another augmenting generative outputs with reward model based selection (OneRec with RM Selection). As shown in Table 12, the pure generative model with RL-based user preference alignment remarkably matched the performance of the entire complex recommendation system. Further applying reward model selection achieved statistically significant improvements of +0.54% and +1.24% in APP Stay Time, and +0.05% and +0.08% in LT7 on these two scenarios, respectively. Notably, improvements of 0.1% in APP Stay Time and 0.01% in LT7 are already considered statistically significant on Kuaishou. Additionally, OneRec demonstrated significant gains across all interaction metrics (likes, follows, comments, etc.), indicating its ability to converge multi-task systems to a more balanced equilibrium without seesaw effects. After validation, we’ve expanded deployment to approximately 25% of total QPS, with implementation details available in Appendix B.

In addition to Kuaishou’s short video recommendation scenarios, experiments have also been conducted in one of its significant business scenes — Local Life Service. The results demonstrate that OneRec achieves a 21.01% growth in GMV, a 17.89% increase in order volume, an 18.58% rise in buyer numbers, and a 23.02% increase in new buyer acquisition. Consequently, the system has now taken over 100% of QPS for this business scenario. After full deployment, we observe even stronger growth across all metrics compared to the initial experimental phase. These results prove OneRec’s generalizability across diverse business contexts for enhanced recommendation performance.

Table 12: The absolute improvement of OneRec compared to the current multi-stage system in the online A/B testing setting.

<table><tbody><tr><td>Scenarios</td><td>Online Metrics</td><td>OneRec</td><td>OneRec with RM Selection</td></tr><tr><td rowspan="8">Kuaishou</td><td>App Stay Time</td><td>+0.01%</td><td>+0.54%</td></tr><tr><td>Watch Time</td><td>+0.07%</td><td>+1.98%</td></tr><tr><td>Video View</td><td>+1.98%</td><td>+2.52%</td></tr><tr><td>Like</td><td>-2.00%</td><td>+2.43%</td></tr></tbody></table>

[^1]: C. Bentz and D. Alikaniotis. The word entropy of natural languages. *arXiv preprint arXiv:1606.06996*, 2016.

[^2]: J. Cao, P. Xu, Y. Cheng, K. Guo, J. Tang, S. Wang, D. Leng, S. Yang, Z. Liu, Y. Niu, et al. Pantheon: Personalized multi-objective ensemble sort via iterative pareto policy optimization. *arXiv preprint arXiv:2505.13894*, 2025.

[^3]: J. Chang, C. Zhang, Z. Fu, X. Zang, L. Guan, J. Lu, Y. Hui, D. Leng, Y. Niu, Y. Song, et al. Twin: Two-stage interest network for lifelong user behavior modeling in ctr prediction at kuaishou. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 3785–3794, 2023.

[^4]: H.-T. Cheng, L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye, G. Anderson, G. Corrado, W. Chai, M. Ispir, et al. Wide & deep learning for recommender systems. In *Proceedings of the 1st workshop on deep learning for recommender systems*, pages 7–10, 2016.

[^5]: A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[^6]: W. Fedus, J. Dean, and B. Zoph. A review of sparse expert models in deep learning. *arXiv preprint arXiv:2209.01667*, 2022.

[^7]: A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[^8]: H. Guo, R. Tang, Y. Ye, Z. Li, and X. He. Deepfm: a factorization-machine based neural network for ctr prediction. *arXiv preprint arXiv:1703.04247*, 2017.

[^9]: T. Henighan, J. Kaplan, M. Katz, M. Chen, C. Hesse, J. Jackson, H. Jun, T. B. Brown, P. Dhariwal, S. Gray, et al. Scaling laws for autoregressive generative modeling. *arXiv preprint arXiv:2010.14701*, 2020.

[^10]: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

[^11]: S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. *arXiv preprint arXiv:2404.06395*, 2024.

[^12]: A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.

[^13]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

[^14]: D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han. Autoregressive image generation using residual quantization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11523–11532, 2022.

[^15]: J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference on machine learning*, pages 19730–19742. PMLR, 2023.

[^16]: A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.

[^17]: X. Luo, J. Cao, T. Sun, J. Yu, R. Huang, W. Yuan, H. Lin, Y. Zheng, S. Wang, Q. Hu, et al. Qarm: Quantitative alignment multi-modal recommendation at kuaishou. *arXiv preprint arXiv:2411.11739*, 2024.

[^18]: L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35:27730–27744, 2022.

[^19]: Q. Pi, G. Zhou, Y. Zhang, Z. Wang, L. Ren, Y. Fan, X. Zhu, and K. Gai. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*, pages 2685–2692, 2020.

[^20]: R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36:53728–53741, 2023.

[^21]: S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion parameter models. In *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pages 1–16. IEEE, 2020.

[^22]: S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost, et al. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems*, 36, 2024.

[^23]: Y. Ren and D. J. Sutherland. Learning dynamics of llm finetuning. *arXiv preprint arXiv:2407.10490*, 2024.

[^24]: S. Rendle. Factorization machines. In *2010 IEEE International conference on data mining*, pages 995–1000. IEEE, 2010.

[^25]: F. Ricci, L. Rokach, and B. Shapira. Introduction to recommender systems handbook. In *Recommender systems handbook*, pages 1–35. Springer, 2010.

[^26]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[^27]: N. Shazeer. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020.

[^28]: M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.

[^29]: Z. Si, L. Guan, Z. Sun, X. Zang, J. Lu, Y. Hui, X. Cao, Z. Yang, Y. Zheng, D. Leng, et al. Twin v2: Scaling ultra-long user behavior sequence modeling for enhanced ctr prediction at kuaishou. In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management*, pages 4890–4897, 2024.

[^30]: R. Thoppilan, D. De Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H.-T. Cheng, A. Jin, T. Bos, L. Baker, Y. Du, et al. Lamda: Language models for dialog applications. *arXiv preprint arXiv:2201.08239*, 2022.

[^31]: X. Wang, J. Cao, Z. Fu, K. Gai, and G. Zhou. Home: Hierarchy of multi-gate experts for multi-task learning at kuaishou. *arXiv preprint arXiv:2408.05430*, 2024.

[^32]: X. Yang, Y. Zhu, Y. Zhang, X. Wang, and Q. Yuan. Large scale product graph construction for recommendation in e-commerce. *CoRR*, abs/2010.05525, 2020.

[^33]: B. Zheng, Y. Hou, H. Lu, Y. Chen, W. X. Zhao, M. Chen, and J.-R. Wen. Adapting large language models by integrating collaborative semantics for recommendation. In *2024 IEEE 40th International Conference on Data Engineering (ICDE)*, pages 1435–1448. IEEE, 2024.

[^34]: G. Zhou, X. Zhu, C. Song, Y. Fan, H. Zhu, X. Ma, Y. Yan, J. Jin, H. Li, and K. Gai. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 1059–1068, 2018.

[^35]: L. Zhu, F. Wei, Y. Lu, and D. Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. *arXiv preprint arXiv:2406.11837*, 2024.

[^36]: D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593*, 2019.