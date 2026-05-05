---
title: "OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment"
source: "https://arxiv.org/html/2502.18965v1"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/05_%E5%85%B6%E4%BB%96/OneRec%2C%20Unifying%20Retrieve%20and%20Rank%20with%20Generative%20Recommender%20and%20Iterative%20Preference%20Alignment%2C%20Jiaxin%20Deng%20et%20al.%2C%202025.no_watermark.zh-CN.dual.pdf"
---
Jiaxin Deng KuaiShou Inc.Beijing, China [dengjiaxin03@kuaishou.com](mailto:dengjiaxin03@kuaishou.com), Shiyao Wang KuaiShou Inc.Beijing, China [wangshiyao08@kuaishou.com](mailto:wangshiyao08@kuaishou.com), Kuo Cai KuaiShou Inc.Beijing, China [caikuo@kuaishou.com](mailto:caikuo@kuaishou.com), Lejian Ren KuaiShou Inc.Beijing, China [renlejian@kuaishou.com](mailto:renlejian@kuaishou.com), Qigen Hu KuaiShou Inc.Beijing, China [huqigen03@kuaishou.com](mailto:huqigen03@kuaishou.com), Weifeng Ding KuaiShou Inc.Beijing, China [dingweifeng@kuaishou.com](mailto:dingweifeng@kuaishou.com), Qiang Luo KuaiShou Inc.Beijing, China [luoqiang@kuaishou.com](mailto:luoqiang@kuaishou.com) and Guorui Zhou KuaiShou Inc.Beijing, China [zhouguorui@kuaishou.com](mailto:zhouguorui@kuaishou.com)

(2018)

###### Abstract.

Recently, generative retrieval-based recommendation systems (GRs) have emerged as a promising paradigm by directly generating candidate videos in an autoregressive manner. However, most modern recommender systems adopt a retrieve-and-rank strategy, where the generative model functions only as a selector during the retrieval stage. In this paper, we propose OneRec, which replaces the cascaded learning framework with a unified generative model. To the best of our knowledge, this is the first end-to-end generative model that significantly surpasses current complex and well-designed recommender systems in real-world scenarios. Specifically, OneRec includes: 1) an encoder-decoder structure, which encodes the user’s historical behavior sequences and gradually decodes the videos that the user may be interested in. We adopt sparse Mixture-of-Experts (MoE) to scale model capacity without proportionally increasing computational FLOPs. 2) a session-wise generation approach. In contrast to traditional next-item prediction, we propose a session-wise generation, which is more elegant and contextually coherent than point-by-point generation that relies on hand-crafted rules to properly combine the generated results. 3) an Iterative Preference Alignment module combined with Direct Preference Optimization (DPO) to enhance the quality of the generated results. Unlike DPO in NLP, a recommendation system typically has only one opportunity to display results for each user’s browsing request, making it impossible to obtain positive and negative samples simultaneously. To address this limitation, We design a reward model to simulate user generation and customize the sampling strategy according to the attributes of the recommendation system’s online learning. Extensive experiments have demonstrated that a limited number of DPO samples can align user interest preferences and significantly improve the quality of generated results. We deployed OneRec in the main scene of Kuaishou, a short video recommendation platform with hundreds of millions of daily active users, achieving a 1.6% increase in watch-time, which is a substantial improvement.

Generative Recommendation, Autoregressive Generation, Semantic Tokenization, Direct Preference Optimization

## 1\. Introduction

To balance efficiency and effectiveness, most modern recommender systems adopt a cascade ranking strategy [^7] [^27] [^44] [^35]. As illustrated in Figure 1(b), a typical cascade ranking system employs a three-stage pipeline: recall [^7] [^20] [^55], pre-ranking [^29] [^47], and ranking [^3] [^16] [^17] [^53] [^54] [^34] [^4]. Each stage is responsible for selecting the top- $k$ items from the received items and passing the results to the next stage, collectively balancing the trade-off between system response time and sorting accuracy.

![Refer to caption](https://arxiv.org/html/2502.18965v1/x1.png)

Figure 1. (a) Our proposed unified architecture for end-to-end generation. (b) A typical cascade ranking system, which includes three stages from the bottom to the top: Retrieval, Pre-ranking, and Ranking.

Although efficient in practice, existing methods typically treat each ranker independently, where the effectiveness of each isolated stage serves as the upper bound for the subsequent ranking stage, thereby limiting the performance of the overall ranking system. Despite various efforts [^14] [^12] [^19] [^35] [^21] [^45] to improve overall recommendation performance by enabling interaction among rankers, they still maintain the traditional cascade ranking paradigm. Recently, generative retrieval-based recommendation systems (GRs) [^37] [^46] [^52] have emerged as a promising paradigm by directly generating the identifier of a candidate item in an autoregressive sequence generation manner. By indexing items with quantized semantic IDs that encode item semantics [^25], recommenders can leverage the abundant semantic information within the items. The generative nature of GRs makes them suitable for directly selecting candidate items through beam search decoding and producing more diverse recommendation results. However, current generative models only act as selectors in the retrieval stage, as their recommendation accuracy does not yet match that of well-designed multiple cascade rankers.

To address the above challenges, we propose a unified end-to-end generative framework for single-stage recommendation named OneRec. First, we present an encoder-decoder architecture. Taking inspiration from the scaling laws observed in training large language models, we find that scaling the capacity of recommendation models also consistently improves the performance. So we scale up the model parameters based on the structure of MoE [^56] [^10] [^8], which significantly improves the model’s ability to characterize user interests. Second, unlike the traditional point-by-point prediction of the next item, we propose a session-wise list generation approach that considers the relative content and order of the items within each session. The point-by-point generation method necessitates hand-craft strategies to ensure coherence and diversity in the generated results. In contrast, the session-wise learning process enables the model to autonomously learn the optimal session structure by feeding it preferred data. Last but not least, we explore preference learning by using direct preference optimization (DPO) [^36] to further enhance the quality of the generated results. For constructing preference pairs, we take inspiration from hard negative sampling [^38] by creating self-hard rejected samples from the beam search results rather than random sampling. We propose an Iterative Preference Alignment (IPA) strategy to rank the sampled responses based on scores provided by the pre-trained reward model (RM), identifying the best-chosen and worst-rejected samples. Our experiments on large-scale industry datasets show the superiority of the proposed method. We also conduct a series of ablation experiments to demonstrate the effectiveness of each module in detail. The main contributions of this work are summarized as follows:

- To overcome the limitations of cascade ranking, we introduce OneRec, a single-stage generative recommendation framework. To the best of our knowledge, this is one of the first industrial solutions capable of handling item recommendations with a unified generation model, significantly surpassing the traditional multi-stage ranking pipeline.
- We highlight the necessity of model capacity and contextual information of target items through a session-wise generation manner, which enables more accurate predictions and enhances the diversity of generated items.
- We propose a novel self-hard negative samples selection strategy based on personalized reward model. With direct preference optimization, we enhance OneRec’s generalization across a broader range of user preference. Extensive offline experiments and online A/B testing demonstrates their effectiveness and efficiency.

![Refer to caption](https://arxiv.org/html/2502.18965v1/x2.png)

Figure 2. The overall framework of OneRec, consists of two stages: (i) the session training stage which train OneRec with session-wise data; (ii) the IPA stage which utilizes iterative direct preference optimization with self-hard negatives.

## 2\. Related Work

### 2.1. Generative Recommendation

In recent years, with the remarkable progress in generative models, generative recommendation has received increasing attention. Unlike traditional embedding-based retrieval methods which largely rely on a two-tower model for calculating the ranking score for each candidate item and utilize an effecient MIPS or ANN [^18] [^32] [^39] [^15] [^22] search system for retrieving top- $k$ relevant items. Generative Retrieval (GR) [^42] method formulates the problem of retrieving relevant documents from the database as a sequence generation task which generate the relevant document tokens sequentially. The document tokens can be the document titles, document IDs or pre-trained semantic IDs [^43]. GENRE [^9] first adopts the transformer architecture for entity retrieval, generating entity names in an autoregressive fashion based on the conditioned context. DSI [^43] first proposes the concept of assigning structured semantic IDs to documents and training encoder-decoder models for generative document retrieval. Following this paradigm, TIGER [^37] introduces the formulation of generative item retrieval models for recommender systems.

In addition to the generation framework, how to index items has also attracted increasing attention. Recent research focuses on the semantic indexing technique [^37] [^43] [^13], which aims to index items based on content information. Specifically, TIGER [^37] and LC-Rec [^52] apply residual quantization (RQ-VAE) to textual embeddings derived from item titles and descriptions for tokenization. Recforest [^13] utilizes hierarchical k-means clustering on item textual embeddings to obtain cluster indexes as tokens. Furthermore, recent studies such as EAGER [^46] explore integrating both semantic and collaborative information into the tokenization process.

### 2.2. Preference Alignment of Language Models

During the post-training [^11] phase of Large Language Models (LLMs), Reinforcement Learning from Human Feedback (RLHF) [^40] [^33] is a prevalent method in aligning LLMs with human values by employing reinforcement learning techniques guided by reward models that represent human feedback. However, RLHF suffers from instability and inefficiency. Direct Preference Optimization (DPO) [^36] is proposed which derives the optimal policy in closed form and enables direct optimization using preference data. Apart from that, several variants have been proposed to further improve the original DPO. For example, IPO [^2] bypasses two approximations in DPO with a general objective. cDPO [^36] alleviates the influence of noisy labels by introducing a hyperparameter $\epsilon$. rDPO [^6] designs an unbiased estimate of the original Binary Cross Entropy loss. Other variants including CPO [^48], simDPO [^6], also enhance or expand DPO in various aspects. However, unlike traditional NLP scenarios where preference data is explicitly annotated through humans, preference learning in recommendation systems faces a unique challenge because of the sparsity of user-item interaction data. This challenge results in adapting DPO for recommendation are still largely unexplored. Different from S-DPO which focuses on incorporating multiple negatives in user preference data for LM-based recommenders, we train a reward model and based on the scores from reward model we choose personalized preference data for different users.

## 3\. Methods

In this section, we propose OneRec, an end-to-end framework that generates target items through a single-stage retrieval manner. In Section 3.1, we first introduce the feature engineering for the single-stage generative recommendation pipeline in industrial applications. Then, in Section 3.2, we formally define the session-wise generative tasks and present the architecture of our proposed OneRec model. Finally, we elaborate on the model’s capability with a personalized reward model for self-hard negative sampling in Section 3.3, and demonstrate how we iteratively improve model performance through direct preference optimization. The overall framework of OneRec is illustrated in Figure 2.

### 3.1. Preliminary

In this section, we introduce the construction of the single-stage generative recommendation pipeline from the perspectives of feature engineering. For user-side feature, OneRec takes the positive historical behavior sequences $\mathcal{H}_{u}=\{\bm{v}^{h}_{1},\bm{v}^{h}_{2},\ldots,\bm{v}^{h}_{n}\}$ as input, where $\bm{v}$ represent the videos that the user has effectively watched or interacted with (likes, follows, shares), and $n$ is the length of behaviour sequence. The output of OneRec is a list of videos, consisting of a session $\mathcal{S}=\{\bm{v}_{1},\bm{v}_{2},...,\bm{v}_{m}\}$, where $m$ is the number of videos within a session (the detailed definition of “session” can be found in Section 3.2).

For each video $\bm{v}_{i}$, we describe them with multi-modal embeddings $\textbf{{e}}_{i}\in\mathbb{R}^{d}$ which are aligned with the real user-item behaviour distribution [^28]. Based on the pretrain multi-modal representation, existing generative recommendation frameworks [^26] [^37] use RQ-VAE [^50] to encode the embedding into semantic tokens. However, such method is suboptimal due to the unbalanced code distribution which is known as the hourglass phenomenon [^24]. We apply a multi-level balanced quantitative mechanism to transform the $\textbf{{e}}_{i}$ with residual K-Means quantization algorithm [^28]. At the first level ($l=1$), the initial residual is defined as $\bm{r}_{i}^{1}=\bm{e}_{i}$. At each level $l$, we have a codebook $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},...,\bm{c}_{K}^{l}\}$, where $K$ is the codebook size. The index of the closest centroid node embedding is generate through $\bm{s}_{i}^{l}=\arg\min_{k}\|\textbf{{r}}_{i}^{l}-\bm{c}_{k}^{l}\|_{2}^{2}$ and for next level $l+1$ the residual is defined as $\bm{r}_{i}^{l+1}=\bm{r}_{i}^{l}-\bm{c}_{\bm{s}_{i}^{l}}^{l}$. So the corresponding codebook tokens are generated through hierarchical indexing:

$$
\displaystyle\bm{s}_{i}^{1}=\arg\min_{k}\|\bm{r}_{i}^{1}-\bm{c}_{k}^{1}\|_{2}^%
{2},\quad\bm{r}_{i}^{2}=\bm{r}_{i}^{1}-\bm{c}_{\bm{s}_{i}^{1}}^{1}
$$
 
$$
\displaystyle\bm{s}_{i}^{2}=\arg\min_{k}\|\bm{r}_{i}^{2}-\bm{c}_{k}^{2}\|_{2}^%
{2},\quad\bm{r}_{i}^{3}=\bm{r}_{i}^{2}-\bm{c}_{\bm{s}_{i}^{2}}^{2}
$$
 
$$
\displaystyle\qquad\qquad\vdots
$$
 
$$
\displaystyle\bm{s}_{i}^{L}=\arg\min_{k}\|\bm{r}_{i}^{L}-\bm{c}_{k}^{L}\|_{2}^%
{2}
$$

where $L$ is the total layers of sematic ID.

To construct a balanced codebook $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},\ldots,\bm{c}_{K}^{l}\}$, we apply the Balanced K-means as detailed in Algorithm 1 for itemset partitioning. Given the total video set $\mathcal{V}$, this algorithm partitions the set into $K$ clusters, where each cluster contains exactly $w=|\mathcal{V}|/K$ videos. During iterative computation, each centroid is sequentially assigned its $w$ nearest unallocated videos based on Euclidean distance, followed by centroid recalibration using mean vectors of assigned videos. The termination criterion is satisfied when cluster assignments reach convergence.

Input: Item set $\mathcal{V}$, number of clusters $K$

Compute $w\leftarrow|\mathcal{V}|/K$

Initialize centroids $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},\ldots,\bm{c}_{K}^{l}\}$ with random selection;

repeat

Initialize unassigned set $\mathcal{U}\leftarrow\mathcal{V}$

for *each cluster $k\in\{1,\ldots,K\}$* do

Sort $\mathcal{U}$ by ascending distance from centroid $\bm{c}_{k}^{l}$;

Assign $\mathcal{V}_{k}\leftarrow\mathcal{U}[0:w-1]$;

Update centroid $\bm{c}_{k}^{l}\leftarrow\frac{1}{w}\sum_{\bm{r}^{l}\in\mathcal{V}_{k}}\bm{r}^{l}$;

Remove assigned items $\mathcal{U}\leftarrow\mathcal{U}\setminus\mathcal{V}_{k}$;

end for

until *Assignment convergence*;

Output: Optimized codebook $\mathcal{C}_{l}=\{\bm{c}_{1}^{l},\ldots,\bm{c}_{K}^{l}\}$

Algorithm 1 Balanced K-means Clustering

### 3.2. Session-wise List Generation

Different from traditional point-wise recommendation methods that only predict the next video, session-wise generation aims to generate a list of high-value sessions based on users’ historical interaction sequences, which enables the recommendation model to capture the dependencies between videos in the recommended list. Specifically, a session refers to a batch of short videos returned in response to a user’s request, typically consisting of 5 to 10 videos. The videos within a session generally take into account factors such as user interest, coherence, and diversity. We have devised several criteria to identify high-quality sessions, including:

- The number of short videos actually watched by the user within a session is greater than or equal to 5;
- The total duration for which the user watches the session exceeds a certain threshold;
- The user exhibits interaction behaviors, such as liking, collecting, or sharing the videos;

This approach ensures that our session-wise model learns from real user engagement patterns and captures more accurate contextual information within the session list. So the objective of our session-wise model $\mathcal{M}$ can be formalized as:

$$
\mathcal{S}:=\mathcal{M}(\mathcal{H}_{u})
$$

where $\mathcal{H}_{u}$ is represented from the remantic IDs: $\mathcal{H}_{u}=\{(\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\\
\bm{s}_{1}^{L}),(\bm{s}_{2}^{1},\bm{s}_{2}^{2},\cdots,\bm{s}_{2}^{L}),\cdots,(%
\bm{s}_{n}^{1},\bm{s}_{n}^{2},\cdots,\bm{s}_{n}^{L})\}$ and $\mathcal{S}=\{(\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\bm{s}_{1}^{L}),\\
(\bm{s}_{2}^{1},\bm{s}_{2}^{2},\cdots,\bm{s}_{2}^{L}),\cdots,(\bm{s}_{m}^{1},%
\bm{s}_{m}^{2},\cdots,\bm{s}_{m}^{L})\}$.

As shown in Figure 2 (a), consistent with the T5 [^49] architecture, our model employs a transformer-based framework consisting of two main components: an encoder for modeling user historical interactions and a decoder for session list generation. Specifically, the encoder leverages the stacked multi-head self-attention and feed-forward layers to process the input sequence $\mathcal{H}_{u}$. We denote the encoded historical interaction features as $\textbf{{H}}=Encoder(\mathcal{H}_{u})$.

The decoder takes the semantic IDs of the target session as input and generates the target in an auto-regressive manner. To train a larger model at reasonable economic costs, for the feed-forward neural networks (FNNs) in the decoder, we adopt the MoE architecture [^56] [^10] [^8] commonly used in Transformer-based language models and substitute the $l$ -th FFN with:

$$
\begin{split}\mathbf{H}_{t}^{l+1}&=\sum_{i=1}^{N_{\rm MoE}}\left({g_{i,t}%
\operatorname{FFN}_{i}\left(\mathbf{H}_{t}^{l}\right)}\right)+\mathbf{H}_{t}^{%
l},\\
g_{i,t}&=\begin{cases}s_{i,t},&s_{i,t}\in\operatorname{Topk}(\{s_{j,t}|1\leq j%
\leq N\},K_{\rm MoE}),\\
0,&\text{otherwise},\end{cases}\\
s_{i,t}&=\operatorname{Softmax}_{i}\left({\mathbf{H}_{t}^{l}}^{T}\mathbf{e}_{i%
}^{l}\right),\end{split}
$$

where $N_{\rm MoE}$ represents the total number of experts, $\operatorname{FFN}_{i}(\cdot)$ is the $i$ -th expert FFN, and $g_{i,t}$ denotes the gate value for the $i$ -th expert. The gate value $g_{i,t}$ is sparse, meaning that only $K_{\rm MoE}$ out of $N_{\rm MoE}$ gate values are non-zero. This sparsity property ensures computational efficiency within an MoE layer and each token will be assigned to and computed in only $K_{\rm MoE}$ experts.

For training, we add a start token $\bm{s}_{[\rm BOS]}$ at the beginning of codes to construct the decoder inputs with:

$$
\begin{gathered}{\mathcal{\bar{S}}}=\{\bm{s}_{[\rm BOS]},\bm{s}_{1}^{1},\bm{s}%
_{1}^{2},\cdots,\bm{s}_{1}^{L},\bm{s}_{[\rm BOS]},\bm{s}_{2}^{1},\bm{s}_{2}^{2%
},\cdots,\bm{s}_{2}^{L},\\
\cdots,\bm{s}_{[\rm BOS]},\bm{s}_{m}^{1},\bm{s}_{m}^{2},\cdots,\bm{s}_{m}^{L}%
\}\end{gathered}
$$

We utilize cross-entropy loss for next-token prediction on the sematic IDs of the target session. The NTP loss $\mathcal{L}_{\rm NTP}$ is formulated as:

$$
\begin{gathered}\mathcal{L}_{\rm NTP}=-\sum_{i=1}^{m}\sum_{j=1}^{L}\log P(\bm{%
s}_{i}^{j+1}\mid[\bm{s}_{[\rm BOS]},\bm{s}_{1}^{1},\bm{s}_{1}^{2},\cdots,\bm{s%
}_{1}^{L},\cdots,\\
\bm{s}_{[\rm BOS]},\bm{s}_{i}^{1},\cdots,\bm{s}_{i}^{j}];\Theta).\end{gathered}
$$

After a certain amount of training on session-wise list generation task, we obtain the seed model $\mathcal{M}_{t}$.

### 3.3. Iterative Preference Alignment with RM

The high-quality sessions defined in Section 3.2 provide valuable training data, enabling the model to learn what constitutes a good session, thereby ensuring the quality of generated videos. Building on this foundation, we aim to further enhance the model’s ability by Direct Preference Optimization (DPO). In traditional natural language processing (NLP) scenarios, preference data is explicitly annotated by humans. However, preference learning in recommendation systems confronts a unique challenge due to the sparsity of user-item interaction data, which necessitates a reward model (RM). So we introduce a session-wise reward model in Section 3.3.1. Moreover, we improve the conventional DPO by proposing an iterative direct preference optimization that enables the model to self-improvement described in Section 3.3.2.

for *$\mathit{epoch}\leftarrow t$ to $T$* do

for *$\mathit{sample}\leftarrow 1$ to $N_{\mathrm{sample}}$* do

if *$\mathit{rand}()<r_{\mathrm{DPO}}$* then

for *$i\leftarrow 1$ to $N$* do

$\mathcal{S}_{u}^{i}\sim\mathcal{M}_{t}(\mathcal{H}_{u})$;

$r_{u}^{i}\leftarrow R(\bm{u},\mathcal{S}_{u}^{i})$;

end for

$\mathcal{S}_{u}^{w}\leftarrow\mathcal{S}_{u}^{\arg\max_{i}r_{u}^{i}}$;

$\mathcal{S}_{u}^{l}\leftarrow\mathcal{S}_{u}^{\arg\min_{i}r_{u}^{i}}$;

Compute NTP and DPO loss:

$\mathcal{L}\leftarrow\mathcal{L}_{\mathrm{NTP}}+\lambda\mathcal{L}_{\mathrm{%
DPO}}$;

else

Compute NTP loss:

$\mathcal{L}\leftarrow\mathcal{L}_{\mathrm{NTP}}$;

end if

Update parameters:

$\Theta\leftarrow\Theta-\alpha\nabla_{\Theta}\mathcal{L}$;

end for

Update model snapshot: $\mathcal{M}_{t+1}\leftarrow\mathcal{M}_{t}$;

end for

Output: Optimized parameters $\Theta$

Algorithm 2 Iterative Preference Alignment (IPA)

#### 3.3.1. Reward Model Training

We use $R(\bm{u},\mathcal{S})$ to denote the reward model which selects preference data for different users. Here, the output $r$ represents the reward corresponding to user $u$ ’s (usually represented by user behavior) preference on the session $\mathcal{S}=\{\bm{v}_{1},\bm{v}_{2},\ldots,\bm{v}_{m}\}$. In order to equip the RM with the capacity to rank session, we first extract the target-aware representation $\bm{e}_{i}=\bm{v}_{i}\odot\bm{u}$ of each item $\bm{v}_{i}$ in $\mathcal{S}$, where $\odot$ represents the target-aware operation (such as target attention toward user behavior). So we get the target-aware representation $\bm{h}=\{\bm{e}_{1},\bm{e}_{2},\cdots,\bm{e}_{m}\}$ for session $\mathcal{S}$. Then the items within a session interact with each other through self-attention layers to fuse the necessary information among different items:

$$
\displaystyle\bm{h}_{f}=\mathrm{SelfAttention}(\bm{h}\bm{W}^{Q}_{s},\bm{h}\ %
\bm{W}^{K}_{s},\bm{h}\bm{W}^{V}_{s})
$$

Next we utilize different tower to make predictions on multi-target reward and the RM is pre-trained with abundant recommendation data:

$$
\small\begin{split}\hat{r}^{swt}&=\texttt{Tower}^{swt}\big{(}\texttt{Sum}\big{%
(}\bm{h}_{f}\big{)}\big{)},\hat{r}^{vtr}=\texttt{Tower}^{vtr}\big{(}\texttt{%
Sum}\big{(}\bm{h}_{f}\big{)}\big{)},\\
\hat{r}^{wtr}&=\texttt{Tower}^{wtr}\big{(}\texttt{Sum}\big{(}\bm{h}_{f}\big{)}%
\big{)},\hat{r}^{ltr}=\texttt{Tower}^{ltr}\big{(}\texttt{Sum}\big{(}\bm{h}_{f}%
\big{)}\big{)},\\
&\texttt{whe}\texttt{re}\quad\texttt{Tower}(\cdot)=\texttt{Sigmoid}\big{(}%
\texttt{MLP}(\cdot)\big{)}\end{split}
$$

After getting all the estimated rewards $\hat{r}^{swt},\dots$ and the ground-truth labels $y^{swt},\dots$ for each session, we directly minimize the binary cross-entropy loss to train the RM. The loss function $\mathcal{L}_{\rm RM}$ is defined as follows:

$$
\begin{split}\mathcal{L}_{\rm RM}=-\sum_{{swt,\dots}}^{xtr}\left(y^{xtr}\log{(%
\hat{r}^{xtr})}+(1-y^{xtr})\log{(1-\hat{r}^{xtr})}\right)\end{split}
$$

#### 3.3.2. Iterative Preference Alignment

Based on pre-trained RM $R(\bm{u},\mathcal{S})$ and current OneRec $\mathcal{M}_{t}$, we generate $N$ different responses for each user by beam search:

$$
\mathcal{S}_{u}^{n}\sim M_{t}(\mathcal{H}_{u})\quad\text{for all}\ u\in%
\mathcal{U}\ \text{and}\ n\in[N],
$$

where we use $[N]$ to denote $\{1,2,\dots,N\}$.

Then we computes the reward $r_{u}^{n}$ for each of these responses based on the RM $R(\bm{u},\mathcal{S})$:

$$
r_{u}^{n}=R(\bm{u},\mathcal{S}_{u}^{n})
$$

Next we build the preference pairs $D_{t}^{\text{pairs}}=(\mathcal{S}_{u}^{w},\mathcal{S}_{u}^{l},\mathcal{H}_{u})$ by choosing the winner response $(\mathcal{S}_{u}^{w},\mathcal{H}_{u})$ with the highest reward value and a loser response $(\mathcal{S}_{u}^{l},\mathcal{H}_{u})$ with the lowest reward value. Given the preference pairs, we can now train a new model $M_{t+1}$ which is initialized from model $M_{t}$, and updated with a loss function that combines the DPO loss [^36] for learning from the preference pairs. The loss corresponding to each preference pair is as follows:

$$
\begin{split}\mathcal{L}_{\text{DPO}}&=\mathcal{L}_{\text{DPO}}(\mathcal{S}_{u%
}^{w},\mathcal{S}_{u}^{l}|\mathcal{H}_{u})\\
&=-\log\sigma\left(\beta\log\frac{M_{t+1}(\mathcal{S}_{u}^{w}|\mathcal{H}_{u})%
}{M_{t}(\mathcal{S}_{u}^{w}|\mathcal{H}_{u})}-\beta\log\frac{M_{t+1}(\mathcal{%
S}_{u}^{l}|\mathcal{H}_{u})}{M_{t}(\mathcal{S}_{u}^{l}|\mathcal{H}_{u})}\right%
).\end{split}
$$

As shown in Algorithm 2 and Figure 2 (b), the overall procedure involves training a sequence of models $M_{t},\dots,M_{T}$. To mitigate the computational burden during beam search inference, we randomly sample only $r_{\rm DPO}=1\%$ data for preference alignment. For each successive model $M_{t+1}$, it initializes from previous model $M_{t}$ and utilizes the preference data $D_{t}^{\text{pairs}}$ generated by the $M_{t}$ for training.

![Refer to caption](https://arxiv.org/html/2502.18965v1/x3.png)

Table 1. Offline performance of our proposed OneRec (green) with pointwise methods (brown), listwise methods (blue) and preference alignment methods (yellow). Best results are in bold, sub-optimal results are underlined. Metrics with ↑ \\uparrow indicate higher is better, while ↓ \\downarrow indicates lower is better.

[^2]: Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. 2024. A general theoretical paradigm to understand learning from human preferences. In *International Conference on Artificial Intelligence and Statistics*. PMLR, 4447–4455.

[^3]: Christopher JC Burges. 2010. From ranknet to lambdarank to lambdamart: An overview. *Learning* 11, 23-581 (2010), 81.

[^4]: Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, et al. 2023. TWIN: TWo-stage interest network for lifelong user behavior modeling in CTR prediction at kuaishou. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 3785–3794.

[^5]: Yuxin Chen, Junfei Tan, An Zhang, Zhengyi Yang, Leheng Sheng, Enzhi Zhang, Xiang Wang, and Tat-Seng Chua. 2024. On Softmax Direct Preference Optimization for Recommendation. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*. [https://openreview.net/forum?id=qp5VbGTaM0](https://openreview.net/forum?id=qp5VbGTaM0)

[^6]: Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. 2024. Provably Robust DPO: Aligning Language Models with Noisy Feedback. In *ICML 2024*.

[^7]: Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In *Proceedings of the 10th ACM conference on recommender systems*. 191–198.

[^8]: Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066* (2024).

[^9]: Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Autoregressive entity retrieval. *arXiv preprint arXiv:2010.00904* (2020).

[^10]: Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. 2022. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*. PMLR, 5547–5569.

[^11]: Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783* (2024).

[^12]: Hongliang Fei, Jingyuan Zhang, Xingxuan Zhou, Junhao Zhao, Xinyang Qi, and Ping Li. 2021. GemNN: gating-enhanced multi-task neural networks with feature interaction learning for CTR prediction. In *Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval*. 2166–2171.

[^13]: Chao Feng, Wuchao Li, Defu Lian, Zheng Liu, and Enhong Chen. 2022. Recommender forest for efficient retrieval. *Advances in Neural Information Processing Systems* 35 (2022), 38912–38924.

[^14]: Luke Gallagher, Ruey-Cheng Chen, Roi Blanco, and J Shane Culpepper. 2019. Joint optimization of cascade ranking models. In *Proceedings of the twelfth ACM international conference on web search and data mining*. 15–23.

[^15]: Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. *IEEE transactions on pattern analysis and machine intelligence* 36, 4 (2013), 744–755.

[^16]: Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. *arXiv preprint arXiv:1703.04247* (2017).

[^17]: B Hidasi. 2015. Session-based Recommendations with Recurrent Neural Networks. *arXiv preprint arXiv:1511.06939* (2015).

[^18]: Michael E Houle and Michael Nett. 2014. Rank-based similarity search: Reducing the dimensional dependence. *IEEE transactions on pattern analysis and machine intelligence* 37, 1 (2014), 136–150.

[^19]: Jiri Hron, Karl Krauth, Michael Jordan, and Niki Kilbertus. 2021. On component interactions in two-stage recommender systems. *Advances in neural information processing systems* 34 (2021), 2744–2757.

[^20]: Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In *Proceedings of the 22nd ACM international conference on Information & Knowledge Management*. 2333–2338.

[^21]: Xu Huang, Defu Lian, Jin Chen, Liu Zheng, Xing Xie, and Enhong Chen. 2023. Cooperative Retriever and Ranker in Deep Recommenders. In *Proceedings of the ACM Web Conference 2023*. 1150–1161.

[^22]: Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. *IEEE transactions on pattern analysis and machine intelligence* 33, 1 (2010), 117–128.

[^23]: Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In *2018 IEEE international conference on data mining (ICDM)*. IEEE, 197–206.

[^24]: Zhirui Kuai, Zuxu Chen, Huimu Wang, Mingming Li, Dadong Miao, Wang Binbin, Xusong Chen, Li Kuang, Yuxing Han, Jiaxing Wang, et al. 2024. Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieval. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track*. 677–685.

[^25]: Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive image generation using residual quantization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 11523–11532.

[^26]: Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, and Liqiang Nie. 2024. MMGRec: Multimodal Generative Recommendation with Transformer Model. *arXiv preprint arXiv:2404.16555* (2024).

[^27]: Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. 2017. Cascade ranking for operational e-commerce search. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 1557–1565.

[^28]: Xinchen Luo, Jiangxia Cao, Tianyu Sun, Jinkai Yu, Rui Huang, Wei Yuan, Hezheng Lin, Yichen Zheng, Shiyao Wang, Qigen Hu, et al. 2024. QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. *arXiv preprint arXiv:2411.11739* (2024).

[^29]: Xu Ma, Pengjie Wang, Hui Zhao, Shaoguo Liu, Chuhan Zhao, Wei Lin, Kuang-Chih Lee, Jian Xu, and Bo Zheng. 2021. Towards a better tradeoff between effectiveness and efficiency in pre-ranking: A learnable feature selection based approach. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2036–2040.

[^30]: Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. SimPO: Simple Preference Optimization with a Reference-Free Reward. In *Advances in Neural Information Processing Systems (NeurIPS)*.

[^31]: Eric Mitchell. \[n. d.\]. A note on dpo with noisy preferences and relationship to ipo, 2023. *URL https://ericmitchell. ai/cdpo. pdf* (\[n. d.\]).

[^32]: Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms for high dimensional data. *IEEE transactions on pattern analysis and machine intelligence* 36, 11 (2014), 2227–2240.

[^33]: Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. *Advances in neural information processing systems* 35 (2022), 27730–27744.

[^34]: Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*. 2685–2692.

[^35]: Jiarui Qin, Jiachen Zhu, Bo Chen, Zhirong Liu, Weiwen Liu, Ruiming Tang, Rui Zhang, Yong Yu, and Weinan Zhang. 2022. Rankflow: Joint optimization of multi-stage cascade ranking systems as flows. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 814–824.

[^36]: Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems* 36 (2024).

[^37]: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems* 36 (2023), 10299–10315.

[^38]: Wentao Shi, Jiawei Chen, Fuli Feng, Jizhi Zhang, Junkang Wu, Chongming Gao, and Xiangnan He. 2023. On the theories behind hard negative sampling for recommendation. In *Proceedings of the ACM Web Conference 2023*. 812–822.

[^39]: Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). *Advances in neural information processing systems* 27 (2014).

[^40]: Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems* 33 (2020), 3008–3021.

[^41]: Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM international conference on information and knowledge management*. 1441–1450.

[^42]: Yubao Tang, Ruqing Zhang, Jiafeng Guo, and Maarten de Rijke. 2023. Recent advances in generative information retrieval. In *Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region*. 294–297.

[^43]: Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a differentiable search index. *Advances in Neural Information Processing Systems* 35 (2022), 21831–21843.

[^44]: Lidan Wang, Jimmy Lin, and Donald Metzler. 2011. A cascade ranking model for efficient ranked retrieval. In *Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval*. 105–114.

[^45]: Yunli Wang, Zhiqiang Wang, Jian Yang, Shiyang Wen, Dongying Kong, Han Li, and Kun Gai. 2024a. Adaptive Neural Ranking Framework: Toward Maximized Business Goal for Cascade Ranking Systems. In *Proceedings of the ACM on Web Conference 2024*. 3798–3809.

[^46]: Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024b. EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 3245–3254.

[^47]: Zhe Wang, Liqin Zhao, Biye Jiang, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2020. Cold: Towards the next generation of pre-ranking system. *arXiv preprint arXiv:2007.16122* (2020).

[^48]: Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. 2024b. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. *arXiv preprint arXiv:2401.08417* (2024).

[^49]: Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. 2024a. Openp5: An open-source platform for developing, training, and evaluating llm-based recommender systems. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 386–394.

[^50]: Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2021. Soundstream: An end-to-end neural audio codec. *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 30 (2021), 495–507.

[^51]: Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level deeper self-attention network for sequential recommendation.. In *IJCAI*. 4320–4326.

[^52]: Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In *2024 IEEE 40th International Conference on Data Engineering (ICDE)*. IEEE, 1435–1448.

[^53]: Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 33. 5941–5948.

[^54]: Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1059–1068.

[^55]: Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1079–1088.

[^56]: Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. 2022. Designing effective sparse expert models. *arXiv preprint arXiv:2202.08906* 2, 3 (2022), 17.