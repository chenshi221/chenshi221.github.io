---
title: "SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model"
source: "https://arxiv.org/html/2510.12709v3"
author:
published: 2025-11-02
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/02_%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B/SAIL-Embedding%20Technical%20Report%2C%20Omni-modal%20Embedding%20Foundation%20Model%2C%20Lin%20Lin%20et%20al.%2C%202025.zh-CN.dual%20%281%29.pdf"
---
###### Abstract

Multimodal embedding models aim to yield informative unified representations that empower diverse cross-modal tasks. Despite promising developments in the evolution from CLIP-based dual-tower architectures to large vision-language models, prior works still face unavoidable challenges in real-world applications and business scenarios, such as the limited modality support, unstable training mechanisms, and industrial domain gaps. In this work, we introduce SAIL-Embedding, an omni-modal embedding foundation model that addresses these issues through tailored training strategies and architectural design. SAIL-Embedding supports multifaceted multimodal retrieval and classification by accommodating arbitrary modality inputs, including transcribed textual information, sampled visual semantics, and acquirable audio signals. To enhance training robustness and scalability, we introduce the dynamic hard negative mining and adaptive multi-source data balancing to consolidate domain expertise and capture effective multimodal representations. In the optimization procedure, we propose a multi-stage training scheme to boost the multifaceted effectiveness of representation learning. Specifically, the content-aware progressive training aims to enhance the model’s adaptability to diverse downstream tasks and master enriched cross-modal proficiency. The collaboration-aware recommendation enhancement training further adapts multimodal representations for recommendation scenarios by distilling knowledge from sequence-to-item and ID-to-item embeddings while mining user historical interests. Concurrently, we develop the stochastic specialization and dataset-driven pattern matching to strengthen model training flexibility and generalizability. Experimental results demonstrate that SAIL-Embedding achieves state-of-the-art performance compared to other methods in item-to-item and query-to-item retrieval tasks across different intents. Furthermore, we provide comprehensive analysis and ablation studies to reveal the necessity of the proposed modules and components. In online experiments across various real-world scenarios integrated with our model, we observe a significant increase in Lifetime (LT), which is a crucial indicator for the recommendation experience. For instance, the model delivers the 7-day LT gain of +0.5% in the Douyin-Selected scenario. Furthermore, through clustering quantification, the model is widely applied across diverse situations such as decentralization, recall, pre-ranking, and re-ranking. For the Douyin feed rank model, the match features produced by SAIL-Embedding yield a +0.1% AUC gain.

newfloatplacementnewfloatnamenewfloatfileextnewfloatwithin

![Refer to caption](https://arxiv.org/html/2510.12709v3/x1.png)

Figure 1: SAIL-Embedding Capability Overview. We present SAIL-Embedding, an omni-modal embedding model adapted from vision-language-audio models. SAIL-Embedding is capable of following instructions and performing various omni-modal embedding tasks, such as Motivation/Tag-CLS (Classification), Search-q2i (Query-to-Item Retrieval), and Copair/Live-i2i (Item-to-Item Retrieval).

## 1 Introduction

Embedding models aim to generate meaningful dense vector representations of data. From the early exploration of distributed word representations such as Word2Vec [^43] and GloVe [^45] to the recent development of large language model (LLM)-based embedding models such as Gemini-Embedding [^33] and Qwen-Embedding [^75], textual embedding models [^39] [^18] [^19] have become a fundamental component of natural language processing and have enabled a wide range of downstream applications, like retrieval-augmented generation (RAG) [^24] [^28] and information extraction [^52] [^44] [^31]. With the rapid progress of multimodal learning [^20] [^64] [^68] [^65] [^59] [^3] [^55] [^56] [^34], multimodal embedding models [^74] [^29] [^42] [^50] [^27] [^60] [^76] [^40] [^4] that map heterogeneous modalities into a unified vector space have also emerged as an active research direction, showing great promises in practical scenarios such as short-video, image recommendation, as well as cross-modal search.

Current multimodal embedding models generally follow two design paradigms, as illustrated in Figure 2. The first paradigm, exemplified by CLIP [^47] and SigLIP [^72], leverages paired multimodal data and employs two large encoders to process each modality independently, with either no fusion or only shallow fusion layers. This design is structurally simple and has proven highly effective for cross-modal retrieval. However, its reliance on shallow fusion restricts the expressiveness and semantic richness of the resulting embeddings. With the recent advances in large language models (LLMs) and multimodal LLMs (MLLMs), a second paradigm has gained increasing attention, as shown in Figure 2(b). This line of work integrates LLMs/MLLMs to achieve deep semantic fusion across modalities [^74] [^29] [^4] [^40]. Although still under active investigation, this paradigm has already exhibited strong representational capacity and is rapidly emerging as the de facto choice for a wide range of downstream tasks [^17].

However, when deployed for real-world scenarios, the models still face significant limitations that hinder their effectiveness. (1) Limited Modalities: Most existing methods rely on only two modalities—typically images and text—for unimodal or cross-modal retrieval. In contrast, industrial applications often require richer and more comprehensive multimodal representations. For example, a Douyin video contains diverse sources of information: visual signals from the cover frame and keyframes, textual cues from tags and captions, background music as audio, and spoken content transcribed via automatic speech recognition (ASR). Each modality contributes essential semantic information, and omitting or misinterpreting even one can severely impact downstream tasks such as recommendation and search, ultimately degrading user experience. (2) Training Instability: These models are typically built upon multimodal large language models, which require careful architectural design and optimization strategies to ensure stable and efficient training. Developing optimization strategies that unlock the practical values of incentive models within business is an indispensable key. (3) Industrial Domain Gap: Many models are trained and evaluated primarily on open-source datasets, but they often underperform on domain-specific industrial data, such as expressive short videos on Douyin, where data distributions and task requirements differ substantially from academic benchmarks.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x2.png)

Figure 2: Embedding Architecture Comparison. (a) CLIP-like dual-tower embedding model architecture. (b) LLM/MLLM-based embedding model architecture. (c) SAIL-Embedding model architecture. In contrast, our model accommodates arbitrary modality inputs and can handle diverse downstream tasks. I, V, T, and O represent image, video, text, and omni-modality, respectively.

To this end, we propose SAIL-Embedding, a powerful omni-modal embedding model of the SAIL families [^13] [^70], along with tailored training strategies to ensure stable large-scale optimization and balance modality contributions. SAIL-Embedding can handle arbitrary combinations of inputs from vision, text, and audio modalities, yielding multi-dimensional representation vectors to fulfil diverse real-world business requirements.

To further enhance the model’s robustness during the training process and improve its scalability to handle large-scale and diverse datasets, we introduce dynamic hard negative mining and adaptive multi-source data balancing. The dynamic hard negative mining helps the model focus on distinguishing challenging negative samples, thereby consolidating the model’s understanding of domain-specific knowledge and reducing the risk of misclassification caused by ambiguous samples. Meanwhile, the adaptive multi-source data balancing dynamically learns weights directly from the data distribution to reduce reliance on manual parameter tuning and maintain the trade-off between data quality and distributional diversity.

SAIL-Embedding presents the multi-stage training procedure that learns unified representations across multiple dimensions through content- and collaboration-awareness. Specifically, content-aware progressive training gradually enhances the embeddings’ discriminative power for diverse task demands and the generalization ability to handle unseen scenarios by leveraging diverse, semantically rich data resources. This process endows the model with comprehensive domain knowledge capabilities, including cross-modal semantic integration, scenario-specific content understanding, and complex concept relation. During the collaboration-aware recommendation enhancement phase, we perform multi-dimensional interest-driven sequence-to-item distillation to incorporate the users’ historical behavioural patterns into the multimodal representations. Subsequently, the combined ID-to-item distillation further aggregates user-specific preference signals within the recommendation system, thereby improving the accuracy of item recommendations.

Extensive experimental results are conducted on multiple benchmark datasets covering different application scenarios, demonstrating that the proposed SAIL-Embedding achieves state-of-the-art (SOTA) performance compared to other advanced baseline methods (i.e., traditional unimodal models, dual-tower fusion models, and large vision-language representation methods) in diverse downstream tasks.

In online experiments, SAIL-Embedding demonstrates its effectiveness in the recommender system of Douyin, which brings substantial gains through diverse application pathways. For cold-start scenarios, our model achieves a 0.05% LT gain in total by performing embedding-based item2item (i2i) recall along with engaging embeddings into the recommendation model. Specifically, SAIL-Embedding improves the AUC of the cold-start model by 0.1% when deployed as a target side feature. We also discretize embeddings info semantic tokens, which ultimately deliver $\sim$ 0.03% LT gain across different stages of the recommender system, including recall, pre-ranking and re-ranking. Specifically, the AUC of the ranking model can be improved by $\sim$ 0.1% with the engagement of both embeddings and sematic tokens.

## 2 Related Works

We have witnessed remarkable progress in multimodal learning in recent years [^3] [^20] [^58] [^57] [^2] [^22] [^38] [^66] [^6] [^37]. Among various research directions, *multimodal embedding learning* has emerged as a fundamental paradigm, aiming to project heterogeneous modalities—such as images, text, audio, and video—into a shared representation space. Such unified embeddings enable a broad range of downstream tasks, including cross-modal retrieval [^60] [^76], video understanding [^67] [^62], and recommendation [^9]. Existing approaches can be broadly categorized into two families: (1) *dual-tower architectures*, which employ independent modality-specific encoders to map each modality to the joint space; and (2) *multimodal large language model (MLLM)-based architectures*, which integrate all modalities into a unified Transformer framework for joint modeling.

### 2.1 Dual-Tower Multimodal Embedding Models

Dual-tower architectures, pioneered by CLIP [^47], employ separate encoders for each modality (e.g., a Vision Transformer [^14] for images and a Transformer-based language model [^54] for text), projecting them into a common embedding space. Training typically relies on large-scale contrastive learning, maximizing the similarity between paired samples and minimizing it for mismatched pairs. This decoupled design allows each encoder to be precomputed and cached, enabling highly efficient retrieval at inference time and scaling well to billions of items. Following CLIP, numerous extensions have been proposed. ALIGN [^26] scales the model capacity and dataset size to improve representation quality. AudioCLIP [^21] incorporates an additional audio branch, while CLIP4Clip [^41] adapts the architecture for video-text retrieval by encoding temporal information. More recently, BLIP [^35] and BLIP-2 [^36] bridge dual-tower and fusion-style paradigms: BLIP adopts a bootstrapped pre-training strategy that unifies vision-language understanding and generation, whereas BLIP-2 introduces a lightweight Q-Former to better align visual features with frozen large language models. SigLIP [^71] further improves contrastive training by replacing the softmax cross-entropy with a sigmoid loss, mitigating the inefficiencies of batch-dependent normalization and allowing more stable training on large-scale noisy datasets. Other works [^11] [^30] investigate robust pretraining on noisy web-scale data by leveraging advanced filtering, tokenization, and data curation strategies. Despite their efficiency and scalability, dual-tower architectures generally fuse modalities only in the final embedding space, which limits their ability to capture fine-grained token-level interactions, temporal dynamics, or higher-level multimodal reasoning required in complex tasks.

### 2.2 MLLM-based Multimodal Embedding Models

In contrast, MLLM-based approaches [^74] [^29] [^42] [^50] [^27] [^60] [^76] [^40] [^4] aim to integrate all modalities within a unified sequence modeling framework, leveraging the generative and reasoning capabilities of large language models. Typically, modality-specific encoders (e.g., visual or audio spectrogram encoders) transform raw inputs into embeddings, which are then aligned to the language token space via linear projections or learned adapters. Representative works include VLM2Vec [^29] and GME [^74]. VLM2Vec [^29] generates fixed-dimensional embeddings for arbitrary combinations of images and text under task instructions, building upon Phi-3.5-V [^1]. Its successor, VLM2Vec-v2 [^42], further extends the framework to support videos and visual documents. GME [^74] constructs an MLLM-based dense retriever to enable unified cross-modal search. Subsequent research has sought to improve this paradigm: mmE5 [^5] leverages synthetic datasets for stronger multilingual performance, MoCa [^4] introduces bidirectional attention through continual pre-training to enhance scalability with both model size and training data, UniMoCo [^46] proposes a modality-completion module that infers visual features from text to address modality-missing issues, UniME [^16] employs discriminative knowledge distillation from a powerful LLM teacher to improve the embedding quality of the language component, and NoteLLM-2 [^73] explores leveraging multimodal large representation models for recommendation.

By naturally modeling cross-modal dependencies through Transformer-based self-attention [^54], this paradigm enables deeper semantic understanding, contextual reasoning, and compositionality—capabilities that dual-tower models often lack. Nonetheless, most existing methods remain constrained to image and text modalities, falling short of fully supporting omni-modal understanding.

## 3 Methodology

In this section, we present the construction of the proposed SAIL-Embedding model. We first describe the collected datasets in § 3.1, together with strategies for balancing heterogeneous data and a hard negative mining method to strengthen representation learning. We then introduce the model architecture in § 3.2, highlighting the fusion of different modalities and the use of prompts to fully leverage the capabilities of multimodal large language models. Finally, we define the training objectives in § 3.3 and outline several techniques designed to improve the training effectiveness.

### 3.1 Data Preparation and Preprocessing

#### 3.1.1 Recommendation-aware Data Construction

The data construction goal for SAIL-Embedding is to enable the model to provide omni-modal understanding capabilities for practical applications, meeting recommendation demands across diverse scenarios, such as Douyin videos and Douyin live. We curate a large-scale dataset of over 10B samples, with statistical details provided in Table 1. Mostly, each sample is a pair consisting of a query and a target for CLIP-like contrastive learning. The query can be a video or just a couple of words, as well as the target. Different training datasets are designed with specific philosophies to encompass diverse content and collaboration semantics. To this end, the meta-task categories for training data are summarized as follows:

Table 1: SAIL-Embedding Training Data Statistics. V, T, and A represent vision, text, and audio modalities, respectively. The training data encompasses multi-faceted retrieval tasks across queries and items with omni-modal information, as well as multi-level label classification tasks. “RSDF” means the Recommendation-side Dense Features.

<table><tbody><tr><td>Meta Task</td><td>Data Partition</td><td>Design Philosophy</td><td>Query</td><td>Target</td><td>Data Magnitude</td></tr><tr><td rowspan="7">Item to ItemRetrieval</td><td>Copair-i2i</td><td>User consumption behaviour-based pair</td><td>V+T+A</td><td>V+T+A</td><td>2.9B</td></tr><tr><td>Search-i2i</td><td>User search behaviour-based pair</td><td>V+T+A</td><td>V+T+A</td><td>0.6B</td></tr><tr><td>Live-i2i</td><td>Live streaming-based behavioral copair</td><td>V+T+A</td><td>V+T+A</td><td>1.4B</td></tr><tr><td>Summary-i2i</td><td>Ngram-based filtering pair</td><td>V+T+A</td><td>V+T+A</td><td>1.1B</td></tr><tr><td>Hashtag-i2i</td><td>User submission-based pair</td><td>V+T+A</td><td>V+T+A</td><td>0.05B</td></tr><tr><td>RSDF-i2i</td><td>RSDF similarity-based filtering pair</td><td>V+T+A</td><td>V+T+A</td><td>0.3B</td></tr><tr><td>ID-i2i</td><td>ID clustering-based pair</td><td>V+T+A</td><td>V+T+A</td><td>0.08B</td></tr><tr><td rowspan="2">Query to ItemRetrieval</td><td>Search-q2i</td><td>Search query and click-based pair</td><td>T</td><td>V+T+A</td><td>1.8B</td></tr><tr><td>Score-q2i</td><td>Pairs of queries and targets with labeled similarity scores for downstream training</td><td>T</td><td>V+T+A</td><td>0.6B</td></tr><tr><td>Classification</td><td>CLS</td><td>Item information and corresponding multi-level tags</td><td>V+T+A</td><td>T</td><td>3.1B</td></tr></tbody></table>

#### 3.1.2 Dynamic Hard Negative Mining

In contrastive learning, the effectiveness of representation learning heavily depends on the quality of both positive and negative samples. While random negatives are often abundant, they tend to be semantically dissimilar to the query, making the discrimination task trivial and limiting the model’s ability to capture fine-grained distinctions. Hard negatives—samples that are challenging to distinguish from positives due to high semantic similarity—play a crucial role in improving model robustness and retrieval performance. However, the notion of “hard” is dataset- and task-dependent, and applying a fixed global similarity threshold often leads to suboptimal results. To address this, we propose a dynamic hard negative mining strategy that adaptively determines the optimal similarity threshold for each dataset.

Formally, let $\mathcal{P}=\{(q_{i},t_{i})\}_{i=1}^{N}$ denote the set of positive pairs, where $q_{i}$ is a query and $t_{i}$ is its corresponding target. Negative pairs are constructed via a Cartesian product excluding positives:

$$
\mathcal{N}=\{(q_{i},t_{j})\mid i\neq j,(q_{i},t_{j})\notin\mathcal{P}\}.
$$

We then compute cosine similarity scores $s_{ij}=\cos(q_{i},t_{j})$ for all negative pairs $(q_{i},t_{j})\in\mathcal{N}$. Merging positive and negative pairs, we obtain a dataset of $(s,y)$, where $y\in\{1\text{ (positive)},0\text{ (negative)}\}$. For each candidate threshold $\lambda$, the binary prediction is defined as:

$$
\hat{y}_{ij}(\lambda)=\begin{cases}1,&s_{ij}\geq\lambda,\\
0,&s_{ij}<\lambda.\end{cases}
$$

Precision and recall are computed accordingly, and the F1 score is defined as:

$$
\text{F1}(\lambda)=\frac{2\cdot\text{Precision}(\lambda)\cdot\text{Recall}(\lambda)}{\text{Precision}(\lambda)+\text{Recall}(\lambda)}.
$$

The optimal similarity threshold is selected by:

$$
\lambda^{*}=\arg\max_{\lambda}\text{F1}(\lambda),
$$

which is then used to identify and mine hard negatives dynamically during training. This adaptive strategy ensures dataset- and task-specific selection of challenging negatives, improving contrastive representation learning and downstream retrieval performance.

Once $\lambda^{*}$ is determined, we construct the hard negative set $\mathcal{H}$ by selecting samples whose similarity scores are less than $\lambda^{*}$ yet among the highest below this threshold:

$$
\mathcal{H}=\left\{x_{i}\mid s(x_{i})<\lambda^{*},\ s(x_{i})\ \text{is among the highest below}\ \lambda^{*}\right\},
$$

where $s(x_{i})$ denotes the similarity score of sample $x_{i}$. This filtering step removes overly similar samples, which we regard as false negatives. For each query $q_{i}$, we sample positives from $\mathcal{P}$, hard negatives from $\mathcal{H}$ and in-batch random negatives from $\mathcal{N}$. The contrastive loss is then defined as:

$$
\mathcal{L}_{\text{contrast}}=-\sum_{i=1}^{N}\left[\log\frac{\exp(s_{ii}/\tau)}{\exp(s_{ii}/\tau)+\sum_{(q_{i},t_{j})\in\mathcal{H}}\exp(s_{ij}/\tau)+\sum_{(q_{i},t_{j})\in\mathcal{N}}\exp(s_{ij}/\tau)}\right],
$$

where $s_{ii}$ is the similarity of the positive pair, $\tau$ is a temperature hyper-parameter. This formulation adaptively enforces discrimination against semantically close but incorrect targets, thereby improving robustness and generalization of the learned representations.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x3.png)

Figure 3: Illustration of Adaptive Multi-Source Data Balancing. For both training and validation datasets, we compute embeddings for different modalities and cluster them. We then construct a similarity matrix to represent the distance between training and validation clusters, which is further aggregated into a single similarity score. Based on these similarity values, each training dataset is assigned a weight using a Softmax function.

#### 3.1.3 Adaptive Multi-Source Data Balancing

Conventional multi-source training pipelines often rely on manually assigned dataset mixing ratios, determined by subjective expertise and task intuition. Such heuristic configurations are difficult to validate empirically and may lead to suboptimal generalization. We introduce an adaptive weighting framework that learns dataset-specific sampling weights directly from the data distribution, rather than from human-designed heuristics. The core idea is to measure the semantic similarity between high-quality benchmark validation sets and pre-training datasets, and to translate this similarity into flexible sampling weights for multi-source training. Compared with hard filtering approaches that remove entire samples based on a single-instance similarity score, our method performs *soft selection* at the dataset level, preserving distributional diversity while maintaining overall data quality. This prevents overfitting to the benchmark domain and avoids distribution collapse, thereby improving generalization to unseen tasks.

Our pipeline works as illustrated in Figure 3. We first construct validation subsets from the training datasets with distributions similar to the downstream test tasks. Given multiple training datasets and one or more benchmark validation datasets, we extract embeddings using an early version of our model. To reduce computation, we randomly sample approximately 10k samples per dataset and apply $k$ -means clustering to obtain cluster centroids. For each dataset pair, we compute a cosine similarity matrix $C$ between their centroids. We then reduce $C$ to a scalar similarity score via the Sinkhorn algorithm [^8], i.e., a weighted sum $\sum P\odot C$, where $P$ is a transport matrix derived from the distance matrix $1-C$ using the Sinkhorn algorithm. This formulation assigns higher weights to more similar clusters. Next, we apply a “fusion first” strategy to determine the optimal modality combination, preferring fused modalities when available, and otherwise selecting unimodal representations. Finally, we compute the similarity between each training set $M_{i}$ and each benchmark $N_{j}$, producing an $m\times n$ similarity matrix. Averaging and normalizing across benchmarks yields the final adaptive sampling weight for each training set.

This method offers three benefits: (1) it learns weights directly from data distributions, reducing reliance on subjective manual tuning; (2) it is modular and can be generalized to other multi-source learning settings; (3) it maintains a balance between high data quality and distributional diversity, mitigating overfitting and enhancing robustness.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x4.png)

Figure 4: Overview of Our SAIL-Embedding Model. For each sample, we extract relevant textual information (e.g., titles, OCR text) and combine it with instructions to obtain textual tokens. Visual and audio information are encoded using dedicated encoders, and the resulting multimodal tokens are concatenated and passed through a fusion module to obtain a unified multimodal embedding. The system is trained by contrasting embeddings from query and target samples.

### 3.2 Architecture Design

#### 3.2.1 Overall Architecture

The core idea of SAIL-Embedding is to transform heterogeneous multimodal information into a unified embedding space, enabling robust cross-modal understanding and retrieval. As illustrated in Figure 4, given an input sample $x$ containing audio $a$, visual $v$, and textual $t$ signals, our framework leverages a large language model (LLM) [^63] [^53] as the central reasoning and integration backbone, which is warmed up following the work [^13].

For the text modality, we adopt conventional preprocessing pipelines, including tokenization and mapping each token to its corresponding word embedding via a trainable embedding layer. For visual and audio modalities, we follow the “foreign language” metaphor: each non-text modality is processed by a modality-specific encoder, namely $\mathcal{E}_{v}$ for vision and $\mathcal{E}_{a}$ for audio, to project their raw features into a natural language-compatible embedding space. These embeddings are then aligned in both dimension and semantics before being fed into the LLM for multimodal fusion. The final representation is extracted from the output of the LLM using mean pooling over all token embeddings.

This design allows for flexible integration of diverse modalities and provides a unified interface for knowledge transfer from large pre-trained language models to multimodal scenarios, without requiring extensive architecture modifications.

#### 3.2.2 Text Tokenizer

![Refer to caption](https://arxiv.org/html/2510.12709v3/x5.png)

Figure 5: Illustration of instructions for different tasks. We design task-specific instructions by explicitly defining the task and its objective. Audio, image, and text tokens are then provided jointly to obtain the final results. For query and target tokens, modality-specific adaptations are applied to accommodate their respective modality combinations.

Real-world short videos contain abundant textual information, such as titles, tags, author labels, OCR texts, and ASR texts. The mainly used fields are described as follows:

1. Title: Title text information of the short video.
2. OCR Text: Textual information recognized from the short video frames.
3. ASR Text: Textual information obtained by converting audio from the short video.
4. Nickname: The author’s nickname associated with the short video.
5. Tags: Text labels generated by tagging models for each item.

We organize these signals into a unified textual format and concatenate them for downstream modeling. Specifically, the title, ASR, and OCR text from the query and candidates, along with additional nickname and tags, are selected as the text-modal data. The processing workflow first performs cleaning and deduplication, followed by the random dropping of partial fields in practice to accommodate scenarios where original fields are missing during online deployment.

Moreover, to effectively leverage the knowledge of the multimodal language model, we design prompts that guide the model to process multimodal information more effectively. As illustrated in Figure 5. Each prompt consists of three parts. The first is the system prompt, which defines the task and specifies its objective, helping the model better interpret and represent subsequent content. Next, we insert the extracted image, text, and audio tokens as user-provided information. Finally, we append the assistant symbol to indicate that the model should generate an answer. In addition, since tasks vary, the instructions for query and target samples are carefully designed to differ, enabling the system to interpret the provided information appropriately for each case.

#### 3.2.3 Vision Encoding Module

For the visual encoding, we adopt a Vision Transformer (ViT)-based backbone [^70]. In the case of video data, each frame is independently patchified, generating a sequence of spatio-temporal tokens. We use a patch size of 14 and resize all frames to a uniform resolution to address resolution variation. Although such dense tokenization preserves fine-grained visual details, it also produces an excessive number of tokens, especially for high-resolution or long-duration videos, thereby imposing a substantial computational burden on downstream fusion.

To address this challenge, we introduce a Visual Perceiver module, inspired by perceiver architectures [^25], serving as a learnable bottleneck for token reduction. Concretely, we concatenate the visual tokens with $N_{q}=16$ learnable latent query tokens, feed them into a Transformer block, and only retain the query token embeddings as the condensed visual representation. This mechanism preserves essential semantic content while significantly reducing sequence length, thereby improving both efficiency and scalability.

#### 3.2.4 Audio Encoding Module

The audio modality processing typically has effective candidates, such as Qwen-audio [^7], Whisper [^48], and CLAP models [^15]. We empirically employ the CLAP model, which is faster than models like Whisper, to balance the efficiency of online deployment with mitigating the long-tail distribution issue in post-sampling audio sequence lengths. This method can extract high-level acoustic semantics to obtain discriminative audio tokens. The pipeline is designed to accommodate diverse audio clip lengths:

1. Short audio ($\leq 10$ s): We apply a *repeat-and-pad* operation to normalize the length before feature extraction, producing a single $1\times\text{dim}$ audio token.
2. Long audio ($>10$ s): We segment the waveform into consecutive non-overlapping 10s chunks, extract features for each chunk with CLAP, and aggregate them (via mean pooling) into a unified $1\times\text{dim}$ representation.

The resulting single audio token is appended to the multimodal token sequence before fusion, ensuring a consistent representation format regardless of the clip length.

#### 3.2.5 Fusion Module

Let $\mathbf{T}_{a}$, $\mathbf{T}_{v}$, and $\mathbf{T}_{t}$ denote the token sequences obtained from the audio, visual, and textual modalities, respectively. We concatenate the unimodal token sequences to form a single multimodal sequence $\mathbf{T}=[\mathbf{T}_{a};\mathbf{T}_{v};\mathbf{T}_{t}]$, where $[\cdot\penalty 10000\ ;\cdot]$ denotes concatenation along the token dimension. The fused sequence $\mathbf{T}$ is then fed into a multimodal Transformer fusion module (i.e., an LLM backbone [^63]), denoted as $\mathcal{F}(\cdot)$, which performs cross-modal reasoning via self-attention layers. To strengthen information exchange across modalities and capture long-range dependencies, we employ a bi-directional attention mechanism. The resulting hidden states are further normalized by a tanh activation, ensuring embeddings remain within a bounded range, and subsequently aggregated by mean pooling to form the final multimodal embedding:

$$
\mathbf{z}=\text{Meanpool}\big(\tanh(\mathcal{F}(\mathbf{T}))\big).
$$

### 3.3 Training Strategies

#### 3.3.1 Content-Aware Progressive Training

To adapt general-purpose large models to specific downstream tasks, we adopt a content-aware progressive training framework. As illustrated in Figure 6, each stage uses datasets with different scales and characteristics: earlier stages employ larger and more diverse datasets, while later stages focus on higher-quality data that closely matches the downstream task requirements. Specifically, in the first stage, we train a base model on large-scale and diverse datasets to acquire fundamental multimodal representation capabilities. Next, we fine-tune the model on a subset of datasets more aligned with the target downstream tasks. To further enhance the model’s ability to capture fine-grained distinctions, we construct hard negatives and perform an additional fine-tuning stage to obtain the refined model. This progressive training framework enables the model to balance general-world knowledge with downstream domain-specific knowledge, while maintaining training stability.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x6.png)

Figure 6: Illustration of Training Techniques. (a) Our progressive training framework gradually shifts from larger, diverse datasets to smaller, domain-specific datasets, balancing general-world knowledge with downstream specialization while maintaining training stability. (b) The conventional training method mixes heterogeneous data into a single batch. (c) Our stochastic specialization training randomly selects a dataset at each iteration to enhance robustness and specialization.

#### 3.3.2 Loss Definition

Our training objective follows the *contrastive learning* paradigm to jointly optimize the multimodal embedding space. The overall loss integrates four complementary components: (i) a Noise-Contrastive Estimation (NCE) loss for global alignment, (ii) a COSENT loss [^51] for fine-grained ranking, (iii) a multimodal In-Context Learning (mICL) loss [^73] to enhance modality-specific discrimination, and (iv) a late fusion loss [^73] to balance visual and textual contributions.

Given a query embedding $e_{q}$ and its positive target embedding $e_{t}^{+}$, the NCE loss enforces global alignment by contrasting against in-batch negatives:

$$
\mathcal{L}_{\mathrm{nce}}=-\log\frac{\exp\left(\cos(e_{q},e_{t}^{+})/\tau\right)}{\exp\left(\cos(e_{q},e_{t}^{+})/\tau\right)+\sum_{i=1}^{B}\exp\left(\cos(e_{q},e_{t}^{-})/\tau\right)},
$$

where $\tau$ is a learnable temperature parameter, initialized differently for each task and each dataset to account for distributional variations.

In real-world applications, storage and computational budgets impose strict constraints on the dimensionality of embedding vectors, making the deployment of high-dimensional representations prohibitive due to memory cost and retrieval latency. To overcome this limitation, we employ *Matryoshka Representation Learning* (MRL) [^32], which enforces multi-granularity supervision on embeddings of different sizes. Specifically, the full $1536$ -dimensional embedding is sliced into multiple contiguous sub-vectors of sizes $768$ and $128$, and each slice, together with the full embedding, is optimized using the same InfoNCE loss objective:

$$
\mathcal{L}_{\mathrm{nce-mrl}}=\mathcal{L}_{\mathrm{nce}}(\mathbf{z}_{128})+\mathcal{L}_{\mathrm{nce}}(\mathbf{z}_{768})+\mathcal{L}_{\mathrm{nce}}(\mathbf{z}_{1536}),
$$

where $\mathbf{z}_{d}$ denotes the sub-embedding of dimensionality $d$. This “nested” training paradigm ensures that all sub-embeddings preserve strong discriminative power, enabling the system to dynamically trade off accuracy and efficiency at inference time without retraining. Such flexibility is particularly critical for deploying resource-constrained multi-modal embedding models.

For text-oriented pre-training datasets, we incorporate a cosine-similarity–based ranking loss (COSENT) [^51] to enforce fine-grained orderings:

$$
\mathcal{L}_{\mathrm{cosent}}=\log\Bigg(1+\sum_{\mathrm{sim}(i,j)>\mathrm{sim}(k,l)}\exp\Big(\tfrac{\cos(e_{k},e_{l})-\cos(e_{i},e_{j})}{\tau}\Big)\Bigg).
$$

To mitigate modality imbalance, we adopt the multimodal In-Context Learning (mICL) [^73]. Instead of compressing multimodal inputs into a single embedding, we separate them into modality-specific embeddings (visual $\mathbf{n}_{v}$ and textual $\mathbf{n}_{t}$), and apply in-batch contrastive learning to encourage consistent alignment across modalities:

$$
\mathcal{L}_{\mathrm{micl}}=\mathcal{L}(\mathbf{n}_{v},\mathbf{n}_{v}^{+})+\mathcal{L}(\mathbf{n}_{t},\mathbf{n}_{t}^{+}),
$$

where each term follows the NCE form but within its respective modality.

In addition, we integrate a late fusion mechanism [^73] to preserve visual fidelity. Given visual embedding $\mathbf{v}$ and multimodal embedding $\mathbf{n}_{m}$, a gated fusion module learns to adaptively combine them:

$$
\mathbf{z}=\sigma\big(W[\mathbf{v},\mathbf{n}_{m}]+b\big),\quad\hat{\mathbf{n}}_{m}=\mathbf{z}\odot\mathbf{v}+(1-\mathbf{z})\odot\mathbf{n}_{m},
$$

where $\sigma$ denotes the sigmoid function. We then introduce a late-fusion contrastive loss $\mathcal{L}_{\mathrm{lf}}$ over $\hat{\mathbf{n}}_{m}$ to reinforce multimodal consistency. The final training loss is a weighted combination:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{nce-mrl}}+\lambda\,\mathcal{L}_{\mathrm{cosent}}+\alpha\,\mathcal{L}_{\mathrm{micl}}+\beta\,\mathcal{L}_{\mathrm{lf}},
$$

where $\lambda$, $\alpha$, and $\beta$ are balancing hyperparameters.

#### 3.3.3 Stochastic Specialization Training

In multi-domain training scenarios, as illustrated by Figure 6 (b), data from heterogeneous datasets are often mixed within each iteration to ensure balanced utilization. However, such domain mixing inevitably fragments the effective batch size for each individual dataset, leading to small per-domain batches and elevated gradient variance. Moreover, synchronizing features or statistics across different domains within a single iteration introduces additional communication and processing overhead, which becomes more severe as the number of datasets grows.

To address this, we propose *Stochastic Specialization Training*, a strategy inspired by meta-learning schemes that improves both supervision focus and computational efficiency. Instead of sampling from all datasets in every iteration, our method stochastically selects a single dataset according to a predefined probability distribution based on methods introduced in Section 3.1.3, and draws the entire batch from it. Across training, all datasets are still visited, but each iteration specializes on one domain, yielding larger per-domain batch sizes while keeping the global batch size unchanged.

This specialization reduces gradient variance within each training step, simplifies the iteration logic, and eliminates the need for dataset-specific processing or inter-domain communication inside the iteration. Furthermore, the approach exhibits strong scalability: adding a new dataset requires only its dataset-specific configuration, without modifying the overall batching or communication pattern.

#### 3.3.4 Dataset-Driven Pattern Matching

To address the challenge of heterogeneous modality availability and imbalance across datasets, we design a *modality-aware matching strategy* that unifies various contrastive objectives under a general query-to-target framework. Specifically, we generalize the CLIP objective beyond the canonical image-to-text setting to a comprehensive *modality-to-modality* paradigm, where any modality can serve as the query while the others become potential targets. This formulation enables flexible alignment tasks such as Image-to-Text Contrastive (ITC), Image-to-Image Contrastive (IIC), Video-to-Text Contrastive (VTC), Video-to-Video Contrastive (VVC), Text-to-Text Contrastive (TTC), and Omni-to-Omni Contrastive (OOC), the latter covering arbitrary cross-modal pairs without restriction.

A configurable data processor consolidates heterogeneous raw inputs (e.g., images, video frames, captions, ASR transcripts, OCR tokens) into a standardized set of modalities. For each training sample, we dynamically construct all valid query–target pairs according to predefined matching patterns that account for modality characteristics. Modalities without valid patterns are excluded from the loss to mitigate noise propagation. Unlike static formulations that fix the number of losses per dataset, our *dynamic multi-pattern* matching evaluates all feasible query–target pairs for each sample within the same forward pass. This not only maximizes the utilization of extracted embeddings but also improves optimization stability, as reflected by smoother convergence in training loss according to our experiments.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x7.png)

Figure 7: Recommendation Enhancement Training. We implement the (a) sequence-to-item distillation and (b) ID-to-item distillation to enhance the SAIL-Embedding’s collaboration-aware capabilities.

#### 3.3.5 Collaboration-aware Recommendation Enhancement Training

To address the demands of downstream recommendation scenarios for capturing user interests and to overcome the limitations of existing single-video content understanding, inspired by [^10], we propose a collaboration-aware recommendation enhancement training strategy.

Sequence-to-Item Distillation. In data construction, we select user query sequence-to-target video pairs from four perspectives. Specifically, we first choose the users’ historical video viewing sequence (1k) and filter it based on labels indicating positive interaction behaviours. The most recent item in the sequence is designated as the target video.

1. Content-aware Single-Peak Interest Modelling: This part of the sequence data is retained by satisfying videos with at least three positive behaviours to mine items with compact interest distributions. Subsequently, items are further filtered based on similarity in content representations.
2. Content-aware Multi-Peak Interest Modelling: This part of the sequence data is retained by satisfying videos with any positive behaviour to mine items with widely distributed interests. Then, we select sequence items and the target item based on the Jaccard coefficient, choosing those with behaviour label thresholds greater than 0.5.
3. Collaboration-aware Single-Peak Interest Modelling: This part of the sequence data is retained by satisfying at least three positive behaviours. The Jaccard coefficient is still employed to measure behaviour consistency with the target item. We perform label-based clustering to select items that satisfy the original user interaction distribution.
4. Collaboration-aware Multi-Peak Interest Modelling: This part of the sequence data is retained by satisfying at least one positive behaviour. Subsequently, we select items corresponding to the proportional distribution of clustered videos as the final sequence data.

As shown in Figure 7(a), we perform contrastive learning by using the representations of the query sequence and the target video. In practice, we find two effective sequence modeling approaches: the mean pooling and the sequence encoder. The former inputs each video in the query sequence into an embedding model to obtain a sequence of query video embeddings, which are then aggregated via pooling. The latter employs a three-layer transformer module to construct a sequence encoder, using a special token to extract the sequence’s overall representation.

ID-to-Item Distillation. In Figure 7(b), we attempt to directly align the model’s output with the recommendation-side representations. In practice, we jointly utilize multiple ID embeddings of each item from the recommender system, aligning them with the omni-modal representation via feature projection to perform feature distillation. Simultaneously, we introduce an auxiliary i2i retrieval task and perform optimization in a multi-task manner to prevent the representation distribution from overly sacrificing content-aware perception capabilities. The above joint training enhances the model’s ability to aggregate video content and mine user interests, thereby better adapting to recommendation scenarios.

## 4 Experiments and Results

### 4.1 Configuration Settings

SAIL-Embedding is trained across multiple NPUs, using DeepSpeed ZeRO2 [^49] and gradient checkpointing strategies. In the modality components, the latent dimension, number of layers, and sampling depth of the visual perceiver are set to 1024, 16, and 6 respectively, with the self-attention enabled. Audio tokens extracted from the CLAP model [^15] have an original dimension of 512 and are mapped via an adapter to a 1536-dimensional alignment hidden state. For the model optimization, we employ the FusedAdam optimizer, which supports mixed-precision computation and gradient fusion, significantly enhancing computational efficiency while maintaining training accuracy. The initial learning rate and weight decay coefficient are set to 1e-5 and 1e-4 to mitigate overfitting risks. The learning rate schedule employs a cosine annealing strategy, with the variation range constrained between 1e-5 and 6e-6 to prevent gradient explosion. In addition, a warm-up strategy is applied during the initial phase to stabilize optimization, and gradient clipping is used to further ensure training stability by limiting gradient norms. In the seq2item distillation, we default to the mean pooling method and set the sequence length to 10 for efficient training. The sequence dataset representing diverse interests comprises 11M samples. The main task dataset in the ID2item distillation contains 220M samples, while the auxiliary task dataset holds 20M samples.

We compare the CLIP-based model [^47] and the standard VLM-based approach [^74] mainly on a wide range of item-to-item (i2i) tasks. Both are fine-tuned on the training data in Table 1 to ensure fair comparisons. Additionally, performance comparisons on query-to-item (q2i) tasks further involve unimodal embedding models, i.e., Doubao-Embedding and Qwen3-Embedding [^75].

### 4.2 Evaluation Metrics

In recommendation scenarios, traditional evaluation metrics for multimodal embeddings often fail to capture their true business impact. A multimodal embedding model may achieve strong standalone performance but contribute little to downstream user engagement. Moreover, recommendation systems evolve rapidly, where inefficient trial-and-error without robust pre-deployment evaluation risks suboptimal online performance. These challenges motivate the design of a multi-dimensional evaluation framework tailored to multimodal embedding models in recommendation tasks.

Given a dataset $\{q_{i},t_{i}\}_{i=1}^{N}$, where $q_{i}$ denotes a query item and $t_{i}$ its matched target, a multimodal encoder $\mathcal{E}$ produces embeddings $\mathcal{E}(q_{i}),\mathcal{E}(t_{i})$. The objective is to assess whether the resulting embedding space $(\mathcal{Q},\mathcal{T})$ exhibits desirable properties: structural similarity, stable ranking, and strong discriminability. Our evaluation protocol consists of four complementary dimensions:

Table 2: Performance Comparison on i2i Tasks. We consider four categories of realistic applications, including content understanding, search, and collaborative perception scenarios.

<table><tbody><tr><td></td><td></td><td></td><td colspan="3">Models</td></tr><tr><td>Scenarios</td><td>Task</td><td>Metric</td><td>CLIP-based Model <sup><a href="#fn:47">47</a></sup></td><td>VLM-based Model <sup><a href="#fn:74">74</a></sup></td><td>SAIL-Embedding</td></tr><tr><td></td><td></td><td>Recall@50</td><td>17.83</td><td>19.99</td><td>20.76</td></tr><tr><td></td><td>University Sub-i2i</td><td>Recall@100</td><td>21.27</td><td>24.68</td><td>24.46</td></tr><tr><td></td><td></td><td>Recall@50</td><td>36.64</td><td>60.71</td><td>62.26</td></tr><tr><td></td><td>Travel Sub-i2i</td><td>Recall@100</td><td>45.87</td><td>70.61</td><td>72.40</td></tr><tr><td></td><td></td><td>Recall@50</td><td>75.98</td><td>81.32</td><td>89.08</td></tr><tr><td></td><td>Film-i2i</td><td>Recall@100</td><td>80.40</td><td>84.79</td><td>91.86</td></tr><tr><td></td><td></td><td>Recall@50</td><td>95.31</td><td>98.75</td><td>98.13</td></tr><tr><td></td><td>Store Visit (Restaurant)-i2i</td><td>Recall@100</td><td>96.69</td><td>99.07</td><td>98.69</td></tr><tr><td></td><td></td><td>Recall@50</td><td>16.90</td><td>22.09</td><td>26.67</td></tr><tr><td></td><td>Store Visit (City)-i2i</td><td>Recall@100</td><td>22.00</td><td>27.65</td><td>32.10</td></tr><tr><td></td><td></td><td>Recall@50</td><td>58.00</td><td>62.19</td><td>62.35</td></tr><tr><td></td><td>Hot Topic-i2i</td><td>Recall@100</td><td>74.43</td><td>80.56</td><td>77.98</td></tr><tr><td></td><td></td><td>Recall@50</td><td>71.38</td><td>70.17</td><td>78.07</td></tr><tr><td></td><td>Comment-i2i</td><td>Recall@100</td><td>82.16</td><td>80.57</td><td>87.62</td></tr><tr><td></td><td></td><td>Recall@50</td><td>74.38</td><td>85.30</td><td>84.55</td></tr><tr><td></td><td>Music Play-i2i</td><td>Recall@100</td><td>79.90</td><td>89.25</td><td>88.72</td></tr><tr><td></td><td></td><td>Recall@50</td><td>71.36</td><td>82.52</td><td>82.08</td></tr><tr><td></td><td>Music Gameplay-i2i</td><td>Recall@100</td><td>77.25</td><td>86.98</td><td>86.72</td></tr><tr><td></td><td></td><td>Recall@50</td><td>67.78</td><td>71.15</td><td>72.83</td></tr><tr><td></td><td>Game Tag-i2i</td><td>Recall@100</td><td>72.15</td><td>74.92</td><td>76.93</td></tr><tr><td></td><td></td><td>Recall@50</td><td>26.54</td><td>45.97</td><td>52.03</td></tr><tr><td></td><td>Brand Vehicle-i2i</td><td>Recall@100</td><td>33.15</td><td>52.22</td><td>57.34</td></tr><tr><td></td><td></td><td>Recall@50</td><td>36.01</td><td>66.31</td><td>72.54</td></tr><tr><td></td><td>Brand Phone-i2i</td><td>Recall@100</td><td>46.84</td><td>76.68</td><td>81.64</td></tr><tr><td></td><td></td><td>Recall@50</td><td>71.00</td><td>79.31</td><td>82.56</td></tr><tr><td></td><td>Spot-i2i</td><td>Recall@100</td><td>75.23</td><td>81.72</td><td>84.64</td></tr><tr><td></td><td></td><td>Recall@50</td><td>67.36</td><td>76.12</td><td>75.52</td></tr><tr><td></td><td>Summary-i2i</td><td>Recall@100</td><td>77.48</td><td>85.60</td><td>85.56</td></tr><tr><td></td><td></td><td>Recall@50</td><td>31.94</td><td>48.33</td><td>74.87</td></tr><tr><td></td><td>Sub-i2i</td><td>Recall@100</td><td>44.59</td><td>61.51</td><td>81.96</td></tr><tr><td></td><td></td><td>Recall@50</td><td>57.27</td><td>49.66</td><td>56.74</td></tr><tr><td>ContentUnderstanding</td><td>ID Fusetag-i2i</td><td>Recall@100</td><td>66.99</td><td>60.13</td><td>67.45</td></tr><tr><td></td><td></td><td>Recall@50</td><td>56.72</td><td>77.45</td><td>80.50</td></tr><tr><td></td><td>Video Search-i2i</td><td>Recall@100</td><td>62.23</td><td>80.90</td><td>84.01</td></tr><tr><td></td><td></td><td>Recall@50</td><td>39.41</td><td>65.79</td><td>68.43</td></tr><tr><td>Search</td><td>Search-i2i</td><td>Recall@100</td><td>44.63</td><td>70.58</td><td>73.00</td></tr><tr><td></td><td></td><td>Recall@50</td><td>49.76</td><td>42.52</td><td>52.46</td></tr><tr><td></td><td>RSDF-i2i</td><td>Recall@100</td><td>55.79</td><td>48.44</td><td>59.06</td></tr><tr><td></td><td></td><td>Recall@50</td><td>53.47</td><td>66.06</td><td>69.17</td></tr><tr><td></td><td>Copair-i2i</td><td>Recall@100</td><td>70.6</td><td>72.70</td><td>75.57</td></tr><tr><td></td><td></td><td>Recall@50</td><td>47.44</td><td>52.55</td><td>54.10</td></tr><tr><td>CollaborativePerception</td><td>Live-i2i</td><td>Recall@100</td><td>56.69</td><td>62.60</td><td>63.80</td></tr></tbody></table>

Table 3: Performance Comparison on q2i Tasks. We consider retrieval and classification applications across nine tasks.

<table><tbody><tr><td>Task</td><td>Metric</td><td>Doubao-Embed.</td><td>Qwen3-Embed.-4B</td><td>Qwen3-Embed.-8B <sup><a href="#fn:75">75</a></sup></td><td>CLIP-based Model <sup><a href="#fn:47">47</a></sup></td><td>VLM-based Model <sup><a href="#fn:74">74</a></sup></td><td>SAIL-Embedding</td></tr><tr><td rowspan="2">Short Video-q2i</td><td>Recall@50</td><td>61.93</td><td>71.73</td><td>72.42</td><td>74.16</td><td>78.53</td><td>86.53</td></tr><tr><td>Recall@100</td><td>64.01</td><td>75.17</td><td>75.77</td><td>76.21</td><td>80.66</td><td>88.54</td></tr><tr><td>Decision-q2i</td><td>AUC</td><td>74.51</td><td>69.79</td><td>69.39</td><td>65.18</td><td>67.34</td><td>82.44</td></tr><tr><td>Longtail-q2i</td><td>AUC</td><td>85.84</td><td>83.90</td><td>83.96</td><td>83.18</td><td>84.02</td><td>91.22</td></tr><tr><td>Search Longtail-q2i</td><td>AUC</td><td>76.86</td><td>74.90</td><td>75.34</td><td>74.82</td><td>74.99</td><td>83.31</td></tr><tr><td>Unbiased-q2i</td><td>AUC</td><td>88.67</td><td>86.03</td><td>86.11</td><td>88.62</td><td>88.06</td><td>93.86</td></tr><tr><td>Unbiased Longtail-q2i</td><td>AUC</td><td>85.17</td><td>82.01</td><td>82.32</td><td>85.06</td><td>85.86</td><td>89.10</td></tr><tr><td>Biased-q2i</td><td>AUC</td><td>87.79</td><td>85.92</td><td>86.10</td><td>88.65</td><td>89.02</td><td>91.65</td></tr><tr><td rowspan="2">Live Summary2i</td><td>Recall@50</td><td>66.31</td><td>70.59</td><td>69.79</td><td>78.25</td><td>81.55</td><td>84.33</td></tr><tr><td>Recall@100</td><td>69.31</td><td>74.45</td><td>73.85</td><td>82.55</td><td>85.62</td><td>88.16</td></tr><tr><td rowspan="2">Live-q2i</td><td>Recall@50</td><td>59.91</td><td>64.23</td><td>64.22</td><td>74.24</td><td>73.91</td><td>79.08</td></tr><tr><td>Recall@100</td><td>64.76</td><td>69.78</td><td>69.57</td><td>79.89</td><td>79.34</td><td>84.38</td></tr></tbody></table>

1. Retrieval Recall. We compute recall@k to quantify retrieval performance:
	$$
	\text{Recall@k}=\frac{\text{Number of relevant items retrieved in top-k}}{\text{Total number of relevant items}}.
	$$
	This is applied to i2i and q2i benchmarks, with $k\in\{1,10,25,50,100\}$.
2. Positive–Negative Separability. For each query, we compute cosine similarity with its ground-truth target and with randomly sampled negatives. The distributional gap between positive and negative similarities reflects the model’s discriminative capability.
3. Group-wise Clustering Consistency. We cluster the embeddings of positive and negative samples offline and adopt *Normalized Mutual Information* (NMI) to measure consistency:
	$$
	\text{NMI}(L_{q},L_{t})=\frac{2I(L_{q};L_{t})}{H(L_{q})+H(L_{t})},
	$$
	where $L_{q}$ and $L_{t}$ denote cluster assignments, $I(\cdot;\cdot)$ is mutual information, and $H(\cdot)$ denotes entropy. A higher NMI indicates that positive and negative samples are distinguished by similar feature dimensions in the multimodal semantic space. For example, both positive and negative samples may cluster into the category “anime,” yet user preferences differ with respect to this content. This consistency highlights two key properties: (1) *Effectiveness of modality information* — multimodal features capture the actual decision basis underlying user behavior (e.g., a user may like or dislike anime due to its “art style”); (2) *Strong interpretability of representations* — the observed consistency suggests that multimodal features bear a potential causal relationship with user behavior, rather than reflecting only superficial correlations.
4. Ranking Consistency. We assess whether query–target similarity rankings remain stable across spaces. Top- $k$ overlap and Kendall’s $\tau$ are used:
	$$
	\tau=\frac{C-D}{C+D},
	$$
	where $C$ and $D$ are the numbers of concordant and discordant pairs, respectively. Higher $\tau$ indicates better preservation of relative semantic ordering.
5. Bijective Alignment Test. We verify whether embeddings exhibit near one-to-one mapping. Specifically, for each query, we retrieve its most similar target, and from this target, retrieve back its most similar query. The proportion of cases where the indices match quantifies embedding fidelity and bijective consistency.
6. AUC Measurement. Additionally, we report the AUC metric on recommendation-oriented tasks.

Our comprehensive evaluation suite provides not only a quantitative measure of embedding quality but also actionable diagnostics on the suitability of multimodal embedding models for integration into large-scale recommendation systems.

### 4.3 Evaluation Dataset Introduction

We curate several evaluation benchmarks to assess our models. Below is the detailed introduction:

For item2item evaluation, we build three types of benchmarks:

- Content Understanding. To evaluate the basic ability to capture semantic similarities, we integrate items with similar visual cues and textual information into pairs according to various rules. Except for the test split of training datasets, we collect some out-of-domain i2i benchmarks to evaluate the generalizability of our model. Leveraging the auto-tagging pipelines and expert models, we group items with the same brands or intellectual property, resulting in Brand Vehicle-i2i, Brand Phone-i2i, Film-i2i and Game Tag-i2i. Moreover, we employ MLLMs to summarize key information for items, and group them with the same n-grams or keywords, resulting in Keywords-i2i and Summary-i2i.
- Search-Based. Taking user actions into consideration, we pair search-and-click items with the same queries. Specifically, Search-i2i consists of items with various forms, e.g., videos and photos. While video search-i2i only contains video items, which are mostly delivered on Douyin.
- Collaborative Perception. To evaluate models’ ability in recommendation scenarios, we collect item pairs based on the co-occurrence mechanism. We construct datasets for video items and live streaming items, respectively. Besides, we directly leverage dense features of the recommender system to compute the cosine similarity scores between items. Items with scores exceeding a particular score are reserved as positive pairs.

For query2item evaluation, we build two types of benchmarks:

- Query2item Retrieval Datasets. Short Video-q2i is an in-domain dataset which is similar to the Search-q2i dataset for training. Besides, we collect two out-of-domain benchmarks from live streaming scenario. Live-q2i utilizes user input search text as queries while Live-summary2i uses the summarized texts of each item.
- Query2item Classification Tasks. To assess the model’s ability to discriminate similar queries and items, we further collect six embedding-based classification datasets from different downstream scenarios. For evaluation, we acquire cosine similarity scores between all queries and items with their embeddings. The scores are regarded as the binary classification probabilities, which predict whether a query and an item are paired positives or unpaired negatives. The AUC metric is computed as the final result. In these datasets, all items contain only text features.

### 4.4 Results Analysis

#### 4.4.1 Item-to-Item Retrieval Task Comparison Results

In Table 2, we present comparison results across 21 i2i tasks spanning four realistic application intents, including content understanding, search, and collaborative perception. We primarily perform omni-modal item retrieval. SAIL-Embedding significantly and consistently outperforms previous models in search and collaborative perception scenarios. Compared to the CLIP-based model, SAIL-Embedding has richer open-world knowledge and feature fusion capabilities, demonstrating stronger multimodal understanding when handling complex and dynamic business scenarios. Compared to the VLM-based model, our model verifies the importance and effectiveness of the audio modality in multimedia retrieval driven by short videos. Furthermore, the well-designed training strategies also better eliminate the performance gaps of large-scale retrieval models in industrial applications. Under the content understanding-oriented scenario, SAIL-Embedding demonstrates competitive performance across most tasks, suggesting the potential of the proposed embedding model in content-semantic learning.

#### 4.4.2 Query-to-Item Retrieval Task Comparison Results

![Refer to caption](https://arxiv.org/html/2510.12709v3/x8.png)

Figure 8: Illustration of the Results with Collaboration-aware Recommendation Enhancement (CRE). (a) Loss curve of the model fine-tuned with distilled ID embeddings. (b) Distribution of similarity scores for positive and negative samples in the original SAIL-Embedding. (c) Distribution of similarity scores after applying CRE.

In Table 3, we further report results for various retrieval methods across 9 q2i tasks, encompassing both uni-modal and vision-language models. These tasks are categorized into retrieval-oriented and classification-oriented tasks, evaluated by Recall and AUC, respectively. SAIL-Embedding significantly outperforms previous methods across all metrics for every task. A plausible explanation is that the omni-modal architecture promotes unified semantic fusion and resolves modality ambiguity, leading to stronger cross-modal semantic understanding. Furthermore, the dataset-driven pattern matching and stochastic specialization training enhance SAIL-Embedding’s cross-domain generalization capabilities and cross-task adaptability.

#### 4.4.3 Collaboration-aware Recommendation Enhancement Results

Through Collaboration-aware Recommendation Enhancement (CRE), we adapt SAIL-Embedding to better suit downstream recommendation tasks. To evaluate the effectiveness of this training phase, we report the results under different metrics in Figure 8 and Table 4. As shown in Figure 8, after incorporating Sequence/ID embedding distillation, the model converges stably after around 20k steps. We measure the separability illustrated in Figures 8(b)&(c). Before and after distillation, the model already exhibits good discrimination between positive and negative samples. However, with distillation, the overlap between positive and negative samples is further reduced and shifts rightward, verifying that the strategy effectively enhances the discriminative power of the base model, particularly for hard negatives with similarity scores around 0.5. In addition, the positive sample distribution becomes more compact, indicating that by jointly incorporating some ID embeddings, the model learns correlations among positive samples with higher confidence. This capability reflects exactly the kind of improvement expected in recommendation, where multimodal models should not only preserve separability but also strengthen relevance modeling for positive samples.

Table 4: CRE Training Quantitative Results. NMI, Ken., Inter., and Acc. represent Normalized Mutual Information in group-wise clustering consistency, Kendall’s coefficient, intersection proportion in ranking consistency, and the proportion of successful hits in the bijective alignment test, respectively. For Ken.\*, the values are multiplied by 10K for clarity.

| Models | NMI | Ken.\* | Inter. | Acc. | Gid-i2iR@50 | Gid-i2iR@100 | Copair-i2iR@50 | Copair-i2iR@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VLM-based Model | 0.53 | 68.6 | 28.29 | 46.51 | 42.52 | 48.44 | 66.06 | 72.70 |
| VLM-based Model (CRE) | 0.61 | 68.6 | 29.62 | 47.22 | 46.34 | 52.18 | 64.78 | 70.52 |
| SAIL-Embedding | 0.60 | 73.4 | 29.73 | 48.28 | 52.46 | 59.06 | 69.17 | 75.57 |
| SAIL-Embedding (CRE) | 0.65 | 77.2 | 31.63 | 48.94 | 59.69 | 66.37 | 66.83 | 73.10 |

Quantitative results are summarized in Table 4. Overall, we observe that the CRE strategy consistently improves performance across different metrics, both for VLM-based models and for our proposed SAIL-Embedding. For the NMI, the ID-distilled version achieves a $+5\%$ gain, indicating that positive and negative samples become more separable in clustering, which facilitates learning of user preference boundaries. For the ranking consistency, the distilled model improves the Kendall correlation by $+3.8\%$ and the Intersection metric by $+1.9\%$, demonstrating enhanced consistency in retrieval orderings and suggesting stronger downstream recommendation performance. For the bijective alignment test, the distilled model improves by $+0.66\%$, showing that incorporating ID embeddings encourages SAIL-Embedding to form more stable matching structures, which may provide valuable guidance for both recall and ranking stages.

Another interesting finding is that CRE training significantly improves model performance on the Gid-i2i benchmark. For instance, the CRE version of SAIL-Embedding achieves 7.23% and 7.31% improvements over the original model on Recall@50 and Recall@100 metrics. The VLM-based model also achieves an average improvement of 3.78%. This observation confirms that training via seq2item and ID2item distillation can further refine the model’s omni-modal representations to perceive collaborative semantics. This is reasonable since Gid-i2i is a paired testing benchmark constructed from gid embeddings’ relevance, which undergoes streaming updates during online execution. Additionally, we observe some performance degradation on the Copair-i2i benchmark constructed based on item-based representation similarity. We consider this variation tolerable to balance the differing application requirements of content-oriented and collaborative behavior-oriented scenarios in industrial settings.

#### 4.4.4 Sequence Modeling Evaluation Results

To verify the model’s understanding of sequence semantics after sequence distillation, we build two evaluation sets from user historical viewing sequences to target videos, named Vanilla Seq2item and Filtered Seq2item. In the former, we collect 100 historical videos viewed by the user in chronological order. The most recent video is selected as the target, while the remaining 10 videos are randomly sampled from the sequence to form the user query sequence. In the Filtered Seq2item, we extend the collected historical sequence to 1k videos and filter them based on content tags or clustered copair-ids to select items highly consistent with the target video as queries. The retained sequence length remains 10.

During implementation, we progressively perform seq2item and ID2item distillation on the content-aware training of SAIL-Embedding. Experimental results are presented in Table 5. The model after seq2item distillation achieves significant performance gains on the Filtered Seq2item. For instance, the Recall@10 and Recall@25 metrics improved by 2.67% and 4.66%, respectively. This observation indicates that the model’s sequence comprehension and content aggregation capabilities have been enhanced. Following further ID2item distillation, SAIL-Embedding consistently achieves higher performance, demonstrating that the combined distillation strategy strengthens the model’s collaboration-aware capabilities in recommendation scenarios. On the Vanilla Seq2item, the model ultimately achieves a 4.76% metric improvement. This phenomenon suggests that recommendation-enhanced learning enhances the model’s ability to extract diverse user interests.

Table 5: Sequence Modeling Evaluation Results. Both test sets evaluate the recall accuracy for the top-1, top-10, and top-25 ranked items. The best results are emphasized in bold.

<table><thead><tr><th rowspan="2">Models</th><th colspan="3">Filtered Seq2item</th><th colspan="3">Vanilla Seq2item</th></tr><tr><th>R@1</th><th>R@10</th><th>R@25</th><th>R@1</th><th>R@10</th><th>R@25</th></tr></thead><tbody><tr><th>SAIL-Embedding</th><td>8.41</td><td>19.35</td><td>25.46</td><td>4.35</td><td>10.46</td><td>14.33</td></tr><tr><th>SAIl-Embedding (Seq2item Distillation)</th><td>8.28</td><td>22.02</td><td>30.12</td><td>4.84</td><td>11.78</td><td>17.79</td></tr><tr><th>SAIl-Embedding (Seq2item + ID2item Distillation)</th><td>9.04</td><td>23.77</td><td>32.41</td><td>5.24</td><td>13.36</td><td>19.09</td></tr></tbody></table>

#### 4.4.5 Systematic Ablation Studies

To comprehensively analyze the necessity of different components and strategies within the model, we conduct systematic ablation studies on subsets of i2i and q2i tasks. Figures 9 and 10 report average performance across different tasks for intuitive observation. We first establish a baseline performance where the BERT model [^12] serves as an encoder for the text modality. Subsequently, the LLM is employed as a fuser to replace the traditional dual-tower pattern. We show consistent improvements across both tasks, with the q2i task achieving a significant gain of 5.01%. This means the LLM enhances the model’s ability to handle complex queries, boosting overall retrieval performance. When the original causal attention in the LLM is replaced with the bidirectional attention, positive incremental gains are demonstrated. The underlying reason is that the bidirectional attention captures global multimodal semantic dependencies, mitigating the semantic bias potentially introduced by the causal attention. This makes it more suitable for the representation embedding scenarios.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x9.png)

Figure 9: Ablation Studies on i2i Tasks. We systematically report on the effects of different modules and strategies on performance.

![Refer to caption](https://arxiv.org/html/2510.12709v3/x10.png)

Figure 10: Ablation Studies on q2i Tasks. We systematically report on the effects of different modules and strategies on performance.

In the initial implementation, SAIL-Embedding adopts a unified instruction format. Upon transitioning to task-specific instruction designs, we notice substantial gains. For instance, average performances on i2i and q2i scenarios increase by 2.62% and 1.88%, respectively. In practice, we observe that the instruction-based multi-task training paradigm guides the model to focus on distinct feature details. This enables adaptation to different task demands for feature expressiveness across various aspects of the omni-modal embeddings. Additionally, we investigate the differences between using the Low-Rank Adaptation (LoRA) [^23] and full-parameter fine-tuning by fully unlocking parameters from the LLM. The results show that full parameter optimization effectively performs cross-modal semantic matching and retrieval between items, enhancing retrieval and classification performance in the q2i tasks.

On the one hand, when enhancing knowledge coverage and business data diversity during model training, SAIL-Embedding achieves significant performance gains on i2i tasks while maintaining results across a wide range of q2i tasks. On the other hand, we discover that as data volume scales up, model performance increases accordingly, adhering to the effect of scaling laws. After incorporating the COSENT loss, the q2i tasks achieve a significant 1.69% gain, implying that fine-grained ranking capabilities have been enhanced in text-oriented situations. Furthermore, we modify the all-in-one training procedure used by the baseline to the progressive training strategy. Based on experimental results, we conclude that progressive training enables the model to acquire substantial domain knowledge in the early phases while establishing robust task dependencies and activating corresponding representational knowledge in subsequent phases.

#### 4.4.6 Extensive Online Experiments

Table 6: Online Results in Recommender System. Feed, Message Pushing, and Coldstart are different channels or modules of Douyin. Douyin-Selected is another application with its own recommendation pipeline.

<table><tbody><tr><td>Scenarios</td><td>Stage</td><td>Feature</td><td>Gain</td></tr><tr><td rowspan="4">Feed</td><td>Recall</td><td>SID</td><td>LT30 + 0.01%</td></tr><tr><td>Pre-Rank</td><td>SID</td><td>LT30 + 0.01%</td></tr><tr><td>Rank</td><td>SID&Embedding</td><td>Finish AUC + 0.1%</td></tr><tr><td>Re-Rank</td><td>SID</td><td>LT30 + 0.01%</td></tr><tr><td rowspan="2">Message Pushing</td><td>Recall</td><td>SID & Embedding</td><td>LT30 + 0.03%</td></tr><tr><td>Rank</td><td>SID</td><td>LT30 + 0.01%</td></tr><tr><td>Coldstart</td><td>Recall</td><td>SID & Embedding</td><td>LT30 + 0.05%</td></tr><tr><td rowspan="2">Douyin-Selected</td><td>Recall</td><td>SID</td><td>LT7 + 0.4%</td></tr><tr><td>Rank</td><td>Embedding</td><td>LT7 + 0.1%</td></tr></tbody></table>

We deploy our model in the real-world recommender system to further verify its effectiveness. Our model mainly provides two types of features for downstream use.:

1. Embeddings. As mentioned before, our model compresses all information of an item, including video frames, title, OCR, ASR, along with the author’s nickname and other textual tags into a dense embedding. This embedding can be used for similarity-based recall, or be used in modules like SIM.
2. Semantic ID (SID). Furthermore, we discretize the embeddings into semantic IDs since most recommenders prefer discrete features. We engage both clustering and vector quantization methods to acquire codebooks with different sizes. The discrete tokens can be used for decentralization as well as the features of candidate items.

Experiments are conducted in various scenarios to verify the effectiveness of our method, and some typical results are shown in Table 6. Firstly, we find that the introduced features can benefit nearly all stages of the recommender system, including recall, pre-rank, rank and re-rank. We believe that engaging features provided by the same embedding model can lead to better consistency through these sequential stages. In addition, different scenarios such as coldstart and message pushing can both be enhanced, achieving 0.05% and 0.04% LT gain respectively. And our model can work on both Douyin and Douyin-Jingxuan, which are two different applications. All these results have demonstrated that our model is both effective and generalizable for recommendation.

Moreover, we find that SIDs bring more gain than dense embeddings. The reason may come from two aspects: (a) SIDs are much easier to use in ruled-based methods than embeddings, such as rule-based decentralization. (b) SIDs can be encoded into trainable embeddings, like item ID, so that they can be further adapted for recommendation models. However, dense embeddings tend to bring more information than SIDs. To effectively leverage such information is worth discovering for future work.

## 5 Conclusion

In this work, we present SAIL-Embedding, an omni-modal embedding foundation model tailored for large-scale recommendation scenarios. By unifying vision, text, and audio modalities, SAIL-Embedding overcomes the limitations of limited modalities of existing multimodal methods. Our contributions include a dynamic hard negative mining strategy and an adaptive multi-source data balancing framework, which jointly enhance training robustness and representation quality. Furthermore, we design a content-aware progressive training procedure and a collaboration-aware recommendation enhancement module, enabling the model to capture both semantic content and collaborative behavioral signals. Extensive experiments across diverse item-to-item and query-to-item benchmarks demonstrate that SAIL-Embedding achieves state-of-the-art performance and superior generalization to real-world industrial applications. Beyond its empirical effectiveness, our systematic ablations validate the necessity of each proposed component. We believe SAIL-Embedding provides a scalable and versatile foundation for future multimodal retrieval and recommendation systems, and we envision extending this framework to broader downstream tasks such as video understanding, personalized content generation, and cross-domain knowledge transfer.

In future work, we plan to further enhance the integration of vision-language models (VLMs) into recommendation systems. First, we will explore training VLMs aligned with recommendation objectives and constructing generative tasks tailored for recommendation, enabling the model to acquire domain-specific knowledge at earlier stages and strengthen its recommendation capability. Second, during representation learning, we aim to better align model training with recommendation goals by mining more paired data from recommendation signals and behavioral feedback, thereby injecting user preferences into multimodal representations. Finally, we will investigate hard negative mining in recommendation scenarios to improve the robustness of representation learning.

## 6 Contributor List

All contributors are listed in alphabetical order by last name initial, with equal contributions within each group; unless otherwise noted, all are members of the Douyin SAIL Team.

Core Contributors:  
Lin Lin  Jiefeng Long  Zhihe Wan  Yuchi Wang (CUHK MMLab)  Dingkang Yang  Shuang Yang  Yueyang Yao

Contributors:  
Xu Chen  Zirui Guo  Shengqiang Li Weiran Li  Hanyu Li  Yaling Mou  Yan Qiu  Haiyang Yu

Project Leader:  
Xiao Liang

Supervisors:  
Xiao Liang  Hongsheng Li (CUHK MMLab)  Chao Feng

## 7 Acknowledgments

We sincerely thank Song Chen, Yaodi Du, Xinjie Huang, Shengyu Li, Zhenming Sun, Junyan Yao, Enwei Zhang, Tengfeng Zhang, Zeqin Zhang, and Sheng Zheng for their valuable support and contributions.

[^1]: Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Hassan Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Singh Behl, Alon Benhaim, and et al. Phi-3 technical report: A highly capable language model locally on your phone. *ArXiv*, abs/2404.14219, 2024. URL [https://api.semanticscholar.org/CorpusID:269293048](https://api.semanticscholar.org/CorpusID:269293048).

[^2]: Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. Uniedit: A unified tuning-free framework for video motion and appearance editing. *ArXiv*, abs/2402.13185, 2024. URL [https://api.semanticscholar.org/CorpusID:267760278](https://api.semanticscholar.org/CorpusID:267760278).

[^3]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. URL [https://api.semanticscholar.org/CorpusID:261101015](https://api.semanticscholar.org/CorpusID:261101015).

[^4]: Haonan Chen, Hong Liu, Yuping Luo, Liang Wang, Nan Yang, Furu Wei, and Zhicheng Dou. Moca: Modality-aware continual pre-training makes better bidirectional multimodal embeddings. *arXiv preprint arXiv:2506.23115*, 2025a.

[^5]: Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. mme5: Improving multimodal multilingual embeddings via high-quality synthetic data. *ArXiv*, abs/2502.08468, 2025b. URL [https://api.semanticscholar.org/CorpusID:276287871](https://api.semanticscholar.org/CorpusID:276287871).

[^6]: Jiawei Chen, Dingkang Yang, Yue Jiang, Mingcheng Li, Jinjie Wei, Xiaolu Hou, and Lihua Zhang. Efficiency in focus: Layernorm as a catalyst for fine-tuning medical visual language pre-trained models. *arXiv preprint arXiv:2404.16385*, 2024.

[^7]: Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. *arXiv preprint arXiv:2311.07919*, 2023.

[^8]: Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in neural information processing systems*, 26, 2013.

[^9]: Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. *arXiv preprint arXiv:2502.18965*, 2025.

[^10]: Xiuqi Deng, Lu Xu, Xiyao Li, Jinkai Yu, Erpeng Xue, Zhongyuan Wang, Di Zhang, Zhaojie Liu, Guorui Zhou, Yang Song, et al. End-to-end training of multimodal model and ranking model. *arXiv preprint arXiv:2404.06078*, 2024.

[^11]: Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 11157–11168, 2020. URL [https://api.semanticscholar.org/CorpusID:219573658](https://api.semanticscholar.org/CorpusID:219573658).

[^12]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)*, pages 4171–4186, 2019.

[^13]: Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran. Scalable vision language model training via high quality data curation. *arXiv preprint arXiv:2501.05952*, 2025.

[^14]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[^15]: Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In *ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 1–5. IEEE, 2023.

[^16]: Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. *ArXiv*, abs/2504.17432, 2025a. URL [https://api.semanticscholar.org/CorpusID:278033532](https://api.semanticscholar.org/CorpusID:278033532).

[^17]: Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. *arXiv preprint arXiv:2504.17432*, 2025b.

[^18]: Michael Günther, Louis Milliken, Jonathan Geuter, Georgios Mastrapas, Bo Wang, and Han Xiao. Jina embeddings: A novel set of high-performance sentence embedding models. *arXiv preprint arXiv:2307.11224*, 2023a.

[^19]: Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, et al. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. *arXiv preprint arXiv:2310.19923*, 2023b.

[^20]: Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, and et al. Seed1.5-vl technical report. *ArXiv*, abs/2505.07062, 2025. URL [https://api.semanticscholar.org/CorpusID:278502305](https://api.semanticscholar.org/CorpusID:278502305).

[^21]: Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas R. Dengel. Audioclip: Extending clip to image, text and audio. *ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 976–980, 2021. URL [https://api.semanticscholar.org/CorpusID:235624127](https://api.semanticscholar.org/CorpusID:235624127).

[^22]: Minghao Han, Dingkang Yang, Jiabei Cheng, Xukun Zhang, Linhao Qu, Zizhi Chen, and Lihua Zhang. Towards unified molecule-enhanced pathology image representation learning via integrating spatial transcriptomics. *arXiv preprint arXiv:2412.00651*, 2024.

[^23]: Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021.

[^24]: Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. *arXiv preprint arXiv:2007.01282*, 2020.

[^25]: Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and João Carreira. Perceiver: General perception with iterative attention. *ArXiv*, abs/2103.03206, 2021. URL [https://api.semanticscholar.org/CorpusID:232110866](https://api.semanticscholar.org/CorpusID:232110866).

[^26]: Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In *International Conference on Machine Learning*, 2021. URL [https://api.semanticscholar.org/CorpusID:231879586](https://api.semanticscholar.org/CorpusID:231879586).

[^27]: Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. E5-v: Universal embeddings with multimodal large language models. *arXiv preprint arXiv:2407.12580*, 2024a.

[^28]: Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 7969–7992, 2023.

[^29]: Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. *arXiv preprint arXiv:2410.05160*, 2024b.

[^30]: Siddharth Joshi, Arnav Jain, Ali Payani, and Baharan Mirzasoleiman. Data-efficient contrastive language-image pretraining: Prioritizing data quality over quantity. In *International Conference on Artificial Intelligence and Statistics*, 2024. URL [https://api.semanticscholar.org/CorpusID:268532161](https://api.semanticscholar.org/CorpusID:268532161).

[^31]: Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In *EMNLP (1)*, pages 6769–6781, 2020.

[^32]: Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, and Ali Farhadi. Matryoshka representation learning. In *Neural Information Processing Systems*, 2022. URL [https://api.semanticscholar.org/CorpusID:252683450](https://api.semanticscholar.org/CorpusID:252683450).

[^33]: Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, et al. Gemini embedding: Generalizable embeddings from gemini. *arXiv preprint arXiv:2503.07891*, 2025.

[^34]: Weixian Lei, Jiacong Wang, Haochen Wang, Xiangtai Li, Jun Hao Liew, Jiashi Feng, and Zilong Huang. The scalability of simplicity: Empirical analysis of vision-language learning with a single transformer. *arXiv preprint arXiv:2504.10462*, 2025.

[^35]: Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *International Conference on Machine Learning*, 2022. URL [https://api.semanticscholar.org/CorpusID:246411402](https://api.semanticscholar.org/CorpusID:246411402).

[^36]: Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International Conference on Machine Learning*, 2023. URL [https://api.semanticscholar.org/CorpusID:256390509](https://api.semanticscholar.org/CorpusID:256390509).

[^37]: Mingcheng Li, Dingkang Yang, Yuxuan Lei, Shunli Wang, Shuaibing Wang, Liuzhen Su, Kun Yang, Yuzheng Wang, Mingyang Sun, and Lihua Zhang. A unified self-distillation framework for multimodal sentiment analysis with uncertain missing modalities. In *Proceedings of the AAAI conference on artificial intelligence*, volume 38, pages 10074–10082, 2024a.

[^38]: Mingcheng Li, Dingkang Yang, Yang Liu, Shunli Wang, Jiawei Chen, Shuaibing Wang, Jinjie Wei, Yue Jiang, Qingyao Xu, Xiaolu Hou, et al. Toward robust incomplete multimodal sentiment analysis via hierarchical representation learning. *Advances in Neural Information Processing Systems*, 37:28515–28536, 2024b.

[^39]: Shiyu Li, Yang Tang, Shizhe Chen, and Xi Chen. Conan-embedding: General text embedding with more and better negative samples. *arXiv preprint arXiv:2408.15710*, 2024c.

[^40]: Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms. *arXiv preprint arXiv:2411.02571*, 2024.

[^41]: Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval. *Neurocomputing*, 508:293–304, 2021. URL [https://api.semanticscholar.org/CorpusID:233296206](https://api.semanticscholar.org/CorpusID:233296206).

[^42]: Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. *arXiv preprint arXiv:2507.04590*, 2025.

[^43]: Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*, 2013.

[^44]: Bhaskar Mitra, Fernando Diaz, and Nick Craswell. Learning to match using local and distributed representations of text for web search. In *Proceedings of the 26th international conference on world wide web*, pages 1291–1299, 2017.

[^45]: Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pages 1532–1543, 2014.

[^46]: Jiajun Qin, Yuan Pu, Zhuolun He, Seunggeun Kim, David Z. Pan, and Bei Yu. Unimoco: Unified modality completion for robust multi-modal embeddings. *ArXiv*, abs/2505.11815, 2025. URL [https://api.semanticscholar.org/CorpusID:278739565](https://api.semanticscholar.org/CorpusID:278739565).

[^47]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PmLR, 2021.

[^48]: Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In *International conference on machine learning*, pages 28492–28518. PMLR, 2023.

[^49]: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pages 1–16. IEEE, 2020.

[^50]: Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. *arXiv preprint arXiv:2402.15449*, 2024.

[^51]: Jianlin Su. Cosent (1): A more effective sentence vector scheme than sentence bert, Jan 2022. URL [https://kexue.fm/archives/8847](https://kexue.fm/archives/8847).

[^52]: Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Gaopeng Gou, and Qi Wu. Missing target-relevant information prediction with world model for accurate zero-shot composed image retrieval. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 24785–24795, 2025.

[^53]: Qwen Team. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024.

[^54]: Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Neural Information Processing Systems*, 2017. URL [https://api.semanticscholar.org/CorpusID:13756489](https://api.semanticscholar.org/CorpusID:13756489).

[^55]: Jiacong Wang, Bohong Wu, Haiyong Jiang, Xun Zhou, Xin Xiao, Haoyuan Guo, and Jun Xiao. World to code: Multi-modal data generation via self-instructed compositional captioning and filtering. *arXiv preprint arXiv:2409.20424*, 2024a.

[^56]: Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. *arXiv preprint arXiv:2506.11991*, 2025a.

[^57]: Yuchi Wang, Junliang Guo, Jianhong Bai, Runyi Yu, Tianyu He, Xu Tan, Xu Sun, and Jiang Bian. Instructavatar: Text-guided emotion and motion control for avatar generation. *ArXiv*, abs/2405.15758, 2024b. URL [https://api.semanticscholar.org/CorpusID:270045800](https://api.semanticscholar.org/CorpusID:270045800).

[^58]: Yuchi Wang, Shuhuai Ren, Rundong Gao, Linli Yao, Qingyan Guo, Kaikai An, Jianhong Bai, and Xu Sun. Ladic: Are diffusion models really inferior to autoregressive counterparts for image-to-text generation? In *North American Chapter of the Association for Computational Linguistics*, 2024c. URL [https://api.semanticscholar.org/CorpusID:269156964](https://api.semanticscholar.org/CorpusID:269156964).

[^59]: Yuchi Wang, Yishuo Cai, Shuhuai Ren, Sihan Yang, Linli Yao, Yuanxin Liu, Yuanxing Zhang, Pengfei Wan, and Xu Sun. Rico: Improving accuracy and completeness in image recaptioning via visual reconstruction. *ArXiv*, abs/2505.22613, 2025b. URL [https://api.semanticscholar.org/CorpusID:278959593](https://api.semanticscholar.org/CorpusID:278959593).

[^60]: Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. In *European Conference on Computer Vision*, pages 387–404. Springer, 2024.

[^61]: Yixuan Wei, Yue Cao, Zheng Zhang, Houwen Peng, Zhuliang Yao, Zhenda Xie, Han Hu, and Baining Guo. iclip: Bridging image classification and contrastive language-image pre-training for visual recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2776–2786, 2023.

[^62]: Zhi Xu, Dingkang Yang, Mingcheng Li, Yuzheng Wang, Zhaoyu Chen, Jiawei Chen, Jinjie Wei, and Lihua Zhang. Debiased multimodal understanding for human language sequences. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pages 14450–14458, 2025.

[^63]: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.

[^64]: Dingkang Yang, Shuai Huang, Haopeng Kuang, Yangtao Du, and Lihua Zhang. Disentangled representation learning for multimodal emotion recognition. In *Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)*, pages 1642–1651, 2022a.

[^65]: Dingkang Yang, Shuai Huang, Shunli Wang, Yang Liu, Peng Zhai, Liuzhen Su, Mingcheng Li, and Lihua Zhang. Emotion recognition for multiple context awareness. In *Proceedings of the European Conference on Computer Vision (ECCV)*, volume 13697, pages 144–162, 2022b.

[^66]: Dingkang Yang, Haopeng Kuang, Kun Yang, Mingcheng Li, and Lihua Zhang. Towards asynchronous multimodal signal interaction and fusion via tailored transformers. *IEEE Signal Processing Letters*, 31:1550–1554, 2024a.

[^67]: Dingkang Yang, Mingcheng Li, Linhao Qu, Kun Yang, Peng Zhai, Song Wang, and Lihua Zhang. Asynchronous multimodal video sequence fusion via learning modality-exclusive and-agnostic representations. *IEEE Transactions on Circuits and Systems for Video Technology*, 2024b.

[^68]: Dingkang Yang, Mingcheng Li, Dongling Xiao, Yang Liu, Kun Yang, Zhaoyu Chen, Yuzheng Wang, Peng Zhai, Ke Li, and Lihua Zhang. Towards multimodal sentiment analysis debiasing via bias purification. In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2024c.

[^69]: Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 19163–19173, 2022c.

[^70]: Weijie Yin, Dingkang Yang, Hongyuan Dong, Zijian Kang, Jiacong Wang, Xiao Liang, Chao Feng, and Jiao Ran. Sailvit: Towards robust and generalizable visual backbones for mllms via gradual feature refinement. *arXiv preprint arXiv:2507.01643*, 2025.

[^71]: Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 11941–11952, 2023a. URL [https://api.semanticscholar.org/CorpusID:257767223](https://api.semanticscholar.org/CorpusID:257767223).

[^72]: Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 11975–11986, 2023b.

[^73]: Chao Zhang, Haoxin Zhang, Shiwei Wu, Di Wu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, and Enhong Chen. Notellm-2: Multimodal large representation models for recommendation. *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1*, 2024a. URL [https://api.semanticscholar.org/CorpusID:270063288](https://api.semanticscholar.org/CorpusID:270063288).

[^74]: Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal llms. *arXiv preprint arXiv:2412.16855*, 2024b.

[^75]: Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. *arXiv preprint arXiv:2506.05176*, 2025.

[^76]: Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. Vista: Visualized text embedding for universal multi-modal retrieval. *arXiv preprint arXiv:2406.04292*, 2024.