---
title: "OpenOneRec Technical Report An Open Foundation Model and Benchmark to Accelerate Generative Recommendation"
source: "https://ar5iv.labs.arxiv.org/html/2512.24762"
author:
published:
created: 2026-04-30
description: "While the OneRec series has successfully unified the fragmented recommendation pipeline into an end-to-end generative framework, a significant gap remains between recommendation systems and general intelligence. Constr…"
tags:
  - "clippings"
---
OneRec Team  
[https://huggingface.co/OpenOneRec](https://huggingface.co/OpenOneRec) [https://github.com/Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)

###### Abstract

While the OneRec series has successfully unified the fragmented recommendation pipeline into an end-to-end generative framework, a significant gap remains between recommendation systems and general intelligence. Constrained by isolated data, existing models struggle to leverage the massive data scaling that drives the emergent capabilities of Large Language Models (LLMs). As a result, they operate as domain specialists—proficient in pattern matching but lacking world knowledge, reasoning capabilities, and instruction following. This limitation is further compounded by the lack of a holistic benchmark to evaluate such integrated capabilities. To address this, our contributions are: 1) RecIF-Bench & Open Data: We propose RecIF-Bench, a holistic benchmark covering 8 diverse tasks that thoroughly evaluate capabilities from fundamental prediction to complex reasoning. Concurrently, we release a massive training dataset comprising 96 million interactions from 160,000 users to facilitate reproducible research. 2) Framework & Scaling: To ensure full reproducibility, we open-source our comprehensive training pipeline, encompassing data processing, co-pretraining, and post-training. Leveraging this framework, we demonstrate that recommendation capabilities can scale predictably while mitigating catastrophic forgetting of general knowledge. 3) OneRec-Foundation: We release OneRec-Foundation (1.7B and 8B), a family of models establishing new state-of-the-art (SOTA) results across all tasks in RecIF-Bench. Furthermore, when transferred to the Amazon benchmark, our models surpass the strongest baselines with an average 26.8% improvement in Recall@10 across 10 diverse datasets (Figure 1). This work marks a step towards building truly intelligent recommender systems. Nonetheless, realizing this vision presents significant technical and theoretical challenges, highlighting the need for broader research engagement in this promising direction.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x1.png)

Figure 1: Holistic Performance Overview. Left: Evaluation on RecIF-Bench and general LLM benchmarks. Our model achieves SOTA performance on recommendation tasks while effectively retaining general knowledge. "Best Baseline" denotes the highest performance achieved by existing methods for each specific task. Right: Amazon Benchmark results. Our model demonstrates exceptional cross-domain transferability, consistently surpassing leading baselines across 10 diverse datasets.

## 1 Introduction

Recent advances in Large Language Models (LLMs) have catalyzed a paradigm shift toward Generative Recommendation [^7] [^33]. Notable efforts, such as the OneRec series [^6] [^40] [^41], have successfully unified the multi-stage recommendation pipeline into an end-to-end generative framework, demonstrating the potential of treating user history as a context for next-item prediction. Despite these structural advancements, a significant chasm remains between current recommendation systems and general intelligence. Existing generative recommenders are often confined by isolated data silos, which sever the data scaling loops essential for the emergent capabilities seen in general LLMs [^12] [^10]. Consequently, these systems function as domain-specific experts that are highly effective at collaborative pattern matching but lack the general world knowledge essential for broader intelligence.

To bridge the semantic gap, recent studies such as LC-Rec [^39] and OneRec-Think [^14] align discrete recommendation identifiers with the linguistic space of LLMs. However, these approaches are typically confined to a limited set of downstream tasks. Such task homogeneity often induces catastrophic forgetting, thereby compromising the backbone’s inherent generalization capabilities. Furthermore, existing benchmarks remain restricted to narrow, specialized tasks, failing to evaluate the holistic capabilities essential for recommendation foundation models. In this work, we propose a unified framework (Figure 2) that integrates scalable pre-training, hybrid post-training, and holistic evaluation. We introduce RecIF-Bench, which, to our knowledge, is the first evaluation suite designed to assess multifaceted capabilities across diverse recommendation scenarios via instruction following. To facilitate reproducibility, we open-source our complete training pipeline, including data processing, co-pretraining, and post-training protocols, along with checkpoints trained on a hundred-billion-token corpus. We summarize our main contributions as follows:

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/Figures/framework.png)

Figure 2: The Overall Framework of OneRec-Foundation. (1) Pre-Training: Integrates collaborative signals with general semantics via Itemic-Text Alignment and mixed-domain Co-Pretraining. (2) Post-Training: Unlocks diverse downstream capabilities via SFT, and balances general reasoning with recommendation performance through alternating General Distillation and Rec-RL. (3) Evaluation: Comprehensively assesses holistic capabilities on RecIF-Bench and validates cross-domain transferability on Amazon datasets.

## 2 Preliminary

In this section, we reframe the recommendation problem through the lens of generative language modeling. We introduce a modality-aligned tokenization strategy for items and formulate the recommendation task as a standard autoregressive generation problem.

### 2.1 Items as Tokens: Bridging the Modality Gap

A fundamental challenge in applying LLMs to recommendation is the mismatch between the continuous linguistic space and the discrete item space. Representing items with detailed textual descriptions is inefficient due to excessively long contexts for user histories [^37] and fails to guarantee the generation of in-corpus items [^39]. To address this, we treat items as a distinct modality, adopting Itemic Tokens [^40] [^15] (see Tokenizer in Figure 2).

We employ RQ-Kmeans [^15] to discretize pre-trained semantic embeddings of item metadata into hierarchical discrete codes. This compresses item semantics into short, fixed-length sequences, enabling efficient long-context modeling while preserving collaborative structure. Crucially, the hierarchical nature of these tokens ensures that items with similar semantics share common prefixes, allowing the model to transfer knowledge based on token proximity—analogous to how semantic relationships are encoded in natural language tokens.

### 2.2 Recommendation as Autoregressive Modeling

With the item modality aligned to the token space, we unify the vocabulary $\mathcal{V}=\mathcal{V}_{text}\cup\mathcal{V}_{item}$. This allows us to treat a user’s interaction history not as a specialized data structure, but simply as a long context sequence containing item tokens, optionally interleaved with text.

We formulate tasks ranging from prediction (e.g., retrieval) to reasoning (e.g., explanation) as a unified Next-Token Prediction problem. Formally, given an instruction $\mathcal{I}$ and a user context $\mathcal{C}$ (comprising the interaction history and optional queries), the model maximizes the likelihood of the target response $Y$:

$$
\mathcal{L}(\theta)=-\sum_{t=1}^{|Y|}\log P_{\theta}(y_{t}|\mathcal{I},\mathcal{C},y_{<t}),
$$

where $Y$ can be a target item’s token sequence (for recommendation tasks) or a natural language sequence (e.g., item metadata, user profile, or reasoning). This formulation allows us to leverage the standard transformer architecture and training infrastructure of LLMs without any task-specific architectural modifications.

## 3 RecIF-Bench: A Holistic Recommendation Instruction-Following Benchmark

The evolution of recommender systems from specialized experts to general-purpose foundation models necessitates a paradigm shift in evaluation. Traditional benchmarks, typically confined to closed-set ranking accuracy within single domains, fall short of assessing the broader capabilities of Large Language Models (LLMs). To bridge this gap, we introduce RecIF-Bench, a comprehensive benchmark designed to rigorously evaluate recommendation foundation models.

### 3.1 Dataset Construction

A prerequisite for a foundation recommendation model is a data substrate that encompasses diverse interaction patterns and rich semantics. RecIF-Bench aggregates approximately 120M interactions from 200K distinct users, spanning three heterogeneous industrial domains. The detailed statistics are presented in Table 1, and the data distributions are visualized in Figure 3.

Table 1: Statistics of RecIF-Bench. The dataset covers three distinct industrial domains with diverse interaction densities.

| Domain | \# Users | \# Items | \# Interactions | Avg. Hist. Item | Avg. Tgt. Item |
| --- | --- | --- | --- | --- | --- |
| Short Video | 195,026 | 13,107,675 | 94,443,611 | 458.1 | 8.6 |
| Ad | 151,259 | 177,548 | 5,341,911 | 29.9 | 5.5 |
| Product | 144,307 | 2,055,240 | 20,087,210 | 132.5 | 6.7 |
| Total | 202,359 | 15,340,463 | 119,872,732 | 574.9 | 17.5 |

##### Multi-Domain Coverage.

The benchmark spans three distinct domains, each capturing different user behavior patterns:

- Short Video (Content Domain): Short-form videos from Kuaishou, covering viewing behaviors across various APP tabs. We provide impression sequences along with the corresponding interaction type for each impression.
- Ad (Commercial Domain): Promotional short videos sponsored by advertisers on the Kuaishou platform, typically containing clickable redirects. We provide click sequences recording user ad click behaviors.
- Product (E-commerce Domain): Products listed in the Kuaishou Mall. We provide click sequences recording user product click behaviors.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x2.png)

Figure 3: Data distribution analysis of RecIF-Bench. (a) Item popularity distribution (log-log scale) across domains. (b-d) Distribution of user history lengths for Short Video, Ad, and Product domains, respectively.

##### Rich Metadata.

Beyond interaction logs, RecIF-Bench provides comprehensive metadata across three dimensions:

- User-side: Each user is represented by a User Portrait—a unified narrative that interleaves natural language descriptions with itemic tokens. This portrait weaves together demographics (gender, age), content creation history, recent searches, followed creator types, viewing preferences, comments, livestream views, purchase records, shopping cart items, local service coupons, ad exposures, and commercial intent signals.
- Item-side: Each item is associated with multimodal embeddings (4096-dim text embedding and 5-frame visual embeddings with 1152-dim per frame). Additionally, we provide dense captions for approximately 13M videos.
- Interaction-side: For each user-video pair in the exposure sequence, we record multi-label behavioral signals including like, follow, comment, effective view, and dislike.

##### Itemic Tokenization.

To align these datasets with the generative paradigm defined in Section 2.1, we apply the hierarchical quantization strategy to all items in the dataset. Each item in RecIF-Bench is thus pre-tokenized into a tuple of discrete tokens $s=(c_{1},c_{2},...,c_{k})$, enabling direct consumption by LLM-based recommenders without further preprocessing. While we provide these pre-computed itemic tokens for convenience, their usage is not mandatory. Researchers can alternatively: (1) leverage the provided multimodal embeddings to train custom itemic tokenizers tailored to their specific needs, or (2) adopt traditional item IDs for conventional recommendation approaches. This flexibility ensures that RecIF-Bench accommodates diverse methodological paradigms while maintaining a consistent evaluation framework.

##### Data Splitting Strategy.

We adopt a strict user-based splitting strategy to evaluate generalization capabilities. From the pool of 200,000 users, we randomly select 20% as the held-out test set. These users and their interactions are entirely excluded from the training phase, ensuring zero leakage of test user data into the training corpus. For each user, we further partition their interaction sequence temporally: interactions before a designated timestamp constitute the history $\mathcal{H}$, while those after serve as the target $Y$ for evaluation.

### 3.2 Task Taxonomy: From Alignment to Reasoning

RecIF-Bench organizes 8 distinct tasks into a four-layer capability hierarchy. This taxonomy allows us to pinpoint exactly where a model’s capability lies on the spectrum from a traditional recommender to an intelligent agent. Table 2 provides a formal summary of these tasks.

##### Task Formulation.

Following the unified formulation in Section 2.2, we instantiate each task as a sequence-to-sequence problem $Y=\mathcal{F}(X)$. An input sample is formalized as $X=[\mathcal{I},\mathcal{C}]$, where $\mathcal{I}$ is the task-specific instruction and $\mathcal{C}$ denotes the personalized user context. Depending on the task, $\mathcal{C}$ can take different forms: (1) the interaction history $\mathcal{H}_{u}=\{s_{i_{1}},s_{i_{2}},...,s_{i_{t}}\}$ representing the user’s sequential behaviors, or (2) the user portrait $\mathcal{P}_{u}$, a unified narrative interleaving natural language descriptions with itemic tokens. The target $Y$ can be a next item ID, a sequence of IDs, or a natural language explanation.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x3.png)

Figure 4: Task Taxonomy of RecIF-Bench. We organize 8 tasks across 4 capability layers, specifying the instruction, context, and target.

Table 2: Task Taxonomy of RecIF-Bench. We formalize 8 tasks across 4 capability layers, specifying the input/output format and evaluation focus.

<table><tbody><tr><td>Layer</td><td>Task</td><td>Input (<math><semantics><mi>X</mi> <annotation>X</annotation></semantics></math>)</td><td>Target (<math><semantics><mi>Y</mi> <annotation>Y</annotation></semantics></math>)</td><td>Metric</td></tr><tr><td>L0: Alignment</td><td>Item Understanding</td><td>Item <math><semantics><mi>i</mi> <annotation>i</annotation></semantics></math></td><td>Item Description</td><td>LLM-as-Judge</td></tr><tr><td rowspan="4">L1: Fundamental Recommendation</td><td>Short Video Rec</td><td>History <math><semantics><msup><mi>ℋ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msup> <annotation>\mathcal{H}^{video}</annotation></semantics></math></td><td>Next item <math><semantics><msub><mi>i</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msub> <annotation>i_{video}</annotation></semantics></math></td><td>Pass@1/32, Recall@32</td></tr><tr><td>Ad Rec</td><td><math><semantics><mrow><msup><mi>ℋ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msup> <mo>+</mo> <msup><mi>ℋ</mi> <mrow><mi>a</mi> <mo></mo><mi>d</mi></mrow></msup></mrow> <annotation>\mathcal{H}^{video}+\mathcal{H}^{ad}</annotation></semantics></math></td><td>Next item <math><semantics><msub><mi>i</mi> <mrow><mi>a</mi> <mo></mo><mi>d</mi></mrow></msub> <annotation>i_{ad}</annotation></semantics></math></td><td>Pass@1/32, Recall@32</td></tr><tr><td>Product Rec</td><td><math><semantics><mrow><msup><mi>ℋ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msup> <mo>+</mo> <msup><mi>ℋ</mi> <mrow><mi>p</mi> <mo></mo><mi>r</mi> <mo></mo><mi>o</mi> <mo></mo><mi>d</mi> <mo></mo><mi>u</mi> <mo></mo><mi>c</mi> <mo></mo><mi>t</mi></mrow></msup></mrow> <annotation>\mathcal{H}^{video}+\mathcal{H}^{product}</annotation></semantics></math></td><td>Next item <math><semantics><msub><mi>i</mi> <mrow><mi>p</mi> <mo></mo><mi>r</mi> <mo></mo><mi>o</mi> <mo></mo><mi>d</mi> <mo></mo><mi>u</mi> <mo></mo><mi>c</mi> <mo></mo><mi>t</mi></mrow></msub> <annotation>i_{product}</annotation></semantics></math></td><td>Pass@1/32, Recall@32</td></tr><tr><td>Label Pred.</td><td><math><semantics><msup><mi>ℋ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msup> <annotation>\mathcal{H}^{video}</annotation></semantics></math> + Item <math><semantics><msub><mi>i</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msub> <annotation>i_{video}</annotation></semantics></math></td><td>Binary (Yes/No)</td><td>AUC</td></tr><tr><td rowspan="2">L2: Instruction Following</td><td>Interactive Rec</td><td>Portrait <math><semantics><mi>𝒫</mi> <annotation>\mathcal{P}</annotation></semantics></math> + Query <math><semantics><mi>q</mi> <annotation>q</annotation></semantics></math></td><td>Item <math><semantics><mi>i</mi> <annotation>i</annotation></semantics></math> user engages with</td><td>Pass@1/32, Recall@32</td></tr><tr><td>Label-Cond. Rec</td><td><math><semantics><msup><mi>ℋ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msup> <annotation>\mathcal{H}^{video}</annotation></semantics></math> + Action <math><semantics><mi>a</mi> <annotation>a</annotation></semantics></math></td><td>Item <math><semantics><mi>i</mi> <annotation>i</annotation></semantics></math> with action <math><semantics><mi>a</mi> <annotation>a</annotation></semantics></math></td><td>Pass@1/32, Recall@32</td></tr><tr><td>L3: Reasoning</td><td>Rec. Explanation</td><td><math><semantics><mi>𝒫</mi> <annotation>\mathcal{P}</annotation></semantics></math> + <math><semantics><msup><mi>ℋ</mi> <mrow><mi>v</mi> <mo></mo><mi>i</mi> <mo></mo><mi>d</mi> <mo></mo><mi>e</mi> <mo></mo><mi>o</mi></mrow></msup> <annotation>\mathcal{H}^{video}</annotation></semantics></math> + Item <math><semantics><mi>i</mi> <annotation>i</annotation></semantics></math></td><td>Explanation</td><td>LLM-as-Judge</td></tr></tbody></table>

#### 3.2.1 Layer 0: Semantic Alignment

This foundational layer verifies whether the model has successfully bridged the modality gap between itemic tokens and natural language—a prerequisite for all downstream tasks.

- Item Understanding. Given an item $i$, the model generates its textual metadata (e.g., title, caption). This inverse mapping tests whether the model aligns itemic tokens with language tokens.

#### 3.2.2 Layer 1: Fundamental Recommendation

This layer evaluates the model’s core capability to capture user preferences and predict item utility across multiple domains.

- Short Video Recommendation. Given the user’s video viewing history $\mathcal{H}^{video}$, the model predicts the next video the user will interact with. This single-domain task serves as the canonical next-item prediction benchmark.
- Ad / Product Recommendation (Cross-Domain). Beyond single-domain modeling, we explore the potential of leveraging cross-domain data for more comprehensive user understanding. By integrating the user’s rich content viewing history ($\mathcal{H}^{video}$) with commercial behaviors ($\mathcal{H}^{Ad}$ or $\mathcal{H}^{Product}$), the model can capture a more holistic view of user interests. Given $\mathcal{H}^{video}$ and $\mathcal{H}^{Ad}$ (or $\mathcal{H}^{Product}$), the model predicts the next ad (or product). This tests the model’s capability in cross-domain interest transfer and holistic user modeling.
- Label Prediction. Given the user’s history $\mathcal{H}^{video}$ and a candidate item $i$, the model predicts whether the user will engage (e.g., effective view) with a binary Yes/No response. This point-wise estimation complements generative recommendation.

#### 3.2.3 Layer 2: Instruction Following

This layer assesses whether the model can adapt its predictions based on explicit natural language instructions, a key differentiator of LLM-based recommenders.

- Interactive Recommendation. Given the user portrait $\mathcal{P}$ and a natural language query $q$ expressing immediate intent (e.g., "something relaxing"), the model predicts items that the user will positively engage with (e.g., click, like, or favorite) after issuing the query.
- Label-Conditional Recommendation. Given the user’s history $\mathcal{H}^{video}$ and a target behavior label $a$ (e.g., Like, ), the model predicts items the user will engage with under that specific behavior, testing fine-grained behavior modeling.

#### 3.2.4 Layer 3: Reasoning

The pinnacle of the hierarchy tests the model’s ability to synthesize information and articulate its understanding in natural language.

- Recommendation Explanation. Given the user portrait $\mathcal{P}$, interaction history $\mathcal{H}^{video}$, and a recommended item $s$, the model generates a natural language justification explaining why this item matches the user’s profile. This tests the model’s ability to reason about user-item compatibility.

Ground Truth for L3: Since reasoning tasks lack natural ground truth, we use Gemini-2.5-Pro with full metadata access to generate high-quality reference outputs.

### 3.3 Evaluation Protocols

We employ a dual-metric evaluation system to cover both recommendation accuracy and generation quality.

##### Recommendation Metrics.

For recommendation tasks (Layer 1 & 2), we use Pass@1, Pass@32, and Recall@32. Pass@K measures whether the ground truth item appears in the top-K generated candidates; Recall@K measures the proportion of relevant items retrieved.

##### Text Generation Metrics.

For text generation tasks (Layer 0 & 3), we employ LLM-as-Judge, prompting an independent LLM to rate the generated text on dimensions such as accuracy and coherence. Details are provided in Appendix B.1.

### 3.4 Comparison with Existing Benchmarks

Evaluating foundation models for recommendation requires assessing capabilities beyond traditional collaborative filtering. We identify three essential dimensions: (1) Data Complexity: handling multi-modal content, cross-domain transfer, long interaction sequences, and diverse user behaviors; (2) Task Generalization: executing multiple recommendation tasks within a unified framework; (3) Instruction Following & Reasoning: processing interleaved inputs (text and itemic tokens) and generating explainable recommendations.

Existing benchmarks address these dimensions only partially. As shown in Table 3, PixelRec and NineRec focus on multi-modal features but lack multi-task and reasoning capabilities. KuaiSAR supports multi-task and multi-behavior settings but omits multi-modal and cross-domain scenarios. Critically, no existing benchmark evaluates Interleaved Data processing or Recommendation Explanation—two capabilities essential for instruction-tuned foundation models. RecIF-Bench is the first benchmark to cover all seven dimensions, providing a comprehensive testbed for next-generation recommendation foundation models.

Table 3: Feature Comparison of Recommendation Benchmarks. RecIF-Bench is the only benchmark that simultaneously supports all seven dimensions. Specifically, it uniquely integrates Multi-Behavioral interactions, Interleaved Data(combining text and itemic tokens), and Recommendation Explanation to evaluate foundation models’ instruction-following ability.

| Benchmark | Long-Seq. | Multi-Task | Multi-Modal | Cross-Domain | Multi-Behav. | Interleaved Data | Reco. Exp. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PixelRec | ✓ |  | ✓ |  |  |  |  |
| KuaiSAR | ✓ | ✓ |  |  | ✓ |  |  |
| NineRec |  |  | ✓ | ✓ |  |  |  |
| Yelp |  |  | ✓ |  |  |  |  |
| Amazon |  |  | ✓ | ✓ |  |  |  |
| RecIF-Bench | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 3.5 Sanity Check: General Intelligence

To assess whether recommendation-focused models retain general intelligence capabilities, we include a sanity check suite covering four categories: (1) Math & Text Reasoning: MATH-500, GSM8K, and AIME’24; (2) General Tasks: MMLU-Pro and GPQA-Diamond; (3) Alignment: IFEVAL (strict prompt); (4) Coding: LiveCodeBench v5. A successful foundation model should maintain competitive performance on these benchmarks while mastering recommendation tasks.

## 4 Pre-Training

In this section, we detail the construction of our pre-training data, followed by the specific training strategies and an empirical analysis of scaling laws in recommendation.

### 4.1 Pre-training Data

#### 4.1.1 Recommendation Corpora

We derive recommendation data from anonymized user logs within Kuaishou, including user-side, item-side and interaction-side raw metadata as detailed in 3.1. To bridge the modality gap between discrete IDs and natural language, we first employ RQ-Kmeans [^15] to quantize item multimodal embeddings into hierarchical itemic tokens. Specifically, we employ a three-layer quantization scheme with a codebook size of 8192 per layer. Each item $i$ is thus mapped to a tuple of hierarchical codes $S_{i}=(c_{1},c_{2},c_{3})$, which is then flattened into a token sequence wrapped by special tokens:

<|item\_begin|><item\_a\_5028><item\_b\_6733><item\_c\_2559><|item\_end|>

To enhance the model’s recommendation-domain capability mentioned in 3.2, we organize these raw metadata into three types of data:

- Itemic Dense Caption Data. To build an initial perception of itemic tokens, we introduce the Itemic Dense Caption data. Given an item represented as itemic tokens, the model is trained to generate a corresponding natural-language caption. This task facilitates a semantic bridge between the abstract, discrete item representations and their rich textual descriptions. Each training sample is formatted as a structured sequence, pairing the item tokens with a target caption.
- Sequential User Behavior Data. To enhance the model’s Fundamental Prediction capability, we utilize Sequential User Behavior data as the core training corpus. This data captures the chronological dynamics of user-item interactions, including views, likes, and shares. By training the model to perform next-item prediction within these long-term sequences, we enable it to internalize fundamental collaborative filtering signals and temporal patterns. This ensures the model excels at the primary recommendation task: accurately predicting a user’s future interest based on their historical behavioral trajectory.
- Interleaved User Persona Grounding Data. To facilitate deep semantic grounding of the quantized space, we construct the Interleaved User Persona Grounding data. We construct narrative-style user portraits $\mathcal{P}_{u}$ by interleaving discrete item representations with heterogeneous user metadata, such as static attributes (e.g., age, gender), active search behaviors (e.g., recent search queries), interactive sequences (represented as itemic token sequences from user interaction history), and summarized user interests (e.g., content creation history, followed creator types, consumption preferences). This interleaved format enables the model to learn rich semantic associations between user characteristics and behavioral patterns, facilitating a deeper understanding of user preferences and item relevance beyond surface-level sequential patterns.

The recommendation-domain datasets are derived from the raw metadata introduced in 3.1, which has been processed by a strict user-based split to avoid data leakage. The primary training corpus encompasses about 160k users, 13 million item captions, and their corresponding interactions. For the OneRec-Pro variant, we scale this to $\sim$ 20 million users and 98 million item captions. To provide a clearer intuition of our pre-training corpus, we provide representative data samples in Appendix B.3. These demos show the structured formats of our training instances and illustrate the seamless integration of hierarchical itemic tokens with textual metadata. All these data and processing scripts will be released in our GitHub repository <sup>1</sup> to facilitate reproducibility.

#### 4.1.2 General-Domain Corpora

While injecting specialized recommendation knowledge is crucial for domain specialization, the substantial distributional shift between recommendation-domain data and the base pretrained model’s original corpus often leads to catastrophic forgetting. To mitigate this degradation and ensure a stable transition of model parameters, we adopt a data-mixing strategy that co-trains the model on recommendation-domain samples alongside high-quality general-domain text corpus [^22] [^2] [^19] [^32] [^35] [^5] [^3] [^29] [^20].

This general-domain text corpus contains multiple languages (including Chinese, English, and others) and various domains, mainly focusing on Coding, STEM(Science, Technology, Engineering, and Mathematics) and Medical. Crucially, to keep and further enhance the model’s reasoning capability, we prioritize reasoning-intensive data, including mathematical derivations, logical puzzles, and code-centric corpora. All these data can be downloaded from the Hugging Face dataset repository <sup>2</sup>. For convenience, we also provide detailed data composition and download links in Appendix B.4.

Specifically, we employ the MinHash algorithm [^4] to conduct efficient fuzzy deduplication, filtering out any general-domain samples that exhibit high similarity to the evaluation benchmarks. This process ensures that the model’s performance reflects genuine generalization, thereby maintaining the reliability of our experimental results.

### 4.2 Training Recipe

In terms of data composition, we blend recommendation-domain metadata with general-domain text at predefined ratios. Specifically, we develop two model variants based on the scale of the training corpus: OneRec and OneRec-Pro. The standard variant is trained exclusively on our publicly released dataset, encompassing 33B tokens across 41.3 million samples, thereby establishing a reproducible baseline for the community. In contrast, the Pro variant leverages an extensive in-house corpus with broader user coverage, totaling 130B tokens and 179.1 million samples to achieve enhanced robustness. A comprehensive breakdown of the token composition and mixing strategies is provided in Appendix B.4.

For both variants, we initialize our model using the Qwen3 backbone [^36], inheriting its full parameters after post-training for instruction following and reinforcement learning. The architecture remains strictly consistent with the original Qwen3 to preserve its foundational linguistic and reasoning capabilities. Building on this architecture, our pre-training methodology is structured into two distinct stages, as illustrated in the pre-training phase of Figure 2.

##### Stage 1: Itemic-Text Alignment.

The first stage focuses on establishing a preliminary alignment between the itemic tokens and text tokens space. We first expand the vocabulary by appending these itemic special tokens to the original Qwen3 tokenizer. The embedding parameters for these itemic tokens are initialized from a multivariate normal distribution parameterized by the mean and covariance of the existing embeddings. During this stage, only the embedding parameters corresponding to itemic tokens are trainable, while all other model’s parameters are frozen. Note that in Qwen3, smaller models (e.g., 0.6B, 1.7B, 4B) employ tied embeddings where the embedding and output projection layers share parameters, while larger models (e.g., 8B and above) have independent output projection parameters. For larger models, the output projection parameters corresponding to itemic tokens are also trainable, ensuring proper alignment in the output space.

##### Stage 2: Full-Parameter Co-Pretraining.

In the second stage, we unfreeze all model parameters and conduct full-parameter pre-training to inject recommendation knowledge into the model. This stage aims to enable the model to capture complex patterns in user behavior, item semantics, and user-item interactions while preserving the general world knowledge inherited from the original Qwen3 model. To prevent catastrophic forgetting, we maintain a considerable proportion of general-domain knowledge data throughout this stage.

##### Training Recipe.

We use the AdamW optimizer with $\beta_{1}=0.9$, $\beta_{2}=0.95$, and weight decay of $0.1$. The learning rate follows a cosine decay schedule with a linear warmup phase, where the peak learning rate is set to $1\times 10^{-3}$ for Stage 1 and $1\times 10^{-4}$ for Stage 2, and the minimum learning rate is set to $1\times 10^{-4}$ and $2\times 10^{-5}$. The warmup duration spans the first $10\%$ of training steps. To accommodate the long sequential nature of user behavior data, we set the maximum context length to $32$ K tokens, enabling the model to process extended user interaction histories and complex recommendation scenarios. This extended context window is crucial for capturing long-term user preferences and understanding intricate patterns in sequential recommendation tasks.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x4.png)

Figure 5: Scaling laws on recommendation-domain data. Top-left: training loss vs. FLOPs for different model sizes, and the smooth convex envelope shown in grey. Top-right: log-log plot of training loss vs. model size under fixed token budgets. Bottom-left: compute-optimal model size N opt ∝ C 0.44 N\_{\\text{opt}}\\propto C^{0.44} as a function of compute budget. Bottom-right: compute-optimal token budget D 0.56 D\_{\\text{opt}}\\propto C^{0.56} as a function of

### 4.3 Scaling Laws in Recommendation

To determine the optimal allocation of compute budget $C$ between model parameters $N$ and training tokens $D$, we follow the rigorous methodology established by [^10]. Utilizing the Qwen3 architecture as our backbone, we evaluate scaling behaviors across a parameter spectrum of $N\in\{0.6,1.7,4,8,14\}\times 10^{9}$. For the scaling analysis, we vary the token budget $D$ in Stage 2 of pretraining, modulating the cosine learning rate schedule to match the training horizon for each run, as recommended to avoid sub-optimal convergence. The training compute budget is estimated using the standard approximation $C\approx 6ND$.

We define the compute-optimal frontier by constructing the convex hull of the final training loss across all model sizes. By interpolating the minimum loss $L_{\text{min}}(C)$ on a continuous FLOPs grid, we extract the optimal model size $N_{\text{opt}}$ and token count $D_{\text{opt}}$ for a given budget. These optimal trajectories are fitted to power-law scaling relations:

$$
N_{\text{opt}}\propto C^{a},\quad D_{\text{opt}}\propto C^{b}
$$

##### Scaling Laws on Recommendation Data.

Figure 5 illustrates the recommendation-domain loss envelope. We observe a smooth, convex frontier, confirming that scaling behavior in recommendation follows predictable power laws analogous to natural language. However, our empirical fit yields the scaling exponents:

$$
N_{\text{opt}}\propto C^{0.44},\quad D_{\text{opt}}\propto C^{0.56}
$$

This allocation deviates significantly from the Chinchilla scaling laws for general text, which imply an equiproportional split ($a\approx 0.5,b\approx 0.5$). In contrast, our results indicate a data-intensive scaling regime ($b>a$), suggesting that in the recommendation domain, optimal compute allocation requires scaling training data volume more aggressively than model parameters as the budget increases.

##### Parametric Fit and Interpretation.

To elucidate the drivers of this behavior, we model the final loss $L(N,D)$ using the parametric function proposed by [^10]:

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

where $E$ represents the irreducible entropy of the data distribution, and the terms $\frac{A}{N^{\alpha}}$ and $\frac{B}{D^{\beta}}$ capture the deviations due to finite model capacity and finite data size, respectively. Fitting this formulation to our experimental data yields:

$$
L(N,D)=0.4232+\frac{502.32}{N^{0.3325}}+\frac{7.02}{D^{0.1865}}
$$

Based on the theoretical relationship derived in [^10], the optimal scaling exponents are governed by the ratio of these decay rates: $a=\frac{\beta}{\alpha+\beta}$ and $b=\frac{\alpha}{\alpha+\beta}$. We derive three critical insights from these coefficients:

- Data-Hungry Scaling ($\alpha>\beta$): Our derived model capacity exponent ($\alpha\approx 0.33$) is consistent with general LLM literature, but the data exponent ($\beta\approx 0.19$) is notably lower than typical text-domain values ($\beta_{\text{text}}\approx 0.28$). Since $\alpha>\beta$, it mathematically necessitates that $b>0.5$. This confirms that because the returns on data quantity diminish more rapidly (lower $\beta$), maintaining optimality requires scaling data volume more aggressively than model size.
- Impact of Warm-Starting (High $A$, Low $B$): We observe a striking imbalance between the scaling coefficients $A$ ($502.32$) and $B$ ($7.02$). In typical training-from-scratch scenarios, $A$ and $B$ are of comparable magnitude. We attribute the extremely low $B$ value to the efficacy of transfer learning from the Qwen3 backbone, where robust pre-trained linguistic and reasoning capabilities lower the entropy of the initial data distribution. Conversely, the inflated coefficient $A$ reflects a conflation of model capacity and pre-training quality: as larger variants are typically trained with more data, their downstream performance gains are statistically captured into $A$ during the fitting process. Notwithstanding this imbalance, our formula aligns well with our empirical observations, reflecting the scaling trajectory between $L$ and the variables $(N,D)$ across all evaluated configurations.
- Low Entropy of Recommendation Tasks (Low $E$): The estimated irreducible loss floor $E=0.42$ is substantially lower than that of natural text ($E\approx 1.69$). This suggests that our recommendation tasks—enriched with structured features such as Itemic Dense Captions—possess lower inherent entropy than open-ended text generation, allowing the model to approach a more deterministic state. Consequently, this underscores the critical need for curating recommendation corpora with greater diversity and higher quality, thereby expanding the information manifold to prevent trivial saturation and foster robust generalization.

Formula 5 provides an initial exploration of the scaling law in recommendation. We acknowledge that the limited experimental scale may introduce some fitting noise. Furthermore, since our models are not trained from scratch, the formula does not fully separate the scaling contribution of the pre-trained backbone. Future work will focus on expanding the experimental scale and design refined formula that better integrate the influence of pre-trained knowledge.

## 5 Post-Training

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x8.png)

Figure 6: Post-training pipeline of the OneRec series models.

After the pre-training process, the model has already learned to align the itemic tokens with the text tokens space, and encoded the collaborative filtering signals into the pretrained models. However, we observed that the pre-trained model exhibits a certain degree of degradation in instruction-following and reasoning capabilities, which is still not capable of complex recommendation tasks.

The post-training is designed to enhance the recommendation capabilities and restore the general task capabilities of the pre-trained model. We adopt three post-training stages: Multi-task Supervised Fine-tuning, On-policy Distillation for General Capability, and Reinforcement Learning for Recommendation. Distinct from the pre-training phase, for this stage, we employ identical training data and equivalent post-training strategies for both the OneRec and OneRec-Pro variants. An overview of these stages is presented in Figure 6.

### 5.1 Multi-task Supervised Fine-tuning

This stage is designed to restore and enhance the model’s foundational instruction-following and reasoning capabilities across both general and recommendation domains, providing a robust base for subsequent on-policy distillation and reinforcement learning.

We begin by designing a suite of complex instruction-response pairs that simulate real-world recommendation trajectories for post-training. Consistent with the pre-training stage, these conversational datasets are synthesized based on the cleaned metadata from 160K users introduced in Section 3.1. This ensures that the instruction-tuning corpus remains strictly partitioned from the evaluation benchmarks. Besides, to prevent the degradation of general intelligence, we also incorporate high-quality open-source general-domain datasets focusing on instruction-following and complex reasoning [^19] [^16] [^13] [^32] [^35] [^26] [^8].

We curate a specialized SFT corpus by blending these general-domain reasoning samples with the aforementioned recommendation-specific tasks, and the detail is provided in Appendix B.5. All instances are organized into a conversational format and serialized using the Qwen3 chat template [^36]. We fine-tune the pre-trained model on this unified dataset using a training recipe consistent with the pre-training phase, but with a reduced learning rate (from $2\times 10^{-5}$ to $5\times 10^{-6}$). Empirical observations suggest that this stage successfully resuscitates the model’s instruction-following capabilities. Notably, we find that the reasoning ability acquired from general-domain data cross-fertilizes with recommendation tasks: the model frequently generates coherent reasoning trajectories for complex recommendation queries, even though such behaviors were not explicitly supervised in the recommendation samples.

### 5.2 On-policy Distillation for General Capability

While previous stages have successfully restored the basic capabilities of instruction-following and thinking, a persistent capability gap in general-domain reasoning is observed, likely due to the distributional shift and the inherent sensitivity of RL-initialized backbones. To address this, we design an on-policy distillation strategy on general tasks.

##### On-Policy Distillation via Policy Gradient.

Unlike traditional off-policy distillation, where the student model learns a teacher’s distribution on a static, pre-generated dataset, on-policy distillation [^1] involves the student model generating its own trajectories, which are subsequently evaluated and supervised by the teacher.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x9.png)

Figure 7: The pipeline of On-policy Distillation via Policy Gradient. The student model (Policy Model) samples a trajectory o for the given prompt q, while the Teacher Model provides feedback through Reverse KL divergence as reward r. The Policy Model is iteratively optimized using policy gradient methods based on reward.

For simplicity and effectiveness, we employ per-token reverse KL divergence as the objective function for on-policy distillation. Specifically, we minimize the divergence between the student’s distribution ($\pi_{\theta}$) and the teacher’s ($\pi_{\text{teacher}}$) at each timestep, conditioned on trajectories sampled by the student:

$$
\mathbb{D}_{KL}\left(\pi_{\theta}\parallel\pi_{\text{teacher}}\right)=\mathbb{E}_{x\sim\pi_{\theta}}\left[\log\pi_{\theta}(x_{t+1}|x_{1..t})-\log\pi_{\text{teacher}}(x_{t+1}|x_{1..t})\right]
$$

Inspired by [^34] [^30], we optimize the policy $\pi_{\theta}$ directly using policy gradient methods. For each input prompt $q$, we sample a trajectory $o$, and employ the reverse KL divergence between the student policy $\pi_{\theta}$ and the teacher policy $\pi_{\text{teacher}}$ as the reward signal. The objective is to maximize the expected reward via gradient ascent, with the gradient estimated as follows:

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{o\sim\mathcal{D},x\sim\pi_{\theta}}\left[\sum_{t=1}^{T}\nabla_{\theta}\log\pi_{\theta}(x_{t}|o,x_{<t})\cdot R_{KL}(o,x)\right]
$$

where $R_{KL}(o,x)$ denotes the per-token reward derived from the teacher’s distribution. To mitigate numerical instability caused by extreme log-probability ratios, we apply a clipping mechanism to the reverse KL divergence. Specifically, the final reward is defined as:

$$
R_{KL}(o,x)=\text{clip}\left(-\mathbb{D}_{KL}(\pi_{\theta}||\pi_{\text{teacher}}),\alpha,\beta\right)
$$

where $\alpha$ and $\beta$ represent the lower and upper clipping thresholds, respectively. This constraint prevents outlier reward signals from destabilizing the training process, thereby enabling the model to robustly recover the teacher’s behavioral distribution through online trajectory sampling. The comprehensive pipeline is illustrated in Figure 7.

##### On-Policy Distillation on General-Domain.

To recover general-domain capabilities, we employ the original Qwen3 model (of the same parameter scale) as a teacher model $\pi_{\text{teacher}}$ to supervise our SFT-refined policy $\pi_{\theta}$.

A critical challenge lies in the vocabulary discrepancy: the teacher model cannot recognize the itemic tokens in our expanded vocabulary. A naive approach is discarding any trajectory $o$ that contains itemic tokens. However, since the reward is estimated via per-token reverse KL, which is inherently a biased approximation, simply dropping these samples introduces significant sampling bias. As training progresses, the model may gradually increase the probabilities of itemic tokens, eventually leading to training collapse.

To address this, we adopt the following robust distillation strategy:

- Prompt Selection: All queries $q$ are sampled exclusively from general-domain datasets. Under these prompts, the policy model is expected to generate pure text without itemic tokens.
- Itemic Token Penalty & Truncation: If a sampled trajectory $o$ contains an itemic token at step $t$, we set $\log\pi_{\text{teacher}}(x_{t}|x_{<t})$ to a minimal value (e.g., $-1e9$) to simulate a zero-probability for that token, and truncate the trajectory after $t$. This penalty, stabilized by the reward clipping mechanism, provides an immediate and strong negative signal for itemic tokens in general-domain contexts.
- Enhanced Exploration: During on-policy sampling, we employ a relatively high temperature coefficient. This encourages the policy model to explore the vocabulary space (including the itemic tokens), allowing the distillation process to actively identify and correct errors of itemic token activations in general-domain tasks.

Through this distillation process, we sample 200K general-domain questions from the SFT dataset. To better recover the instruction-following capabilities on thinking of the original Qwen3, we follow the methodology described in the Qwen3 technical report [^36] and randomly append a suffix (/think, /no\_think, or an empty string) to the user prompt. This strategy is designed to align the model’s behavior with the forced-thinking, non-thinking, and auto-thinking paradigms, ensuring robust control over its reasoning processes across diverse scenarios.

After this stage, we effectively bridge the gap in general intelligence while maintaining the model’s specialized recommendation performance.

### 5.3 Reinforcement Learning for Recommendation

While on-policy distillation effectively restores the model’s general reasoning capabilities, it does not directly optimize the discrete ranking metrics (e.g., Recall or NDCG) that define recommendation quality. Supervised fine-tuning (SFT) primarily focuses on maximizing the likelihood of ground-truth sequences, often suffering from exposure bias and failing to distinguish between "near-misses" and irrelevant recommendations. To bridge this gap, we introduce a final stage of Recommendation-oriented Reinforcement Learning (Rec-RL).

##### Group Relative Policy Optimization

We employ Group Relative Policy Optimization (GRPO) [^24] as our reinforcement learning framework. Unlike traditional Actor-Critic algorithms (e.g., PPO) that require a separate critic model to estimate state values, GRPO computes the advantage of a response relative to a group of sampled trajectories for the same prompt. This significantly reduces computational overhead while maintaining stability.

Formally, for each recommendation prompt $q$, we sample a group of $G$ candidate responses $\{R_{1},R_{2},\dots,R_{G}\}$ from the current policy $\pi_{\theta}$. The objective is to maximize the following:

$$
\mathcal{L}_{GRPO}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\left(\text{Adv}_{i}\cdot\log\pi_{\theta}(R_{i}|q)\right)-\beta\cdot KL(\pi_{\theta}||\pi_{ref})
$$

where $\pi_{ref}$ is the model obtained after on-policy distillation, and $\text{Adv}_{i}$ is the relative advantage calculated by normalizing the rewards within the group.

##### Rule-based Recommendation Reward.

To align the model directly with the ranking accuracy, we design a sparse, rule-based reward function $r(R_{i})$ focused on "Hit" events. For the 5 core recommendation tasks—Short Video Rec, Ad Rec, Product Rec, Interactive Rec, and Label-Conditional Rec—the reward is defined as:

$$
r(R_{i})=\begin{cases}+1.0&\text{if the target itemic token }s\in R_{i}\\
0.0&\text{otherwise}\end{cases}
$$

By sampling multiple candidate sequences (groups) for each user interaction history, GRPO encourages the model to assign higher probability mass to itemic tokens that result in successful hits, effectively performing "Soft Ranking" within the generative space.

##### Implementation Details.

We initialize the RL trainer with the model post-distillation. To ensure the model does not sacrifice its restored general intelligence for domain-specific precision, we maintain a strict KL penalty ($\beta$) against $\pi_{ref}$. We utilize the same dataset as the SFT stage, yet observe continuous improvement in recommendation performance throughout RL training. After this stage, we observe a significant boost on recommendation metrics, demonstrating that Rec-RL effectively aligns the language model’s generative behavior with the goals of a recommender system.

## 6 Evaluation

In this section, we present our detailed experimental results. We first demonstrate the comprehensive capabilities of OneRec-Foundation on RecIF-Bench. Next, we validate the generalization ability of OneRec-Foundation on the Amazon dataset. Finally, we conduct multiple ablation studies to analyze the impact of different components on the model.

### 6.1 Experimental Settings

##### RecIF-Bench.

To demonstrate the effectiveness of OneRec-Foundation, we compare it against two groups of competitive baselines: (1) Discriminative recommender models, including BERT4Rec [^27], GRU4Rec [^9], SASRec [^11], HSTU [^38], and ReaRec [^28]; and (2) Generative recommender models, such as TIGER [^23] and LC-Rec [^39]. We adapted these baseline methods with task-specific modifications to support different RecIF tasks. Notably, these discriminative-based approaches are inherently task-specific, with each task requiring a separately trained model. To ensure a fair comparison at the foundation model scale, we implement LC-Rec as “LC-Rec-8B” using a comparable Qwen3-8B. We evaluate all methods using the metrics in Section 3.3, with comprehensive results reported in Table 4.

##### Amazon.

To evaluate the generalization capability of our model, we utilize ten real-world datasets from the popular Amazon review benchmark [^17], covering diverse domains: Baby, Beauty, Cell Phones and Accessories, Grocery and Gourmet Food, Health and Personal Care, Home and Kitchen, Pet Supplies, Sports and Outdoors, Tools and Home Improvement, and Toys and Games. We adopt the same baselines as mentioned above. Consistent with previous works [^23] [^31], we focus on the sequential recommendation setting. For data pre-processing, we discard sparse users and items with fewer than 5 interactions. We employ the leave-one-out strategy [^23] [^31] to split the datasets for training and evaluation. For performance comparison, we report Recall@ $K$ and NDCG@ $K$ with $K\in\{5,10\}$.

### 6.2 Main Results on RecIF-Bench

Table 4: Unified performance comparison across all tasks. For each task, the bold results highlight the best results (with an orange background), while the second-best ones are underlined. Missing entries are marked as “–”.

<table><tbody><tr><td>Task</td><td>Metric</td><td>SASRec</td><td>BERT4Rec</td><td>GRU4Rec</td><td>HSTU</td><td>ReaRec</td><td>TIGER</td><td>LC-Rec-8B</td><td>OneRec-1.7B</td><td>OneRec-8B</td><td>OneRec-1.7B-Pro</td><td><p></p><p>OneRec-8B-Pro</p><p></p></td></tr><tr><td rowspan="3">Short Video Rec</td><td>Pass@1</td><td>0.0045</td><td>0.0040</td><td>0.0051</td><td>0.0043</td><td>0.0052</td><td>0.0168</td><td>0.0341</td><td>0.0496</td><td>0.0542</td><td>0.0456</td><td>0.0548</td></tr><tr><td>Pass@32</td><td>0.1003</td><td>0.0951</td><td>0.0993</td><td>0.1010</td><td>0.1002</td><td>0.1061</td><td>0.1306</td><td>0.1710</td><td>0.2104</td><td>0.1706</td><td>0.2122</td></tr><tr><td>Recall@32</td><td>0.0119</td><td>0.0113</td><td>0.0117</td><td>0.0119</td><td>0.0120</td><td>0.0132</td><td>0.0180</td><td>0.0272</td><td>0.0355</td><td>0.0274</td><td>0.0369</td></tr><tr><td rowspan="3">Ad Rec</td><td>Pass@1</td><td>0.0044</td><td>0.0061</td><td>0.0059</td><td>0.0076</td><td>0.0035</td><td>0.0125</td><td>0.0197</td><td>0.0169</td><td>0.0219</td><td>0.0190</td><td>0.0259</td></tr><tr><td>Pass@32</td><td>0.0980</td><td>0.1225</td><td>0.1102</td><td>0.1266</td><td>0.1054</td><td>0.1769</td><td>0.2096</td><td>0.2037</td><td>0.2490</td><td>0.2126</td><td>0.2700</td></tr><tr><td>Recall@32</td><td>0.0293</td><td>0.0381</td><td>0.0336</td><td>0.0409</td><td>0.0327</td><td>0.0581</td><td>0.0723</td><td>0.0707</td><td>0.0877</td><td>0.0735</td><td>0.0964</td></tr><tr><td rowspan="3">Product Rec</td><td>Pass@1</td><td>0.0052</td><td>0.0054</td><td>0.0047</td><td>0.0055</td><td>0.0030</td><td>0.0120</td><td>0.0178</td><td>0.0144</td><td>0.0187</td><td>0.0158</td><td>0.0223</td></tr><tr><td>Pass@32</td><td>0.0914</td><td>0.0936</td><td>0.0821</td><td>0.0914</td><td>0.0907</td><td>0.1276</td><td>0.1809</td><td>0.1571</td><td>0.1971</td><td>0.1761</td><td>0.2290</td></tr><tr><td>Recall@32</td><td>0.0175</td><td>0.0193</td><td>0.0161</td><td>0.0178</td><td>0.0189</td><td>0.0283</td><td>0.0416</td><td>0.0360</td><td>0.0470</td><td>0.0405</td><td>0.0538</td></tr><tr><td rowspan="3">Label-Cond. Rec</td><td>Pass@1</td><td>0.0026</td><td>0.0026</td><td>0.0032</td><td>0.0026</td><td>0.0027</td><td>0.0044</td><td>0.0079</td><td>0.0064</td><td>0.0097</td><td>0.0067</td><td>0.0099</td></tr><tr><td>Pass@32</td><td>0.0380</td><td>0.0372</td><td>0.0393</td><td>0.0383</td><td>0.0381</td><td>0.0337</td><td>0.0420</td><td>0.0431</td><td>0.0535</td><td>0.0420</td><td>0.0549</td></tr><tr><td>Recall@32</td><td>0.0140</td><td>0.0135</td><td>0.0143</td><td>0.0139</td><td>0.0137</td><td>0.0123</td><td>0.0170</td><td>0.0184</td><td>0.0228</td><td>0.0182</td><td>0.0235</td></tr><tr><td>Label Pred.</td><td>AUC</td><td>0.6244</td><td>0.6598</td><td>0.6640</td><td>0.6581</td><td>0.6204</td><td>0.6675</td><td>0.6139</td><td>0.6184</td><td>0.6615</td><td>0.6071</td><td>0.6912</td></tr><tr><td rowspan="3">Interactive Rec</td><td>Pass@1</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>0.0890</td><td>0.0660</td><td>0.1230</td><td>0.0800</td><td>0.1250</td></tr><tr><td>Pass@32</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>0.3730</td><td>0.3170</td><td>0.4570</td><td>0.3370</td><td>0.5080</td></tr><tr><td>Recall@32</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>0.2394</td><td>0.1941</td><td>0.3032</td><td>0.2024</td><td>0.3458</td></tr><tr><td>Item Understand.</td><td>LLM-Judge Score</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>0.2517</td><td>0.3175</td><td>0.3202</td><td>0.3133</td><td>0.3209</td></tr><tr><td>Rec. Explanation</td><td>LLM-Judge Score</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>3.9350</td><td>3.3540</td><td>3.6774</td><td>3.5060</td><td>4.0381</td></tr></tbody></table>

Table 5: Performance comparison on general capability (Thinking). For each task, the bold results highlight the best results, while the second-best ones are underlined.

<table><tbody><tr><td>Category</td><td>Task</td><td>Qwen3-1.7B</td><td>OneRec-1.7B</td><td>OneRec-1.7B-Pro</td><td>Qwen3-8B</td><td>OneRec-8B</td><td>OneRec-8B-Pro</td></tr><tr><td rowspan="3">Math & Text Reasoning</td><td>MATH-500</td><td>0.8780</td><td>0.8840</td><td>0.8840</td><td>0.9520</td><td>0.9460</td><td>0.9380</td></tr><tr><td>GSM8K</td><td>0.9121</td><td>0.8984</td><td>0.8999</td><td>0.9568</td><td>0.9575</td><td>0.9575</td></tr><tr><td>AIME’24</td><td>0.4938</td><td>0.4104</td><td>0.4146</td><td>0.7917</td><td>0.7250</td><td>0.7188</td></tr><tr><td rowspan="2">General Tasks</td><td>MMLU-Pro</td><td>0.5422</td><td>0.3548</td><td>0.3932</td><td>0.7235</td><td>0.5342</td><td>0.5204</td></tr><tr><td>GPQA-Diamond</td><td>0.3788</td><td>0.3232</td><td>0.3333</td><td>0.5606</td><td>0.5000</td><td>0.5051</td></tr><tr><td>Alignment Tasks</td><td>IFEVAL <math><semantics><msub><mtext>strict prompt</mtext></msub> <annotation>{}_{\text{strict prompt}}</annotation></semantics></math></td><td>0.6969</td><td>0.5471</td><td>0.5416</td><td>0.8577</td><td>0.7893</td><td>0.7634</td></tr><tr><td>Coding</td><td>LiveCodeBench v5</td><td>0.3907</td><td>0.2832</td><td>0.2832</td><td>0.5484</td><td>0.4910</td><td>0.4667</td></tr></tbody></table>

Table 6: Performance comparison on general capability (Non-Thinking). For each task, the bold results highlight the best results, while the second-best ones are underlined.

<table><tbody><tr><td>Category</td><td>Task</td><td>Qwen3-1.7B</td><td>OneRec-1.7B</td><td>OneRec-1.7B-Pro</td><td>Qwen3-8B</td><td>OneRec-8B</td><td>OneRec-8B-Pro</td></tr><tr><td rowspan="3">Math & Text Reasoning</td><td>MATH-500</td><td>0.6980</td><td>0.7060</td><td>0.6940</td><td>0.8380</td><td>0.8240</td><td>0.7980</td></tr><tr><td>GSM8K</td><td>0.8218</td><td>0.8036</td><td>0.8158</td><td>0.9303</td><td>0.9310</td><td>0.9196</td></tr><tr><td>AIME’24</td><td>0.1313</td><td>0.1271</td><td>0.1250</td><td>0.2729</td><td>0.2417</td><td>0.2271</td></tr><tr><td rowspan="2">General Tasks</td><td>MMLU-Pro</td><td>0.4384</td><td>0.3072</td><td>0.2804</td><td>0.6632</td><td>0.5795</td><td>0.4521</td></tr><tr><td>GPQA-Diamond</td><td>0.3030</td><td>0.3131</td><td>0.2778</td><td>0.3990</td><td>0.4040</td><td>0.3939</td></tr><tr><td>Alignment Tasks</td><td>IFEVAL <math><semantics><msub><mtext>strict prompt</mtext></msub> <annotation>{}_{\text{strict prompt}}</annotation></semantics></math></td><td>0.6747</td><td>0.4769</td><td>0.5250</td><td>0.8392</td><td>0.7357</td><td>0.7098</td></tr><tr><td>Coding</td><td>LiveCodeBench v5</td><td>0.1219</td><td>0.1219</td><td>0.1147</td><td>0.2760</td><td>0.2401</td><td>0.2401</td></tr></tbody></table>

Based on the evaluation results, we highlight some key conclusions on RecIF-Bench:

- State-of-the-Art Recommendation Performance: OneRec-Foundation consistently outperforms all baselines across the vast majority of tasks. Notably, the results demonstrate scaling from two dimensions: (1) Data scaling: OneRec-Pro consistently surpasses OneRec at the same model size; (2) Model scaling: 8B models outperform 1.7B models across all variants.
- Trade-off on General Capabilities: As shown in Table 5 and Table 6, our model successfully retains most of the general capabilities of the Qwen3 backbone, especially with minimal degradation on mathematical benchmarks. Nevertheless, we observe a performance trade-off in general knowledge and recommendation capabilities. This suggests that while the distillation process effectively preserves reasoning proficiency, the limited diversity of the general data used may have constrained the model’s broader capabilities, indicating that a more refined data strategy is required to better balance recommendation and general capabilities.

Table 7: Cross-Domain generalization performance on Amazon domains. Best results are indicated with bold fonts and orange background.

<table><tbody><tr><td>Model</td><td>Metric</td><td>Baby</td><td>Beauty</td><td>Cell</td><td>Grocery</td><td>Health</td><td>Home</td><td>Pet</td><td>Sports</td><td>Tools</td><td>Toys</td></tr><tr><td rowspan="4">SASRec</td><td>R@5</td><td>0.0232</td><td>0.0393</td><td>0.0482</td><td>0.0480</td><td>0.0295</td><td>0.0133</td><td>0.0377</td><td>0.0240</td><td>0.0269</td><td>0.0420</td></tr><tr><td>R@10</td><td>0.0381</td><td>0.0639</td><td>0.0782</td><td>0.0789</td><td>0.0506</td><td>0.0212</td><td>0.0607</td><td>0.0389</td><td>0.0437</td><td>0.0658</td></tr><tr><td>N@5</td><td>0.0137</td><td>0.0209</td><td>0.0281</td><td>0.0262</td><td>0.0173</td><td>0.0070</td><td>0.0222</td><td>0.0130</td><td>0.0149</td><td>0.0217</td></tr><tr><td>N@10</td><td>0.0185</td><td>0.0289</td><td>0.0378</td><td>0.0361</td><td>0.0242</td><td>0.0098</td><td>0.0296</td><td>0.0178</td><td>0.0203</td><td>0.0294</td></tr><tr><td rowspan="4">BERT4Rec</td><td>R@5</td><td>0.0117</td><td>0.0219</td><td>0.0325</td><td>0.0307</td><td>0.0204</td><td>0.0063</td><td>0.0218</td><td>0.0151</td><td>0.0145</td><td>0.0200</td></tr><tr><td>R@10</td><td>0.0228</td><td>0.0419</td><td>0.0569</td><td>0.0534</td><td>0.0353</td><td>0.0113</td><td>0.0412</td><td>0.0261</td><td>0.0264</td><td>0.0362</td></tr><tr><td>N@5</td><td>0.0065</td><td>0.0120</td><td>0.0190</td><td>0.0174</td><td>0.0117</td><td>0.0038</td><td>0.0123</td><td>0.0083</td><td>0.0083</td><td>0.0102</td></tr><tr><td>N@10</td><td>0.0101</td><td>0.0185</td><td>0.0268</td><td>0.0247</td><td>0.0165</td><td>0.0054</td><td>0.0186</td><td>0.0119</td><td>0.0121</td><td>0.0154</td></tr><tr><td rowspan="4">GRU4Rec</td><td>R@5</td><td>0.0202</td><td>0.0322</td><td>0.0430</td><td>0.0362</td><td>0.0256</td><td>0.0090</td><td>0.0264</td><td>0.0174</td><td>0.0176</td><td>0.0266</td></tr><tr><td>R@10</td><td>0.0346</td><td>0.0539</td><td>0.0676</td><td>0.0591</td><td>0.0423</td><td>0.0156</td><td>0.0449</td><td>0.0278</td><td>0.0305</td><td>0.0453</td></tr><tr><td>N@5</td><td>0.0124</td><td>0.0201</td><td>0.0275</td><td>0.0230</td><td>0.0164</td><td>0.0058</td><td>0.0163</td><td>0.0110</td><td>0.0116</td><td>0.0171</td></tr><tr><td>N@10</td><td>0.0170</td><td>0.0271</td><td>0.0355</td><td>0.0303</td><td>0.0217</td><td>0.0079</td><td>0.0222</td><td>0.0144</td><td>0.0158</td><td>0.0231</td></tr><tr><td rowspan="4">HSTU</td><td>R@5</td><td>0.0226</td><td>0.0456</td><td>0.0475</td><td>0.0458</td><td>0.0330</td><td>0.0134</td><td>0.0362</td><td>0.0227</td><td>0.0231</td><td>0.0489</td></tr><tr><td>R@10</td><td>0.0350</td><td>0.0643</td><td>0.0725</td><td>0.0712</td><td>0.0485</td><td>0.0197</td><td>0.0521</td><td>0.0347</td><td>0.0337</td><td>0.0649</td></tr><tr><td>N@5</td><td>0.0156</td><td>0.0308</td><td>0.0314</td><td>0.0297</td><td>0.0215</td><td>0.0092</td><td>0.0239</td><td>0.0151</td><td>0.0159</td><td>0.0339</td></tr><tr><td>N@10</td><td>0.0196</td><td>0.0368</td><td>0.0395</td><td>0.0378</td><td>0.0265</td><td>0.0112</td><td>0.0290</td><td>0.0190</td><td>0.0193</td><td>0.0391</td></tr><tr><td rowspan="4">ReaRec</td><td>R@5</td><td>0.0197</td><td>0.0488</td><td>0.0444</td><td>0.0454</td><td>0.0326</td><td>0.0150</td><td>0.0299</td><td>0.0231</td><td>0.0219</td><td>0.0517</td></tr><tr><td>R@10</td><td>0.0320</td><td>0.0702</td><td>0.0711</td><td>0.0730</td><td>0.0481</td><td>0.0210</td><td>0.0486</td><td>0.0348</td><td>0.0310</td><td>0.0706</td></tr><tr><td>N@5</td><td>0.0123</td><td>0.0341</td><td>0.0269</td><td>0.0289</td><td>0.0213</td><td>0.0101</td><td>0.0189</td><td>0.0152</td><td>0.0143</td><td>0.0369</td></tr><tr><td>N@10</td><td>0.0163</td><td>0.0409</td><td>0.0355</td><td>0.0378</td><td>0.0263</td><td>0.0121</td><td>0.0249</td><td>0.0189</td><td>0.0173</td><td>0.0430</td></tr><tr><td rowspan="4">TIGER</td><td>R@5</td><td>0.0191</td><td>0.0413</td><td>0.0540</td><td>0.0447</td><td>0.0328</td><td>0.0142</td><td>0.0343</td><td>0.0216</td><td>0.0228</td><td>0.0367</td></tr><tr><td>R@10</td><td>0.0318</td><td>0.0628</td><td>0.0786</td><td>0.0691</td><td>0.0534</td><td>0.0216</td><td>0.0542</td><td>0.0331</td><td>0.0344</td><td>0.0527</td></tr><tr><td>N@5</td><td>0.0125</td><td>0.0277</td><td>0.0350</td><td>0.0295</td><td>0.0222</td><td>0.0094</td><td>0.0232</td><td>0.0145</td><td>0.0148</td><td>0.0255</td></tr><tr><td>N@10</td><td>0.0162</td><td>0.0346</td><td>0.0429</td><td>0.0373</td><td>0.0289</td><td>0.0118</td><td>0.0295</td><td>0.0182</td><td>0.0184</td><td>0.0307</td></tr><tr><td rowspan="4">LC-Rec</td><td>R@5</td><td>0.0232</td><td>0.0495</td><td>0.0585</td><td>0.0501</td><td>0.0412</td><td>0.0199</td><td>0.0388</td><td>0.0269</td><td>0.0288</td><td>0.0350</td></tr><tr><td>R@10</td><td>0.0344</td><td>0.0764</td><td>0.0883</td><td>0.0790</td><td>0.0616</td><td>0.0293</td><td>0.0612</td><td>0.0418</td><td>0.0438</td><td>0.0549</td></tr><tr><td>N@5</td><td>0.0151</td><td>0.0338</td><td>0.0392</td><td>0.0328</td><td>0.0272</td><td>0.0138</td><td>0.0247</td><td>0.0177</td><td>0.0187</td><td>0.0221</td></tr><tr><td>N@10</td><td>0.0187</td><td>0.0424</td><td>0.0488</td><td>0.0421</td><td>0.0338</td><td>0.0168</td><td>0.0320</td><td>0.0225</td><td>0.0235</td><td>0.0285</td></tr><tr><td rowspan="4">Ours</td><td>R@5</td><td>0.0352</td><td>0.0646</td><td>0.0717</td><td>0.0688</td><td>0.0534</td><td>0.0279</td><td>0.0563</td><td>0.0365</td><td>0.0412</td><td>0.0693</td></tr><tr><td>R@10</td><td>0.0513</td><td>0.0924</td><td>0.1036</td><td>0.1029</td><td>0.0768</td><td>0.0390</td><td>0.0834</td><td>0.0547</td><td>0.0593</td><td>0.0953</td></tr><tr><td>N@5</td><td>0.0238</td><td>0.0456</td><td>0.0490</td><td>0.0460</td><td>0.0376</td><td>0.0202</td><td>0.0389</td><td>0.0252</td><td>0.0295</td><td>0.0496</td></tr><tr><td>N@10</td><td>0.0289</td><td>0.0545</td><td>0.0593</td><td>0.0570</td><td>0.0452</td><td>0.0237</td><td>0.0476</td><td>0.0310</td><td>0.0354</td><td>0.0579</td></tr><tr><td colspan="2">Improve (%) R@10</td><td>34.6  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>20.9  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>17.3  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>30.3  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>24.7  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>33.1  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>36.3  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>30.9  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>35.4  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>35.0  <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr></tbody></table>

### 6.3 Transfer Learning on Amazon Benchmark

To evaluate the transferability of OneRec-Foundation, we conduct comprehensive experiments on the Amazon benchmark, encompassing 10 distinct domains. These experiments serve to rigorously validate whether our foundation model, pre-trained on diverse open-domain data, confers a fundamental transfer advantage for modeling specific downstream recommendation distributions.

##### Main Results.

As detailed in Table 7, OneRec-Foundation establishes new state-of-the-art results across all 10 datasets. Specifically, our model achieves an average improvement of 26.8% in Recall@10 over the second-best baseline on each domain. These results empirically confirm that large-scale generative pre-training endows the model with robust transfer capabilities that far exceed traditional collaborative filtering approaches.

Through our transfer learning experiments, we identify two critical factors that significantly influence performance: Comprehensive Utilization of Pre-trained Knowledge, where we design adaptive strategies to maximally leverage the collaborative filtering signals and semantic understanding capabilities encoded during pre-training (detailed in Table 8), and Multi-Domain Joint Training, which enables the model to extract universal recommendation patterns across heterogeneous domains, as demonstrated in Figure 8. We elaborate on these two critical aspects in the following subsections.

#### 6.3.1 Adaptive Strategies for Pre-trained Model Utilization

A primary challenge in transfer learning is the distributional shift of item identifiers. Our pre-trained tokenizer is optimized on a broad, open-domain corpus (e.g., short videos, e-commerce products), resulting in a codebook that may not granularly distinguish items within a specific vertical like Amazon products. Direct application leads to a high collision rate (>30%), causing catastrophic information loss. To address this, we systematically explore three strategies to adapt pre-trained representations to the target domain:

##### Strategy 1: Extended Residual Quantization.

We extend the hierarchical depth by computing residuals from the pre-trained third layer and applying Finite Scalar Quantization (FSQ) [^18] to generate a fourth-layer code, reducing collisions to 3.05%. Remaining collisions are resolved via popularity-based decoding. This strategy achieves a 10.0% improvement in average R@10 over LC-Rec, validating effective transfer of collaborative filtering knowledge. However, the non-pretrained fourth layer disrupts the original hierarchical semantics, motivating alternative approaches.

##### Strategy 2: Text-Only Adaptation.

We bypass itemic tokens entirely, representing each item via 5 distinctive keywords extracted from its metadata, reducing collisions to 4.27%. This strategy achieves an 18.8% improvement in average R@10 over Extended Residual Quantization: the model’s linguistic core remains intact, enabling robust semantic understanding, while natural language representations prove more expressive in narrow domains. However, this approach sacrifices collaborative filtering signals embedded in pre-trained itemic tokens.

##### Strategy 3: Text-Augmented Itemic Tokens.

We concatenate the original three-layer pre-trained itemic tokens with keyword representations: \[itemic\_tokens\] + \[keywords\]. Critically, we preserve the original pre-trained itemic tokens without structural extension, maintaining hierarchical semantics. Keywords provide semantic disambiguation (collision rate 0.47%) and enable full utilization of linguistic capabilities. Table 8 shows that this strategy achieves state-of-the-art performance across nearly all datasets. The consistent gains validate that effective transfer learning requires maximizing utilization of the foundation model’s diverse capabilities—collaborative filtering, knowledge, and semantic understanding—while strictly preserving pre-trained structural integrity.

Table 8: Performance Comparison of Adaptive Strategies for Pre-trained Model Utilization. Extended Residual Quantization (collision rate: 3.05%), Text-Only Adaptation (collision rate: 4.27%), and Text-Augmented Itemic Tokens (collision rate: 0.47%).

<table><tbody><tr><td>Strategy</td><td>Metric</td><td>Baby</td><td>Beauty</td><td>Cell</td><td>Grocery</td><td>Health</td><td>Home</td><td>Pet</td><td>Sports</td><td>Tools</td><td>Toys</td></tr><tr><td rowspan="4">Extended Residual Quantization</td><td>R@5</td><td>0.0288</td><td>0.0534</td><td>0.0574</td><td>0.0562</td><td>0.0479</td><td>0.0227</td><td>0.0518</td><td>0.0315</td><td>0.0350</td><td>0.0511</td></tr><tr><td>R@10</td><td>0.0407</td><td>0.0799</td><td>0.0830</td><td>0.0861</td><td>0.0673</td><td>0.0313</td><td>0.0758</td><td>0.0447</td><td>0.0495</td><td>0.0701</td></tr><tr><td>N@5</td><td>0.0201</td><td>0.0364</td><td>0.0389</td><td>0.0383</td><td>0.0333</td><td>0.0162</td><td>0.0356</td><td>0.0215</td><td>0.0243</td><td>0.0360</td></tr><tr><td>N@10</td><td>0.0239</td><td>0.0449</td><td>0.0471</td><td>0.0480</td><td>0.0396</td><td>0.0190</td><td>0.0433</td><td>0.0258</td><td>0.0289</td><td>0.0421</td></tr><tr><td rowspan="4">Text-Only Adaptation</td><td>R@5</td><td>0.0317</td><td>0.0630</td><td>0.0688</td><td>0.0687</td><td>0.0529</td><td>0.0285</td><td>0.0548</td><td>0.0368</td><td>0.0414</td><td>0.0668</td></tr><tr><td>R@10</td><td>0.0448</td><td>0.0883</td><td>0.0985</td><td>0.1048</td><td>0.0752</td><td>0.0398</td><td>0.0850</td><td>0.0548</td><td>0.0615</td><td>0.0931</td></tr><tr><td>N@5</td><td>0.0227</td><td>0.0445</td><td>0.0473</td><td>0.0460</td><td>0.0368</td><td>0.0199</td><td>0.0382</td><td>0.0256</td><td>0.0288</td><td>0.0483</td></tr><tr><td>N@10</td><td>0.0269</td><td>0.0526</td><td>0.0569</td><td>0.0576</td><td>0.0440</td><td>0.0235</td><td>0.0478</td><td>0.0314</td><td>0.0354</td><td>0.0568</td></tr><tr><td rowspan="4">Text-Augmented Itemic Tokens</td><td>R@5</td><td>0.0352</td><td>0.0646</td><td>0.0717</td><td>0.0688</td><td>0.0534</td><td>0.0285</td><td>0.0563</td><td>0.0368</td><td>0.0414</td><td>0.0693</td></tr><tr><td>R@10</td><td>0.0513</td><td>0.0924</td><td>0.1036</td><td>0.1029</td><td>0.0768</td><td>0.0398</td><td>0.0834</td><td>0.0547</td><td>0.0593</td><td>0.0953</td></tr><tr><td>N@5</td><td>0.0238</td><td>0.0456</td><td>0.0490</td><td>0.0460</td><td>0.0376</td><td>0.0202</td><td>0.0389</td><td>0.0256</td><td>0.0295</td><td>0.0496</td></tr><tr><td>N@10</td><td>0.0289</td><td>0.0545</td><td>0.0593</td><td>0.0576</td><td>0.0452</td><td>0.0237</td><td>0.0478</td><td>0.0314</td><td>0.0354</td><td>0.0579</td></tr></tbody></table>

#### 6.3.2 Domain-Specific Training vs. Multi-Domain Joint Training

Beyond item representation, another critical factor is the training strategy across domains. We compare Domain-Specific Training versus Multi-Domain Joint Training to investigate whether our pre-trained foundation model can benefit from multi-domain knowledge integration, in contrast to traditional generative recommenders like TIGER.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2512.24762/assets/x10.png)

Figure 8: Impact of Training Strategies (Domain-Specific vs Multi-Domain Joint) and Few-Shot Learning on Transfer Performance. We compare OneRec-Foundation (Ours) against TIGER across four Amazon domains under three settings: (1) Few-shot learning with 10% training data, (2) Full-data training with domain-specific strategy, and (3) Full-data training with joint multi-domain strategy. The green dashed line represents the performance gain (Recall@10 difference) of Ours over TIGER.

The results in Figure 8 reveal a striking divergence. TIGER exhibits a consistent performance decline under joint training, with an average 10.6% drop in Recall@10. In stark contrast, OneRec-Foundation demonstrates an average 2.3% improvement.

This divergence highlights the fundamental advantage of our pre-trained foundation model. Traditional models like TIGER primarily memorize domain-specific collaborative statistics. When confronted with heterogeneous multi-domain data, they struggle to reconcile conflicting patterns, leading to performance degradation.

In contrast, OneRec-Foundation’s success stems from the unique combination of rich recommendation knowledge and semantic understanding capabilities acquired during pre-training. This enables the model to extract generalizable patterns rather than memorize domain-specific statistics. Multi-domain joint training further enriches the model by exposing it to diverse interaction patterns, enabling effective cross-domain knowledge transfer. The massive parameter capacity provides sufficient representational space to encode domain-specific nuances while maintaining shared high-level patterns.

##### Few-Shot Learning: Amplified Transfer Advantage.

Beyond training strategies, Figure 8 also reveals that the transfer learning advantage of foundation models becomes significantly more pronounced under data scarcity. While OneRec-Foundation surpasses TIGER by an average of 77.7% in Recall@10 with full training data, this gap widens dramatically to 219.7% in the 10% few-shot regime. Crucially, OneRec-Foundation preserves 45.2% of its full-data performance when restricted to 10% data, whereas TIGER retains only 23.0%. This striking resilience validates that large-scale pre-training confers robust, transferable representations that enable effective domain adaptation under severe data constraints.

### 6.4 Ablation Study

#### 6.4.1 Ablation Study on Pre-training Strategies

Table 9: Ablation study on pre-training strategies. We compare our model with ablated variants. w/o Align: removing the itemic-text alignment stage. Results are reported across three model sizes (0.6B, 1.7B, 8B). Bold indicates the best result for each task within each model size.

<table><tbody><tr><td rowspan="2">Task</td><td rowspan="2">Metric</td><td colspan="2">0.6B</td><td colspan="2">1.7B</td><td colspan="2">8B</td></tr><tr><td>Ours</td><td>w/o Align</td><td>Ours</td><td>w/o Align</td><td>Ours</td><td>w/o Align</td></tr><tr><td rowspan="2">Short Video Rec</td><td>Pass@32</td><td>0.1401</td><td>0.1397</td><td>0.1636</td><td>0.1605</td><td>0.2034</td><td>0.1933</td></tr><tr><td>Recall@32</td><td>0.0210</td><td>0.0210</td><td>0.0254</td><td>0.0251</td><td>0.0334</td><td>0.0310</td></tr><tr><td rowspan="2">Ad Rec</td><td>Pass@32</td><td>0.1740</td><td>0.1680</td><td>0.1961</td><td>0.1922</td><td>0.2350</td><td>0.2401</td></tr><tr><td>Recall@32</td><td>0.0586</td><td>0.0569</td><td>0.0673</td><td>0.0669</td><td>0.0821</td><td>0.0841</td></tr><tr><td rowspan="2">Product Rec</td><td>Pass@32</td><td>0.1139</td><td>0.1064</td><td>0.1512</td><td>0.1395</td><td>0.1893</td><td>0.1911</td></tr><tr><td>Recall@32</td><td>0.0257</td><td>0.0243</td><td>0.0343</td><td>0.0312</td><td>0.0447</td><td>0.0442</td></tr><tr><td rowspan="2">Label-Cond. Rec</td><td>Pass@32</td><td>0.0350</td><td>0.0343</td><td>0.0426</td><td>0.0401</td><td>0.0537</td><td>0.0537</td></tr><tr><td>Recall@32</td><td>0.0146</td><td>0.0145</td><td>0.0181</td><td>0.0171</td><td>0.0227</td><td>0.0230</td></tr><tr><td rowspan="2">Interactive Rec</td><td>Pass@32</td><td>0.2460</td><td>0.2360</td><td>0.3110</td><td>0.3050</td><td>0.4650</td><td>0.4490</td></tr><tr><td>Recall@32</td><td>0.1402</td><td>0.1357</td><td>0.1908</td><td>0.1770</td><td>0.3039</td><td>0.2910</td></tr><tr><td>Label Pred.</td><td>AUC</td><td>0.6488</td><td>0.5807</td><td>0.6392</td><td>0.5796</td><td>0.6879</td><td>0.6285</td></tr><tr><td>Item Understanding</td><td>LLM-Judge Score</td><td>0.3174</td><td>0.3112</td><td>0.3170</td><td>0.3181</td><td>0.3225</td><td>0.3103</td></tr><tr><td>Rec. Explanation</td><td>LLM-Judge Score</td><td>2.9960</td><td>2.8635</td><td>3.0922</td><td>3.3160</td><td>3.9420</td><td>3.9329</td></tr></tbody></table>

We conducted an ablation study to quantify the contribution of the itemic-text alignment (Stage 1) within our pre-training pipeline. We compare the full-stage model against an ablated variant (w/o Align) that bypasses this initial phase. Both versions utilized an identical pre-training recipe, including the optimizer, learning-rate schedule, and context length. Since the raw pre-trained checkpoints lack inherent instruction-following capabilities, we applied the Multi-task SFT described in Section 5.1 to both variants before benchmarking across three model scales (0.6B, 1.7B, and 8B).

The results, summarized in Table 9, reveal that Stage 1 serves as a fundamental semantic bridge for cold-started itemic token embeddings. By aligning these initialized parameters with the pre-trained latent space before full-parameter fine-tuning, Stage 1 establishes a robust semantic foundation, which is particularly necessary for smaller models(0.6B, 1.7B). The marginal gains from Stage 1 scale inversely with model size, likely because larger backbones possess greater inherent generalization. However, this stage remains essential for domain-specific precision, particularly in label prediction and interactive recommendation. These findings underscore that explicit alignment is a prerequisite for optimizing recommendation performance across all model scales.

#### 6.4.2 Evolution of Model Capabilities Across Post-training Stages

We also analyze the performance of our model after each key post-training phase: Multi-task Supervised Fine-tuning, On-policy Distillation for General Capability, and Reinforcement Learning for Recommendation. As shown in Table 10, Table 11, and Table 12, each stage plays a distinct role in balancing the performance between recommendation-domain and general-domain.

Table 10: Performance comparison on general capability across post-training stages (Thinking).

|  | math\_500 | gsm8k | AIME’24 | mmlu\_pro | gpqa\_diamond | IFEVAL | LiveCodeBench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B (Base) | 0.952 | 0.9568 | 0.7917 | 0.7235 | 0.5606 | 0.8577 | 0.5484 |
| Stage 1: Multi-task SFT | 0.936 | 0.9083 | 0.5104 | 0.5307 | 0.4949 | 0.6174 | 0.4516 |
| Stage 2: On-Policy Distillation | 0.948 | 0.9538 | 0.7125 | 0.5454 | 0.5 | 0.7653 | 0.4659 |
| Stage 3: Reinforcement Learning | 0.938 | 0.9575 | 0.7188 | 0.5204 | 0.5051 | 0.7634 | 0.4667 |

Table 11: Performance comparison on general capability across post-training stages (Non-Thinking).

|  | math\_500 | gsm8k | AIME’24 | mmlu\_pro | gpqa\_diamond | IFEVAL | LiveCodeBench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B (Base) | 0.838 | 0.9303 | 0.2729 | 0.6632 | 0.399 | 0.8392 | 0.276 |
| Stage 1: Multi-task SFT | 0.876 | 0.906 | 0.0688 | 0.4909 | 0.3384 | 0.5638 | 0.1756 |
| Stage 2: On-Policy Distillation | 0.848 | 0.9234 | 0.2521 | 0.583 | 0.4091 | 0.7689 | 0.2545 |
| Stage 3: Reinforcement Learning | 0.798 | 0.9196 | 0.2271 | 0.4521 | 0.3939 | 0.7098 | 0.2401 |

Impact of On-policy Distillation on General Capabilities. As shown in Table 10 and Table 11, a comparison between Stage 1 (Multi-task SFT) and Stage 2 (On-policy Distillation) reveals that on-policy distillation significantly restores general capabilities, effectively realigning the model with the Qwen3 baseline on most general benchmarks. Despite this marked improvement, a performance gap persists relative to the original Qwen3 base model across several metrics. This gap is likely attributable to the current data composition and quality during the distillation phase, suggesting that further refinement of the data strategy is required to fully bridge the remaining gap.

Interestingly, after the Multi-task SFT stage, we observed that several metrics in the "Non-Thinking" mode are unexpectedly high, occasionally even surpassing the Qwen3 base model. Qualitative results indicate this is due to instruction drift, where the model disregards the /no\_think tag and generates "thinking" trajectories (CoT), leading to inflated scores. This issue is effectively mitigated through On-policy Distillation, which restores the model’s ability to faithfully switch between distinct reasoning modes.

Advancements through RL for Recommendation. The final reinforcement learning stage demonstrates targeted improvements on core recommendation tasks. As illustrated in Table 12, the RL-trained model achieves consistent gains on Reco. tasks. These improvements stem from the rule-based Hit reward that directly optimizes for ranking accuracy, encouraging the model to assign higher probability mass to target itemic tokens. Notably, the Reco Reason task also benefits from RL training. This suggests that the refined “recommendation intuition” acquired through RL transfers to explanation generation, producing more coherent and relevant reasoning.

Table 12: Recommendation benchmark performance across post-training stages.

| Model | Video Rec | Ad Rec | Product Rec | Label Cond. | Interactive | Label Pred. | Item Understanding | Reco Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage 1: Multi-task SFT | 0.0324 | 0.0925 | 0.0532 | 0.0229 | 0.3461 | 0.6979 | 0.3274 | 3.8795 |
| Stage 2: On-Policy Distillation | 0.0304 | 0.0596 | 0.0330 | 0.0200 | 0.2419 | 0.6944 | 0.3319 | 3.9479 |
| Stage 3: Reinforcement Learning | 0.0370 | 0.0967 | 0.0536 | 0.0236 | 0.3458 | 0.6908 | 0.3209 | 4.0381 |

## 7 Conclusion, Limitations, and Future Directions

In this work, we presented OpenOneRec, a comprehensive framework designed to bridge the gap between traditional recommendation systems and Large Language Models. We proposed RecIF-Bench, the first holistic benchmark for evaluating recommendation instruction-following capabilities, encompassing diverse tasks from fundamental prediction to complex reasoning. To facilitate reproducibility and scalable research, we open-sourced a full-stack training pipeline—including data processing, co-pretraining, and post-training protocols—and validated the scaling laws of recommendation capabilities. Extensive experiments demonstrate that our OpenOneRec-Foundation models achieve state-of-the-art performance across RecIF-Bench and show exceptional transferability to external domains, proving the efficacy of our unified generative paradigm. Despite these advancements, several limitations remain that point towards important future research directions.

First, while our experiments confirm that the recommendation backbone significantly enhances downstream performance, the magnitude of these gains is currently constrained by tokenizer transferability. A promising avenue for future work lies in maximizing the reuse of foundation model priors while simultaneously ensuring high-quality item indexing (code quality) for downstream tasks. Second, maintaining the model’s general intelligence and reasoning capabilities necessitates mixing vast amounts of general-domain text during training. Investigating the optimal data mixing ratios and improving data utilization efficiency are urgent challenges to balance domain-specific precision with general capabilities. Third, we observe that Chain-of-Thought reasoning currently yields improvements only in limited settings. This underscores the need for a more rigorous exploration of test-time scaling strategies to unlock consistent reasoning gains across diverse recommendation scenarios.

We believe that addressing these challenges requires collective effort from the community. By open-sourcing our entire framework, we invite contributions and encourage researchers to build upon OpenOneRec. We hope this work serves as a solid foundation for future exploration, accelerating the evolution towards truly intelligent recommendation systems.

\*

## References

## Appendix A Contributions

Within each role, authors are listed alphabetically by their first name.

Core Contributors  
Guorui Zhou  
Honghui Bao  
Jiaming Huang  
Jiaxin Deng  
Jinghao Zhang  
Junda She  
Kuo Cai  
Lejian Ren  
Lu Ren  
Qiang Luo  
Qianqian Wang  
Qigen Hu  
Rongzhou Zhang  
Ruiming Tang  
Shiyao Wang  
Wuchao Li  
Xiangyu Wu  
Xinchen Luo  
Xingmei Wang  
Yifei Hu  
Yunfan Wu  
Zhanyu Liu  
Zhiyang Zhang  
Zixing Zhang

Contributors  
Bo Chen  
Bin Wen  
Chaoyi Ma  
Chengru Song  
Chenglong Chu  
Defu Lian  
Fan Yang  
Feng Jiang  
Hongtao Cheng  
Huanjie Wang  
Kun Gai  
Pengfei Zheng  
Qiang Wang  
Rui Huang  
Siyang Mao  
Tingting Gao  
Wei Yuan  
Yan Wang  
Yang Zhou  
Yi Su  
Zexuan Cheng  
Zhixin Ling  
Ziming Li

## Appendix B Appendix

### B.1 Evaluation Details for Item Understanding

For the Item Understanding task (Layer 0), we employ a rigorous LLM-as-Judge framework to evaluate the semantic alignment between the model-generated captions and the ground truth. The ground truth captions are generated by Gemini-2.5-Pro given the video frames and metadata. The evaluation process consists of three steps: Information Point Extraction, Semantic Matching, and Weighted Scoring. We use Gemini-2.5-Flash-Lite as the judge for extraction and matching.

#### B.1.1 WIP Extraction and Matching

We first prompt the LLM to decompose captions into atomic Weighted Information Points (WIPs), each containing a fact statement and an importance score (1-5). Then, we employ the LLM to align model-generated WIPs with ground truth WIPs, identifying valid matches, hallucinations (unmatched model WIPs), and omissions (unmatched GT WIPs). The specific prompts are detailed in Section B.2.

#### B.1.2 Scoring Protocol

To compute the final score, we calculate a Double-Weighted F1 Score that incorporates both the semantic importance of the information and the quality of the match. For each matched pair $(w_{gt},w_{model})$, we compute a match quality score $q\in[0,1]$, which is the F1 score calculated by BERTScore (using bert-base-chinese).

The contributions to True Positives ($TP_{i}$), False Negatives ($FN_{i}$), and False Positives ($FP_{i}$) for each sample $i$ are calculated as follows:

$$
\displaystyle TP_{i}
$$
 
$$
\displaystyle=\sum_{(w_{gt},w_{model})\in\text{Matches}}\text{score}(w_{gt})\times q
$$
 
$$
\displaystyle FN_{i}
$$
 
$$
\displaystyle=\sum_{(w_{gt},w_{model})\in\text{Matches}}\text{score}(w_{gt})\times(1-q)+\sum_{w_{gt}\in\text{Unmatched GT}}\text{score}(w_{gt})
$$
 
$$
\displaystyle FP_{i}
$$
 
$$
\displaystyle=\sum_{(w_{gt},w_{model})\in\text{Matches}}\text{score}(w_{model})\times(1-q)+\sum_{w_{model}\in\text{Unmatched Model}}\text{score}(w_{model})
$$

The F1 score for sample $i$ is computed as:

$$
F1_{i}=\frac{2\cdot TP_{i}}{2\cdot TP_{i}+FP_{i}+FN_{i}}
$$

The final score is the LLM-Judge Score, computed as the average of per-sample F1 scores:

$$
\text{LLM-Judge Score}=\frac{1}{N}\sum_{i=1}^{N}F1_{i}
$$

This metric penalizes both hallucinations (high FP) and omissions (high FN), while rewarding high-quality matches on important information points.

### B.2 Prompts for Item Understanding

We provide the core instructions of the prompts used for the Item Understanding evaluation below.

<svg id="A2.SS2.p2.pic1" height="176.02" overflow="visible" version="1.1" viewBox="0 0 600 176.02" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,176.02) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.87 L 0 156.34 C 0 158.48 1.73 160.22 3.87 160.22 L 596.13 160.22 C 598.27 160.22 600 158.48 600 156.34 L 600 3.87 C 600 1.73 598.27 0 596.13 0 L 3.87 0 C 1.73 0 0 1.73 0 3.87 Z"></path></g><g style="--ltx-fill-color:#FAFAFA;" fill="#FAFAFA" fill-opacity="1.0"><path style="stroke:none" d="M 1.11 3.87 L 1.11 156.34 C 1.11 157.87 2.35 159.11 3.87 159.11 L 596.13 159.11 C 597.65 159.11 598.89 157.87 598.89 156.34 L 598.89 3.87 C 598.89 2.35 597.65 1.11 596.13 1.11 L 3.87 1.11 C 2.35 1.11 1.11 2.35 1.11 3.87 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 0 152.35)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 23.68 L 271.56 23.68 L 271.56 0 Z"></path></g><g style="--ltx-fill-color:#000000;" fill="#000000" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 23.68 L 271.56 23.68 L 271.56 0 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.81 8.38)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:16.29em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.23em;" width="247.93" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A2.SS2.p2.pic1.1.1.1.1.1.1.1" style="--ltx-fg-color:#FFFFFF;">Prompt for WIP Extraction (Abridged)</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.35 141.34)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.43em;--ltx-fo-height:0.6em;--ltx-fo-depth:9.45em;" width="573.31" height="139.06" transform="matrix(1 0 0 -1 0 8.3)" overflow="visible" color="#000000"><span id="A2.SS2.p2.pic1.2.2.2.1.1" style="width:48.74em;"><span id="A2.SS2.p2.pic1.2.2.2.1.1.1"><span id="A2.SS2.p2.pic1.2.2.2.1.1.1.1" style="font-size:80%;">任务<span id="A2.SS2.p2.pic1.2.2.2.1.1.1.1.1">: 你是一位信息抽取专家。请将描述性文本分解为结构化的【原子化且唯一】的"信息点 (WIPs)"列表。</span></span></span> <span id="A2.SS2.p2.pic1.2.2.2.1.1.2"><span id="A2.SS2.p2.pic1.2.2.2.1.1.2.1" style="font-size:80%;">输出结构<span id="A2.SS2.p2.pic1.2.2.2.1.1.2.1.1">:</span></span></span> <span id="A2.I1"><span id="A2.I1.i1" style="list-style-type:none;">• <span id="A2.I1.i1.p1"><span id="A2.I1.i1.p1.1"><span id="A2.I1.i1.p1.1.1" style="font-size:80%;">info_point</span><span id="A2.I1.i1.p1.1.2" style="font-size:80%;">: 一个简洁的、陈述事实的短语。</span></span> </span></span><span id="A2.I1.i2" style="list-style-type:none;">• <span id="A2.I1.i2.p1"><span id="A2.I1.i2.p1.1"><span id="A2.I1.i2.p1.1.1" style="font-size:80%;">importance_score</span><span id="A2.I1.i2.p1.1.2" style="font-size:80%;">: 整数 [1-5] (5=核心, 1=琐碎)。</span></span> </span></span></span><span id="A2.SS2.p2.pic1.2.2.2.1.1.3"><span id="A2.SS2.p2.pic1.2.2.2.1.1.3.1" style="font-size:80%;">关键原则<span id="A2.SS2.p2.pic1.2.2.2.1.1.3.1.1">:</span></span></span> <span id="A2.I2"><span id="A2.I2.i1" style="list-style-type:none;">1. <span id="A2.I2.i1.p1"><span id="A2.I2.i1.p1.1"><span id="A2.I2.i1.p1.1.1" style="font-size:80%;">原子性</span><span id="A2.I2.i1.p1.1.2" style="font-size:80%;">: 每个点只包含一个独立事实。</span></span> </span></span><span id="A2.I2.i2" style="list-style-type:none;">2. <span id="A2.I2.i2.p1"><span id="A2.I2.i2.p1.1"><span id="A2.I2.i2.p1.1.1" style="font-size:80%;">唯一性</span><span id="A2.I2.i2.p1.1.2" style="font-size:80%;">: 概念上不重复。</span></span> </span></span><span id="A2.I2.i3" style="list-style-type:none;">3. <span id="A2.I2.i3.p1"><span id="A2.I2.i3.p1.1"><span id="A2.I2.i3.p1.1.1" style="font-size:80%;">合并</span><span id="A2.I2.i3.p1.1.2" style="font-size:80%;">: 将相似描述合并为一个代表性点。</span></span> </span></span></span><span id="A2.SS2.p2.pic1.2.2.2.1.1.4"><span id="A2.SS2.p2.pic1.2.2.2.1.1.4.1" style="font-size:80%;">打分指南<span id="A2.SS2.p2.pic1.2.2.2.1.1.4.1.1">: 5 (灵魂/核心), 4 (骨架/关键事件), 3 (血肉/重要细节), 2 (背景/次要细节), 1 (琐碎信息)。</span></span></span></span></foreignObject></g></g></svg>

<svg id="A2.SS2.p3.pic1" height="149.18" overflow="visible" version="1.1" viewBox="0 0 600 149.18" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,149.18) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.87 L 0 129.5 C 0 131.64 1.73 133.38 3.87 133.38 L 596.13 133.38 C 598.27 133.38 600 131.64 600 129.5 L 600 3.87 C 600 1.73 598.27 0 596.13 0 L 3.87 0 C 1.73 0 0 1.73 0 3.87 Z"></path></g><g style="--ltx-fill-color:#FAFAFA;" fill="#FAFAFA" fill-opacity="1.0"><path style="stroke:none" d="M 1.11 3.87 L 1.11 129.5 C 1.11 131.03 2.35 132.27 3.87 132.27 L 596.13 132.27 C 597.65 132.27 598.89 131.03 598.89 129.5 L 598.89 3.87 C 598.89 2.35 597.65 1.11 596.13 1.11 L 3.87 1.11 C 2.35 1.11 1.11 2.35 1.11 3.87 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 0 125.5)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 23.68 L 265.92 23.68 L 265.92 0 Z"></path></g><g style="--ltx-fill-color:#000000;" fill="#000000" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 23.68 L 265.92 23.68 L 265.92 0 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.81 8.38)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:15.94em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.23em;" width="242.69" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible" color="#000000"><span id="A2.SS2.p3.pic1.1.1.1.1.1.1.1" style="--ltx-fg-color:#FFFFFF;">Prompt for WIP Matching (Abridged)</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.35 115.05)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.43em;--ltx-fo-height:0.56em;--ltx-fo-depth:7.55em;" width="573.31" height="112.22" transform="matrix(1 0 0 -1 0 7.75)" overflow="visible" color="#000000"><span id="A2.SS2.p3.pic1.2.2.2.1.1" style="width:48.74em;"><span id="A2.SS2.p3.pic1.2.2.2.1.1.1"><span id="A2.SS2.p3.pic1.2.2.2.1.1.1.1" style="font-size:80%;">任务<span id="A2.SS2.p3.pic1.2.2.2.1.1.1.1.1">: 你是一位语义匹配专家。请对比"Ground Truth WIPs"和"Model-Generated WIPs"，识别匹配项、幻觉和漏报。</span></span></span> <span id="A2.SS2.p3.pic1.2.2.2.1.1.2"><span id="A2.SS2.p3.pic1.2.2.2.1.1.2.1" style="font-size:80%;">匹配规则<span id="A2.SS2.p3.pic1.2.2.2.1.1.2.1.1">:</span></span></span> <span id="A2.I3"><span id="A2.I3.i1" style="list-style-type:none;">1. <span id="A2.I3.i1.p1"><span id="A2.I3.i1.p1.1"><span id="A2.I3.i1.p1.1.1" style="font-size:80%;">语义核心</span><span id="A2.I3.i1.p1.1.2" style="font-size:80%;">: 基于</span> <span id="A2.I3.i1.p1.1.3" style="font-size:80%;">info_point</span> <span id="A2.I3.i1.p1.1.4" style="font-size:80%;">的核心含义进行匹配。</span></span> </span></span><span id="A2.I3.i2" style="list-style-type:none;">2. <span id="A2.I3.i2.p1"><span id="A2.I3.i2.p1.1"><span id="A2.I3.i2.p1.1.1" style="font-size:80%;">部分匹配</span><span id="A2.I3.i2.p1.1.2" style="font-size:80%;">: 只要有语义重叠即可匹配 (例如："篮球比赛" 匹配 "球员打篮球")。</span></span> </span></span><span id="A2.I3.i3" style="list-style-type:none;">3. <span id="A2.I3.i3.p1"><span id="A2.I3.i3.p1.1"><span id="A2.I3.i3.p1.1.1" style="font-size:80%;">一对一</span><span id="A2.I3.i3.p1.1.2" style="font-size:80%;">: 寻找最佳匹配对。</span></span> </span></span></span><span id="A2.SS2.p3.pic1.2.2.2.1.1.3"><span id="A2.SS2.p3.pic1.2.2.2.1.1.3.1" style="font-size:80%;">输出结构<span id="A2.SS2.p3.pic1.2.2.2.1.1.3.1.1">:</span></span></span> <span id="A2.I4"><span id="A2.I4.i1" style="list-style-type:none;">• <span id="A2.I4.i1.p1"><span id="A2.I4.i1.p1.1"><span id="A2.I4.i1.p1.1.1" style="font-size:80%;">matches</span><span id="A2.I4.i1.p1.1.2" style="font-size:80%;">: 语义相似的配对列表</span> <span id="A2.I4.i1.p1.1.3" style="font-size:80%;">{model_wip, gt_wip}</span> <span id="A2.I4.i1.p1.1.4" style="font-size:80%;">。</span></span> </span></span><span id="A2.I4.i2" style="list-style-type:none;">• <span id="A2.I4.i2.p1"><span id="A2.I4.i2.p1.1"><span id="A2.I4.i2.p1.1.1" style="font-size:80%;">unmatched_model_wips</span><span id="A2.I4.i2.p1.1.2" style="font-size:80%;">: 幻觉 (False Positives)。</span></span> </span></span><span id="A2.I4.i3" style="list-style-type:none;">• <span id="A2.I4.i3.p1"><span id="A2.I4.i3.p1.1"><span id="A2.I4.i3.p1.1.1" style="font-size:80%;">unmatched_gt_wips</span><span id="A2.I4.i3.p1.1.2" style="font-size:80%;">: 漏报 (False Negatives)。</span></span></span></span></span></span></foreignObject></g></g></svg>

### B.3 Samples of Recommendation-Domain Pre-training Data

In this section, we provide some simple samples for the three recommendation-domain pre-training data discussed in Section 4.1.1.

#### B.3.1 Itemic Dense Caption Data

<svg id="A2.SS3.SSS1.p1.pic1" height="424.25" overflow="visible" version="1.1" viewBox="0 0 600 424.25" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,424.25) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.87 L 0 406.11 C 0 408.25 1.73 409.98 3.87 409.98 L 596.13 409.98 C 598.27 409.98 600 408.25 600 406.11 L 600 3.87 C 600 1.73 598.27 0 596.13 0 L 3.87 0 C 1.73 0 0 1.73 0 3.87 Z"></path></g><g style="--ltx-fill-color:#FAFAFA;" fill="#FAFAFA" fill-opacity="1.0"><path style="stroke:none" d="M 1.11 3.87 L 1.11 406.11 C 1.11 407.64 2.35 408.88 3.87 408.88 L 596.13 408.88 C 597.65 408.88 598.89 407.64 598.89 406.11 L 598.89 3.87 C 598.89 2.35 597.65 1.11 596.13 1.11 L 3.87 1.11 C 2.35 1.11 1.11 2.35 1.11 3.87 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 0 402.11)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 22.14 L 197.48 22.14 L 197.48 0 Z"></path></g><g style="--ltx-fill-color:#000000;" fill="#000000" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 22.14 L 197.48 22.14 L 197.48 0 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.81 7.61)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:11.42em;--ltx-fo-height:0.63em;--ltx-fo-depth:0.18em;" width="173.86" height="12.3" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" color="#000000"><span id="A2.SS3.SSS1.p1.pic1.1.1.1.1.1.1.1" style="--ltx-fg-color:#FFFFFF;">Itemic Dense Caption Data</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.35 10.58)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.43em;--ltx-fo-height:28.1em;--ltx-fo-depth:0em;" width="573.31" height="388.83" transform="matrix(1 0 0 -1 0 388.83)" overflow="visible" color="#000000"><span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1" style="width:48.74em;"><span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1.1"><span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1.1.1" style="font-size:80%;">视频 <span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1.1.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_5028&gt;&lt;item_b_6733&gt;&lt;item_c_2559&gt;&lt;|item_end|&gt;</span> 展示了以下内容：视频内容聚焦在庆祝冬至这一重要节日的习俗，特别是享受饺子与汤圆等美食。视频表达了冬至节气的特色意义，以及人们对新一年开始的寓意。内容上，显现出浓浓的节日气氛与家庭温暖，可能会触动那些寻求传统节日体验和家的感觉的观众。视频还可能激发观众对中华传统文化的兴趣，以及对家人团聚时的美好记忆。通过美食与节日的结合，观众可感受到温馨和幸福，为冬至节日的到来营造了欢乐与期盼。</span></span> <span style="width:433.6pt;height:0.4pt;--ltx-bg-color:black;display:inline-block;"></span><span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1.2"><span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1.2.1" style="font-size:80%;--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_5028&gt;&lt;item_b_6733&gt;&lt;item_c_2559&gt;&lt;|item_end|&gt;</span> <span id="A2.SS3.SSS1.p1.pic1.2.2.2.1.1.2.2" style="font-size:80%;">focuses on the customs of celebrating the Winter Solstice, a significant traditional festival, with a particular emphasis on enjoying delicacies such as dumplings and tangyuan. It conveys the unique cultural significance of the Winter Solstice solar term and its symbolism as the beginning of a new cycle. In terms of content, the video exudes a profound festive atmosphere and familial warmth, likely resonating with viewers seeking traditional holiday experiences and a sense of home. Furthermore, it may spark interest in traditional Chinese culture and evoke cherished memories of family reunions. By blending culinary traditions with the holiday spirit, the video allows the audience to experience warmth and happiness, fostering a sense of joy and anticipation for the arrival of the Winter Solstice.</span></span></span></foreignObject></g></g></svg>

#### B.3.2 Sequential User Behavior Data

<svg id="A2.SS3.SSS2.p1.pic1" height="252.74" overflow="visible" version="1.1" viewBox="0 0 600 252.74" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,252.74) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.87 L 0 234.59 C 0 236.73 1.73 238.47 3.87 238.47 L 596.13 238.47 C 598.27 238.47 600 236.73 600 234.59 L 600 3.87 C 600 1.73 598.27 0 596.13 0 L 3.87 0 C 1.73 0 0 1.73 0 3.87 Z"></path></g><g style="--ltx-fill-color:#FAFAFA;" fill="#FAFAFA" fill-opacity="1.0"><path style="stroke:none" d="M 1.11 3.87 L 1.11 234.59 C 1.11 236.12 2.35 237.36 3.87 237.36 L 596.13 237.36 C 597.65 237.36 598.89 236.12 598.89 234.59 L 598.89 3.87 C 598.89 2.35 597.65 1.11 596.13 1.11 L 3.87 1.11 C 2.35 1.11 1.11 2.35 1.11 3.87 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 0 230.59)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 22.14 L 219.1 22.14 L 219.1 0 Z"></path></g><g style="--ltx-fill-color:#000000;" fill="#000000" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 22.14 L 219.1 22.14 L 219.1 0 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.81 7.61)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:12.89em;--ltx-fo-height:0.63em;--ltx-fo-depth:0.18em;" width="196.25" height="12.3" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" color="#000000"><span id="A2.SS3.SSS2.p1.pic1.1.1.1.1.1.1.1" style="--ltx-fg-color:#FFFFFF;">Sequential User Behavior Data</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.35 13.35)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.43em;--ltx-fo-height:15.5em;--ltx-fo-depth:0.2em;" width="573.31" height="217.31" transform="matrix(1 0 0 -1 0 214.54)" overflow="visible" color="#000000"><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1" style="width:48.74em;"><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1"><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1" style="font-size:80%;">用户的曝光序列为 <span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_1023&gt;&lt;s_b_5426&gt;&lt;s_c_6422&gt;&lt;|item_end|&gt;</span>,<br><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1.2" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_3168&gt;&lt;s_b_7950&gt;&lt;s_c_4134&gt;&lt;|item_end|&gt;</span>,……；<br>其中长播列表是 <span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1.3" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_4988&gt;&lt;s_b_7436&gt;&lt;s_c_2477&gt;&lt;|item_end|&gt;</span>,<br><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1.4" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_5087&gt;&lt;s_b_7888&gt;&lt;s_c_4759&gt;&lt;|item_end|&gt;</span>,……；<br>点赞列表是 <span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1.5" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_3168&gt;&lt;s_b_7950&gt;&lt;s_c_4134&gt;&lt;|item_end|&gt;</span>,<br><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.1.1.6" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_250&gt;&lt;s_b_2310&gt;&lt;s_c_4925&gt;&lt;|item_end|&gt;</span>,……</span></span> <span style="width:433.6pt;height:0.4pt;--ltx-bg-color:black;display:inline-block;"></span><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2"><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1" style="font-size:80%;">The user’s exposure sequence is <span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_1023&gt;&lt;s_b_5426&gt;&lt;s_c_6422&gt;&lt;|item_end|&gt;</span>,<br><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1.2" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_3168&gt;&lt;s_b_7950&gt;&lt;s_c_4134&gt;&lt;|item_end|&gt;</span>, …;<br>The long-viewed list is <span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1.3" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_4988&gt;&lt;s_b_7436&gt;&lt;s_c_2477&gt;&lt;|item_end|&gt;</span>,<br><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1.4" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_5087&gt;&lt;s_b_7888&gt;&lt;s_c_4759&gt;&lt;|item_end|&gt;</span>, …;<br>The like list is <span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1.5" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_3168&gt;&lt;s_b_7950&gt;&lt;s_c_4134&gt;&lt;|item_end|&gt;</span>,<br><span id="A2.SS3.SSS2.p1.pic1.2.2.2.1.1.2.1.6" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;s_a_250&gt;&lt;s_b_2310&gt;&lt;s_c_4925&gt;&lt;|item_end|&gt;</span>, …</span></span></span></foreignObject></g></g></svg>

#### B.3.3 Interleaved User Persona Grounding Data

<svg id="A2.SS3.SSS3.p1.pic1" height="409.02" overflow="visible" version="1.1" viewBox="0 0 600 409.02" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,409.02) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 3.87 L 0 390.88 C 0 393.02 1.73 394.76 3.87 394.76 L 596.13 394.76 C 598.27 394.76 600 393.02 600 390.88 L 600 3.87 C 600 1.73 598.27 0 596.13 0 L 3.87 0 C 1.73 0 0 1.73 0 3.87 Z"></path></g><g style="--ltx-fill-color:#FAFAFA;" fill="#FAFAFA" fill-opacity="1.0"><path style="stroke:none" d="M 1.11 3.87 L 1.11 390.88 C 1.11 392.41 2.35 393.65 3.87 393.65 L 596.13 393.65 C 597.65 393.65 598.89 392.41 598.89 390.88 L 598.89 3.87 C 598.89 2.35 597.65 1.11 596.13 1.11 L 3.87 1.11 C 2.35 1.11 1.11 2.35 1.11 3.87 Z"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 0 386.88)"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#4D4D4D;" fill="#4D4D4D" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 22.14 L 251.25 22.14 L 251.25 0 Z"></path></g><g style="--ltx-fill-color:#000000;" fill="#000000" fill-opacity="1.0"><path style="stroke:none" d="M 0 0 L 0 22.14 L 251.25 22.14 L 251.25 0 Z"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 11.81 7.61)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:15.06em;--ltx-fo-height:0.63em;--ltx-fo-depth:0.18em;" width="229.16" height="12.3" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" color="#000000"><span id="A2.SS3.SSS3.p1.pic1.1.1.1.1.1.1.1" style="--ltx-fg-color:#FFFFFF;">Interleaved User Persona Grounding</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.35 10.58)"><foreignObject style="--ltx-fg-color:#000000;--ltx-fo-width:41.43em;--ltx-fo-height:27em;--ltx-fo-depth:0em;" width="573.31" height="373.6" transform="matrix(1 0 0 -1 0 373.6)" overflow="visible" color="#000000"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1" style="width:48.74em;"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.1"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.1.1" style="font-size:80%;">平台上有一名用户，她创作内容涵盖：8个其他，1个美食，1个数码，1个明星娱乐。</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.2"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.2.1" style="font-size:80%;">她近期的搜索记录包括：怎么拍游戏视频、黑白头像可爱、……。</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.3"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.3.1" style="font-size:80%;">她近期的购买记录包括：商品 <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.3.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_6133&gt;&lt;item_b_5060&gt;&lt;item_c_5431&gt;&lt;|item_end|&gt;</span> ，具体类型为【女装-裤子-休闲裤】，花费290元。</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.4"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.4.1" style="font-size:80%;">她近期在视频 <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.4.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_3316&gt;&lt;item_b_7440&gt;&lt;item_c_2022&gt;&lt;|item_end|&gt;</span> 下评论了"这个短剧叫什么名字啊"；在视频 <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.4.1.2" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_7822&gt;&lt;item_b_1648&gt;&lt;item_c_5756&gt;&lt;|item_end|&gt;</span> 下评论了"嘻嘻嘻，真的吗？我也喜欢玩蛋仔派对，早就关注你了"；……。</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.5"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.5.1" style="font-size:80%;">她点赞了视频 <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.5.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_5743&gt;&lt;item_b_930&gt;&lt;item_c_1231&gt;&lt;|item_end|&gt;</span> ……；</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.6"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.6.1" style="font-size:80%;">收藏了视频 <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.6.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_468&gt;&lt;item_b_8186&gt;&lt;item_c_5877&gt;&lt;|item_end|&gt;</span> ……；</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.7"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.7.1" style="font-size:80%;">分享了视频……。她关注的博主类型有：【其他】占47.58%，【颜值】占16.52%，【明星娱乐】占8.37%，……。</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.8"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.8.1" style="font-size:80%;">她近期观看的直播类型包括：【闲聊互动-热闹闲聊】分类下的直播点赞了6次，评论了59次；……</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.9"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.9.1" style="font-size:80%;">她过去30天观看时间最长的1种短剧类型分别是:[解密_悬疑]看了30.0分钟</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.10"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.10.1" style="font-size:80%;">……</span></span> <span style="width:433.6pt;height:0.4pt;--ltx-bg-color:black;display:inline-block;"></span><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.11"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.11.1" style="font-size:80%;">There is a user on the platform whose content creation covers: 8 in "Other," 1 in "Food," 1 in "Digital Tech," and 1 in "Celebrity &amp; Entertainment."</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.12"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.12.1" style="font-size:80%;">Her recent search history includes: "how to film game videos," "cute black and white avatars," …</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.13"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.13.1" style="font-size:80%;">Her recent purchase records include the item <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.13.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_6133&gt;&lt;item_b_5060&gt;&lt;item_c_5431&gt;&lt;|item_end|&gt;</span>, categorized as [Women’s Wear - Pants - Casual Pants], costing 290 RMB.</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.14"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.14.1" style="font-size:80%;">In terms of recent social interactions, she commented "What is the name of this short drama?" under the video <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.14.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_3316&gt;&lt;item_b_7440&gt;&lt;item_c_2022&gt;&lt;|item_end|&gt;</span>; and commented "Really? I love playing Eggy Party too! Actually, I’ve been following you for a while" under the video <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.14.1.2" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_7822&gt;&lt;item_b_1648&gt;&lt;item_c_5756&gt;&lt;|item_end|&gt;</span>; …</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.15"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.15.1" style="font-size:80%;">She also liked the video <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.15.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_5743&gt;&lt;item_b_930&gt;&lt;item_c_1231&gt;&lt;|item_end|&gt;</span> …;</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.16"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.16.1" style="font-size:80%;">Bookmarked the video <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.16.1.1" style="--ltx-fg-color:#2563EB;">&lt;|item_begin|&gt;&lt;item_a_468&gt;&lt;item_b_8186&gt;&lt;item_c_5877&gt;&lt;|item_end|&gt;</span> …</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.17"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.17.1" style="font-size:80%;">Shared the video …</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.18"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.18.1" style="font-size:80%;">The types of creators she follows are: [Others] accounting for 47.58%, [Beauty] accounting for 16.52%, [Entertainment] accounting for 8.37%, and …</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.19"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.19.1" style="font-size:80%;">Her recent livestream viewing history includes: 6 likes and 59 comments in the [Chat &amp; Interaction - Lively Chat] category; and 9 comments in another…</span></span> <span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.20"><span id="A2.SS3.SSS3.p1.pic1.2.2.2.1.1.20.1" style="font-size:80%;">Over the past 30 days, her most-watched short drama genre was [Mystery/Suspense], with a total viewing time of 30.0 minutes…</span></span></span></foreignObject></g></g></svg>

### B.4 Detailed Data Composition and Token Budgets for Pre-training

Table 13 presents the main datasets used in Stage 2 of our Open model’s pre-training. To ensure the model maintains robust general reasoning capabilities while adapting to recommendation tasks, we curate a diverse data mixture that is primarily composed of two parts:

General Domain Data: All general-domain datasets listed in the table are publicly available and can be downloaded from HuggingFace repository <sup>3</sup>. We incorporate a significant proportion of mathematical and coding datasets (e.g., Nemotron and OpenMath series) alongside general reasoning corpora to prevent the catastrophic forgetting of logic and instruction-following abilities.

Recommendation Domain Data: The three recommendation-domain datasets (Itemic Dense Caption Data, Interleaved User Persona Grounding Data, and Sequential User Behavior Data) are constructed from the raw meta data we have released, following the formats illustrated in the examples above (Section B.3).

Table 13: Data mixture for Pre-training. The table presents the distribution across general domains and recommendation domains, showing the sampling weight of each dataset and the subtotal ratio for each category.

<table><tbody><tr><td>Dataset</td><td>Weight (%)</td><td>Category</td><td>Subtotal (%)</td><td>Token Budget</td></tr><tr><td>Nemotron_CC_Math_v1 <sup>4</sup></td><td>37.41%</td><td>Math</td><td rowspan="11">62.34%</td><td rowspan="11">29B</td></tr><tr><td>Nemotron_Pretraining_Code_v1 <sup>5</sup></td><td>12.91%</td><td>Code</td></tr><tr><td>Nemotron_CC_v2 <sup>6</sup></td><td>5.59%</td><td>Math, General, Code</td></tr><tr><td>reasoning_v1_20m <sup>7</sup></td><td>4.04%</td><td>General</td></tr><tr><td>OpenMathReasoning <sup>8</sup></td><td>1.16%</td><td>Math</td></tr><tr><td>NuminaMath-QwQ-CoT-5M <sup>9</sup></td><td>0.79%</td><td>Math</td></tr><tr><td>OpenCodeReasoning <sup>10</sup></td><td>0.26%</td><td>Code</td></tr><tr><td>KodCode_V1_SFT_R1 <sup>11</sup></td><td>0.09%</td><td>Code</td></tr><tr><td>Chinese-Reasoning-Distil-Data <sup>12</sup></td><td>0.07%</td><td>General</td></tr><tr><td>medical-o1-reasoning-SFT <sup>13</sup></td><td>0.02%</td><td>Medical</td></tr><tr><td>Bespoke-Stratos-17k <sup>14</sup></td><td>0.01%</td><td>Math, Code, Science</td></tr><tr><td>Itemic Dense Caption Data</td><td>37.66%</td><td>Reco</td><td rowspan="3">37.35%</td><td rowspan="3">4B</td></tr><tr><td>Interleaved User Persona Grounding Data</td><td>0.21%</td><td>Reco</td></tr><tr><td>Sequential User Behavior Data</td><td>0.38%</td><td>Reco</td></tr></tbody></table>

Table 14: Data Composition and Token Budgets for Pre-training Stages. This table illustrates the training configurations for the Open and Pro model variants across different stages, specifying the parameter focus, data domain distribution, and allocated token budgets.

<table><tbody><tr><td>Model</td><td>Stage</td><td>Training</td><td>General-Domain</td><td>Reco-Domain</td><td>Token Budget</td></tr><tr><td rowspan="2">OneRec</td><td>stage1</td><td>Itemic-related Parameters</td><td rowspan="2">62.34%</td><td rowspan="2">37.66%</td><td>16B</td></tr><tr><td>stage2</td><td>Full-Parameter</td><td>33B</td></tr><tr><td rowspan="2">OneRec-Pro</td><td>stage1</td><td>Itemic-related Parameters</td><td>19%</td><td>81%</td><td>30B</td></tr><tr><td>stage2</td><td>Full-Parameter</td><td>53%</td><td>47%</td><td>130B</td></tr></tbody></table>

For the OneRec variant, Stage 2 uses all datasets listed in Table 13, totaling 33B tokens, while Stage 1 training data is sampled from this corpus. Table 14 details the comprehensive data composition and token budgets allocated for each pre-training stage, characterizing the strategic balance between general-domain and recommendation-domain knowledge.

In contrast, the OneRec-Pro model leverages a significantly expanded budget of 130B tokens to achieve superior performance, by increasing the sampling ratios of the general-purpose datasets from Table 13, integrating additional proprietary in-house datasets, and substantially expanding the scale of user-interaction data within the recommendation domain.

### B.5 Detailed Data Composition and Token Budgets for Multi-task SFT

Table 15: Data Mixture for Multi-task SFT. The table presents the distribution across reasoning and recommendation domains, showing the sampling weight of each dataset and the subtotal ratio for each category.

<table><tbody><tr><td>Dataset</td><td>Weight (%)</td><td>Category</td><td>Subtotal (%)</td></tr><tr><td>OpenMathReasoning <sup>15</sup></td><td>12.971%</td><td>Math</td><td rowspan="9">64.978%</td></tr><tr><td>R1-Distill-SFT <sup>16</sup></td><td>12.784%</td><td>General</td></tr><tr><td>Infinity_Instruct <sup>17</sup></td><td>11.359%</td><td>Instruction</td></tr><tr><td>OpenCoderReasoning <sup>18</sup></td><td>11.130%</td><td>Code</td></tr><tr><td>Chinese-Reasoning-Distil-Data <sup>19</sup></td><td>4.552%</td><td>General</td></tr><tr><td>Reasoning_Multi_subject_RLVR <sup>20</sup></td><td>4.376%</td><td>Multi-subject</td></tr><tr><td>Reasoning_KodCode_V1_SFT_R1 <sup>21</sup></td><td>4.167%</td><td>Code</td></tr><tr><td>DeepMath103K <sup>22</sup></td><td>2.362%</td><td>Math</td></tr><tr><td>medical-o1-reasoning-SFT <sup>23</sup></td><td>1.277%</td><td>Medical</td></tr><tr><td>Label prediction</td><td>7.800%</td><td>Reco</td><td rowspan="8">35.022%</td></tr><tr><td>SID to Caption generation</td><td>7.493%</td><td>Reco</td></tr><tr><td>Interactive recommendation</td><td>6.392%</td><td>Reco</td></tr><tr><td>Video recommendation</td><td>3.971%</td><td>Reco</td></tr><tr><td>Label conditional recommendation</td><td>3.575%</td><td>Reco</td></tr><tr><td>Product recommendation</td><td>2.878%</td><td>Reco</td></tr><tr><td>Ad recommendation</td><td>2.820%</td><td>Reco</td></tr><tr><td>Recommendation reasoning</td><td>0.094%</td><td>Reco</td></tr></tbody></table>

[^1]: R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In *The Twelfth International Conference on Learning Representations*, 2024.

[^2]: A. Basant, A. Khairnar, A. Paithankar, A. Khattar, A. Renduchintala, A. Malte, A. Bercovich, A. Hazare, A. Rico, A. Ficek, et al. Nvidia nemotron nano 2: An accurate and efficient hybrid mamba-transformer reasoning model. *arXiv preprint arXiv:2508.14444*, 2025.

[^3]: BespokeLabs. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation. https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation, 2025. Accessed: 2025-01-22.

[^4]: A. Z. Broder. On the resemblance and containment of documents. In *Proceedings of Compression and Complexity of Sequences*, pages 21–29. IEEE, 1997. URL [https://ieeexplore.ieee.org/document/666900](https://ieeexplore.ieee.org/document/666900).

[^5]: J. Chen, Z. Cai, K. Ji, X. Wang, W. Liu, R. Wang, J. Hou, and B. Wang. Huatuogpt-o1, towards medical complex reasoning with llms, 2024. URL [https://arxiv.org/abs/2412.18925](https://arxiv.org/abs/2412.18925).

[^6]: J. Deng, S. Wang, K. Cai, L. Ren, Q. Hu, W. Ding, Q. Luo, and G. Zhou. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. *arXiv preprint arXiv:2502.18965*, 2025.

[^7]: S. Geng, S. Liu, Z. Fu, Y. Ge, and Y. Zhang. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). *arXiv preprint arXiv:2203.13366*, 2022.

[^8]: Z. He, T. Liang, J. Xu, Q. Liu, X. Chen, Y. Wang, L. Song, D. Yu, Z. Liang, W. Wang, Z. Zhang, R. Wang, Z. Tu, H. Mi, and D. Yu. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. 2025. URL [https://arxiv.org/abs/2504.11456](https://arxiv.org/abs/2504.11456).

[^9]: B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk. Session-based recommendations with recurrent neural networks. In *ICLR*. ICLR, 2016.

[^10]: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

[^11]: W.-C. Kang and J. McAuley. Self-attentive sequential recommendation. In *ICDM*, pages 197–206. IEEE, 2018.

[^12]: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

[^13]: J. Li, L. Du, H. Zhao, B. wen Zhang, L. Wang, B. Gao, G. Liu, and Y. Lin. Infinity instruct: Scaling instruction selection and synthesis to enhance language models, 2025. URL [https://arxiv.org/abs/2506.11116](https://arxiv.org/abs/2506.11116).

[^14]: Z. Liu, S. Wang, X. Wang, R. Zhang, J. Deng, H. Bao, J. Zhang, W. Li, P. Zheng, X. Wu, et al. Onerec-think: In-text reasoning for generative recommendation. *arXiv preprint arXiv:2510.11639*, 2025.

[^15]: X. Luo, J. Cao, T. Sun, J. Yu, R. Huang, W. Yuan, H. Lin, Y. Zheng, S. Wang, Q. Hu, et al. Qarm: Quantitative alignment multi-modal recommendation at kuaishou. In *Proceedings of the 34th ACM International Conference on Information and Knowledge Management*, pages 5915–5922, 2025.

[^16]: S. T. Madhusudhan, S. Radhakrishna, J. Mehta, and T. Liang. Millions scale dataset distilled from r1-32b. https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT.

[^17]: J. McAuley, C. Targett, Q. Shi, and A. Van Den Hengel. Image-based recommendations on styles and substitutes. In *Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 43–52. ACM, 2015.

[^18]: F. Mentzer, D. Minnen, E. Agustsson, and M. Tschannen. Finite scalar quantization: Vq-vae made simple. *arXiv preprint arXiv:2309.15505*, 2023.

[^19]: I. Moshkov, D. Hanley, I. Sorokin, S. Toshniwal, C. Henkel, B. Schifferer, W. Du, and I. Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. *arXiv preprint arXiv:2504.16891*, 2025.

[^20]: Mxode. Chinese-reasoning-distil-data: A multidisciplinary chinese reasoning dataset distilled from frontier models. [https://huggingface.co/datasets/Mxode/Chinese-Reasoning-Distil-Data](https://huggingface.co/datasets/Mxode/Chinese-Reasoning-Distil-Data), 2024.

[^21]: A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.

[^22]: S. P. M. P. M. S. B. C. Rabeeh Karimi Mahabadi, Sanjeev Satheesh. Nemotron-cc-math: A 133 billion-token-scale high quality math pretraining dataset. 2025. URL [https://arxiv.org/abs/2508.15096](https://arxiv.org/abs/2508.15096).

[^23]: S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost, et al. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems*, 36:10299–10315, 2023.

[^24]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[^25]: G. Sheng, Y. Chi, K. Yao, Y. Zhao, Y. Xia, W. Li, H. Zhao, S. Wang, and Y. Wang. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv:2409.19256*, 2024.

[^26]: Y. Su, D. Yu, L. Song, J. Li, H. Mi, Z. Tu, M. Zhang, and D. Yu. Expanding rl with verifiable rewards across diverse domains. *arXiv preprint arXiv:2503.23829*, 2025.

[^27]: F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In *CIKM*, pages 1441–1450. ACM, 2019.

[^28]: J. Tang, S. Dai, T. Shi, J. Xu, X. Chen, W. Chen, J. Wu, and Y. Jiang. Think before recommend: Unleashing the latent reasoning power for sequential recommendation. *arXiv preprint arXiv:2503.22675*, 2025.

[^29]: G. A. Team. Reasoning-v1-20m: A massive synthetic dataset of 22.2 million general domain reasoning traces. [https://huggingface.co/datasets/glaiveai/reasoning-v1-20m](https://huggingface.co/datasets/glaiveai/reasoning-v1-20m), 2025.

[^30]: Thinking Machines. On-policy distillation. Thinking Machines Blog, Oct 2025. URL [https://thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation). [https://thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation), Last accessed on: 2025-12-20.

[^31]: W. Wang, H. Bao, X. Lin, J. Zhang, Y. Li, F. Feng, S.-K. Ng, and T.-S. Chua. Learnable item tokenization for generative recommendation. In *CIKM*. ACM, 2024.

[^32]: S. M. A. F. S. J. J. H. V. N. B. G. Wasi Uddin Ahmad, Sean Narenthiran. Opencodereasoning: Advancing data distillation for competitive coding. 2025. URL [https://arxiv.org/abs/2504.01943](https://arxiv.org/abs/2504.01943).

[^33]: L. Wu, Z. Zheng, Z. Qiu, H. Wang, H. Gu, T. Shen, C. Qin, C. Zhu, H. Zhu, Q. Liu, et al. A survey on large language models for recommendation. *arXiv preprint arXiv:2305.19860*, 2023.

[^34]: H. Xu, Q. Zhu, H. Deng, J. Li, L. Hou, Y. Wang, L. Shang, R. Xu, and F. Mi. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning. *arXiv preprint arXiv:2506.02208*, 2025a.

[^35]: Z. Xu, Y. Liu, Y. Yin, M. Zhou, and R. Poovendran. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding. 2025b. URL [https://arxiv.org/abs/2503.02951](https://arxiv.org/abs/2503.02951).

[^36]: A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.

[^37]: L. Yang, A. Subbiah, H. Patel, J. Y. Li, Y. Song, R. Mirghaderi, V. Aggarwal, and Q. Wang. Item-language model for conversational recommendation. *arXiv preprint arXiv:2406.02844*, 2024.

[^38]: J. Zhai, L. Liao, X. Liu, Y. Wang, R. Li, X. Cao, L. Gao, Z. Gong, F. Gu, M. He, et al. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152*, 2024.

[^39]: B. Zheng, Y. Hou, H. Lu, Y. Chen, W. X. Zhao, M. Chen, and J.-R. Wen. Adapting large language models by integrating collaborative semantics for recommendation. In *2024 IEEE 40th International Conference on Data Engineering (ICDE)*, pages 1435–1448. IEEE, 2024.

[^40]: G. Zhou, J. Deng, J. Zhang, K. Cai, L. Ren, Q. Luo, Q. Wang, Q. Hu, R. Huang, S. Wang, et al. Onerec technical report. *arXiv preprint arXiv:2506.13695*, 2025a.

[^41]: G. Zhou, H. Hu, H. Cheng, H. Wang, J. Deng, J. Zhang, K. Cai, L. Ren, L. Ren, L. Yu, et al. Onerec-v2 technical report. *arXiv preprint arXiv:2508.20900*, 2025b.