---
title: "OneRec-Think: In-Text Reasoning for Generative Recommendation"
source: "https://arxiv.org/html/2510.11639v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/05_%E5%85%B6%E4%BB%96/OneRec-Think%2C%20In-Text%20Reasoning%20for%20Generative%20Recommendation%2C%20Zhanyu%20Liu%20et%20al.%2C%202025.no_watermark.zh-CN.dual.pdf"
---
Zhanyu Liu <sup>*</sup>, Shiyao Wang <sup>*</sup>, Xingmei Wang, Rongzhou Zhang, Jiaxin Deng,  
Honghui Bao, Jinghao Zhang, Wuchao Li, Pengfei Zheng, Xiangyu Wu,  
Yifei Hu, Qigen Hu, Xinchen Luo, Lejian Ren, Zixing Zhang,  
Qianqian Wang, Kuo Cai, Yunfan Wu, Hongtao Cheng, Zexuan Cheng,  
Lu Ren, Huanjie Wang, Yi Su, Ruiming Tang, Kun Gai, Guorui Zhou <sup>†</sup>  
Kuaishou Inc., Beijing, China  
{liuzhanyu,wangshiyao08,zhouguorui}@kuaishou.com  
\*: Equal contributions.†: Corresponding Author.

###### Abstract

The powerful generative capacity of Large Language Models (LLMs) has instigated a paradigm shift in recommendation. However, existing generative models (e.g., OneRec) operate as implicit predictors, critically lacking the capacity for explicit and controllable reasoning—a key advantage of LLMs. To bridge this gap, we propose OneRec-Think, a unified framework that seamlessly integrates dialogue, reasoning, and personalized recommendation. OneRec-Think incorporates: (1) Itemic Alignment: cross-modal Item-Textual Alignment for semantic grounding; (2) Reasoning Activation: Reasoning Scaffolding to activate LLM reasoning within the recommendation context; and (3) Reasoning Enhancement, where we design a recommendation-specific reward function that accounts for the multi-validity nature of user preferences. Experiments across public benchmarks show state-of-the-art performance. Moreover, our proposed "Think-Ahead" architecture enables effective industrial deployment on Kuaishou, achieving a 0.159% gain in APP Stay Time and validating the practical efficacy of the model’s explicit reasoning capability.

OneRec-Think: In-Text Reasoning for Generative Recommendation

Zhanyu Liu <sup>*</sup>, Shiyao Wang <sup>*</sup>, Xingmei Wang, Rongzhou Zhang, Jiaxin Deng, Honghui Bao, Jinghao Zhang, Wuchao Li, Pengfei Zheng, Xiangyu Wu, Yifei Hu, Qigen Hu, Xinchen Luo, Lejian Ren, Zixing Zhang, Qianqian Wang, Kuo Cai, Yunfan Wu, Hongtao Cheng, Zexuan Cheng, Lu Ren, Huanjie Wang, Yi Su, Ruiming Tang, Kun Gai, Guorui Zhou <sup>†</sup> Kuaishou Inc., Beijing, China {liuzhanyu,wangshiyao08,zhouguorui}@kuaishou.com

<sup>†</sup> <sup>†</sup>

## 1 Introduction

![Refer to caption](https://arxiv.org/html/2510.11639v2/x1.png)

Figure 1: Examples of OneRec-Think’s Unified Dialogue, Reasoning and Recommendation Framework.

The rapid advancement of Large Language Models (LLMs) has fundamentally reshaped recommender systems, ushering in the generative retrieval paradigm(GR) [^15] [^37] [^3] [^13] [^25]. This approach represents a profound shift from traditional query-candidate matching, utilizing Transformer-based sequence-to-sequence models to autoregressively decode the identifiers of target candidates. Capitalizing on this, a major research frontier is the development of end-to-end generative frameworks, including OneRec, OneLoc, OneSug, and OneSearch [^4] [^39] [^40] [^29] [^7] [^2]. These unified models replace the traditional multi-stage recommendation funnel (involving separate retrieval and ranking stages), enabling holistic optimization towards the final objective and concentrating computational resources for better industrial scaling and performance.

While these models successfully harness the LLMs’ capacity for output generation, they fundamentally lack the explicit, verifiable reasoning pathways that define modern LLM breakthroughs, such as text-based Chain-of-Thought (CoT) [^16] [^38] [^27]. To bridge this critical gap, we propose OneRec-Think, a novel framework that integrates dialogue, reasoning, and personalized generative recommendations within a single, unified model. It is capable of generating high-quality, interpretable textual reasoning paths, significantly enhancing both recommendation accuracy and user trustworthiness. The model’s inherent dialogic nature further enables dynamic tailoring of suggestions to specific user constraints (as shown in Fig. 1). Our approach is realized through a three-stage framework: (1) Itemic Alignment, which maps item semantics into the LLM’s textual embedding space, establishing a unified representational continuum that unlocks the model’s capacity for reasoning. (2) Reasoning Activation, which aims to induce the LLM’s inherent reasoning ability directly within the context of recommender systems; (3) Reasoning Enhancement, which utilizes a recommendation-specific reward function that captures the multi-validity (i.e., multiple valid choices) nature of user preferences. Furthermore, we introduce the OneRec-Think Inference Architecture to ensure efficient deployment and real-time responsiveness in industrial-scale serving scenarios. Our contributions are summarized as follows:

- We introduce a unified framework that bridges the semantic gap between discrete recommendation items and continuous reasoning spaces, enabling seamless integration of personalized recommendation within LLMs’ natural language understanding.
- We design a novel reasoning paradigm that orchestrates multi-step deliberation with recommendation optimization, achieving interpretable and accuracy-aware personalized recommendation through synergistic training.
- The proposed approach achieves state-of-the-art results on multiple public benchmarks, while our deployment-friendly "Think-Ahead" architecture enables significant industrial impact with a 0.159% gain in APP Stay Time.

## 2 Related Work

### 2.1 Reasoning in Large Language Models

Large language models achieve complex reasoning through various prompting techniques, with CoT prompting [^28] being the foundational approach that decomposes problems into intermediate reasoning steps. This has inspired numerous extensions including zero-shot CoT [^12], self-consistency decoding [^26], and tree-of-thoughts [^31]. These techniques enable test-time scaling where additional computational budget during inference improves performance [^20]. Recent work has shifted focus from prompting to post-training enhancement of reasoning capabilities by using techniques such as Reinforcement Learning. Models including DeepSeek-R1 [^6] and Seed-1.5 [^17] optimize reasoning behaviors via techniques such as GRPO [^18], DAPO [^32], and VAPO [^33], which demonstrates promising advancements in this direction.

### 2.2 Reasoning-Based Recommendation

Although generative recommendation models such as TIGER [^16], HSTU [^34], and OneRec [^39] have demonstrated effectiveness, they inherently lack reasoning capabilities. Recently, the Reasoning-based recommendation systems aim to perform multi-step deduction for more accurate and interpretable recommendations. Existing approaches fall into two categories: explicit reasoning methods generate human-readable rationales but are confined to discriminative tasks [^23] [^1] [^5] [^11], while implicit reasoning methods [^35] [^22] perform latent reasoning without textual interpretability. Our work introduces explicit reasoning into generative recommendation, bridging this gap to enable both interpretable rationales and scalable item generation.

![Refer to caption](https://arxiv.org/html/2510.11639v2/x2.png)

Figure 2: The framework of the OneRec-Think. In the first stage, we achieve item-level semantic alignment through multi-task pre-training. In the second stage, we activate explicit reasoning by prompting the model to generate preference rationales. In the third stage, we refine the reasoning paths through RL based on a reward tailored for recommendations.

## 3 Preliminary

##### Itemic Token

An itemic token is a discrete, semantic-rich representation unit for an item, analogous to a word token in natural language. Following OneRec [^39] [^16], we map each item $v$ to a sequence of such tokens $\bm{s}_{v}=(s_{v}^{1},\dots,s_{v}^{L})$, which are generated from the item’s multi-modal and collaborative content.

##### Problem Definition

Let $\mathcal{U}$ and $\mathcal{V}$ denote the sets of users and items, respectively. Each user $u\in\mathcal{U}$ has a chronological interaction history $V_{u}=(v^{u}_{1},v^{u}_{2},\dots,v^{u}_{n_{u}})$ of length $n_{u}$. For brevity, we omit the user index $u$. By using the itemic tokens, the user’s interaction history is thus represented by the sequence $S_{u}=(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}})$.

Conventional generative recommenders [^16] [^39] [^40] [^38] define their task with a generation target of the next itemic tokens as:

$$
\bm{s}_{v_{n+1}}\sim P(\cdot|\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}};\theta)
$$

In contrast, we reformulate this task to unify reasoning and recommendation in a single autoregressive pass. Conditioned on a prompted user history, we generate tokens sequentially, beginning with a reasoning sequence $\bm{\tau}=(r_{1},\dots,r_{M})$ and concluding with the next itemic tokens $\bm{s}_{v_{n+1}}$. This end-to-end process is captured by:

$$
\displaystyle\bm{\tau}
$$
 
$$
\displaystyle\sim P\left(\cdot\mid\mathcal{P}(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}});\theta\right)
$$
 
$$
\displaystyle\bm{s}_{v_{n+1}}
$$
 
$$
\displaystyle\sim P\left(\cdot\mid\mathcal{P}(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}}),\bm{\tau};\theta\right)
$$

where $\mathcal{P}(\cdot)$ means a valid prompt constructed for the recommendation.

## 4 Methodolody

We now present OneRec-Think, a scalable framework for end-to-end generative reasoning recommendation. Our approach comprises three core components: an Itemic Alignment stage, a Reasoning Activation stage, and a "Think Ahead" architecture for industrial deployment. The illustration of OneRec-Think is shown in Fig. 2.

### 4.1 Itemic Alignment through Multi-Task Pre-training

To align recommender knowledge with the LLM’s linguistic space, we design a Multi-Task Pre-training strategy enabling seamless processing of natural language and itemic tokens via four complementary tasks under Next Token Prediction.

##### Interleaved User Persona Grounding

Unlike prior work that uses either pure textual data or isolated item sequences, this task interleaves the itemic tokens and text tokens of User Persona. It includes serialized static attributes, active search behaviors, interactive sequences, and summarized user interests. This composition creates rich, dual-modality training instances where itemic tokens are grounded in their semantic context.

##### Sequential Preference Modeling

As the core recommendation task, this task constructs data that teaches the model to predict subsequent item interactions from chronological user histories.

##### Itemic Dense Captioning

This task requires the model to decode an item’s descriptive content from its itemic tokens. By learning to generate detailed textual descriptions, the model establishes a fundamental understanding of the semantic characteristics represented by item combinations.

##### General Language Modeling

This task continues pre-training the model on general text corpora, preserving the model’s fundamental language capabilities during applying the model to recommendation scenarios.

To enable effective knowledge integration while preserving the linguistic capabilities of the model, we implement a two-substage training strategy to ensure stable alignment. The Token Warm-up substage exclusively trains itemic token embeddings on the Interleaved User Persona Grounding task while keeping the base LLM frozen. The subsequent Multi-Task Integration substage jointly optimizes all parameters on the combined task using a designed ratio (see Appendix A.3 for details).

### 4.2 Reasoning Activation

Despite robust itemic alignment, direct application to industrial recommendation scenarios often fails to yield effective CoT reasoning due to the noisy and lengthy nature of real-world user behavior sequences. To address this, we propose a supervised fine-tuning framework that first extracts coherent reasoning trajectories from pruned user contexts, then leverages these trajectories to guide rationale generation over raw behavioral data, enabling effective contextual distillation for noisy industrial settings (as shown in Fig. 2(b)).

Bootstrapping with Pruned Contexts: To bootstrap reasoning capabilities, we first construct easy-to-learn instances where logical relationships are preserved despite sequence pruning. For each user, we select the target item $\bm{s}_{v_{n+1}}$ and form a context-target pair $<(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}}),\bm{s}_{v_{n+1}}>$. We then retrieve the top- $k$ most relevant historical items using a similarity function $g(\cdot,\cdot)$:

$$
g((\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}}),\bm{s}_{v_{n+1}})=(\bm{s}_{w_{1}},\dots,\bm{s}_{w_{k}}).
$$

Using these relevant items, we query our pre-aligned model to generate a rationale $\bm{\tau}$ explaining the target interaction:

$$
\displaystyle\bm{\tau}
$$
 
$$
\displaystyle\sim P\left(\cdot\mid\mathcal{P}_{r}((\bm{s}_{w_{1}},\dots,\bm{s}_{w_{k}}),\bm{s}_{v_{n+1}});\theta\right)
$$

where $\mathcal{P}_{r}(a,b)$ means constructs a prompt to query the rationale why a user who interacts with item sequence $a$ would interact with item $b$. This process yields high-quality rationales that are both logically sound and target-aligned, providing ideal training signals for reasoning induction.

Learning to Reason from Noisy Sequences: The distilled rationales serve as supervision for learning to reason from raw sequences. The training objective minimizes the negative log-likelihood of generating both the rationale and target item:

$$
\displaystyle\mathcal{L}_{\text{RA}}=-\Biggl(\sum_{i=1}^{M}\log P(r_{i}|\mathcal{P}(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}}),r_{<i};\theta)
$$
 
$$
\displaystyle+\sum_{j=1}^{L}\log P(s_{v_{n+1}}^{j}|\mathcal{P}(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}}),\bm{\tau},{s}_{v_{n+1}}^{<j};\theta)\Biggr),
$$

where $\bm{\tau}=\{r_{1},\dots,r_{M}\}$ represents the rationale tokens and $\bm{s}_{v_{n+1}}=\{s_{v_{n+1}}^{1},\dots,s_{v_{n+1}}^{L}\}$ denotes the target item tokens. By optimizing $\mathcal{L}_{\text{RA}}$, the model learns to internally distill relevant context from noisy sequences and generate coherent rationales that bridge user history to target interactions, significantly enhancing its CoT capabilities in challenging recommendation scenarios.

### 4.3 Reasoning Enhancement

Building upon the CoT capabilities from Reasoning Activation, we address the challenge of ensuring consistently high-quality reasoning through Reinforcement Learning. This stage refines the recommendation accuracy using a novel reward mechanism tailored for generative recommendation.

Beam Candidate Reward Maximization: Standard verifiable pass rewards face significant sparsity challenges in recommendation scenarios, as most reasoning rollouts fail to hit the target item and consequently yield identical zero rewards—thereby neutralizing group advantages in algorithms such as GRPO [^18]. To overcome this, we introduce the Rollout-Beam reward that evaluates reasoning capability by the model’s best achievable performance within a constrained beam. Our approach employs beam search with width $K$ to explore multiple generation candidates after reasoning trajectory generation:

$$
\mathcal{R}_{\text{Rollout-Beam}}=\max_{\hat{s}_{v_{n+1}}\in\mathcal{B}}\sum_{l=1}^{L}\mathbb{I}(\hat{s}_{v_{n+1}}^{l}=s_{v_{n+1}}^{l}),
$$

where the beam search result set is defined as:

$$
\begin{split}\mathcal{B}&=\bigl\{\bigl(\hat{s}_{v_{n+1}}^{1,(j)},\cdots,\hat{s}_{v_{n+1}}^{L,(j)}\bigr)_{j=1}^{K}\bigr\},\\
&=\text{BeamSearch}\Bigl(P(\bm{s}_{v_{n+1}}\mid\bm{H},\bm{\tau};\theta\bigr),K\Bigr)\end{split}
$$

which contains the items with the top- $K$ probabilty in the beam search within the distribution $P\left(\bm{s}_{v_{n+1}}\mid\bm{H},\bm{\tau};\theta\right)$. $\bm{H}=\mathcal{P}(\bm{s}_{v_{1}},\dots,\bm{s}_{v_{n}})$ is a valid prompt of history sequence. $\text{BeamSearch}(P,K)$ means the top- $K$ result of beam search within distribution $P$. Subsequently, we optimize the model using GRPO [^18] based on $\mathcal{R}_{\text{Rollout-Beam}}$, which effectively leverages the enriched reward signals from the multi-validity nature of user preferences.

Overall, this design establishes training-inference consistency by aligning reward computation with beam search-based inference, providing denser learning signals through multi-path evaluation.

Table 1: Overall performance comparison between the baselines and OneRec-Think on three datasets. The bold results highlight the best results, while the second-best ones are underlined.

<table><thead><tr><th>Dataset</th><th>Method</th><th>BERT4Rec</th><th>HGN</th><th>GRU4Rec</th><th>SASRec</th><th>TIGER</th><th>HSTU</th><th>ReaRec</th><th>OneRec-Think</th></tr></thead><tbody><tr><th rowspan="4">Beauty</th><td>R@5</td><td>0.0232</td><td>0.0319</td><td>0.0395</td><td>0.0402</td><td>0.0405</td><td>0.0424</td><td>0.0450</td><td>0.0563</td></tr><tr><td>R@10</td><td>0.0396</td><td>0.0536</td><td>0.0584</td><td>0.0607</td><td>0.0623</td><td>0.0652</td><td>0.0704</td><td>0.0791</td></tr><tr><td>N@5</td><td>0.0146</td><td>0.0196</td><td>0.0265</td><td>0.0254</td><td>0.0267</td><td>0.0280</td><td>0.0262</td><td>0.0398</td></tr><tr><td>N@10</td><td>0.0199</td><td>0.0266</td><td>0.0326</td><td>0.0320</td><td>0.0337</td><td>0.0353</td><td>0.0344</td><td>0.0471</td></tr><tr><th rowspan="4">Sports</th><td>R@5</td><td>0.0102</td><td>0.0183</td><td>0.0190</td><td>0.0199</td><td>0.0215</td><td>0.0268</td><td>0.0214</td><td>0.0288</td></tr><tr><td>R@10</td><td>0.0175</td><td>0.0313</td><td>0.0312</td><td>0.0301</td><td>0.0347</td><td>0.0343</td><td>0.0332</td><td>0.0412</td></tr><tr><td>N@5</td><td>0.0065</td><td>0.0109</td><td>0.0122</td><td>0.0106</td><td>0.0137</td><td>0.0173</td><td>0.0116</td><td>0.0199</td></tr><tr><td>N@10</td><td>0.0088</td><td>0.0150</td><td>0.0161</td><td>0.0141</td><td>0.0179</td><td>0.0226</td><td>0.0154</td><td>0.0239</td></tr><tr><th rowspan="4">Toys</th><td>R@5</td><td>0.0215</td><td>0.0326</td><td>0.0330</td><td>0.0448</td><td>0.0337</td><td>0.0366</td><td>0.0523</td><td>0.0579</td></tr><tr><td>R@10</td><td>0.0332</td><td>0.0517</td><td>0.0490</td><td>0.0626</td><td>0.0547</td><td>0.0566</td><td>0.0764</td><td>0.0797</td></tr><tr><td>N@5</td><td>0.0131</td><td>0.0192</td><td>0.0228</td><td>0.0300</td><td>0.0209</td><td>0.0245</td><td>0.0298</td><td>0.0412</td></tr><tr><td>N@10</td><td>0.0168</td><td>0.0254</td><td>0.0279</td><td>0.0358</td><td>0.0276</td><td>0.0309</td><td>0.0376</td><td>0.0482</td></tr></tbody></table>

### 4.4 Industrial Deployment: A "Think-Ahead" Architecture

The deployment of OneRec-Think in industrial recommendation systems presents a fundamental challenge: reconciling the computational demands of multi-step reasoning with the stringent latency requirements of real-time user interactions.

To address this critical bottleneck, we introduce a novel *Think-Ahead* Inference Architecture. Our solution strategically decouples the model’s inference into two stages: In the first stage, the computationally intensive reasoning path and the initial item-tokens (e.g., the first two itemic tokens) are generated offline by the full OneRec-Think model. These initial tokens are designed to capture the user’s broad intent or general preference context. Subsequently, the second stage then employs a real-time updated OneRec model following [^39] for online finalization. It utilizes the pre-generated item-tokens as a constrained prefix to rapidly produce the final itemic token. This design ensures real-time responsiveness and achieves production-grade performance by leveraging current contextual data. The details of this architecture are in Appendix A.3.4.

## 5 Experiments

![Refer to caption](https://arxiv.org/html/2510.11639v2/x3.png)

Figure 3: Demonstration of context-aware recommendation adaptation: our model dynamically shifts recommendations to relaxing content based on the user’s command.

![Refer to caption](https://arxiv.org/html/2510.11639v2/x4.png)

Figure 4: Demonstration of fine-grained interest reasoning, which shows the end-to-end process from user behavior analysis to interpretable recommendations.

### 5.1 Experimental Settings

Datasets and Baselines.

We use three real-world recommendation datasets from the popular Amazon review benchmark <sup>1</sup>: Beauty, Toys, and Sports. We compare OneRec-Think against two groups of competitive baselines: (1) Classic sequential methods like BERT4Rec [^21], HGN [^14], GRU4Rec [^8], and SASRec [^10]; and (2) Generative Recommender Models, such as TIGER [^16], HSTU [^34], and ReaRec [^22]. Top-K Recall (R@K) and NDCG (N@K) with K=5 and 10 are used as metrics, following [^16]. Implementation <sup>2</sup> details are in Appendix A.1.

### 5.2 Overall Performance

The results are shown in Table 1. We could observe that models leveraging powerful reasoning-based architectures (ReaRec and our OneRec-Think) consistently outperform both traditional sequential recommenders and generative recommenders. This robust trend confirms that effective sequential prediction necessitates robust inference and contextual reasoning capabilities. Furthermore, OneRec-Think further gets the best performance across all benchmarks. This significant superiority is directly attributed to explicit, text-based reasoning ability for item generation, in contrast to the more implicit, purely learned generation mechanisms in prior works.

### 5.3 Ablation Study

Table 2: Ablation Study of different variants of OneRec-Think on Beauty dataset.

| Training Method | R@5 | R@10 | N@5 | N@10 |
| --- | --- | --- | --- | --- |
| Base | 0.0460 | 0.0654 | 0.0314 | 0.0377 |
| Base+IA | 0.0532 | 0.0735 | 0.0342 | 0.0402 |
| Base+IA+R | 0.0563 | 0.0791 | 0.0398 | 0.0471 |

We conduct an ablation study on the Beauty dataset, comparing three configurations: the Base model tuned by the raw itemic token sequence, the Base+IA model enhanced with Itemic Alignment, and the full Base+IA+R model incorporating our enhanced reasoning mechanism, as shown in Table 2. It demonstrates that each component is indispensable: Itemic Alignment provides a foundational boost by creating coherent semantic representations of itemic tokens, while the reasoning mechanism yields a further significant gain, confirming that both components synergistically address core challenges in sequential recommendation.

### 5.4 Industrial Experiments

#### 5.4.1 Training Settings

We adopt Qwen-8B [^30] as our backbone model, initializing its parameters from the publicly available pre-trained weights. The model’s vocabulary is extended with 24,576 new tokens representing the three-level hierarchical itemic tokens (8,192 tokens per level), plus two special boundary tokens, <|item\_begin|> and <|item\_end|>. For our production environment, we implement a daily incremental training pipeline. The model is updated each day on a cluster of 80 flagship GPUs, processing approximately 20B tokens per day to stay current with newly generated user interaction data. The details are shown in Appendix A.3.

Table 3: The relative improvement of our online A/B testing on a short-video recommendation scenario.

| Online Metrics | OneRec-Think |
| --- | --- |
| App Stay Time | +0.159% |
| Watch Time | +0.169% |
| Video View | +0.150% |
| Follow | +0.431% |
| Forward | +0.758% |
| Like | +0.019% |
| Collect | +0.098% |

#### 5.4.2 Results

##### Online A/B Result.

We deploy OneRec-Think on Kuaishou, a short-video platform with hundreds of millions of daily active users. Using a 1.29% traffic experimental group, we compare OneRec-Think with our online model for one week and report the result in Table 3, where the primary metric is APP Stay Time (reflecting total user engagement time). The primary metric, App Stay Time, shows significant gains that increase by 0.159%. Note that in industrial recommendation systems, 0.1% improvements are considered substantial. Furthermore, interactive metrics such as Video View and Forward exhibit positive trends, indicating enhanced user engagement. We conducted multiple experiments at different times and consistently observed significant improvements in stay time and related interaction metrics.

Table 4: The Bertscore for User Understanding and Short Video Understanding Benchmark.

<table><thead><tr><th rowspan="2">Benchmark</th><th rowspan="2">Qwen3</th><th>Qwen3</th><th>Qwen3</th></tr><tr><th>+ TW</th><th>+ TW + MI</th></tr></thead><tbody><tr><td>User</td><td>0.6588</td><td>0.6492</td><td>0.7053</td></tr><tr><td>Short Video</td><td>0.6031</td><td>0.6443</td><td>0.7300</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2510.11639v2/x5.png)

Figure 5: The model’s reasoning process evolves from broad interest matching (left) to fine-grained theme specification (middle), with recommendations (right) showing semantic consistency with each reasoning step.

##### Ablation on Itemic Alignment on industrial benchmark

We evaluate Token Warm-up (TW) and Multi-Task Integration (MI) of the Itemic Alignment stage on our industrial User and Short Video Understanding benchmarks using BertScore [^36] (details in Appendix A.3.2). Results in Table 4 reveal distinct roles for each component. On the text-heavy User Understanding task, TW provides limited gain over the strong Base model since the LLM can effectively process the abundant textual information directly, while MI delivers a substantial boost by translating aligned representations into actionable insights. In contrast, in the pure itemic token Short Video Understanding task, it shows progressive gains from both TW and MI, confirming their necessity for interpreting non-textual item information. These results validate that these two substages both contribute to the final performance of Itemic Alignment.

![Refer to caption](https://arxiv.org/html/2510.11639v2/x6.png)

Figure 6: Demonstration of itemic-textual interleaved reasoning.

### 5.5 Case Study

Our case studies demonstrate the model’s sophisticated reasoning capabilities across different scenarios. In conversational settings (Fig. 3), when the user expresses negative emotions, the model detects this affective signal and strategically shifts recommendations from general interests toward relaxing and positive content, demonstrating its ability to actively optimize the viewing experience through the interaction with the user. In reasoning-based short-video recommendation (Fig. 4), the model generates diverse reasoning paths that capture fine-grained user preferences, such as specific gameplay mechanics and narrative patterns, enabling more precise recommendations beyond coarse topic matching. Furthermore, our consistency analysis (Fig. 5) reveals strong alignment between reasoning textx and recommended items when applying beam search at intermediate reasoning steps, confirming that the reasoning process genuinely guides recommendation generation rather than serving as post-hoc justification. Notably, our model achieves itemic-textual interleaved reasoning paths (Fig. 6). Through precise content anchoring by itemic tokens and causal articulation by textual tokens, the interleaved reasoning delivers enhanced recommendation accuracy and transparent explanations beyond isolated modality approaches. These results collectively validate our model’s capacity for authentic, multi-faceted reasoning by demonstrating its ability to adapt to real-time interactions, capture fine-grained preferences, and maintain semantic consistency across diverse recommendation scenarios.

## 6 Conclusion

We present OneRec-Think, a novel framework that bridges reasoning capabilities with generative recommendation through three key innovations: hierarchical itemic token alignment, reasoning activation via CoT supervised fine-tuning, and reinforcement-based reasoning refinement. Our method fundamentally transforms recommendation systems from mere item predictors into reasoning-aware models that generate interpretable rationales alongside high-quality recommendations. Extensive experiments demonstrate that OneRec-Think not only achieves state-of-the-art performance across multiple benchmarks, but also translates to concrete industrial impact with a 0.15% gain in primary metrics like APP Stay Time. Future work will focus on exploring user long-sequence modeling and dense RL reward for finer-grained preference modeling, further bridging LLM-based reasoning with industrial recommendation systems.

## Limitations

Despite promising empirical results, current public datasets exhibit quality constraints through their limited behavior sequence lengths and restricted item spaces. These limitations hinder our Reasoning Activation and Reasoning Enhancement modules from acquiring high-quality reasoning capabilities comparable to those learned from industrial-scale data. Consequently, we simplify and adapt our approach to achieve a stable yet simplified reasoning capacity, which remains robust within the public datasets. To address these issues, we are actively constructing a large-scale benchmark with extended behavioral trajectories and diversified item catalogs that will enable more comprehensive evaluation of reasoning capabilities for reasoning-based recommendation models.

## Ethics Statement

In this work, we have conducted experiments for two settings: one for open-source benchmark datasets and one for the industrial scenario. For the experiments for open-source benchmark datasets, all datasets are publicly available from previous works or public APIs while maintaining anonymity. For the industrial scenario, we utilize user interaction data collected from our platform to train the recommendation model. All data collection and usage strictly comply with our platform’s privacy policy and terms of service, to which users have provided explicit consent. Importantly, our training process operates solely on aggregated behavioral sequences, textual content, and user base information without accessing or processing any personally identifiable information.

## References

## Appendix A Appendix

### A.1 Experiment settings

##### Details of Baselines

We compare OneRec-Think with competitive baselines within two groups of work, traditional recommender models and generative recommender models: 1) BERT4Rec [^21] leverages BERT’s pre-trained language representations to capture semantic user-item relationships. 2) HGN [^14] utilizes graph neural networks to learn user and item representations for predicting user-item interactions. 3) GRU4Rec [^8] is a lightweight graph convolutional network model focusing on high-order connections between users and items. 4) SASRec [^10] employs self-attention mechanisms to capture long-term dependencies in user interaction history. 5) TIGER [^16] introduces codebook-based identifiers via RQ-VAE, which quantizes semantic information into code sequences for LLM-based generative recommendation. 6) HSTU [^34] reformulates recommendation problems as sequential transduction tasks within a generative modeling framework and proposes a new architecture for streaming data. 7) ReaRec [^22] enhances user representations through implicit multi-step reasoning within an inference-time computing framework for recommendation. Evaluation Metrics. We use two metrics: top- $K$ Recall (R@ $K$) and NDCG (N@ $K$) with $K$ = 5 and 10, following [^16].

![Refer to caption](https://arxiv.org/html/2510.11639v2/x7.png)

Figure 7: A reasoning example for a short-video recommendation scenario.

##### Details of experiments on open-source datasets

We adopt Qwen3-1.7B [^30] as our backbone model. The model’s vocabulary is extended with 1,024 new tokens representing the four-level hierarchical semantic IDs (256 tokens per level), in addition to two special boundary tokens, <|item\_begin|> and <|item\_end|>. All models were trained on a server equipped with flagship GPUs. To generate the top-K recommendations during evaluation, we employ a beam search strategy with a beam width of 10. Given the inherent challenge of deriving a robust reasoning path $\bm{\tau}$ from the short and sparse item sequences typical of public benchmarks, we strategically employ manually constructed category-based CoT as the pruned content for Reasoning Activation to ensure stable semantic guidance. For the training data, we adopt the pre-processing techniques from previous work [^16] [^24], discarding sparse users and items with interactions less than 5. We consider the sequential recommendation setting and use leave-one-out strategy [^16] [^38] to split datasets. For training, we follow [^10] to restrict the number of items in a user’s history to 50.

### A.2 Model Demonstration

##### Reasoning Cases for Short-video Recommendation

In this part, we show some cases of short video recommendations of another user. Figure 7 presents two reasoning paths generated by our model, demonstrating its capacity for multi-step interest inference and underlying need identification. In the first case, the model connects the user’s gaming preferences with hardware comparison behaviors to deduce an unstated need for performance optimization, ultimately recommending monitor analysis videos. The second case reveals deeper psychological needs by associating sports/military viewing history with adolescent rebellion searches, identifying parenting challenges as the core concern. Both examples showcase our model’s ability to transcend superficial topic matching and perform causal reasoning about user motivations, enabling recommendations that address both expressed interests and latent needs.

##### Semantic Comprehension Validation

After semantic alignment, the model acquires the capability to comprehend and articulate the semantic meaning of item tokens through natural language. To validate this emergent ability, we evaluate the capacity of the model to generate descriptive captions for itemic tokens without explicit training on this task. When prompted to explain what an itemic token represents, the model leverages the learned semantic correspondences to produce coherent textual descriptions that accurately capture the characteristics of the items. This demonstrates that the alignment process successfully establishes genuine semantic understanding rather than superficial pattern matching, as the model can now bidirectionally translate between discrete itemic tokens and their rich natural language semantics. The following cases in Beauty datset showcase the model’s caption generation capability for item semantic tokens:

<svg height="3215.38" id="A1.SS2.SSS0.Px2.p2.pic1" overflow="visible" version="1.1" viewBox="0 0 600 3215.38" width="600"><g stroke="#000000" style="--ltx-stroke-color:#000000;" transform="translate(0,3215.38) matrix(1 0 0 -1 0 0)"><g fill="#004D4D" fill-opacity="1.0" stroke-width="0.5pt" style="--ltx-fill-color:#004D4D;"><path d="M 0 5.91 L 0 3196.37 C 0 3199.63 2.64 3202.28 5.91 3202.28 L 594.09 3202.28 C 597.36 3202.28 600 3199.63 600 3196.37 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#000000" stroke-width="0.4pt" style="--ltx-fill-color:#000000;"><g fill="#F2F9F9" fill-opacity="1.0" style="--ltx-fill-color:#F2F9F9;"><path d="M 1.97 5.91 L 1.97 3196.37 C 1.97 3198.55 3.73 3200.31 5.91 3200.31 L 594.09 3200.31 C 596.27 3200.31 598.03 3198.55 598.03 3196.37 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 13.84 3192.43)"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)"><g fill="#505050" fill-opacity="1.0" style="--ltx-fill-color:#505050;"><path d="M 0 0.69 L 0 22.25 C 0 22.63 0.31 22.94 0.69 22.94 L 228.84 22.94 C 229.23 22.94 229.54 22.63 229.54 22.25 L 229.54 0.69 C 229.54 0.31 229.23 0 228.84 0 L 0.69 0 C 0.31 0 0 0.31 0 0.69 Z" style="stroke:none"></path></g><g fill="#505050" fill-opacity="1.0" style="--ltx-fill-color:#505050;"><path d="M 0.69 0.69 L 0.69 22.25 L 228.84 22.25 L 228.84 0.69 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.5 8.01)"><foreignObject color="#000000" height="12.3" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :13.54em;--fo_height:0.63em;--fo_depth :0.18em;" transform="matrix(1 0 0 -1 0 9.61)" width="206.07"><span style="--ltx-fg-color:#FFFFFF;">Case: Itemic Token Explaining 1</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 22.51 5.91)"><foreignObject color="#000000" height="3190.47" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :40.11em;--fo_height:230.57em;--fo_depth :0em;" transform="matrix(1 0 0 -1 0 3190.47)" width="554.98"><span style="width:40.11em;">Question + Answer <span style="--ltx-bg-color:#F2F9F9;"><a download="" href="data:text/plain;base64,PHxpbV9zdGFydHw+c3lzdGVtCllvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudC48fGltX2VuZHw+Cjx8aW1fc3RhcnR8PnVzZXIKQ2FuIHlvdSBleHBsYWluIHRoZSBtZWFuaW5nIG9mIDx8aXRlbV9iZWdpbnw+PGl0ZW1fYV8yMTk+PGl0ZW1fYl8xODk+PGl0ZW1fY180OT48aXRlbV9kXzIyNz48fGl0ZW1fZW5kfD4/PHxpbV9lbmR8Pgo8fGltX3N0YXJ0fD5hc3Npc3RhbnQKYFx0ZXh0YmZ7SXRzIHRpdGxlIGlzIE1hbnVrYSBIb25leSBhbmQgTWFudWthIE9pbCBTa2luIENyZWFtLCBpdHMgY2F0ZWdvcnkgaXMgQmVhdXR5ID4gU2tpbiBDYXJlID4gRmFjZSA+IENsZWFuc2Vycy59YA==">⬇</a> <span id="lstnumberx1"><span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">im_start</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> <span style="--ltx-fg-color:#323232;">system</span> </span><span id="lstnumberx2"><span style="--ltx-fg-color:#323232;">You</span> <span style="--ltx-fg-color:#323232;">are</span> <span style="--ltx-fg-color:#323232;">a</span> <span style="--ltx-fg-color:#323232;">helpful</span> <span style="--ltx-fg-color:#323232;">assistant</span><span style="--ltx-fg-color:#323232;">.&lt;|</span> <span style="--ltx-fg-color:#323232;">im_end</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> </span><span id="lstnumberx3"><span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">im_start</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> <span style="--ltx-fg-color:#323232;">user</span> </span><span id="lstnumberx4"><span style="--ltx-fg-color:#323232;">Can</span> <span style="--ltx-fg-color:#323232;">you</span> <span style="--ltx-fg-color:#323232;">explain</span> <span style="--ltx-fg-color:#323232;">the</span> <span style="--ltx-fg-color:#323232;">meaning</span> <span style="--ltx-fg-color:#323232;">of</span> <span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">item_begin</span> <span style="--ltx-fg-color:#323232;">|&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_a_219</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_b_189</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_c_49</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_d_227</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;|</span> <span style="--ltx-fg-color:#323232;">item_end</span> <span style="--ltx-fg-color:#323232;">|&gt;?&lt;|</span> <span style="--ltx-fg-color:#323232;">im_end</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> </span><span id="lstnumberx5"><span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">im_start</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> <span style="--ltx-fg-color:#323232;">assistant</span> </span><span id="lstnumberx6"><span style="--ltx-fg-color:#323232;--ltx-bg-color:#F2F9F9;">Its title is Manuka Honey and Manuka Oil Skin Cream, its category is Beauty &gt; Skin Care &gt; Face &gt; Cleansers.</span></span></span> Ground Truth <span style="--ltx-bg-color:#F2F9F9;"><a download="" href="data:text/plain;base64,InRpdGxlIjogIk1hbnVrYSBIb25leSBhbmQgTWFudWthIE9pbCBTa2luIENyZWFtIgoiY2F0ZWdvcmllcyI6ICJCZWF1dHkgPiBTa2luIENhcmUgPiBGYWNlID4gQ3JlYW1zICYgTW9pc3R1cml6ZXJzID4gRmx1aWRzICYgTG90aW9ucyA+IExvdGlvbnMiCiJzaWQiOiAiPHxpdGVtX2JlZ2lufD48aXRlbV9hXzIxOT48aXRlbV9iXzE4OT48aXRlbV9jXzQ5PjxpdGVtX2RfMjI3Pjx8aXRlbV9lbmR8PiI=">⬇</a> <span id="lstnumberx7"><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">title</span> <span style="--ltx-fg-color:#323232;">":</span><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Manuka</span> <span style="--ltx-fg-color:#323232;">Honey</span> <span style="--ltx-fg-color:#323232;">and</span> <span style="--ltx-fg-color:#323232;">Manuka</span> <span style="--ltx-fg-color:#323232;">Oil</span> <span style="--ltx-fg-color:#323232;">Skin</span> <span style="--ltx-fg-color:#323232;">Cream</span> <span style="--ltx-fg-color:#323232;">"</span> </span><span id="lstnumberx8"><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">categories</span> <span style="--ltx-fg-color:#323232;">":</span><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Beauty</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Skin</span> <span style="--ltx-fg-color:#323232;">Care</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Face</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Creams</span> <span style="--ltx-fg-color:#323232;">&amp;</span> <span style="--ltx-fg-color:#323232;">Moisturizers</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Fluids</span> <span style="--ltx-fg-color:#323232;">&amp;</span> <span style="--ltx-fg-color:#323232;">Lotions</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Lotions</span> <span style="--ltx-fg-color:#323232;">"</span> </span><span id="lstnumberx9"><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">sid</span> <span style="--ltx-fg-color:#323232;">":</span><span style="--ltx-fg-color:#323232;">"&lt;|</span> <span style="--ltx-fg-color:#323232;">item_begin</span> <span style="--ltx-fg-color:#323232;">|&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_a_219</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_b_189</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_c_49</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_d_227</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;|</span> <span style="--ltx-fg-color:#323232;">item_end</span> <span style="--ltx-fg-color:#323232;">|&gt;"</span></span></span></span></foreignObject></g></g></g></svg>

<svg height="3032.73" id="A1.SS2.SSS0.Px2.p3.pic1" overflow="visible" version="1.1" viewBox="0 0 600 3032.73" width="600"><g stroke="#000000" style="--ltx-stroke-color:#000000;" transform="translate(0,3032.73) matrix(1 0 0 -1 0 0)"><g fill="#004D4D" fill-opacity="1.0" stroke-width="0.5pt" style="--ltx-fill-color:#004D4D;"><path d="M 0 5.91 L 0 3013.72 C 0 3016.98 2.64 3019.63 5.91 3019.63 L 594.09 3019.63 C 597.36 3019.63 600 3016.98 600 3013.72 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#000000" stroke-width="0.4pt" style="--ltx-fill-color:#000000;"><g fill="#F2F9F9" fill-opacity="1.0" style="--ltx-fill-color:#F2F9F9;"><path d="M 1.97 5.91 L 1.97 3013.72 C 1.97 3015.9 3.73 3017.66 5.91 3017.66 L 594.09 3017.66 C 596.27 3017.66 598.03 3015.9 598.03 3013.72 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 13.84 3009.79)"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)"><g fill="#505050" fill-opacity="1.0" style="--ltx-fill-color:#505050;"><path d="M 0 0.69 L 0 22.25 C 0 22.63 0.31 22.94 0.69 22.94 L 228.84 22.94 C 229.23 22.94 229.54 22.63 229.54 22.25 L 229.54 0.69 C 229.54 0.31 229.23 0 228.84 0 L 0.69 0 C 0.31 0 0 0.31 0 0.69 Z" style="stroke:none"></path></g><g fill="#505050" fill-opacity="1.0" style="--ltx-fill-color:#505050;"><path d="M 0.69 0.69 L 0.69 22.25 L 228.84 22.25 L 228.84 0.69 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.5 8.01)"><foreignObject color="#000000" height="12.3" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :13.54em;--fo_height:0.63em;--fo_depth :0.18em;" transform="matrix(1 0 0 -1 0 9.61)" width="206.07"><span style="--ltx-fg-color:#FFFFFF;">Case: Itemic Token Explaining 2</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 22.51 5.91)"><foreignObject color="#000000" height="3007.82" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :40.11em;--fo_height:217.37em;--fo_depth :0em;" transform="matrix(1 0 0 -1 0 3007.82)" width="554.98"><span style="width:40.11em;">Question + Answer <span style="--ltx-bg-color:#F2F9F9;"><a download="" href="data:text/plain;base64,PHxpbV9zdGFydHw+c3lzdGVtCllvdSBhcmUgYSBoZWxwZnVsIGFzc2lzdGFudC48fGltX2VuZHw+Cjx8aW1fc3RhcnR8PnVzZXIKQ2FuIHlvdSBleHBsYWluIHRoZSBtZWFuaW5nIG9mIDx8aXRlbV9iZWdpbnw+PGl0ZW1fYV84PjxpdGVtX2JfMTg+PGl0ZW1fY184Nj48aXRlbV9kXzEzMT48fGl0ZW1fZW5kfD4/PHxpbV9lbmR8Pgo8fGltX3N0YXJ0fD5hc3Npc3RhbnQKYFx0ZXh0YmZ7SXRzIHRpdGxlIGlzIDI1IENvbG9yIFByb2Zlc3Npb25hbCBDYW1vdWZsYWdlIE1ha2V1cCBQYWxldHRlLCBpdHMgY2F0ZWdvcnkgaXMgQmVhdXR5ID4gTWFrZXVwID4gRmFjZSA+IENvbmNlYWxlciBcJiBCYXNlLn1g">⬇</a> <span id="lstnumberx10"><span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">im_start</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> <span style="--ltx-fg-color:#323232;">system</span> </span><span id="lstnumberx11"><span style="--ltx-fg-color:#323232;">You</span> <span style="--ltx-fg-color:#323232;">are</span> <span style="--ltx-fg-color:#323232;">a</span> <span style="--ltx-fg-color:#323232;">helpful</span> <span style="--ltx-fg-color:#323232;">assistant</span><span style="--ltx-fg-color:#323232;">.&lt;|</span> <span style="--ltx-fg-color:#323232;">im_end</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> </span><span id="lstnumberx12"><span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">im_start</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> <span style="--ltx-fg-color:#323232;">user</span> </span><span id="lstnumberx13"><span style="--ltx-fg-color:#323232;">Can</span> <span style="--ltx-fg-color:#323232;">you</span> <span style="--ltx-fg-color:#323232;">explain</span> <span style="--ltx-fg-color:#323232;">the</span> <span style="--ltx-fg-color:#323232;">meaning</span> <span style="--ltx-fg-color:#323232;">of</span> <span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">item_begin</span> <span style="--ltx-fg-color:#323232;">|&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_a_8</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_b_18</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_c_86</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_d_131</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;|</span> <span style="--ltx-fg-color:#323232;">item_end</span> <span style="--ltx-fg-color:#323232;">|&gt;?&lt;|</span> <span style="--ltx-fg-color:#323232;">im_end</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> </span><span id="lstnumberx14"><span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">im_start</span> <span style="--ltx-fg-color:#323232;">|&gt;</span> <span style="--ltx-fg-color:#323232;">assistant</span> </span><span id="lstnumberx15"><span style="--ltx-fg-color:#323232;--ltx-bg-color:#F2F9F9;">Its title is 25 Color Professional Camouflage Makeup Palette, its category is Beauty &gt; Makeup &gt; Face &gt; Concealer &amp; Base.</span></span></span> Ground Truth <span style="--ltx-bg-color:#F2F9F9;"><a download="" href="data:text/plain;base64,InRpdGxlIjogIlByb2Zlc3Npb25hbCAxNSBDb2xvciBDb25jZWFsZXIgQ2Ftb3VmbGFnZSBNYWtldXAgUGFsZXR0ZSIsCiJjYXRlZ29yaWVzIjogIkJlYXV0eSA+IE1ha2V1cCA+IEZhY2UgPiBDb25jZWFsZXJzICYgTmV1dHJhbGl6ZXJzIiwKInNpZCI6ICI8fGl0ZW1fYmVnaW58PjxpdGVtX2FfOD48aXRlbV9iXzE4PjxpdGVtX2NfODY+PGl0ZW1fZF8xMzE+PHxpdGVtX2VuZHw+Ig==">⬇</a> <span id="lstnumberx16"><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">title</span> <span style="--ltx-fg-color:#323232;">":</span><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Professional</span> <span style="--ltx-fg-color:#323232;">15</span> <span style="--ltx-fg-color:#323232;">Color</span> <span style="--ltx-fg-color:#323232;">Concealer</span> <span style="--ltx-fg-color:#323232;">Camouflage</span> <span style="--ltx-fg-color:#323232;">Makeup</span> <span style="--ltx-fg-color:#323232;">Palette</span> <span style="--ltx-fg-color:#323232;">",</span></span> <span id="lstnumberx17"><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">categories</span> <span style="--ltx-fg-color:#323232;">":</span><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Beauty</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Makeup</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Face</span> <span style="--ltx-fg-color:#323232;">&gt;</span> <span style="--ltx-fg-color:#323232;">Concealers</span> <span style="--ltx-fg-color:#323232;">&amp;</span> <span style="--ltx-fg-color:#323232;">Neutralizers</span> <span style="--ltx-fg-color:#323232;">",</span></span> <span id="lstnumberx18"><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">sid</span> <span style="--ltx-fg-color:#323232;">":</span><span style="--ltx-fg-color:#323232;">"&lt;|</span> <span style="--ltx-fg-color:#323232;">item_begin</span> <span style="--ltx-fg-color:#323232;">|&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_a_8</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_b_18</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_c_86</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_d_131</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;|</span> <span style="--ltx-fg-color:#323232;">item_end</span> <span style="--ltx-fg-color:#323232;">|&gt;"</span></span></span></span></foreignObject></g></g></g></svg>

### A.3 Implementation Details

#### A.3.1 Itemic Alignment

##### Task details

Here, we first introduce the aforementioned four types of tasks.

1\. Interleaved User Persona Grounding. This part contains the interleaved itemic tokens with rich, natural-language text extracted from static user profiles. This process forces the model to create a robust mapping between the itemic tokens and their real-world meanings, grounded in user attributes, stated interests, and historical behaviors. The sample data is shown below.

<svg height="7697.53" id="A1.SS3.SSS1.Px1.p3.pic1" overflow="visible" version="1.1" viewBox="0 0 600 7697.53" width="600"><g stroke="#000000" style="--ltx-stroke-color:#000000;" transform="translate(0,7697.53) matrix(1 0 0 -1 0 0)"><g fill="#B4B4B4" fill-opacity="1.0" stroke-width="0.5pt" style="--ltx-fill-color:#B4B4B4;"><path d="M 0 5.91 L 0 7678.52 C 0 7681.78 2.64 7684.43 5.91 7684.43 L 594.09 7684.43 C 597.36 7684.43 600 7681.78 600 7678.52 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#000000" stroke-width="0.4pt" style="--ltx-fill-color:#000000;"><g fill="#F5F5F5" fill-opacity="1.0" style="--ltx-fill-color:#F5F5F5;"><path d="M 1.97 5.91 L 1.97 7678.52 C 1.97 7680.7 3.73 7682.46 5.91 7682.46 L 594.09 7682.46 C 596.27 7682.46 598.03 7680.7 598.03 7678.52 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 13.84 7674.59)"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 1 0 0)"><g fill="#505050" fill-opacity="1.0" style="--ltx-fill-color:#505050;"><path d="M 0 0.69 L 0 22.25 C 0 22.63 0.31 22.94 0.69 22.94 L 284.9 22.94 C 285.28 22.94 285.59 22.63 285.59 22.25 L 285.59 0.69 C 285.59 0.31 285.28 0 284.9 0 L 0.69 0 C 0.31 0 0 0.31 0 0.69 Z" style="stroke:none"></path></g><g fill="#505050" fill-opacity="1.0" style="--ltx-fill-color:#505050;"><path d="M 0.69 0.69 L 0.69 22.25 L 284.9 22.25 L 284.9 0.69 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 12.5 8.01)"><foreignObject color="#000000" height="12.3" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :17.22em;--fo_height:0.63em;--fo_depth :0.18em;" transform="matrix(1 0 0 -1 0 9.61)" width="262.12"><span style="--ltx-fg-color:#FFFFFF;">Interleaved User Persona Grounding data</span></foreignObject></g></g></g> <g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 22.51 5.91)"><foreignObject color="#000000" height="7672.62" overflow="visible" style="--ltx-fg-color:#000000;--fo_width :40.11em;--fo_height:554.5em;--fo_depth :0em;" transform="matrix(1 0 0 -1 0 7672.62)" width="554.98"><span style="width:40.11em;"><span style="--ltx-bg-color:#F5F5F5;"><a download="" href="data:text/plain;base64,IyBVc2VyIFByb2ZpbGUgTmFycmF0aXZlClRoZSB1c2VyIGlzIGEgMjUtMzAgeWVhci1vbGQgbWFsZSBiYXNlZCBpbiBCZWlqaW5nLgoKIyBVc2VyIFByb2ZpbGUgTmFycmF0aXZlClRoZSB1c2VyJ3MgcmVjZW50IHNlYXJjaGVzIG9uIHRoZSBwbGF0Zm9ybSBpbmNsdWRlOiAiYmVzdCBzcGFjZSBvcGVyYSBub3ZlbHMsIiBhbmQgImxhdGVzdCBOQVNBIGRpc2NvdmVyaWVzLiIKCiMgTGl2ZSBTdHJlYW0gQmVoYXZpb3IKSGUgcmVjZW50bHkgY29tbWVudGVkIDUgdGltZXMgb24gbGl2ZSBzdHJlYW1zIGluIHRoZSAiU2NpZW5jZSAmIFRlY2giIGNhdGVnb3J5LgoKIyBMaWtlIEJlaGF2aW9yCkhlIHJlY2VudGx5IGxpa2VkIHZpZGVvIDx8aXRlbV9iZWdpbnw+PGl0ZW1fYV8xMTIzPjxpdGVtX2JfNTgxMz48aXRlbV9jXzQyMTI+PHxpdGVtX2VuZHw+LCBjYXB0aW9uZWQgIkV4cGxvcmluZyB0aGUgQW5kcm9tZWRhIEdhbGF4eSB3aXRoIHRoZSBKYW1lcyBXZWJiIFRlbGVzY29wZS4iOyBhbmQgdmlkZW8gPHxpdGVtX2JlZ2lufD48aXRlbV9hXzM0MjE+PGl0ZW1fYl84ODEyPjxpdGVtX2NfMTIzND48fGl0ZW1fZW5kfD4sIGNhcHRpb25lZCAiVG9wIDEwIFBhcmFkb3hlcyBvZiBUaW1lIFRyYXZlbC4iCi4uLgoKIyBDb21tZW50IEJlaGF2aW9yCkhlIHJlY2VudGx5IGNvbW1lbnRlZCAiSW5jcmVkaWJsZSBmb290YWdlISIgb24gVmlkZW86IDx8aXRlbV9iZWdpbnw+PGl0ZW1fYV81ODEzPjxpdGVtX2JfMTEyMz48aXRlbV9jXzk4NzY+PHxpdGVtX2VuZHw+LCBhIGRvY3VtZW50YXJ5IGFib3V0IGJsYWNrIGhvbGVzOyBhbmQgY29tbWVudGVkICJNaW5kLWJsb3dpbmcgY29uY2VwdCEiIG9uIFZpZGVvOiA8fGl0ZW1fYmVnaW58PjxpdGVtX2FfODgxMj48aXRlbV9iXzM0MjE+PGl0ZW1fY181NDMyPjx8aXRlbV9lbmR8PiwgZXhwbGFpbmluZyB0aGUgRmVybWkgUGFyYWRveC4KLi4uCgojIEZvbGxvd2VkIENyZWF0b3JzCkhlIGZvbGxvd3MgY3JlYXRvcnMgb24gdGhlIHBsYXRmb3JtIGFjcm9zcyB2YXJpb3VzIGZpZWxkcywgaW5jbHVkaW5nIHNjaWVuY2UgcG9wdWxhcml6ZXJzLCBib29rIHJldmlld2VycywgYW5kIGZpbG0gY3JpdGljcyBzcGVjaWFsaXppbmcgaW4gc2NpLWZpLgoKIyBVc2VyIFN1bW1hcnkKUHJpbWFyeSBJbnRlcmVzdHM6IFRoaXMgdXNlciBlbmpveXMgc2NpZW5jZSwgZXNwZWNpYWxseSBhc3Ryb25vbXkuIFRoaXMgdXNlciBhbHNvIGVuZ2FnZXMgd2l0aCBIb25vciBvZiBLaW5ncyBjb250ZW50LCBpbmRpY2F0aW5nIGEgY2FzdWFsIGdhbWluZyBpbnRlcmVzdC4KU2Vjb25kYXJ5IEludGVyZXN0czogRGl2ZXJzZSBleHBsb3JhdGlvbnMgaW5jbHVkZSBwZXQgKGNhdCkgdmlkZW9zLCB0cmFkaXRpb25hbCBjdWx0dXJlLCBhbmQgbG9jYWwgZm9vZCBjb250ZW50Lgo=">⬇</a> <span id="lstnumberx19"><span style="--ltx-fg-color:#323232;">#</span> <span style="--ltx-fg-color:#323232;">User</span> <span style="--ltx-fg-color:#323232;">Profile</span> <span style="--ltx-fg-color:#323232;">Narrative</span> </span><span id="lstnumberx20"><span style="--ltx-fg-color:#323232;">The</span> <span style="--ltx-fg-color:#323232;">user</span> <span style="--ltx-fg-color:#323232;">is</span> <span style="--ltx-fg-color:#323232;">a</span> <span style="--ltx-fg-color:#323232;">25-30</span> <span style="--ltx-fg-color:#323232;">year</span> <span style="--ltx-fg-color:#323232;">-</span> <span style="--ltx-fg-color:#323232;">old</span> <span style="--ltx-fg-color:#323232;">male</span> <span style="--ltx-fg-color:#323232;">based</span> <span style="--ltx-fg-color:#323232;">in</span> <span style="--ltx-fg-color:#323232;">Beijing</span><span style="--ltx-fg-color:#323232;">.</span></span> <span id="lstnumberx22"><span style="--ltx-fg-color:#323232;">#</span> <span style="--ltx-fg-color:#323232;">User</span> <span style="--ltx-fg-color:#323232;">Profile</span> <span style="--ltx-fg-color:#323232;">Narrative</span> </span><span id="lstnumberx23"><span style="--ltx-fg-color:#323232;">The</span> <span style="--ltx-fg-color:#323232;">user</span> <span style="--ltx-fg-color:#323232;">’</span> <span style="--ltx-fg-color:#323232;">s</span> <span style="--ltx-fg-color:#323232;">recent</span> <span style="--ltx-fg-color:#323232;">searches</span> <span style="--ltx-fg-color:#323232;">on</span> <span style="--ltx-fg-color:#323232;">the</span> <span style="--ltx-fg-color:#323232;">platform</span> <span style="--ltx-fg-color:#323232;">include</span><span style="--ltx-fg-color:#323232;">:</span><span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">best</span> <span style="--ltx-fg-color:#323232;">space</span> <span style="--ltx-fg-color:#323232;">opera</span> <span style="--ltx-fg-color:#323232;">novels</span><span style="--ltx-fg-color:#323232;">,"</span> <span style="--ltx-fg-color:#323232;">and</span> <span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">latest</span> <span style="--ltx-fg-color:#323232;">NASA</span> <span style="--ltx-fg-color:#323232;">discoveries</span><span style="--ltx-fg-color:#323232;">."</span> </span><span id="lstnumberx25"><span style="--ltx-fg-color:#323232;">#</span> <span style="--ltx-fg-color:#323232;">Live</span> <span style="--ltx-fg-color:#323232;">Stream</span> <span style="--ltx-fg-color:#323232;">Behavior</span> </span><span id="lstnumberx26"><span style="--ltx-fg-color:#323232;">He</span> <span style="--ltx-fg-color:#323232;">recently</span> <span style="--ltx-fg-color:#323232;">commented</span> <span style="--ltx-fg-color:#323232;">5</span> <span style="--ltx-fg-color:#323232;">times</span> <span style="--ltx-fg-color:#323232;">on</span> <span style="--ltx-fg-color:#323232;">live</span> <span style="--ltx-fg-color:#323232;">streams</span> <span style="--ltx-fg-color:#323232;">in</span> <span style="--ltx-fg-color:#323232;">the</span> <span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Science</span> <span style="--ltx-fg-color:#323232;">&amp;</span> <span style="--ltx-fg-color:#323232;">Tech</span> <span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">category</span><span style="--ltx-fg-color:#323232;">.</span></span> <span id="lstnumberx28"><span style="--ltx-fg-color:#323232;">#</span> <span style="--ltx-fg-color:#323232;">Like</span> <span style="--ltx-fg-color:#323232;">Behavior</span> </span><span id="lstnumberx29"><span style="--ltx-fg-color:#323232;">He</span> <span style="--ltx-fg-color:#323232;">recently</span> <span style="--ltx-fg-color:#323232;">liked</span> <span style="--ltx-fg-color:#323232;">video</span> <span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">item_begin</span> <span style="--ltx-fg-color:#323232;">|&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_a_1123</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_b_5813</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_c_4212</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;|</span> <span style="--ltx-fg-color:#323232;">item_end</span> <span style="--ltx-fg-color:#323232;">|&gt;,</span><span style="--ltx-fg-color:#323232;">captioned</span> <span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Exploring</span> <span style="--ltx-fg-color:#323232;">the</span> <span style="--ltx-fg-color:#323232;">Andromeda</span> <span style="--ltx-fg-color:#323232;">Galaxy</span> <span style="--ltx-fg-color:#323232;">with</span> <span style="--ltx-fg-color:#323232;">the</span> <span style="--ltx-fg-color:#323232;">James</span> <span style="--ltx-fg-color:#323232;">Webb</span> <span style="--ltx-fg-color:#323232;">Telescope</span><span style="--ltx-fg-color:#323232;">.";</span><span style="--ltx-fg-color:#323232;">and</span> <span style="--ltx-fg-color:#323232;">video</span> <span style="--ltx-fg-color:#323232;">&lt;|</span> <span style="--ltx-fg-color:#323232;">item_begin</span> <span style="--ltx-fg-color:#323232;">|&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_a_3421</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_b_8812</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;</span> <span style="--ltx-fg-color:#323232;">item_c_1234</span> <span style="--ltx-fg-color:#323232;">&gt;&lt;|</span> <span style="--ltx-fg-color:#323232;">item_end</span> <span style="--ltx-fg-color:#323232;">|&gt;,</span><span style="--ltx-fg-color:#323232;">captioned</span> <span style="--ltx-fg-color:#323232;">"</span> <span style="--ltx-fg-color:#323232;">Top</span> <span style="--ltx-fg-color:#323232;">10</span> <span style="--ltx-fg-color:#323232;">Paradoxes</span> <span style="--ltx-fg-color:#323232;">of</span> <span style="--ltx-fg-color:#323232;">Time</span> <span style="--ltx-fg-color:#323232;">Travel</span><span style="--ltx-fg-color:#323232;">."</span> </span><span id="lstnumberx30"><span style="--ltx-fg-color:#323232;">...</span></span> <span id="lstnumberx32"><span style="--ltx-fg-color:#323232;">#</span></span></span></span></foreignObject></g></g></g></svg>

[^1]: Millennium Bismay, Xiangjue Dong, and James Caverlee. 2024. Reasoningrec: Bridging personalized recommendations and human-interpretable explanations through llm reasoning. *arXiv preprint arXiv:2410.23180*.

[^2]: Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, and 1 others. 2025. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. *arXiv preprint arXiv:2509.03236*.

[^3]: Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A review of modern recommender systems using generative models (gen-recsys). In *Proceedings of the 30th ACM SIGKDD conference on Knowledge Discovery and Data Mining*, pages 6448–6458.

[^4]: Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. *arXiv preprint arXiv:2502.18965*.

[^5]: Yi Fang, Wenjie Wang, Yang Zhang, Fengbin Zhu, Qifan Wang, Fuli Feng, and Xiangnan He. 2025. Reason4rec: Large language models for recommendation with deliberative user preference alignment. *CoRR*.

[^6]: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*.

[^7]: Xian Guo, Ben Chen, Siyuan Wang, Ying Yang, Chenyi Lei, Yuqing Ding, and Han Li. 2025b. Onesug: The unified end-to-end generative framework for e-commerce query suggestion. *arXiv preprint arXiv:2506.06913*.

[^8]: Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In *ICLR*. ICLR.

[^9]: Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3.

[^10]: Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In *ICDM*, pages 197–206. IEEE.

[^11]: Jieyong Kim, Hyunseo Kim, Hyunjin Cho, SeongKu Kang, Buru Chang, Jinyoung Yeo, and Dongha Lee. Review-driven personalized preference reasoning with large language models for recommendation. corr, abs/2408.06276, 2024. doi: 10.48550. *arXiv preprint ARXIV.2408.06276*.

[^12]: Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213.

[^13]: Lei Li, Yongfeng Zhang, Dugang Liu, and Li Chen. 2023. Large language models for generative recommendation: A survey and visionary discussions. *arXiv preprint arXiv:2309.01157*.

[^14]: Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical gating networks for sequential recommendation. In *KDD*. ACM.

[^15]: Qiyao Peng, Hongtao Liu, Hua Huang, Qing Yang, and Minglai Shao. 2025. A survey on llm-powered agents for recommender systems. *arXiv preprint arXiv:2502.10050*.

[^16]: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, and 1 others. 2023. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems*, 36:10299–10315.

[^17]: ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, and 1 others. 2025. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. *arXiv preprint arXiv:2504.13914*.

[^18]: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*.

[^19]: Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv: 2409.19256*.

[^20]: Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*.

[^21]: Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In *CIKM*, pages 1441–1450. ACM.

[^22]: Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and Yuning Jiang. 2025. Think before recommend: Unleashing the latent reasoning power for sequential recommendation. *arXiv preprint arXiv:2503.22675*.

[^23]: Alicia Tsai, Adam Kraft, Long Jin, Chenwei Cai, Anahita Hosseini, Taibai Xu, Zemin Zhang, Lichan Hong, Ed H. Chi, and Xinyang Yi. 2024. [Leveraging LLM reasoning enhances personalized recommender systems](https://doi.org/10.18653/v1/2024.findings-acl.780). In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 13176–13188, Bangkok, Thailand. Association for Computational Linguistics.

[^24]: Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024a. Learnable item tokenization for generative recommendation. In *CIKM*. ACM.

[^25]: Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, and Tat-Seng Chua. 2023. Generative recommendation: Towards next-generation recommender paradigm. *arXiv preprint arXiv:2304.03516*.

[^26]: Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*.

[^27]: Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, and 1 others. 2024b. Content-based collaborative generation for recommender systems. In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management*, pages 2420–2430.

[^28]: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837.

[^29]: Zhipeng Wei, Kuo Cai, Junda She, Jie Chen, Minghao Chen, Yang Zeng, Qiang Luo, Wencong Zeng, Ruiming Tang, Kun Gai, and 1 others. 2025. Oneloc: Geo-aware generative recommender systems for local life service. *arXiv preprint arXiv:2508.14646*.

[^30]: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*.

[^31]: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models, 2023. *URL https://arxiv. org/abs/2305.10601*, 3:1.

[^32]: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*.

[^33]: Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, and 1 others. 2025. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. *arXiv preprint arXiv:2504.05118*.

[^34]: Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, and 1 others. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152*.

[^35]: Junjie Zhang, Beichen Zhang, Wenqi Sun, Hongyu Lu, Wayne Xin Zhao, Yu Chen, and Ji-Rong Wen. 2025a. Slow thinking for sequential recommendation. *arXiv preprint arXiv:2504.09627*.

[^36]: Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. *arXiv preprint arXiv:1904.09675*.

[^37]: Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, and Yong Li. 2025b. A survey of large language model empowered agents for recommendation and search: Towards next-generation information retrieval. *arXiv preprint arXiv:2503.05659*.

[^38]: Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In *ICDE*. IEEE.

[^39]: Guorui Zhou, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo, Qianqian Wang, Qigen Hu, Rui Huang, Shiyao Wang, and 1 others. 2025a. Onerec technical report. *arXiv preprint arXiv:2506.13695*.

[^40]: Guorui Zhou, Hengrui Hu, Hongtao Cheng, Huanjie Wang, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Lu Ren, Liao Yu, and 1 others. 2025b. Onerec-v2 technical report. *arXiv preprint arXiv:2508.20900*.