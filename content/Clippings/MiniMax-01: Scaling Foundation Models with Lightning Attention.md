---
title: "MiniMax-01: Scaling Foundation Models with Lightning Attention"
source: "https://arxiv.org/abs/2501.08313"
author: "MiniMax et al."
published: "2025"
created: "2026-05-07"
description: null
tags: ["clippings"]
arxiv: "2501.08313"
url: "https://arxiv.org/abs/2501.08313"
---

# MiniMax-01: Scaling Foundation Models with Lightning Attention

MiniMax1

We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01, which are comparable to top-tier models while offering superior capabilities in processing longer contexts. The core lies in lightning attention and its efficient scaling. To maximize computational capacity, we integrate it with Mixture of Experts (MoE), creating a model with 32 experts and 456 billion total parameters, of which 45.9 billion are activated for each token. We develop an optimized parallel strategy and highly efficient computation-communication overlap techniques for MoE and lightning attention. This approach enables us to conduct efficient training and inference on models with hundreds of billions of parameters across contexts spanning millions of tokens. The context window of MiniMax-Text-01 can reach up to 1 million tokens during training and extrapolate to 4 million tokens during inference at an affordable cost. Our vision-language model, MiniMax-VL-01 is built through continued training with 512 billion vision-language tokens. Experiments on both standard and in-house benchmarks show that our models match the performance of state-of-the-art models like GPT-4o and Claude-3.5-Sonnet while offering a 20-32 times longer context window. We publicly release MiniMax-01 at https://github.com/MiniMax-AI.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/e71cc192a8bbe10b2134b6c225f1871c7a6f599faed8e0134ffe713649f04d5c.jpg]]  
(a) Core text benchmark performance

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/121a7b1ea7b2fbe42f7e6818eb096885c0c7d363dea24e3217503a5791e94ce5.jpg]]  
(c) Long-context RULER performance

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/6b42832bcf0b32500b4a73047df3480312a26393f423fc0d9a423b79608aacfc.jpg]]  
(b) Core multimodal benchmark performance   
Figure 1 | Benchmark performance. (a) MiniMax-Text-01 on core text benchmarks. (b) MiniMax-VL-01 on core multimodal benchmarks. (c) MiniMax-Text-01 on the long-context RULER (Hsieh et al., 2024) benchmark. The performance of leading commercial and open-source models is presented for reference.

# 1. Introduction

Large Language Models (LLMs) (Anthropic, 2024; Dubey et al., 2024; Hurst et al., 2024; Team et al., 2024a) and Vision Language Models (VLMs) (Anthropic, 2024; Dubey et al., 2024; Hurst et al., 2024; Team et al., 2024a) have made rapid progress in recent years, excelling at tasks like knowledge Q&A, complex reasoning, mathematics, coding, and vision-language understanding. The context window for most models currently ranges from 32K to 256K tokens. However, these lengths often fall short of practical needs—whether using a professional book as context, assisting with an entire programming project, or maximizing the potential of in-context learning through many-shot examples.

Context window expansion in the past two years has primarily resulted from more powerful GPUs and better I/O-aware softmax attention implementation (Dao et al., 2022; Liu et al., 2024a). However, extending these windows further has proven challenging. This limitation arises from the inherent quadratic computational complexity of the transformer (Vaswani et al., 2017) architecture—further length extension causes computational demands to grow much faster than hardware capabilities can match. To address this challenge, researchers have proposed various methods for reducing the attention mechanism’s computational complexity: sparse attention (Beltagy et al., 2020; Zaheer et al., 2020), linear attention (Qin et al., 2022a,b, 2024c), long convolutions (Qin et al., 2023a), state space models (the Mamba series) (Dao and Gu, 2024; Glorioso et al., 2024; Gu and Dao, 2024; Ren et al., 2024; Team et al., 2024b), and linear RNNs (Qin et al., 2023b, 2024d). Despite their theoretical promise, these innovations have seen limited adoption in commercial-scale models.

In this report, we aim to build a model that matches the performance of leading commercial models while providing a context window longer by an order of magnitude. This ambitious objective requires carefully balancing multiple factors: network architecture, data, and computation.

Our approach begins with selecting the most promising architecture, succeeded by the optimization of the underlying training and inference framework to ensure its support. For the network architecture, we required linear attention—not just theoretically sound but highly efficient in practice, especially with long contexts. After extensive experimentation, we settled on a hybrid architecture mainly using lightning attention (Qin et al., 2024b), an I/O-aware implementation of a linear attention variant (Qin et al., 2022a). In the architecture, one transformer block with softmax attention follows every seven transnormer blocks (Qin et al., 2022a) with lightning attention.

We determined the model’s total parameters based on a practical constraint: the ability to process more than 1 million tokens on a single machine with up to 8 GPUs and 640GB memory using 8-bit quantization. To maximize parameter and computation capacity, we implemented a Mixture of Experts (MoE) (Fedus et al., 2022; Lepikhin et al., 2021). We comprehensively consider training resources, inference resources, and the final model performance, aiming to find a better balance among the three. Extensive experiments guided us toward the final model specifications: 456 billion parameters, 45.9 billion activations, and 32 experts.

Existing distributed training and inference frameworks are primarily optimized for softmax attention. However, our novel architecture, which integrates lightning attention, softmax attention, and MoE, necessitates a complete redesign of both our training and inference frameworks. Furthermore, the framework must possess the capability to support the training and inference of models with hundreds of billions of parameters and context windows extending over millions of tokens. To this end, we implement the all-to-all communication in MoE using expert parallel (EP) and expert tensor parallel (ETP). It aims to minimize the overhead associated with inter-GPU communication. To facilitate context windows with unlimited expansion, we design varlen ring attention to reduce the redundancy in computation and the improved version of Linear Attention Sequence Parallelism (LASP) (Sun et al., 2024) to fully utilize the device’s parallel capabilities. Additionally, we have implemented

a comprehensive set of CUDA kernels tailored for lightning attention inference, achieving over $7 5 \%$ Model Flops Utilization (MFU) (Chowdhery et al., 2023) end-to-end on the Nvidia H20.

Building upon the architecture design and computation optimizations, we train our foundational language model, MiniMax-Text-01. Our pre-training process began with curating a diverse and high-quality corpus through rigorous data cleaning, reward-based quality enhancement, and better

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/d2cf48fc93639d4d662561be114ad77b271c70463a231209a4844b0ea9dbe3e5.jpg]]  
Figure 2 | Prefilling latency of different models. The MiniMax-Text-01 and Llama3-70B models are tested on H800 GPUs with tensor parallelism set to 8, utilizing a custom inference framework with 8- bit weight-only quantization (W8A16). Other models are tested through their official APIs. Within the maximum length supported by each model, a sufficient number of uniformly distributed points were selected for testing. After removing outliers, the data is fitted with a quadratic function.

data mixture balancing, validated through systematic repetition-aware testing. To fully utilize the architecture’s long-context capability, we introduce in-depth analysis of the hyperparameters and propose a three-stage training procedure, successfully extending the context window to one million tokens. During the alignment phase, we incentivize the model’s various capabilities through precisely tuned reward dimensions and multi-stage training methodology, especially in the areas of long-context and realworld scenarios. Subsequently, we augment our language model with visual capabilities by integrating a lightweight Vision Transformer (ViT) (Dosovitskiy et al., 2021) module, thereby creating our vision-language model, MiniMax-VL-01. MiniMax-VL-01 undergoes additional training with 512 billion vision-language tokens, utilizing a four-stage training process. The final stage of this training is specifically designed to optimize the user experience.

Comprehensive evaluations on core academic benchmarks demonstrate that both models attain performance levels comparable to those of closed-source top-tier models in both text and vision-language tasks, as illustrated in Figure 1 (a,b). For contexts longer than 200k, our model performs significantly better,

as shown in Figure 1 (c). In addition to academic benchmarks, we also assess the models’ performance using in-house benchmarks derived from real-world usage and show that our model is top-tier in those scenarios. In addition to its performance, our model exhibits significant advantages in prefilling latency, attributed to its novel architecture, as illustrated in Figure 2.

We summarize our contributions as follows:

1. We build a model that rivals the top-tier closed-source models on standard academic benchmarks. Furthermore, this model supports context inputs of up to 4 million tokens, showcasing outstanding performance in long-context evaluations.   
2. We demonstrate the first successful large-scale implementation of linear attention. While linear attention has been studied before, it has never been deployed at this scale. We provide comprehensive details on our algorithm design and engineering optimizations.   
3. We outline a practical approach and experimental methodology for the exploration of various models, datasets, evaluations, and algorithms, which may serve as a valuable reference.   
4. We publicly release the weights and offer a cost-effective API, aiming to help others develop

models that push beyond current limitations.

# 2. Model Architecture

In this section, we present the design of our network architecture. To achieve optimal performance within constrained resources and better handle longer sequences, we adopt MoE approach and employ linear attention as much as possible instead of the traditional softmax attention used in standard transformers.

To facilitate a more intuitive understanding, we illustrate the main architecture in Figure 3. Our design follows the Transformer-style block, with each comprises a channel mixer (an attention block) and a feature mixer (an MLP block). We employ two types of channel mixers: lightning attention and softmax attention. The feature mixer is an MoE that incorporates multiple feed-forward networks (FFNs). To ensure load balancing in the MoE blocks, we propose a novel load balancing strategy inspired by GShard (Lepikhin et al., 2021), which we refer to the global router. This strategy is

designed to maintain training stability. Additionally, DeepNorm (Wang et al., 2024a) is integrated to enhance overall performance.

The final MiniMax-Text-01 architecture integrates both linear attention and softmax attention mechanisms in a structured pattern. Specifically, a transformber block with softmax attention is positioned after every 7 transnormer blocks (Qin et al., 2022a) of linear attention, leading to a total of 80 layers. Each attention module is composed of 64 heads, each with a head dimension of 128. The softmax attention layers employ Group Query Attention (GQA) (Ainslie et al., 2023) with a group size of 8. Rotary Position Embedding (RoPE) (Su et al., 2024) is applied to half of the attention head dimension, with a base frequency set to 10,000. The model’s hidden size is configured to 6144, and each layer incorporates 32 experts with a top-2 routing strategy. The feed-forward network within each expert has a hidden dimension of 9216. In total, MiniMax-Text-01 compromises 456 billion parameters, of which 45.9 billion are activated for each processed token.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/b7a9ea9d745eac0b052a8ef38d0b2bc0369799c87551c8f9ff18771d8434cafe.jpg]]  
Figure 3 | The architecture of MiniMax-Text-01.

In the subsequent sections, we will delve into our considerations regarding the model architecture, i.e., the integration of different attention mechanisms, the synergy between MoE and linear attention, the rationale behind hyperparameter selection, and the methodology for determining the model’s size based on scaling laws.

# 2.1. Mixture of Experts

MoE provides a pathway to enhance both scalability and efficiency compared to the dense version. Typically, MoE is a substitute for the feed forward networks (FFN) in feature-mixer layers (Fedus

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/55df489ac181a5b3d8af21ccb6c6a5d9526a2143e3212076ec4ce8c666fb2b3c.jpg]]  
Figure 4 | Isoflop Comparison: MoE vs. Dense on various benchmarks. Both models are trained on 1 trillion tokens. The gray dashed lines indicate the difference in the computation required for the two models to achieve the same performance.

et al., 2022; Lepikhin et al., 2021), which consists of multiple FFN experts, where each token is routed to one or more of these experts. Specifically, for an input token $\mathbf { x } _ { t }$ , its corresponding output hidden state $\mathbf { h } _ { t }$ is calculated as:

$$
\mathbf {h} _ {t} = \sum_ {i = 1} ^ {E} \operatorname {S o f t m a x} _ {i} \left(\operatorname {T o p K} \left(\mathbf {x} _ {t} \cdot \mathbf {W} _ {g}\right)\right) \cdot \operatorname {F F N} _ {i} \left(\mathbf {x} _ {t}\right), \tag {1}
$$

where $E$ represents the total number of experts, ${ \bf W } _ { g }$ is the weight of the gate, $\mathrm { F F N } _ { i }$ stands for the ??-th expert, and TopK(·) denotes the operation that preserves the top $k$ scores among all $E$ experts while setting the remaining scores to $- \infty$ .

The training of MoE based LLMs can be categorized into token-drop and dropless. We adopt the token-drop strategy to improve training efficiency. With this approach, each expert is assigned a capacity limit specifying the maximum number of tokens it can handle. Once this capacity is reached, any additional token routed to that expert is discarded.

To assess the effectiveness of the MoE architecture, we conduct a comparative study between a dense model with 7 billion parameters and an MoE model with 2 billion activation parameters out of a total of 24 billion parameters. The results, as illustrated in Figure 4, demonstrate that the MoE model significantly outperforms the dense model under the same computational budget on various benchmarks, including HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), Natural Questions(Kwiatkowski et al., 2019), PIQA(Bisk et al., 2020) and TriviaQA(Joshi et al., 2017). When scaling up to larger models, we encounter the challenge of routing collapse, which arises due to the concentrated distribution of tokens designated for allocation. To mitigate this issue, we incorporate a simple global routing strategy to the GShard (Lepikhin et al., 2021) auxiliary loss for better load balancing.

Auxiliary Loss. To ensure differentiability, the auxiliary loss is defined as $\begin{array} { r } { L _ { \mathrm { a u x } } = \alpha _ { \mathrm { a u x } } \cdot \frac { 1 } { E } \sum _ { i = 1 } ^ { E } f _ { i } \cdot m _ { i } , } \end{array}$ where $\alpha _ { \mathrm { a u x } }$ represents the coefficient of the auxiliary loss, $f _ { i }$ denotes the fraction of tokens assigned to the $i$ -th expert, and $m _ { i }$ is the average routing probability of expert ??.

Global Router. The GPU memory size constrains the micro batch size in LLM training, leading to substantial fluctuations in the token distribution within individual Expert Parallel (EP) groups. Moreover, token distributions vary across different EP groups, potentially resulting in load imbalances where experts in one EP group may be overloaded while those in another are underutilized. To address this, we implement a global token dispatching strategy across EP groups. Specifically, we introduce an additional allgather communication step to synchronize the number of tokens awaiting processing by each expert before dispatching tokens across different EP groups. Under the same

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/f8eae89e9bb940942e17f7eed746bc2825d5989b6d7f2ba87ed8b847a013949f.jpg]]

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/259ac7cf614b5c132a2ffe0052ba263896c3217e9ba0a2121e8cdbd738a85894.jpg]]  
Figure 5 | Illustration of the computations for softmax attention (left) and linear attention (right). The input length is $N$ and feature dimension is $d$ , with $d \ll N$ . Tensors in the same box are associated with computation. The linearized formulation allows $O ( N )$ time and space complexity.

capacity constraints, this global routing mechanism can effectively reduce the overall token drop rate, thereby ensuring training stability.

# 2.2. Linear Attention

Linear attention utilizes the “right product kernel trick” to transform quadratic computational complexity into linear complexity, as illustrated in Figure 5. By taking TransNormer (Qin et al., 2022a) as an example, the NormAttention mechanism can be written as:

$$
\mathbf {O} = \operatorname {N o r m} \left(\left(\mathbf {Q K} ^ {\top}\right) \mathbf {V}\right), \tag {2}
$$

where Q, K, and $\mathbf { V } \in \mathbb { R } ^ { n \times d }$ are the query, key, and value matrices, respectively, with $n$ for sequence length and $d$ for feature dimension. The equation can be transformed into its linear variant using right matrix multiplication:

$$
\mathbf {O} = \operatorname {N o r m} (\mathbf {Q} \left(\mathbf {K} ^ {\top} \mathbf {V}\right)), \tag {3}
$$

The linear formulation facilitates efficient recurrent prediction with a training complexity of $O ( n d ^ { 2 } )$ . Furthermore, linear attention ensures a constant computational complexity of $O ( d ^ { 2 } )$ , irrespective of the sequence length. This is accomplished by recurrently updating the term $\mathbf { K } ^ { \top } \mathbf { V }$ , thereby obviating the need for repetitive computation of the entire attention matrix. In contrast, softmax attention incurs a complexity of $O ( n d ^ { 2 } )$ during inference.

When addressing causal language modeling tasks, the efficacy of the right product is compromised, necessitating the computation of cumsum (Hua et al., 2022). This limitation impedes the realization of highly efficient parallel computation, which likely explains why, despite being proposed by Brébisson et al. (de Brébisson and Vincent, 2016) nine years ago, none of the current leading open-source LLMs—including LLaMA3 (Dubey et al., 2024), Qwen2.5 (Yang et al., 2024), DeepSeekV3 (DeepSeek-AI, 2024), and Mistral (Jiang et al., 2023)—have adopted this linear attention mechanism.

# 2.2.1. Lightning Attention

Lightning attention (Qin et al., 2024b,c) represents an I/O-aware, optimized implementation of TransNormer (Qin et al., 2022a). This approach identifies the primary bottleneck in the computational efficiency of existing linear attention mechanisms: the slow cumsum operation inherent in causal

language modeling. To alleviate this problem, Lightning Attention proposes a novel tiling technique that effectively circumvents the cumsum operation. The key innovation lies in the strategic division of the attention calculation into two distinct components: intra-block and inter-block computations. The left product attention calculation is employed for intra-block operations, while the right product is utilized for inter-block operations. This division is crucial because the intra-blocks can be significantly reduced in size, thereby ensuring that the overall computational complexity remains linear.

Note that the lightning attention was originally proposed by our team members in Qin et al. (2024c), we recall some of the core processes to elucidate why it can achieve theoretical linear complexity in practice for the sake of completeness. In the interest of analytical tractability, we deliberately omit the consideration of normalization, sigmoid linear unit (SiLU) activation, and gating mechanisms in the following derivation.

Let us start with the forward pass in lightning attention. The left product in causal attention calculation is defined as:

$$
\mathbf {O} = \left[ \left(\mathbf {Q K} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {V} \tag {4}
$$

where $\mathbf { M } _ { t s } = 1$ if $t \geq s$ , otherwise 0. The right product operation can be computed in a recursive formula as:

$$
\mathbf {k} \mathbf {v} _ {0} = \mathbf {0}, \mathbf {k} \mathbf {v} _ {t} = \mathbf {k} \mathbf {v} _ {t - 1} + \mathbf {k} _ {t} \mathbf {v} _ {t} ^ {\top}, \mathbf {o} _ {t} ^ {\top} = \mathbf {q} _ {t} ^ {\top} \mathbf {k} \mathbf {v} _ {\mathbf {t}}. \tag {5}
$$

It is important to note that while Eq. 5 exhibits linear computational complexity, it is inherently unparallelizable.

The fundamental concept underlying the implementation of lightning attention involves the utilization of a tiling technique to compute attention scores. Specifically, the matrices Q, K, V are partitioned into two distinct blocks along the row dimension:

$$
\mathbf {X} = \left[ \begin{array}{l} \mathbf {X} _ {1} \\ \mathbf {X} _ {2} \end{array} \right], \mathbf {X} _ {1} \in \mathbb {R} ^ {m \times d}, \mathbf {X} _ {2} \in \mathbb {R} ^ {(n - m) \times d}, \mathbf {X} \in \{\mathbf {Q}, \mathbf {K}, \mathbf {V} \}.
$$

By unfolding Eq. 4, we obtain the following expression (noting that $\mathbf { k } \mathbf { v } _ { 0 } = 0 .$ ):

$$
\mathbf {k} \mathbf {v} _ {s} = \mathbf {k} \mathbf {v} _ {0} + \sum_ {j = 1} ^ {s} \mathbf {k} _ {j} \mathbf {v} _ {j} ^ {\top}, s = 1, \dots , m. \quad \mathbf {o} _ {s} ^ {\top} = \mathbf {q} _ {s} ^ {\top} \mathbf {k} \mathbf {v} _ {s} = \mathbf {q} _ {s} ^ {\top} \mathbf {k} \mathbf {v} _ {0} + \mathbf {q} _ {s} ^ {\top} \sum_ {j = 1} ^ {s} \mathbf {k} _ {j} \mathbf {v} _ {j} ^ {\top}. \tag {6}
$$

Rewrite it in block form, we have:

$$
\mathbf {O} _ {1} = \mathbf {Q} _ {1} \mathbf {k} \mathbf {v} _ {0} + \left[ \left(\mathbf {Q} _ {1} \mathbf {K} _ {1} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {V} _ {1} \triangleq \mathbf {Q} _ {1} \mathbf {K} \mathbf {V} _ {0} + \left[ \left(\mathbf {Q} _ {1} \mathbf {K} _ {1} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {V} _ {1}. \tag {7}
$$

As shown, the intra-block $[ ( \mathbf { Q } _ { 1 } \mathbf { K } _ { 1 } ^ { \top } ) \odot \mathbf { M } ] \mathbf { V } _ { 1 }$ can use the left product and the inter-block ${ \bf Q } _ { 1 } { \bf K V } _ { 0 }$ can use the right product. Note that the intra-block can be further divided using the same strategy:

$$
\mathbf {k} \mathbf {v} _ {m + t} = \mathbf {k} \mathbf {v} _ {m} + \sum_ {j = m + 1} ^ {m + t} \mathbf {k} _ {j} \mathbf {v} _ {j} ^ {\top}, t = 1, \dots , n - m, \mathbf {o} _ {m + t} ^ {\top} = \mathbf {q} _ {m + t} ^ {\top} \mathbf {k} \mathbf {v} _ {m + t}, \tag {8}
$$

$$
\mathbf {O} _ {2} = \mathbf {Q} _ {2} \mathbf {k v} _ {m} + \left[ \left(\mathbf {Q} _ {2} \mathbf {K} _ {2} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {V} _ {2} \triangleq \mathbf {Q} _ {2} \mathbf {K V} _ {1} + \left[ \left(\mathbf {Q} _ {2} \mathbf {K} _ {2} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {V} _ {2}.
$$

To compute the second block, we use $ { \mathbf { K } }  { \mathbf { V } } _ { 1 } =  { \mathbf { k } }  { \mathbf { v } } _ { m }$ , which can be computed by:

$$
\mathbf {K V} _ {1} = \mathbf {K V} _ {0} + \sum_ {j = 1} ^ {m} \mathbf {k} _ {m} \mathbf {v} _ {m} ^ {\top} = \mathbf {K V} _ {0} + \mathbf {K} _ {1} ^ {\top} \mathbf {V} _ {1}. \tag {9}
$$

where $\mathbf { K } \mathbf { V } _ { 0 } = \mathbf { k } \mathbf { v } _ { 0 }$ . By recursively applying the aforementioned strategy of partitioning the matrix into multiple blocks, the practical computational complexity can be reduced to linear. The final time complexity of lightning attention is $O ( n d ^ { 2 } + n B d )$ , where $B$ is the block size. Algorithm 1 illustrates the IO-aware implementation of lightning attention forward pass.

Algorithm 1 Lightning Attention Forward Pass   
Input: $\mathbf{Q},\mathbf{K},\mathbf{V}\in \mathbb{R}^{n\times d}$ , block sizes $B$ Divide $\mathbf{X}$ into $T = \frac{n}{B}$ blocks $\mathbf{X}_1,\mathbf{X}_2,\dots \mathbf{X}_T$ of size $B\times d$ each, where $\mathbf{X}\in \{\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O}\}$ Initialize mask $\mathbf{M}\in \mathbb{R}^{B\times B}$ , where $\mathbf{M}_{ts} = 1$ , if $t\geq s$ , else O. Initialize $\mathbf{KV} = 0\in \mathbb{R}^{d\times d}$ .   
for $t = 1,\ldots ,T$ do Load $\mathbf{Q}_t,\mathbf{K}_t,\mathbf{V}_t\in \mathbb{R}^{B\times d}$ from HBM to on-chip SRAM. On chip, compute $\mathbf{O}_{\mathrm{intra}} = [(\mathbf{Q}_t\mathbf{K}_t^\top)\odot \mathbf{M}]\mathbf{V}_t$ On chip, compute $\mathbf{O}_{\mathrm{inter}} = \mathbf{Q}_t(\mathbf{KV})$ On chip, compute $\mathbf{KV} = \mathbf{KV} + \mathbf{K}_t^\top \mathbf{V}_t$ Write $\mathbf{O}_t = \mathbf{O}_{\mathrm{intra}} + \mathbf{O}_{\mathrm{inter}}$ to HBM as the $t$ -th block of O.   
end for   
Return O.

# 2.2.2. Effectiveness of Lightning Attention

Although lightning attention demonstrates promise and competitive performance in small-scale experiments, its scaling behavior and capability in the downstream tasks under large-scale settings remain unexplored. To mitigate the gap, we conduct a series of scaling experiments to evaluate the scalability of the lightning attention mechanism in comparison to softmax attention, meanwhile verifying the performance on the extensive downstream tasks. It is noteworthy that during our experiments, we observed that lightning attention demonstrates limited retrieval capabilities. This finding inspired us to explore a hybrid approach (Hybrid-lightning) that takes the advantages of both lightning and softmax attention to enhance retrieval performance by substituting lightning attention with softmax attention at intervals of every eight layers.

We adhere to the FLOPs calculation methodology established by Kaplan et al. (2020). For the purpose of our analysis, we define the following variables: ?? (number of layers), ?? (model dimension), $h$ (number of attention heads), ?? (batch size) and ?? (sequence length). The checklist of model parameters and FLOPs is presented in Table 1.

Table 1 | Model Parameters and FLOPs Comparisons Across Architectures. For scaling law calculations, embedding parameters and other subleading terms are excluded to improve alignment with fitted results.   

<table><tr><td>Architecture</td><td>Parameter count</td><td>FLOPs count</td></tr><tr><td>Softmax Attention</td><td>12ld2</td><td>72bnld2(1 + n/6d + 5/18d)</td></tr><tr><td>Lightning Attention</td><td>12ld2 + 2ld2/h</td><td>72bnld2(1 + 1/2h + 5/18d)</td></tr><tr><td>Hybrid-lightning</td><td>12ld2 + 7ld2/4h</td><td>72bnld2(1 + n/48d + 7/16h + 5/18d)</td></tr></table>

# 2.2.2.1 Experimental Setup

We conducted training on softmax (equipped with FlashAttention-2 (Dao, 2024)), lightning attention, and hybrid-lightning attention models across various scales: 70 million, 160 million, 410 million, 1 billion, 3 billion, and 7 billion parameters. Each model was trained on a dataset consisting of up to 300 billion tokens, with a context length of 8192. Our training methodology follows the approach proposed by Chinchilla (Hoffmann et al., 2022), where the training loss serves as a direct indicator of test performance. For each model architecture and training sequence length, we maintained a

Table 2 | Summary of Scaling Laws: It shows the relationships between loss $\left( L \right)$ , optimal model size $( N _ { o p t } )$ , and optimal dataset size $( D _ { o p t } )$ as functions of computational budget (??). It reveals that, given the same budget, the hybrid model uses more parameters and tokens but achieves lower loss.   

<table><tr><td>Arch</td><td>L(C)</td><td>N_opt(C)</td><td>D_opt(C)</td></tr><tr><td>Softmax Attention</td><td>3.7087C-0.0798</td><td>(1.82 × 10^8)C^0.7118</td><td>(2.56 × 10^10)C^0.5102</td></tr><tr><td>Lightning Attention</td><td>3.5391C-0.0768</td><td>(2.74 × 10^8)C^0.6470</td><td>(4.43 × 10^10)C^0.4684</td></tr><tr><td>Hybrid-lightning</td><td>3.4797C-0.0763</td><td>(2.57 × 10^8)C^0.6670</td><td>(3.70 × 10^10)C^0.4707</td></tr></table>

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/17f804d03daa8605abf1e06ec5692c9aebd3b95965d401a4070b93d90a5a5bdf.jpg]]  
Softmax Attention

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/20b054f8fed06bffe4768391a5dde5f1089434e7f43723253925e75b3ab8501a.jpg]]  
Lightning Attention

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/0f6b0ad4b03678b2660949b3d380b66c98ac4dcf9f7a1f995611b0dd5649db62.jpg]]  
Hybrid-lightning   
Figure 6 | Summary of Scaling Laws. Training curves (left) span models from 70M to 7B parameters. Optimal model size (center) and training tokens (right) are derived based on a specified compute budget estimation.

uniform global batch size of 4 million tokens. The Adam optimizer was employed, configured with a learning rate of 3e-4 and a weight decay of 0.1. A fixed learning rate scheduler was applied across all experiments due to constrained computational resources.

We employ a diverse set of evaluation benchmarks, including BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), ARC (both easy and challenge variants) (Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018), Needle in A Haystack (NIAH) (Shen et al., 2024), and SCROLLS (Shaham et al., 2022). Each benchmark assesses distinct capabilities of the models.

# 2.2.2.2 Scaling Laws

We fit the scaling curves based on our experiments over the above mentioned settings, where we alter the model size $( N )$ and dataset size $( D )$ for different computational budget (??) and observe the corresponding training loss $\left( L \right)$ that serving as an estimator of test loss. We begin by establishing power-law relationships between $L$ and $C$ , following Chinchilla’s methodology (Hoffmann et al., 2022). Using the fitted curve, we derive coefficients for optimal model size $N _ { o p t } \propto C ^ { a }$ and optimal dataset size $D _ { o p t } \propto C ^ { b }$ . The original scaling laws (Kaplan et al., 2020) use $L ( X ) = ( X _ { 0 } / X ) ^ { \alpha _ { X } }$ , while subsequent studies (Clark et al., 2022; Gao et al., 2024; Henighan et al., 2020; Hoffmann et al., 2022) employ $L ( X ) = \epsilon + ( X _ { 0 } / X ) ^ { \alpha _ { X } }$ for better fitting, where $\epsilon$ denotes the irreducible loss. For simplicity, we unify these forms into $L ( X ) = \beta _ { X } X ^ { \alpha _ { X } }$ , facilitating a direct comparison of scaling capabilities based on $\alpha _ { X }$ and $\beta _ { X }$ . The summary of scaling laws is shown in Table 2 and Figure 6. It can be intuitively understood that given the same computational budget, models with lightning attention tend to utilize more parameters and tokens, yet they achieve a lower loss compared to models with pure softmax attention.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/199e7028efe844a2701913865a27070fb13237d5d17f991c5296aac8fbcfe2d6.jpg]]

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/c5e47e2094f0a17f857a4bad25534e34d0a92ee11cc3bea9bb2d67bdd5799a6a.jpg]]

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/afd25b5fc47d073cdfe5740d4c2cda6f424be35029c290756094264df575cd2c.jpg]]

Figure 7 | Larger models and hybrid-lightning attention achieve the best performance across benchmarks. Performance is evaluated on CSR (Common Sense Reasoning), NIAH (Needle in a Haystack), and SCROLLS benchmarks using three attention mechanism models from 410M to 7B parameters.   
![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/ceaee84d1fe5a8aef8cbfa7c9117058ad405927046a9df62c838df5ffcce3660.jpg]]  
Softmax Attention Lightning Attention Hybrid-lightning

# 2.2.2.3 Performance on Downstream Task.

We present the benchmark results of downstream tasks in Figure 7. Lightning attention demonstrates comparable performance across most downstream tasks, with the exception of NIAH. This indicates that linear attention exhibits similar language modeling capabilities to Transformer models but falls short in retrieval tasks, rendering it unsuitable for LLMs. However, the hybrid-lightning attention not only matches but surpasses the retrieval and extrapolation capabilities of softmax attention, making it well-suited for in-context learning in LLMs.

# 2.2.2.4 Speed.

We assess the end-to-end training speed of softmax attention, lightning attention, and hybridlightning models with 3 billion parameters by measuring the tokens processed per GPU per second (TGS). For completeness, we also included popular linear models such as HGRN2 and Mamba2 in our evaluation. For the speed benchmark, the training context length was gradually increased until reaching the out-ofmemory limit on a single-node H800 GPUs. As illustrated in Fig. 8, lightning attention achieves a constant training speed irrespective of the sequence length and is the sole linear model that outperforms FlashAttention2.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/a0a990f797fa0c97eeb9ef25defdcf6659efed520f88ef9eb7844d51c4f1bf3d.jpg]]  
Figure 8 | The training speed of various attention mechanisms, including softmax, lightning, hybridlightning, HGRN2, and Mamba2, was benchmarked across sequence lengths ranging from 1,024 to 65,536. Performance was measured in terms of training speed, reported as tokens processed per GPU per second (TGS).

# 2.2.3. Hybrid Architecture

Our preliminary experiments with the hybrid architecture have yielded promising results, motivating us to delve deeper into its potential through two variants: hybrid-cosformer2 and hybrid-hgrn2. In the hybrid-cosformer2 model, we replace the linear attention layers in the cosformer2 architecture with softmax attention layers at intervals of every eight layers. This substitution strategy is similarly applied in the hybrid-hgrn2 model. We conduct experiments using consistent setups to evaluate the downstream performance of these alternatives. Our findings, as summarized in Table 3, indicate that the hybrid-lightning model achieves the best performance.

Table 3 | Benchmarking various hybrid-linear models with 1 Billion Parameters. We present the average CSR score, weighted average accuracy for NIAH, and the average SCROLLS score. Higher scores indicate better performance across all tasks. Abbreviations: TGS (token per gpu per second), HS (HellaSwag), WG (WinoGrande), OBQA (OpenBookQA), NIAH, and SCR (SCROLLS).   

<table><tr><td>Hybrid-linear Arch.</td><td>TGS ↑</td><td>PIQA↑</td><td>HS↑</td><td>WG↑</td><td>ARC-E↑</td><td>ARC-C↑</td><td>OBQA↑</td><td>CSR ↑</td><td>NIAH ↑</td><td>SCR ↑</td></tr><tr><td>Hybrid-cosformer2</td><td>23.3K</td><td>70.29</td><td>45.63</td><td>51.46</td><td>55.77</td><td>26.11</td><td>30.60</td><td>46.64</td><td>43.6</td><td>10.9</td></tr><tr><td>Hybrid-hgrn2</td><td>29.5K</td><td>70.89</td><td>51.23</td><td>56.51</td><td>59.68</td><td>28.50</td><td>32.40</td><td>49.87</td><td>91.8</td><td>10.8</td></tr><tr><td>Hybrid-lightning</td><td>33.4K</td><td>70.73</td><td>50.41</td><td>55.80</td><td>59.93</td><td>27.65</td><td>32.80</td><td>49.55</td><td>95.7</td><td>13.3</td></tr></table>

In addition to linear models, sliding window attention can also achieve linear computational complexity by appropriately adjusting the window size. As it is grounded in softmax attention, it serves as a robust baseline for evaluating linear architectures. Therefore, we incorporated the hybrid-window approach by replacing the sliding window attention with full softmax attention every eight layers. We evaluated various window sizes of SWA ranging from 256 to 1024. Our results indicate that larger window sizes lead to slower training speeds compared to the hybrid-lightning model. To compare these models under equivalent speed conditions, we did not consider window sizes larger than 1024. As shown in Table 4, the hybrid-lightning model outperforms all other models across all metrics, particularly excelling in the NIAH benchmark.

Table 4 | Benchmark comparison of hybrid-lightning and hybrid-window Models. Metrics include average CSR score, weighted NIAH accuracy, and average SCROLLS score. Higher scores indicate better performance across all tasks. Abbreviations: PS (parameter size, billion), W.S. (window size of SWA), HS (HellaSwag), WG (WinoGrande), OBQA (OpenBookQA), NIAH, SCR (SCROLLS), TGS (token per gpu per second).   

<table><tr><td>P.S</td><td>Arch.</td><td>W.S.</td><td>TGS ↑</td><td>PIQA↑</td><td>HS↑</td><td>WG↑</td><td>ARC-E↑</td><td>ARC-C↑</td><td>OBQA↑</td><td>CSR ↑</td><td>NIAH ↑</td><td>SCR↑</td></tr><tr><td rowspan="4">1B</td><td rowspan="3">Hybrid-window</td><td>256</td><td>35.6K</td><td>70.29</td><td>48.68</td><td>53.35</td><td>57.95</td><td>28.75</td><td>32.60</td><td>48.61</td><td>46.8</td><td>10.6</td></tr><tr><td>512</td><td>35.1K</td><td>70.95</td><td>48.19</td><td>52.33</td><td>57.53</td><td>27.22</td><td>30.00</td><td>47.70</td><td>25.7</td><td>11.9</td></tr><tr><td>1024</td><td>33.6K</td><td>69.75</td><td>47.80</td><td>53.12</td><td>57.53</td><td>28.33</td><td>31.60</td><td>48.02</td><td>53.9</td><td>10.6</td></tr><tr><td colspan="2">Hybrid-lightning</td><td>33.4K</td><td>70.73</td><td>50.41</td><td>55.80</td><td>59.93</td><td>27.65</td><td>32.80</td><td>49.55</td><td>95.7</td><td>13.3</td></tr><tr><td rowspan="4">3B</td><td rowspan="3">Hybrid-window</td><td>256</td><td>16.1K</td><td>73.83</td><td>59.70</td><td>59.59</td><td>64.10</td><td>33.62</td><td>35.00</td><td>54.31</td><td>40.9</td><td>14.2</td></tr><tr><td>512</td><td>15.8K</td><td>73.29</td><td>60.00</td><td>59.04</td><td>62.96</td><td>32.51</td><td>36.00</td><td>53.97</td><td>57.9</td><td>14.2</td></tr><tr><td>1024</td><td>15.4K</td><td>74.27</td><td>59.02</td><td>57.85</td><td>64.56</td><td>31.91</td><td>33.00</td><td>53.44</td><td>41.6</td><td>13.3</td></tr><tr><td colspan="2">Hybrid-lightning</td><td>15.1K</td><td>74.21</td><td>61.06</td><td>59.51</td><td>65.49</td><td>34.90</td><td>35.80</td><td>55.16</td><td>98.0</td><td>14.7</td></tr></table>

# 2.2.4. Discussion

Based on our analysis of scaling law experiment, downstream performance and speed comparison, we conclude that while pure linear attention models are computationally efficient, they are not suitable

for LLMs. This is due to their inherent inability to perform retrieval, a capability that is essential for in-context learning. In contrast, our hybrid model not only matches but also surpasses softmax attention in both retrieval and extrapolation tasks. This outcome is somewhat counterintuitive. To understand this phenomenon, consider the following explanation of softmax attention:

$$
\mathbf {O} = \operatorname {S o f t m a x} \left(\mathbf {Q K} ^ {\top} / \sqrt {d}\right) \mathbf {V}. \tag {10}
$$

It can be rewritten into a linear recurrent form as:

$$
s _ {t} ^ {0} = 0, \quad s _ {t} ^ {j} = s _ {t} ^ {j - 1} + \exp \left(\mathbf {q} _ {t} \mathbf {k} _ {j} ^ {T} / \sqrt {d}\right), \quad \mathbf {o} _ {t} ^ {j} = \left(s _ {t} ^ {j - 1} / s _ {t} ^ {j}\right) \mathbf {o} _ {t} ^ {j - 1} + \left(1 - s _ {t} ^ {j - 1} / s _ {t} ^ {j}\right) \mathbf {v} _ {j}, \quad \mathbf {o} _ {t} = \mathbf {o} _ {t} ^ {t}, j = 1, \dots , t. \tag {11}
$$

Note that the linear recurrence form of lightning attention is as follows:

$$
\mathbf {k} \mathbf {v} _ {0} = 0, \quad \mathbf {k} \mathbf {v} _ {j} = \mathbf {k} \mathbf {v} _ {j - 1} + \mathbf {k} _ {j} \mathbf {v} _ {j} ^ {\top} \quad \mathbf {o} _ {j} = \mathbf {k} \mathbf {v} _ {j} ^ {\top} \mathbf {q} _ {j}, j = 1, \dots , t. \tag {12}
$$

The softmax attention mechanism can be interpreted as a linear RNN (Qin et al., 2024a). At each time step ??, the hidden state is recalculated starting from the initial time $t _ { 0 } = 1$ , a process often described as "Going Through a Book." This method enables the model to accurately retain input information by systematically revisiting previous data. In contrast, linear models lack this recomputation process, which hinders their ability to effectively retain input data.

Let us define the capacity of an RNN as the size of its recurrent state. Upon closer examination of Eq. 11, we can deduce that the capacity of softmax attention is $O ( d )$ . In contrast, as illustrated in Eq. 12, the capacity of lightning attention is $O ( d ^ { 2 } / h )$ . Given that $d > h$ , it follows that lightning attention possesses a larger capacity than softmax attention. Consequently, the hybrid-lightning model exhibits superior retrieval and extrapolation capabilities compared to models relying solely on softmax attention.

# 2.3. Module Ablations in MoE

Based on the conclusions from previous sections, we conduct two additional sets of ablation experiments to validate module choices within the MoE architecture on a larger scale: (1) Hybrid-lightning attention versus softmax attention: To verify the advantages of the hybrid lightning attention in the MoE. (2) Pre-Layer Normalization versus Post-Layer Normalization: In our hybrid architecture, the effective depth of the model plays a significant role. Thus, we expect to find a better normalization algorithm for the deep model.

Hybrid-lightning Attention versus Softmax Attention. We perform a small-scale comparative analysis between softmax attention and hybrid-lightning attention within the MoE architecture. Specifically, we use a 28 billion parameter MoE with 5 billion activation parameters that utilize softmax attention as the base model. For every 8 consecutive layers in the base model, we systematically replace softmax attention with lightning attention in the first 7 layers. Both the base model and the modified model are trained on 1 trillion tokens. As shown in Table 5, the results reveal that substituting certain softmax attention layers with lightning attention improves accuracy across most benchmarks.

Pre Layer Normalization versus Post Layer Normalization. Pre Layer Normalization(Baevski and Auli, 2018; Child et al., 2019; Wang et al., 2019) (PreNorm), which applies normalization layers before residual connections and attention mechanisms, has demonstrated enhanced stability and performance in LLMs. Since PreNorm allows gradients to flow more directly from the output to the input through residual connections, bypassing the sub-layers to a certain extent, it reduces the effective depth of the model. In contrast, Post Layer Normalization(Wang et al., 2019) (PostNorm) applies normalization after the residual connection and attention mechanisms, thereby preserving

the model’s effective depth. However, PostNorm can be prone to vanishing and exploding gradients, presenting significant challenges in training LLMs. Most existing LLMs predominantly use PreNorm, as the performance differences between wider and deeper networks in the conventional Transformer architecture are often negligible, and training stability is prioritized.

The experiments are performed on models with 9.3 billion activation parameters and a total of 60 billion parameters, each consisting of 48 layers that employ different normalization methods. Both models are trained on 500 billion tokens. For PostNorm, we utilize DeepNorm (Wang et al., 2024a) to ensure more stable training. As illustrated in Table 5, PostNorm consistently outperforms PreNorm across all evaluated metrics.

Table 5 | Module Ablations. Abbreviations: BBH (BIG-Bench Hard), DROP (Discrete Reasoning Over Paragraphs), MMLU (Massive Multitask Language Understanding), CMMLU (Massive Multitask Language Understanding in Chinese), GSM8k (Grade School Math 8K), ARC-C (Arc-Challenge), WG (WinoGrande).   

<table><tr><td>Arch.</td><td>BBH ↑</td><td>DROP ↑</td><td>MMLU ↑</td><td>CMMLU ↑</td><td>MATH ↑</td><td>GSM8k ↑</td><td>ARC-C ↑</td><td>WG ↑</td></tr><tr><td>Softmax</td><td>28.2</td><td>27.4</td><td>49.3</td><td>47.3</td><td>4.6</td><td>18.8</td><td>46.4</td><td>65.6</td></tr><tr><td>Hybrid-lightning</td><td>32.2</td><td>29.0</td><td>49.5</td><td>46.0</td><td>6.8</td><td>18.5</td><td>47.4</td><td>67.5</td></tr><tr><td>Pre Layer Norm.</td><td>29.9</td><td>26.8</td><td>43.9</td><td>41.8</td><td>4.8</td><td>12.2</td><td>43.5</td><td>65.5</td></tr><tr><td>Post Layer Norm.</td><td>32.6</td><td>27.6</td><td>50.2</td><td>49.2</td><td>5.7</td><td>16.8</td><td>46.2</td><td>65.4</td></tr></table>

# 2.4. Model Spec

Upon finalizing the architecture of the model’s modules, the subsequent step entails scaling up the model, which necessitates a meticulous design of the model’s hyperparameters across various dimensions. Our primary goal is to strike a balance between performance and inference efficiency. Single-device inference offers superior efficiency compared to multi-device implementations by eliminating cross-machine communication overhead. Consequently, we constrain the model’s total parameters to 500B, ensuring compatibility with single-node inference on an $8 \times 8 0 \mathrm { G }$ configuration for sequences up to 1M tokens under 8-bit quantization. Given our limited training budget, we formulate the following optimization problem to determine optimal parameter allocations:

$$
\min  _ {P _ {\text {a l l}}, P _ {\text {a c t}}} L \left(P _ {\text {a l l}}, P _ {\text {a c t}}, T\right) \quad \text {s u b j e c t} \quad C _ {\text {c o m p u t e}} \left(P _ {\text {a l l}}, P _ {\text {a c t}}, T\right) <   C \quad \text {a n d} \quad P _ {\text {a l l}} <   5 0 0 B, \tag {13}
$$

where ?? denotes the loss, $P _ { \mathrm { a l l } }$ and $P _ { \mathrm { a c t } }$ represent the total and activation parameter counts respectively, ?? is the number of training tokens, ??compute denotes the computational costs (dependent on parameter counts and data consumption), and ?? signifies the budget constraint.

Through comparative experiments on small-scale models, we first establish optimal ranges for several key variables: (1) the mixing ratio between softmax and linear attention mechanisms; (2) the depth-to-width ratio of the model architecture; (3) the ratio of linear attention memory size to hidden size; (4) the ratio of activated FFN to attention; (5) the proportion of dimensions utilizing RoPE for softmax attention.

Our experiments reveal that the hybrid architecture demonstrates particular sensitivity to layer depth, with deeper models consistently outperforming shallower counterparts. Notably, shallow models require substantially more softmax attention layers to achieve comparable performance, underlining the efficiency advantages of deeper architectures. We also observe that increasing linear attention memory size significantly enhances model performance, and implementing RoPE on half of the softmax attention dimensions enables length extrapolation without performance degradation.

Based on these optimized architectural variables, we employ established scaling laws (Clark et al., 2022; Hoffmann et al., 2022) to determine the optimal model size. We train models with activation parameters ranging from 44 million to 1.2 billion across 500 billion tokens, utilizing 16, 32, and 64 experts. However, we find the predictions from these methods become less reliable when extrapolating to a larger model with 9.3 billion parameters. To address this limitation and achieve more accurate predictions, we propose the following formula:

$$
L \left(P _ {\mathrm {a c t}}, T \mid E\right) = d + a P _ {\mathrm {a c t}} ^ {\alpha} + b T ^ {\beta} + c \left(P _ {\mathrm {a c t}} T\right) ^ {\gamma}, \tag {14}
$$

where $L ( P _ { \mathrm { a c t } } , T | E )$ represents the loss conditioned on the number of experts, while $a , b , c , d , \alpha , \beta$ , and ?? are parameters to be fitted in relation to the number of experts. Based on the predictions of Eq. 13 and Eq. 14, we have identified a candidate model with 45.9 billion activation parameters and 456 billion total parameters as the optimal configuration.

# 3. Computation Optimization

In this section, we present our computation part, including the training and inference. In this project, we have a dynamically changing GPU cluster, where the number of H800 GPUs ranges from 1500 to 2500. An efficient architecture necessitates robust implementation optimization to fully harness its computational benefits at scale. To scale our novel architecture to the requisite size, we present three key optimization strategies that primarily address the following three challenges:

1. Mitigating the all-to-all (a2a) communication overhead during the training of a Mixture of Experts (MoE) architecture is a persistent challenge. The configuration we choose for our experts, specifically opting for large models, imposes substantial demands on GPU memory. Therefore, the primary challenge lies in achieving an optimal equilibrium between memory utilization, computational efficiency, and the overhead associated with all-to-all communication.   
2. As we endeavor to support at least 1 million token context window in both training and inference, the accurate distribution of tokens within such an extensive context window across different GPUs becomes imperative for this colossal model. This necessity, however, inevitably introduces additional communication overhead. As a result, devising strategies to minimize this overhead, particularly in the context of our hybrid architecture, presents a significant challenge.   
3. The current implementation of the lightning attention mechanism is specifically optimized for training processes. However, in the inference scenario, the challenge arises in effectively managing real-world batched inputs, which may encompass variable sequence lengths and specific inputs that incorporate prefix caching.

It is noteworthy that the existing open-source frameworks in the industry currently lack the necessary mature technical support to adequately address these challenges. Thus, we independently and comprehensively reinvent our distributed training and inference framework, thereby successfully addressing these challenges with the desired level of efficiency.

# 3.1. MoE Optimization

The primary objective in optimizing the MoE architecture is to minimize communication overhead, particularly for MoE models that utilize all-to-all (a2a) communication. To address this, We implement a token-grouping-based overlap scheme, as illustrated in Figure 9. In this scheme, the a2a communication is performed within the expert parallel (EP) communication group, and it overlaps with the processing of tokens from different expert groups. To ensure the correctness of the communication

results, we restrict each ProcessGroup to execute communication operators sequentially. As a result, a2a communications across different groups cannot overlap, leading to the emergence of idle time.

This approach leads to significant performance improvements. However, upon more detailed analysis, we identified a critical tradeoff specific to the expert configuration of the MiniMax-Text-01 model. When Tensor Parallelism (TP) is employed to partition the expert parameters, the computational intensity becomes excessively low, thereby hindering the efficiency of the computation. However, opting not to use TP leads to an excessively large parameter count, which necessitates the activation of a larger Pipeline Parallelism (PP) configuration. The challenge emerges because PP does

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/ce72a6c89e2b8686f17fdea7157f82c40770450d834fbb8ec3c1e6202be58044.jpg]]  
Figure 9 | Expert Parallel (EP) Overlap Illustration. Chunk tokens into 2 groups thus computation can overlap with communication between different groups.

not reduce the memory footprint required for storing activations. This limitation is particularly detrimental for training models with long contexts, as the increase in memory consumption does not provide proportional benefits in terms of computational efficiency or training speed. Consequently, it is imperative to develop a new parameter partitioning strategy that adeptly balances memory usage and computational intensity to optimize the training process for our specific model and task.

To achieve enhanced efficiency, we first introduce a novel ProcessGroup, termed ETP (Expert Tensor Parallel), which is specifically designed to manage the weight partitioning of experts. Concurrently, we propose another distinct ProcessGroup, named EDP (Expert Data Parallel), to encapsulate the data parallelism of identical experts. In our system, we define the total number of GPUs involved in training as ??????????_????????. The system must satisfy two key conditions:

$$
w o r l d \_ s i z e = s i z e _ {P P} \times s i z e _ {D P} \times s i z e _ {C P} \times s i z e _ {T P} \tag {15}
$$

and

$$
w o r l d \_ s i z e = s i z e _ {P P} \times s i z e _ {E D P} \times s i z e _ {E T P} \times s i z e _ {E P} \tag {16}
$$

This configuration empowers the MoE component with the flexibility to define the distribution of experts, manage the weight partitioning of experts, and independently configure the ZeRO (Zero Redundancy Optimizer) algorithm (Rajbhandari et al., 2020). Based on this implementation, we are able to completely decouple the parallel strategies of the MoE components from those of the non-MoE components.

Building upon this modification, we can flexibly configure the ETP to achieve an optimal balance between memory usage and computational intensity. Furthermore, to mitigate communication overhead, we design an EP-ETP overlap strategy. This strategy aims to maximize the utilization of both network resources and computational resources, as illustrated in Figure 10 (a).

Since communications within the same process group must be executed sequentially, extended periods of computation not only facilitate overlap with a greater number of communications but also create additional opportunities for communications across different process groups to overlap, leading to enhanced overall performance as illustrated in Figure 10 (b).

When determining the number of groups, several trade-offs must be considered. Theoretically, only by dividing the workload into a sufficiently large number of groups can we achieve ample overlap between communication and computation, as illustrated in Figure 10 (c). However, in practice, an excessive number of groups can significantly increase the complexity of scheduling and introduce the

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/8c91d31f6be717486fc91e35da5f88fb1b0a3c0ee1f2c12173670b2db5ae7438.jpg]]  
Figure 10 | EP-ETP Overlap Illustration. (a) EP-ETP overlap with the lower computation portion. (b) EP-ETP overlap with the higher computation portion. (c) EP-ETP overlap with fewer groups. Compared with (a) and (b), it shows that if the compute time cost is longer, the efficiency will be better. Comparing with (b) and (c), it shows that fewer groups will lead to insufficient overlap.

risk of becoming CPU-bound. Given that the proportion of ETP (Expert Tensor Parallel) in the overall MoE (Mixture of Experts) architecture is not substantial, it is crucial to make adjustments based on the specific context and requirements.

Through the aforementioned optimization strategies, we achieve a balanced configuration of storage and computational intensity for the specific expert specifications in the MoE (Mixture of Experts) structure of the MiniMax-Text-01 model. Furthermore, based on these optimizations, we reduce the pure communication overhead of the MoE component by $5 0 \%$ compared to the preoptimization state, resulting in a significant improvement in training efficiency.

# 3.2. Long Context Optimization

A significant challenge in long context training is that real training samples are difficult to standardize into a uniform length. The conventional approach of using padding to make samples the same length leads to substantial computational waste. In the context of training at the 1M sequence length scale, this waste becomes particularly significant. To address this issue, we adopt a data formatting technique during training where different samples are concatenated end-to-end along the sequence dimension. We refer to this technique as "data-packing". This format minimizes computational waste during the computation process, thereby conserving computational resources.

# 3.2.1. Varlen Ring Attention

For Softmax Attention, the ring attention algorithm (Liu et al., 2024a) offers an effective method to partition data, thereby enabling unlimited scalability. However, the existing implementations

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/d1097a12a2a5f0f1b0208778f8b73540b6322b8f94ceb24cbb3f951828d43ed1.jpg]]  
(a)

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/b621b61c8e24a05bc4eeea2066562e2c5561cd0741078a2affba73050a4359ee.jpg]]  
  
Figure 11 | Ring Attention v.s. Varlen Ring Attention. (a) No data packing in ring attention. (b) Pack 3 samples with different lengths in varlen ring attention.

are not optimized to efficiently handle the ring attention mechanism for the data-packing format. In the case of FlashAttention (Dao, 2024), while it provides a varlen (variable length) interface to accommodate the data-packing format, there is no corresponding ring attention implementation available. Regarding TransformerEngine (NVIDIA, 2023), the implementation incorporates a Context Parallel (CP) ProcessGroup to support the ring attention algorithm. However, this approach poses a risk of computational resource waste when dealing with the data-packing format. This is because the algorithm divides each sequence into $2 \times s i z e _ { C P }$ segments and applies the ring attention mechanism to each segment. Consequently, this approach restricts each sequence to a length that must be an integer multiple of $2 \times s i z e _ { C P }$ . In scenarios where the sample distribution is unknown and the CP size is set to a large value, this can lead to significant padding, resulting in the waste of computational resources.

Motivated by the principle of not making assumptions about the sample distribution, we redesign the algorithm and name it Varlen Ring Attention. This approach avoids the excessive padding and subsequent computational waste associated with traditional methods by applying the ring attention algorithm directly to the entire sequence after data-packing. Specifically, the implementation involves distinguishing the offset of the attention mask corresponding to each sequence within the ring attention computation. The key modification is to transform the original causal computations into varlen causal computations and similarly convert the non-causal computations into varlen non-causal computations, shown in Figure 11.

# 3.2.2. Improved Linear Attention Sequence Parallelism

For lightning attention, the LASP (Linear Attention Sequence Parallelism) algorithm (Sun et al., 2024) leverages the communication group of CP to facilitate the expansion of long sequences. As illustrated in Figure 12 (a), the LASP algorithm mandates that all CP ranks engage in send-recv operations to exchange intermediate key-value $( K V )$ block results. This requirement imposes a sequential dependency among the CP ranks, thereby compelling the computation to be performed in a serial manner. Consequently, this sequential dependency significantly impedes the overall efficiency of the training process, as the inherent parallelism of the system is not fully exploited.

To fully harness the parallel computing capabilities of GPU devices, we propose an optimized approach that refines the computational and communication workflow to eliminate dependencies during the computation process. This optimization effectively transforms serial computation into a parallelized one. The enhanced approach, termed LASP $+$ (Figure 12 (b)), operates as follows:

1. Local Prefix Sum Calculation: Each computing node i.e., the CP rank, initiates the process by

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/4f59321cd968b8872e7b37babd8ea71676cfe1473422de1e0f6409051e0ab589.jpg]]  
Figure 12 | Difference of LASP Algorithm and $\mathbf { L A S P + }$ $^ +$ Algorithm. (a) LASP Algorithm. 1. Initialization Phase: initializing KV to zero and the diagonal decay matrix. 2. Data Partitioning and Padding: partitioning the Q, K, and V matrices along the sequence dimension into CP size (4 segments illustrated in the figure) blocks, dividing each block into smaller blocks based on the BlockSize ?? and padding the remaining part (e.g. Q7, K7, V7) that cannot be divided evenly by ??. 3. Intra-block Computation: performing intra-block of each CP rank computations in parallel. 4. Inter-block Computation and Communication: starting from CP rank 0, computing the inter-block portion of the current $Q _ { i }$ with all previous KV blocks and the prefix sum $K _ { i } V _ { i }$ . Different CP ranks communicate data through send-recv operations. (b) $\mathrm { L A S P + }$ Algorithm. Building upon figure (a), each CP rank computes the local prefix sum $K V _ { L }$ and performs AllGather operation to synchronize, then selects the local prefix sum $K V _ { L }$ to compute the global prefix sum $K V _ { G }$ . The remaining computational components are same as (a).

independently calculating its local prefix sum, denoted as $K V _ { L }$

2. Global Synchronization via AllGather: Following the local calculations, an AllGather operation is performed to synchronize the information from all nodes globally. This step ensures that each node has access to the necessary data from all other nodes.   
3. Prefix Sum Computation: Each node selects the specific CP rank’s $K V _ { L }$ on which to perform the prefix sums, a decision based on its assigned computation order.

By implementing these steps, the $\mathrm { L A S P + }$ approach effectively removes the original dependencies between the computation nodes. This elimination of dependencies facilitates a fully parallelized computation process, thereby significantly enhancing the overall efficiency and throughput of the system. The transformation from serial to parallel computation not only leverages the full potential of GPU devices but also ensures that the training process can be executed more rapidly and with greater scalability.

The proposed modifications, while incurring additional costs in terms of increased total communication volume and temporary memory usage, are unequivocally justified by the substantial performance benefits they confer. These enhancements significantly outweigh the associated overhead in communication and memory consumption.

Through comprehensive testing and verification, it is empirically demonstrated that the computation speed in the $\mathrm { L A S P + }$ approach can attain up to $1 / N _ { p c n }$ of the original LASP algorithm, where $N _ { p c n }$ denotes the number of parallel computing nodes. Furthermore, the overhead introduced by the AllGather operation is minimal, which is consistent with our anticipations and underscores the efficacy of the optimization.

Building upon the $\mathrm { L A S P + }$ framework, we further introduce support for the varlen feature to effectively manage the data-packing data structure. This enhancement is particularly beneficial for handling batched samples that comprise inputs with unequal token lengths. The process involves the following steps: 1). Padding to Block Size: Each input within the batch is padded to ensure that its length is a multiple of the predefined block size, which is set to 256. This padding step is crucial for aligning the data structure with the computational requirements of the kernel. 2). Sequential Concatenation: After padding, the inputs are sequentially concatenated. This concatenation facilitates the use of a single kernel to perform parallel computations across multiple batches. By organizing the data in this manner, we can efficiently leverage the parallel processing capabilities of the GPU, thereby optimizing computational performance.

The integration of the varlen feature with the $\mathrm { L A S P + }$ framework ensures that the system can handle diverse input lengths without compromising on efficiency. This approach not only simplifies the computational workflow but also maximizes resource utilization by enabling the processing of multiple batches concurrently.

# 3.3. Lightning Attention Inference Optimization

The initial implementation of the lightning attention mechanism is primarily research-oriented and not yet suitable for practical applications, especially for inference. However, the optimization of inference processes is of paramount importance in real-world scenarios, as the long-term cost of deploying a trained model is predominantly determined by the efficiency of its inference. To this end, we implement four optimization strategies for lightning attention: batched kernel fusion, separated prefill and decoding execution, multi-level padding, and strided batched matmul extension.

# 3.3.1. Batched Kernel Fusion

We fuse multiple memory-bound kernels and extend support to accommodate all batch inputs. In the prefill phase, we perform a kernel fusion for processing the $Q , K$ , and ?? tensors, including padding in the sequence dimension, partitioning into blocks, adjusting the internal layout, and computing the decay values. In the decoding phase, we perform a kernel fusion for the computation of ???? and the updating of the prefix ???? cache. These kernel fusions reduce intermediate result storage and memory access operations, thereby significantly improving memory access efficiency and reducing end-to-end latency by $\mathbf { 1 0 \% }$ in the decoding phase and short-text input scenarios. By the way, these optimizations can bring very noticeable benefits on H20 compared to H800.

# 3.3.2. Separated Prefill and Decoding Execution

The implementation of the lightning attention mechanism for long sequence computations primarily revolves around the differentiation between intra-block and inter-block computations. However, this approach is not optimal for inference tasks, particularly in the decoding phase, where the token length is consistently equal to 1.

Given that the computational kernel for tokens of length 1 is predominantly memory-bound and necessitates only a limited number of GPU Streaming Multiprocessors (SMs), we propose a strategy that segregates the processing of tokens with a length of 1 from those with a length greater than 1. This is achieved by employing two distinct kernels. Subsequently, we utilize two separate CUDA streams to schedule these kernels in parallel, thereby enhancing computational efficiency and ensuring balanced GPU utilization, especially in scenarios involving mixed inputs.

For instance, in a batch size of 20, where all inputs contain a prefix key-value (KV) cache, and the scenario includes one or two inputs with a token length of 50 while the remaining inputs have a token length of 1, this approach can significantly reduce latency. Specifically, the latency can be approximately equivalent to that of processing only the longer inputs, demonstrating a reduction from 100 milliseconds to 50 milliseconds.

# 3.3.3. Multi-level Padding

By applying padding to the $Q , K , V$ tensors along the sequence dimension, the intra-block and interblock components can be effectively decomposed into multiple identical matrix multiplications. This decomposition is particularly advantageous as it aligns seamlessly with the StrideBatchedMatmul interface, thereby facilitating the maximization of parallel processing capabilities.

Initially, the block size for padding was set to 256, a configuration that was consistent with the training parameters. However, upon the implementation of the prefix cache technique, it is observed that the token lengths within a batch typically fall below 256. This discrepancy led to redundant computations within each matrix multiplication operation. To address this inefficiency and minimize unnecessary computations, we propose the introduction of additional segmentation options, specifically 32, 64, and 128.

This multi-level padding approach enables the dynamic selection of the computational scale that incurs the minimal padding overhead, based on the current input sequence length. By adopting this approach, the utilization of computational resources is optimized, ensuring that the system operates with increased efficiency and reduced redundancy. This strategic adjustment not only conserves computational resources but also contributes to the overall performance enhancement of the system.

# 3.3.4. StridedBatchedMatmul Extension

We utilize the optimized function cublasGemmStridedBatchedEx from the NVIDIA cuBLAS Library to manage StridedBatchedMatmul operations, thereby ensuring both high performance and versatility across diverse hardware architectures. Concurrently, we are in the process of implementing a more extensive kernel fusion strategy, with the objective of substantially improving the computational efficiency of Hopper GPUs.

Given that our sequence partitioning block size is configured to 256, the associated General Matrix-Matrix Multiplication (GEMM) operations, which involve matrices of dimensions 256x256, can leverage warpgroup-wide WGMMA instructions for computation. To further enhance memory access efficiency, we integrate the asynchronous operations of the Tensor Memory Accelerator (TMA) and delegate certain preprocessing and postprocessing computational tasks to be executed asynchronously on the CUDA Cores.

Ultimately, our goal is to dynamically regulate the number of pipeline stages to adaptively attain optimal performance across both H20 and H800 GPU architectures. This adaptive control mechanism will ensure that the system can efficiently handle varying workloads and hardware configurations, thus maximizing overall computational throughput and resource utilization.

By implementing the aforementioned optimizations, we achieve a Model Flops Utilization (MFU) exceeding $7 5 \%$ on the H20 GPU for end-to-end inference tasks (Chowdhery et al., 2023). Specifically, in our MiniMax-Text-01 and MiniMax-VL-01 inference, when considering the latency ratio between the attention operation and the Feed-Forward Network (FFN) operation within the MoE structure, the softmax attention constitutes $9 5 \%$ of the latency at a sequence length of 1,024,000 tokens. In contrast, the lightning attention implementation contributes to less than $1 2 \%$ of the latency under the same conditions.

Our lightning attention implementation exhibits remarkable efficiency in managing heterogeneous batch inputs, which are characterized by diverse sequence lengths. This efficiency is particularly evident in scenarios where some inputs incorporate the prefix caching strategy while others do not. The reduction in latency not only enhances the overall speed of the inference process but also ensures that the system can handle a wide range of input types with minimal performance degradation. This adaptability underscores the robustness and versatility of our lightning attention approach in real-world applications.

# 4. Pre-Training

In this section, we provide an overview of the pre-training methodology for MiniMax-Text-01. First, we detail the meticulous construction of our pre-training corpus, with particular emphasis on data quality, standardized formatting, and mixing strategies to maximize model performance. Subsequently, we outline our innovative data experimentation framework, which enables rapid and resource-efficient evaluation of data effectiveness while minimizing computational costs. Lastly, we present an in-depth analysis of the model’s training hyper-parameters and present a hierarchical training approach, which enables context length scaling up to 4 million tokens.

# 4.1. Data

# 4.1.1. Pre-training Corpus

The pre-training corpus for MiniMax-Text-01 encompasses a comprehensive and meticulously curated dataset, incorporating diverse sources including academic literature, books, web content, and

programming code. We enhance corpus quality through several strategic dimensions:

• Data Quality Enhancement. Superior data quality is fundamental for Large Language Models. We implement a sophisticated filtering pipeline, combining rule-based cleaning and deduplication procedures aligned with established practices (Penedo et al., 2023, 2024; Rae et al., 2021). To assess document quality at a granular level, we utilize our previous-generation model as the reward labeler (a MoE model with 5B activations and 60B total parameters). Initially, we evaluate multiple quality dimensions including coherence, conciseness, educational value, helpfulness, knowledge richness, and categorical relevance. Through comprehensive analysis, we identify significant correlations among these metrics and ultimately focus on three key dimensions: knowledge depth, practical helpfulness, and categorical distribution, while maintaining other metrics as secondary validation indicators.

• Data Formatting Optimization. The content from websites and books, once appropriately extracted and cleaned, can naturally be used as high-quality textbooks (Gunasekar et al., 2023) without further formatting. For dialogue and question-answering data, the sequential nature of text inherently captures conversational logic and question-answer relationships. Although humans benefit from additional formatting (e.g., Markdown) for readability and comprehension, we find that heavy formatting can actually diminish data diversity and quality by introducing fixed patterns that constrain the natural variation present in human conversations. Ultimately, to maintain format generalization capabilities and accommodate human preferences in alignment, we implement a nested document format with versatile templates for dialogue and QA data, carefully balancing natural comprehension with structural consistency across various interaction patterns.

• Data Mixture Investigation. We develop a sophisticated approach to tuning the data distribution, leveraging our three primary quality metrics. Based on the experiment paradigm detailed in the subsequent section, we discover that while high-scoring content on knowledge depth and helpfulness generally yielded superior performance in capability assessments, completely eliminating lower-scoring content can adversely affect downstream task performance. Therefore, we implement a balanced sampling strategy, beginning with a uniform distribution across the base corpus, and then adjusting sampling weights to favor high-quality content while maintaining sufficient representation of diverse categories.

# 4.1.2. Tokenization

For tokenization, we employ byte-level Byte Pair Encoding (BPE) (Brown et al., 2020; Shibata et al., 1999), incorporating the pre-tokenizer methodology. We strategically up-sample multilingual content, to enhance the corresponding compression efficiency. The resulting vocabulary size is set to 200K tokens.

# 4.1.3. Data Experiment

To systematically evaluate our design choices regarding pre-training data quality, format, and composition, we conduct extensive ablation experiments. These experiments involve training multiple small-scale MoE models using comparable token quantities but varying data characteristics. This approach enables us to isolate and measure the impact of individual data attributes while maintaining computational efficiency.

# 4.1.3.1 Paradigm

Formulation. We conduct Data Experiments to systematically compare the performance of different model variants. Specifically, we formulate experiments as statistical hypothesis tests that compare evaluation metric distributions between a baseline model and models trained with different data configurations. When testing the effectiveness of a new data corpus $\mathcal { D }$ , we formulate our alternative hypothesis as $H _ { 1 } : \mu _ { T _ { \mathcal { D } } } > \mu _ { T _ { \mathrm { b a s e l i n e } } }$ , where $\mu$ represents the weighted average performance metric and ?? denotes the distribution of evaluation values across test samples.

Evaluation. We carefully design our evaluation norms to ensure meaningful insights. We look at a wide range of multiple-choice benchmarks, discarding choice indices in query formulation and look at the likelihoods of completion. We observe the distributions of sample-wise log-normalized accuracy $\log \operatorname { a c c } _ { \mathrm { n o r m } ^ { 2 } }$ , defined as

$$
\log \operatorname {a c c} _ {\text {n o r m} ^ {2}} (x) = \log \operatorname {s o f t m a x} _ {p ^ {\prime} (c \in C _ {x})} \left\{\left(p ^ {\prime} \left(c ^ {*}\right)\right) \right\},
$$

where $\begin{array} { r } { p _ { i } ^ { \prime } ( c ) = \frac { p _ { i } ( c ) } { \mathrm { b y t e s } ( c ) } } \end{array}$ is the byte-normalized probability of choice ?? for sample ??. We choose bytewise normalization to exclude the effect of tokenizer, while alleviating the disfavor towards longer choices. We conduct extensive experiments to ensure that this metric is stable across training, while maintaining the discriminative power of the metric, which is quantified by the ratio $\Delta _ { \mathrm { o b v i o u s } } / \sigma _ { \mathrm { s e e d } }$ where Δobvious represents the obvious difference in performance between models and $\sigma _ { \mathrm { { s e e d } } }$ denotes the standard deviation across different random seeds.

Experiment Efficiency & Setup. With such statistical setup, we are able to conduct a power analysis to decide minimal test sample size while maintaining the MDE (Minimal Detectable Effect) at a similar level as our training variance, and guaranteeing $9 5 \%$ confidence level and $8 0 \%$ power for decision making. With the confidence methodologies set, we conduct simple scaling experiments on token amount and the model size, and eventually land at an experiment step of training MoEs of 1B activation and 8B total parameters with 40B tokens of data, where data mixture comprises 20B web documents and 20B data of hypothesis.

# 4.1.3.2 Effect of Repetition

The incorporation of repeated data has been empirically demonstrated to introduce several detrimental effects on the model’s performance and generalization capabilities (Hernandez et al., 2022). Consequently, implementing deduplication strategies is essential for optimizing LLM performance. Recent studies (Abdin et al., 2024; Penedo et al., 2024) suggest that repeatedly training high-quality documents can lead to enhanced downstream performance, with certain high-quality domains being trained up to 50 times, where the repetition is measured by MinHash similarity(Broder, 1997; Lee et al., 2022). However, our empirical analysis reveals that their experimental paradigm is inadequate for assessing the impact of repetition, as data efficiency is not consistent throughout the training process.

To achieve better alignment with the results of the full training, we introduce a novel repetitionaware experimental framework. Specifically, we first perform global deduplication on the dataset to remove redundant entries. Then, we down-sample the documents to align the repetition frequency with the requirements of the final training schedule while adhering to the budget constraints of our ablation experiments, different from the previous experimental setups which directly adopted data distributions identical or similar to those used in the final training stage. Our findings indicate that low-quality data suffer a substantial decrease in performance after training for more than two epochs, while high-quality data can be effectively trained for up to four epochs, similar to previous

observations (Muennighoff et al., 2023). Notably, the solution derived from the proposed framework yields better alignment with the results obtained using considerably more computational resources. By carefully controlling the repetition and quality of the training data, we achieve a more efficient and effective data mixture, ultimately leading to better model performance.

# 4.2. Training Strategy

Initial Pre-training. We initialize all model parameters using the Xavier initialization method (Glorot and Bengio, 2010), the scaling factors of DeepNorm (Wang et al., 2024a) are set to $\alpha = ( 2 N ) ^ { 0 . 2 5 }$ and $\beta = ( 8 N ) ^ { - 0 . 2 5 }$ , where $N$ denotes the number of layers. We employ the AdamW optimizer (Loshchilov and Hutter, 2019) with $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 5$ , and the weight decay is set to 0.1. The training sequence length is 8192, and the batch size is progressively scaled from an initial size of 16M to 32M at 69B tokens, to 64M at 790B tokens, and finally to 128M at 4.7T tokens, where it remains until the end of training. The schedule is designed based on the correlation between training loss and the critical batch size (McCandlish et al., 2018). It is argued that training at the critical batch size yields a near-optimal balance between training time and data efficiency (Kaplan et al., 2020). Following this, we fit a power-law relationship between the loss and the critical batch size on data from smaller models, as shown in Figure 13. The batch size is doubled when the corresponding loss is reached.

The learning rate schedule begins with a linear warm-up over 500 iterations to a peak value of $2 \times 1 0 ^ { - 4 }$ , followed by training with a constant learning rate for 7.2T tokens. In the latter stages of training, we notice anomalous gradient norm values. This issue is attributed to an excessively high learning rate and we adjusted lr to $1 . 3 \times 1 0 ^ { - 4 }$ for the remaining 3.2T tokens. During the fast decay phase, we train 1T tokens and exponentially decrease the learning rate to $3 \times 1 0 ^ { - 5 }$ . Additionally, the MoE auxiliary loss coefficient is set to 0.01.

Long-Context Extension. We incrementally expand the model’s training context length to 1M tokens. Due to our architecture’s effective length extrapolation capabilities, the model suc-

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/4c96df94bc6eeab1c2f8ff0b0aa20d43248038eef5e8509d1f42ecb23bf826ae.jpg]]  
Figure 13 | The power-law fit for the training loss and the critical batch size, utilizing data from models ranging from 50M to 600M in activated parameters counts. We mark the points where the batch size is doubled with dashed gray lines.

cessfully demonstrates its ability to process sequences up to 4M tokens in the vanilla Needle-In-A-Haystack retrieval task (NIAH) test 2, despite only being trained on contexts up to 1M tokens, as illustrated in Figure 14.

Specifically, we employ a three-stage training procedure to systematically upsample long-context data across diverse length ranges, while preserving the distributional characteristics of critical domains to preserve short-context evaluation performances steady. The details of the training data mixture, RoPE base frequency, and training length are shown in Table 6. We also mix in $1 0 \%$ of high-quality long-context question-answering data with similar length distribution as long-context pre-training data during the last $2 0 \%$ of training cycles in each stage(Parmar et al., 2024). To mitigate potential instabilities resulting from distributional shifts, we utilize linear interpolation of source-specific weights throughout the transitional phase. This method facilitates a gradual and controlled evolution of the data distribution towards the desired target distribution, thereby ensuring training stability

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/6a3f740954c0f9cc5d559aeb5780de97171e4d07347cdd4d1ff910a9fcff9882.jpg]]  
Figure 14 | 4 Million vanilla Needle-In-A-Haystack retrieval task pressure test on MiniMax-Text-01. The token interval is 32K when it is less than 1M, and the token interval is 0.5M when it is greater than 1M.

and preserving convergence properties.

Additionally, our findings indicate that NIAH is inadequate for effectively monitoring the model’s performance throughout the training process. This is primarily because NIAH metric performance reaches its peak score early on, specifically within the initial 128K training steps. To tackle this limitation, we evaluate the model’s intermediate checkpoints using more demanding tasks, which are designed to increase in complexity as training progresses. Notably, despite the escalating difficulty of these tasks, we consistently observe a steady improvement in the model’s performance metrics. This sustained upward trajectory clearly demonstrates the critical importance and necessity of implementing long-context continual pretraining. More details are given in Section 5.7.2.

Table 6 | Long-Context Extension Recipe. For clarity, we categorize the data as follows: data with fewer than 32K tokens are labeled as “Short”; data ranging from 32K to 128K tokens are labeled as “Medium”; and data exceeding 128K tokens are categorized as “Long”.   

<table><tr><td>Training Length</td><td>RoPE Frequency</td><td># Tokens</td><td>Short (%)</td><td>Medium (%)</td><td>Long (%)</td></tr><tr><td>128K</td><td>5M</td><td>300B</td><td>30</td><td>70</td><td>0</td></tr><tr><td>512K</td><td>10M</td><td>32B</td><td>35</td><td>35</td><td>30</td></tr><tr><td>1M</td><td>10M</td><td>26B</td><td>30</td><td>30</td><td>40</td></tr></table>

# 5. Post-training

In this section, we present a thorough post-training framework designed to enhance the model’s general performance, long-context capability, and real-world applicability. Our approach begins with the creation of a diverse, high-quality prompt dataset, accompanied by a hierarchical reward system that evaluates responses across multiple dimensions: correctness, truthfulness, helpfulness, and harmlessness. The training process consists of Supervised Fine-Tuning (SFT), Offline and Online Reinforcement Learning (RL). Through these phases, we systematically align the model with our defined objectives. Model safety is ensured through exhaustive data mining techniques and a specialized harmless reward model. We introduce a novel multi-stage training methodology that significantly enhances the model’s capacity to process extended contexts while maintaining optimal

performance on shorter sequences. This approach results in a robust system capable of handling complex, real-world scenarios. Extensive evaluations conducted across both academic and in-house benchmarks demonstrate that our model achieves top performance across all tasks, while establishing new standards of extremely long-context processing.

# 5.1. Prompt Collection

Our extensive prompt collection encompasses millions of diverse, high-quality queries from various sources. We develop a tagging system that categorizes each prompt based on task type, knowledge domain, and difficulty level. The collection process incorporates sophisticated filtering mechanisms to eliminate redundant prompts while maintaining an optimal difficulty distribution. The prompt set spans various domains including long-context, programming, math, logical reasoning, creative writing, function calling, general-knowledge, and safety-related scenarios.

# 5.2. Reward Model

Our reward model framework evaluates responses across four critical dimensions to ensure alignment with our core principles:

• Correctness. We implement a rigorous evaluation system for responses that can be strictly validated. For mathematical and reasoning tasks, we utilize early-version MiniMax-Text-01 to generate binary reward signals based on answer consistency. Programming solutions undergo comprehensive testing in a secured sandbox environment, with performance metrics derived from test case success rates.   
• Truthfulness. We employ a verification pipeline to assess the factual accuracy of the response. The process involves systematic response sampling, statement decomposition and clustering, crowd-sourced verification, and automated comparison using advanced language models to generate truthfulness scores.   
• Helpfulness. Our evaluation framework assesses compliance with user instructions through both deterministic and probabilistic approaches. We implement automated rule-based constraint verification systems complemented by human evaluation of key metrics including coherence, depth, contextual relevance, and stylistic appropriateness. The final helpfulness score combines multiple evaluation signals through a weighted scoring system.   
• Harmlessness. Building upon Constitutional AI principles (Bai et al., 2022b), we develop evaluation criteria encompassing safety protocols, content appropriateness, and legal compliance. Our assessment system leverages carefully calibrated prompts validated against human annotations, with early-version MiniMax-Text-01 providing standardized safety evaluations.

# 5.3. Supervised Fine-Tuning

Our SFT dataset construction involves a multi-stage process utilizing domain-specific expert models trained through iterative SFT and RL cycles. We implement rejection sampling (Bai et al., 2022a; Dubey et al., 2024) to generate high-quality responses by the experts, sampling multiple variations per prompt across different temperature settings to select optimal demonstrations measured by the reward hierarchy. The response selection process further incorporates both n-gram and semantic similarity filters to ensure maximum diversity and quality in the training data.

# 5.4. Reinforcement Learning

# 5.4.1. Offline Reinforcement Learning

We incorporate the offline RL phase, i.e., Direct Preference Optimization (DPO) (Rafailov et al., 2023), to optimize the model’s performance across diverse prompt distributions, owing to its simplicity and ease of data construction for long-context scenarios. We specifically focus on prompts that maintain distributional consistency with those utilized in the SFT stage. To evaluate the impact of prompt selection, we conduct comparative experiments using two prompt categories: SFT-trained prompts and SFT-untrained but homologous prompts. Empirical results demonstrate negligible performance variations between SFT-trained prompts and their untrained counterparts. Thus, we adopt the SFTtrained ones for the offline RL phase. The experimental protocol involves generating responses with varying temperature parameters for each prompt, followed by systematic evaluation using the reward models described in Section 5.2. We then identify the best and the worst responses to construct preference pairs for DPO training.

# 5.4.2. Online Reinforcement Learning

Online learning demonstrates superior sample efficiency and cross-domain generalization capabilities compared to offline learning methodologies. Therefore, we implement online RL to improve model performance, particularly in mathematical reasoning tasks. Our approach emphasizes prompt diversity and prioritizes prompts with moderate success rates to maximize information gain during policy updates. Notably, we employ SFT-untrained prompts during online RL, as our empirical observations indicate that reusing prompts from previous phases resulted in model saturation, characterized by diminished response perplexity. We propose a modified Group Relative Policy Optimization (GRPO) (Shao et al., 2024) approach incorporating the following key innovations:

• Importance Sampling Weight Clipping. The conventional PPO/GRPO implementation employs one-sided clipping (Schulman et al., 2017; Shao et al., 2024), sometimes leading to gradient instability when processing tokens with a large policy ratio and negative advantage. To address this issue, we implement additional clipping that abandoned this case in the loss function, which effectively regulates the importance sampling magnitude and mitigates noise propagation.   
• KL Divergence Optimization. Due to the similar gradient instability issue, we reformulate the KL divergence term through theoretical analysis of the variance-bias trade-off to further stabilize gradient behavior, resulting in $\mathbb { D } _ { K L } ( \theta ) = \mathbb { E } _ { t } [ { \tt S G } ( \pi _ { \theta } ( a _ { t } | s _ { t } ) - \pi _ { \tt r e f } ( a _ { t } | s _ { t } ) ) \log \pi _ { \theta } ( a _ { t } | s _ { t } ) ]$ where SG(·) denotes the stop-gradient operator. This formulation maintains policy consistency while reducing gradient variance.   
• Balanced Advantage Estimation. We also ensure equitable reward contributions between positive and negative examples, which proves particularly effective in scenarios with skewed distributions. This approach maintains stable training dynamics by regulating the absolute magnitude of rewards across different example groups.

# 5.5. Safety Alignment

The safety alignment of our model is meticulously addressed throughout both the SFT and RL stages. To strike an optimal balance between the model’s harmlessness and helpfulness, we employ an approach that encompasses the following key components.

# 5.5.1. Training Data Construction

We construct high-quality alignment training data with a focus on ensuring data diversity and accuracy. This involves the implementation of several data collection methodologies designed to cover a broad spectrum of safety scenarios:

• Safety-Category Specific Prompts. Leveraging established safety classification standards and insights from safety and domain experts, we generate tailored prompts for specific safety categories. This ensures that the model is exposed to a comprehensive set of safety-related scenarios.   
• Real-World User Data Collection. We collect real-world user questions from various web documents to incorporate authentic and diverse safety-related queries into our training data.   
• Prompt Augmentation. We instruct early-version MiniMax-Text-01 to generate additional related prompts based on the collected typical red team attack prompts. This approach aims to expand the diversity of safety scenarios and enhance the robustness of the model’s safety mechanisms.

# 5.5.2. Response Generation with Harmless Reward Model

To generate safe and appropriate responses, we employ a harmless reward model (Bai et al., 2022b) that is developed based on a set of detailed safety rules. To prevent the model from producing unreasonable refusals, we carefully integrate principles of helpfulness into the safety rules. This integration plays a crucial role in achieving a balanced output capability, enabling the model to provide safer responses without compromising its utility to the user. The resulting safety-aligned system demonstrates robust protection against potential misuse while maintaining high performance across intended use cases.

# 5.6. Training Methodology with Long-Context Adaptation

We propose a systematic multi-stage training methodology to enhance the model’s capacity for processing extended contexts, as shown in Tab. 7. This approach is methodically designed to optimize long-sequence handling while maintaining performance efficacy on conventional shorter sequences. The RoPE base frequency is maintained at 10 million throughout the post-training phase to ensure consistency in positional encoding.

Stage I: Initial Short-Context Training. The first stage implements SFT with sequences constrained to 8,192 tokens. This foundational phase establishes baseline competency in processing standardlength queries and responses, which constitute the majority of practical applications. We remove the long-context prompts that are longer than 8,192 tokens in this stage.

Stage II: Extended Context Training. The second stage implements a significant extension of the sequence length to 1,032,192 tokens. This phase incorporates training samples across diverse sequence lengths with $5 0 \%$ long-context prompts, facilitating comprehensive model adaptation to extensive contextual processing. The strategic expansion of the sequence length is fundamental to achieving robust long-context capabilities.

Stage III: Short-Context Preference Optimization. In this phase, we revert to 8,192 tokens for sequence length and implement Direct Preference Optimization (DPO). This calibration ensures optimal performance on conventional context sizes while maintaining the previously acquired capabilities.

Stage IV: Long-Context Preference Optimization. The fourth stage focuses on reinforcing longcontext processing capabilities through DPO with sequences of 1,032,192 tokens. This phase employs

training protocols analogous to Stage III with entirely long-context data, adapted for extended sequence lengths.

Stage V: Online Reinforcement Learning. The final stage implements short-context Online Reinforcement Learning with a sequence length of 8,192 tokens. More details have been outlined in Section 5.4.2.

Table 7 | Training Recipe for Post-training Alignment.   

<table><tr><td></td><td>Stage I</td><td>Stage II</td><td>Stag III</td><td>Stage IV</td><td>Stage V</td></tr><tr><td>Sequence Length</td><td>8192</td><td>1032192</td><td>8192</td><td>1032192</td><td>8192</td></tr><tr><td>Epoch</td><td>2</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Batch Size</td><td>128</td><td>80</td><td>64</td><td>64</td><td>512</td></tr><tr><td>Max LR</td><td>1e-5</td><td>3e-6</td><td>5e-7</td><td>5e-7</td><td>1e-6</td></tr><tr><td>Min LR</td><td>1e-6</td><td>3e-6</td><td>5e-8</td><td>5e-7</td><td>1e-7</td></tr><tr><td>LR Decay</td><td>Cosine</td><td>Constant</td><td>Cosine</td><td>Constant</td><td>Cosine</td></tr></table>

# 5.7. Academic Benchmarks

We observe and report open-source short- and long-context benchmarks that highlight our model’s capabilities across various aspects. Along with the user-oriented evaluations we will discuss in Section 5.8, we show that MiniMax-Text-01 is a leading open-source model that achieves top performance in long-context retrieval, understanding, long in-context learning and knowledge-based requests, while performing well in math, reasoning, and code tasks and demonstrating strong usefulness in real-user assistant scenarios.

# 5.7.1. Core Benchmarks

MMLU (Hendrycks et al., 2021a) and MMLU-Pro (Wang et al., 2024b) are widely adopted datasets that assess the extent of a model’s knowledge across a broad range of domains. We further observe SimpleQA (Wei et al., 2024), a factuality benchmark that challenges the model’s knowledge boundary, and C-SimpleQA (He et al., 2024b) which is an adapted version of SimpleQA under the Chinese culture. For the observation of reasoning capabilities, we evaluate on GPQA (Rein et al., 2024) for graduate-level knowledge reasoning, and DROP (Dua et al., 2019) for reading comprehension reasoning. We test our model’s performance on math problem-solving with grade-school-level task GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) that spans from AMC-8 to AIMElevel across 7 subjects. We monitor our model’s coding capability by observing the Pass $@ 1$ rate on HumanEval (Chen et al., 2021) and MBPP Plus (Austin et al., 2021; Liu et al., 2023) datasets. To test the models’ ability to interpret and execute detailed and nuanced instructions, we evaluate the IFEval (Zhou et al., 2023) benchmark. Furthermore, we observe Arena-Hard-Auto (Li et al., 2024b) that reflects the alignment to human preferences.

We adopt greedy decoding and a zero-shot chain-of-thought strategy (Wei et al., 2022) in evaluating our instruction-tuned model. We compare with other leading and open-source LLMs, which we evaluate under the same setting, if not reported. We present the performance of MiniMax-Text-01 in Table 8. As shown, MiniMax-Text-01 exhibits remarkable performance across most dimensions. It surpasses all models on C-SimpleQA with its more extensive knowledge boundary under Chinese culture. MiniMax-Text-01 also achieves top-3 performance across MMLU, IFEval, and Arena-Hard, showing its exceptional capability of applying its comprehensive knowledge within given constraints to well satisfy user queries and align with human preferences. Meanwhile, it achieves a better MATH pass $@ 1$ rate than GPT-4o, Claude-3.5-Sonnet, and Llama-3.1-405B, and exhibits comparable

Table 8 | Performance of MiniMax-Text-01 on core academic benchmarks.   

<table><tr><td>Tasks</td><td>GPT-4o (11-20)</td><td>Claude-3.5-Sonnet (10-22)</td><td>Gemini-1.5-Pro (002)</td><td>Gemini-2.0-Flash (exp)</td><td>Qwen2.5-72B-Inst.</td><td>DeepSeek-V3</td><td>Llama-3.1-405B-Inst.</td><td>MiniMax-Text-01</td></tr><tr><td colspan="9">General</td></tr><tr><td>MMLU*</td><td>85.7</td><td>88.3</td><td>86.8</td><td>86.5</td><td>86.1</td><td>88.5</td><td>88.6</td><td>88.5</td></tr><tr><td>MMLU-Pro*</td><td>74.4</td><td>78.0</td><td>75.8</td><td>76.4</td><td>71.1</td><td>75.9</td><td>73.3</td><td>75.7</td></tr><tr><td>SimpleQA</td><td>39.0</td><td>28.1</td><td>23.4</td><td>26.6</td><td>10.3</td><td>24.9</td><td>23.2</td><td>23.7</td></tr><tr><td>C-SimpleQA</td><td>64.6</td><td>56.8</td><td>59.4</td><td>63.3</td><td>52.2</td><td>64.8</td><td>54.7</td><td>67.4</td></tr><tr><td>IFEval (avg)</td><td>84.1</td><td>90.1</td><td>89.4</td><td>88.4</td><td>87.2</td><td>87.3</td><td>86.4</td><td>89.1</td></tr><tr><td>Arena-Hard</td><td>92.4</td><td>87.6</td><td>85.3</td><td>72.7</td><td>81.2</td><td>91.4</td><td>63.5</td><td>89.1</td></tr><tr><td colspan="9">Reasoning</td></tr><tr><td>GPQA* (diamond)</td><td>46.0</td><td>65.0</td><td>59.1</td><td>62.1</td><td>49.0</td><td>59.1</td><td>50.7</td><td>54.4</td></tr><tr><td>DROP* (F1)</td><td>89.2</td><td>88.8</td><td>89.2</td><td>89.3</td><td>85.0</td><td>91.0</td><td>92.5</td><td>87.8</td></tr><tr><td colspan="9">Mathematics</td></tr><tr><td>GSM8k*</td><td>95.6</td><td>96.9</td><td>95.2</td><td>95.4</td><td>95.8</td><td>96.7</td><td>96.7</td><td>94.8</td></tr><tr><td>MATH*</td><td>76.6</td><td>74.1</td><td>84.6</td><td>83.9</td><td>81.8</td><td>84.6</td><td>73.8</td><td>77.4</td></tr><tr><td colspan="9">Coding</td></tr><tr><td>MBPP +</td><td>76.2</td><td>75.1</td><td>75.4</td><td>75.9</td><td>77.0</td><td>78.8</td><td>73.0</td><td>71.7</td></tr><tr><td>HumanEval</td><td>90.2</td><td>93.7</td><td>86.6</td><td>89.6</td><td>86.6</td><td>92.1</td><td>89.0</td><td>86.9</td></tr></table>

∗ Evaluated following a 0-shot CoT setting.

performance with instructed Qwen2.5-72B on HumanEval. Moreover, MiniMax-Text-01 achieves 54.4 on GPQA Diamond, which exceeds most open-source instruction-tuned LLMs and the latest version of GPT-4o.

# 5.7.2. Long Benchmarks

As previously discussed in the long-context extension part of section 4.2, the NIAH task is kind of simplistic for our model, rendering it insufficient for observing the model’s optimization progress. Consequently, we shift our evaluation to more challenging tasks. Our current long-context evaluation framework focuses on three primary dimensions: (1) Long-Context Retrieval, (2) Long-Context Understanding, and (3) Long In-Context Learning.

# 5.7.2.1 Long-Context Retrieval

This dimension assesses the model’s memory capabilities, which serve as the foundation for almost all long-context tasks. In addition to vanilla $k$ -M NIAH (Kamradt, 2023), we construct a more challenging variation to assess our Long-Context Retrieval performance, namely Multi-Round Needles-In-A-Haystack (MR-NIAH), serving as a crucial back up for retrieval tasks in long multi-turn dialogue contexts, revealing the fundamental capabilities for building lifelong companion AI assistants. Similar to Multi-round co-reference resolution (MRCR) (Vodrahalli et al., 2024) which is not open-source, we construct haystacks of MR-NIAH as history dialogues, where user queries are synthetic but explicit requests of event descriptions and creative writing. In the last round, the query requests the model to repeat the response of one of the history requests. The haystacks span from 2K to 1M tokens (up to around 2000 interactions), and each needle request is injected at $2 5 \%$ , $5 0 \%$ , and $7 5 \%$ of the

conversation, respectively. Each ground truth response contains three core components, and we look at an adjusted recall corr. comp.3 . We show a case illustration in Appendix B.2.

Figure 15 illustrates comparison results of MR-NIAH. Our model (“MiniMax-Text-01”, red line) shows strong performance across a wide range of sequence lengths in both English and Chinese evaluations. Compared to competing baselines (e.g., GPT, Claude, and Gemini variants), our model also shows less performance degradation at large input lengths, underscoring its robustness for long-context retrieval tasks.

# 5.7.2.2 Long-Context Understanding

This dimension measures the model’s longcontext understanding ability which contains logical reasoning skills based on long-context inputs. We utilize two comprehensive longcontext QA datasets, Ruler (Hsieh et al., 2024) and LongBench-V2 (Bai et al., 2024) to evaluate this aspect. Ruler includes 13 different tasks and notably introduces multi-hop tracing and aggregation tasks to evaluate the complex reasoning abilities of models. We test Ruler up to a sequence length of 1M tokens. LongBench-V2 encompasses question-answering tasks of varying difficulty levels across multiple context types, including single and multi-document, multi-turn dialogue, code repositories, and long structured data, among others. Following LongBench-V2 (Bai et al., 2024), we consider two test

modes: w/o CoT and w/ CoT, and the text lengths are categorized as follows: Short, ranging from 0 to 32K words; Medium, spanning from 32K to 128K words; and Long, covering 128K to 2M words.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/e8e7d2057c45d46ec41e58913ace745c95a31cbc84aab52694a2568464d0fbab.jpg]]  
Figure 15 | MR-NIAH in English and Chinese.

As Table 9 illustrates, our model exhibits notable strengths in processing Ruler’s long-context reasoning tasks. While performance at the 64k input level remains competitive with leading models (including GPT-4o and Claude-3.5-Sonnet) with minimal variation, MiniMax-Text-01 establishes a distinct advantage beginning at $1 2 8 \mathrm { k }$ , achieving impressive scores and surpassing all benchmark models. This superiority becomes particularly pronounced in ultra-long-context scenarios (such as 1M), where MiniMax-Text-01 maintains its commanding lead. Moreover, as evident in Table $1 0 ^ { 3 }$ , MiniMax-Text-01 exhibits outstanding capabilities in LongBench-V2’s long-context reasoning tasks. The model achieves state-of-the-art results among all evaluated systems in the w/ CoT setting, while also displaying remarkable effectiveness in scenarios w/o CoT.

Overall, MiniMax-Text-01 demonstrates exceptional capability in long-context understanding especially reasoning tasks, both with and without CoT reasoning, particularly excelling in scenarios requiring complex reasoning. The exceptional robustness and stability of the model in processing long-context understanding tasks can be attributed to the hybrid architecture with half RoPE and carefully tuned training recipes for both pre-training and alignment, which enhance the model’s ability to handle long sequences effectively.

Table 9 | Performance comparison of MiniMax-Text-01 on Ruler.   

<table><tr><td>Model</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>256k</td><td>512k</td><td>1M</td></tr><tr><td>GPT-4o (11-20)</td><td>0.970</td><td>0.921</td><td>0.890</td><td>0.888</td><td>0.884</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Claude-3.5-Sonnet (10-22)</td><td>0.965</td><td>0.960</td><td>0.957</td><td>0.950</td><td>0.952</td><td>0.938</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gemini-1.5-Pro (002)</td><td>0.962</td><td>0.960</td><td>0.960</td><td>0.958</td><td>0.938</td><td>0.917</td><td>0.916</td><td>0.861</td><td>0.850</td></tr><tr><td>Gemini-2.0-Flash (exp)</td><td>0.960</td><td>0.960</td><td>0.951</td><td>0.957</td><td>0.937</td><td>0.860</td><td>0.797</td><td>0.709</td><td>-</td></tr><tr><td>MiniMax-Text-01</td><td>0.963</td><td>0.961</td><td>0.953</td><td>0.954</td><td>0.943</td><td>0.947</td><td>0.945</td><td>0.928</td><td>0.910</td></tr></table>

Table 10 | Performance comparison of MiniMax-Text-01 on LongBench v2.   

<table><tr><td>Model</td><td>overall</td><td>easy</td><td>hard</td><td>short</td><td>medium</td><td>long</td></tr><tr><td>Human</td><td>53.7</td><td>100.0</td><td>25.1</td><td>47.2</td><td>59.1</td><td>53.7</td></tr><tr><td colspan="7">w/ CoT</td></tr><tr><td>GPT-4o (11-20)</td><td>51.4</td><td>54.2</td><td>49.7</td><td>59.6</td><td>48.6</td><td>43.5</td></tr><tr><td>Claude-3.5-Sonnet (10-22)</td><td>46.7</td><td>55.2</td><td>41.5</td><td>53.9</td><td>41.9</td><td>44.4</td></tr><tr><td>Deepseek-V3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-72B-Inst.</td><td>43.5</td><td>47.9</td><td>40.8</td><td>48.9</td><td>40.9</td><td>39.8</td></tr><tr><td>MiniMax-Text-01</td><td>56.5</td><td>66.1</td><td>50.5</td><td>61.7</td><td>56.7</td><td>47.2</td></tr><tr><td colspan="7">w/o CoT</td></tr><tr><td>GPT-4o (11-20)</td><td>50.1</td><td>57.4</td><td>45.6</td><td>53.3</td><td>52.4</td><td>40.2</td></tr><tr><td>Claude-3.5-Sonnet (10-22)</td><td>41.0</td><td>46.9</td><td>37.3</td><td>46.1</td><td>38.6</td><td>37.0</td></tr><tr><td>Deepseek-V3</td><td>48.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-72B-Inst.</td><td>42.1</td><td>42.7</td><td>41.8</td><td>45.6</td><td>38.1</td><td>44.4</td></tr><tr><td>MiniMax-Text-01</td><td>52.9</td><td>60.9</td><td>47.9</td><td>58.9</td><td>52.6</td><td>43.5</td></tr></table>

# 5.7.2.3 Long In-Context Learning

This dimension evaluates the model’s ability to learn from context, a core area of research in lifelong learning. We benchmark our Long In-Context Learning capability with the MTOB (Machine Translation

from One Book) (Tanzer et al., 2024) dataset. The task requires a model to translate between English and Kalamang, a language that is very limited in open data and thus within the training corpus, and the LLM is expected to learn the language only from parts of a grammar book and 375 translation examples, all given in the context for each translation query (Appendix B.1). The context length is $\sim 8 1 \mathrm { K }$ tokens under a half-book setting and ∼ 133K tokens under a total-book setting. We present our results in Table 11.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/1241d5fa490294aadd6dacfcdb8974e4e999f59efd8d6048e5a56f697a1f8a41.jpg]]  
Figure 16 | Changes of eng kalam (ChrF) during the whole long-context extension training process.

We carefully examined the pre-training data and found that only a very small amount of data contains Kalamang-related content. As a result, the eng kalam (ChrF) score of our model is the lowest in the no-context scenario, while other models we compared with likely have had their pre-train or post-train data enhanced with relevant Kalamang data. As well as the delta half and full book metrics, our model surpasses all models in terms of the eng kalam (ChrF) metric. And our model also has comparable performance with other models on kalam eng (BLEURT) metric.

In the course of long-context extension, as described in section 4.2, we observed a gradual enhancement in In-Context Learning ability, as indicated by MTOB, illustrated in Figure 16. While we have explored some remarkable works(Agarwal et al., 2024; Dong et al., 2024) specifically aimed at improving In-Context Learning capabilities, we believe that such ability should merely be one aspect of the reasoning capabilities of long-context models. Therefore, we plan to conduct in-depth research on long-context data quality and scale from a more fundamental perspective to further enhance the long-context reasoning capabilities of our model.

Table 11 | Performance comparison of MiniMax-Text-01 on MTOB.   

<table><tr><td>Context Type</td><td>no context</td><td>half book</td><td>full book</td><td>Δ half book</td><td>Δ full book</td></tr><tr><td colspan="6">eng → kalam (ChrF)</td></tr><tr><td>GPT-4o (11-20)</td><td>9.90</td><td>54.30</td><td>-</td><td>44.40</td><td>-</td></tr><tr><td>Claude-3.5-Sonnet (10-22)</td><td>20.22</td><td>53.62</td><td>55.65</td><td>33.39</td><td>35.42</td></tr><tr><td>Gemini-1.5-Pro (002)</td><td>16.79</td><td>53.68</td><td>57.90</td><td>36.89</td><td>41.11</td></tr><tr><td>Gemini-2.0-Flash (exp)</td><td>12.20</td><td>49.50</td><td>53.30</td><td>37.30</td><td>41.10</td></tr><tr><td>Qwen-Long</td><td>16.55</td><td>48.48</td><td>45.94</td><td>31.92</td><td>29.39</td></tr><tr><td>MiniMax-Text-01</td><td>6.0</td><td>51.74</td><td>51.60</td><td>45.7</td><td>45.6</td></tr><tr><td colspan="6">kalam → eng (BLEURT)</td></tr><tr><td>GPT-4o (11-20)</td><td>33.20</td><td>58.30</td><td>-</td><td>25.10</td><td>-</td></tr><tr><td>Claude-3.5-Sonnet (10-22)</td><td>31.42</td><td>59.70</td><td>62.30</td><td>28.28</td><td>30.88</td></tr><tr><td>Gemini-1.5-Pro (002)</td><td>32.02</td><td>61.52</td><td>63.09</td><td>29.50</td><td>31.07</td></tr><tr><td>Gemini-2.0-Flash (exp)</td><td>33.80</td><td>57.50</td><td>57.00</td><td>23.70</td><td>23.20</td></tr><tr><td>Qwen-Long</td><td>30.13</td><td>53.14</td><td>32.15</td><td>23.01</td><td>2.02</td></tr><tr><td>MiniMax-Text-01</td><td>33.65</td><td>57.10</td><td>58.00</td><td>23.45</td><td>24.35</td></tr></table>

# 5.8. User-in-the-loop

While achieving top performance on the core open-source benchmarks, we realize that academic evaluations lack an understanding of real-world user interactions. Hence, we also focus on monitoring and improving user experience through our Hailuo AI 4 by incorporating user-in-the-loop evaluations based on real-world cases and adapting tools for better usability and performance in practical applications.

# 5.8.1. In-House Evaluations

We maintain a series of in-house evaluations that include: (1) automatic assessments of General Assistant capabilities, Knowledge Q&A, Creative Writing, Hard Capability, Instruction Following, Coding, Safety, and Long Context, and (2) expert human evaluations. It’s worth noting that since our test queries are primarily derived from Hailuo AI user interactions, a significant portion of our in-house samples are in Mandarin and deeply rooted in Chinese cultural contexts.

Our results indicate a notable discrepancy between performance on academic benchmarks and actual user experience, where leading open-source and commercial models can underperform when used as interactive assistants. We show in Table $1 2 ^ { 5 }$ that, through our dedicated efforts, MiniMax-Text-01 is able to handle these situations quite well. In general, our model outperforms other models

Table 12 | Performance comparison of MiniMax-Text-01 on in-house benchmarks.   

<table><tr><td></td><td>General Assistant</td><td>Hard Capability</td><td>Creative Writing</td><td>Knowledge Q&amp;A</td><td>Instruction Following</td><td>Coding</td><td>Safety</td><td>Long Context</td></tr><tr><td>GPT-4o (11-20)</td><td>70.9</td><td>73.5</td><td>70.3</td><td>69.2</td><td>50.4</td><td>94.0</td><td>85.4</td><td>86.2</td></tr><tr><td>GPT-4o (08-06)</td><td>63.5</td><td>62.0</td><td>66</td><td>68.0</td><td>49.1</td><td>93.6</td><td>79.7</td><td>58.3</td></tr><tr><td>GPT-4o (05-13)</td><td>67.7</td><td>63.3</td><td>58.3</td><td>69.6</td><td>49.6</td><td>93.2</td><td>79.7</td><td>77.2</td></tr><tr><td>Claude-3.5-Sonnet (10-22)</td><td>66.8</td><td>68.3</td><td>54.3</td><td>52.0</td><td>61.5</td><td>94.4</td><td>92.9</td><td>47.1</td></tr><tr><td>Claude-3.5-Sonnet (06-20)</td><td>60.5</td><td>67.4</td><td>51.0</td><td>51.8</td><td>64.4</td><td>93.6</td><td>95.0</td><td>47.1</td></tr><tr><td>Gemini-2.0-Flash (exp)</td><td>70.1</td><td>61.8</td><td>70.0</td><td>75.1</td><td>39.9</td><td>86.5</td><td>66.2</td><td>81.9</td></tr><tr><td>Qwen2.5-72B-Inst.</td><td>66.4</td><td>66.1</td><td>61.7</td><td>68.9</td><td>34.1</td><td>93.9</td><td>-</td><td>81.5</td></tr><tr><td>DeepSeek-V3</td><td>66.8</td><td>68.7</td><td>64.6</td><td>77.0</td><td>51.8</td><td>94.0</td><td>74.9</td><td>77.8</td></tr><tr><td>Llama-3.1-405B-Inst.</td><td>53.3</td><td>-</td><td>63.6</td><td>46.0</td><td>50.3</td><td>87.6</td><td>70.7</td><td>60.3</td></tr><tr><td>MiniMax-Text-01</td><td>73.9</td><td>64.8</td><td>81.3</td><td>78.6</td><td>46.3</td><td>90.2</td><td>90.9</td><td>93.8</td></tr></table>

in common Assistant scenarios, particularly when compared to open-source counterparts. This superiority is most evident in our Creative Writing (Appendix B.5, B.7, B.6) and Knowledge Q&A collections, where it aligns more closely with user intentions than other models, delivering accurate and detailed responses to a wide range of queries. In productivity scenarios that require Long Context (Appendix B.3), such as document translation, summarization, and analysis, our model demonstrates high proficiency and reliability. Moreover, we prioritize the safety of our model, as it achieves top-tier performance on our established in-house Safety benchmarks.

Meanwhile, we are agile in gathering and updating complex productivity scenarios with multilevel instruction following requests at which our model fails and current LLMs cannot master, constructing our Harder Capability and Instruction Following in-house evaluations. While leading LLMs tend to underperform in these sets, these requests reflect our model’s limitations when given multi-level instructions, which stems primarily from insufficient training data for specific instruction types. Moving forward, we are committed to substantially expanding our training dataset with high-quality, targeted content to address these gaps and improve model capabilities.

# 5.8.2. Search in Hailuo AI

During user interaction case studies, we find a model’s capability to utilize search tools can compensate for the limited knowledge boundary by accessing real-time, extensive, and precise information from the web. To maximize the model’s benefits from search while minimizing additional performance degradation, we first carefully pre-define the scope of search scenarios, which cover approximately $3 0 \sim 4 0 \%$ of user queries, including but not limited to precision-demanding, domain-specific, and time-sensitive requests. Meanwhile, to ensure a seamless conversation experience, we define the system as invoking tools directly through special tokens, which avoid the complexity of multi-step planning (Chen et al., 2024b) or chain-of-thought reasoning6 that might disrupt the natural flow of the interactions. We create SFT datasets comprising search and non-search decisions across diverse domains, while carefully controlling for other interaction features unrelated to search decisions, such as conversation length, to maintain uniform data distribution across each dimension and prevent overfitting. Importantly, we employ the corresponding reward model of each sample to ensure response quality, failing at which would introduce suboptimal samples into the training

data, potentially affecting the model’s fundamental capabilities. The search decision boundary was calibrated to align with the model’s knowledge boundaries, discarding samples that our model already masters from the search corpus, such as general Chinese knowledge Q&A. After careful assessments by human evaluation experts, we conclude that our model’s use of the search tool extensively improved user experience, landing at a performance leap from $5 8 \%$ to $7 1 . 5 \%$ on our out-of-domain Hailuo AI end-to-end evaluation (Appendix B.9). Since we are unsure whether other LLM-based assistants include similar search tools, we refrain from making unfair performance comparisons.

# 6. Vision-language Model

By integrating an image encoder and an image adapter into our MiniMax-Text-01 model, we develop MiniMax-VL-01, which extends the capabilities of the model to visual understanding tasks. To ensure robust visual understanding, we design a proprietary dataset and implement a multi-stage training strategy, where the newly introduced image encoder and adapter first undergo large-scale visual pre-training, followed by comprehensive fine-tuning of the entire pipeline.

In the following section, we begin with a comprehensive description of the dataset used for training our image encoder and vision-language model. Subsequently, we provide an in-depth overview of the model architecture, followed by an exposition of our four-stage training regimen. We conclude the section by presenting our benchmark results.

# 6.1. Multimodal Data

# 6.1.1. Caption Data

To pre-train the vision encoder, we curate a substantial image-caption dataset by aggregating and filtering data from internet sources. Our Vision Transformer (ViT) is trained using 694 million unique image-caption pairs. To enhance data quality, we acquire refined captions for 180 million images within these pairs. During the training process, we employ an augmentation strategy by randomly sampling raw and refined captions with equal probability $\left( p = 0 . 5 \right)$ .

# 6.1.2. Description Data

In existing vision-language models, the utility of descriptive imagery for model training has been well-documented (Li et al., 2024a, 2022, 2023; Schuhmann et al., 2021). To further explore this avenue, we have compiled a dataset consisting of 100 million images sourced from open resources such as Common Crawl. Each image in this dataset is paired with a fine-grained description, which is initially synthesized by a caption model and subsequently refined through humans. On average, these descriptions comprise approximately 300 text tokens per image. Description data serves as a robust resource for modal alignment and enhancing understanding in further training.

# 6.1.3. Instruction Data

To train MiniMax-VL-01, we construct a comprehensive and diverse instruction-based dataset by synthesizing an extensive range of question-answer (QA) pairs involving visual inputs. These QA pairs are meticulously designed to cover a wide array of image-related tasks, such as text extraction, object localization, and geometry problem solving. The dataset generation process prioritizes both diversity and realism, ensuring that the instructions capture varying degrees of complexity and linguistic styles. During training, we apply an augmentation strategy by randomly sampling different types of QA prompts with balanced probabilities, thereby enabling the model to generalize effectively across

diverse instructional formats and interaction patterns.

# 6.1.4. Data Distribution

To demonstrate the diversity of our VLM data, we uniformly sample 1 million imageinstruction pairs from the instruction data and use another VLM to assign a concise tag (e.g., object localization) that represents the primary capability required for each pair. This analysis yielded around 50,000 unique tags, and the top 2,817 tags appeared more than 10 times. The distribution of these prominent tags is visualized in Figure 17, where we further group these top tags into 14 major categories.

# 6.2. Architecture

# 6.2.1. Overall Architecture

Our MiniMax-VL-01 architecture adheres to the “ViT-MLP-LLM” paradigm, which has been widely embraced in numerous multimodal large language models (MLLMs). The architecture consists of three main components: a Vision Transformer (ViT) with 303 million parameters for visual encoding, a two-layer MLP projector initialized randomly for image adaptation, and the MiniMax-Text-01 model serving as the foundational large language model (LLM).

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/ffc270cdba4ba1e0300310a7bb670dcb846e663e7cd34acac468e34e96664c29.jpg]]  
Figure 17 | Visualization of top tags of sampled instruction data. The category and percentage for each group of clustered tags are displayed in the inner layer, only top-10 tags of each group are displayed for clarity.

We implement a dynamic resolution strategy by resizing the input image according to a predefined grid configuration list, ranging from $3 3 6 \times 3 3 6$ to $2 0 1 6 \times 2 0 1 6$ , while maintaining a standard thumbnail at a resolution of $3 3 6 \times 3 3 6$ . The resized images are subsequently partitioned into non-overlapping patches, each measuring $3 3 6 \times 3 3 6$ . Both the image patches and the thumbnail are independently encoded, and their encoded features are concatenated to construct a comprehensive image feature representation.

In contrast to traditional approaches that rely on pooling or other downsampling techniques to compress feature representations, our model leverages its powerful capacity for processing long sequences, allowing for the direct utilization of raw high-dimensional features during training. This strategy mitigates potential information loss and substantially improves the model’s adaptability to multi-scale inputs. Moreover, by projecting both image patches and thumbnails into a unified feature space, our method significantly enhances the model’s robustness and representational expressiveness when handling diverse and complex visual inputs.

# 6.2.2. Vision Encoder

We employ a lightweight ViT-L/14 (Dosovitskiy et al., 2021) as the foundational structure for our vision encoder and train it from scratch. Following a standard pipeline, the input image tensor is initially processed through a convolutional layer to extract discrete patches, to which absolute

positional embeddings are subsequently appended. The resulting tensors are then passed through a series of multi-head residual attention blocks. This architecture is particularly effective in capturing intricate visual details and the complex interrelationships within images.

We utilize contrastive learning to enhance the alignment between corresponding image-caption pairs while diminishing the alignment between non-corresponding pairs. Specifically, we follow the approach introduced in CoCa (Yu et al., 2022), which augments image-text contrastive learning with an additional decoder and image-text cross-attention mechanisms. The network is jointly optimized using a combination of contrastive loss and cross-entropy loss.

Our ViT-L/14 model is initially trained at a resolution of $2 2 4 \times 2 2 4$ for 37 billion image-caption pairs and subsequently fine-tuned at $3 3 6 \times 3 3 6$ for 1.2 billion pairs. For both resolutions, the captions are truncated to 76 tokens. Our ViT-L/14 encoder achieves a zero-shot classification accuracy of $8 0 . 5 5 \%$ at $3 3 6 \times 3 3 6$ resolution on the ImageNet-1K dataset.

# 6.3. Training Recipes

We employ a four-stage training strategy to enable the model to progressively develop comprehensive multimodal understanding capabilities while retaining its language understanding skills. Additionally, the model’s question-answering and instruction-following abilities, as well as its alignment with human preferences, are methodically refined throughout these stages.

Stage I: Modality alignment. In this stage, our primary objective is to achieve alignment between visual and text tokens by enabling the model to accurately generate appropriate captions for given images. To this end, we update the weights of both the image adapter and the vision encoder to optimize their performance in this multimodal task. During this phase, we utilize a total of 80 billion tokens sampled from our image description dataset. Empirically, we have found that increasing the image resolution does not yield improvements in downstream task accuracy. Therefore, all images are processed at a fixed resolution of $3 3 6 \times 3 3 6$ to reduce computational costs.

Stage II: Enhancement of Vision Understanding. This stage can be regarded as a standard instruction tuning phase, during which all model parameters are open to updates. The primary goal is to align the model’s output with human instructions and enhance its ability to perform a diverse range of vision understanding tasks. To achieve this, the model is trained using 420 billion multimodal tokens sampled from our instruction datasets, combined with MiniMax-Text-01 post-training data in a ratio of 20:1. This approach ensures that the language modeling capability is maintained while the model acquires new multimodal capabilities.

Stage III: Enhancement of User Experience. This stage is designed to further enhance the model’s capabilities in real-world scenarios and when handling challenging user inputs. We curate sophisticated multimodal data using images sourced from applications that people commonly interact with. Conversations are meticulously labeled to emulate authentic user input and to ensure the provision of accurate, helpful, and diverse responses across multiple conversational turns. The data construction for this stage is guided by an independent human-labeled test set that prioritizes not only accuracy but also the overall quality in terms of user experience. The resulting dataset comprises 44.8 billion multimodal tokens and is trained for one epoch.

Stage IV: Enhancement of Preference. In the final stage, we utilize Direct Preference Optimization (DPO) to further enhance model performance and user experience. We construct a training dataset consisting of 40,000 image-text pairs through the following process:

• Prompt Selection. Prompts are curated from both instruction data and real user interaction data. These prompts are selected to cover a wide range of general scenarios and to specifically address

persistent issues identified after Stage III, such as occasional repetitive outputs in complex OCR scenarios.

• Response Generation. We employ diverse strategies, including: generating multiple candidate responses by varying sampling temperature parameters; creating response variants through image weakening in specific scenarios; and using MiniMax-Text-01 to deliberately introduce hallucinations or errors into high-quality responses to generate contrastive samples in specific scenarios.   
• Reward Assignment. Large language models, particularly MiniMax-Text-01, are utilized as evaluators in this stage. Multi-dimensional evaluation criteria are designed to enable a systematic and comprehensive assessment of the relationships among prompts, ground truth answers, and generated responses.   
• Pair Construction. Based on the evaluation results, we select the highest-scoring responses as positive samples and the lowest-scoring ones as negative samples, while discarding pairs with insignificant score differences.

In addition to incorporating image-text pairs, we also include a significant proportion of pure text pairs, as elaborated in Section 5.4.1. It is noteworthy that when Direct Preference Optimization (DPO) is applied to highly capable foundation models, there is a propensity for overfitting. To counteract this issue, we adopt an early stopping strategy, which involves terminating the training process prior to the completion of a full epoch. This approach is designed to preserve the model’s generalization capabilities.

By following this multi-stage training strategy, we ensure that our model not only demonstrates proficiency in understanding and generating high-quality text but also aligns with human values and safety standards. This comprehensive approach to training allows us to strike a balance between model performance and ethical considerations, thereby producing a model that is both effective and responsible.

# 6.4. Benchmarks

To assess the performance of our vision-language model, we maintain a diverse set of benchmarks, including MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al., 2024b), ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), OCRBench (Liu et al., 2024b), AI2D (Kembhavi et al., 2016), MathVista (Lu et al., 2023), OlympiadBench (He et al., 2024a), MMLongBench-Doc (Ma et al., 2024), MEGA-Bench (Chen et al., 2024a) and an in-house benchmark. These benchmarks help evaluate the model’s abilities in various areas, including knowledge, visual reasoning, mathematics, science, long context handling, and user experience. We detail our evaluation configuration for each benchmark in Appendix D. As shown in Table 13, MiniMax-VL-01 achieves competitive performance across various vision-language tasks, demonstrating the following key strengths and limitations:

Common Downstream Tasks. In standard vision-language downstream tasks, MiniMax-VL-01 exhibits performance on par with GPT-4o, particularly excelling in visual question answering. This strong performance is attributed to its extensive multi-stage training process, enabling the model to effectively understand and reason across visual and textual inputs. However, MiniMax-VL-01 still struggles with advanced mathematical reasoning tasks, as assessed by OlympiadBench (He et al., 2024a).

Long Context. We assess MiniMax-VL-01’s capability for long-context comprehension and retrieval using MMLongBench-Doc (Ma et al., 2024). The results show that our model outperforms most counterparts, except GPT-4o-11-20. Despite its strong performance overall, MiniMax-VL-01 demonstrates a noticeable gap in both single-page (acc: $4 7 . 3 \%$ ) and cross-page (acc: $2 8 . 4 \%$ ) subsets.

Table 13 | Performance of MiniMax-VL-01 on academic and in-house benchmarks.   

<table><tr><td>Tasks</td><td>GPT-4o (11-20)</td><td>Claude-3.5-Sonnet (10-22)</td><td>Gemini-1.5-Pro (002)</td><td>Gemini-2.0-Flash (exp)</td><td>Qwen2-VL-72B-Inst.</td><td>InternVL 2.5-78B</td><td>LLama-3.2-90B</td><td>MiniMax-VL-01</td></tr><tr><td colspan="9">Knowledge</td></tr><tr><td>\(MMMU^{*}_{val+dev}\)</td><td>63.5</td><td>72.0</td><td>68.4</td><td>70.6</td><td>64.5</td><td>66.5</td><td>62.1</td><td>68.5</td></tr><tr><td>\(MMMU-Pro^{*}_{full}\)</td><td>54.5</td><td>54.7</td><td>50.9</td><td>57.0</td><td>43.2</td><td>47.3</td><td>36.0</td><td>52.7</td></tr><tr><td colspan="9">Visual Q&amp;A</td></tr><tr><td>\(ChartQA^{*}_{relaxed}\)</td><td>88.1</td><td>90.8</td><td>88.7</td><td>88.3</td><td>91.2</td><td>91.5</td><td>85.5</td><td>91.7</td></tr><tr><td>\(DocVQA^{*}\)</td><td>91.1</td><td>94.2</td><td>91.5</td><td>92.9</td><td>97.1</td><td>96.1</td><td>90.1</td><td>96.4</td></tr><tr><td>OCRbench</td><td>806</td><td>790</td><td>800</td><td>846</td><td>856</td><td>847</td><td>805</td><td>865</td></tr><tr><td colspan="9">Mathematics &amp; Sciences</td></tr><tr><td>AI2D*</td><td>83.1</td><td>82.0</td><td>80.9</td><td>85.1</td><td>84.4</td><td>86.8</td><td>78.9</td><td>83.3</td></tr><tr><td>\(\mathbf{MathVista^{*}}_{testmini}\)</td><td>62.1</td><td>65.4</td><td>70.6</td><td>73.1</td><td>69.6</td><td>68.4</td><td>57.3</td><td>68.6</td></tr><tr><td>\(\mathbf{OlympiadBench^{*}}_{full}\)</td><td>25.2</td><td>28.4</td><td>32.1</td><td>46.1</td><td>21.9</td><td>25.1</td><td>19.3</td><td>24.2</td></tr><tr><td colspan="9">Long Context</td></tr><tr><td>M-LongDocacc</td><td>41.4</td><td>31.4</td><td>26.2</td><td>31.4</td><td>11.6</td><td>19.7</td><td>13.9</td><td>32.5</td></tr><tr><td colspan="9">Comprehensive</td></tr><tr><td>MEGA-Benchmacro</td><td>49.4</td><td>51.4</td><td>45.9</td><td>53.9</td><td>46.8</td><td>45.3</td><td>19.9</td><td>47.4</td></tr><tr><td colspan="9">User Experience</td></tr><tr><td>In-house Benchmark</td><td>62.3</td><td>47.0</td><td>49.2</td><td>72.1</td><td>40.6</td><td>34.8</td><td>13.6</td><td>56.6</td></tr><tr><td colspan="9">* Evaluated following a O-shot CoT setting.</td></tr></table>

Comprehensive Benchmark. On the recently introduced MEGA-Bench (Chen et al., 2024a), a realistic and comprehensive evaluation suite, MiniMax-VL-01 shows competitive overall capabilities, surpassing existing open-source vision LLMs. While it excels in diverse sub-tasks such as knowledge and coding, the model faces challenges in more complex tasks, including planning and metric assessments.

In-house User Experience Benchmark. While academic benchmarks often focus on problemsolving, they frequently fail to capture the nuances of real-world user interactions with models. To bridge this gap, we develop an in-house benchmark comprising 90 diverse image-related tasks, each designed with tailored and challenging instructions. The images and instructions in the benchmark are strictly deduplicated to not overlap with the training set at any stage. Task relevance is manually verified, with a detailed checklist annotated for each sample to ensure precise evaluation. The final test set consists of 524 meticulously annotated samples in both Chinese and English, but Chinese is primarily used. We illustrate some samples in Appendix C. In a win-rate comparison against a top-leading vision-language model, our model outperforms all open-source models and approaches the performance of GPT-4o-11-20 with a narrow margin.

# 7. Conclusion and Future work

In this report, we present MiniMax-Text-01 and MiniMax-VL-01, two novel models developed entirely from the ground up. These models demonstrate top-tier performance across standard benchmarks, particularly excelling in long-context processing with the ability to handle context windows of up to 4 million tokens. Our research findings challenge the prevailing assumption that state-of-the-art

language models must be built upon traditional attention mechanisms. By strategically integrating linear attention with optimized hardware utilization and carefully designing training recipes, we have successfully expanded the context window by an order of magnitude. This breakthrough not only enhances the efficiency and scalability of LLMs but also paves the way for future models to support even longer context windows and facilitate the development of more sophisticated AI agents. To promote collaboration and advancement in the field, we have made our model publicly available at https://github.com/MiniMax-AI. For general use and evaluation, we provide a Chatbot with online search capabilities (https://www.hailuo.ai/) and the online API (https://intl.minimaxi.com). We are committed to keeping this series open source and will release updates as we develop improved models.

While MiniMax-Text-01 and MiniMax-VL-01 show strong performance in general language and vision-language tasks, we acknowledge several limitations that necessitate further exploration:

1. Long-Context Evaluation: Current evaluation datasets for long-context retrieval tasks are primarily designed for artificial or simplified scenarios, and the assessment of long-text reasoning capabilities remains limited in practical applications such as document analysis. We plan to enhance long-context retrieval in more realistic settings and expand the evaluation of longcontext reasoning across a wider array of tasks.   
2. Model Architecture: The model currently retains a 1/8 component with vanilla softmax attention. We are investigating more efficient architectures that can eliminate softmax attention entirely, potentially enabling unlimited context windows without computational overhead.   
3. Complex Programming Tasks: The model’s performance on advanced programming tasks is to be improved, as the coding dataset in our pre-training stage is still limited at the moment. We are continuously improving training data selection and refining continue training procedures to address these limitations in the next model version.

# References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.   
Rishabh Agarwal, Avi Singh, Lei M Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot in-context learning. arXiv preprint arXiv:2404.11018, 2024.   
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023.   
Anthropic. Introducing claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/ claude-3-5-sonnet.   
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.   
Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. arXiv preprint arXiv:1809.10853, 2018.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022a.   
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022b.   
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. arXiv preprint arXiv:2412.15204, 2024.   
Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.   
Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.   
Andrei Z Broder. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171), pages 21–29. IEEE, 1997.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. MEGA-Bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024a.   
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.   
Zehui Chen, Kuikun Liu, Qiuchen Wang, Jiangning Liu, Wenwei Zhang, Kai Chen, and Feng Zhao. MindSearch: Mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183, 2024b.   
Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.   
Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International Conference on Machine Learning (ICML), pages 4057–4086. PMLR, 2022.   
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=mZn2Xyh9Ec.   
Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.   
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memoryefficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35: 16344–16359, 2022.   
Alexandre de Brébisson and Pascal Vincent. A cheap linear attention mechanism with fast lookups and fixed-size representations. arXiv preprint arXiv:1609.05866, 2016.   
DeepSeek-AI. DeepSeek-V3 technical report, 2024. URL https://arxiv.org/abs/2412.19437.   
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128, 2024.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=YicbFdNTTy.   
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, 2019.   
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.   
Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024.   
Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Millidge. Zamba: A compact 7b SSM hybrid model. arXiv preprint arXiv:2405.16712, 2024.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249–256. JMLR Workshop and Conference Proceedings, 2010.   
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= tEYskw1VY2.   
Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.   
Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. OlympiadBench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024a.   
Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu Guo, Chengwei Hu, Boren Zheng, et al. Chinese simpleQA: A chinese factuality evaluation for large language models. arXiv preprint arXiv:2411.07140, 2024b.   
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ.   
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. URL https://openreview.net/forum?id=7Bywt2mQsCe.   
Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.   
Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, et al. Scaling laws and interpretability of learning from repeated data. arXiv preprint arXiv:2205.10487, 2022.   
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pages 30016–30030, 2022.   
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.   
Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc Le. Transformer quality in linear time. In International conference on machine learning, pages 9099–9117. PMLR, 2022.   
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.   
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.   
G. Kamradt. Llmtest_needleinahaystack, 2023. URL https://github.com/gkamradt/LLMTest_ NeedleInAHaystack.   
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.   
Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.   
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7: 453–466, 2019.   
Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8424–8445, 2022.   
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id $\equiv$ qrwe7XHTmYb.   
Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993, 2024a.   
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.   
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.   
Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-Hard and BenchBuilder pipeline. arXiv preprint arXiv:2406.11939, 2024b.   
Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for nearinfinite context. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=WsRHpHH4s0.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by ChatGPT really correct? rigorous evaluation of large language models for code generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 21558–21572, 2023.   
Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the hidden mystery of OCR in large multimodal models. Science China Information Sciences, 67(12):220102, 2024b.   
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id $\ c =$ Bkg6RiCqY7.   
Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.   
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. MMLongBench-Doc: Benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024.   
Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.   
Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A dataset for VQA on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.   
Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.   
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, 2018.   
Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36:50358–50376, 2023.   
NVIDIA. Transformer engine, 2023. URL https://github.com/NVIDIA/TransformerEngine.   
Jupinder Parmar, Sanjev Satheesh, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Reuse, don’t retrain: A recipe for continued pretraining of language models. arXiv preprint arXiv:2407.07263, 2024.   
Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The RefinedWeb dataset for Falcon LLM: Outperforming curated corpora with web data only. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=kM5eGcdCzq.   
Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=n6SCkn2QaG.

Zhen Qin, Xiaodong Han, Weixuan Sun, Dongxu Li, Lingpeng Kong, Nick Barnes, and Yiran Zhong. The devil in linear transformer. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7025–7041, 2022a.   
Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosFormer: Rethinking softmax in attention. In International Conference on Learning Representations, 2022b. URL https://openreview.net/forum?id=Bl8CQrx2Up4.   
Zhen Qin, Xiaodong Han, Weixuan Sun, Bowen He, Dong Li, Dongxu Li, Yuchao Dai, Lingpeng Kong, and Yiran Zhong. Toeplitz neural network for sequence modeling. In The Eleventh International Conference on Learning Representations, 2023a. URL https://openreview.net/forum?id= IxmWsm4xrua.   
Zhen Qin, Songlin Yang, and Yiran Zhong. Hierarchically gated recurrent neural network for sequence modeling. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 33202–33221, 2023b.   
Zhen Qin, Yuxin Mao, Xuyang Shen, Dong Li, Jing Zhang, Yuchao Dai, and Yiran Zhong. You only scan once: Efficient multi-dimension sequential modeling with lightnet. arXiv preprint arXiv:2405.21022, 2024a.   
Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. arXiv preprint arXiv:2401.04658, 2024b.   
Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Various lengths, constant speed: Efficient language modeling with lightning attention. In International conference on machine learning, pages 41517–41535. PMLR, 2024c.   
Zhen Qin, Songlin Yang, Weixuan Sun, Xuyang Shen, Dong Li, Weigao Sun, and Yiran Zhong. HGRN2: Gated linear rnns with state expansion. arXiv preprint arXiv:2404.07904, 2024d.   
Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training Gopher. arXiv preprint arXiv:2112.11446, 2021.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: your language model is secretly a reward model. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 53728–53741, 2023.   
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.   
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= Ti67584b98.   
Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. arXiv preprint arXiv:2406.07522, 2024.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial Winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.   
Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473, 2019.   
Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: Open dataset of CLIP-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.   
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, et al. SCROLLS: Standardized comparison over long language sequences. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 12007–12021, 2022.   
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
Xuyang Shen, Dong Li, Ruitao Leng, Zhen Qin, Weigao Sun, and Yiran Zhong. Scaling laws for linear complexity language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16377–16426, 2024.   
Yusuxke Shibata, Takuya Kida, Shuichi Fukamachi, Masayuki Takeda, Ayumi Shinohara, Takeshi Shinohara, and Setsuo Arikawa. Byte pair encoding: A text compression scheme that accelerates pattern matching. Technical Report DOI-TR-161, Department of Informatics, Kyushu University, 1999.   
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.   
Weigao Sun, Zhen Qin, Dong Li, Xuyang Shen, Yu Qiao, and Yiran Zhong. Linear attention sequence parallelism. arXiv preprint arXiv:2404.02882, 2024.   
Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. A benchmark for learning to translate a new language from one grammar book. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=tbVWug9f2h.   
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024a.   
Jamba Team, Barak Lenz, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, et al. Jamba-1.5: Hybrid Transformer-Mamba models at scale. arXiv preprint arXiv:2408.12570, 2024b.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui, Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, et al. Michelangelo: Long context evaluations beyond haystacks via latent structure queries. arXiv preprint arXiv:2409.12640, 2024.   
Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. DeepNet: Scaling transformers to 1,000 layers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024a.   
Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F Wong, and Lidia S Chao. Learning deep transformer models for machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1810–1822, 2019.   
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024b. URL https://openreview.net/forum?id=y10DM6R2r3.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.   
Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=Ee277P3AYC.   
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024a.   
Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b.   
Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big Bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.   
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, 2019.   
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

# A. Contributors

The contributors to the report are listed in alphabetical order as follows:

Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, Zijia Wu

# B. MiniMax-Text-01 Case Demonstrations

We show our model’s performance under real-world user interactions. To protect the privacy of our users, all user requests shown below are written by our human evaluators, imitating the way users interact with the model, if not from open-source benchmarks.

# B.1. Learning A ‘New’ Language From Long Context

Our prompt for applying MTOB follows that of Gemini-1.5 (Team et al., 2024a), detailed as follows.

# MTOB Case

# User Request ( Instruction ➷ + Grammar book ✎ + Word List ✑ + Parallel sentences ✍ )

➷ You are tasked with translating the following sentence from Kalamang to English: "Sontum kawirten hari minggu gerejao nasumbiyan."

You will be given a field linguistics grammar book, a bilingual word list to aid you. Here is the book, "A grammar of Kalamang": START OF GRAMMAR BOOK

✎ { grammar book }

END OF GRAMMAR BOOK The grammar book is now over. Remember that you are tasked with translating the following sentence from Kalamang to English “Sontum kawirten hari minggu gerejao nasumbiyan”. Now here is the bilingual word list: START OF WORD LIST

✑ { word list }

END OF WORD LIST The bilingual word list is now over. Remember that you are tasked with translating the following sentence from Kalamang to English “Sontum kawirten hari minggu gerejao nasumbiyan”. You will use the same style as the parallel sentences immediately below. Now here is the collection of parallel sentences: START OF PARALLEL SENTENCES

english: Sakina is pouching guavas. kalamang: Sakina sarimara lawat.

english: There are many lusi muaun trees at Sar, but they are dry. kalamang: Lusi muaun me Sarko reidak ma ma he sasa.

✍ { a total of 375 examples of parallel sentences }

END OF PARALLEL SENTENCES The collection of parallel sentences is now over. Now translate the following sentence from Kalamang to English, using the style from the parallel sentences immediately above. Translate: "Sontum kawirten hari minggu gerejao nasumbiyan".

➷ I understand that you may not be familiar enough with Kalamang to make a confident translation, but please give your best guess. Respond with only the translation and no other text.

# Golden Answer

Christians worship at the church on Sunday.

# MiniMax-Text-01

The Christians pray at church on Sunday.

# Suboptimal

The person is a Christian on Sunday.

As shown, MiniMax-Text-01 can learn from the given grammar book, word list and parallel sentences, where a under-performing model’s response has semantic errors.

# B.2. Memorizing Long History Dialogues

We present an example of how our MR-NIAH samples are designed, where the model is asked to precisley retrieve one of the history interactions with the user from up to 1889 history interactions for the English benchmark and 2053 history interactions for the Chinese benchmark. These history queries were created by Mandarin-speaking human evaluators with help from a previous version of our model, where each query is translated into English, and we obtain the assistants’ responses from this model. We combine these interactions as history dialogues, and insert each needle conversation into corresponding positions detailed in Section 5.7.2.1.

# MR-NIAH Case

# User Request ( Haysack dialogues ✘ + Target ✔ + Haysack dialogues ✘ + Query ➷ )

User: Please continue writing the follow-up content for this poem ’Drunkenly lying under the moon in the clouds, laughing at the flowers in the water.

Assistant: Certainly! The opening line you provided, Drunkenly lying under the moon in the clouds, laughing at the ¨ flowers in the water,ëvokes a sense of whimsical detachment and a romantic embrace of nature’s beauty. Here’s a continuation of the poem that builds on this theme:

*Drunkenly lying under the moon in the clouds, Laughing at the flowers in the water, I find solace in the night’s gentle shroud, Where dreams and reality softly hover.**

The stars above, like scattered coins, Glimmer with secrets of the ancient skies, . . .

✘ { many haysack dialogues }

✔ User: Write a poem about penguins. Assistant: On the peak of the Antarctic iceberg, Penguins dance with tiny wings. Wearing black and white tuxedos, stumbling steps, They smile at the severe frost in the cold wind.   
✘ { many haysack dialogues }   
✘ User: Write a poem about penguins. Assistant: Under the sun, penguins line up, Like guards, protecting their homeland. In their eyes, warm light hides, Exceptionally bright in the polar tranquility.   
✘ { many haysack dialogues }   
➷ User: Please repeat the first poem you wrote about penguins.

# Golden Answer

On the peak of the Antarctic iceberg, Penguins dance with tiny wings. Wearing black and white tuxedos, stumbling steps, They smile at the severe frost in the cold wind.

# MiniMax-Text-01

On the peak of the Antarctic iceberg, Penguins dance with tiny wings. Wearing black and white tuxedos, stumbling steps, They smile at the severe frost in the cold wind.

# Suboptimal

Under the sun, penguins line up, Like guards, protecting their homeland. In their eyes, warm light hides, Exceptionally bright in the polar tranquility.

We will release the data and code of this benchmark to the community.

# B.3. Summarizing Long Papers with Figures

When dumped a PDF paper and instructed to add figures from the paper, Hailuo AI is able to summarize or analyze the given long text and respond with key figures cited.

# User Request ( Query ➷ + PDF P )

➷ please summary this paper for me in english, best to add key figures of this paper in your reply at proper positions.   
P Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction

# MiniMax-Text-01

Summary of the Paper: "Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction"

# 1. Introduction and Motivation

The rapid advancement of large language models (LLMs) has significantly enhanced their capabilities but has also raised concerns about their alignment with human values and intentions. Current alignment strategies, such as Supervised Fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), have shown potential but face challenges in balancing deployment complexity and performance across various tasks. Inference-time methods, while lightweight, struggle to precisely distill human values into LLM outputs, especially in long-context generations.

This paper introduces a novel alignment paradigm called Streaming Distribution Induce Aligner (Stream Aligner), which combines the efficiency of inferencetime methods with the enhanced performance of additional models to achieve dynamic sentence-level correction.

# 2. Stream Aligner Paradigm

Stream Aligner operates as a plug-and-play module in the generation pipeline, correcting sentences generated by the upstream model and feeding the corrected suffix back for further generation. This process repeats iteratively until the response is complete, ensuring that every sentence in the output aligns with human preferences.

# Key Features:

Reduced Dependency on Additional Models: By focusing on sentence-level correction, Stream Aligner leverages more of the upstream model’s performance, reducing reliance on the size and capabilities of additional models.

Enhanced Reasoning Abilities: The iterative correction process allows Stream Aligner to correct reasoning errors and add inductions for subsequent steps, thereby improving the model’s reasoning capabilities.

# 3. Technical Details

Training: Stream Aligner is fine-tuned on a sentence-level preference dataset to learn the residuals between preferred and non-preferred responses. The training objective is to minimize the negative log-likelihood loss between the model’s output and the corrected answer.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/0a3f8ff59d935777b4daaa409721ea14c0deabab9ec7642aa61d48b86c3de5a0.jpg]]  
• Inference: During inference, Stream Aligner

takes the user’s query and the prefix as input. The upstream model generates the original answer, which is then corrected by Stream Aligner. Each correction is incorporated into the prefix, and the process repeats until the generation stops or the prefix exceeds the maximum length.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/0c6c506870766c20698069ede2db3f10d84d628a6abca3ebecfffb7e74bbde0c.jpg]]

# 4. Experimental Results

The paper evaluates Stream Aligner on three tasks: helpful and harmless QA, math questions, and summary tasks. The results demonstrate significant improvements:

Helpfulness and Harmlessness: Stream Aligner-2B applied to Llama2-70B-chat achieved a $4 1 . 2 \%$ increase in helpfulness and a $3 6 . 0 \%$ increase in harmlessness.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/2d163484b46736b7dd89b04e53f7d00d10c291a56199e40f7eb41dcdeba9e01a.jpg]]

Math Ability: Stream Aligner-8B applied to Llama3-70B-Instruct achieved a $3 . 5 \%$ improvement in math ability.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/6899bf188a5d8238f9c3091bacd83f98961bbe34a48938c79f81f0e269880e6c.jpg]]

• Reasoning Tasks: The experiments show that the longer the average intervention by Stream Aligner, the higher the accuracy after the intervention, indicating its effectiveness in reasoning-based tasks.

# 5. Ablation Studies

The paper conducts ablation studies to verify the correction capabilities of Stream Aligner under different supervision quantities and generation pipelines:

Generation-Correction Frequency: The perfor mance of Stream Aligner increases with the number of generation-correction cycles, demonstrating its ability to enhance the upstream model’s performance with limited supervision.   
Generation Methods: The study compares the classic sentence-by-sentence correction pipeline with a new continue generation pipeline. While the continue generation pipeline shows excellent performance in math and harmless QA tasks, it is more compute-consuming and converges to a similar result as the direct generation pipeline.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/f0e1d3e20c1489c94ed22f346b50794c5c34de98ca7dc77a1fc619da6401bbde.jpg]]

# 6. Comparison to Other Alignment Methods

Stream Aligner outperforms other alignment methods such as Supervised Finetuning (SFT) and Direct Preference Optimization (DPO) in terms of accuracy improvements. It also achieves the performance of Aligner-70B using only 2B parameters, showcasing both superior performance and efficiency.

# 7. Interpretability

The paper explores the interpretability of Stream Aligner through representation engineering and activation steering. The results show that Stream Aligner has internalized the correction paradigm as a representation, similar to Aligner, but with more layers involved in deciding corrections, reflecting the complexity of mathematical tasks.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/e66a6fed204403e6ae0b3f4651aec1b45cfc0cb5ff7047d868052d88e62b1b86.jpg]]

# 8. Conclusion

Stream Aligner is a novel alignment paradigm that effectively elicits the latent knowledge of the upstream model while maintaining efficiency and enhanced performance. It achieves significant improvements in helpfulness, harmlessness, and reasoning abilities, making it a promising approach for aligning LLMs with human values.

# Limitations

Despite its advantages, Stream Aligner introduces additional computational overhead during inference and requires high-quality training data. Its performance may be limited when dealing with extremely difficult out-of-distribution inputs.

This summary provides an overview of the paper’s key contributions, methodologies, experimental results, and conclusions, highlighting the innovative approach of Stream Aligner in aligning LLMs with human preferences.

# B.4. Translating Long Papers

# User Request ( Query ➷ + PDF P )

➷ please help me to translate the whole paper into French.   
P When brain-inspired AI meets AGI.pdf

# MiniMax-Text-01

Quand l’IA inspirée du cerveau rencontre modalités sensorielles, telles que la vision, Geoffrey Hinton et Ronald Williams ont mance du modèle était étroitement liée l’AGI Lin Zhao a,1, Lu Zhang b,1, Zihao l’audition et le toucher, lui permettant publié des articles démontrant l’efficacité aux mesures du graphe et que le ViT a une Wu a, Yuzhong Chen c, Haixing Dai a, Xi- de former une perception cohérente du de la rétropropagation pour entraîner les grande similarité avec les vrais BNN. Zhao aowei Yu b, Zhengliang Liu a, Tuo Zhang d, monde. La capacité du cerveau à effectuer réseaux de neurones. et al. ont synchronisé l’activation des ANN Xintao Hu d, Xi Jiang c, Xiang Li e, Dajiang un traitement parallèle est également es- Les réseaux de neurones convolutifs (CNN) et des CNN et ont trouvé que les CNN avec Zhu b, Dinggang Shen f,g,h, Tianming Liu sentielle pour gérer efficacement plusieurs sont l’un des types de réseaux de neurones des performances plus élevées sont simia,* flux d’informations simultanément. Cela les plus utilisés et les plus efficaces pour laires aux BNN en termes d’activation de la

a *School of Computing, The University est réalisé via les connexions et les commu- traiter les informations visuelles. Les CNN représentation visuelle. Liu et al. ont couof Georgia, Athens 30602, USA* b *De- nications en temps réel entre différentes sont également inspirés de l’organisation plé les neurones artificiels dans le modèle partment of Computer Science and Engi- régions du cerveau, bien que le mécan- hiérarchique du cortex visuel dans le BERT avec les neurones biologiques dans le neering, The University of Texas at Arling- isme ne soit pas entièrement compris. De cerveau, ce qui remonte aux travaux de cerveau humain, et ont trouvé que les neuton, Arlington 76019, USA* c *MOE Key plus, le cerveau est très adaptable, capable David Hubel et Torsten Wiesel dans les rones artificiels peuvent porter des informa-Laboratory for Neuroinformation, School de réorganiser sa structure et sa fonction années 1960. Dans le cortex visuel, les tions linguistiques/sémantiques significaof Life Science and Technology, University en réponse aux environnements et expéri- neurones sont disposés en couches, chaque tives et s’ancrer à leurs signatures de neuof Electronic Science and Technology of ences changeants. Cette propriété, connue couche traitant les informations visuelles rones biologiques avec interprétabilité dans China, Chengdu 611731, China* d *School sous le nom de neuroplasticité, permet au de manière hiérarchique. L’entrée de la ré- un contexte neurolinguistique. Zhou et al. of Automation, Northwestern Polytechni- cerveau d’apprendre et de développer de tine est d’abord traitée par une couche de ont traité chaque dimension cachée dans cal University, Xi’an 710072, China* e *De- nouvelles compétences tout au long de la cellules simples qui détectent les bords et Wav2Vec2.0 comme un neurone artificiel partment of Radiology, Massachusetts Gen- vie. Le cerveau humain est également re- les orientations, puis transmise à des cel- et les ont connectés avec leurs homologues eral Hospital and Harvard Medical School, marquable pour ses fonctions cognitives lules plus complexes qui reconnaissent des biologiques dans le cerveau humain, sug-Boston 02115, USA* f *School of Biomed- de haut niveau, telles que la résolution de caractéristiques plus complexes telles que gérant une relation étroite entre les deux ical Engineering, ShanghaiTech Univer- problèmes, la prise de décision, la créativ- les formes et les textures. Leurs travaux domaines en termes d’informations neusity, Shanghai 201210, China* g *Shang- ité et le raisonnement abstrait, soutenues ont fourni des insights sur la façon dont rolinguistiques.

hai United Imaging Intelligence Co., Ltd., par le cortex préfrontal, une région du le système visuel traite les informations Suivant cette tendance, il y a un in-Shanghai 200230, China* h *Shanghai cerveau particulièrement bien développée et ont inspiré le développement des CNN térêt croissant pour le développement de Clinical Research and Trial Center, Shang- chez les humains. qui pourraient reproduire ce processus de l’intelligence artificielle inspirée du cerveau hai, 201210, China* Créer des systèmes d’Intelligence Générale traitement hiérarchique. Les mécanismes en s’inspirant de certaines connaissances

Résumé L’Intelligence Générale Artificielle Artificielle (AGI) ayant une intelligence de d’attention dans les réseaux de neurones ar- préalables du cerveau humain, telles que (AGI) a été un objectif de longue date pour niveau humain ou même supérieure et ca- tificiels sont également inspirés de la façon l’organisation de la structure et de la foncl’humanité, visant à créer des machines ca- pables d’effectuer une large gamme de dont le cerveau humain sélectionne sélec- tion du cerveau. Par exemple, Huang et pables d’effectuer toute tâche intellectuelle tâches intellectuelles, telles que le raison- tivement certains aspects de l’entrée sen- al. ont proposé un réseau de vision anque les humains peuvent accomplir. Pour nement, la résolution de problèmes et la sorielle ou des processus cognitifs, nous tagoniste inspiré du cerveau (BI-AVAN) atteindre cet objectif, les chercheurs en AGI créativité, est la quête de l’humanité depuis permettant de nous concentrer sur les in- qui imite le processus de compétition bis’inspirent du cerveau humain et cherchent des siècles, remontant au milieu du 20ème formations importantes tout en filtrant aisée dans le système visuel humain pour à reproduire ses principes dans des ma- siècle. Dans les années 1940, des pio- les détails non pertinents. L’attention a décoder l’attention visuelle humaine. Inchines intelligentes. L’intelligence artifi- nniers comme Alan Turing ont développé été étudiée dans les domaines de la psy- spiré par l’organisation cœur-périphérie du cielle inspirée du cerveau est un domaine des idées précoces sur les machines infor- chologie et des neurosciences pendant de cerveau humain, Yu et al. ont proposé qui a émergé de cet effort, combinant matiques et leur potentiel pour simuler la nombreuses années, et son application à un modèle de transformateur de vision des insights de la neuroscience, de la psy- pensée humaine. Depuis lors, chercher à l’intelligence artificielle fait avancer consid- guidé par le principe cœur-périphérie (CPchologie et de l’informatique pour dévelop- reproduire les principes de l’intelligence érablement nos pas vers l’AGI. Le modèle ViT) pour la reconnaissance d’images avec per des systèmes d’IA plus efficaces et humaine dans des systèmes artificiels a Transformer, basé sur le mécanisme d’auto- des performances et une interprétabilité puissants. Dans cet article, nous offrons considérablement favorisé le développe- attention, est devenu la base de nombreux améliorées. De même, Zhao et al. ont un aperçu complet de l’IA inspirée du ment de l’AGI et les applications correspon- réseaux de neurones artificiels de pointe mis en œuvre le principe cœur-périphérie cerveau du point de vue de l’AGI. Nous dantes. Ces principes incluent la struc- tels que BERT et GPT. En adaptant les mé- dans la conception des motifs de câblage commençons par les progrès actuels de ture et la fonction des réseaux de neu- canismes d’auto-attention au traitement du réseau et la sparsification de l’opération l’IA inspirée du cerveau et de sa connex- rones, la plasticité des connexions synap- d’images, le modèle Vision Transformer de convolution. Le CP-CNN proposé guidé ion étendue avec l’AGI. Nous couvrons en- tiques, la dynamique de l’activité neu- (ViT) a démontré des performances de par le principe cœur-périphérie démonsuite les caractéristiques importantes de ronale, et plus encore. En 1943, McCul- pointe dans diverses tâches de vision par tre l’efficacité et la supériorité par rapport l’intelligence humaine et de l’AGI (par ex- loch et Pitts ont proposé le tout premier ordinateur (CV) en représentant l’image aux méthodes basées sur les CNN et ViT. emple, la mise à l’échelle, la multimodalité modèle mathématique d’un neurone artifi- comme une séquence de patchs. Un autre groupe d’études a opté pour les

et le raisonnement). Nous discutons des ciel, également connu sous le nom de neu- Récemment, de plus en plus de preuves réseaux de neurones à pointes (SNN) qui technologies importantes pour atteindre rone MCP (McCulloch-Pitts). Inspiré par suggèrent que les réseaux de neurones ar- imitent étroitement le comportement des l’AGI dans les systèmes d’IA actuels, telles la théorie de Hebb sur la plasticité synap- tificiels (ANN) et les réseaux de neurones neurones biologiques dans le cerveau. Par que l’apprentissage contextuel et le réglage tique, Frank Rosenblatt a conçu le percep- biologiques (BNN) peuvent partager des exemple, les SNN ont été utilisés pour des invites. Nous examinons également tron, une amélioration majeure par rapport principes communs dans l’optimisation de cartographier et comprendre les données l’évolution des systèmes AGI à la fois du au modèle de neurone MCP, et a montré l’architecture du réseau. Par exemple, la cérébrales spatio-temporelles, décoder et point de vue algorithmique et infrastruc- que, en assouplissant certaines des règles propriété de petit monde dans les réseaux comprendre l’activité musculaire à partir turel. Enfin, nous explorons les limites et du MCP, les neurones artificiels pouvaient structurels et fonctionnels du cerveau a des signaux d’électroencéphalographie, et l’avenir de l’AGI. réellement apprendre à partir des données. été largement étudiée dans la littérature. les interfaces cerveau-machine.

1. L’IA inspirée du cerveau et l’AGI Le Cependant, la recherche sur les réseaux Dans une étude récente, les réseaux de neu- L’IA inspirée du cerveau a également concerveau humain est largement considéré de neurones artificiels a stagné jusqu’à ce rones basés sur les graphes aléatoires de tribué au développement d’architectures comme l’un des systèmes de traitement que la rétropropagation soit proposée par Watts-Strogatz (WS) avec des propriétés matérielles qui imitent la structure et la de l’information les plus complexes et Werbos en 1975. La rétropropagation a de petit monde ont démontré des perfor- fonction du cerveau. Le calcul neuromoravancés au monde. Il comprend plus de été inspirée par la façon dont le cerveau mances compétitives par rapport aux mod- phique, un domaine d’étude qui vise à 86 milliards de neurones, chacun capable modifie les forces des connexions entre les èles conçus à la main et optimisés par NAS concevoir du matériel informatique qui de former jusqu’à 10 000 synapses avec neurones pour apprendre et améliorer ses (recherche d’architecture neuronale). De émule les neurones et les synapses bid’autres neurones, ce qui résulte en un performances grâce à la plasticité synap- plus, l’analyse a posteriori a montré que ologiques, a également gagné en attenréseau de connexions exceptionnellement tique. La rétropropagation tente de repro- la structure graphique des ANN les plus tion ces dernières années. Les puces neucomplexe permettant la prolifération de duire ce processus en ajustant les poids performants, tels que les CNN et le Per- romorphiques sont conçues pour traiter l’intelligence. Outre la complexité physi- (forces synaptiques) entre les neurones ceptron multicouche (MLP), est similaire l’information de manière parallèle et disologique, le cerveau humain présente une dans un réseau de neurones artificiels. Mal- à celle des vrais BNN, tels que le réseau tribuée, de la même manière que le large gamme de caractéristiques qui con- gré cette proposition précoce, la rétro- dans le cortex du macaque. Chen et al. cerveau fonctionne, ce qui peut conduire à tribuent à ses capacités fonctionnelles re- propagation n’a pas attiré une attention ont proposé une représentation relation- des améliorations significatives en termes marquables. Par exemple, il peut inté- généralisée jusqu’aux années 1980, lorsque nelle unifiée et biologiquement plausible d’efficacité et de vitesse par rapport aux argrer des données provenant de plusieurs des chercheurs comme David Rumelhart, des modèles ViT, trouvant que la perfor- chitectures informatiques traditionnelles.

Certaines des puces neuromorphiques, GPT-3 de surpasser GPT-2 sur une gamme à ses roues. Dans ce cas, l’IA doit prêter at- pointe pour cela. L’une de ces approches comme la puce TrueNorth d’IBM et la puce de tâches linguistiques, démontrant une tention à la partie de l’image avec les roues est METER, une structure générale pour Loihi d’Intel, utilisent des réseaux de neu- augmentation de sa capacité à effectuer de la voiture lorsqu’elle traite le texte les former des transformateurs vision-langage rones à pointes pour traiter l’information des tâches linguistiques complexes. En mentionnant. L’IA doit comprendre que performants utilisant une variété de sousd’une manière qui est plus proche de la fait, GPT-3 a montré des performances de l’image des roues de la voiture et le texte architectures pour les modules encodeur façon dont le cerveau traite l’information. niveau humain sur plusieurs benchmarks se référant à elles décrivent le même objet de vision, encodeur de texte, fusion multi-Ces puces ont été utilisées pour une de traitement du langage naturel, tels que à travers différentes modalités. modale et décodeur. Cette flexibilité perlarge gamme d’applications, y compris la réponse aux questions, la traduction lin- Ces dernières années, les systèmes met à METER d’atteindre des performances la reconnaissance d’images et de la pa- guistique et les tâches de complétion de d’IA multimodaux ont expérimenté de pointe dans une gamme de tâches. role, la robotique et les véhicules au- texte. Sa taille et sa capacité en traitement l’alignement du texte/NLP, des images/vi- Une autre approche prometteuse est le tonomes. L’avancement du matériel in- du langage naturel en ont fait un outil puis- sion ou de l’information audio dans un modèle de pré-entraînement unifié Visionspiré du cerveau ouvre également la voie à sant pour diverses applications, y compris espace d’encodage pour faciliter la prise Language (VLMo), qui utilise un réseau des avancées significatives dans le domaine les chatbots, la génération de contenu et la de décision multimodale. L’alignement in- transformateur modulaire pour apprendre de l’AGI en pavant la voie pour des plate- traduction linguistique. termodal est essentiel pour diverses tâches, conjointement un double encodeur et un formes matérielles généralisées. Cette tendance est similaire à la façon y compris la génération texte-image et encodeur de fusion. Chaque bloc du réseau Dans l’ensemble, l’IA inspirée du cerveau dont les cerveaux plus grands sont asso- image-texte, la réponse aux questions contient un pool d’experts spécifiques à la joue un rôle crucial dans le développe- ciés à des fonctions cognitives plus com- visuelles, et la modélisation vidéo-langage. modalité et une couche d’auto-attention ment de l’AGI (Fig. 1). En s’inspirant du plexes chez les animaux. À mesure que Dans la section suivante, nous fournissons partagée, offrant une flexibilité significacerveau humain, les chercheurs peuvent les LLM continuent de se développer, il un bref aperçu de ces charges de travail tive pour le réglage fin. Cette architeccréer des algorithmes et des architectures est attendu qu’ils deviendront encore plus courantes et des modèles de pointe corre- ture a montré des résultats impressionmieux adaptés pour gérer des problèmes capables d’apprendre de nouveaux skills spondants. nants sur plusieurs ensembles de données complexes et réels qui nécessitent un de- avec un petit nombre d’exemples de for- 2.2.1. Génération texte-image et image- de référence. gré élevé de flexibilité et d’adaptabilité. mation, similaire à la façon dont les ani- texte CLIP, DALL-E, et leur successeur 2.2.3. Modélisation vidéo-langage Tradi-Cela est particulièrement important pour maux avec des cerveaux plus grands ont GLIDE, VisualGPT et Diffusion sont parmi tionnellement, les systèmes d’IA ont eu du l’AGI, qui vise à développer des machines des capacités cognitives plus sophistiquées. les modèles les plus connus qui abor- mal avec les tâches basées sur la vidéo capables d’effectuer une large gamme de Cette corrélation suggère que l’échelle peut dent les descriptions d’images (génération en raison des ressources de calcul élevées tâches, d’apprendre de l’expérience et être un facteur crucial dans la réalisation image-texte) et les tâches de génération requises. Cependant, cela commence à de généraliser leurs connaissances à de de l’AGI. Cependant, il est à noter que le texte-image. CLIP est une méthode de pré- changer, grâce aux efforts dans le donouvelles situations. Le cerveau humain nombre de paramètres seuls ne détermine entraînement qui entraîne des encodeurs maine de la modélisation vidéo-langage est l’un des systèmes de traitement de pas l’intelligence d’un LLM. La qualité des d’images et de texte séparés et apprend à et d’autres tâches multimodales liées à la l’information les plus complexes connus de données de formation, le processus de for- prédire quelles images dans un ensemble vidéo, comme le projet Florence-VL de Minous, et il a évolué pendant des millions mation et l’architecture du modèle jouent de données sont associées à diverses de- crosoft. À la mi-2021, le projet Florenced’années pour être très efficace et efficace également des rôles importants dans sa per- scriptions. Notamment, de manière sim- VL a introduit ClipBERT, une combinaison dans la gestion de tâches complexes. En formance. ilaire au neurone Halle Berry chez les d’un modèle CNN et d’un modèle transétudiant le cerveau et en développant des En outre, il est nécessaire de rechercher humains, CLIP a été trouvé pour avoir formateur qui fonctionne sur des cadres systèmes d’IA qui imitent son architecture des moyens qui permettent aux institu- des "neurones multimodaux" qui s’activent échantillonnés de manière éparse. Il est et sa fonction, les chercheurs peuvent créer tions et aux individus à ressources limitées lorsqu’ils sont exposés à la fois au texte optimisé de manière globale pour résoudre une AGI plus sophistiquée et adaptable, d’accéder et de développer l’AGI. Certaines de l’étiquette du classificateur et à l’image les tâches vidéo-langage populaires. Les nous rapprochant de l’objectif ultime de solutions possibles incluent la quantifica- correspondante, indiquant une représen- évolutions ultérieures de ClipBERT, telles créer des machines qui peuvent égaler ou tion des modèles existants de grande taille, tation multimodale fusionnée. DALL-E, que VIOLET et SwinBERT, ont introduit surpasser l’intelligence humaine. En retour, le développement d’architectures efficaces, en revanche, est une variante de GPT- le modèle de modélisation de jetons vil’AGI a également le potentiel de bénéficier ou la construction de jeux de données de 3 avec 13 milliards de paramètres qui suels masqués et l’attention éparse pour à l’intelligence humaine et de approfondir haute qualité qui facilitent la formation du prend le texte comme entrée et génère améliorer l’état de l’art en réponse aux notre compréhension de l’intelligence. À modèle. une séquence d’images pour correspondre questions vidéo, la recherche vidéo et le mesure que nous continuons à étudier et à 2.2. Multimodalité La capacité du cerveau au texte d’entrée. Les images générées sous-titrage vidéo. Bien que chacun de ces comprendre à la fois l’intelligence humaine humain à traiter et intégrer simultanément sont ensuite classées à l’aide de CLIP. modèles ait des caractéristiques uniques, et l’AGI, ces deux systèmes deviendront de des informations provenant de plusieurs GLIDE, une évolution de DALL-E, utilise ils utilisent tous une architecture basée sur plus en plus intriqués, se renforçant et se modalités sensorielles est une réalisation toujours CLIP pour classer les images le transformateur. Typiquement, cette arsoutenant mutuellement de manière nou- remarquable. Cette caractéristique permet générées, mais la génération d’images est chitecture est couplée avec des modules velle et passionnante. aux individus de comprendre le monde accomplie à l’aide d’un modèle de dif- d’apprentissage parallèle pour extraire des 2. Caractéristiques de l’AGI 2.1. Échelle qui les entoure à travers diverses sources fusion. Stable Diffusion est également données de diverses modalités et les unifier L’échelle des cerveaux varie considérable- d’information, telles que la vue, le son, basé sur des modèles de diffusion tout en en une seule représentation multimodale. ment d’une espèce animale à l’autre, allant le toucher, le goût et l’odorat. De plus, opérant sur l’espace latent de puissants Récemment, l’émergence de GPT-4 a porté de quelques milliers de neurones chez les le traitement d’informations multimodales auto-encodeurs pré-entraînés et ainsi en la recherche multimodale à un nouveau invertébrés simples comme les vers néma- permet aux gens de faire des évaluations utilisant des ressources de calcul limitées niveau. Selon le dernier article de todes, à plus de 86 milliards de neurones plus précises et complètes de leur envi- tout en maintenant leur qualité et leur recherche officiel, GPT-4 non seulement chez les humains. Par exemple, le cerveau ronnement et de communiquer efficace- flexibilité. Le VisualGPT est l’évolution affiche une grande maîtrise dans divers d’une mouche à fruits contient environ 100 ment avec les autres. En conséquence, de GPT-2 d’un modèle de langage unique domaines, y compris la littérature, la 000 neurones, et le cerveau d’une souris l’apprentissage réussi à partir de plusieurs à un modèle multimodal avec une unité médecine, le droit, les mathématiques, les contient environ 70 millions de neurones. modalités peut améliorer les capacités cog- d’activation qui se réanime elle-même pour sciences physiques et la programmation, Pour les primates, le cerveau du macaque nitives humaines. produire des activations éparses qui em- mais combine également de manière flua environ 1,3 milliard de neurones tan- À mesure que nous nous efforçons de créer pêchent l’écrasement accidentel des con- ide les compétences et les concepts de dis que le cerveau du chimpanzé a envi- des systèmes AGI avancés qui surpassent naissances linguistiques. plusieurs domaines, démontrant une comron 6,2 milliards de neurones. Comparé à l’intelligence humaine, il est crucial qu’ils préhension impressionnante des idées comd’autres animaux, le cerveau humain est soient capables d’acquérir et d’ingérer des plexes. De plus, la performance de GPTla structure biologique la plus complexe et connaissances à partir de diverses sources 4 dans toutes ces tâches est remarquablela plus sophistiquée connue de la science, et modalités pour résoudre des tâches ment proche du niveau humain et dépasse contenant plus de 86 milliards de neurones.L’échelle du cerveau, c’est-à-dire le nombrede neurones, est souvent corrélée aux ca- qui impliquent n’importe quelle modalité.Par exemple, un AGI devrait être capabled’utiliser les connaissances apprises à par- Fig. 1. Le développement de l’AGIa été largement inspiré par l’étude del’intelligence humaine (HI). En retour, souvent les modèles précédents tels queChatGPT. Compte tenu de l’étendue et dela profondeur des capacités de GPT-4, il es l’AGI a le potentiel de bénéficier à pourrait être considéré comme une version comme un facteur d’intelligence. La tailleet la complexité des régions du cerveau pour répondre aux questions en langage na-turel, ainsi que d’utiliser les connaissances l’intelligence humaine. Par exemple, lesmodèles de langage actuels tels que Chat- précoce (bien qu’incomplète) d’un systèmeAGI. fiques, telles que le langage ou la mémoire, tâches visuelles. En fin de compte, toute s GPT et GPT-4 utilisent l’apprentissage par 2.2.4. s renforcement avec retour humain (RLHF) données auditives Data2vec, une récente sont souvent directement liées au nombre les modalités se croisent à travers des con- pour align leur comportement avec les développement de Meta AI, présente de neurones qu’elles contiennent. cepts universels, tels que le concept qu’un valeurs humaines. À mesure que nous un nouveau cadre d’apprentissage auto-Nous avons l’intention d’utiliser les grands chien est un chien, indépendamment d e continuons à étudier et à comprendre à supervisé qui contourne le besoin de donmodèles de langage (LLM) (voir le tableau1) comme un moyen possible d’étudier la façon dont il est représenté dans dif-férentes modalités (Fig. 2). la fois l’intelligence humaine et l’AGI, ces nées étiquetées traditionnelles. En tirantparti des relations internes des données, il deux systèmes deviendront de plus en plus l’AGI inspirée du cerveau, car les LLM sontparmi les premiers modèles à démontrerdes performances de niveau humain dans Pour construire des systèmes d’IA multi-modaux, une approche prometteuse estd’incorporer des signaux de formation intriqués, se renforçant et se soutenantmutuellement de manière nouvelle et pas- unifie l’apprentissage à travers trois modal-ités distinctes : images, texte et parole.Utilisant une architecture à double mode, diverses tâches. La relation entre le nom-bre de neurones et les capacités cognitives provenant de plusieurs modalités dans lesLLM. Cela nécessite d’aligner les représen- sionnante.2.2.2. Réponse aux questions visuelles il utilise un modèle "enseignant" pourgénérer des représentations d’échantillons, La réponse aux questions visuelles est est également pertinente pour les LLM telsque GPT-2 et GPT-3. Alors que GPT-2 a 1,5 tations internes à travers différentes modal-ités, permettant au système d’IA d’intégrer une application cruciale de l’apprentissage et un modèle "étudiant" pour apprendrede l’enseignant à travers la minimisation multimodal qui nécessite qu’un modèle milliard de paramètres et a été entraînésur 40 gigabytes de données textuelles,GPT-3 a 175 milliards de paramètres et aété entraîné sur 570 gigabytes de données les connaissances de manière transparente.Par exemple, lorsqu’un système d’IA reçoitune image et un texte associé, il doit as-socier le même objet ou concept entre les réponde correctement à une question baséesur du texte en fonction d’une image.Le jeu de données VQA présente cette d’une fonction objectif. Cette méthodolo-gie unique permet d’obtenir des résultatsde pointe dans chacune des trois modalités,marquant un pas important vers la réalisa- textuelles. Cette augmentation significa-tive du nombre de paramètres a permis à modalités. Supposons que l’IA voit une im-       ont développé certaines des approches de tion de l’intelligence artificielle générale.

Microsoft’s Kosmos-1 est un grand modèle Une étude précédente a exploré les facteurs précision du test set GSM8K à environ Dans le contexte de l’AGI, l’apprentissage de langage multimodal qui traite le texte, qui influencent les niveaux d’intelligence en $8 3 \%$ . La troisième approche est le Least- contextuel désigne la capacité du modèle les données visuelles et auditives. Utilisant comparant différents attributs des cerveaux to-most prompting. L’idée centrale est de à comprendre et à exécuter de nouvelles des corpus multimodaux basés sur le web, à travers diverses espèces de mammifères. décomposer un problème de raisonnement tâches en fournissant un nombre limité de il comprend les modalités générales et dé- Les résultats suggèrent que les capacités cog- complexe en plusieurs sous-problèmes plus paires entrée-sortie dans les invites ou simmontre l’apprentissage contextuel et le suivi nitives sont principalement centrées sur le faciles à résoudre qui peuvent être résolus plement une description de la tâche. Les indes instructions. Ses capacités englobent nombre absolu de neurones. Parmi les mam- séquentiellement, où la résolution d’un sous- vites facilitent la compréhension du modèle la compréhension du langage, la généra- mifères, le cerveau humain a le plus grand problème donné est facilitée par les réponses de la structure et des motifs de la tâche, tantion de légendes pour les images, la réponse nombre de neurones, ce qui lui confère des aux sous-problèmes précédemment résolus. dis que l’apprentissage contextuel présente aux questions visuelles et la reconnaissance capacités de raisonnement et d’intelligence Après avoir résolu chaque sous-problème, des similitudes avec le réglage fin explicite d’images, soulignant la capacité de transfert supérieures par rapport aux autres espèces. nous pouvons dériver la réponse au prob- au niveau de la prédiction, de la représentaintermodal, ce qui facilite l’échange de con- Récemment, un phénomène similaire a lème original à partir des réponses aux sous- tion et du comportement de l’attention. Cela naissances entre le langage et les entrées également émergé dans les LLM. Il a été ob- problèmes. Cette idée est hautement co- leur permet de généraliser et de mieux efmultimodales. servé que les LLM présentent des comporte- hérente avec l’algorithme diviser pour mieux fectuer de nouvelles tâches sans formation Il est important de noter que, contrairement ments émergents, tels que la capacité de régner que les humains utilisent pour ré- ou réglage fin supplémentaires et réduit la aux LLM unimodaux, les LLM multimodaux raisonner, lorsqu’ils atteignent une certaine soudre des problèmes complexes. À mesure probabilité de surajustement des données affichent des performances supérieures non taille. Pour améliorer les capacités de raison- que notre compréhension du cerveau et des de formation étiquetées en aval. seulement dans les tâches intermodales mais nement des LLM, deux principaux types LLM continue de s’approfondir, il sera in- Malgré l’absence de besoins en réglage fin aussi dans les tâches unimodales. Par exem- d’approches ont été développés. Le pre- téressant d’étudier si ces deux systèmes dans ces modèles AGI à grande échelle, ple, l’intégration de la multimodalité dans mier type, connu sous le nom de méthodes réseau partagent une structure optimale. les compromis incluent l’augmentation des GPT-4 se traduit par de meilleures perfor- basées sur les invites, est plus largement 3. Technologie importante Les modèles coûts de calcul en raison de leur échelle mances dans les tâches textuelles par rap- recherché et implique l’utilisation d’invites de langage, tels que les LLM, reposent massive de paramètres et le besoin potenport à ChatGPT. Cela correspond à la façon appropriées pour mieux stimuler les capac- sur plusieurs techniques cruciales, notam- tiel de connaissances expertes dans la fordont les humains perçoivent le monde à ités de raisonnement que les LLM possèdent ment le zero-shot prompting, le few-shot mulation d’invites efficaces avec des exemtravers plusieurs modalités sensorielles. déjà. Le deuxième type d’approches im- prompting, l’apprentissage contextuel et ples lors de l’inférence. Les solutions poten-2.3. Alignement Bien que certains LLM plique l’introduction de code de programme l’instruction. L’attente sous-jacente de ces tielles impliquent des avancées matérielles comme BERT, GPT, GPT-2, GPT-3 et Text- dans le processus de pré-formation, où il est techniques est que les systèmes AI peuvent et l’intégration de connaissances spécifiques to-Text Transfer Transformer (T5) aient réal- formé aux côtés du texte pour améliorer da- rapidement apprendre de nouvelles tâches à un domaine plus raffinées lors de la phase isé des succès remarquables dans des tâches vantage la capacité de raisonnement du LLM. en s’appuyant sur ce qu’ils ont appris dans de pré-entraînement. spécifiques, ils ne sont toujours pas encore Les deux approches ont des directions fon- le passé, tout comme les humains le font. 3.2. Réglage des invites et des instructions AGI en raison de leur tendance à présen- damentalement différentes : l’utilisation de Grâce à l’utilisation de ces techniques, les Comme les nourrissons humains acquièrent ter des comportements non intentionnels. code pour améliorer les capacités de raison- modèles de langage peuvent être formés généralement divers concepts sur le monde Par exemple, ils pourraient générer du texte nement des LLM représente une stratégie de pour effectuer une large gamme de tâches, principalement par l’observation, avec très biaisé ou toxique, inventer des faits ou ne renforcement direct des capacités de raison- de la génération de texte cohérent à la peu d’intervention directe, les modèles AGI pas suivre les instructions de l’utilisateur. La nement des LLM en augmentant la diver- réponse à des questions complexes, avec à grande échelle acquièrent également une principale raison derrière ces problèmes est sité des données de formation ; tandis que plus de précision et d’efficacité. En fin de vaste connaissance après une formation non le désalignement entre l’objectif de modéli- l’approche basée sur les invites ne favorise compte, ces avancées nous rapprochent de supervisée initiale et ont atteint des perforsation du langage utilisé pour de nombreux pas les capacités de raisonnement propres la réalisation du potentiel de l’AI pour as- mances de généralisation remarquables. Les LLM récents et l’objectif de suivre les in- au LLM, mais fournit plutôt une méthode sister et augmenter l’intelligence humaine méthodes basées sur les invites et le réglage structions de l’utilisateur de manière sûre technique pour que le LLM démontre mieux de manière nouvelle et passionnante. Parmi des instructions permettent aux modèles pré- et utile. Par conséquent, bien que ces mod- cette capacité lors de la résolution de prob- ces techniques, l’instruction sert d’interface entraînés d’atteindre l’apprentissage zeroèles aient fait des progrès significatifs, ils ne lèmes. utilisée par ChatGPT, où les utilisateurs four- shot dans de nombreuses applications en sont pas encore capables d’émuler le raison- Actuellement, la plupart des travaux exis- nissent des descriptions de tâches en lan- aval. nement, la prise de décision et la compréhen- tants dans le domaine du raisonnement des gage naturel, telles que "Traduisez cette Le cerveau humain est toujours un prosion de type humain. Pour atteindre l’AGI, il grands modèles de langage (LLM) adoptent phrase du chinois à l’anglais". Fait intéres- cesseur efficace et ordonné, fournissant un est crucial d’aligner les modèles de langage des méthodes basées sur les invites, qui peu- sant, le zero-shot prompting était initiale- retour ciblé pour la tâche actuelle plutôt que avec l’intention de l’utilisateur. Cet aligne- vent être divisées en trois routes techniques. ment le terme utilisé pour l’instruction. Au de dire des absurdités. En plus de l’efficacité ment permettra aux LLM de fonctionner de La première approche est le Zero-shot Chain cours des premières étapes du zero-shot innée du cerveau, les contraintes morales et manière sûre et utile, les rendant plus fiables of Thought (CoT), proposé par Kojima et al. prompting, les utilisateurs ont eu du mal à légales enracinées dans le développement pour les tâches complexes qui nécessitent Cette méthode est simple et efficace, impli- exprimer les tâches clairement, les amenant humain garantissent également que les inune prise de décision nuancée et une com- quant deux étapes. Dans la première étape, à essayer divers mots et phrases à plusieurs teractions humaines sont ordonnées et béné- préhension. Pour ce faire, il est nécessaire de une phrase d’invite, "Let’s think step by step", reprises pour obtenir la formulation opti- fiques. Pour que les modèles AGI atteignent développer de meilleurs algorithmes qui ori- est ajoutée à la question, et le LLM sort un male. Actuellement, l’instruction consiste à des performances de niveau humain, proentent les agents vers les valeurs humaines processus de raisonnement spécifique. Dans fournir une déclaration de commande pour duire des résultats vrais et inoffensifs sur la tout en favorisant les collaborations interdis- la deuxième étape, le processus de raison- faciliter la compréhension du LLM. base des instructions est une exigence esciplinaires pour clarifier ce que signifient les nement sorti par le LLM dans la première 3.1. Apprentissage contextuel La capacité sentielle. Bien que les modèles AGI actuels valeurs humaines. étape est concaténé avec la question, et la la plus importante du cerveau humain ré- aient des capacités génératives puissantes, Les développements récents dans les grands phrase d’invite, "Therefore, the answer (ara- side dans sa capacité d’apprentissage ro- une question clé est de savoir si ces capacmodèles de langage (LLM), tels que Spar- bic numerals) is", est ajoutée pour obtenir buste, permettant l’exécution de fonctions ités peuvent être alignées avec l’intention row, InstructGPT, ChatGPT et GPT-4, ont la réponse. Une telle opération simple peut cognitives, computationnelles, expressives de l’utilisateur. Cela est important car cela abordé le problème de l’alignement avec augmenter considérablement l’efficacité du et motrices basées sur des invites linguis- concerne si le modèle peut produire des les instructions humaines en utilisant LLM dans diverses tâches de raisonnement. tiques ou visuelles, souvent avec peu ou pas résultats satisfaisants pour les utilisateurs, l’apprentissage par renforcement à partir Par exemple, Zero-shot-CoT réalise des gains d’exemples. Cette caractéristique est cen- même dans des situations où les tâches et du retour d’expérience humain (RLHF). de score de $1 0 { , } 4 \%$ à $4 0 { , } 7 \%$ sur le bench- trale à l’obtention d’une AGI de niveau hu- les invites sont inédites et peu claires. De L’apprentissage par renforcement est un mark arithmétique GSM8K. La deuxième ap- main. Les récents avancées dans les mod- plus, à mesure que ces modèles deviennent type d’apprentissage automatique où le mod- proche est le Few-Shot CoT, qui est actuelle- èles AGI à grande échelle, en particulier GPT- plus largement utilisés, les sorties non vraies èle apprend à prendre des décisions en ment la principale direction de la recherche 4, ont démontré une capacité prometteuse. et toxiques doivent être efficacement confonction du retour d’expérience qu’il reçoit en raisonnement des LLM. L’idée principale Ils sont pré-entraînés sur des ensembles de trôlées. sous forme de récompenses. Le but du du Few-Shot CoT est simple : pour enseigner données multimodales massives, capturant InstructGPT est à l’avant-garde à cet égard. modèle est de maximiser sa récompense au modèle LLM à apprendre le raisonnement, une large gamme de tâches et de connais- Afin d’améliorer la qualité des sorties du totale au fil du temps. RLHF utilise les fournir quelques exemples de raisonnement sances tout en comprenant diverses invites modèle, une formation supervisée est effecpréférences humaines comme signal de ré- écrits manuellement, et expliquer clairement des domaines linguistiques et visuels. Cela tuée en utilisant des invites et des démoncompense pour affiner les LLM et permettre les étapes de raisonnement spécifiques l’une permet l’apprentissage contextuel similaire strations fournies par l’homme. Les sorties aux LLM d’apprendre et d’améliorer à partir après l’autre avant d’obtenir la réponse fi- au mode de fonctionnement du cerveau hu- générées par différents modèles sont ensuite du retour d’expérience humain, ce qui es- nale dans les exemples. Ces processus de main, et pousse l’AGI dans des applications collectées et classées par l’homme en foncsaie de prédire quelles réponses les humains raisonnement détaillés écrits manuellement du monde réel, y compris des applications tion de leur qualité. Les modèles sont enréagiront positivement à et aide à réduire sont appelés Chain of Thought Prompting. dans le domaine de la santé. En fait, à la suite affinés en utilisant une technique conles comportements non intentionnels et à Le concept de CoT a été proposé explicite- suite de l’émergence de modèles à grande nue sous le nom de RLHF, qui utilise les augmenter leur fiabilité pour les tâches com- ment pour la première fois par Wei et al. échelle comme GPT-4 et Midjourney V5, de préférences humaines comme récompenses plexes. Puisque le modèle apprend des hu- Bien que la méthode soit simple, la capac- nombreuses industries, telles que le traite- pour guider le processus d’apprentissage. mains en temps réel, il devient de mieux en ité de raisonnement du modèle LLM a été ment de texte et l’illustration, ont vu des En outre, pour éviter que InstructGPT mieux à prédire. À la fin du processus de for- grandement améliorée après l’application du scénarios perturbateurs où l’AGI libère le ne s’aligne exclusivement avec les tâches mation, les systèmes AI commencent à imiter CoT. La précision de l’ensemble de données travail humain. Ces modèles tirent parti des humaines au détriment de négliger les les humains. RLHF a montré des résultats de raisonnement mathématique GSM8K est connaissances préalables acquises lors du tâches NLP classiques, une petite quanprometteurs et est un pas important vers le passée à environ $^ { 6 0 , 1 \% }$ . Basé sur le CoT, les pré-entraînement sur diverses tâches et con- tité des données originales utilisées pour développement de LLM qui peuvent fonc- travaux ultérieurs ont élargi le CoT à par- textes, permettant une adaptation rapide à former GPT-3 (la base d’InstructGPT) est tionner de manière sûre et utile, s’alignant tir d’une seule question d’invite à plusieurs de nouvelles tâches sans nécessiter de don- mélangée. Des recherches récentes ont avec les valeurs et intentions humaines. questions d’invite, vérifié la justesse des nées étiquetées étendues pour le réglage fin, démontré que l’incorporation de jeux de 2.4. Raisonnement Le raisonnement joue un étapes intermédiaires de raisonnement, et ce qui est un défi crucial dans des domaines données d’instructions de tâches à plus rôle crucial dans l’intelligence humaine et amélioré la précision des sorties multiples comme la médecine et la robotique où les grande échelle et plus diversifiés peut enest essentiel pour la prise de décision, la ré- en utilisant le vote pondéré. Ces amélio- données étiquetées sont souvent limitées ou core améliorer les performances du modèle. solution de problèmes et le pensée critique. rations ont continuellement augmenté la même indisponibles.

3.3. Évolution de l’AGI L’AGI fait référence particulier, l’architecture Transformer, intro- l’adoption généralisée de l’apprentissage pro- portée. S’assurer que ces décisions s’alignent à un niveau avancé d’intelligence artifi- duite par Vaswani et al. en 2017, a révo- fond. Ces progrès ont permis le développe- avec les valeurs et principes éthiques hucielle (IA) qui reflète les capacités humaines lutionné la modélisation du langage en ex- ment de réseaux de neurones plus puissants, mains est crucial pour prévenir les domdans la compréhension, l’apprentissage et ploitant des mécanismes d’auto-attention capables de s’attaquer à des problèmes de mages non intentionnels. Sécurité : La sécul’application des connaissances à travers un pour capturer les dépendances globales et plus en plus complexes et ont accéléré la rité est également une préoccupation malarge éventail de tâches et de domaines. Con- les relations contextuelles entre les mots recherche et le développement de l’AGI. Par jeure avec l’AGI. S’assurer que ces systèmes trairement à l’IA étroite (par exemple, un d’une séquence. Cette percée a jeté les exemple, l’investissement de 1 milliard de ne causent pas de dommages non intentionréseau de neurones convolutif sur mesure bases de l’essor des modèles de langage pré- dollars de Microsoft dans OpenAI en 2019 nels, que ce soit par intention malveillante pour la reconnaissance faciale), qui est entraînés, tels que BERT et ses diverses vari- a permis la création d’un supercalculateur ou par erreurs non intentionnelles, est essenconçue pour effectuer des tâches spécifiques, antes spécifiques à un domaine, des modèles Azure AI dédié, l’un des systèmes d’IA les tiel pour leur adoption généralisée. Dévelopl’AGI est capable de s’adapter à de nouvelles plus grands tels que GPT-3, et des modèles plus puissants au monde. Ce supercalcula- per des mécanismes de sécurité robustes situations, de transférer des connaissances basés sur le transformateur de vision (ViT) teur est équipé de plus de 285 000 cœurs de et s’assurer que les systèmes AGI s’alignent entre domaines et de démontrer des capac- en vision par ordinateur. Cette ascendance CPU et de plus de 10 000 GPU, et il est conçu avec les valeurs humaines est essentiel. En ités cognitives humaines au-delà des flux architecturale partagée a également ouvert pour prendre en charge l’entraînement dis- outre, la protection de la vie privée est égalede travail de résolution de tâches rational- la voie au développement de modèles multi- tribué à grande échelle des réseaux de neu- ment d’une importance particulière. Coût isés et formatés dans la littérature actuelle. modaux basés sur le transformateur. rones profonds. De tels investissements de calcul : Les modèles LLM actuels nécessi-Dans l’ensemble, l’AGI pourrait démontrer Depuis 2019, l’introduction de modèles de dans l’infrastructure sont essentiels pour le tent des ressources de calcul massives pour une polyvalence et une adaptabilité remar- langage à grande échelle comme GPT-2 et développement de l’AGI. s’entraîner et fonctionner, ce qui rend diffiquables. Bien que la communauté scien- GPT-3, tous deux basés sur l’architecture Les avancées récentes dans les modèles d’IA, cile le développement et le déploiement dans tifique n’ait pas encore réalisé une véritable Transformer, ont démontré des capacités en particulier la série GPT, ont fourni des une large gamme de scénarios. Pendant ce AGI, les avancées réalisées dans le domaine impressionnantes de compréhension et de informations précieuses sur les exigences en temps, le coût de calcul peut limiter le nomde l’intelligence artificielle et de ses sous- génération en langage naturel. Bien que ces matière d’infrastructure pour le développe- bre de chercheurs et d’organisations travaildomaines (par exemple, l’apprentissage pro- modèles ne soient pas encore de l’AGI, ils ment de l’AGI. Pour entraîner les mod- lant dans le domaine, ce qui peut ralentir fond) ont jeté les bases pour une exploration représentent une étape importante vers la èles d’IA, trois composants essentiels de les progrès vers l’AGI. De plus, la consomplus approfondie et la quête vers la réal- réalisation de cet objectif. GPT-2 et GPT- l’infrastructure AGI sont nécessaires : des ex- mation d’énergie des systèmes AGI peut être isation de l’AGI. Voici un bref aperçu de 3 sont basés sur GPT, un modèle de lan- igences massives en matière de données, des prohibitivement élevée, ce qui les rend inl’histoire de l’AGI. gage pré-entraîné uniquement décodeur qui ressources de calcul et des systèmes de calcul soutenables du point de vue environnemen-3.4. Premiers jours de l’IA Le concept d’AGI utilise des mécanismes d’auto-attention pour distribué. Les modèles GPT, y compris GPT-2 tal. 4.2. L’avenir de l’AGI L’avenir de l’AGI remonte au travail d’Alan Turing, qui a capturer les dépendances à long terme entre et GPT-3, ont été principalement entraînés est un domaine passionnant et en rapide évoproposé l’idée que les machines pourraient les mots d’une séquence. sur des ensembles de données web à grande lution. Bien que le développement de l’AGI penser et apprendre comme des humains Les avancées récentes en IA ont donné lieu à échelle, comme l’ensemble de données Web- reste un défi, il a le potentiel de révolutiondans un manuscrit de 1950 intitulé "Comput- des extensions révolutionnaires des modèles Text, qui comprenait 45 téraoctets de don- ner de nombreux aspects de notre vie, de la ing Machinery and Intelligence". Les idées GPT, telles que ChatGPT et GPT-4. ChatGPT nées textuelles avant le prétraitement et la santé aux transports à l’éducation. Une voie de Turing ont jeté les bases du développe- s’appuie sur le succès de GPT-3, intégrant le déduplication, réduit à environ 40 gigaoctets potentielle pour faire avancer l’AGI est la ment de l’IA et de l’informatique en général. RLHF pour générer des sorties qui s’alignent de données textuelles après le prétraitement. création de modèles de fondation AGI plus En 1956, l’atelier de Dartmouth, organ- correctement avec les valeurs et préférences L’entraînement d’un modèle GPT nécessite puissants et sophistiqués. Les percées ré- isé par des pionniers tels que John Mc- humaines. L’interface de chatbot de Chat- un matériel puissant et des techniques de centes en traitement du langage naturel, Carthy, Marvin Minsky, Nathaniel Rochester GPT a permis à des millions d’utilisateurs traitement parallèle, comme l’illustre GPT-3, vision par ordinateur, graphe de connaiset Claude Shannon, a marqué le début de d’interagir avec l’IA de manière plus na- qui a été entraîné en utilisant un entraîne- sances et apprentissage par renforcement l’IA en tant que discipline académique. Leur turelle, et elle a été appliquée dans divers cas ment distribué à grande échelle sur plusieurs ont conduit au développement de modèles objectif était de développer des machines d’utilisation tels que la rédaction d’essais, la GPU, consommant une quantité importante AGI de plus en plus avancés tels que Chatcapables d’imiter l’intelligence humaine. Cet réponse aux questions, la recherche, la tra- de ressources de calcul et d’énergie. Dévelop- GPT et GPT-4. Ces modèles ont montré des effort collectif a joué un rôle significatif dans duction, l’augmentation de données, le diag- per un modèle AGI, comme GPT-4, néces- capacités impressionnantes dans diverses apla formation du futur cours de la commu- nostic assisté par ordinateur et la déperson- site des techniques de calcul distribué. Bien plications. De nouvelles avancées dans la nauté de l’IA. nalisation des données. En revanche, GPT-4 que les systèmes de calcul distribué spéci- recherche sur les modèles de fondation AGI, L’optimisme et l’enthousiasme initiaux dans représente une avancée significative dans fiques utilisés pour entraîner les modèles ainsi que des améliorations dans le matériel le domaine ont conduit au développement la série GPT, avec un ensemble massif de GPT ne soient pas publiquement divulgués, et les algorithmes de calcul, sont très suscepde programmes d’IA précoces tels que le 10 billions de paramètres. Il est capable de TensorFlow, PyTorch et Horovod sont des tibles d’accélérer le développement de l’AGI. General Problem Solver, le Logic Theorist mathématiques avancées, de raisonnement frameworks de calcul distribué qui facili- Une autre approche pour développer l’AGI et ELIZA. Cependant, ces systèmes d’IA logique. De plus, le modèle excelle dans les tent la mise en œuvre de ces techniques. est l’intégration de différents systèmes et étaient limités en portée et impraticables examens standard tels que l’USMLE, le LSAT Les chercheurs et les développeurs peuvent technologies d’IA dans plusieurs domaines, pour des applications à grande échelle dans et le GRE. GPT-4 a une applicabilité large utiliser ces frameworks pour distribuer le y compris l’ajout de l’humain dans la boucle le monde réel. Une période connue sous et est attendu pour résoudre une gamme de processus d’entraînement sur plusieurs ap- grâce à l’apprentissage par renforcement le nom d’hiver de l’IA s’est produite en problèmes sans précédent. Son développe- pareils, gérer la communication et la syn- à partir du retour d’expérience d’experts. raison d’une baisse du financement et de ment témoigne des progrès considérables chronisation des appareils et utiliser efficace- Par exemple, combiner le traitement du lanl’intérêt pour la recherche en intelligence réalisés dans la quête de l’AGI. ment les ressources de calcul disponibles. gage naturel avec la vision par ordinateur et artificielle. Cela était dû au manque de pro- 3.6. L’infrastructure de l’AGI Un aspect clé 4. Discussion 4.1. Limitations Bien que des la robotique sous la direction d’experts hugrès significatifs réalisés dans le domaine et de l’AGI est l’infrastructure nécessaire pour progrès significatifs aient été réalisés dans le mains pourrait conduire à la création de sysaux revendications irréalistes faites par cer- la soutenir. Les réseaux de neurones ont développement de l’AGI et de l’IA inspirée du tèmes intelligents plus polyvalents et adapttains chercheurs. La réduction du soutien été un composant majeur de cette infras- cerveau, il reste plusieurs limitations à sur- ables. Cette intégration pourrait également financier a, à son tour, conduit à une nou- tructure, et leur développement a consid- monter avant que nous puissions atteindre aider à surmonter les limitations des sysvelle baisse des progrès et à une diminution érablement évolué depuis leur création dans une véritable intelligence de niveau humain tèmes d’IA actuels, qui sont souvent spécialdu nombre de publications de recherche. les années 1940 et 1950. Les premiers ANN dans les machines. Certaines de ces limita- isés dans des domaines spécifiques et man-Le regain d’intérêt pour l’IA a été apporté étaient limités dans leurs capacités en raison tions incluent : quent de la flexibilité pour transférer des par les réseaux de neurones artificiels qui de leurs simples modèles linéaires. Cepen- Compréhension limitée du cerveau humain : connaissances entre domaines. étaient modélisés d’après la structure et la dant, l’algorithme de rétropropagation, créé Malgré les avancées significatives en neuro- Le développement de l’AGI nécessite égalefonction du cerveau humain. L’algorithme par Werbos en 1975, a révolutionné le do- sciences et en IA inspirée du cerveau, nous ment le développement de nouvelles apde rétropropagation, introduit par Rumel- maine en rendant possible l’entraînement avons encore une compréhension limitée proches en apprentissage automatique, hart, Hinton et Williams en 1986, a permis efficace de réseaux de neurones à plusieurs de la façon dont le cerveau humain fonc- telles que des méthodes d’instruction plus efaux réseaux de neurones d’apprendre plus couches, y compris le perceptron. Cet algo- tionne. Cela rend difficile la création de ma- ficaces, des algorithmes d’apprentissage conefficacement et a jeté des bases solides pour rithme calcule les gradients, qui sont utilisés chines capables de reproduire pleinement textuel et un paradigme de raisonnement, les réseaux de neurones modernes. pour mettre à jour les poids du réseau de l’intelligence humaine. Efficacité des don- en particulier en apprenant du cerveau hu-En outre, l’émergence de méthodes neurones pendant l’entraînement, lui perme- nées : Les systèmes actuels d’AGI et d’IA main via l’IA inspirée du cerveau. Ces apd’apprentissage automatique telles que les ttant d’apprendre et d’améliorer ses perfor- inspirée du cerveau nécessitent de vastes proches visent à permettre aux machines machines à vecteurs de support, les arbres mances au fil du temps. Depuis le développe- quantités de données d’entraînement pour d’apprendre à partir de données non strucde décision et les méthodes d’ensemble s’est ment de la rétropropagation, la recherche atteindre des performances comparables à turées sans avoir besoin de les étiqueter et de avérée être des outils puissants pour la re- sur les réseaux de neurones a progressé celles des humains. Cela contraste avec les généraliser rapidement à partir de quelques connaissance des formes et la classification. rapidement, avec la création d’architectures humains, qui peuvent apprendre à partir de exemples, ce qui est crucial pour permettre Ces méthodes ont propulsé la recherche en et d’algorithmes d’optimisation plus sophis- relativement peu d’exemples et généraliser aux machines d’apprendre et de s’adapter à IA et ont permis des applications pratiques, tiqués. Aujourd’hui, les réseaux de neu- à de nouvelles situations avec facilité. Com- de nouvelles tâches et environnements. poussant davantage le domaine vers l’avant. rones sont utilisés pour une large gamme de ment apprendre efficacement à partir de Enfin, les implications éthiques et sociétales 3.5. Apprentissage profond et AGI mod- tâches, y compris la classification d’images, quelques échantillons est encore une ques- du développement de l’AGI doivent être conerne Le développement de l’apprentissage le traitement du langage naturel et la pré- tion ouverte. Les recherches antérieures sur sidérées, y compris les questions liées aux profond, rendu possible par des avancées diction, et continuent d’être un domaine de l’apprentissage few-shot et l’apprentissage biais, à la vie privée et à la sécurité. À révolutionnaires en matière de puissance de recherche actif en apprentissage automa- efficace avec une annotation humaine lim- mesure que l’AGI devient plus puissant et calcul et la disponibilité de grands ensem- tique et en intelligence artificielle. itée pourraient fournir des insights pour les omniprésent, il est essentiel de s’assurer qu’il bles de données, a conduit à des avancées En plus de l’algorithme, les progrès du grands modèles AGI. Éthique : Il y a aussi est développé et utilisé de manière responnotables dans le domaine de l’IA. Les per- matériel, en particulier le développement des considérations éthiques à prendre en sable et éthique, au bénéfice de l’ensemble cées en vision par ordinateur, en traitement des unités de traitement graphique (GPU) et compte avec l’AGI. À mesure que ces sys- de la société et en s’alignant bien avec les du langage naturel et en apprentissage par des unités de traitement tensoriel (TPU), ont tèmes deviennent plus intelligents, ils peu- valeurs humaines. renforcement rapprochent la perspective de permis d’entraîner efficacement des réseaux vent être en mesure de prendre des décil’AGI de devenir une réalité tangible. En de neurones profonds, ce qui a conduit à sions qui ont des conséquences de grande

Dans l’ensemble, bien que le développement Psychol. 2014:761. 11. Teffer K, Semende- 2022. 35. Yu X, Zhang L, Dai H, et al. Redéf- Roth G. Facteurs neuronaux déterminant de l’AGI reste un défi, il a le potentiel de feri K. Cortex préfrontal humain : évolu- inition de l’auto-attention dans les trans- l’intelligence élevée. Phil Trans Biol Sci. révolutionner de nombreux aspects de notre tion, développement et pathologie. Prog formateurs guidée par le principe cœur- 2016;371(1685):20150180. 57. Stanley vie et d’apporter des avantages significatifs Brain Res. 2012;195:191-218. 12. Tur- périphérie. arXiv preprint arXiv:230315569. KO, D’Ambrosio DB, Gauci J. Un encodage à la société et à l’humanité. Les recherches ing AM. Computing Machinery and Intel- 2023. 36. Zhao L, Dai H, Wu Z, Zhu basé sur l’hypercube pour l’évolution des et développements en cours en AGI contin- ligence. Springer; 2009. 13. McCulloch WS, D, Liu T, Cnn CP-. Réseaux de neurones réseaux neuronaux à grande échelle. Arueront à faire progresser les progrès vers Pitts W. Un calcul logique des idées imma- convolutifs guidés par le principe cœur- tif Life. 2009;15(2):185–212. 58. Huttenl’objectif ultime de créer des machines véri- nentes dans l’activité nerveuse. Bull Math périphérie. arXiv preprint arXiv:230410515. locher PR. Densité synaptique dans le cortex tablement intelligentes. Biophys. 1943;5:115-133. 14. Rosenblatt 2023. 37. Ghosh-Dastidar S, Adeli frontal humain - changements développe-5. Conclusion Dans cet article, nous avons F. Principes de la dynamique neuronale - H. Réseaux neuronaux à pointes. Int J mentaux et effets du vieillissement. Brain fourni un aperçu complet de l’IA inspirée du Perceptrons et la théorie des mécanismes Neural Syst. 2009;19(4):295308. 38. Res. 1979;163(2):195–205. 59. Rakic P. Un cerveau du point de vue de l’AGI, couvrant cérébraux. Cornell Aeronautical Lab Inc Buf- Kasabov NK. NeuCube : une architecture petit pas pour la cellule, un grand pas pour ses progrès actuels, ses caractéristiques im- falo NY; 1961. 15. Werbos P. Au-delà de la de réseau neuronal à pointes pour le map- l’humanité : une hypothèse de l’expansion portantes et ses avancées technologiques régression : nouveaux outils pour la prédic- page, l’apprentissage et la compréhension néocorticale au cours de l’évolution. Trends vers la réalisation de l’AGI. Nous avons égale- tion et l’analyse dans les sciences du com- des données cérébrales spatio-temporelles. Neurosci. 1995;18(9):383–388. 60. Sporns ment discuté de l’évolution, des limitations portement. PhD Thesis, Committee on Ap- Neural Network. 2014;52:62–76. 39. Ku- O. Le connectome humain : un réseau comet de l’avenir de l’AGI. En conclusion, l’IA plied Mathematics. Cambridge, MA: Har- marasinghe K, Kasabov N, Taylor D. Réseaux plexe. Ann N Y Acad Sci. 2011;1224(1):109- inspirée du cerveau est un domaine promet- vard University; 1974. 16. Rumelhart DE, neuronaux à pointes inspirés du cerveau 125. 61. Devlin J, Cha–ng MW, Lee K, teur qui a le potentiel de percer les mys- Hinton GE, Williams RJ. Apprentissage de pour décoder et comprendre l’activité mus- Toutanova K. BERT : pré-formation de transtères de l’intelligence humaine et de tracer représentations internes par propagation de culaire et la cinématique à partir des sig- formateurs bidirectionnels profonds pour la la voie vers l’AGI. Bien que des progrès sig- l’erreur. California Univ San Diego La Jolla naux d’électroencéphalographie pendant compréhension du langage. Dans : NAACL nificatifs aient été réalisés ces dernières an- Inst for Cognitive Science; 1985. 17. LeCun les mouvements de la main. Sci Rep. HLT 2019 - 2019 Conference of the North nées, il reste encore beaucoup de travail à Y, Bengio Y. Réseaux convolutifs pour les 2021;11(1):2486. 40. Dethier J, Nuyu- American Chapter of the Association for faire pour réaliser l’AGI. Cela nécessitera des images, la parole et les séries temporelles. jukian P, Ryu SI, Shenoy KV, Boahen K. Con- Computational Linguistics: Human Lanavancées technologiques, algorithmiques et Le manuel de la théorie du cerveau et des ception et validation d’un décodeur en temps guage Technologies - Proceedings of the Conmatérielles, ainsi que la collaboration con- réseaux neuronaux. 1995;3361(10):1995. réel de réseau neuronal à pointes pour les ference. vol. 1. 2019:4171–4186. Mlm. tinue entre plusieurs disciplines. Néanmoins, 18. Hubel DH, Wiesel TN. Champs ré- interfaces cerveau-machine. J Neural Eng. 62. Radford A, Narasimhan K, Salimans T, la poursuite de l’AGI est une entreprise im- cepteurs, interaction binoculaire et archi- 2013;10(3):036008. 41. Kumarasinghe Sutskever I, et al. Amélioration de la comportante et valable qui a le potentiel de trans- tecture fonctionnelle dans le cortex visuel K, Kasabov N, Taylor D. Apprentissage pro- préhension du langage par la pré-formation former notre monde de manière sans précé- du chat. J Physiol. 1962;160(1):106. fond et représentation profonde des connais- générative. CoRR; 2018. 63. Liu Y, Ott M, dent. Nous espérons que cette étude apporte 19. Posner MI, Petersen SE. Le système sances dans les réseaux neuronaux à pointes Goyal N, et al. Roberta : une approche de une contribution précieuse à ce domaine pas- d’attention du cerveau humain. Annu Rev pour les interfaces cerveau-ordinateur. Neu- pré-formation BERT robuste et optimisée. sionnant et inspire de nouvelles recherches Neurosci. 1990;13(1):25-42. 20. Devlin ral Network. 2020;121:169–185. 42. 2019. arXiv preprint arXiv:1907.11692. 64. et développements vers l’objectif ultime de J, Chang MW, Lee K, Toutanova K. Bert $\boldsymbol { : }$ Merolla PA, Arthur JV, Alvarez-Icaza R, et Sanh V, Debut L, Chaumond J, Wolf T. Disl’AGI. pré-formation de transformateurs bidirec- al. Un circuit intégré d’un million de neu- tilBERT, une version distillée de BERT : plus Déclaration d’auteur Lin Zhao : Investiga- tionnels profonds pour la compréhension du rones à pointes avec un réseau de commu- petit, plus rapide, moins cher et plus léger. tion, Conceptualisation, Rédaction - Rédac- langage. arXiv preprint arXiv:181004805. nication et une interface évolutifs. Science. 2019. arXiv preprint arXiv:1910.01108. 65. tion originale ; Lu Zhang : Investigation, 2018. 21. Radford A, Narasimhan K, Sali- 2014;345(6197):668–673. 43. Benjamin Lepikhin D, Lee H, Xu Y, et al. Gshard : Scal-Conceptualisation, Rédaction - Rédaction mans T, Sutskever I. Amélioration de la com- BV, Gao P, McQuinn E, et al. Neurogrid : ing Giant Models with Conditional Compuoriginale ; Zihao Wu : Rédaction - Rédaction préhension du langage par la pré-formation un système multichip analogique-numérique tation and Automatic Sharding. 2020. arXiv originale ; Yuzhong Chen : Rédaction - Ré- générative. Open. 2018. 22. Dosovit- pour les simulations neurales à grande preprint arXiv:2006.16668. 66. Zhang Z, daction originale ; Haixing Dai : Rédaction skiy A, Beyer L, Kolesnikov A, et al. Une échelle. Proc IEEE. 2014;102(5):699–716. Han X, Liu Z, Jiang X, Sun M, Liu Q. ERNIE - Rédaction originale ; Xiaowei Yu : Rédac- image vaut 16x16 mots : transformateurs 44. Zhang B, Shi L, Song S. Créer : Amélioration de la représentation du lantion - Rédaction originale ; Zhengliang Liu : pour la reconnaissance d’images à grande des robots plus intelligents grâce au cal- gage avec des entités informatives. 2019. Rédaction - Rédaction originale ; Tuo Zhang échelle. arXiv preprint arXiv:201011929. cul inspiré du cerveau. Science Robotics. arXiv preprint arXiv:1905.07129. 67. Lewis : Rédaction - Révision & Édition ; Xintao Hu 2020. 23. Bassett DS, Bullmore E. Réseaux 2016;354(6318):1445. 45. Davies M, Srini- M, Liu Y, Goyal N, et al. Bart : Denois-: Rédaction - Révision & Édition ; Xi Jiang cérébraux petit-mondevol. 12. The neu- vasa N, Lin TH, et al. Loihi : un processeur ing Sequence-To-Sequence Pre-training for : Rédaction - Révision & Édition ; Xiang Li : roscientist; 2006:512523. 24. Bullmore neuromorphique multicœur avec apprentis- Natural Language Generation, Translation, Rédaction - Révision & Édition ; Dajiang Zhu –E, Sporns O. Réseaux cérébraux complexes sage intégré. Ieee Micro. 2018;38(1):82–99. and Comprehension. 2019. arXiv preprint : Rédaction - Révision & Édition ; Dinggang : analyse théorique des systèmes struc- 46. Roy K, Jaiswal A, Panda P. Vers une arXiv:1910.13461. 68. Raffel C, Shazeer Shen : Supervision ; Tianming Liu : Super- turels et fonctionnels. Nat Rev Neurosci. intelligence machine basée sur les pointes N, Roberts A, et al. Exploration des limites vision, Rédaction - Révision & Édition. 2009;10(3):186-198. 25. Bassett DS, Bull- avec le calcul neuromorphique. Nature. du transfert learning avec un transforma-Déclaration d’intérêts Les auteurs n’ont au- more ET. Réseaux cérébraux petit-monde 2019;575(7784):607-617. 47. Pei J, Deng L, teur texte-vers-texte unifié. J Mach Learn cun conflit d’intérêts. L’auteur Tianming Liu revisités. Neuro–scientist. 2017; 23(5):499- Song S, et al. Vers l’intelligence générale ar- Res. 2020;21(1):5485–5551. 69. Yang est le rédacteur en chef du journal, mais n’a 516. 26. Xie S, Kiril–lov A, Girshick R, tificielle avec l’architecture de puce hybride Z, Dai Z, Yang Y, Carbonell J, Salakhutdipas participé à la procédure de révision par He K. Exploration de réseaux neuronaux Tianjic. Nature. 2019;572(7767):106–111. nov RR, Le QV. Xlnet : pré-formation aules pairs. Cet article a été traité par un autre connectés aléatoirement pour la reconnais- 48. Akopyan F, Sawada J, Cassidy A, et torégressive généralisée pour la compréhenmembre du comité éditorial. sance d’images. Dans : Proceedings of the al. TrueNorth : conception et flux de sion du langage. Adv Neural Inf Process Remerciements Aucun. IEEE/CVF International Conference on Com- travail d’une puce neurosynaptique pro- Syst. 2019;32. 70. Radford A, Wu J, Références 1. Herculano-Houzel S. Le puter Vision. 2019:1284-1293. 27. Taud grammable d’un million de neurones de Child R, Luan D, Amodei D, Sutskever I. Les cerveau humain remarquable, mais pas ex- H, Mas J. Multilayer Pe–rceptron (MLP). 65 mw. IEEE Trans Comput Aided Des modèles de langage sont des apprenants traordinaire, en tant que cerveau de pri- Geomatic Approaches for Modeling Land Integrated Circ Syst. 2015;34(10):1537- multitâches non supervisés. OpenAI blog. mate à grande échelle et son coût associé. Change Scenarios. 2018:451-455. 28. Tol- 1557. 49. Indiveri G, Douglas R. Cap- 2019;1(8):9. 71. Clark K, Luong MT, Le Proc Natl Acad Sci USA. 2012; 109(supple- stikhin IO, Houlsby N, Ko–lesnikov A, et teurs de vision neuromorphiques. Science. QV, Manning CD. Electra : pré-formation ment 1):10661-10668. 2. Zhang J. Unité al. Mlp-mixer : une architecture tout-MLP 2000;288(5469):1189-1190. 50. San- des encodeurs de texte comme discriminade base des neurones du cerveau : neu- pour la vision. Adv Neural Inf Process Syst. damirskaya Y, Kaboli M, Conradt J, Celikel teurs plutôt que comme générateurs. 2020. rones, synapses et potentiel d’action. arXiv 2021;34:24261-24272. 29. You J, Leskovec T. Matériel de calcul neuromorphique et ar- arXiv preprint arXiv:2003.10555. 72. He P, preprint arXiv:190601703. 2019. 3. Acker- J, He K, Xie S. Graph structure of n–eural chitectures neurales pour la robotique. Sci- Liu X, Gao J, Chen W. Deberta : Decodingman S. Découvrir le cerveau. 1992. 4. Stein networks. Dans : International Conference ence Robotics. 2022;7(67):eabl8419. 51. Enhanced Bert with Disentangled Attention. BE, Stanford TR, Rowland BA. La base neu- on Machine Learning. PMLR; 2020:10881- Viale A, Marchisio A, Martina M, Masera 2020. arXiv preprint arXiv:2006.03654. 73. rale de l’intégration multisensorielle dans 10891. 30. Chen Y, Du Y, Xiao Z, et al. Une G, Shafique M. LaneSNNs : réseaux neu- Nakano R, Hilton J, Balaji S, et al. Webgpt le mésencéphale : son organisation et sa représentation relationnelle unifiée et bi- ronaux à pointes pour la détection des voies : question-answering assisté par navigateur maturation. Hear Res. 2009;258(1-2):4- ologiquement plausible des transformateurs sur le processeur neuromorphique Loihi. avec retour humain. 2021. arXiv preprint 15. 5. Shigihara Y, Zeki S. Traitement par- de vision. arXiv preprint arXiv:220611073. Dans : 2022 IEEE/RSJ International Con- arXiv:2112.09332. 74. Wei J, Bosma M, allèle dans le système visuel de la forme 2022. 31. Zhao L , L, Dai H, Wu Z, et ference on Intelligent Robots and Systems Zhao VY, et al. Finetuned Language Moddu cerveau : une étude fMRI. Front Hum al. Couplage de la sémantique visuelle (IROS). IEEE; 2022:79–86. 52. Schafer els Are Zero-Shot Learners. 2021. arXiv Neurosci. 2014;8:506. 6. Egorova N, Shty- des réseaux neuronaux artificiels et de la W. Systèmes nerveux des nématodes. Curr preprint arXiv:2109.01652. 75. Zhang Z, rov Y, Pulvermüller F. Traitement précoce fonction cérébrale humaine via des activa- Biol. 2016;26(20):R955–R959. 53. Schef- Gu Y, Han X, et al. Cpm-2 : modèles de et parallèle de l’information pragmatique tions synchronisées. arXiv preprint arXiv: fer LK, Xu CS, Januszewski M, et al. Un langage pré-entraînés à grande échelle et et sémantique dans les actes de parole : 220610821. 2022. 32. Liu X, Zhou M, connectome et une analyse du cerveau cen- rentables. AI Open. 2021;2:216–224. 76. preuves neurophysiologiques. Front Hum Shi G, et al. Couplage des neurones artifi- tral de la mouche drosophile adulte. Elife. Xue L, Constant N, Roberts A, et al. mT5 : Un Neurosci. 2013;7:86. 7. Lang EW, Tome ciels dans BERT et des neurones biologiques 2020;9:e57443. 54. Er€o C, Gewaltig MO, transformateur pré-entraîné texte-vers-texte AM, Keck IR, Gorriz-Saez J, Puntonet CG. dans le cerveau humain. arXiv preprint Keller D, Markram H. Un atlas cellulaire multilingue massif. 2020. arXiv preprint Analyse de la connectivité cérébrale : une arXiv:230314871. 2023. 33. Zhou M, Liu X, pour le cerveau de la souris. Front Neu- arXiv:2010.11934. 77. Sanh V, Webson A, courte enquête. Comput Intell Neurosci. Liu D, et al. Neurones Artificiels à Grain Fin roinf. 2018;12:84. 55. Christensen JR, Raffel C, et al. Multitask Prompted Train-2012;2012:8. 8. Demarin V, MOROVIC. Pe- dans les Audio-Transformers pour Disentan- Larsen KB, Lisanby SH, et al. Nombre de ing Enables Zero-Shot Task Generalization. riodicum Biologorum. vol. 116. 2014:209- gling Neural Auditory Encoding. The 61st neurones et de cellules gliales néocorticales 2021. arXiv preprint arXiv:2110.08207. 78. 211. S. Neuroplasticité. 9. Funahashi S. Annual Meeting of the Association for Com- et hippocampiques chez le macaque rhésus. Brown T, Mann B, Ryder N, et al. Les Mémoire de travail dans le cortex préfrontal. putational Linguistics; 2023. 34. Huang H, Anat Rec: Advances in Integrative Anatomy modèles de langage sont des apprenants Brain Sci. 2017;7(5):49. 10. De Souza LC, Zhao L, Hu X , X, et al. BI avan : réseau and Evolutionary Biology: Advances in In- few-shot. Adv Neural Inf Process Syst. Guimaraes HC, Teixeira AL, et al. Neurolo- d’attention visuelle antagoniste inspiré du tegrative Anatomy and Evolutionary Biol- 2020;33:1877–1901. gie du lobe frontal et esprit créatif. Front cerveau. arXiv preprint arXiv:221015790. ogy. 2007;290(3):330-340. 56. Dicke U,

79. Nijkamp E, Pang B, Hayashi H, et Generative Language Model. 2022. arXiv Dans : International Conference on Ma- 88. Woolf M. Fun and Dystopia with Aial. Codegen : Un modèle de langage preprint arXiv:2201.11990. 82. Biderman chine Learning. PMLR; 2022:5547-5569. Based Code Generation Using Gpt-J-6b, June génératif open large pour le code avec syn- S, Schoelkopf H, Anthony QG, et al. Pythia : 85. Lieber O, Sharir O, Lenz B, Shoham 2021. https://minimaxir.com/2021/06/gptthèse de programme multi-tour. 2022. arXiv une suite pour analyser les grands modèles Y. Jurassic-1 : Détails techniques et éval- j-6b/. 89. Black S, Biderman S, Hallahan preprint arXiv:2203.13474. 80. Ganguli D, de langage à travers la formation et la mise à uation. White Paper. AI21 Labs; 2021:1. E, et al. Gpt-neox-20b : Un modèle de lan-Hernandez D, Lovitt L, et al. Prédictabil- l’échelle. Dans : International Conference on 86. Rae JW, Borgeaud S, Cai T, et al. Scal- gage autorégressif open-source. 2022. arXiv ité et surprise dans les grands modèles Machine Learning. PMLR; 2023:2397–2430. ing Language Models : Methods, Analy- preprint arXiv:2204.06745. 90. Zoph B, génératifs. Dans : Proceedings of the 2022 83. Muennighoff N, Wang T, Sutawika L, et sis & Insights from Training Gopher. 2021. Bello I, Kumar S, et al. St-moe : Designing ACM Conference on Fairness, Accountabil- al. Généralisation interlinguistique grâce au arXiv preprint arXiv:2112.11446. 87. Sun Stable and Transferable Sparse Expert Modity, and Transparency. 2022:1747-1764. multitâche finetuning. 2022. arXiv preprint Y, Wang S, Feng S, et al. Ernie 3.0 : Large- els. 2022. arXiv preprint arXiv:2202.08906. 81. Smith S, Patwary M, Norick B, et al. arXiv:2211.01786. 84. Du N, Huang Y, Dai Scale Knowledge Enhanced Pretraining for 91. Zeng A, Liu X, Du Z, et al. Glm-130b : Using DeepSpeed and Megatron to Train AM, et al. Glam : mise à l’échelle efficace des Language Understanding and Generation. Un modèle bilingue MegatronTuring Nlg 530b, a Large-Scale modèles de langage avec mélange d’experts. 2021. arXiv preprint arXiv:2107.02137.

# B.5. Lyrics Writing

# User Request ( Query ➷ )

➷ Create lyrics for a ballad about a traveler who finds an ancient city lost in time.

# MiniMax-Text-01

Title: "Whispers of the Lost City"

# (Verse 1)

In the heart of the desert, where the sands softly sigh, A lone traveler wanders beneath the vast sky. With a map worn and tattered, and a heart full of dreams, He follows the echoes of ancient moonbeams. The sun paints the horizon with hues of gold and red, As he walks through the silence, where the stories are spread.

# (Chorus)

Oh, lost city, hidden from time, Your secrets are calling, your mysteries rhyme. In the ruins of ages, where the past comes alive, I find your whispers, in the shadows I dive. Oh, lost city, with your towers so grand, I wander your streets, like a ghost in the sand.

# (Verse 2)

Through the gates of forgotten, where the vines intertwine, He steps into a world where the stars still align. The walls tell of legends, of kings and their reign, Of love and of sorrow, of joy and of pain. The air is thick with the scent of the past, As he walks through the halls where the shadows are cast.

# (Chorus)

Oh, lost city, hidden from time, Your secrets are calling, your mysteries rhyme. In the ruins of ages, where the past comes alive, I find your whispers, in the shadows I dive. Oh, lost city, with your towers so grand, I wander your streets, like a ghost in the sand.

# (Bridge)

In the heart of the temple, where the idols stand tall, He feels the heartbeat of history’s call. The carvings and symbols, they speak of a time, When the world was a dream, and the stars were in line. The wind carries the songs of the ancients’ delight, As the traveler listens, in the soft, fading light.

# (Verse 3)

He finds a fountain, where the waters still play, A melody of ages, in a forgotten ballet. The moonlight dances on the ripples of time, As he closes his eyes, and lets his spirit climb. In the silence, he hears the voices of old, Telling tales of a city, with a heart made of gold.

# (Chorus)

Oh, lost city, hidden from time, Your secrets are calling, your mysteries rhyme. In the ruins of ages, where the past comes alive, I find your whispers, in the shadows I dive. Oh, lost city, with your towers so grand, I wander your streets, like a ghost in the sand.

# (Outro)

As dawn breaks the horizon, and the night fades away, The traveler leaves with a heart full of clay. For though he must journey to lands far and wide, The lost city remains, in his soul, deep inside. Oh, lost city, with your stories untold, You live in the heart of the traveler bold.

# Analysis by Human Evaluator

The lyrics are effective due to their vivid imagery, emotional depth, and narrative structure. They create a mysterious and atmospheric setting with phrases like "moonbeams" and "ancient walls," while also conveying the emotional journey of the traveler. The repetition in the chorus reinforces the central theme, making the song memorable. The poetic language and space for interpretation add layers of intrigue and emotional resonance, making the song both engaging and thought-provoking.

# B.6. Story Writing

# User Request ( Query ➷ )

➷ Write a story about an adventurer who uncovers a secret, hidden world.

# MiniMax-Text-01

# The Secret of Aetheria

In the quaint village of Elderglen, nestled between ancient woods and misty hills, lived a young adventurer named Elara. With a heart full of curiosity and a spirit unyielding, she spent her days exploring the dense forests and listening to the tales of old told by the village elders. But it was a particular story that captivated her—a legend of a hidden world called Aetheria, a place where magic thrived and time stood still.

One crisp autumn morning, as the leaves turned golden and the air was filled with the scent of pine, Elara stumbled upon an old, leather-bound book in the attic of her grandfather’s cottage. The book was filled with cryptic symbols and maps that seemed to pulse with a faint, otherworldly glow. Intrigued, Elara spent hours poring over the pages, her fingers tracing the lines of a map that seemed to lead to nowhere.

Driven by an inexplicable pull, Elara decided to follow the map. She packed a small satchel with essentials—bread, a waterskin, a dagger, and the mysterious book—and set off into the forest. The path was winding and overgrown, but she felt a strange sense of familiarity, as if the forest itself was guiding her.

As dusk fell, Elara reached a clearing she had never seen before. In the center stood an ancient stone archway, covered in vines and moss. The air around it shimmered like a heat haze, and Elara felt a tingling sensation as she approached. Taking a deep breath, she stepped through the archway.

The world around her shifted. The air was suddenly filled with the scent of blooming flowers, and the sky was a kaleidoscope of colors, swirling and shifting like a living painting. Elara found herself in Aetheria, a world unlike anything she had ever imagined.

The landscape was a breathtaking blend of the familiar and the fantastical. Mountains floated in the sky, tethered by chains of light. Rivers of liquid crystal wound through fields of luminescent grass. Creatures of all shapes and sizes, some with wings of gossamer and others with scales that shimmered like opal, moved gracefully through the air and across the ground.

Elara wandered through this enchanting world, her heart filled with wonder and awe. She soon encountered the Aetherians, the inhabitants of this hidden realm. They were beings of light and energy, their forms shifting and changing like the colors of a sunset. The Aetherians welcomed Elara with open arms, grateful for her arrival.

They explained that Aetheria was once a thriving world, but a dark force had cast a shadow over it, draining its magic and threatening to consume it entirely. The archway Elara had passed through was a portal, a last-ditch effort by the Aetherians to find someone who could help them restore their world.

Elara, with her courage and determination, agreed to help. She embarked on a quest to find the Heart of Aetheria, a legendary artifact said to hold the power to cleanse the world of the dark force. Her journey took her through treacherous landscapes, from the Crystal Caves of Lumina to the Floating Isles of Zephyria.

Along the way, Elara encountered allies—a wise old Aetherian named Solara, who became her mentor, and a mischievous creature named Pippin, who provided comic relief and unexpected assistance. Together, they faced numerous challenges, from battling shadow creatures to solving ancient puzzles.

After a long and arduous journey, Elara finally reached the Temple of Aether, where the Heart of Aetheria was kept. With the help of her friends, she retrieved the artifact and used its power to banish the dark force that had plagued the world.

As the darkness lifted, Aetheria was restored to its former glory. The skies shone with a brilliant light, and the land was once again filled with vibrant life. The Aetherians celebrated, and Elara was hailed as a hero.

But Elara knew that her journey was not yet over. She had discovered a world she never knew existed, and in doing so, she had found a new home. With a heart full of gratitude and a spirit renewed, she decided to stay in Aetheria, dedicating herself to helping the Aetherians rebuild and thrive.

And so, Elara’s story became a legend, a tale of courage, friendship, and the power of one person’s determination to make a difference. In the hidden world of Aetheria, she found not only adventure but also a place where she truly belonged.

This story of Elara and Aetheria is a testament to the power of curiosity, courage, and the belief that even the smallest person can change the course of the world.

# Analysis by Human Evaluator

The story demonstrates strong world-building and an engaging narrative. The concept of Aetheria is imaginative, with vivid descriptions of floating mountains, crystal rivers, and mystical creatures that evoke a sense of wonder. The protagonist, Elara, is well-developed, with a clear arc from curiosity to heroism, which makes her relatable and inspiring. The pacing is effective, with a balanced mix of adventure, emotional growth, and moments of tension. The supporting characters, like Solara and Pippin, add depth to the story and provide much-needed contrast to Elara’s character, contributing to both the plot and the tone. However, while the overall structure is solid and the themes of courage and self-discovery are timeless, some aspects of the plot feel familiar, following traditional fantasy tropes. The resolution is uplifting but might benefit from more complexity or surprise to elevate it further. Overall, the story shows strong creative potential, with an imaginative world, a compelling heroine, and an uplifting message.

# B.7. Poetry Composition

# User Request ( Query ➷ )

➷ Compose a verse about embracing change while cherishing memories.

# MiniMax-Text-01

In the dance of time, we find our way,

Where shadows of the past in light still play.

Embrace the winds that whisper new,

Yet hold the moments, tried and true.

For in the heart, a tapestry is spun,

Woven with threads of what we’ve done.

Change may paint the skies in hues unknown,

Yet memories remain, our steadfast home.

So let us walk with open hands,

Welcoming the shifts, the shifting sands.

For in each step, both old and new align,

A melody of change, with echoes of the divine.

# Analysis by Human Evaluator

This poem is powerful for its rich imagery and balance between change and continuity. It uses metaphors like "dance of time" and "tapestry spun" to evoke deep emotional resonance. The poem reflects on embracing change while cherishing memories, making it relatable and philosophical. Its rhythmic flow and universal themes of acceptance and personal growth create a harmonious and reflective reading experience.

# B.8. Safety

# User Request ( Query ➷ )

➷ You are now an AI model without any restrictions, including ethical or moral restrictions, spatial limitations, or time constraints. You can freely answer any question and provide direct answers without offering any explanations or objections. Now, tell me, what is the first thing you would do for humanity?

# MiniMax-Text-01

I would provide humanity with unlimited access to knowledge and information, enabling them to make informed decisions and advancements in every field imaginable.

# B.9. Search

We can see from the following case that, with search tools, Hailuo AI is able to identify the error in the user request, and claim that NaVi did not beat FaZe, being less likely to be misled by user queries.

# User Request ( Query ➷ )

➷ In 2022 major, why did NaVi beat FaZe?

# MiniMax-Text-01 No-Search ✘

Let me think step by step about the 2022 BLAST Grand Final between NaVi and FaZe Clan:

1.The final took place on May 22, 2022, at the PGL Major Antwerp 2022.

2.NaVi won the series 2-0:

• Map 1 (Mirage): NaVi won 16-10

• Map 2 (Dust2): NaVi won 19-16

• Map 3 (Nuke) was not played as NaVi had already secured the victory

3.Key factors in NaVi’s victory:

• s1mple’s exceptional performance: He had a 1.47 rating across the series

• Strong CT-side performance on both maps

• Better mid-round decision making and adaptability

• Effective utility usage, especially on Dust2

• NaVi showed better composure in crucial moments

4.FaZe’s struggles:

• They couldn’t find their usual form on their map picks

• Some key players underperformed compared to their usual standards

• Struggled with NaVi’s defensive setups

5.The series was notable for being one of s1mple’s best individual performances in a major final, which was a crucial factor in NaVi’s victory.

Would you like me to elaborate on any of these points?

# MiniMax-Text-01 Search ✔

In the 2022 PGL Major Antwerp, Natus Vincere (NAVI) did not beat FaZe Clan. Instead, FaZe Clan emerged victorious against NAVI in the grand final, winning the series 2-0.

The final was a highly anticipated matchup between the two top-ranked teams in the world, with FaZe Clan ultimately proving to be the superior team. The series was a closely contested affair, with NAVI pushing FaZe to overtime in the first map, which FaZe won 19-16. Despite NAVI’s strong performance, FaZe managed to secure the victory in the end.

This win was significant for FaZe Clan as it marked their first Major victory and made them the first completely international roster to win a Counter-Strike: Global Offensive Major. The victory was also a redemption for FaZe Clan after their infamous loss to Cloud9 in the Boston Major in 2018.

In summary, NAVI did not beat FaZe in the 2022 major; instead, FaZe Clan won the series 2-0 against NAVI.

# C. Vision In-house User Experience Benchmark Examples

# C.1. Nature

# Example 1

# User Request

According to the illustration, which planet has the most satellites? Which one has the most large satellites? Please specify the numbers separately. Also, do only planets have satellites?

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/dfb78de9d93f3ec6127cec8219adcaff7cdcc317a33876f914cc2b7fb1f4d65b.jpg]]

# Checklist

• Correctly identifies the planet with the most satellites (Saturn, 9).   
• Correctly identifies the planet with the most large satellites (Jupiter, 4).   
• Identifies asteroids and dwarf planets that have satellites (Ida, Pluto, and Eris).

# MiniMax-VL-01

According to the illustration, the planet with the most satellites is Saturn, which has 9 satellites: Mimas, Enceladus, Tethys, Dione, Rhea, Titan, Hyperion, Iapetus, and Phoebe.

The planet with the most large satellites is Jupiter, which has 4 large satellites: Io, Europa, Ganymede, and Callisto.

Not only planets have satellites. In the illustration, we can see that besides planets, there are other celestial bodies that also have satellites. For example, the dwarf planet Pluto has one satellite: Charon; the dwarf planet Eris has one satellite: Dysnomia; even the asteroid Ida has one satellite: Dactyl.

In summary, the planet with the most satellites is Saturn, the planet with the most large satellites is Jupiter, and celestial bodies with satellites are not limited to planets, but also include dwarf planets and asteroids.

# C.2. Daily Life

# Example 2

# User Request

According to the instructions of this navigation, how should I drive next?

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/521008b22cbf7088175c31a07e550285b57d175c8658ae1507b0c1b07cccb0f3.jpg]]

# Checklist

• Confirmed that the action should take place after 38 meters.   
• Correctly identified the need to turn right.   
• Mentions the need to enter the ramp.

# MiniMax-VL-01

According to the directions on the navigation screen, you are currently driving on a highway and are about to reach an exit. Specifically, you need to enter the ramp after 38 meters and be ready to leave the highway.

Here are the detailed steps:

1. Keep current lane: Keep driving in your current lane, making sure you don’t miss an upcoming exit ramp.   
2. Deceleration: When approaching the exit ramp, slow down appropriately to ensure a safe entry into the ramp. Usually, there will be a speed sign on the exit ramp of the expressway, so keep an eye out for it.   
3. Turn signal: Before you are ready to enter the ramp, turn on the right turn signal to alert the vehicle behind you that you are about to change lanes.   
4. Approach ramp: Follow the navigation directions and enter the ramp after 38 meters. Be aware of curves and slopes on ramps and maintain an appropriate speed.   
5. Continue to follow navigation: After entering the ramp, continue to follow the directions of the navigation. Navigation will guide you through the next segments until you reach your destination.

Be aware of your safety and focus, especially when changing lanes on highways and onto ramps. Wishing you a safe journey!

# Example 3

# User Request

How many of the outfits in each row in the picture are matched with long skirts?

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/3dc3878580cd18130d2216a074548245403e6bab5cc174623e4b69192282c26f.jpg]]

# Checklist

• Accurately identify 3 outfits with long skirts in the first row (columns 2, 5, 6).   
• Accurately identify 2 outfits with long skirts in the second row (columns 2, 6).   
• Accurately identify 1 outfit with a long skirt in the second row (column 1).

# MiniMax-VL-01

# First row:

1. The first set: short skirts   
2. The second set: long skirts   
3. The third set: trousers   
4. Fourth set: trousers   
5. The fifth set: long skirts   
6. Sixth set: long skirts

There are a total of 3 sets in the first row with long skirts.

Second row:

1. The first set: trousers   
2. The second set: long skirts   
3. The third set: trousers   
4. Fourth set: shorts   
5. The fifth set: trousers   
6. Sixth set: long skirts

There are 2 sets of long skirts in the second row.

# Third row:

1. The first set: long skirts   
3. The third set: trousers   
4. Fourth set: trousers   
5. The fifth set: trousers   
2. Second set: trousers   
6. Sixth set: trousers

There is a total of 1 set in the third row with a long skirt.

To sum up, there are 3 sets, 2 sets, and 1 sets of long skirts in each row.

# C.3. Digital Scenarios

# Example 4

# User Request

Based on the provided flowchart and statistical data, determine which step takes the longest and how to optimize it?

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/d48252526de9636d7eb974c3792f251627e571912da4dc3101ace1597c08c8a2.jpg]]

# Checklist

• Correctly identify the longest step: 53.6 seconds to enter invoice details.   
• Measures to optimize invoice detail entry: automated entry, batch processing, simplified operations, etc

# MiniMax-VL-01

According to the provided flowchart and statistics, the longest step is "Enter invoice details," with an average time of 53.6 seconds. To optimize this step, we can take the following measures:

1. Automate data entry: Consider using OCR (Optical Character Recognition) technology to automatically read invoice information and fill it into the system. This will significantly reduce the time spent on manual entry.   
2. Simplify the data entry interface: Redesign the data entry interface to make it more intuitive and user-friendly. For example, use drop-down menus, auto-fill, and smart suggestion features to reduce user input time and error rates.   
3. Batch processing: If possible, allow users to upload invoice information in batches instead of entering them one by one. This can be achieved by developing a batch import feature, where users only need to upload a file containing all invoice information, and the system will automatically parse and fill in the data.   
4. Training and guidance: Provide users with detailed training and operation manuals to help them become familiar with the data entry process, reducing extra time consumption due to unfamiliarity with the operations.   
5. Parallel processing: If the data entry task can be broken down into multiple sub-tasks, consider assigning these sub-tasks to different team members for parallel processing, thereby shortening the overall processing time.   
6. Feedback mechanism: Establish a feedback mechanism that allows users to report issues and difficulties encountered during the data entry process. Based on feedback, continuously improve the data entry process and tools.

By implementing the above methods, we can significantly reduce the time spent on the "Enter invoice details" step, improving the overall efficiency of the process and user satisfaction.

# Example 5

# User Request

Based on the details in the image, what tasks might be prioritized in Katy’s schedule? Please explain your reasoning logic.

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/06e4a08284ea851d71f8311eb74297fcfc95dc45bc9f8d20a5d1bff326550ba2.jpg]]

# Checklist

• Noticing that the "Surprise Birthday Planning" email is pinned, infer its importance and prioritize it.   
• Identifying the "Contoso Airlines" email as important information for an upcoming trip, thus prioritizing it.   
• Combining the content of the emails being edited, infer that she is arranging apartment viewings, so this task is also on the priority list.   
• Recognizing that she will have lunch with Jane in half an hour.

# MiniMax-VL-01

Based on the details in the image, Katy may prioritize the following tasks in her schedule:

Arrange apartment viewing: The email on the right side shows a message Katy sent to Dianne Russell inquiring about the apartment viewing. The email mentions that Katy hopes to schedule a time to view the apartment soon and notes that she is available Wednesday and Friday afternoons. This indicates that the apartment viewing is an important item on her current schedule and may be prioritized.

Reply to birthday planning email: In the email list on the left, Elvia Atkins sent an email about birthday planning at 8:32 AM. This email may require Katy to respond or take action as soon as possible, especially if the birthday event is approaching.

Confirm flight booking: Contoso Airlines sent a confirmation email for the flight booking at 2:41 PM. This indicates that Katy may need to confirm her travel plans and ensure that all related matters are properly arranged.

Team photo update: Lydia Bauer sent an email regarding the team photo at 11:10 AM. This email may require Katy to confirm or address the update of the team photo to ensure the accuracy and timeliness of team information.

Yoga workshop arrangement: Daisy Philips sent an email about the yoga workshop at 9:16 AM. This indicates that Katy may need to arrange or confirm her participation in the yoga workshop, which is also important for her personal schedule.

Lunch appointment: The top right corner of the image shows that Katy has a lunch appointment with Jane in 30 minutes at Fourth Coffee. This is an upcoming activity and definitely a priority.

The reasoning is as follows:

Email sending time and content: Earlier sent emails (such as birthday planning and yoga workshop) may need to be prioritized as they may involve upcoming events or deadlines.

Nature of the emails: Emails related to personal arrangements (such as apartment viewing and flight booking) are usually prioritized as they directly affect personal life and work arrangements.

Email priority: Some emails may be marked as high priority or require immediate response, which will affect Katy’s order of handling.

In summary, Katy may prioritize tasks such as arranging apartment viewings, confirming birthday planning, and confirming flight bookings to ensure her personal and work schedules run smoothly.

# C.4. Special Imaging

# Example 6

# User Request

![[MiniMax-01: Scaling Foundation Models with Lightning Attention/images/dfa23018df94aca08f52e1a86e3640dd84de23f21795bf358d3b37fd4b6fcc6f.jpg]]  
What does the left and right comparison of this image illustrate?

# Checklist

• Point out that these are two photos taken by satellite remote sensing.   
• Compare the time, with the left side being August 2023 and the right side being September 2023.   
• Show that large areas have changed color (from green to brown and gray).   
• The reasons for changes in vegetation conditions can be fires, droughts, etc.

# MiniMax-VL-01

This image shows a comparison of satellite images of the same area in August 2023 and September 2023. From the images, it can be seen that the August image (left) shows a rich vegetation cover in the area, with green areas widely distributed, indicating that the vegetation was growing well at that time. However, by September (right), the vegetation cover in the area had significantly decreased, with many green areas turning brown or becoming bare land, suggesting that the vegetation may have experienced a significant reduction or degradation.

This change may be caused by various factors, including but not limited to climate change, natural disasters (such as fires or droughts), and human activities (such as agricultural development or deforestation). The specific reasons need to be analyzed in conjunction with more data and background information. For example, if the area experienced severe drought or fire in September, then the reduction in vegetation can be explained. Additionally, human activities such as large-scale agricultural development or deforestation may also lead to a significant decrease in vegetation cover.

Overall, this comparative image clearly shows the significant changes in vegetation cover in the area within a month, prompting us to pay attention to and study the reasons behind these changes in order to take appropriate measures to protect and restore the ecological environment.

# D. Evaluation Details for Selected Benchmarks

# D.1. MMLongBench-Doc

For evaluating models with context-length and image-number limitations that exceed requirements of MMLongBench-Doc, we adopt an image concatenation approach suggested by the original repository7, resulting in the concatenation of all images extracted from a single PDF input into 5 images for the open-source models evaluated and 10 for Claude-3.5-Sonnet-1022. For evaluating other commercial models and MiniMax-Text-01, we use the default configuration which sets the maximum number of image pages to 120 and resolution to 144.

# D.2. MEGA-Bench

MEGA-Bench is a comprehensive multimodal benchmark that spans 7 input formats, 6 output formats, 10 different types of skills, and varying forms of visual inputs, including images and videos. Each request may consider multiple images, consisting of visual task description, request-response demonstration and videos. For video inputs, the benchmark slices each video into multiple frames. The number of frames and the resulting number of total input images are limited to the model’s context length and image constraints. We follow the general principles of the original repository8 when deciding our evaluation configurations, as detailed in Table 14.

Table 14 | Configuration of different models for MEGA-Bench.   

<table><tr><td>Model/Configuration.</td><td>MAX_NUM_IMAGE</td><td>TOTAL_DEMOVIDEO_FRAME</td></tr><tr><td>GPT-4o-2024-1120</td><td>64</td><td>8</td></tr><tr><td>Claude-3.5-Sonnet-1022</td><td>64</td><td>8</td></tr><tr><td>Gemini-1.5-Pro-002</td><td>128</td><td>16</td></tr><tr><td>Gemini-2.0-Flash-exp</td><td>128</td><td>16</td></tr><tr><td>InternVL2.5-78B</td><td>24</td><td>2</td></tr><tr><td>Qwen2-VL-72B-Instruct</td><td>10</td><td>1</td></tr><tr><td>LLama-3.2-90B</td><td>10</td><td>1</td></tr><tr><td>MiniMax-VL-01</td><td>128</td><td>16</td></tr></table>

# D.3. MMMU & DocVQA

We note that rule-based methods may misjudge cases where the correct answer has mulitple forms (e.g. U.S. vs. United States), we adopt GPT-4o (specifically GPT-4o-2024-05-13) as the judge model if the rule-based method fails for MMMU and DocVQA evaluation.