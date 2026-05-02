---
title: "OmniGen2: Towards Instruction-Aligned Multimodal Generation"
source: "https://arxiv.org/html/2506.18871v4"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Chenyuan Wu <sup>1,2</sup>, Jiahao Wang <sup>1,3∗</sup>, Pengfei Zheng <sup>1,2∗</sup>, Ruiran Yan <sup>1,2∗</sup>, Shitao Xiao <sup>1∗</sup> <sup>4</sup>, Xin Luo <sup>1,2∗</sup>,  
Yueze Wang <sup>1∗</sup>, Wanli Li <sup>1,4</sup>, Xiyan Jiang <sup>1,4</sup> <sup>2</sup>, Yexin Liu <sup>1</sup> <sup>2</sup>, Junjie Zhou <sup>1</sup>, Ziyi Xia <sup>1</sup>,  
Ze Liu <sup>1,2</sup>, Chaofan Li <sup>1</sup>, Haoge Deng <sup>1,3</sup>, Kun Luo <sup>1,3</sup>, Bo Zhang <sup>4</sup>, Jiajun Zhang <sup>3</sup>,  
Dong Liu <sup>2</sup>, Defu Lian <sup>2</sup>, Xinlong Wang <sup>1</sup>, Zhongyuan Wang <sup>1</sup>, Tiejun Huang <sup>1</sup>, Zheng Liu <sup>1</sup>  
<sup>1</sup> Beijing Academy of Artificial Intelligence, <sup>2</sup> University of Science and Technology of China,  
<sup>3</sup> Institute of Automation, Chinese Academy of Sciences, <sup>4</sup> Zhejiang University  
{stxiao, yzwang}@baai.ac.cn  zhengliu1026@gmail.com  
Co-first Authors and Listed in Alphabetical OrderCore ContributorCorresponding AuthorProject Lead

###### Abstract

Multimodal generative models can process instructions in various modalities and demonstrate outstanding performance across a wide range of image generation tasks. However, their robustness in complex real-world scenarios remains limited due to insufficient generalized instruction alignment. We introduce OmniGen2, a unified multimodal generator designed to follow complex, fine-grained instructions. Our core contribution is a two-stage design that first builds a strong, world-knowledge-grounded foundation model and then aligns it using a progressive, multi-task instruction tuning strategy. The foundation model features a streamlined architecture with decoupled decoding for versatile multimodal generation and a novel positional encoding scheme to improve learning efficiency. We ground this model in real-world knowledge using large-scale data construction pipelines. Building on this foundation, we propose a progressive, reinforcement-based alignment process. This phase carefully schedules training tasks and reward signals to foster cross-task knowledge transfer, significantly improving the model’s instruction-following capabilities. Our models demonstrate competitive performance on standard benchmarks and our dedicated in-context generation benchmark, OmniContext. We have released our models, code, benchmark, and training datasets at [https://github.com/VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2).

## 1 Introduction

![Refer to caption](https://arxiv.org/html/2506.18871v4/x1.png)

Figure 1: Overview of versatile abilities of OmniGen2.

Multimodal image generation has witnessed rapid progress in the past year. Generative models such as GPT-Image-1 [^34], Flux [^40], Qwen-Image [^94], Seedream [^78] and NanoBanana [^29] demonstrate increasingly broad and versatile capabilities such as stylization, text rendering, in-context generation, and knowledge-driven generation, marking a significant step toward general-purpose generative intelligence. Given these diverse capabilities, it is essential to perform multimodal instruction alignment to ensure the controllability, semantic consistency and overall generation quality. This involves two key challenges. The first is constructing a robust and versatile foundation model. The model must be endowed with nascent instruction following capabilities and broad world knowledge, while strictly avoiding over-training. The second is aligning the foundation model. This alignment requires explicit and comprehensive reward signals and must ensure consistency across all generation tasks.

Existing open-source generation models are somewhat deficient as initial base model. Some models are specialized and can not handle tasks beyond their training scope while some are over-optimized towards specific aesthetic preferences, resulting in a severe loss of plasticity. Meanwhile, instruction alignment requires the foundation model to possess a deep understanding of multimodal semantic and task intent. Therefore, we aim to first establish a base model which is simple, versatile and flexible.

The versatility of a generative model depends a lot on the scale and diversity of its training data. Existing datasets are typically generated either via inpainting models [^93], which have limited task coverage, or by retrieving images from the Internet [^98], which results in limited data volume and low image quality. To address this, we develop extensive data construction pipelines that leverage video sources, providing richer in-context and editing examples.

A strong architecture is equally crucial. OmniGen2 achieves unified multimodal generation by conditioning the diffusion transformer on the variable-length hidden states of a Vision Language Model (VLM), effectively leveraging the VLM’s deep semantic understanding and rich world knowledge. To support diverse tasks, we introduce Omni-RoPE, which enhances spatial consistency across images and improves cross-image localization. While conceptually similar to MetaQuery [^66], OmniGen2 differs in execution: rather than using fixed-length query tokens, it conditions the diffusion decoder on the VLM’s variable-length hidden states, avoiding information bottlenecks. During the majority of the training process, the VLM is frozen, and optimization focuses on image rendering, making OmniGen2 more efficient than models like Mogao [^46] and BAGEL [^18].

Once the foundation model is obtained via pre-training and fine-tuning, we apply progressive reinforcement learning to facilitate instruction alignment of OmniGen2. Specifically, we adopt Group Relative Policy Optimization (GRPO) [^79] and divide the instruction alignment process into multiple sequential stages. At each stage, appropriate reward is selected to optimize the alignment for specific target tasks. The training sequence is carefully organized to promote inter-task transfer.

Our extensive evaluation of OmniGen2 reveals its competitive performance across various task domains, including text-to-image (T2I) generation, image editing (Edit), and in-context generation (IC). Instruction alignment consistently and significantly improves the performance of the base model across all these tasks. Notably, for the in-context generation task, there is currently a lack of well-established public benchmark to systematically assess and compare the key capabilities of different models. To mitigate this limitation, we introduce the OmniContext benchmark, comprising eight task categories specifically designed to evaluate consistency across individuals, objects, and scenes.

In summary, our main contributions are as follows:

- We introduce OmniGen2, a powerful multimodal generative model that is systematically instruction aligned. The model demonstrates superior instruction following ability, context consistency, and generation quality across diverse task scenarios.
- We establish an end-to-end pipeline to achieve comprehensive instruction alignment. This pipeline spans from strong foundation model construction to dedicated multi-task alignment.
- We present the OmniContext benchmark, a rigorous suite designed to evaluate in-context image generation, providing the community with a standardized tool to measure progress in this key area.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x2.png)

Figure 2: Architecture of OmniGen2. OmniGen2 employs separate transformers for autoregressive and diffusion. Two distinct image encoders are utilized: ViT encodes images for input into the text transformer, while VAE encodes images for the diffusion transformer.

## 2 Dataset Construction

To build a versatile and robust base model, high-quality, large-scale, and diverse training data is essential. Our goal is to equip the base model with broad world knowledge and the ability to generate varied content. A key challenge is the lack of high-quality public data for complex tasks like detailed image editing and consistent in-context generation. Therefore, we not only gather existing datasets but also build our own scalable pipelines to create the high-quality data needed to fill these gaps.

Foundational Knowledge and General Capabilities. To build a strong foundation, we first curate a massive dataset covering both multimodal understanding and text-to-image (T2I) generation. For the former, we adopt LLaVA-OneVision [^41]. For T2I, we collect approximately 140M open-source image-text pairs from diverse datasets [^43] [^11] [^12] [^77] [^9] [^63] [^44] [^84] [^10], supplemented with 10M proprietary images annotated by Qwen2.5-VL-72B [^3].

Advanced Capabilities for Editing and In-Context Tasks. To address the data scarcity in more complex domains, we develop dedicated construction pipelines. For image editing, we integrate public datasets such as SEED-Data-Edit [^25] and OmniEdit [^93], and further construct high-quality editing data using inpainting and video-based pipelines. For in-context generation and editing, we build our datasets from video sources to model consistent subjects across varying scenarios. We employ vision-language models for subject detection, segmentation, and semantic filtering, resulting in diverse and semantically consistent triplets for training.

Fostering Higher-Level Reasoning. Finally, to push the base model’s capabilities beyond simple generation, we construct interleaved and reflection datasets to enhance temporal reasoning and self-correction capabilities in multimodal models. Detailed pipeline steps, examples and the capability of reflection are provided in Appendix 9.2, 9.3, 9.4, 9.5, 9.6.

## 3 Method

![Refer to caption](https://arxiv.org/html/2506.18871v4/x3.png)

Figure 3: Illustration of Omni-RoPE. Each token in the k -th image is assigned a three-dimensional positional identifier ( Δ I ), h w (\\Delta\_{I}^{(k)},h,w), where \\Delta\_{I}^{(k)} denotes the instance identity shared by all tokens within the same image, and (h,w) are the local 2D spatial coordinates computed from 0 (0,0). This decomposition enables the model to distinguish different images while preserving local spatial consistency for tasks such as image editing.

OmniGen2 is built on three key components: (1) a decoupled architecture for unified generation, (2) Omni-RoPE for efficient contextual learning, and (3) a multi-stage training and alignment curriculum that progresses from broad knowledge to fine-grained instruction following.

### 3.1 Overall Architecture

We aim to design a simple, efficient, and versatile architecture for multimodal generation. Following this principle, OmniGen2 utilizes decoupled pathways for text and image generation. It employs two distinct transformer modules to efficiently facilitate the concurrent support of both understanding and generation capabilities, as illustrated in Figure 2. The autoregressive transformer model is initialized from a VLM (Qwen2.5-VL-3B [^3]). This VLM provides extensive world knowledge and deep understanding of multimodal instructions. The diffusion transformer is randomly initialized and dedicated solely to high-fidelity image synthesis.

The two modules operate in sequence. First, the VLM processes the input multimodal context. A special token, <|img|>, is learned to distinguish the understanding and generation tasks. The generation of this token triggers the image generation. The corresponding hidden states from the VLM are extracted and fed to the diffusion decoder as a condition. It encodes high-level semantic instruction. Besides the high-level semantic encoding from the VLM, we incorporate low-level image features to ensure consistency of fine-grained visual details for tasks like image editing. We utilize Flux-VAE [^39] for this purpose. This approach avoids complex architectural modifications to the pre-trained VLM, thereby preserving its powerful instruction understanding capabilities.

### 3.2 Diffusion Decoder

OmniGen2 employs computational efficient conditioning mechanism for the diffusion decoder. Existing methods such as MetaQuery [^66] compressing instructions into a fixed set of learnable query tokens can create an information bottleneck. In contrast, OmniGen2 directly leverages the rich hidden states from the VLM’s final layer. Furthermore, we utilize only the hidden states corresponding to text tokens, as the VAE features already provide sufficient visual detail.

Within the diffusion decoder, we adopt a unified transformer backbone, following the architecture of Lumina-Image 2.0 [^69], where the parameters are shared across modalities. This design choice is motivated by the motivation that language and vision share substantial semantic representations. Consequently, parameter sharing provides a more natural and efficient means of cross-modal alignment than maintaining separate pathways [^39] [^40] [^18]. Meanwhile, this design facilitates more consistent information exchange between modalities. Before processed by the core transformer blocks, input conditioning signals (VLM hidden states, VAE features, and noisy latents) are aligned by a lightweight two-layer transformer refiner. This refiner shares the same architecture as the transformer block employed in Lumina-Image 2.0.

<table><thead><tr><th>Method</th><th>PosID <math><semantics><mrow><msub><mi>PosID</mi> <mi>k</mi></msub> <mo>⁡</mo> <mrow><mo>(</mo><mi>h</mi><mo>,</mo><mi>w</mi><mo>)</mo></mrow></mrow> <annotation>\operatorname{PosID}_{k}(h,w)</annotation></semantics></math></th><th>Steps to Target <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th>Final Loss <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>Lumina-Image-2.0’s</th><td><math><semantics><mrow><mo>(</mo><mn>0</mn><mo>,</mo><mrow><mi>h</mi> <mo>+</mo> <msub><mi>Δ</mi> <mi>h</mi></msub></mrow><mo>,</mo><mrow><mi>w</mi> <mo>+</mo> <msub><mi>Δ</mi> <mi>w</mi></msub></mrow><mo>)</mo></mrow> <annotation>(0,\,h+\Delta_{h},\,w+\Delta_{w})</annotation></semantics></math></td><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 2,500</td><td>0.017</td></tr><tr><th>Qwen2-VL’s</th><td><math><semantics><mrow><mo>(</mo><msub><mi>Δ</mi> <mi>I</mi></msub><mo>,</mo><mrow><mi>h</mi> <mo>+</mo> <msub><mi>Δ</mi> <mi>I</mi></msub></mrow><mo>,</mo><mrow><mi>w</mi> <mo>+</mo> <msub><mi>Δ</mi> <mi>I</mi></msub></mrow><mo>)</mo></mrow> <annotation>(\Delta_{I},\,h+\Delta_{I},\,w+\Delta_{I})</annotation></semantics></math></td><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 1,200</td><td>0.005</td></tr><tr><th>Omni-RoPE (Ours)</th><td rowspan="2"><math><semantics><mrow><mo>(</mo><msub><mi>Δ</mi> <mi>I</mi></msub><mo>,</mo><mi>h</mi><mo>,</mo><mi>w</mi><mo>)</mo></mrow> <annotation>(\Delta_{I},\,h,\,w)</annotation></semantics></math></td><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 800</td><td>0.003</td></tr><tr><th>   + Image Index Emb.</th><td><math><semantics><mo>∼</mo> <annotation>\sim</annotation></semantics></math> 800</td><td>0.002</td></tr></tbody></table>

Table 1: Comparison of RoPE designs in the toy reconstruction task. Models are trained to reproduce the $k$ -th image among randomly sampled inputs. We report the number of steps required to reach the target (loss $<$ 0.014). Omni-RoPE achieves both faster convergence and lower final loss. Note: $\Delta_{h}$ and $\Delta_{w}$ denote the accumulated coordinate offsets in the height and width dimensions, respectively, while $\Delta_{I}$ represents the accumulated offset in the instance dimension.

### 3.3 Omni-RoPE: Unified Positional Encoding

We introduce Omni-RoPE, a positional encoding scheme tailored for multimodal contexts with complex structural correspondence. Conventional positional encodings cannot reliably distinguish multiple images or preserve spatial alignment across editing operations. Omni-RoPE addresses this limitation by extending Rotary Position Embedding (RoPE) [^83] to a unified multimodal setting.

Unified formulation. As shown in Figure 3, each token at coordinates $(h,w)$ in the $k$ -th image is assigned a three-dimensional positional identifier:

$$
\operatorname{PosID}_{k}(h,w)=(\Delta_{I}^{(k)},\,h,\,w),
$$

where $\Delta_{I}^{(k)}$ denotes the instance identity, which distinguishes different images or modalities, and $(h,w)$ are the 2D spatial coordinates. All tokens from the same image share the same $\Delta_{I}^{(k)}$, while the spatial mapping remains unchanged, i.e., $\mathcal{P}_{h}^{(k)}(h)=h$ and $\mathcal{P}_{w}^{(k)}(w)=w$. For text tokens, this formulation naturally reduces to a standard 1D positional index.

This decomposition separates image identity from intra-image spatial layout. Spatial coordinates are computed locally from $(0,0)$ within each image, ensuring that corresponding patches in input and output images receive identical embeddings, thereby preserving spatial alignment and edit consistency. Meanwhile, $\Delta_{I}^{(k)}$ provides an explicit channel for distinguishing visual instances, which is critical for in-context image generation and multi-image reasoning.

Toy experiment verification. To evaluate positional correspondence, we design a controlled toy task in which a randomly initialized model is trained to reconstruct the $k$ -th image from multiple randomly sampled input images, thereby isolating the effect of positional encoding. We measure efficiency by the number of training steps required to reach a high-fidelity reconstruction target (loss $<$ 0.014).

As reported in Table 1, the RoPE variants used in Lumina-Image-2.0 and Qwen2-VL [^90] require substantially more training steps to converge, indicating weaker alignment across visual instances. In contrast, Omni-RoPE converges markedly faster and achieves the lowest reconstruction loss, demonstrating stronger spatial correspondence and instance discrimination. Incorporating an image index embedding [^13] further improves the final reconstruction fidelity at no additional cost.

### 3.4 Foundation Model Training

We construct the foundation model using a two-stage training pipeline comprising from-scratch pre-training followed by supervised fine-tuning. To accommodate variable context lengths in unified multi-task training, we employ FlashAttention2 [^17] for efficient sequence processing. The model is optimized using the Rectified Flow objective [^55] [^49] [^2].

Pre-training. This stage focuses on learning general-purpose visual and semantic representations from large-scale datasets. The model is trained through a resolution-based curriculum (256 <sup>2</sup> $\rightarrow$ 512 <sup>2</sup> $\rightarrow$ 1024 <sup>2</sup>). For each resolution, we first conduct training on the text-to-image (T2I) task to establish strong text–image alignment. Then, we introduce a curated mixed-task dataset (covering image editing and in-context generation) to diversify the model’s capabilities.

Supervised Fine-Tuning. After pre-training, the model undergoes SFT at 1024 <sup>2</sup> resolution to refine high-level reasoning and compositional skills. We train on a mixture of curated datasets and distilled data from proprietary models, aiming to enhance instruction following and visual fidelity.

Through two-stage training, the model acquired initial instruction-following skills and versatile generation capability, setting a foundation for subsequent alignment. Detailed configurations for each stage are provided in Appendix 9.7.

### 3.5 Instruction Alignment

We perform online reinforcement learning for multi-task alignment via a progressive curriculum instead of a single joint training stage to ensure stability and avoid task interference. The key challenge is to achieve synergistic gains across tasks without degrading individual performance.

We define a sequence of training tasks $\mathcal{S}=\langle\mathcal{T}_{1},\dots,\mathcal{T}_{N}\rangle$, where each task $\mathcal{T}=(\tau,\delta,\mathcal{R})$ consists of a task type $\tau\in\{\text{T2I, Edit, IC}\}$, a task instance $\delta$, and a reward signal $\mathcal{R}$. Our goal is to cover all fundamental task types for comprehensive alignment.

For $\tau=\text{Edit}$ and $\tau=\text{IC}$, we adopt general-purpose tasks to enhance instruction-following and compositional abilities. As these tasks lack verifiable rewards, we employ learned reward models: EditScore [^57] for Edit and Qwen2.5-VL-72B [^3] for IC. For $\tau=\text{T2I}$, we select GenEval, which provides verifiable rewards and exhibits strong overlap with Edit and IC.

We exclude T2I tasks with limited generalization or high reward-hacking risk. In particular, aesthetic rewards such as HPSv3 [^58] are omitted due to reward hacking, and specialized tasks (e.g., OCR) are excluded as they lack synergy with general instruction-following.

This yields a three-stage curriculum $\langle\mathcal{T}_{1},\mathcal{T}_{2},\mathcal{T}_{3}\rangle$, trained with Flow-GRPO [^52]:

- $\mathcal{T}_{1}=(\text{Edit, general editing, EditScore})$,
- $\mathcal{T}_{2}=(\text{T2I, GenEval, Verifiable Reward})$,
- $\mathcal{T}_{3}=(\text{IC, general in-context, Qwen2.5-VL-72B})$.

Our RL data includes 50k T2I prompts from Flow-GRPO [^55], 110k editing samples from EditScore [^57], and 180k in-context data from Echo-4o [^102].

## 4 OmniContext Benchmark

Rigorous evaluation is essential for generalized instruction alignment, particularly for reference-based tasks testing core consistency. However, existing benchmarks fall short, lacking support for multiple input images and diverse tasks. For instance, DreamBench [^76] only contains 30 objects and 25 prompt templates. And relying on simplistic metrics like CLIP-I fails on multi-subject evaluation and offers no explainability. To address these critical gaps, we introduce OmniContext, a comprehensive benchmark designed to assess a model’s ability to generate content consistent with user-specified context images.

To bridge these gaps, we construct OmniContext using a large-scale, manually collected dataset of high-quality images including personal photos, open-source images, animation stills and AI-generated images. These images are grouped into three distinct categories — Character, Object, and Scene — and exhibit diverse coverage across various domains, as illustrated in Figure 4. We define three task categories(SINGLE, MULTIPLE, and SCENE), each with 50 examples per subtask. SINGLE uses one context image, MULTIPLE combines multiple subjects, and SCENE conditions on environmental context.

Image–prompt pairs are constructed through a hybrid process combining MLLMs and manual annotation. MLLMs first filter low-quality samples, after which experts select images based on clarity, aesthetics, and diversity. Prompts are generated with GPT-4o and refined for semantic and syntactic variety.

We use GPT-4.1 to assess outputs on three metrics: Prompt Following (PF), Subject Consistency (SC), and an Overall Score (geometric mean of PF and SC). Following VIEScore [^38], GPT-4.1 provides both scores (0–10) and rationales to justify its evaluations. We believe the OmniContext will serve as a valuable resource for driving future research in controllable, reference-based image generation.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x4.png)

Figure 4: Overview of OmniContext benchmark. Left: Image genres included in OmniContext. Right: Example images for each genre in OmniContext.

## 5 Experiments

<table><tbody><tr><td>Model</td><td># Params</td><td colspan="3">Understanding</td><td colspan="2">Image Generation</td><td colspan="2">Image Editing</td><td colspan="3">In-context Generation</td></tr><tr><td></td><td></td><td>MMB <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>MMMU <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>MM-Vet <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>GenEval <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>OneIG-Bench-EN <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>ImgEdit-Bench <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>GEdit-Bench-EN <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Single <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Multiple <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Scene <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>LLaVA-1.5 <sup><a href="#fn:51">51</a></sup></td><td>-</td><td>36.4</td><td>67.8</td><td>36.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLaVA-NeXT <sup><a href="#fn:50">50</a></sup></td><td>-</td><td>79.3</td><td>51.1</td><td>57.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SDXL <sup><a href="#fn:67">67</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.55</td><td>0.32</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SD3-medium <sup><a href="#fn:1">1</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FLUX.1-dev <sup><a href="#fn:39">39</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.66</td><td>0.43</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen-Image <sup><a href="#fn:94">94</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.91</td><td>0.54</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Instruct-P2P <sup><a href="#fn:5">5</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.88</td><td>3.68</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MagicBrush <sup><a href="#fn:108">108</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.90</td><td>1.86</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AnyEdit <sup><a href="#fn:104">104</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.45</td><td>3.21</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Step1X-Edit <sup><a href="#fn:54">54</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.06</td><td>6.70</td><td>-</td><td>-</td><td>-</td></tr><tr><td>IC-Edit <sup><a href="#fn:110">110</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.05</td><td>4.84</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UNO <sup><a href="#fn:97">97</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>6.72</td><td>4.48</td><td>3.59</td></tr><tr><td>InfiniteYou <sup><a href="#fn:37">37</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>6.05</td><td>-</td><td>-</td></tr><tr><td>DreamO <sup><a href="#fn:62">62</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>7.65</td><td>7.05</td><td>4.52</td></tr><tr><td>UMO <sup><a href="#fn:15">15</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>7.78</td><td>7.14</td><td>6.78</td></tr><tr><td>Show-o <sup><a href="#fn:99">99</a></sup></td><td>-</td><td>-</td><td>27.4</td><td>-</td><td>0.68</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Janus-Pro <sup><a href="#fn:14">14</a></sup></td><td>-</td><td>75.5</td><td>36.3</td><td>39.8</td><td>0.80</td><td>0.26</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Emu3 <sup><a href="#fn:91">91</a></sup></td><td>-</td><td>58.5</td><td>31.6</td><td>37.2</td><td>0.54 / <math><semantics><msup><mn>0.66</mn> <mo>†</mo></msup> <annotation>0.66^{\dagger}</annotation></semantics></math></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MetaQuery-XL <sup><a href="#fn:66">66</a></sup></td><td>7B + 1.6B <sup>∗</sup></td><td>83.5</td><td>58.6</td><td>66.6</td><td><math><semantics><msup><mn>0.80</mn> <mo>†</mo></msup> <annotation>0.80^{\dagger}</annotation></semantics></math></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BLIP3-o 8B <sup><a href="#fn:10">10</a></sup></td><td>7B + 1.4B <sup>∗</sup></td><td>83.5</td><td>58.6</td><td>66.6</td><td><math><semantics><msup><mn>0.84</mn> <mo>†</mo></msup> <annotation>0.84^{\dagger}</annotation></semantics></math></td><td>0.31</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BAGEL <sup><a href="#fn:18">18</a></sup></td><td>7B + 7B <sup>∗</sup></td><td>85.0</td><td>55.3</td><td>67.2</td><td>0.82 / 0.88 <sup>†</sup></td><td>0.36</td><td>3.20</td><td>6.52</td><td>6.25</td><td>6.02</td><td>5.08</td></tr><tr><td>UniWorld-V1 <sup><a href="#fn:48">48</a></sup></td><td>7B + 12B <sup>∗</sup></td><td>83.5</td><td>58.6</td><td>67.1</td><td><math><semantics><msup><mn>0.84</mn> <mo>†</mo></msup> <annotation>0.84^{\dagger}</annotation></semantics></math></td><td>-</td><td>3.26</td><td>4.85</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen-Image-Edit-2509 <sup><a href="#fn:94">94</a></sup></td><td>7B + 20B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.41</td><td>7.54</td><td>8.74</td><td>8.13</td><td>6.55</td></tr><tr><td>Gemini 2.5 Flash Image <sup><a href="#fn:29">29</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.55</td><td>4.28</td><td>7.10</td><td>8.77</td><td>8.06</td><td>7.01</td></tr><tr><td>OmniGen <sup><a href="#fn:98">98</a></sup></td><td>3.8B</td><td>-</td><td>-</td><td>-</td><td>0.68</td><td>-</td><td>2.96</td><td>5.06</td><td>6.46</td><td>5.26</td><td>4.34</td></tr><tr><td>myblueOmniGen2</td><td>3B + 4B <sup>∗</sup></td><td>79.1</td><td>53.1</td><td>61.8</td><td>0.95</td><td>0.47</td><td>3.69</td><td>7.21</td><td>8.41</td><td>7.73</td><td>7.86</td></tr></tbody></table>

Table 2: Comparison of different models across Understanding, Generation, Editing, and In-context Generation tasks. \*: The first term represents the number of parameters for text generation, while the second term corresponds to the number of parameters allocated for image generation. $\dagger$ refers to the methods using LLM rewriter. ^7fc440

In this section, we conduct a comprehensive evaluation of OmniGen2 to demonstrate its unified capabilities across a wide spectrum of generation tasks. The overall comparison results are presented in Table 2.

### 5.1 Visual Understanding

OmniGen2 leverages Qwen2.5-VL-3B-Instruct [^3] for visual understanding. As shown in Table 2, our model demonstrates robust multimodal comprehension, achieving solid scores of 79.1 on MMBench [^56], 53.1 on MMMU [^107], and 61.8 on MM-Vet [^105]. These results confirm that OmniGen2 possesses a solid foundation for interpreting complex visual and textual instructions, which is essential for high-quality, instruction-aligned generation. ^bcdee9

### 5.2 Text-to-Image Generation

We assess OmniGen2’s T2I generation capabilities on two standard benchmarks: GenEval [^26], which evaluates compositional generation, and OneIG-Bench [^7] which evaluate models across multiple dimensions, including prompt-image alignment, text rendering precision, reasoning-generated content, stylization, and diversity.

As shown in Table 2, OmniGen2 delivers strong image generation performance on complex, compositional prompts, achieving an overall score of 0.95 on GenEval. This surpasses other powerful unified models such as UniWorld-V1 (0.84) and BAGEL (0.88), as well as Qwen-Image, a model specialized for T2I generation, highlighting the effectiveness of the RL alignment strategy in OmniGen2. On the more comprehensive OneIG-Bench, OmniGen2 continues to demonstrate competitive realism, achieving an overall score of 0.47, outperforming most existing models and trailing only behind large-scale models such as Gemini 2.5 Flash Image and Qwen-Image. More details are provided in Appendix 8.1 and 9.10.

### 5.3 Image Editing

<table><tbody><tr><th>Method</th><td colspan="3">Emu-Edit</td><td colspan="3">GEdit-Bench-EN</td></tr><tr><th></th><td>CLIP-I <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>CLIP-Out <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DINO <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PQ <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>O <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><th>Gemini-2.0-Flash-Image <sup><a href="#fn:28">28</a></sup></th><td>-</td><td>-</td><td>-</td><td>6.73</td><td>6.61</td><td>6.32</td></tr><tr><th>Gemini-2.5-Flash-Image <sup><a href="#fn:29">29</a></sup></th><td>-</td><td>-</td><td>-</td><td>7.41</td><td>7.96</td><td>7.10</td></tr><tr><th>GPT-4o <sup><a href="#fn:64">64</a></sup></th><td>-</td><td>-</td><td>-</td><td>7.85</td><td>7.62</td><td>7.53</td></tr><tr><th>Instruct-Pix2Pix <sup><a href="#fn:5">5</a></sup></th><td>0.856</td><td>0.292</td><td>0.773</td><td>3.58</td><td>5.49</td><td>3.68</td></tr><tr><th>MagicBrush <sup><a href="#fn:108">108</a></sup></th><td>0.877</td><td>0.298</td><td>0.807</td><td>4.68</td><td>5.66</td><td>4.52</td></tr><tr><th>AnyEdit <sup><a href="#fn:104">104</a></sup></th><td>-</td><td>-</td><td>-</td><td>3.18</td><td>5.82</td><td>3.21</td></tr><tr><th>OmniGen <sup><a href="#fn:98">98</a></sup></th><td>-</td><td>-</td><td>-</td><td>5.96</td><td>5.89</td><td>5.06</td></tr><tr><th>ICEdit <sup><a href="#fn:110">110</a></sup></th><td>0.907</td><td>0.305</td><td>0.866</td><td>5.11</td><td>6.85</td><td>4.84</td></tr><tr><th>Step1X-Edit <sup><a href="#fn:54">54</a></sup></th><td>0.860</td><td>0.304</td><td>0.782</td><td>7.09</td><td>6.76</td><td>6.70</td></tr><tr><th>BAGEL <sup><a href="#fn:18">18</a></sup></th><td>0.839</td><td>0.307</td><td>0.753</td><td>7.36</td><td>6.83</td><td>6.52</td></tr><tr><th>UniWorld-V1 <sup><a href="#fn:48">48</a></sup></th><td>-</td><td>-</td><td>-</td><td>4.93</td><td>7.43</td><td>4.85</td></tr><tr><th>Qwen-Image-Edit-2509 <sup><a href="#fn:94">94</a></sup></th><td>-</td><td>-</td><td>-</td><td>8.15</td><td>7.86</td><td>7.54</td></tr><tr><th>myblueOmniGen2</th><td>0.896</td><td>0.311</td><td>0.876</td><td>7.58</td><td>7.94</td><td>7.21</td></tr></tbody></table>

Table 3: Quantitative comparison on Emu-Edit [^81] and GEdit-Bench-EN [^54]. For Emu-Edit, CLIP-I/DINO measure consistency with the source image, while CLIP-Out measures alignment with the caption of target image, CLIP-B/32 [^71] and DINO-S/16 [^6] are leveraged for feature calculation. For GEdit-Bench, SC (Semantic Consistency) evaluates instruction following, and PQ (Perceptual Quality) assesses image naturalness and artifacts. Higher scores are better for all metrics.

Image editing is a cornerstone of OmniGen2’s capabilities. We rigorously evaluate its performance across three diverse benchmarks: Emu-Edit [^80], GEdit-Bench-EN [^54] and ImgEdit-Bench [^103]. The results collectively demonstrate that OmniGen2 achieve a strong performance in instruction-based image editing.

As shown in Table 3, OmniGen2 exhibits an exceptional balance between edit accuracy and image preservation. On Emu-Edit, our model achieves the highest CLIP-Out score (0.311), indicating it most effectively applies the requested edits among all compared models. Concurrently, it secures the second-best scores for CLIP-I (0.896) and best scores for DINO (0.876), which measure the preservation of unedited regions. This combination highlights OmniGen2’s proficiency in making precise, localized changes without disturbing the rest of the image. This strong instruction-following capability is further confirmed on GEdit-Bench, where OmniGen2 achieves the second-highest Semantic Consistency (SC) score of 7.58 and the highest Perceptual Quality (PQ) score of 7.94. This leads to a strong overall score of 7.21, placing it among the top-tier models. This score outperforms Gemini-2.5-Flash-Image and is second only to Qwen-Image-Edit among open-source models. As detailed in Table 2, OmniGen2 demonstrates compelling performance on the comprehensive ImgEdit-Bench, notably surpassing some strong open-source models like BAGEL. More details are provided in Appendix 8.2.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">SINGLE</td><td colspan="3">MULTIPLE</td><td colspan="3">SCENE</td><td rowspan="2">Average <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Character</td><td>Object</td><td>Character</td><td>Object</td><td>Char. + Obj.</td><td>Character</td><td>Object</td><td>Char. + Obj.</td></tr><tr><td>Flux.1 Kontext max <sup><a href="#fn:40">40</a></sup></td><td>8.48</td><td>8.68</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gemini-2.0-Flash-Image <sup><a href="#fn:28">28</a></sup></td><td>5.06</td><td>5.17</td><td>2.91</td><td>2.16</td><td>3.80</td><td>3.02</td><td>3.89</td><td>2.92</td><td>3.62</td></tr><tr><td>Gemini-2.5-Flash-Image <sup><a href="#fn:29">29</a></sup></td><td>8.62</td><td>8.91</td><td>7.88</td><td>8.92</td><td>7.39</td><td>7.29</td><td>7.05</td><td>6.68</td><td>7.84</td></tr><tr><td>GPT-4o <sup><a href="#fn:64">64</a></sup></td><td>8.90</td><td>9.01</td><td>9.07</td><td>8.95</td><td>8.54</td><td>8.90</td><td>8.44</td><td>8.60</td><td>8.80</td></tr><tr><td>InfiniteYou <sup><a href="#fn:37">37</a></sup></td><td>6.05</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UNO <sup><a href="#fn:97">97</a></sup></td><td>6.60</td><td>6.83</td><td>2.54</td><td>6.51</td><td>4.39</td><td>2.06</td><td>4.33</td><td>4.37</td><td>4.71</td></tr><tr><td>BAGEL <sup><a href="#fn:18">18</a></sup></td><td>5.48</td><td>7.03</td><td>5.17</td><td>6.64</td><td>6.24</td><td>4.07</td><td>5.71</td><td>5.47</td><td>5.73</td></tr><tr><td>Qwen-Image-Edit-2509 <sup><a href="#fn:94">94</a></sup></td><td>8.35</td><td>9.13</td><td>7.65</td><td>8.85</td><td>7.90</td><td>5.16</td><td>7.75</td><td>6.73</td><td>7.69</td></tr><tr><td>OmniGen <sup><a href="#fn:98">98</a></sup></td><td>7.21</td><td>5.71</td><td>5.65</td><td>5.44</td><td>4.68</td><td>3.59</td><td>4.32</td><td>5.12</td><td>4.34</td></tr><tr><td>myblueOmniGen2</td><td>8.19</td><td>8.63</td><td>7.45</td><td>7.91</td><td>7.93</td><td>7.75</td><td>7.91</td><td>7.93</td><td>7.95</td></tr></tbody></table>

Table 4: Overall comparison of existing models on our proposed OmniContext benchmark. ”Char. + Obj.” indicates Character + Object.

### 5.4 In-context Generation

A distinguishing feature of OmniGen2 is its capacity to perform in-context generation. We introduce the OmniContext benchmark to provide a comprehensive evaluation of the performance of the existing model in this domain. OmniContext comprises eight subtasks, with overall scores for each subtask presented in Table 4. As the inaugural model evaluated on this benchmark, OmniGen2 establishes a strong baseline, achieving an overall score of 7.95, which surpass the powerful open-sourced model Qwen-Image-Edit-2509. These results show OmniGen2’s proficiency in disentangling the subject’s identity from its original background and re-rendering it accurately according to new textual instructions. OmniGen2 exhibits significant improvements over competing models in all types of tasks, demonstrating superior prompt-following ability and subject consistency. Among closed-source models, GPT-4o [^64] achieves the highest scores in the Overall metrics. More details are provided in Appendix 8.3, 9.8.

### 5.5 Ablation Study

Our ablation study, detailed in Table 5, validates our principled curriculum by demonstrating the critical importance of two key factors: the selection of tasks and reward signals, and the scheduling of the training sequence. For task selection, we highlight four crucial findings: (1) Tasks with limited skill overlap can cause negative interference, as shown by OCR only training which degrades the GEdit Overall score from 6.28 to 6.13. (2) Conversely, well-chosen tasks with skill overlap such as instruction following exhibit strong synergy; the Edit & GenEval strategy surpasses both single-task baselines on their respective metrics (GenEval: 0.95 vs. 0.94; GEdit Overall: 7.19 vs. 7.01). (3) Reward signals about human preference pose significant risks, with Edit & HPSv3 confirming reward hacking by inflating the PQ score to 8.22 at the severe cost of collapsing SC and IC scores. (4) accuracy reward signal is vital. As shown by Edit only, whose IC score is higher than IC only (7.71 vs. 7.38) because of excel performance of EditScore [^57] to enhance instruction following. Beyond selection, the training sequence is equally vital. This is confirmed by comparing our final curriculum (Edit & Geneval & IC) against an alternative ordering (Edit & IC & Geneval), which results in a marked performance drop (GEdit Overall: 7.21 vs. 7.06). We also observe that prioritizing editing tasks leads to consistently better performance than T2I-first. We hypothesize this is because editing tasks with richer supervision build a robust foundation for subsequent learning. And Additional results on out-of-distribution (OOD) benchmarks are provided in Appendix 9.10, showing consistent improvements under our RL curriculum. These findings collectively prove that both careful selection and scheduling are essential to our principled alignment strategy.

<table><tbody><tr><th rowspan="2">Strategy</th><td rowspan="2">GenEval <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td rowspan="2">OmniContext <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td colspan="3">GEdit <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>SC</td><td>PQ</td><td>Overall</td></tr><tr><th>Base (w/o RL)</th><td>0.78</td><td>7.18</td><td>6.72</td><td>7.20</td><td>6.28</td></tr><tr><th colspan="6">Single-Task</th></tr><tr><th>Edit only</th><td>0.79</td><td>7.71</td><td>7.30</td><td>7.95</td><td>7.01</td></tr><tr><th>GenEval only</th><td>0.94</td><td>7.24</td><td>6.78</td><td>7.20</td><td>6.30</td></tr><tr><th>OCR only</th><td>0.78</td><td>7.33</td><td>6.65</td><td>7.15</td><td>6.13</td></tr><tr><th>IC only</th><td>0.78</td><td>7.38</td><td>6.97</td><td>6.98</td><td>6.39</td></tr><tr><th colspan="6">Multi-Tasks</th></tr><tr><th>Edit & GenEval</th><td>0.95</td><td>7.68</td><td>7.52</td><td>7.95</td><td>7.19</td></tr><tr><th>Edit & OCR</th><td>0.81</td><td>7.70</td><td>7.28</td><td>7.96</td><td>7.06</td></tr><tr><th>Edit & HPSv3</th><td>0.77</td><td>6.82</td><td>6.87</td><td>8.22</td><td>6.88</td></tr><tr><th>Edit & IC & GenEval</th><td>0.93</td><td>7.65</td><td>7.33</td><td>7.92</td><td>7.06</td></tr><tr><th>Geneval & Edit & IC</th><td>0.94</td><td>7.80</td><td>7.49</td><td>7.97</td><td>7.21</td></tr><tr><th>myblueEdit & GenEval & IC (Ours)</th><td>0.95</td><td>7.95</td><td>7.58</td><td>7.94</td><td>7.21</td></tr></tbody></table>

Table 5: Ablation study of multi-task reinforcement learning strategies. T2I, Edit, and IC tasks are trained for 1500, 700, and 200 steps, respectively.

## 6 Related Works

### 6.1 Multimodal Generation

Recent advances in multimodal generation have produced models capable of both understanding and generating content across text, images, and video. Diffusion-based models, including the Stable Diffusion series [^74] [^67] [^22], DALL·E [^72], and Imagen [^35], have achieved high-fidelity image synthesis, while methods like ControlNet [^109] and T2I-Adapter [^61] improve controllability, and StyleShot, InstructPix2Pix, and EMU-Edit [^24] [^5] [^81] support fine-grained, instruction-guided editing. Unified image generation models such as OmniGen [^98], UniReal [^13], and related works [^98] [^59] [^13] [^88] extend this further, integrating multiple tasks into a single model. Building on this foundation, autoregressive multimodal models provide an alternative paradigm for unified generation [^87] [^85] [^91]. There are also hybrid approaches such as Show-o and Transfusion [^99] [^112] [^31] [^20] [^82] [^66] combining autoregressive text generation with diffusion-based image modeling. Several works focus on adapting large language models for multimodal generation [^10] [^46] [^18] [^89] [^95]: These works are trained with vast amount of data, obtaining powerful image understanding and image generation capabilities.

### 6.2 Reinforcement Learning in Diffusion Model

Reinforcement learning has increasingly been adopted to improve alignment in diffusion and flow-based models. For text-to-image (T2I) generation, early works such as DDPO and DPOK [^4] [^23] formulated diffusion sampling as a sequential decision process and optimized via KL-regularized policy updates. Follow-up approaches including ReFL, AlignProp [^8] [^68] refined this paradigm with more stable reward optimization, improved credit assignment across denoising steps, and scalable preference-learning from human or synthetic feedback. More recently GRPO [^79] has become prominent due to its training stability and efficiency. GRPO-based alignment method like DanceGRPO [^100], Flow-GRPO [^52], and Mix-GRPO [^42] further push the boundaries of alignment technology, outperforming traditional methods in both accuracy and scalability [^32] [^92] [^45]. For image editing or in-context generation, RL has also been used to enforce text alignment and editing faithfulness to ensure consistency between input and output [^33] [^60] [^57] [^15]. Despite the rapid progress, most existing approaches optimize RL for a single task or a narrow alignment objective. In contrast, our work introduces a multi-task RL pipeline that jointly aligns the model’s behavior across all three critical scenarios, achieving comprehensive, all-around alignment.

## 7 Conclusion

In this work, we present OmniGen2, a generative model that is systematically instruction aligned. OmniGen2 explores two directions to enhance alignment performance: constructing a robust and flexible base model, and developing a multi-task RL alignment scheme. OmniGen2 utilizes a simple, efficient and flexible architecture to support diverse multimodal generation tasks. Our experiments across standard benchmarks and our propose novel OmniContext benchmark demonstrate OmniGen2’s semantic consistency, versatile capabilities, and superior generation quality. Instruction alignment has consistently and significantly enhanced the base model across various tasks. These results suggest that instruction alignment may represent a crucial step toward realizing general multimodal systems.

## References

Supplementary Material

## 8 More Qualitative Results

In this section, we present more qualitative results of OmniGen2. OmniGen2 demonstrates unified and versatile capabilities across a wide range of tasks, including text-to-image generation, image editing, in-context generation, and others. Moreover, it exhibits strong generalization abilities, allowing it to effectively tackle complex generation scenarios with remarkable consistency and fidelity.

### 8.1 Text to Image

Figure 5 showcases OmniGen2’s robust capabilities in text-to-image (T2I) synthesis. The model demonstrates high fidelity across a wide spectrum of conceptual and thematic prompts, from fantasy scenes like a celestial staircase to photorealistic portraits and dynamic actions. OmniGen2 excels in rendering intricate details, such as the water droplets on the blue rose, and displays a sophisticated understanding of complex lighting and composition, as seen in the dramatic glow of the girl wielding lightning and the serene ambiance of the underwater scene. Crucially, these examples also highlight the model’s native support for arbitrary aspect ratios, generating high-quality portrait, landscape, and square images without distortion. This combination of conceptual diversity, high fidelity, and flexible aspect ratio support validates OmniGen2 as a powerful and versatile T2I generator.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x5.png)

Figure 5: Qualitative text-to-image generation by OmniGen2. Examples showcasing the model’s high fidelity to various text prompts and its support for diverse aspect ratios.

### 8.2 Image Editing

Figure 6 demonstrates OmniGen2’s comprehensive suite of image editing capabilities, showcasing its ability to interpret a wide range of user instructions with high fidelity. The model adeptly handles localized object manipulations, including precisely adding (a hat), removing (a cat), replacing (a sword with a hammer), and extracting subjects from their backgrounds. Beyond object-level changes, OmniGen2 excels at nuanced semantic alterations, such as modifying facial expressions (adding a smile) and character motion (changing a pose to waving). Furthermore, the model is capable of executing complex, global modifications that affect the entire image, from changing backgrounds to performing complete stylistic transformations (converting a photograph into a 3D figurine). A key strength observed across all examples is the model’s ability to preserve the identity of the subject and the integrity of unmodified regions, ensuring coherent and believable results.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x6.png)

Figure 6: Versatile image editing with OmniGen2. The model skillfully handles a wide variety of instructions, from simple object modifications to complex motion change and stylistic alterations.

### 8.3 In context generation

Figure 7 showcases OmniGen2’s advanced capabilities in in-context generation and editing, a challenging task requiring the model to comprehend and manipulate subjects provided in reference images. The model adeptly performs compositional tasks, such as seamlessly integrating a subject into a new environment (OBJECT + SCENE, PERSON + SCENE) or combining multiple distinct subjects into a coherent new image (ANIME + ANIME). In these examples, OmniGen2 successfully preserves the high-fidelity identity of each subject while harmonizing them with the new context through appropriate adjustments in lighting, scale, and placement. Furthermore, the model handles complex in-context editing instructions. This includes replacing a subject within an existing scene (OBJECT REPLACE, PERSON REPLACE), where it not only swaps the main element but also intelligently adapts its appearance (e.g., color and accessories) to fit the new setting. These results demonstrate a sophisticated level of visual reasoning, where the model goes beyond simple generation to perform compositional and conditional editing based on multiple visual inputs.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x7.png)

Figure 7: Qualitative results of in-context generation and in-context edit.

### 8.4 Limitations

we also find several limitations of OmniGen2:

1) Performance Disparity Between English and Chinese Prompts. As shown in the first row of Figure 8, prompts in English generally yield better results than those in Chinese. For instance, when using a Chinese prompt, the generated image exhibits a minor inconsistency between input image and edited image.

2) Limited Generalization to Certain Instructions. The second row highlights OmniGen2’s difficulty in modifying human body shapes, likely due to the scarcity of real-world data capturing such variations.

3) Sensitivity to Input Image Quality. As illustrated in Figure 8, the quality of the generated output is highly sensitive to the quality of the input image. When we input a low-quality image (generated by adding noise to the raw image), the resulting images exhibit significant degradation, with details becoming notably blurred. Furthermore, downsampling the input image to a maximum dimension of 256 pixels leads to further loss of clarity and detail, and the model’s ability to accurately follow generation instructions is substantially reduced.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x8.png)

Figure 8: Visualization of OmniGen2’s Limitations. Line 1: The model performs poorly when processing Chinese prompts and low-quality images. Line 2: The model often struggles to modify human body shapes accurately. Line 3: The model is sensitive to ambiguous instructions involving multiple image sources.

4) Ambiguity in Multi-Image Inputs. The third row of Figure 8 demonstrates that the model’s performance improves when the prompt explicitly specifies the correspondence between objects and their source images (e.g., “the bird from image 1 and the desk from image 2”), indicating a sensitivity to ambiguous multi-source instructions.

5) In in-context generation tasks, the model occasionally fails to perfectly reproduce objects from the provided context. Increasing the guidance scale of image can partially alleviate this issue; however, it does not offer a complete solution. We hypothesize that significant improvements on such complex tasks may require further scaling of the model size.

## 9 Other Experimental Details

![Refer to caption](https://arxiv.org/html/2506.18871v4/x9.png)

Figure 9: Full loss curves for the Omni-RoPE toy reconstruction experiment. Omni-RoPE converges substantially faster than prior positional encoding schemes. The inset shows late-stage optimization, where adding image index embeddings yields the lowest and most stable final loss.

### 9.1 Toy Experiment Verification for Omni-RoPE

Figure 9 provides the full loss curves for the toy reconstruction experiment introduced in Section 3.3. The trends are consistent with the quantitative results in Table 1 of the main paper: both Omni-RoPE variants reduce the reconstruction loss much faster than the prior positional encoding designs. In particular, Qwen2-VL’s RoPE shows a noticeable optimization plateau before converging, while Lumina-Image-2.0’s design remains unstable for a substantially longer period and converges to a worse final solution.

The zoomed-in view further highlights the late-stage behavior. Omni-RoPE already achieves a low and stable loss floor, and adding the image index embedding mainly improves the final fidelity and variance in the last stage of training, yielding the best overall reconstruction quality. These results support our key design choice of disentangling image identity from local spatial coordinates: it preserves patch-wise correspondence across images while avoiding the optimization difficulty introduced by entangled global offsets.

### 9.2 Data Construction Pipeline

For multimodal understanding tasks, we utilize the dataset provided by LLaVA-OneVision [^41]. For T2I generation, our training corpus comprises approximately 140 million open-source images sourced from Recap-DataComp [^43], SAM-LLaVA [^11], ShareGPT4V [^12], LAION-Aesthetic [^77], ALLaVA-4V [^9], DOCCI [^63], DenseFusion [^44], JourneyDB [^84], and BLIP3-o [^10]. Furthermore, we incorporate 10 million proprietary images, for which we generate synthetic annotations using the Qwen2.5-VL-72B [^3]. For image editing tasks, we collect publicly available datasets, including SEED-Data-Edit [^25], UltraEdit [^111], OmniEdit [^93], PromptFix [^106], and ImgEdit [^103]. However, these open-source resources often suffer from suboptimal image quality, limited instruction accuracy, and insufficient task diversity. To overcome these constraints and better serve our research objectives, we have meticulously constructed a new comprehensive training dataset for this study. The subsequent sections provide a detailed account of our data construction pipeline.

### 9.3 In-Context Data

The in-context image generation task [^97] [^101] [^40] [^86] focuses on extracting a visual concept—such as a specific object, identity or individual—from input images and accurately reproducing it within newly generated images. This task, also known as subject-driven generation [^76], parallels in-context learning in large language models: the image generation model produces personalized outputs in real time based solely on the provided context, without the need for additional fine-tuning. While in-context image generation has been extensively explored due to its broad range of applications, the community still faces a notable shortage of high-quality datasets tailored to this task.

#### 9.3.1 In-Context Generation

![Refer to caption](https://arxiv.org/html/2506.18871v4/x10.png)

Figure 10: In-Context Generation Dataset Construction Pipeline. The final input images are outlined with a red border and the target image is marked by a blue boundary.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x11.png)

Figure 11: In-Context Editing Dataset Construction Pipeline. The final input and target images are outlined by red and blue consistent with Figure 10.

In-context generation tasks require modeling the diverse appearances of an object across different scenarios. To address this, we leverage video data, which inherently capture the same subjects under varying conditions across frames. This temporal diversity enables the construction of training pairs in which subjects remain semantically consistent but exhibit differences in pose, viewpoint, and illumination. As illustrated in Figure 10, our data pipeline begins by extracting keyframes from each video and designating a base frame. Using Qwen2.5-VL-7B-Instruct [^3], we identify the primary subjects within the base frame, capitalizing on the model’s vision-language capabilities to focus on semantically salient entities while filtering out irrelevant background objects. The subject bounding boxes are then obtained via Grounding DINO [^53], conditioned on the tags generated by the vision-language model. Subsequently, SAM2 [^73] is employed to segment and track identified subjects in subsequent frames, with the last valid frame containing all subjects selected to maximize appearance variation. To mitigate tracking errors—such as the inclusion of visually similar but incorrect objects—we introduce a VLM-based filtering step to ensure subject consistency. To further enhance visual diversity, FLUX.1-Fill-dev <sup>1</sup> is used to outpaint the subject with a novel background in the input frame. We apply DINO [^6] -based similarity filtering to discard samples where the subject’s appearance deviates significantly, and Qwen2.5-VL-7B-Instruct is leveraged to assess both the semantic quality and consistency of the generated samples. Additionally, Qwen2.5-VL-7B-Instruct is used to generate concise object descriptions and detailed captions for the base image, which are then integrated into natural language instructions. The final training triplet comprises the instruction, the repainted image as input, and the original image as output, providing semantically rich and visually diverse supervision for multi-subject generation tasks.

#### 9.3.2 In-Context Edit

We further extend the in-context generation paradigm to editing tasks, introducing a new task termed in-context editing, as illustrated in Figure 11. Here, the model extracts relevant elements from a context image and utilizes them to edit a target input image.

The data source for in-context editing mirrors that of in-context generation: two frames containing the same object are selected, with one serving as the context clip and the other as the target clip. Initially, object masks for both frames are obtained using SAM2 [^73]. For the context image, FLUX.1-Fill-dev is applied to generate a new background for the object via outpainting, encouraging the model to focus on object-specific features. Subsequently, FLUX.1-Fill-dev is used to inpaint the target clip, removing the object while preserving the original background to create the input clip. Finally, Qwen2.5-VL-72B-Instruct [^3] generates a natural language description of the transformation from the input clip to the target clip, which is combined with the object description from the context clip to produce comprehensive natural language instructions.

### 9.4 Image Editing Data

![Refer to caption](https://arxiv.org/html/2506.18871v4/x12.png)

Figure 12: Create image edit pairs from videos. We first filter out frames belonging to different scenes to ensure contextual consistency, and then remove frames that exhibit significant changes in viewpoint.

#### 9.4.1 Inpaint Data

Although most existing editing datasets are constructed through inpainting techniques, they suffer from two primary limitations: (1) substandard image quality, stemming from both inherent low resolution and post-processing degradation during inpainting. (2) Editing instructions are inaccurate: previous work predefines editing instructions and uses inpainting models to generate images based on these instructions, but inpainting models have poor instruction-following capabilities, causing a mismatch between editing instructions and original-inpainted image pairs.

In this work, we select a small set of high-quality images from text-to-image data as our data source, applying FLUX.1-Fill-dev for inpainting. We use the inpainted images as inputs and the original images as targets to ensure high-quality target images. Additionally, we do not input instructions to the inpainting model, allowing it to fill content randomly. After obtaining image pairs, we employ a MLLM to write editing instructions based on these pairs. We find that the latest MLLM(e.g., Qwen2.5-VL) excels at writing editing instructions for original-inpainted image pairs, resulting in a high-accuracy editing dataset.

#### 9.4.2 Video Data

Traditional inpainting methods are inherently limited in their capacity to construct diverse types of data, rendering them inadequate for tasks such as action modification, object movement, or expression changes. To address these limitations, we additionally extract editing pairs from video sources.

We show the pipeline in Figure 12. Image editing tasks typically require localized modifications while preserving the integrity of the surrounding context. To construct suitable image editing pairs from videos, it is essential to identify frame pairs that exhibit only local changes. We begin by segmenting videos into distinct scenes to avoid pairing frames across discontinuous contexts. Scene boundaries are detected by analyzing average RGB pixel intensities, while a rolling average of differences in the HSV color space enhances robustness to rapid motion. Within each identified scene, we extract multiple frame pairs and evaluate their differences using both DINOv2 [^65] and CLIP [^71]. Pairs exhibiting substantial differences—indicative of viewpoint changes—or negligible differences are filtered out.

Since camera viewpoints in videos often change even within a single scene, further refinement is necessary. Existing approaches, such as vision-language models, are computationally expensive and prone to inaccuracies, while methods based on color histograms or pixel-level similarity are either insensitive to spatial structure or overly susceptible to noise. To address these challenges, we divide each image into multiple blocks and compare the color histograms of corresponding blocks to assess their similarity, effectively reducing the impact of noise. The proportion of similar blocks is then computed to impose spatial constraints, serving as a reliable indicator of viewpoint consistency. This strategy efficiently filters out frame pairs with viewpoint changes while maintaining computational efficiency.

Finally, for each retained image pair with a consistent camera viewpoint, we employ Qwen2.5-VL-72B-Instruct [^3] to generate precise editing instructions, thereby facilitating the construction of high-quality image editing datasets.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x13.png)

Figure 13: Multimodal Reflection for image generation.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x14.png)

Figure 14: Example of generation with reflection using OmniGen2. Left and middle: Successful correction via one round of reflection. Right: an example of failed reflection, where the correct answer is incorrectly judged as wrong due to over-reflection.

### 9.5 Interleave Data

#### 9.5.1 Interleaved Frames

We initially segment videos based on detected scene transitions and extract key frames from each segment. Subsequently, we construct two types of video frame sequences, each comprising up to five frames: intra-scene interleaved sequence composed of frames within identical scene and inter-scene interleaved sequence composed of frames across different scenes. Following frame sequence extraction, we annotate each pair of consecutive frames with descriptive captions using an MLLM to describe changes in object actions and behaviors, variations in environment and background, and differences in object appearances. Given the substantial volume of required annotations, we employ Qwen2.5-VL-7B-Instruct for this process. Consequently, we obtain 0.8 million interleaved data samples from video sources, which serve to pretrain the model’s capacity for processing continuous multimodal sequences.

![Refer to caption](https://arxiv.org/html/2506.18871v4/x15.png)

Figure 15: An illustrative example of evaluating the output image in the OmniContext benchmark.

#### 9.5.2 Reflection Data

Inspired by previous advances in test-time scaling and self-reflection of large language models [^30] [^36] [^47], we further explore the integration of reflection capabilities into multimodal generation models and demonstrate how test-time scaling can enhance the quality of image generation. In this section, we focus on describing the construction of the reflection data for subsequent model fine-tuning. The reflection data comprise an interleaved sequence of text and images, beginning with a user instruction, followed by the multimodal model’s generated image, and step-by-step reflections on the previous generated outputs. Each reflection addresses two key aspects: 1) an analysis of the deficiencies or unmet requirements in relation to the original instruction, and 2) proposed solutions to address the previous image’s limitation.

To construct self-reflection data, we select a small subset from the training data (in the current experiment, we only use data from the text-to-image task) and generate images through the model. Subsequently, we use an MLLM to assess whether the generated images meet the instruction requirements. If the images fail to adequately follow instructions or exhibit other quality issues, the model identifies specific errors and suggests modifications. Initially, we experimented with the DSG [^16] evaluation framework to assess instruction-image alignment. However, this approach frequently led to hallucinations. Later, we discovered that powerful multimodal models could handle this task directly, so we employed Doubao-1.5-pro [^21] to output issues and modification suggestions. After obtaining the first round of reflections, we append the generated images and corresponding reflections to the original instructions and fine-tune the model on these data. Once training is complete, we continue inferring data (using the first round of reflection data) to obtain a second round of images and corresponding reflective data. This iterative process yields multiple rounds of self-reflection data.

There is currently limited research on employing reflection mechanisms to enhance image generation tasks within multimodal generative models. We hope that our present work will contribute to advancing the development of reasoning capabilities in the field of multimodal generation. After the model acquires initial reflective capabilities through training with the current data, online reinforcement learning algorithms can further enhance these capabilities, which we leave for future exploration.

### 9.6 Reflection Fine-Tuning

We fine-tune OmniGen2 on reflection dataset following the illustrated in Figure 13. The model’s enhanced reflection capabilities are demonstrated through the examples in Figure 14. In the successful cases, the model effectively reflects on the initial generated image, identifies its shortcomings, makes appropriate corrections and terminate the generation process at an appropriate point. However, it still faces challenges in reflection and correction. The model may over-reflect on simple instructions, generating unnecessary requirements or fails to revise the image. These issues arise from the limited perception of the 3B-scale MLLM and insufficient reflection data. In future work, we plan to scale up the model and employ reinforcement learning to improve reflection quality.

### 9.7 Training pipeline details

Detailed configurations for each stage are summarized in Table 6.

<table><tbody><tr><td>Stage</td><td>Resolution</td><td>Task Type</td><td>Steps</td></tr><tr><td rowspan="6">1. Pre-training</td><td rowspan="2">256 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 256</td><td>T2I-only</td><td>50k</td></tr><tr><td>Mixed-Task</td><td>50k</td></tr><tr><td rowspan="2">512 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 512</td><td>T2I-only</td><td>30k</td></tr><tr><td>Mixed-Task</td><td>30k</td></tr><tr><td rowspan="2">1024 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1024</td><td>T2I-only</td><td>50k</td></tr><tr><td>Mixed-Task</td><td>50k</td></tr><tr><td>2. SFT</td><td>1024 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1024</td><td>Mixed-Task</td><td>100k</td></tr><tr><td>3. RL Alignment</td><td>512 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 512</td><td>Mixed-Task</td><td>2.4k</td></tr></tbody></table>

Table 6: Details of the OmniGen2 staged training pipeline. The curriculum progresses from general pre-training to general instruction alignment, with increasing task complexity and resolution.

<table><tbody><tr><th rowspan="3">Method</th><td colspan="9">SINGLE <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="3">Character</td><td colspan="3">Object</td><td colspan="3">Average</td></tr><tr><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td></tr><tr><th>Flux.1 Kontext max <sup><a href="#fn:40">40</a></sup></th><td>7.98</td><td>9.24</td><td>8.48</td><td>8.78</td><td>8.76</td><td>8.68</td><td>8.38</td><td>9.00</td><td>8.58</td></tr><tr><th>Gemini-2.0-flash <sup><a href="#fn:28">28</a></sup></th><td>5.54</td><td>5.98</td><td>5.06</td><td>6.17</td><td>5.89</td><td>5.17</td><td>5.86</td><td>5.93</td><td>5.11</td></tr><tr><th>GPT-4o <sup><a href="#fn:64">64</a></sup></th><td>8.89</td><td>9.03</td><td>8.90</td><td>9.40</td><td>8.74</td><td>9.01</td><td>9.14</td><td>8.88</td><td>8.95</td></tr><tr><th>InfiniteYou <sup><a href="#fn:37">37</a></sup></th><td>7.81</td><td>5.15</td><td>6.05</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>UNO <sup><a href="#fn:97">97</a></sup></th><td>7.56</td><td>6.48</td><td>6.60</td><td>7.78</td><td>6.65</td><td>6.83</td><td>7.67</td><td>6.56</td><td>6.72</td></tr><tr><th>BAGEL <sup><a href="#fn:18">18</a></sup></th><td>7.72</td><td>4.86</td><td>5.48</td><td>8.56</td><td>6.06</td><td>7.03</td><td>8.14</td><td>5.46</td><td>6.25</td></tr><tr><th>OmniGen <sup><a href="#fn:98">98</a></sup></th><td>7.12</td><td>7.58</td><td>7.21</td><td>7.66</td><td>5.04</td><td>5.71</td><td>7.39</td><td>6.31</td><td>6.46</td></tr><tr><th>myblueOmniGen2</th><td>7.92</td><td>8.68</td><td>8.19</td><td>8.98</td><td>8.44</td><td>8.63</td><td>8.45</td><td>8.56</td><td>8.41</td></tr></tbody></table>

Table 7: Comparison on task type SINGLE from OmniContext. Prompt Following (PF), Subject Consistency (SC), and Overall scores are reported (higher is better, $\uparrow$).

<table><tbody><tr><th rowspan="3">Method</th><td colspan="12">MULTIPLE <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="3">Character</td><td colspan="3">Object</td><td colspan="3">Char. + Obj.</td><td colspan="3">Average</td></tr><tr><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td></tr><tr><th>Gemini-2.0-flash <sup><a href="#fn:28">28</a></sup></th><td>3.65</td><td>3.02</td><td>2.91</td><td>2.50</td><td>5.02</td><td>2.16</td><td>4.26</td><td>5.80</td><td>3.80</td><td>3.47</td><td>4.62</td><td>2.96</td></tr><tr><th>GPT-4o <sup><a href="#fn:64">64</a></sup></th><td>9.17</td><td>9.03</td><td>9.07</td><td>9.06</td><td>8.90</td><td>8.95</td><td>8.34</td><td>8.89</td><td>8.54</td><td>8.86</td><td>8.94</td><td>8.86</td></tr><tr><th>UNO <sup><a href="#fn:97">97</a></sup></th><td>3.88</td><td>2.38</td><td>2.54</td><td>7.46</td><td>5.86</td><td>6.51</td><td>5.10</td><td>4.10</td><td>4.39</td><td>5.48</td><td>4.11</td><td>4.48</td></tr><tr><th>BAGEL <sup><a href="#fn:18">18</a></sup></th><td>6.14</td><td>4.86</td><td>5.17</td><td>7.54</td><td>6.10</td><td>6.64</td><td>6.74</td><td>6.28</td><td>6.24</td><td>6.81</td><td>5.75</td><td>6.02</td></tr><tr><th>OmniGen <sup><a href="#fn:98">98</a></sup></th><td>5.92</td><td>6.18</td><td>5.65</td><td>5.60</td><td>5.46</td><td>5.44</td><td>4.64</td><td>4.96</td><td>4.68</td><td>5.39</td><td>5.53</td><td>5.26</td></tr><tr><th>myblueOmniGen2</th><td>7.30</td><td>8.10</td><td>7.45</td><td>7.98</td><td>7.74</td><td>7.80</td><td>7.60</td><td>8.34</td><td>7.93</td><td>7.63</td><td>8.06</td><td>7.73</td></tr></tbody></table>

Table 8: Comparison on task type MULTIPLE from OmniContext. Prompt Following (PF), Subject Consistency (SC), and Overall scores are reported (higher is better, $\uparrow$).

<table><tbody><tr><th rowspan="3">Method</th><td colspan="12">SCENE <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td colspan="3">Character</td><td colspan="3">Object</td><td colspan="3">Char. + Obj.</td><td colspan="3">Average</td></tr><tr><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td><td>PF</td><td>SC</td><td>Overall</td></tr><tr><th>Gemini-2.0-flash <sup><a href="#fn:28">28</a></sup></th><td>3.76</td><td>3.33</td><td>3.02</td><td>4.02</td><td>5.22</td><td>3.89</td><td>2.89</td><td>4.63</td><td>2.92</td><td>3.56</td><td>4.39</td><td>3.28</td></tr><tr><th>GPT-4o <sup><a href="#fn:64">64</a></sup></th><td>9.05</td><td>8.88</td><td>8.90</td><td>8.33</td><td>8.62</td><td>8.44</td><td>8.71</td><td>8.57</td><td>8.60</td><td>8.70</td><td>8.69</td><td>8.65</td></tr><tr><th>UNO <sup><a href="#fn:97">97</a></sup></th><td>2.74</td><td>2.50</td><td>2.06</td><td>5.62</td><td>3.52</td><td>4.33</td><td>5.22</td><td>3.86</td><td>4.37</td><td>4.53</td><td>3.29</td><td>3.59</td></tr><tr><th>BAGEL <sup><a href="#fn:18">18</a></sup></th><td>4.56</td><td>3.94</td><td>4.07</td><td>6.12</td><td>5.50</td><td>5.71</td><td>5.90</td><td>5.30</td><td>5.47</td><td>5.53</td><td>4.91</td><td>5.08</td></tr><tr><th>OmniGen <sup><a href="#fn:98">98</a></sup></th><td>4.14</td><td>3.42</td><td>3.59</td><td>5.24</td><td>3.72</td><td>4.32</td><td>5.56</td><td>4.84</td><td>5.12</td><td>4.98</td><td>3.99</td><td>4.34</td></tr><tr><th>myblueOmniGen2</th><td>8.02</td><td>7.64</td><td>7.75</td><td>8.10</td><td>7.80</td><td>7.91</td><td>8.08</td><td>7.88</td><td>7.93</td><td>8.07</td><td>7.77</td><td>7.86</td></tr></tbody></table>

Table 9: Comparison on task type SCENE from OmniContext. Prompt Following (PF), Subject Consistency (SC), and Overall scores are reported (higher is better, $\uparrow$).

### 9.8 OmniContext Details

The detailed metrics for each subtask are presented in Tables 7, 8 and 9. The pipeline to evaluate models on OmniContext as shown in Figure 15.

### 9.9 RL Generalization to Out-of-Distribution Benchmarks

To further evaluate the generalization ability of our multi-stage RL curriculum, we conduct experiments on out-of-distribution (OOD) benchmarks that are not directly aligned with our training rewards.

Specifically, we evaluate on Emu-Edit and OneIG-Bench, which assess editing fidelity and general image generation quality under diverse and challenging conditions. These benchmarks differ from our training objectives and thus provide a reliable measure of transferability.

As shown in Table 10, our full curriculum (Edit → GenEval → IC) consistently outperforms the base model and alternative training orders across all OOD metrics. This demonstrates that our alignment strategy does not overfit to specific reward signals, but instead learns generalizable capabilities that transfer across tasks.

<table><thead><tr><th rowspan="2">Strategy</th><th colspan="3">Emu-Edit</th><th>OneIG</th></tr><tr><th>CLIP-I <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>CLIP-Out <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>DINO <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th><th>Align.<math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>Base</th><td>0.877</td><td>0.309</td><td>0.823</td><td>0.7870</td></tr><tr><th>GenEval+Edit</th><td>0.909</td><td>0.311</td><td>0.894</td><td>0.8160</td></tr><tr><th>GenEval+Edit+IC</th><td>0.886</td><td>0.312</td><td>0.858</td><td>0.8212</td></tr><tr><th>Edit+IC+GenEval</th><td>0.868</td><td>0.310</td><td>0.826</td><td>0.8242</td></tr><tr><th>Edit+GenEval+IC</th><td>0.896</td><td>0.311</td><td>0.876</td><td>0.8289</td></tr></tbody></table>

Table 10: RL curriculum ablation on out-of-distribution (OOD) benchmarks. Base denotes the model without RL.

### 9.10 GenEval Results

As shown in the Table 11, OmniGen2 excels at generating images from complex, compositional prompts. Our model achieves an impressive overall score of 0.95. This result surpasses other powerful unified models like UniWorld-V1 (0.84) and BAGEL (0.88). It is crucial to note that this SOTA performance is achieved with exceptional efficiency. OmniGen2 utilizes only 4B trainable parameters and was trained on 15M T2I pairs and 50k prompts used in RL.

| Method | Single object $\uparrow$ | Two object $\uparrow$ | Counting $\uparrow$ | Colors $\uparrow$ | Position $\uparrow$ | Color attribution $\uparrow$ | Overall $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SDv2.1 [^75] | 0.98 | 0.51 | 0.44 | 0.85 | 0.07 | 0.17 | 0.50 |
| SDXL [^67] | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| IF-XL | 0.97 | 0.74 | 0.66 | 0.81 | 0.13 | 0.35 | 0.61 |
| LUMINA-Next [^113] | 0.92 | 0.46 | 0.48 | 0.70 | 0.09 | 0.13 | 0.46 |
| SD3-medium [^1] | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
| FLUX.1-dev [^39] | 0.99 | 0.81 | 0.79 | 0.74 | 0.20 | 0.47 | 0.67 |
| NOVA [^19] | 0.99 | 0.91 | 0.62 | 0.85 | 0.33 | 0.56 | 0.71 |
| OmniGen [^98] | 0.98 | 0.84 | 0.66 | 0.74 | 0.40 | 0.43 | 0.68 |
| TokenFlow-XL [^70] | 0.95 | 0.60 | 0.41 | 0.81 | 0.16 | 0.24 | 0.55 |
| Janus [^96] | 0.97 | 0.68 | 0.30 | 0.84 | 0.46 | 0.42 | 0.61 |
| Janus Pro [^14] | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| $\text{Emu3-Gen}^{\dagger}$ [^91] | 0.99 | 0.81 | 0.42 | 0.80 | 0.49 | 0.45 | 0.66 |
| Show-o [^99] | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
| $\text{MetaQuery-XL}^{\dagger}$ [^66] | \- | \- | \- | \- | \- | \- | 0.80 |
| $\text{BLIP3-o}^{\dagger}$ 4B [^10] | \- | \- | \- | \- | \- | \- | 0.81 |
| $\text{BLIP3-o}^{\dagger}$ 8B [^10] | \- | \- | \- | \- | \- | \- | 0.84 |
| BAGEL [^18] | 0.99 | 0.94 | 0.81 | 0.88 | 0.64 | 0.63 | 0.82 |
| $\text{BAGEL}^{\dagger}$ [^18] | 0.98 | 0.95 | 0.84 | 0.95 | 0.78 | 0.77 | 0.88 |
| UniWorld-V1 [^48] | 0.99 | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 | 0.80 |
| $\text{UniWorld-V1}^{\dagger}$ [^48] | 0.98 | 0.93 | 0.81 | 0.89 | 0.74 | 0.71 | 0.84 |
| myblueOmniGen2 | 0.99 | 1 | 0.93 | 0.91 | 1 | 0.86 | 0.95 |

Table 11: Evaluation of text-to-image generation ability on GenEval [^27] benchmark. $\dagger$ refers to the methods using LLM rewriter.

[^1]: S. AI (2024) SD3-medium. Note: [https://stability.ai/news/stable-diffusion-3-medium](https://stability.ai/news/stable-diffusion-3-medium) Cited by: Table 2, Table 11.

[^2]: M. S. Albergo and E. Vanden-Eijnden (2022) Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571. Cited by: §3.4.

[^3]: S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025) Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. Cited by: §2, §3.1, §3.5, §5.1, §9.2, §9.3.1, §9.3.2, §9.4.2.

[^4]: K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine (2024) Training diffusion models with reinforcement learning. External Links: 2305.13301, [Link](https://arxiv.org/abs/2305.13301) Cited by: §6.2.

[^5]: T. Brooks, A. Holynski, and A. A. Efros (2023) Instructpix2pix: learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402. Cited by: Table 2, Table 3, §6.1.

[^6]: M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin (2021) Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660. Cited by: Table 3, Table 3, §9.3.1.

[^7]: J. Chang, Y. Fang, P. Xing, S. Wu, W. Cheng, R. Wang, X. Zeng, G. Yu, and H. Chen (2025) OneIG-bench: omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977. Cited by: §5.2.

[^8]: C. Chen, A. Wang, H. Wu, L. Liao, W. Sun, Q. Yan, and W. Lin (2024) Enhancing diffusion models with text-encoder reinforcement learning. In European Conference on Computer Vision, pp. 182–198. Cited by: §6.2.

[^9]: G. H. Chen, S. Chen, R. Zhang, J. Chen, X. Wu, Z. Zhang, Z. Chen, J. Li, X. Wan, and B. Wang (2024) ALLaVA: harnessing gpt4v-synthesized data for a lite vision-language model. External Links: 2402.11684 Cited by: §2, §9.2.

[^10]: J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, et al. (2025) Blip3-o: a family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568. Cited by: §2, Table 2, §6.1, §9.2, Table 11, Table 11.

[^11]: J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu, et al. (2023) Pixart-alpha: fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426. Cited by: §2, §9.2.

[^12]: L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin (2023) Sharegpt4v: improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793. Cited by: §2, §9.2.

[^13]: X. Chen, Z. Zhang, H. Zhang, Y. Zhou, S. Y. Kim, Q. Liu, Y. Li, J. Zhang, N. Zhao, Y. Wang, et al. (2025) Unireal: universal image generation and editing via learning real-world dynamics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12501–12511. Cited by: §3.3, §6.1.

[^14]: X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan (2025) Janus-pro: unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811. Cited by: Table 2, Table 11.

[^15]: Y. Cheng, W. Wu, S. Wu, M. Huang, F. Ding, and Q. He (2025) UMO: scaling multi-identity consistency for image customization via matching reward. arXiv preprint arXiv:2509.06818. Cited by: Table 2, §6.2.

[^16]: J. Cho, Y. Hu, J. Baldridge, R. Garg, P. Anderson, R. Krishna, M. Bansal, J. Pont-Tuset, and S. Wang (2024) Davidsonian scene graph: improving reliability in fine-grained evaluation for text-to-image generation. In ICLR, Cited by: §9.5.2.

[^17]: T. Dao (2023) Flashattention-2: faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691. Cited by: §3.4.

[^18]: C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. (2025) Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683. Cited by: §1, §3.2, Table 2, Table 3, Table 4, §6.1, Table 11, Table 11, Table 7, Table 8, Table 9.

[^19]: H. Deng, T. Pan, H. Diao, Z. Luo, Y. Cui, H. Lu, S. Shan, Y. Qi, and X. Wang (2024) Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169. Cited by: Table 11.

[^20]: R. Dong, C. Han, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J. Sun, H. Zhou, H. Wei, et al. (2023) Dreamllm: synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499. Cited by: §6.1.

[^21]: Doubao (2025) Doubao-1.5-pro. Note: [https://seed.bytedance.com/zh/special/doubao\_1\_5\_pro](https://seed.bytedance.com/zh/special/doubao_1_5_pro) Cited by: §9.5.2.

[^22]: P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. (2024) Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, Cited by: §6.1.

[^23]: Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee (2023) Dpok: reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems 36, pp. 79858–79885. Cited by: §6.2.

[^24]: J. Gao, Y. Liu, Y. Sun, Y. Tang, Y. Zeng, K. Chen, and C. Zhao (2024) StyleShot: a snapshot on any style. arXiv preprint arXiv:2407.01414. Cited by: §6.1.

[^25]: Y. Ge, S. Zhao, C. Li, Y. Ge, and Y. Shan (2024) SEED-data-edit technical report: a hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007. Cited by: §2, §9.2.

[^26]: D. Ghosh, H. Hajishirzi, and L. Schmidt (2023) Geneval: an object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36, pp. 52132–52152. Cited by: §5.2.

[^27]: D. Ghosh, H. Hajishirzi, and L. Schmidt (2024) Geneval: an object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36. Cited by: Table 11, Table 11.

[^28]: Google (2025) Gemini 2.0 flash. Note: [https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation) Cited by: Table 3, Table 4, Table 7, Table 8, Table 9.

[^29]: Google (2025) Introducing gemini 2.5 flash image. Note: [https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/](https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/) Accessed: 2025-09-18 Cited by: §1, Table 2, Table 3, Table 4.

[^30]: D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. (2025) DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645 (8081), pp. 633–638. Cited by: §9.5.2.

[^31]: A. Gupta, L. Fan, S. Ganguli, and L. Fei-Fei (2022) Metamorph: learning universal controllers with transformers. arXiv preprint arXiv:2203.11931. Cited by: §6.1.

[^32]: X. He, S. Fu, Y. Zhao, W. Li, J. Yang, D. Yin, F. Rao, and B. Zhang (2025) Tempflow-grpo: when timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324. Cited by: §6.2.

[^33]: Z. Huang, Y. Shu, H. Fang, Q. Long, W. Wang, Q. Guo, T. Ge, and L. Gan (2025) From competition to synergy: unlocking reinforcement learning for subject-driven image generation. arXiv preprint arXiv:2510.18263. Cited by: §6.2.

[^34]: A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. (2024) Gpt-4o system card. arXiv preprint arXiv:2410.21276. Cited by: §1.

[^35]: Imagen-Team-Google (2024) Imagen 3. External Links: 2408.07009, [Link](https://arxiv.org/abs/2408.07009) Cited by: §6.1.

[^36]: A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. (2024) Openai o1 system card. arXiv preprint arXiv:2412.16720. Cited by: §9.5.2.

[^37]: L. Jiang, Q. Yan, Y. Jia, Z. Liu, H. Kang, and X. Lu (2025) InfiniteYou: flexible photo recrafting while preserving your identity. arXiv preprint arXiv:2503.16418. Cited by: Table 2, Table 4, Table 7.

[^38]: M. Ku, D. Jiang, C. Wei, X. Yue, and W. Chen (2023) VIEScore: towards explainable metrics for conditional image synthesis evaluation. External Links: 2312.14867 Cited by: §4.

[^39]: B. F. Labs (2024) FLUX. Note: [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux) Cited by: §3.1, §3.2, Table 2, Table 11.

[^40]: B. F. Labs (2025) FLUX.1 kontext: flow matching for in-context image generation and editing in latent space.. Cited by: §1, §3.2, Table 4, §9.3, Table 7.

[^41]: B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, et al. (2024) Llava-onevision: easy visual task transfer. arXiv preprint arXiv:2408.03326. Cited by: §2, §9.2.

[^42]: J. Li, Y. Cui, T. Huang, Y. Ma, C. Fan, M. Yang, and Z. Zhong (2025) MixGRPO: unlocking flow-based grpo efficiency with mixed ode-sde. External Links: 2507.21802, [Link](https://arxiv.org/abs/2507.21802) Cited by: §6.2.

[^43]: X. Li, H. Tu, M. Hui, Z. Wang, B. Zhao, J. Xiao, S. Ren, J. Mei, Q. Liu, H. Zheng, Y. Zhou, and C. Xie (2024) What if we recaption billions of web images with llama-3?. arXiv preprint arXiv:2406.08478. Cited by: §2, §9.2.

[^44]: X. Li, F. Zhang, H. Diao, Y. Wang, X. Wang, and L. Duan (2024) DenseFusion-1m: merging vision experts for comprehensive multimodal perception. 2407.08303. Cited by: §2, §9.2.

[^45]: Y. Li, Y. Wang, Y. Zhu, Z. Zhao, M. Lu, Q. She, and S. Zhang (2025) Branchgrpo: stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040. Cited by: §6.2.

[^46]: C. Liao, L. Liu, X. Wang, Z. Luo, X. Zhang, W. Zhao, J. Wu, L. Li, Z. Tian, and W. Huang (2025) Mogao: an omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472. Cited by: §1, §6.1.

[^47]: H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2023) Let’s verify step by step. arXiv preprint arXiv:2305.20050. Cited by: §9.5.2.

[^48]: B. Lin, Z. Li, X. Cheng, Y. Niu, Y. Ye, X. He, S. Yuan, W. Yu, S. Wang, Y. Ge, et al. (2025) UniWorld: high-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147. Cited by: Table 2, Table 3, Table 11, Table 11.

[^49]: Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2022) Flow matching for generative modeling. arXiv preprint arXiv:2210.02747. Cited by: §3.4.

[^50]: H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee (2024-01) LLaVA-next: improved reasoning, ocr, and world knowledge. External Links: [Link](https://llava-vl.github.io/blog/2024-01-30-llava-next/) Cited by: Table 2.

[^51]: H. Liu, C. Li, Q. Wu, and Y. J. Lee (2024) Visual instruction tuning. Advances in neural information processing systems 36. Cited by: Table 2.

[^52]: J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang (2025) Flow-grpo: training flow matching models via online rl. arXiv preprint arXiv:2505.05470. Cited by: §3.5, §6.2.

[^53]: S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu, et al. (2023) Grounding dino: marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499. Cited by: §9.3.1.

[^54]: S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. (2025) Step1x-edit: a practical framework for general image editing. arXiv preprint arXiv:2504.17761. Cited by: §5.3, Table 2, Table 3, Table 3, Table 3.

[^55]: X. Liu, C. Gong, and Q. Liu (2022) Flow straight and fast: learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003. Cited by: §3.4, §3.5.

[^56]: Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. (2024) Mmbench: is your multi-modal model an all-around player?. In European conference on computer vision, pp. 216–233. Cited by: §5.1.

[^57]: X. Luo, J. Wang, C. Wu, S. Xiao, X. Jiang, D. Lian, J. Zhang, D. Liu, and Z. Liu (2025) EditScore: unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909. Cited by: §3.5, §3.5, §5.5, §6.2.

[^58]: Y. Ma, Y. Shui, X. Wu, K. Sun, and H. Li (2025) HPSv3: towards wide-spectrum human preference score. CoRR abs/2508.03789. External Links: [Link](https://doi.org/10.48550/arXiv.2508.03789), [Document](https://dx.doi.org/10.48550/ARXIV.2508.03789), 2508.03789 Cited by: §3.5.

[^59]: C. Mao, J. Zhang, Y. Pan, Z. Jiang, Z. Han, Y. Liu, and J. Zhou (2025) Ace++: instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487. Cited by: §6.1.

[^60]: Y. Miao, W. Loh, S. Kothawade, P. Poupart, A. Rashwan, and Y. Li (2024) Subject-driven text-to-image generation via preference-based reinforcement learning. Advances in Neural Information Processing Systems 37, pp. 123563–123591. Cited by: §6.2.

[^61]: C. Mou, X. Wang, L. Xie, Y. Wu, J. Zhang, Z. Qi, Y. Shan, and X. Qie (2023) T2I-adapter: learning adapters to dig out more controllable ability for text-to-image diffusion models. External Links: 2302.08453, [Link](https://arxiv.org/abs/2302.08453) Cited by: §6.1.

[^62]: C. Mou, Y. Wu, W. Wu, Z. Guo, P. Zhang, Y. Cheng, Y. Luo, F. Ding, S. Zhang, X. Li, et al. (2025) Dreamo: a unified framework for image customization. arXiv preprint arXiv:2504.16915. Cited by: Table 2.

[^63]: Y. Onoe, S. Rane, Z. Berger, Y. Bitton, J. Cho, R. Garg, A. Ku, Z. Parekh, J. Pont-Tuset, G. Tanzer, et al. (2024) DOCCI: descriptions of connected and contrasting images. arXiv preprint arXiv:2404.19753. Cited by: §2, §9.2.

[^64]: OpenAI (2025) GPT-4o. Note: [https://openai.com/index/introducing-4o-image-generation](https://openai.com/index/introducing-4o-image-generation) Cited by: §5.4, Table 3, Table 4, Table 7, Table 8, Table 9.

[^65]: M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. (2023) Dinov2: learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. Cited by: §9.4.2.

[^66]: X. Pan, S. N. Shukla, A. Singh, Z. Zhao, S. K. Mishra, J. Wang, Z. Xu, J. Chen, K. Li, F. Juefei-Xu, et al. (2025) Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256. Cited by: §1, §3.2, Table 2, §6.1, Table 11.

[^67]: D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach (2023) Sdxl: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952. Cited by: Table 2, §6.1, Table 11.

[^68]: M. Prabhudesai, A. Goyal, D. Pathak, and K. Fragkiadaki (2023) Aligning text-to-image diffusion models with reward backpropagation. Cited by: §6.2.

[^69]: Q. Qin, L. Zhuo, Y. Xin, R. Du, Z. Li, B. Fu, Y. Lu, J. Yuan, X. Li, D. Liu, et al. (2025) Lumina-image 2.0: a unified and efficient image generative framework. arXiv preprint arXiv:2503.21758. Cited by: §3.2.

[^70]: L. Qu, H. Zhang, Y. Liu, X. Wang, Y. Jiang, Y. Gao, H. Ye, D. K. Du, Z. Yuan, and X. Wu (2025) Tokenflow: unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2545–2555. Cited by: Table 11.

[^71]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021) Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. Cited by: Table 3, Table 3, §9.4.2.

[^72]: A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen (2022) Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1 (2), pp. 3. Cited by: §6.1.

[^73]: N. Ravi, V. Gabeur, Y. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al. (2024) Sam 2: segment anything in images and videos. arXiv preprint arXiv:2408.00714. Cited by: §9.3.1, §9.3.2.

[^74]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022) High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695. Cited by: §6.1.

[^75]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022) High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695. Cited by: Table 11.

[^76]: N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman (2023) Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510. Cited by: §4, §9.3.

[^77]: C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. (2022) Laion-5b: an open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35, pp. 25278–25294. Cited by: §2, §9.2.

[^78]: T. Seedream, Y. Chen, Y. Gao, L. Gong, M. Guo, Q. Guo, Z. Guo, X. Hou, W. Huang, Y. Huang, et al. (2025) Seedream 4.0: toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427. Cited by: §1.

[^79]: Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024) Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: §1, §6.2.

[^80]: S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman (2024) Emu edit: precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871–8879. Cited by: §5.3.

[^81]: S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman (2024) Emu edit: precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871–8879. Cited by: Table 3, Table 3, §6.1.

[^82]: W. Shi, X. Han, C. Zhou, W. Liang, X. V. Lin, L. Zettlemoyer, and L. Yu (2024) LlamaFusion: adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188. Cited by: §6.1.

[^83]: J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu (2024) Roformer: enhanced transformer with rotary position embedding. Neurocomputing 568, pp. 127063. Cited by: §3.3.

[^84]: K. Sun, J. Pan, Y. Ge, H. Li, H. Duan, X. Wu, R. Zhang, A. Zhou, Z. Qin, Y. Wang, et al. (2024) Journeydb: a benchmark for generative image understanding. Advances in Neural Information Processing Systems 36. Cited by: §2, §9.2.

[^85]: Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang (2024) Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14398–14409. Cited by: §6.1.

[^86]: Z. Tan, S. Liu, X. Yang, Q. Xue, and X. Wang (2024) Ominicontrol: minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098. Cited by: §9.3.

[^87]: C. Team (2024) Chameleon: mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818. Cited by: §6.1.

[^88]: X. Tian, W. Li, B. Xu, Y. Yuan, Y. Wang, and H. Shen (2025) Mige: a unified framework for multimodal instruction-based image generation and editing. arXiv preprint arXiv:2502.21291. Cited by: §6.1.

[^89]: G. Wang, S. Zhao, X. Zhang, L. Cao, P. Zhan, L. Duan, S. Lu, M. Fu, X. Chen, J. Zhao, Y. Li, and Q. Chen (2025) Ovis-u1 technical report. External Links: 2506.23044, [Link](https://arxiv.org/abs/2506.23044) Cited by: §6.1.

[^90]: P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. (2024) Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191. Cited by: §3.3.

[^91]: X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. (2024) Emu3: next-token prediction is all you need. arXiv preprint arXiv:2409.18869. Cited by: Table 2, §6.1, Table 11.

[^92]: Y. Wang, Z. Li, Y. Zang, Y. Zhou, J. Bu, C. Wang, Q. Lu, C. Jin, and J. Wang (2025) Pref-grpo: pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751. Cited by: §6.2.

[^93]: C. Wei, Z. Xiong, W. Ren, X. Du, G. Zhang, and W. Chen (2024) Omniedit: building image editing generalist models through specialist supervision. In The Thirteenth International Conference on Learning Representations, Cited by: §1, §2, §9.2.

[^94]: C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. Yin, S. Bai, X. Xu, Y. Chen, Y. Chen, Z. Tang, Z. Zhang, Z. Wang, A. Yang, B. Yu, C. Cheng, D. Liu, D. Li, H. Zhang, H. Meng, H. Wei, J. Ni, K. Chen, K. Cao, L. Peng, L. Qu, M. Wu, P. Wang, S. Yu, T. Wen, W. Feng, X. Xu, Y. Wang, Y. Zhang, Y. Zhu, Y. Wu, Y. Cai, and Z. Liu (2025) Qwen-image technical report. External Links: 2508.02324, [Link](https://arxiv.org/abs/2508.02324) Cited by: §1, Table 2, Table 2, Table 3, Table 4.

[^95]: C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. Yin, S. Bai, X. Xu, Y. Chen, et al. (2025) Qwen-image technical report. arXiv preprint arXiv:2508.02324. Cited by: §6.1.

[^96]: C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, et al. (2025) Janus: decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12966–12977. Cited by: Table 11.

[^97]: S. Wu, M. Huang, W. Wu, Y. Cheng, F. Ding, and Q. He (2025) Less-to-more generalization: unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160. Cited by: Table 2, Table 4, §9.3, Table 7, Table 8, Table 9.

[^98]: S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu (2025) Omnigen: unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13294–13304. Cited by: §1, Table 2, Table 3, Table 4, §6.1, Table 11, Table 7, Table 8, Table 9.

[^99]: J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou (2024) Show-o: one single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528. Cited by: Table 2, §6.1, Table 11.

[^100]: Z. Xue, J. Wu, Y. Gao, F. Kong, L. Zhu, M. Chen, Z. Liu, W. Liu, Q. Guo, W. Huang, et al. (2025) DanceGRPO: unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818. Cited by: §6.2.

[^101]: H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang (2023) Ip-adapter: text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721. Cited by: §9.3.

[^102]: J. Ye, D. Jiang, Z. Wang, L. Zhu, Z. Hu, Z. Huang, J. He, Z. Yan, J. Yu, H. Li, et al. (2025) Echo-4o: harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987. Cited by: §3.5.

[^103]: Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan (2025) Imgedit: a unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275. Cited by: §5.3, §9.2.

[^104]: Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang (2025) Anyedit: mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26125–26135. Cited by: Table 2, Table 3.

[^105]: W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang (2023) Mm-vet: evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490. Cited by: §5.1.

[^106]: Y. Yu, Z. Zeng, H. Hua, J. Fu, and J. Luo (2024) Promptfix: you prompt and we fix the photo. arXiv preprint arXiv:2405.16785. Cited by: §9.2.

[^107]: X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. (2024) Mmmu: a massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567. Cited by: §5.1.

[^108]: K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su (2024) Magicbrush: a manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36. Cited by: Table 2, Table 3.

[^109]: L. Zhang, A. Rao, and M. Agrawala (2023-10) Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3836–3847. Cited by: §6.1.

[^110]: Z. Zhang, J. Xie, Y. Lu, Z. Yang, and Y. Yang (2025) In-context edit: enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690. Cited by: Table 2, Table 3.

[^111]: H. Zhao, X. S. Ma, L. Chen, S. Si, R. Wu, K. An, P. Yu, M. Zhang, Q. Li, and B. Chang (2024) Ultraedit: instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems 37, pp. 3058–3093. Cited by: §9.2.

[^112]: C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy (2024) Transfusion: predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039. Cited by: §6.1.

[^113]: L. Zhuo, R. Du, H. Xiao, Y. Li, D. Liu, R. Huang, W. Liu, X. Zhu, F. Wang, Z. Ma, et al. (2024) Lumina-next: making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems 37, pp. 131278–131315. Cited by: Table 11.