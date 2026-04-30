---
title: "Step1X-Edit: A Practical Framework for General Image Editing"
source: "https://arxiv.org/html/2504.17761v5"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Step1X-Image Team  
StepFun  
[https://github.com/stepfun-ai/Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit)

###### Abstract

In recent years, image editing technology has witnessed remarkable and rapid development. The recent unveiling of cutting-edge multimodal models such as GPT-4o and Gemini2 Flash has presented highly promising image editing capabilities. These models demonstrate an impressive aptitude for fulfilling a vast majority of user-driven editing requirements, marking a significant advancement in the field of image manipulation. However, there is still a large gap between the open-source algorithm with these closed-source models. To this end, we introduce a state-of-the-art image editing model, Step1X-Edit, which aims to provide comparable performance against the closed-source models like GPT-4o and Gemini2 Flash. More specifically, we adopt the Multimodal LLM to process the reference image and the user’s editing instruction. A latent embedding has been extracted and integrated with a diffusion image decoder to obtain the target image. To train this model, we build a data generation pipeline covering 11 editing tasks to produce a high-quality dataset. For evaluation, we develop the GEdit-Bench, a novel benchmark rooted in real-world user instructions. Experimental results on GEdit-Bench demonstrate that Step1X-Edit outperforms existing open-source baselines by a substantial margin and approaches the performance of leading proprietary models, thereby making significant contributions to the field of image editing.

## 1 Introduction

Image editing with natural language instructions has become an increasingly important task in vision-language research. It offers intuitive interaction for end users while posing unique technical challenges: understanding nuanced semantics, precisely localizing regions to edit, and preserving image fidelity. While diffusion models [^42] [^5] [^7] [^11] [^41] have dramatically improved image generation quality, the existing design by integrating text encoder, e.g., CLIP [^44] and T5 [^45], with diffusion transformer often struggles with following editing instruction to maintain alignment between input image and edit instruction, especially when edit instructions are subtle or compositional.

Recent advances in proprietary multimodal foundation models, such as GPT-4o [^37], Gemini2 Flash [^15], and SeedEdit/Doubao [^50], have pushed the frontier of instruction-based image editing. These systems leverage large-scale vision-language modeling capabilities to perform high-fidelity edits across diverse scenarios. However, their closed nature limits reproducibility and transparency. In parallel, open-source efforts like OmniGen [^61] and ACE++ [^34] aim to replicate similar capabilities but still fall short in terms of overall generalization, edit accuracy, and the quality of generated images.

In this work, we aim to narrow the performance gap between open-source and closed-source editing systems, while also pushing the boundary of practical and user-grounded editing evaluation. Although researchers have open-sourced editing datasets like AnyEdit [^64] and OmniEdit [^59], we argue that the quality and diversity of these datasets are not good enough to obtain comparable performance against the close-source algorithms like GPT-4o. Thus, to target the image edit problem, we first try to build a large-scale high quality dataset for training. More specifically, we identify $11$ major editing task categories based on the commonly used editing instructions. Guided by this taxonomy, we develop a scalable and flexible data pipeline to generate over $1$ million high-quality training data. These image-instruction pairs encompass a broad spectrum of editing operations, including object manipulation, attribute modification, layout adjustment, and stylization, ensuring comprehensive coverage of real-world editing scenarios.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x1.png)

Figure 1: Overview of Step1X-Edit. Step1X-Edit is an open-source general editing model that achieves proprietary-level performance with comprehensive editing capabilities.

Building on this dataset, we propose, Step1X-Edit, a unified image editing model that combines the strong semantic reasoning of Multimedia Large Language Model (MLLM), e.g., Qwen-VL [^3], with a DiT-style diffusion architecture. The reference image and editing prompts can be processed by the MLLM to generate a target image latent condition which will be integrated with the diffusion model to obtain the output image. Our approach maintains a good balance between reference image reconstruction and editing prompt following. To train the model, we start from a text-to-image model to retain aesthetic quality and visual consistency, which can be easily replace by the existing text-to-image models like SD3 [^11], FLUX [^5] [^7] [^26], HiDream-I1 [^18] and Flex [^38]. To evaluate the existing editing models, we introduce a new benchmark named GEdit-Bench. By carefully collecting the images and editing prompts, GEdit-Bench ensures both real-world editing requirements and the diversity of the editing prompts. The experiments on GEdit-Bench validate that Step1X-Edit outperforms existing open-source baselines with a large-margin and approaches the performance of leading proprietary models, e.g., GPT-4o.

In summary, there will be three contributions of our work:

- We will open-source our Step1X-Edit model, to reduce the performance gap between open-source and closed-source image editing systems and boost further research in the field of image editing.
- A data generation pipeline is designed to produce high-quality image editing data. It ensures that the dataset is diverse, representative, and of sufficient quality to support the development of effective image editing models. The availability of such a pipeline provides a valuable resource for researchers and developers working on similar projects.
- A new benchmark, named GEdit-Bench, grounded in real-world usages is developed to support more authentic and comprehensive evaluation. This benchmark, which is carefully curated to reflect actual user editing needs and a wide range of editing scenarios, enables more authentic and comprehensive evaluations of image editing models.

## 2 Related Work

### 2.1 Controllable Image Generation and Edit

Autoregressive (AR) models have been actively studied for controllable image generation and editing by modeling images as sequences of discrete tokens. Works such as ControlAR [^29], ControlVAR [^27], and CAR [^63] incorporate spatial and pixel-level guidance—such as edges, segmentation masks, and depth maps—into the decoding process, enabling localized and structured control. Extensions like Training-Free VAR [^58], M2M [^48], and Instruct-imagen [^20] further improve editing flexibility and broaden application scenarios. UniFluid [^12] explores unified autoregressive generation and understanding with continuous visual tokens. However, due to reliance on discrete tokens and sequence length constraints, AR models often struggle to produce high-resolution and photorealistic results, especially in complex scenes.

Diffusion models have become the dominant approach for high-fidelity image synthesis, offering strong capabilities in photorealism, structural consistency, and diversity. Beginning with DDPM [^19] and DDIM [^51], and further advanced by Latent Diffusion [^47] [^42], diffusion models operate in latent spaces for improved scalability. With the introduction of DiT architectures [^41], diffusion models have made significant strides in generalization, image quality, and knowledge capacity, becoming the predominant architecture in modern image generation [^1] [^5] [^7]. Based on the above text2image models, ControlNet [^67], and T2I-Adapter [^36] inject spatial or task-specific control into the generation process. BrushNet [^23], PowerPaint [^73], and FLUX-Fill [^6] further improve inpainting quality and versatility. Despite these advances, diffusion models often rely on static prompts or fixed conditions and lack the capacity for multi-turn reasoning or flexible language alignment, limiting their application in open-ended editing scenarios.

These limitations have led to growing interest in unified image editing frameworks that combine the symbolic control of AR models with the generative fidelity of diffusion. Such models aim to tightly couple instruction understanding, spatial reasoning, and photorealistic synthesis within a single architecture, offering more flexible, general, and user-controllable editing capabilities.

### 2.2 Instruction-based Image Editing Models

Instruction-based image editing models aim to bridge semantic instruction understanding and precise visual manipulation. Early approaches such as InstructEdit [^55], InstructPix2Pix [^8], MagicBrush [^66], and BrushEdit [^28] adopt modular pipelines where MLLMs generate prompts, spatial cues, or synthetic instruction-image pairs to guide diffusion-based editing.

Recent works move toward tighter integration between instruction and generation. SmartEdit [^21], X2I [^33], RPG [^62], AnyEdit [^64], and UltraEdit [^71] enhance multimodal interaction and instruction fidelity through improved model architectures, task-aware routing, and fine-grained editing capabilities. Meanwhile, unified generation and editing frameworks such as OmniGen [^61], ACE [^16], ACE++ [^34], and Lumina-OmniLV [^43] consolidate diverse visual tasks under a single architecture. Methods like Qwen2VL-Flux [^31], DreamEngine [^9] and MetaQueries [^39] further explore efficient control integration and direct latent-level fusion between MLLMs and diffusion decoders. Moreover, Hidream-E1 [^17] incorporates instructions and edited image descriptions as inputs, enabling more detailed edit information. Additionally, some researchers focus on more efficient methods to ensure editing performance while reducing training costs. For example, ICEdit [^70] employs LoRA-MoE hybrid tuning and identifies better initial noise distributions, while SuperEdit [^35] uses higher-quality training data and introduces contrastive supervision signals.

More generally, models such as Gemini [^15] and GPT-4o [^37] demonstrate strong visual fluency through joint vision-language training, showing promising capabilities in understanding and generating consistent, context-aware images. Collectively, these developments reflect a shift from loosely coupled systems toward tightly integrated, instruction-driven editing frameworks.

However, existing approaches still face key limitations. Most methods are task-specific and lack general-purpose editability. They typically do not support incremental editing, fine-grained region correspondence, or instruction feedback refinement. Moreover, architectural coupling remains shallow in many designs, failing to unify instruction understanding and generation into a cohesive framework. These challenges motivate Step1X-Edit, which tightly integrates MLLM-based multimodal reasoning with diffusion-based controllable synthesis, enabling scalable, interactive, and instruction-faithful image editing across diverse editing goals.

## 3 Step1X-Edit

### 3.1 Data Creation

#### 3.1.1 Data Pipeline

![Refer to caption](https://arxiv.org/html/2504.17761v5/x2.png)

Figure 2: Data Volume Comparison.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x3.png)

Figure 3: Data Construction Pipeline and Sub-Task Distribution.

In the existing literature, current image editing datasets are constrained either by the scale or the quality of the collected data. To address this gap, this report endeavors to assemble a large-scale, high-quality dataset specifically tailored for image editing tasks.

We initiate the dataset collection process by web crawling a diverse set of image editing examples from the Internet. Through in-depth analysis of these examples, we systematically categorize the image editing problem into 11 distinct categories, which has been partly referenced by [^64] [^40]. These categories are designed to comprehensively encompass the vast majority of image editing requirements in practice. An overview of these 11 categories, along with the detailed data collection pipeline, is illustrated in Fig. 3.

To collect a large-scale high-quality triplets consisting of a source image, an editing instruction, and a target image, we designed a sophisticated data pipeline, which enabled us to generate over $20$ million instruction-images triplets. Following rigorous filtering using both Multimodal LLMs, e.g. step-1o [^52], and human annotators, we retained more than $1$ million high-quality triplets. In Fig. 2, we present a side-by-side comparison of all existing editing datasets [^13] [^14] [^74] [^59] [^65] [^53] [^22] [^66] [^2] [^64]. Our Step1X-Edit dataset surpasses all others in scale. Even after a rigorous filtering process (with a retention ratio of 20:1), the Step1X-Edit-HQ subset remains on par with other datasets in terms of absolute magnitude. The full data collection pipeline for each subtask is outlined below.

Subject Addition & Removal: For subject-add and subject-remove tasks, we begin by annotating our proprietary dataset using Florence-2 [^60], which supports diverse semantic granularities, spatial hierarchies, and annotation types such as object detection and classification. We then apply SAM-2 [^46] for segmentation and use ObjectRemovalAlpha [^30] to perform inpainting. Editing instructions are generated using a combination of Step-1o model [^52] and GPT-4o, followed by manual review to ensure data validity.

Subject Replacement & Background Change: This category shares similar preprocessing steps with subject-add/remove, including Florence-2 [^60] annotation and SAM-2 [^46] segmentation. However, for these tasks, we utilize Qwen2.5-VL [^3] and the Recognize-Anything Model [^69] to identify target objects or keywords, followed by Flux-Fill [^6] for content-aware inpainting. The instructions are automatically generated by Step-1o and the triplets are human-verified.

Color Alteration & Material Modification: After detecting objects in the image, we employ Zeodepth [^4] for depth estimation to understand object geometry. Based on the identified target transformation (e.g., change of color or material), we use ControlNet [^67] with diffusion model [^1] to generate new images that preserve object identity while altering appearance attributes such as texture or color.

Text Modification: For text-editing tasks, we differentiate between valid and invalid text edits. We use PPOCR [^10], which focuses on recognizing correct characters, alongside the Step-1o model to distinguish correct and incorrect regions of text. Based on this classification, we generate corresponding editing instructions. All outputs are finalized via human post-processing (e.g., manual retouching of text).

Motion Change: To handle motion-related transformations, we leverage videos from Koala-36M [^56], extracting frame pairs as input. We use BiRefNet [^72] and RAFT [^54] for foreground-background separation and optical flow estimation. Specifically, we compute the mean of the foreground flow norm and the norm of the background flow mean, ensuring robustness in selecting pairs where only the foreground exhibits motion. Finally, GPT-4o is used to annotate the change in motion between frames as editing instructions.

Portrait Editing and Beautification: Data are collected and created by two major sources: (a) Beautification pairs from public sources. Faces are detected and passed through Step-1o to assess layout and background consistency. (b) Beautification of the human editor, we invite the human editor to conduct beatification on collected data. All data are manually validated.

Style Transfer: We handle stylization in two directions depending on the target visual domain: For styles such as Ghibli, ink painting, or 3D anime style, generating photorealistic images from stylized inputs yields better alignment. We extract edges from stylized images and generate realistic outputs using controlled diffusion model [^67] [^1]. Conversely, for styles like oil painting or pixel art, we begin with realistic images and generate stylized outputs using the same edge-to-image pipeline.

Tone Transformation: This category focuses on global tonal adjustments, including color grading, dehazing, deraining, and seasonal transformations. These changes are largely driven by algorithmic tools and automated filters to simulate realistic environmental changes.

#### 3.1.2 Caption Strategy

To obtain high-quality and fine-grained editing instruction–image pairs, we adopt the following annotation strategies:

Redundancy-Enhanced Annotation: Given the well-known limitations of Vision-Language Models (VLMs)—such as vague background descriptions and susceptibility to hallucinations—we employ a multi-round annotation strategy. Specifically, the annotation results from a previous round are fed into the next round as contextual input. This recursive refinement strengthens semantic consistency across annotations and significantly mitigates hallucination-related issues. Deterministic information is reinforced through repeated confirmations, ensuring higher reliability of the final annotation.

Stylized Annotation via Contextual Examples: During the captioning process, we provide annotators (or models) with a large set of style-aligned examples as contextual references. These examples guide the tone, structure, and granularity of the captions, ensuring a consistent and stylized annotation format throughout the dataset.

Bilingual Annotation (Chinese-English): All our annotations are conducted bilingually, in both Chinese and English. This not only enhances accessibility and usability across different linguistic communities but also lays the groundwork for multilingual model training and evaluation.

### 3.2 Our Method

![Refer to caption](https://arxiv.org/html/2504.17761v5/x4.png)

Figure 4: Framework of Step1X-Edit. Step1X-Edit leverages the image understanding capabilities of MLLMs to parse editing instructions and generate editing tokens, which are then decoded into images using a DiT-based network.

As illustrated in Fig. 4, our algorithm involves three key components: a Multimedia Large Language Model (MLLM), a connector module, and a Diffusion in Transformer (DiT) [^41]. The input editing instruction, accompanied by the reference image, is first introduced to the MLLM; e.g., Qwen-VL [^3] (hereafter abbreviated as Qwen). In conjunction with a system prefix, these inputs are jointly processed through a single forward pass of the MLLM, enabling the model to capture the semantic relationships between the instruction and the visual content. To isolate and emphasize the semantic elements relevant to the editing task, we selectively discard the token embeddings associated with the prefix. This filtering process retains only the token embeddings that directly align with the edit information, ensuring that subsequent processing focuses precisely on the editing requirements.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x5.png)

Figure 5: DiT module details.

The extracted embeddings are then fed into a lightweight connector module, such as the token refiner [^32] [^24]. This module restructures the embeddings into a more compact multimodal feature representation, which is subsequently used as the multimodal embedding input to the downstream DiT network. Furthermore, we calculate the mean of valid embeddings from Qwen. This mean value is then projected through a linear layer, generating a global guidance vector. By doing so, the image editing network can leverage Qwen’s enhanced semantic comprehension capabilities, enabling more accurate and context-aware editing operations.

To effectively train the Token Refiner and enable rich cross-modal conditioning, we carefully design our feature aggregation strategy. Compared to approaches such as FLUX-Fill [^6], which uses channel concatenation, and methods like SeedEdit [^50], which introduce additional causal self-attention mechanisms, we follow OminiControl [^53] and adopt token concatenation to better balance responsiveness to editing instructions with the preservation of fine-grained image details. During training, the reference image is encoded by a VAE encoder, and its latent features are linearly projected into reference image tokens. As illustrated in Fig. 5, the image tokens (highlighted in the green box) are concatenated with noise image tokens along the token length dimension to construct the final visual input.

Recent research has also explored alternative approaches to enhance cross-model understanding by leveraging MLLMs. For instance, Qwen2VL-FLUX [^31] introduces a pioneering approach by replacing the traditional T5 [^45] text encoder in DiT-based text-to-image models with MLLMs to enhance multi-modal understanding and generation. However, it still retains T5 [^45] for text encoding, which limits its ability to perform comprehensive cross-modal reasoning. In contrast, approaches like DreamEngine [^9] leverage Qwen to align image and text modalities, with their features serving as external conditional inputs for SD3.5 [^1] in the image synthesis. This method establishes a shared representation space that facilitates a more coherent process of generation and understanding. Nevertheless, in DreamEngine, the challenge remains in fully capturing the fine-grained details of reference images using only MLLM features.

Compared to these approaches, our model not only retains cross-modal understanding but also enhances the extraction of image details. By combining structured visual-language guidance, detailed visual condition, and strong pretrained backbones within a unified framework, our method significantly boosts the system’s capability to perform high-fidelity, semantically aligned image edits across a diverse range of user instructions. During training, we jointly optimize the connector and the downstream DiT using only the diffusion loss, following the rectified flow [^11] formulation. Moreover, our method ensures stable training without relying on the mask loss trick, which distinguishes it from OmniGen [^61]. The learning rate is fixed at $1e^{-5}$ to ensure a good trade-off between training stability and convergence speed.

## 4 Benchmark and Evaluation

### 4.1 GEdit(Genuine Edit)-Bench

To evaluate the performance of the image editing models, we collect a new benchmark called GEdit(Genuine Edit)-Bench. The main motivation of the benchmark is to collect the real-world user editing instances in order to evaluate how the existing editing algorithms can be suffice for the practical editing instructions. More specifically, we collect more than 1K user editing instances from the Internet, e.g., reddit, and manually split these editing instructions into the $11$ categories. To keep the diversity of the benchmark, we filter those editing instructions with similar purpose. Finally, we obtain 606 testing examples whose reference images are from the real-world cases which make it more genuine for the applications. Based on GEdit-Bench, we evaluate the existing open-source image editing algorithms like ACE++ and AnyEdit, as well as the closed-source algorithms like GPT-4o and Gemini2 Flash.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x6.png)

Figure 6: De-Identification Process.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x7.png)

(a) VIEScore for the Intersection-subset.

| Benchmarks | Size | Real Image | Genuine Instruction | Human Filtering | #Sub-tasks | Public Availability |
| --- | --- | --- | --- | --- | --- | --- |
| EditBench [^57] | 240 | ✓ | ✗ | ✗ | 1 | ✓ |
| EmuEdit [^49] | 3,055 | ✓ | ✗ | ✗ | 7 | ✓ |
| HIVE [^68] | 1,000 | ✓ | ✗ | ✓ | 1 | ✓ |
| HQ-Eidt [^22] | 1,640 | ✗ | ✗ | ✗ | 7 | ✓ |
| MagicBrush [^66] | 1,053 | ✓ | ✗ | ✓ | 7 | ✓ |
| AnyEdit [^48] | 1,250 | ✓ | ✗ | ✗ | 25 | ✓ |
| ICE-Bench [^40] | 6,538 | ✓ | ✗ | ✓ | 31 | ✗ |
| GEdit-Bench(Ours) | 606 | ✓ | ✓ | ✓ | 11 | ✓ |

Table 1: Key Attributes of Open-source Edit Benchmarks. The reliance of existing open-source benchmarks on synthetic user inputs and minimal human involvement highlights the necessity of our proposed GEdit-Bench.

To safeguard privacy, a comprehensive de-identification protocol was meticulously implemented for all user-uploaded images prior to their utilization within the benchmarking framework as shown in Fig. 6. For each individual original image, a multi-faceted reverse image search strategy was employed, spanning across multiple public search engines. This process aimed to identify publicly accessible alternative images that demonstrated both visual similarity and semantic consistency with the original one, thereby aligning seamlessly with the corresponding editing instructions. In instances where public image alternatives could not be procured through this search methodology, a systematic approach to modifying the editing instructions was adopted. These modifications were carefully calibrated to maintain the highest degree of fidelity between the anonymized image-instruction example and the original user intents. This approach not only ensures the ethical integrity of the benchmark dataset but also preserves the essential characteristics required for accurate and meaningful evaluation of image editing models.

<table><tbody><tr><th rowspan="2">Model</th><td colspan="6">GEdit-Bench-EN (Intersection subset) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td colspan="6">GEdit-Bench-EN (Full set) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><td>G_SC</td><td>G_PQ</td><td>G_O</td><td>Q_SC</td><td>Q_PQ</td><td>Q_O</td><td>G_SC</td><td>G_PQ</td><td>G_O</td><td>Q_SC</td><td>Q_PQ</td><td>Q_O</td></tr><tr><th>Instruct-Pix2Pix <sup><a href="#fn:8">8</a></sup></th><td>3.335</td><td>6.210</td><td>3.234</td><td>4.833</td><td>6.992</td><td>4.691</td><td>3.296</td><td>6.189</td><td>3.219</td><td>4.746</td><td>6.913</td><td>4.578</td></tr><tr><th>MagicBrush <sup><a href="#fn:66">66</a></sup></th><td>4.564</td><td>6.335</td><td>4.236</td><td>5.814</td><td>7.149</td><td>5.653</td><td>4.517</td><td>6.371</td><td>4.185</td><td>5.752</td><td>7.069</td><td>5.558</td></tr><tr><th>AnyEdit <sup><a href="#fn:64">64</a></sup></th><td>3.122</td><td>5.865</td><td>2.919</td><td>3.873</td><td>6.754</td><td>3.789</td><td>3.053</td><td>5.882</td><td>2.854</td><td>3.713</td><td>6.730</td><td>3.635</td></tr><tr><th>OmniGen <sup><a href="#fn:61">61</a></sup></th><td>6.037</td><td>5.856</td><td>5.154</td><td>7.033</td><td>6.775</td><td>6.557</td><td>5.879</td><td>5.871</td><td>5.005</td><td>6.845</td><td>6.700</td><td>6.352</td></tr><tr><th>Step1X-Edit</th><td>7.289</td><td>6.962</td><td>6.618</td><td>7.501</td><td>7.264</td><td>7.189</td><td>7.131</td><td>6.998</td><td>6.444</td><td>7.388</td><td>7.279</td><td>7.067</td></tr><tr><th>Step1X-Edit-v1.1</th><td>7.905</td><td>7.366</td><td>7.189</td><td>7.737</td><td>7.425</td><td>7.436</td><td>7.658</td><td>7.354</td><td>6.969</td><td>7.652</td><td>7.408</td><td>7.346</td></tr><tr><th>Gemini <sup><a href="#fn:15">15</a></sup></th><td>6.816</td><td>7.408</td><td>6.483</td><td>7.295</td><td>7.314</td><td>6.996</td><td>6.866</td><td>7.436</td><td>6.509</td><td>7.274</td><td>7.327</td><td>6.971</td></tr><tr><th>Doubao <sup><a href="#fn:50">50</a></sup></th><td>7.396</td><td>7.899</td><td>7.137</td><td>7.427</td><td>7.651</td><td>7.285</td><td>7.222</td><td>7.885</td><td>6.983</td><td>7.353</td><td>7.651</td><td>7.230</td></tr><tr><th>GPT-4o <sup><a href="#fn:37">37</a></sup></th><td>7.867</td><td>8.097</td><td>7.590</td><td>7.905</td><td>7.723</td><td>7.752</td><td>7.743</td><td>8.133</td><td>7.494</td><td>7.847</td><td>7.705</td><td>7.692</td></tr></tbody></table>

Table 2: Quantitative evaluation on GEdit-Bench-EN. All metrics are reported as higher-is-better ($\uparrow$). The Intersection subset reflects the subset of prompts where all methods return valid responses with a total of 434 instances; the Full set includes all the 606 instances. G\_SC, G\_PQ, and G\_O refer to the metrics evaluated by GPT-4.1, while Q\_SC, Q\_PQ, and Q\_O refer to the metrics evaluated by Qwen2.5-VL-72B.

<table><tbody><tr><th rowspan="2">Model</th><td colspan="6">GEdit-Bench-CN (Intersection subset) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td colspan="6">GEdit-Bench-CN (Full set) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><td>G_SC</td><td>G_PQ</td><td>G_O</td><td>Q_SC</td><td>Q_PQ</td><td>Q_O</td><td>G_SC</td><td>G_PQ</td><td>G_O</td><td>Q_SC</td><td>Q_PQ</td><td>Q_O</td></tr><tr><th>Gemini <sup><a href="#fn:15">15</a></sup></th><td>5.316</td><td>7.571</td><td>5.208</td><td>5.658</td><td>7.372</td><td>5.566</td><td>5.259</td><td>7.600</td><td>5.143</td><td>5.622</td><td>7.370</td><td>5.525</td></tr><tr><th>Doubao <sup><a href="#fn:50">50</a></sup></th><td>7.228</td><td>7.800</td><td>6.915</td><td>7.109</td><td>7.687</td><td>7.054</td><td>7.168</td><td>7.794</td><td>6.837</td><td>7.098</td><td>7.676</td><td>7.040</td></tr><tr><th>GPT-4o <sup><a href="#fn:37">37</a></sup></th><td>7.606</td><td>7.991</td><td>7.336</td><td>7.772</td><td>7.658</td><td>7.599</td><td>7.518</td><td>8.023</td><td>7.301</td><td>7.726</td><td>7.652</td><td>7.552</td></tr><tr><th>Step1X-Edit</th><td>7.464</td><td>7.076</td><td>6.779</td><td>7.527</td><td>7.410</td><td>7.259</td><td>7.299</td><td>7.142</td><td>6.658</td><td>7.490</td><td>7.384</td><td>7.212</td></tr><tr><th>Step1X-Edit-v1.1</th><td>7.838</td><td>7.401</td><td>7.115</td><td>7.636</td><td>7.367</td><td>7.327</td><td>7.647</td><td>7.398</td><td>6.983</td><td>7.532</td><td>7.370</td><td>7.240</td></tr></tbody></table>

Table 3: Quantitative evaluation on GEdit-Bench-CN. All metrics are reported as higher-is-better ($\uparrow$). The Intersection subset reflects the subset of prompts where all methods return valid responses with a total of 422 instances; the Full set includes all the 606 instances. G\_SC, G\_PQ, and G\_O refer to the metrics evaluated by GPT-4.1, while Q\_SC, Q\_PQ, and Q\_O refer to the metrics evaluated by Qwen2.5-VL-72B.

### 4.2 Experimental Results

#### 4.2.1 Evaluation on GEdit-Bench

Based on the GEdit-Bench, we evaluated a diverse range of image editing algorithms, covering state-of-the-art open-source solutions such as Instruct-Pix2Pix [^8], MagicBrush [^66], AnyEdit [^64], OmniGen [^61], as well as proprietary algorithms like GPT-4.1 [^37] <sup>1</sup>, doubao [^50] <sup>2</sup>, and Gemini2 Flash [^15] <sup>3</sup>. Following VIEScore [^25], we adopt three metrics: SQ (Semantic Consistency), PQ (Perceptual Quality), and O (Overall Score). SQ assesses the degree to which the edited results conform to the given editing instruction, with a score ranging from $0$ to $10$. PQ evaluates the naturalness of the image and the presence of artifacts, also using a scoring scale that ranges from $0$ to $10$. The overall score is calculated based on these evaluations. To perform the automatic evaluation for VIEScore, we adopt the state-of-art MLLM model GPT-4o <sup>4</sup>. Also, the evaluation based on the open-source model Qwen2.5-VL-72B [^3] is included for reproduction. To comprehensively assess model capabilities on different languages, each image in our benchmark is paired with one English (EN) and one Chinese (CN) instruction. For EN instructions (GEdit-Bench-EN), both closed and open-source models are evaluated. For CN instructions (GEdit-Bench-CN), only those models which supports Chinese prompts, i.e., the close-source systems, are tested. During the evaluation process, we find that close-source image editing system such as GPT-4o may refuse certain instructions due to safety policies. To address this issue, we report two scores for two testing sets: (1) Intersection Subset — the subset of images whose results can be successfully returned from all the tested models, and (2) Full set — all the testing samples from GEdit-Bench. For the full set of results, we will calculate the average scores only for the cases where the models successfully generate and return the target image. For each evaluated model, instances where no result image is returned due to reasons such as safety concerns will be excluded from the averaging process.

Fig. 7 demonstrates the groundbreaking capacity of Step1X-Edit, which outperforms open-source counterparts across 11 distinct evaluation axes. When compared to closed models, it surpasses Gemini2 Flash [^15] and even beats GPT-4o [^37] in axes such as style change and color alteration. As detailed in Tab. 2, Step1X-Edit significantly outperforms the existing open-source models like OmniGen [^61] and has comparable results against the closed model like Gemini2 Flash and Doubao. Furthermore, as shown in Tab. 3, Step1X-Edit demonstrates consistent performance, even surpassing Gemini2 Flash and Doubao when handling the Chinese editing instructions in GEdit-Bench-CN benchmark. These results highlight the outstanding performance of our model across all dimensions with a unified architecture, eliminating the requirement for masks during the editing process. Fig. 8 and Fig. 9 provide illustrative examples for English instructions and Chinese instructions, respectively.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x11.png)

Figure 8: A Comparative Illustration of Open-Source Approaches and Commercial systems for English Editing Instructions.

![Refer to caption](https://arxiv.org/html/2504.17761v5/x12.png)

Figure 9: A Comparative Illustration of state-of-art algorithms for Chinese Editing Instructions.

#### 4.2.2 User Study

To assess the subjective quality of image editing results, we conduct a comprehensive user preference study built upon the GEdit-Bench. A total of 55 participants are recruited to evaluate the outputs of four algorithms—Gemini2 Flash [^15], Doubao [^50], GPT-4o [^37], and our method, Step1X-Edit. Each participant is presented with a series of test images and asked to rank the editing results generated by the four methods. This evaluation is performed in a blinded and subjective setting to minimize bias and ensure fairness.

Participants rate the outputs using a five-level quality scale, ranging from worst to excellent. To facilitate consistent comparison with quantitative evaluation metrics such as VIEScores, we map these qualitative ratings to numerical scores: worst = $2$, poor = $4$, fair = $6$, good = $8$, and excellent = $10$. For each editing task, we compute the mean preference score across all participants. The overall performance of each method is then summarized by averaging the scores over all editing tasks.

The results, presented in Tab. 4 and Fig. 10, highlight the effectiveness of Step1X-Edit. Notably, our method achieves comparable subjective quality to other state-of-the-art approaches, reinforcing its capability in producing visually pleasing and user-preferred edits. It is worth noting that Gemini2 Flash achieves an astonishingly high user preference score primarily attributed to its strong identity-preserving capabilities in the testing examples. This characteristic was more favored by the participants in the user study.

| Model | Gemini [^15] | Doubao [^50] | GPT-4o [^37] | Step1X-Edit |
| --- | --- | --- | --- | --- |
| UP-IS ($\uparrow$) | 7.109 | 6.320 | 6.961 | 6.544 |
| UP-Full ($\uparrow$) | 6.603 | 5.678 | 7.134 | 6.939 |

Table 4: Overall user preference (UP) evaluation on GEdit-Bench. UP-IS and UP-Full represent user preference score for Intersection subset (IS) and Full set (Full), respectively. All metrics are reported as higher-is-better ($\uparrow$).

![Refer to caption](https://arxiv.org/html/2504.17761v5/x13.png)

(a) User Preference score in the Intersection subset.

## 5 Conclusion

In this report, we present a new general image editing algorithm called Step1X-Edit, which will be publicly released to foster further innovation and research within the image editing community. To train the model effectively, we propose a new data generation pipeline which can generate large-scale high-quality image editing triples, each consisting of a reference image, an editing instruction, and a corresponding target image. Based on the collected dataset, we train our Step1X-Edit model by seamlessly integrating powerful Multimedia Large Language Model with a diffusion-based image decoder. According to the evaluations on our collected GEdit-Bench, our proposed algorithm outperforms the existing open-source image editing algorithms with a substantial margin.

## Contributors and Acknowledgments

We designate core contributors as those who have been involved in the development of Step1X-Edit throughout its entire process, while contributors are those who worked on the early versions or contributed part-time.

Core Contributors: Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Xianfang Zeng, Gang Yu.

Contributors: Honghao Fu, Ruoyu Wang, Yongliang Wu, Tianyu Wang, Haozhen Sun, Wen Sun, Bizhu Huang, Mei Chen, Kang An, Shuli Gao, Wei Ji, Tianhao You, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Daxin Jiang.

Corresponding Authors: Xianfang Zeng(zengxianfang@stepfun.com), Gang Yu (yugang@stepfu  
n.com), Daxin Jiang (djiang@stepfun.com).

[^1]: Stability AI. Stable diffusion 3.5. [https://huggingface.co/stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large), 2024. Accessed: 2025-04-17.

[^2]: Jinbin Bai, Wei Chow, Ling Yang, Xiangtai Li, Juncheng Li, Hanwang Zhang, and Shuicheng Yan. Humanedit: A high-quality human-rewarded dataset for instruction-based image editing. arXiv preprint arXiv:2412.04280, 2024.

[^3]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[^4]: Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.

[^5]: Black Forest Labs. Flux.1 \[dev\]. [https://huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), 2024.

[^6]: Black Forest Labs. Flux.1 fill \[dev\]. [https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), 2024. Accessed: 2025-04-19.

[^7]: Black Forest Labs. Flux.1 \[schnell\]. [https://huggingface.co/black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), 2024.

[^8]: Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402, 2022.

[^9]: Liang Chen, Shuai Bai, Wenhao Chai, Weichu Xie, Haozhe Zhao, Leon Vinci, Junyang Lin, and Baobao Chang. Multimodal representation alignment for image generation: Text-image interleaved control is easier than you think, 2025.

[^10]: Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing Dang, and Haoshuang Wang. Pp-ocr: A practical ultra lightweight ocr system. arXiv preprint arXiv:2009.09941, 2020.

[^11]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[^12]: Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025.

[^13]: Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Xihui Liu, and Hongsheng Li. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639, 2025.

[^14]: Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024.

[^15]: Google Gemini2. Experiment with gemini 2.0 flash native image generation, 2025.

[^16]: Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. Ace: All-round creator and editor following instructions via diffusion transformer. arXiv preprint arXiv:2410.00086, 2024.

[^17]: HiDream-ai. Hidream-e1. [https://github.com/HiDream-ai/HiDream-E1](https://github.com/HiDream-ai/HiDream-E1), 2025.

[^18]: HiDream-ai. Hidream-i1. [https://github.com/HiDream-ai/HiDream-I1](https://github.com/HiDream-ai/HiDream-I1), 2025.

[^19]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[^20]: Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with multi-modal instruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4754–4763, 2024.

[^21]: Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, and Ying Shan. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8362–8371, 2024.

[^22]: Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.

[^23]: Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision, pages 150–168. Springer, 2024.

[^24]: Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[^25]: Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. arXiv preprint arXiv:2312.14867, 2023.

[^26]: Black Forest Labs. Flux. [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux), 2024.

[^27]: Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, and Bhiksha Raj. Controlvar: Exploring controllable visual autoregressive modeling. arXiv preprint arXiv:2406.09750, 2024.

[^28]: Yaowei Li, Yuxuan Bian, Xu Ju, Zhaoyang Zhang, Ying Shan, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. ArXiv, abs/2412.10316, 2024.

[^29]: Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Controlar: Controllable image generation with autoregressive models. In International Conference on Learning Representations, 2025.

[^30]: lrzjason. Objectremovalalpha dataset. [https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha](https://huggingface.co/datasets/lrzjason/ObjectRemovalAlpha), 2025. Accessed: 2025-04-19.

[^31]: Pengqi Lu. Qwen2vl-flux: Unifying image and text guidance for controllable image generation, 2024.

[^32]: Bingqi Ma, Zhuofan Zong, Guanglu Song, Hongsheng Li, and Yu Liu. Exploring the role of large language models in prompt encoding for diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[^33]: Jiancang Ma, Qirong Peng, Xu Guo, Chen Chen, H. Lu, and Zhenyu Yang. X2i: Seamless integration of multimodal understanding into diffusion transformer via attention distillation. ArXiv, abs/2503.06134, 2025.

[^34]: Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025.

[^35]: Li Ming, Gu Xin, Chen Fan, Xing Xiaoying, Wen Longyin, Chen Chen, and Zhu Sijie. Superedit: Rectifying and facilitating supervision for instruction-based image editing. arXiv preprint arXiv:2505.02370, 2025.

[^36]: Chong Mou, Xintao Wang, Liangbin Xie, Jing Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. ArXiv, abs/2302.08453, 2023.

[^37]: OpenAI. Introducing 4o image generation, 2025.

[^38]: ostris. Flex.2-preview. [https://huggingface.co/ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview), 2025.

[^39]: Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

[^40]: Yulin Pan, Xiangteng He, Chaojie Mao, Zhen Han, Zeyinzi Jiang, Jingfeng Zhang, and Yu Liu. Ice-bench: A unified and comprehensive benchmark for image creating and editing. arXiv preprint arXiv:2503.14482, 2025.

[^41]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[^42]: Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.

[^43]: Yuandong Pu, Le Zhuo, Kaiwen Zhu, Liangbin Xie, Wenlong Zhang, Xiangyu Chen, Pneg Gao, Yu Qiao, Chao Dong, and Yihao Liu. Lumina-omnilv: A unified multimodal framework for general low-level vision. arXiv preprint arXiv:2504.04903, 2025.

[^44]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[^45]: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

[^46]: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations, 2025.

[^47]: Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021.

[^48]: Ying Shen, Yizhe Zhang, Shuangfei Zhai, Lifu Huang, Joshua M Susskind, and Jiatao Gu. Many-to-many image generation with auto-regressive diffusion models. arXiv preprint arXiv:2404.03109, 2024.

[^49]: Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.

[^50]: Yichun Shi, Peng Wang, and Weilin Huang. Seededit: Align image re-generation to image editing. arXiv preprint arXiv:2411.06686, 2024.

[^51]: Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. International Conference on Learning Representations (ICLR), 2021.

[^52]: StepFun. step-1o-turbo-vision. [https://platform.stepfun.com/](https://platform.stepfun.com/), 2025.

[^53]: Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.

[^54]: Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020.

[^55]: Qian Wang, Biao Zhang, Michael Birsak, and Peter Wonka. Instructedit: Improving automatic masks for diffusion-based image editing with user instructions. ArXiv, abs/2305.18047, 2023.

[^56]: Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, and Di Zhang. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content, 2024.

[^57]: Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18359–18369, 2023.

[^58]: Yufei Wang, Lanqing Guo, Zhihao Li, Jiaxing Huang, Pichao Wang, Bihan Wen, and Jian Wang. Training-free text-guided image editing with visual autoregressive model. arXiv preprint arXiv:2503.23897, 2025.

[^59]: Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024.

[^60]: Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024.

[^61]: Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

[^62]: Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. ArXiv, abs/2401.11708, 2024.

[^63]: Ziyu Yao, Jialin Li, Yifeng Zhou, Yong Liu, Xi Jiang, Chengjie Wang, Feng Zheng, Yuexian Zou, and Lei Li. Car: Controllable autoregressive modeling for visual generation. arXiv preprint arXiv:2410.04671, 2024.

[^64]: Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024.

[^65]: Yongsheng Yu, Ziyun Zeng, Hang Hua, Jianlong Fu, and Jiebo Luo. Promptfix: You prompt and we fix the photo. arXiv preprint arXiv:2405.16785, 2024.

[^66]: Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

[^67]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3813–3824, 2023.

[^68]: Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9026–9036, 2024.

[^69]: Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514, 2023.

[^70]: Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv, 2025.

[^71]: Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.

[^72]: Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 3:9150038, 2024.

[^73]: Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In European Conference on Computer Vision, pages 195–211. Springer, 2024.

[^74]: Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Señorita-2m: A high-quality instruction-based dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025.