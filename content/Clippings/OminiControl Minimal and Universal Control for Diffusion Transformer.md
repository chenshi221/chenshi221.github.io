---
title: "OminiControl: Minimal and Universal Control for Diffusion Transformer"
source: "https://arxiv.org/html/2411.15098v6"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Zhenxiong Tan  Songhua Liu  Xingyi Yang  Qiaochu Xue  Xinchao Wang  
National University of Singapore  
{zhenxiong, songhua.liu, xyang, e1352520}@u.nus.edu xinchao@nus.edu.sg

###### Abstract

We present OminiControl, a novel approach that rethinks how image conditions are integrated into Diffusion Transformer (DiT) architectures. Current image conditioning methods either introduce substantial parameter overhead or handle only specific control tasks effectively, limiting their practical versatility. OminiControl addresses these limitations through three key innovations: (1) a minimal architectural design that leverages the DiT’s own VAE encoder and transformer blocks, requiring just 0.1% additional parameters; (2) a unified sequence processing strategy that combines condition tokens with image tokens for flexible token interactions; and (3) a dynamic position encoding mechanism that adapts to both spatially-aligned and non-aligned control tasks. Our extensive experiments show that this streamlined approach not only matches but surpasses the performance of specialized methods across multiple conditioning tasks. To overcome data limitations in subject-driven generation, we also introduce Subjects200K, a large-scale dataset of identity-consistent image pairs synthesized using DiT models themselves. This work demonstrates that effective image control can be achieved without architectural complexity, opening new possibilities for efficient and versatile image generation systems.

![[Uncaptioned image]](https://arxiv.org/html/2411.15098v6/x1.png)

Figure 1: Results of our OminiControl on both subject-driven generation (top) and spatially-aligned tasks (bottom). The small images in red boxes show the input conditions.

## 1 Introduction

The ability to generate high-quality images with precise user control remains a central challenge in computer vision. While diffusion models [^17] [^44] have significantly advanced image generation, surpassing traditional GAN-based approaches [^11] in both quality and diversity, fine-grained control remains problematic. Text-conditioned models [^44] [^41] [^9] [^24] serve as the primary controllable generation paradigm, yet fundamentally lack the capacity to specify exact spatial details and visual attributes that users often need. To address this limitation, some works have explored image-based control methods [^61] [^60] [^37] [^63] [^27] [^50] [^57] [^28] [^42] [^56] [^29] [^52] [^15] [^2] [^30] [^18] [^14] [^26] [^46] [^57] [^54], enabling users to specify their intentions through reference images or visual hints, offering more precise guidance than text alone [^44].

However, current image control methods face several key challenges: First, existing methods require substantial architectural modifications with dedicated control modules [^61] [^60] [^38]. Second, these approaches show clear task bias – they typically work well for either spatially aligned controls (e.g., edge-guided or depth-guided generation [^61] [^37] [^63] [^42] [^29] [^52] [^15] [^2] [^30]) or spatially unaligned ones (e.g., style transfer or subject-driven generation [^60] [^28] [^27] [^50] [^18] [^14] [^33]), but rarely both. Third, current approaches are built primarily for UNet architectures [^45]. These approaches yield suboptimal results when applied to newer Diffusion Transformer (DiT) models [^40] due to fundamental architectural differences, despite the latter’s superior generation capabilities [^24] [^4] [^9].

These challenges, particularly within the emerging paradigm of Diffusion Transformer models, motivate us to rethink the fundamental approach to image control. We propose OminiControl, an omni-capable yet minimal framework that achieves effective and flexible control.

For minimal architecture, OminiControl employs a parameter reuse strategy that leverages DiT’s existing components—particularly its VAE encoder and transformer blocks—which already possess the necessary capabilities to process visual control signals. By reusing these components with minimal fine-tuning, our approach dramatically reduces parameter overhead while maintaining control effectiveness, requiring only 0.1% additional parameters compared to the base model.

To achieve omni-capability across diverse tasks, OminiControl introduces a unified sequence processing approach that fundamentally differs from previous methods’ rigid feature addition. By directly concatenating condition tokens with noisy image tokens in a unified sequence, we allow the multi-modal attention mechanism to discover appropriate relationships between tokens—whether spatial or semantic—without imposing artificial constraints. This flexibility is crucial for handling both spatially-aligned tasks like edge-guided generation and non-aligned tasks like subject-driven generation within a single framework.

Building on this unified approach, OminiControl incorporates a dynamic positioning strategy that adaptively assigns position indices based on the task type, enabling true omni-capability without task-specific architectural modifications. Furthermore, for practical flexibility, we introduce an attention bias mechanism that allows users to dynamically adjust the influence of image conditions at inference time, providing crucial control over the generation process.

The effectiveness of our architecture depends on high-quality training data, particularly for subject-driven generation. Recognizing that state-of-the-art DiT models can generate remarkably consistent image pairs of the same subject, we develop an automated data synthesis pipeline that creates Subjects200K—a dataset of over 200,000 diverse, identity-consistent images that provides the rich training signals needed to fully realize OminiControl ’s potential.

In summary, we highlight our contributions as follows:

- We propose OminiControl, a minimal and universal control framework for DiT models that requires only 0.1% additional parameters while effectively handling both spatially-aligned and non-aligned tasks, demonstrating that extensive architectural modifications are unnecessary for effective image conditioning.
- We identify two key technical innovations that enable omni-capability: (1) unified sequence processing that outperforms traditional feature addition, and (2) adaptive position encoding that strategically assigns position indices based on task requirements.
- We design a flexible attention bias mechanism that allows precise adjustment of conditioning strength at inference time, enhancing practical control within the multi-modal framework without compromising performance.
- We develop and release Subjects200K, a large-scale dataset containing over 200,000 identity-consistent images to advance future research.

## 2 Related works

### 2.1 Diffusion models

Diffusion models have emerged as a powerful framework for image generation [^17] [^44], demonstrating success across diverse tasks including text-to-image synthesis [^44] [^4] [^49], image-to-image translation [^48], and image editing [^35] [^1]. Recent advances have led to significant improvements in both quality and efficiency, notably through the introduction of latent diffusion models [^44]. To further enhance generative capabilities, large-scale transformer architectures have been integrated into these frameworks, leading to advanced models like DiT [^40] [^4] [^24] [^5]. Building on these architectural innovations, FLUX [^24] incorporates transformer-based design with flow matching objectives [^31], achieving state-of-the-art generation performance.

### 2.2 Controllable generation

Controllable generation has been extensively studied in the context of diffusion models. Text-to-image models [^44] [^41] have established a foundation for conditional generation, while various approaches have been developed to incorporate additional control signals such as image. Notable methods include ControlNet [^61], enabling spatially aligned control in diffusion models, and T2I-Adapter [^37], which improves efficiency with lightweight adapters. UniControl [^42] uses Mixture-of-Experts (MoE) to unify different spatial conditions, further reducing model size. However, these methods rely on spatially adding condition features to the denoising network’s hidden states, inherently limiting their effectiveness for spatially non-aligned tasks like subject-driven generation. IP-Adapter [^60] addresses this by introducing cross-attention through an additional encoder, and SSR-Encoder [^62] further enhances identity preservation in image-conditioned tasks. Despite these advances [^27] [^32] [^10] [^38] [^47] [^18] [^23] [^14] [^33], a unified solution for both spatially aligned and non-aligned tasks remains elusive.

## 3 Methods

### 3.1 Preliminary

The Diffusion Transformer (DiT) model [^40], employed in architectures like FLUX.1 [^24], Stable Diffusion 3 [^44], and PixArt [^4], uses transformer as denoising network to refine noisy image tokens iteratively.

A DiT model processes two types of tokens: noisy image tokens ${X}\in\mathbb{R}^{N\times d}$ and text condition tokens ${C}_{\text{T}}\in\mathbb{R}^{M\times d}$, where $d$ is the embedding dimension, $N$ and $M$ are the number of image and text tokens respectively (Figure 2(a)). Throughout the network, these tokens maintain consistent shapes as they pass through multiple transformer blocks.

In FLUX.1, each DiT block consists of layer normalization followed by Multi-Modal Attention (MMA) [^39], which incorporates Rotary Position Embedding (RoPE) [^51] to encode spatial information. For image tokens ${X}$, RoPE applies rotation matrices based on the token’s position $(i,j)$ in the 2D grid:

$$
{X}_{i,j}\rightarrow{X}_{i,j}\cdot R(i,j),
$$

where $R(i,j)$ is the rotation matrix at position $(i,j)$. Text tokens ${C}_{\text{T}}$ undergo the same transformation with their positions set to $(0,0)$.

The multi-modal attention mechanism then projects the position-encoded tokens into query $Q$, key $K$, and value $V$ representations. It enables the computation of attention between all tokens:

$$
\text{MMA}([{X};{C}_{\text{T}}])=\text{softmax}\left(\frac{QK^{\top}}{\sqrt{d}%
}\right)V,
$$

where $[{X};{C}_{\text{T}}]$ denotes the concatenation of image and text tokens. This formulation enables bidirectional attention.

### 3.2 OminiControl

![Refer to caption](https://arxiv.org/html/2411.15098v6/x2.png)

(a) Overview of the Diffusion Transformer (DiT) architecture and integration methods for image conditioning.

![Refer to caption](https://arxiv.org/html/2411.15098v6/x4.png)

(a) Training losses for different image condition integration methods.

Building upon the DiT architecture, with FLUX.1 [^24] as our base model, we aim to develop a minimal, omni-capable and controllable generation framework that accepts flexible control signals. This vision leads to our OminiControl, which we describe in this section.

#### 3.2.1 Minimal architecture

To minimize the extra architectural and parameter overhead, OminiControl adopts a *parameter reuse strategy*.

As its name implies, we reuses the VAE [^22] [^44] encoder from the base DiT model encode the condition image. These images are projected into the same latent space as the noisy input tokens, ensuring compatibility without introducing new modules. Following this, OminiControl processes both noisy image tokens and condition tokens jointly through the original DiT blocks. To adapt the shared DiT blocks for handling condition tokens, only lightweight LoRA fine-tuning [^8] is applied, avoiding costly full-parameter updates.

This approach contrasts with previous methods that rely on separate feature extractors like CLIP [^43] [^60] or additional control modules [^62], significantly reducing architectural complexity. Meanwhile, LoRA fine-tuning ensures parameter efficiency compared to duplicating the entire network as done in ControlNet [^61].

#### 3.2.2 Omni-capable token interaction

To achieve effective control across diverse tasks, OminiControl needs to enable flexible interactions between condition tokens and image tokens. We address this challenge through two key mechanisms: unified sequence processing and position-aware token interaction.

##### Unified sequence processing.

Building upon the shared latent space, we now think how to integrate condition tokens into the model for flexible control across.

Previous methods [^61] [^37] incorporate condition images through direct feature adding:

$$
{h_{X}}\leftarrow{h_{X}}+h_{{C}_{\text{I}}},
$$

where the condition features $h_{{C}_{\text{I}}}$ are spatially aligned and added to the noisy image features $h_{X}$.

We first implement this direct adding approach as illustrated in Figure 2(a). While the bare effectiveness is shown in Figure 2(b), this approach faces two limitations: (1) it lacks flexibility for non-aligned scenarios where spatial correspondence doesn’t exist, and (2) the rigid addition operation constrains potential interactions between condition and image tokens.

We then explored the unified sequence processing to integrate condition tokens into the model. Specifically, this approach directly concatenates condition tokens with noisy image tokens $[{X};{C}_{\text{T}};{C}_{\text{I}}]$ for multi-modal attention processing. This formulation enables flexible token interactions through DiT’s multi-modal attention mechanism, allowing direct relationships to emerge between any pair of tokens without imposing rigid spatial constraints.

As shown in Figure 4, this approach effectively handles both spatially aligned and non-aligned tasks, with attention maps revealing clear cross-token relationships and interaction patterns. Empirically, this unified sequence approach consistently achieves lower training loss compared to direct feature adding (Figure 3(a)), demonstrating its superior conditioning capability across diverse generation scenarios.

![Refer to caption](https://arxiv.org/html/2411.15098v6/x6.png)

Figure 4: (a) Attention maps for the Canny-to-image task (with setting from Figure 2(b) ), showing interactions between noisy image tokens X 𝑋 italic\_X and image condition tokens C I subscript 𝐶 𝐼 C\_{I} italic\_C start\_POSTSUBSCRIPT italic\_I end\_POSTSUBSCRIPT. Strong diagonal patterns indicate effective spatial alignment. (b) Subject-driven generation task, with input condition and output image. ( Prompt: This item is placed on a table with Christmas decorations around it. ) Attention maps for → i 𝑖 X\\to C\_{i} italic\_X → italic\_C start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT and C\_{i}\\to X italic\_C start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT → italic\_X illustrate accurate subject-focused attention.

##### Position-aware token interaction.

While the unified sequence approach allows flexible token interactions, encoding position information for the newly appended conditional tokens is not straightforward.

FLUX.1 adopts the RoPE mechanism to encode spatial information for both image and text tokens. Specifically, for a 512 $\times$ 512 input image, the VAE encoder produces a 32 $\times$ 32 grid of latent tokens, with each token assigned a position index $(i,j)$ where $i,j\in[0,31]$.

We first assign the same position indices to condition tokens as noisy image tokens, it works well for spatially aligned tasks like edge-guided generation. However, for non-aligned tasks like subject-driven generation, this shared position indexing can lead to suboptimal performance due to spatial overlap between condition and noisy image tokens. But if we shift the position indices of condition tokens by a fixed offset $\Delta$ (e.g., $(0,32)$), ensuring no spatial overlap with noisy image tokens, the training convergence accelerates significantly and final performance improves. (see Figure 3(b) and Figure 5).

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/shifting3.jpg)

Figure 5: The results from models with shared and shifted position indices. Both models are fully trained with 15k iterations.

This observation suggests that for spatially aligned tasks, a shared position indexing facilitates direct spatial correspondence between condition and image tokens, while for non-aligned tasks, shared position indexing can constrain the model’s ability to establish semantic relationships. Hence, we propose a dynamic positioning strategy based on the control task:

$$
(i,j)_{{C}_{\text{I}}}=\begin{cases}(i,j)_{{X}}&\text{for aligned tasks}\\
(i,j)+\Delta&\text{for non-aligned tasks}\end{cases}
$$

which makes OminiControl truly omni-capable by adapting to both spatially aligned and non-aligned control tasks.

#### 3.2.3 Control with flexibility

Although our unified sequence processing and position-aware token interaction mechanisms effectively enable joint attention between conditions and images, they also present a new challenge. Unlike previous methods [^60] [^61] that could simply scale condition features (e.g., $h_{X}\leftarrow h_{X}+\alpha\cdot h_{C_{I}}$ where $\alpha$ controls strength), the joint attention approach of OminiControl does not inherently support adjustable conditioning strength, which is crucial for practical applications where users need to balance text and image influences.

To address this limitation, we design a flexible control mechanism by introducing a bias term into the multi-modal attention computation. Specifically, for a given strength factor $\gamma$, we modify the attention operation in Equation 2 to:

$$
\text{MMA}([{X};{C}_{\text{T}};{C}_{\text{I}}])=\text{softmax}\left(\frac{QK^{%
\top}}{\sqrt{d}}+B(\gamma)\right)V,
$$

where $B(\gamma)$ is a bias matrix modulating the attention between concatenated tokens $[{X};{C}_{\text{T}};{C}_{\text{I}}]$. Given ${C}_{\text{T}}\in\mathbb{R}^{M\times d}$ and ${X},{C}_{\text{I}}\in\mathbb{R}^{N\times d}$, the bias matrix has the structure:

$$
B(\gamma)=\begin{bmatrix}\mathbf{0}_{M\times M}&\mathbf{0}_{M\times N}&\log(%
\gamma)\mathbf{1}_{N\times N}\\
\mathbf{0}_{N\times M}&\mathbf{0}_{N\times N}&\mathbf{0}_{N\times N}\\
\log(\gamma)\mathbf{1}_{N\times N}&\mathbf{0}_{M\times N}&\mathbf{0}_{N\times N%
}\end{bmatrix}.
$$

This formulation preserves the original attention patterns within each token type while scaling attention weights between ${X}$ and ${C}_{\text{I}}$ by $\log(\gamma)$. At test time, setting $\gamma=0$ removes the condition’s influence, while $\gamma>1$ enhances it, which makes OminiControl more flexible.

#### 3.2.4 Comparison with concurrent works

Several recent works also explore controllable image generation with DiT models. Some approaches [^24] [^34] [^12] employ channel concatenation to integrate condition tokens, which offers less flexibility than our unified sequence processing. Others [^3] [^6] focus exclusively on specific tasks such as style transfer or subject-driven generation, limiting their flexibility and generality. In contrast, our approach combines dynamic positioning with a flexible control mechanism to provide a more comprehensive and adaptable control framework compared to existing methods.

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/subject200k.jpg)

Figure 6: Examples from our Subjects200K dataset. Each pair of images shows the same object in varying contexts.

### 3.3 Subjects200K datasets

A critical challenge in developing universal control frameworks lies in obtaining high-quality training data for subject-driven generation. While OminiControl ’s architecture enables flexible control, effective training requires data that maintains subject consistency while incorporating natural variations in pose, lighting, and context.

Existing solutions often rely on identical image pairs [^60] or limited-scale datasets [^47] [^23], which present several limitations. Using identical pairs can lead to overfitting in our framework, causing the model to simply reproduce the input. Meanwhile, existing datasets with natural variations often lack either the scale or diversity needed for robust training.

To address this data challenge, we leverage a key observation: state-of-the-art DiT models like FLUX.1 [^25] can generate identity-consistent image pairs when provided with appropriate prompts [^20] [^19]. Building on this, we develop an pipeline to create Subjects200K, a large-scale dataset specifically designed for subject-driven generation <sup>1</sup>:

- Prompt Generation. We use GPT-4o to generate over 30,000 diverse subject descriptions. Each description represents the same subject in multiple scenes.
- Paired-image synthesis. We then reorganized the collected descriptions into structured prompts. Each prompt describes the same subject in two different scenes. The template of such prompt is shown in the Figure S3 of supplementary material.These prompts are then fed into FLUX to generate image pairs. Each pair is designed to maintain subject consistency while varying in context.
- Quality Assesment. Finally, we use GPT-4o to evaluate the generated pairs. Misaligned pairs are removed to ensure identity consistency and high image quality.

The resulting dataset (Figure 6) contains over 200,000 high-quality images spanning diverse categories. Each subject appears in multiple contexts, providing rich training signals for learning robust subject-driven control. To facilitate future research, we release both the dataset and our complete generation pipeline <sup>2</sup>.

## 4 Experiment

### 4.1 Setup

Tasks and base model. We evaluate our method on two categories of conditional generation tasks: spatially aligned tasks (including Canny-to-image, depth-to-image, masked-based inpainting, and colorization) and subject-driven generation. We build our method upon FLUX.1 [^24], a latent rectified flow transformer model for image generation. By default, we use FLUX.1-dev to generate images for spatially aligned tasks. In subject-driven generation tasks, we switch to FLUX.1-schnell as we observed it tend to produce better visual quality.

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/res_comp.jpg)

Figure 7: Qualitative comparison. Left: Spatially aligned tasks. Right: Subject-driven generation with beverage can and shoes. Our method demonstrates superior controllability and quality across all tasks. (More results are provided in supplementary materials.)

Implement details. Our method utilizes LoRA [^8] for fine-tuning the base model with a default rank of 4. To preserve the model’s original capabilities and achieve flexibility, the LoRA scale is set to 0 by default when processing non-condition tokens.

Training. Our model is trained with a batch size of 1 and gradient accumulation over 8 steps (effective batch size of 8). We employ the Prodigy optimizer [^36] with safeguard warmup and bias correction enabled, setting the weight decay to 0.01. For spatially aligned tasks, we use text-to-image-2M [^13] dataset with the last 300,000 images. For subject-driven generation, we utilize our proposed Subjects200K dataset. The experiments are conducted on 2 NVIDIA H100 GPUs (80GB each). For spatially aligned tasks, models are trained for 50,000 iterations, while subject-driven generation models are trained for 15,000 iterations.

Baselines. For spatially aligned tasks, we compare our method with both the original ControlNet [^61] and T2I-Adapter [^37] on Stable Diffusion 1.5, as well as ControlNetPro [^25], the FLUX.1 implementation of ControlNet. For subject-driven generation, we compare with IP-Adapter [^60], evaluating its implementations FLUX.1 [^58]. Additionally, we also with the official FLUX.1 Tools [^24] implementation.

Evaluation of spatially aligned tasks. We evaluate our model on both spatially aligned tasks and subject-driven generation. For spatially aligned tasks, we assess two aspects: generation quality and controllability. Generation quality is measured using FID [^16], SSIM, CLIP-IQA [^53], MAN-IQA [^59], MUSIQ [^21], and PSNR [^7] for visual fidelity, along with CLIP Text and CLIP Image scores [^43] for consistency. For controllability, we compute F1 Score between extracted and input edge maps in edge-conditioned generation, and MSE between extracted and original condition maps for other tasks (using Depth Anything for depth, color channel separation for colorization, etc.). We use 5,000 images from COCO 2017 validation set, and resize them to 512 $\times$ 512, then generate task-specific conditions and associated captions as prompts with a fixed seed of 42.

Evaluation of subject-driven generation. For subject-driven generation, we propose a five-criteria framework evaluating both preservation of subject characteristics (identity preservation, material quality, color fidelity, natural appearance) and accuracy of requested modifications, with all assessments conducted through GPT-4o’s vision capabilities to ensure systematic evaluation. Details are presented in the Section B.2 of supplementary material. We test on 750 text-condition pairs (30 subjects × 25 prompts) from DreamBooth [^47] dataset with 5 different seeds, using one selected image per subject as the condition.

![Refer to caption](https://arxiv.org/html/2411.15098v6/x7.png)

Figure 8: Radar charts visualization comparing our method (blue) with baselines across five evaluation metrics.

### 4.2 Main result

Spatially aligned tasks As shown in Table 2, we comprehensively evaluate our method against existing approaches on five spatially aligned tasks. Our method achieves the highest F1-Score of 0.38 on depth-to-image generation, significantly outperforming both SD1.5-based methods ControlNet [^61] and T2I-Adapter [^37], as well as FLUX.1-based ControlNetPro [^25]. In terms of general quality metrics, our approach demonstrates consistent superiority across most tasks, showing notably better performance in SSIM [^55], CLIP-IQA [^53], MAN-IQA [^59], MUSIQ [^21] and PSNR [^7] scores. For challenging tasks like deblurring and colorization, our method achieves substantial improvements: the MSE is reduced by 77% and 93% respectively compared to ControlNetPro, while the FID scores [^16] improve from 30.38 to 11.49 for deblurring. The CLIP-Text and CLIP-Image metrics [^43] indicate that our method maintains high consistency across all tasks, suggesting effective preservation of semantic alignment and image alignment while achieving better control and visual quality. As shown in Figure 7, our method produces sharper details and more faithful color reproduction in colorization tasks, while maintaining better structural fidelity in edge-guided generation and deblurring scenarios.

<table><tbody><tr><td>Methods</td><td>Base model</td><td>Parameters</td><td>Ratio</td></tr><tr><td>ControlNet</td><td rowspan="3">SD1.5 / 860M</td><td>361M</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 42%</td></tr><tr><td>T2I-Adapter</td><td>77M</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 9.0%</td></tr><tr><td>IP-Adapter</td><td>449M</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 52.2%</td></tr><tr><td>ControlNet</td><td rowspan="3">FLUX.1 / 12B</td><td>3.3B</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 27.5%</td></tr><tr><td>IP-Adapter</td><td>918M</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 7.6%</td></tr><tr><td>FLUX.1 Tools</td><td>612M</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 5.1%</td></tr><tr><td>Ours</td><td>FLUX.1 / 12B</td><td>14.5M / 48.7M w/ Encoder</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 0.1% / <math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation> <annotation>∼</annotation></semantics></math> 0.4% w/ Encoder</td></tr></tbody></table>

Table 1: Additional parameters introduced by different image conditioning methods. For IP-Adapter, the parameter count includes the CLIP Image encoder. For our method, we also report results when using the original VAE encoder from FLUX.1.

<table><tbody><tr><td rowspan="2">Task</td><td rowspan="2">Methods / Setting</td><td rowspan="2">Base Model</td><td>Controllability</td><td colspan="6">Image Quality</td><td colspan="2">Alignment</td></tr><tr><td>F1 <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math> / MSE <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>FID <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>SSIM <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>CLIP-IQA <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>MAN-IQA <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>MUSICQ <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>PSNR <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>CLIP Text <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>CLIP Image <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><td rowspan="5">Canny</td><td>ControlNet</td><td rowspan="2">SD1.5</td><td>0.35</td><td>18.74</td><td>0.36</td><td>0.65</td><td>0.45</td><td>67.81</td><td>10.27</td><td>0.305</td><td>0.752</td></tr><tr><td>T2I Adapter</td><td>0.22</td><td>20.06</td><td>0.35</td><td>0.57</td><td>0.39</td><td>67.89</td><td>9.53</td><td>0.305</td><td>0.748</td></tr><tr><td>ControlNet Pro</td><td rowspan="3">FLUX.1</td><td>0.21</td><td>98.69</td><td>0.25</td><td>0.48</td><td>0.37</td><td>56.91</td><td>9.22</td><td>0.192</td><td>0.537</td></tr><tr><td>Flux Tools</td><td>0.20</td><td>22.13</td><td>0.32</td><td>0.66</td><td>0.60</td><td>75.47</td><td>9.69</td><td>0.308</td><td>0.701</td></tr><tr><td>Ours</td><td>0.50</td><td>24.20</td><td>0.45</td><td>0.66</td><td>0.62</td><td>74.87</td><td>11.34</td><td>0.305</td><td>0.785</td></tr><tr><td rowspan="5">Depth</td><td>ControlNet</td><td rowspan="2">SD1.5</td><td>923</td><td>23.03</td><td>0.34</td><td>0.64</td><td>0.47</td><td>70.73</td><td>10.63</td><td>0.308</td><td>0.726</td></tr><tr><td>T2I Adapter</td><td>1560</td><td>24.72</td><td>0.28</td><td>0.61</td><td>0.39</td><td>69.99</td><td>9.50</td><td>0.309</td><td>0.721</td></tr><tr><td>ControlNet Pro</td><td rowspan="3">FLUX.1</td><td>2958</td><td>62.20</td><td>0.26</td><td>0.55</td><td>0.39</td><td>66.85</td><td>9.38</td><td>0.212</td><td>0.547</td></tr><tr><td>Flux Tools</td><td>767</td><td>24.56</td><td>0.32</td><td>0.68</td><td>0.59</td><td>75.30</td><td>10.15</td><td>0.308</td><td>0.715</td></tr><tr><td>Ours</td><td>537</td><td>31.04</td><td>0.39</td><td>0.68</td><td>0.60</td><td>74.04</td><td>10.53</td><td>0.305</td><td>0.749</td></tr><tr><td rowspan="3">Mask</td><td>ControlNet</td><td>SD1.5</td><td>7588</td><td>13.14</td><td>0.68</td><td>0.58</td><td>0.42</td><td>67.22</td><td>18.96</td><td>0.300</td><td>0.848</td></tr><tr><td>Flux Tools</td><td rowspan="2">FLUX.1</td><td>6610</td><td>11.40</td><td>0.73</td><td>0.56</td><td>0.45</td><td>68.92</td><td>18.37</td><td>0.305</td><td>0.874</td></tr><tr><td>Ours</td><td>6351</td><td>10.20</td><td>0.78</td><td>0.59</td><td>0.49</td><td>70.78</td><td>19.59</td><td>0.305</td><td>0.892</td></tr><tr><td rowspan="2">Colorization</td><td>ControlNet Pro</td><td rowspan="2">FLUX.1</td><td>994</td><td>30.38</td><td>0.75</td><td>0.40</td><td>0.31</td><td>54.38</td><td>16.23</td><td>0.279</td><td>0.781</td></tr><tr><td>Ours</td><td>73</td><td>10.37</td><td>0.92</td><td>0.56</td><td>0.48</td><td>69.40</td><td>21.56</td><td>0.305</td><td>0.884</td></tr><tr><td rowspan="2">Deblur</td><td>ControlNet Pro</td><td rowspan="2">FLUX.1</td><td>338</td><td>16.27</td><td>0.64</td><td>0.55</td><td>0.43</td><td>70.95</td><td>20.53</td><td>0.294</td><td>0.853</td></tr><tr><td>Ours</td><td>62</td><td>18.89</td><td>0.58</td><td>0.59</td><td>0.54</td><td>70.98</td><td>21.82</td><td>0.301</td><td>0.856</td></tr></tbody></table>

Table 2: Quantitative comparison with baseline methods on five spatially aligned tasks. We evaluate methods based on Controllability (F1-Score for Canny, MSE for others), Image Quality (SSIM, FID, CLIP-IQA, MAN-IQA, MUSIQ, PSNR), and Alignment (CLIP Text and CLIP Image). For F1-Score (used in Canny to Image task), higher is better; for MSE, lower is better. Best results are shown in bold.The second-best results are highlighted with underlines.

Subject driven generation Figure 8 presents a comprehensive comparison against existing baselines. Our method demonstrates superior performance, particularly in identity preservation and modification accuracy. Averaging over random seeds, we achieve 75.8% modification accuracy compared to IP-Adapter (FLUX)’s 57.7%, while maintaining 50.6% identity preservation against IP-Adapter (SD 1.5)’s 29.4%. The advantage amplifies in best-seed scenarios, achieving 90.7% modification accuracy and 82.3% identity preservation - surpassing the strongest baselines by 15.8 and 18.0 percentage points, demonstrating effective subject-fidelity editing. These quantitative results are further corroborated by user studies presented in supplementary material B.2.

Paramter efficiency As shown in Table 1, our approach achieves remarkable parameter efficiency compared to existing methods. For the 12B parameter FLUX.1 model, our method requires only 14.5M trainable parameters (approximately 0.1%), which is significantly lower than ControlNet (27.5%) and IP-Adapter (7.6%). Even when utilizing the original VAE encoder from FLUX.1, our method still maintains high efficiency with just 0.4% additional parameters, demonstrating the effectiveness of our parameter-efficient design.

Condition strength factor We evaluate our condition strength control mechanism (Section 3.2.3) through qualitative experiments. Figure 9 shows generated results with varying strength factor $\gamma$. Results show that $\gamma$ effectively controls the generation process for both spatially aligned tasks like depth-to-image generation and non-aligned tasks like subject-driven generation, enabling flexible control over the condition’s influence.

![Refer to caption](https://arxiv.org/html/2411.15098v6/x8.png)

Figure 9: Demonstration of the condition strength control.

### 4.3 Ablation Studies

To better understand the key factors that influence our model’s control capabilities, we conducted comprehensive ablation studies examining parameter efficiency, architectural decisions, and component contributions.

Impact of LoRA rank. We conducted extensive experiments with different LoRA ranks (1, 2, 4, 8, and 16) for the Canny-to-image task. As shown in Table 3, our experiments show that increasing the LoRA rank generally improves model performance, with rank 16 achieving the best results across multiple aspects. However, even with smaller ranks (e.g., rank 1), the model demonstrates competitive performance, especially in text-image alignment, showing the efficiency of our approach even with limited parameters.

<table><tbody><tr><td>Study</td><td>Setting</td><td>FID <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>SSIM <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>F1 Score <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>CLIP Score <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><td rowspan="5">LoRA Rank</td><td>1</td><td>21.09</td><td>0.412</td><td>0.385</td><td>0.765</td></tr><tr><td>2</td><td>21.28</td><td>0.411</td><td>0.377</td><td>0.751</td></tr><tr><td>4</td><td>20.63</td><td>0.407</td><td>0.380</td><td>0.761</td></tr><tr><td>8</td><td>21.40</td><td>0.404</td><td>0.3881</td><td>0.761</td></tr><tr><td>16</td><td>19.71</td><td>0.425</td><td>0.407</td><td>0.764</td></tr><tr><td rowspan="2">Condition Blocks</td><td>Early</td><td>25.66</td><td>0.369</td><td>0.23</td><td>0.72</td></tr><tr><td>Full</td><td>20.63</td><td>0.407</td><td>0.38</td><td>0.76</td></tr></tbody></table>

Table 3: Ablation studies on (1) LoRA rank for the Canny-to-image task and (2) condition signal integration approaches. Results show that LoRA rank of 16 and full-depth integration achieve the best performance. Rows with blue background indicate our default settings (LoRA rank=4, Full condition integration). Best results are in bold.

Conditioning depth. FLUX.1’s transformer architecture features two distinct types of blocks: early blocks that employ separate normalization modules for different modalities tokens (text and image) and later blocks that share unified normalization across all tokens. As shown in Table 3, experiments reveal that restricting condition signal integration to only these early blocks results in insufficient controllability. This suggests that allowing the condition signals to influence the entire transformer stack is crucial for generate better results. Notably, this finding indicates that the preview approaches [^25] [^58] [^60] [^61] [^37] of injecting condition signals primarily in early blocks, which were effective in UNet-based architectures, may not fully translate to DiT-based models like FLUX.1.

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/LoRA.jpg)

Figure 10: Ablation study of critical modules for conditional generation. Given the censored Mona Lisa image and text prompt (bottom left), we test removing LoRA from different components.

Critical module analysis. To identify essential components for effective control, we conducted fine-grained ablation studies as shown in Figure 10. Results demonstrate that normalization layers and attention projections—particularly query ($W_{Q}$) and key ($W_{K}$)—are critical for maintaining control quality. Removing LoRA from these components significantly degrades conditional rendering, while value projections ($W_{V}$) have less impact. These insights reveal that control signals might propagate through normalization pathways and attention routing mechanisms rather than feature transformation processes.

## 5 Conclusion

OminiControl offers parameter-efficient image-conditional control for DiT across diverse tasks using a unified token approach, requiring only 0.1% additional parameters. The Subjects200K dataset — featuring over 200,000 high-quality, subject-consistent images—further supports advancements in subject-driven generation, with experimental results confirming OminiControl ’s effectiveness in both spatially-aligned and non-aligned tasks.

However, the unified sequence approach increases the total number of tokens processed through the network, potentially limiting computational efficiency during inference. Addressing this token efficiency challenge while maintaining our method’s control capabilities represents an important direction for future research in parameter-efficient conditional generation.

## 6 Acknowledgment

We would like to acknowledge that the computational work involved in this research work is partially supported by NUS IT’s Research Computing group using grant numbers NUSREC-HPC-00001.

## References

Supplementary Material

## Appendix A Details of Subjects200K datasets

We present a comprehensive synthetic dataset constructed to address the limitations in scale and image quality found in previous datasets [^47] [^23] [^27] [^28]. Our approach leverages FLUX.1-dev [^24] to generate high-quality, consistent images of the same subject under various conditions.

Subjects200K dataset currently consists of two splits, both generated using similar pipelines. Split-1 contains paired images of objects in different scenes, while Split-2 pairs each object’s scene image with its corresponding studio photograph. Due to their methodological similarities, we primarily focus on describing the synthesis process and details of Split-2, although both splits are publicly available. Our complete Subjects200K dataset can be fully accessed via this [link](https://github.com/Yuanshi9815/Subjects200K).

### A.1 Generation pipeline

Our dataset generation process consists of three main stages: description generation, image synthesis, and quality assessment.

Description Generation We employed ChatGPT-4o to create a hierarchical structure of descriptions: We first generated 42 diverse object categories, including furniture, vehicles, electronics, clothing, and others. For each category, we created multiple object instances, totaling 4,696 unique objects. Each object entry consists of: (1) A brief description, (2) Eight diverse scene descriptions, (3) One studio photo description. Figure S2 shows a representative example of our structured description format.

Image Synthesis We designed a prompt template to leverage FLUX’s capability of generating paired images containing the same subject. Our template synthesizes a comprehensive prompt by combining a brief object description with two distinct scene descriptions, ensuring subject consistency while introducing environmental variations.

The detailed prompt structure is illustrated in Figure S3. For each prompt, we set the image dimensions to 1056×528 pixels and generated five images using different random seeds to ensure diversity in our dataset. During the training process, we first split the paired images horizontally, then performed central cropping to obtain 512×512 pixel image pairs. This padding strategy was implemented to address cases where the generated images were not precisely bisected, preventing potential artifacts from appearing in the wrong half of the split images.

![Refer to caption](https://arxiv.org/html/2411.15098v6/x9.png)

Figure S1: Examples of successful and failed generation results from Subjects200K dataset. Green checks indicate successful cases where subject identity and characteristics are well preserved, while red crosses show failure cases.

Quality assessment We leveraged ChatGPT-4o’s vision capabilities to rigorously evaluate the quality of images generated by FLUX.1-dev. The assessment focused on multiple critical aspects:

- Image composition: Verifying that each image properly contains two side-by-side views.
- Subject consistency: Ensuring the subject maintains identity across both views.
- Image quality: Confirming high resolution and visual fidelity.

To maintain stringent quality standards, each image underwent five independent evaluations by ChatGPT-4o. Only images that passed all five evaluations were included in our training dataset. Figure S1 presents representative examples from our quality-controlled dataset.

[⬇](data:text/plain;base64,ewogICAgImJyaWVmX2Rlc2NyaXB0aW9uIjoKICAgICAgICAiQSBmaW5lbHktY3JhZnRlZCB3b29kZW4gc2VhdGluZyBwaWVjZS4iLAogICAgInNjZW5lX2Rlc2NyaXB0aW9ucyI6IFsKICAgICAgICAiU2V0IG9uIGEgc2FuZHkgc2hvcmUgYXQgZHVzaywgaXQgZmFjZXMgdGhlIG9jZWFuIHdpdGggYSBnZW50bGUgYnJlZXplIHJ1c3RsaW5nIG5lYXJieSBwYWxtcywgYmF0aGVkIGluIHNvZnQsIHdhcm0gdHdpbGlnaHQuIiwKICAgICAgICAiUG9zaXRpb25lZCBpbiBhIGJ1c3RsaW5nIHVyYmFuIGNhZmUsIGl0IHN0YW5kcyBvdXQgYWdhaW5zdCBleHBvc2VkIGJyaWNrIHdhbGxzLCBjYXB0dXJpbmcgdGhlIG1pZGRheSBzdW4gdGhyb3VnaCBhIHdpZGUgYmF5IHdpbmRvdy4iCiAgICAgICAgLy8gQWRkaXRpb25hbCBzaXggc2NlbmUgZGVzY3JpcHRpb25zIG9taXR0ZWQKICAgIF0sCiAgICAic3R1ZGlvX3Bob3RvX2Rlc2NyaXB0aW9uIjoKICAgICAgICAiSW4gYSBwcm9mZXNzaW9uYWwgc3R1ZGlvIGFnYWluc3QgYSBwbGFpbiB3aGl0ZSBiYWNrZHJvcCwgaXQgaXMgY2FwdHVyZWQgaW4gdGhyZWUtcXVhcnRlciB2aWV3IHVuZGVyIHVuaWZvcm0gaGlnaC1rZXkgbGlnaHRpbmcsIHNob3djYXNpbmcgdGhlIGRlbGljYXRlIGdyYWluIGFuZCBzbW9vdGggb2YgaXRzIGZpbmVseS1jcmFmdGVkIHN1cmZhY2VzLiIKfQ==)

{

"brief\_description":

"A finely-crafted wooden seating piece.",

"scene\_descriptions": \[

"Set on a sandy shore at dusk, it faces the ocean with a gentle breeze rustling nearby palms, bathed in soft, warm twilight.",

"Positioned in a bustling urban cafe, it stands out against exposed brick walls, capturing the midday sun through a wide bay window."

// Additional six scene descriptions omitted

\],

"studio\_photo\_description":

"In a professional studio against a plain white backdrop, it is captured in three-quarter view under uniform high-key lighting, showcasing the delicate grain and smooth of its finely-crafted surfaces."

}

Figure S2: An example of our structured description format for dataset generation.

[⬇](data:text/plain;base64,cHJvbXB0XzEgPSBmIlR3byBzaWRlLWJ5LXNpZGUgaW1hZ2VzIG9mIHRoZSBzYW1lIG9iamVjdDoge2JyaWVmX2Rlc2NyaXB0aW9ufSIKcHJvbXB0XzIgPSBmIkxlZnQ6IHtzY2VuZV9kZXNjcmlwdGlvbjF9Igpwcm9tcHRfMyA9IGYiUmlnaHQ6IHtzY2VuZV9kZXNjcmlwdGlvbjJ9Igpwcm9tcHRfaW1hZ2UgPSBmIntwcm9tcHRfMX07IHtwcm9tcHRfMn07IHtwcm9tcHRfM30i)

prompt\_1 = f"Two side-by-side images of the same object: {brief\_description}"

prompt\_2 = f"Left: {scene\_description1}"

prompt\_3 = f"Right: {scene\_description2}"

prompt\_image = f"{prompt\_1}; {prompt\_2}; {prompt\_3}"

Figure S3: Our prompt template for paired image generation. The template combines a brief object description with two distinct scene descriptions to maintain subject consistency while varying environmental conditions.

### A.2 Dataset Statistics

In Split-2, we first generated 42 distinct object categories, from which we created and curated a set of 4,696 detailed object instances. Then we combine these descriptions to generate 211,320 subject-consistent image pairs. Through rigorous quality control using GPT-4o, we selected 111,767 high-quality image pairs for our final dataset. This extensive filtering process ensured the highest standards of image quality and subject consistency, resulting in a collection of 223,534 high-quality training images.

## Appendix B Additional experimental results

### B.1 Effect of training data

![Refer to caption](https://arxiv.org/html/2411.15098v6/x10.png)

Figure S4: Comparison of models trained with different data. The model trained by data augmentation tends to copy inputs directly, while model trained by our Subjects200K generates novel views while preserving identity.

For subject-driven generation, our model takes a reference image of a subject (e.g., a plush toy or an object) and a text description as input, aiming to generate novel images of the same subject following the text guidance while preserving its key characteristics.

To validate the effectiveness of our Subjects200K dataset described in Section 3.3, we compare two training strategies for this task. The first approach relies on traditional data augmentation, where we apply random cropping, rotation, scaling, and adjustments to contrast, saturation, and color to the original images. The second approach utilizes our Subjects200K dataset. As shown in Figure S4, the model trained with data augmentation only learns to replicate the input conditions with minimal changes. In the first row, it simply places the taco plush toy in a bright room setting while maintaining its exact appearance and pose. Similarly, in the second row, the yellow alarm clock is reproduced with nearly identical details despite the window-side placement instruction. In contrast, our Subjects200K -trained model demonstrates the ability to generate diverse yet consistent views of the subjects while faithfully following the text prompts.

### B.2 Evaluation for subject-driven generation

Framework and criteria. To systematically evaluate subject-driven generation quality, we establish a framework with five criteria assessing both preservation of subject characteristics and accuracy of requested modifications:

- Identity Preservation: Evaluates preservation of essential identifying features (e.g., logos, brand marks, distinctive patterns)
- Material Quality: Assesses if material properties and surface characteristics are accurately represented
- Color Fidelity: Evaluates if colors remain consistent in regions not specified for modification
- Natural Appearance: Assesses if the generated image appears realistic and coherent
- Modification Accuracy: Verifies if the changes specified in the text prompt are properly executed

<table><tbody><tr><td>Method</td><td>Identity</td><td>Material</td><td>Color</td><td>Natural</td><td>Modification</td><td>Average</td></tr><tr><td></td><td>preservation</td><td>quality</td><td>fidelity</td><td>appearance</td><td>accuracy</td><td>score</td></tr><tr><td colspan="7">Average over 5 random seeds</td></tr><tr><td>IP-Adapter (SD 1.5)</td><td>29.4</td><td>86.1</td><td>45.3</td><td>97.9</td><td>17.0</td><td>55.1</td></tr><tr><td>SSR-Encoder</td><td>46.0</td><td>92.0</td><td>54.2</td><td>96.3</td><td>28.5</td><td>63.4</td></tr><tr><td>IP-Adapter (FLUX)</td><td>11.8</td><td>65.8</td><td>30.8</td><td>98.1</td><td>57.7</td><td>52.8</td></tr><tr><td>Ours</td><td>50.6</td><td>84.3</td><td>55.0</td><td>98.5</td><td>75.8</td><td>72.8</td></tr><tr><td colspan="7">Best score over 5 random seeds</td></tr><tr><td>IP-Adapter (SD 1.5)</td><td>56.3</td><td>98.9</td><td>70.1</td><td>99.7</td><td>37.2</td><td>72.5</td></tr><tr><td>SSR-Encoder</td><td>64.3</td><td>99.2</td><td>74.4</td><td>99.1</td><td>53.6</td><td>78.1</td></tr><tr><td>IP-Adapter (FLUX)</td><td>27.5</td><td>86.1</td><td>53.6</td><td>99.9</td><td>74.9</td><td>68.4</td></tr><tr><td>Ours</td><td>82.3</td><td>98.0</td><td>88.4</td><td>100.0</td><td>90.7</td><td>91.9</td></tr></tbody></table>

Table S1: Quantitative evaluation results (in percentage) across different evaluation criteria. Higher values indicate better performance.

![Refer to caption](https://arxiv.org/html/2411.15098v6/x11.png)

Figure S5: User study results comparing different methods across three metrics: identity consistency, text-image alignment, and visual coherence.

User studies. To further validate our approach, we conducted user studies collecting 375 valid responses. Participants evaluated the generated images across three key dimensions: identity consistency, text-image alignment, and visual coherence between subjects and backgrounds. The results shown in Figure S5 corroborate our quantitative findings, with our method achieving superior performance across all evaluation criteria.

### B.3 Additional generation results

We showcase more generation results from our method. Figure S7 presents additional results on the DreamBooth dataset, while Figure S8 demonstrates our method’s effectiveness on other subject-driven generation tasks.

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/full_comp.jpg)

Figure S6: More results on Dreambooth dataset.

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/dreambooth.jpg)

Figure S7: More results on Dreambooth dataset.

![Refer to caption](https://arxiv.org/html/2411.15098v6/extracted/6603175/fig/more_results.jpg)

Figure S8: More results on other subject-driven generation tasks.

[^1]: Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 18208–18218, 2022.

[^2]: Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023.

[^3]: Shengqu Cai, Eric Chan, Yunzhi Zhang, Leonidas Guibas, Jiajun Wu, and Gordon Wetzstein. Diffusion self-distillation for zero-shot customized image generation. *arXiv preprint arXiv:2411.18616*, 2024.

[^4]: Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart- $\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. *arXiv preprint arXiv:2310.00426*, 2023.

[^5]: Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart- $\{$ $\backslash$ delta $\}$: Fast and controllable image generation with latent consistency models. *arXiv preprint arXiv:2401.05252*, 2024.

[^6]: Jooyoung Choi, Chaehun Shin, Yeongtak Oh, Heeseung Kim, and Sungroh Yoon. Style-friendly snr sampler for style-driven generation. *arXiv preprint arXiv:2411.14793*, 2024.

[^7]: Wikipedia contributors. Peak signal-to-noise ratio, 2024. Accessed: 2024-03-04.

[^8]: Shilpa Devalal and A Karthikeyan. Lora technology-an overview. In *2018 second international conference on electronics, communication and aerospace technology (ICECA)*, pages 284–290. IEEE, 2018.

[^9]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.

[^10]: Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. *arXiv preprint arXiv:2208.01618*, 2022.

[^11]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. *Communications of the ACM*, 63(11):139–144, 2020.

[^12]: Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. Ace: All-round creator and editor following instructions via diffusion transformer. *arXiv preprint arXiv:2410.00086*, 2024.

[^13]: Jacky Hate. Text-to-image-2m dataset. [https://huggingface.co/datasets/jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M), 2024.

[^14]: Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. Anystory: Towards unified single and multiple subject personalization in text-to-image generation, 2025.

[^15]: Qingdong He, Jinlong Peng, Pengcheng Xu, Boyuan Jiang, Xiaobin Hu, Donghao Luo, Yong Liu, Yabiao Wang, Chengjie Wang, Xiangtai Li, and Jiangning Zhang. Dynamiccontrol: Adaptive condition selection for improved text-to-image generation, 2024.

[^16]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017.

[^17]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.

[^18]: Miao Hua, Jiawei Liu, Fei Ding, Wei Liu, Jie Wu, and Qian He. Dreamtuner: Single image is enough for subject-driven generation, 2023.

[^19]: Lianghua Huang, Wei Wang, Zhi-Fan Wu, Huanzhang Dou, Yupeng Shi, Yutong Feng, Chen Liang, Yu Liu, and Jingren Zhou. Group diffusion transformers are unsupervised multitask learners. 2024a.

[^20]: Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. *arXiv preprint arXiv:2410.23775*, 2024b.

[^21]: Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 5148–5157, 2021.

[^22]: Diederik P Kingma. Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*, 2013.

[^23]: Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1931–1941, 2023.

[^24]: Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024a. Accessed: 2024-11-12.

[^25]: Shakker Labs. Flux.1-dev-controlnet-union-pro. [https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro), 2024b.

[^26]: Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. *Advances in Neural Information Processing Systems*, 36:30146–30166, 2023.

[^27]: Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. *Advances in Neural Information Processing Systems*, 36, 2024a.

[^28]: Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8640–8650, 2024b.

[^29]: Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Controlar: Controllable image generation with autoregressive models, 2025.

[^30]: Kuan Heng Lin, Sicheng Mo, Ben Klingher, Fangzhou Mu, and Bolei Zhou. Ctrl-x: Controlling structure and appearance for text-to-image generation without guidance, 2024.

[^31]: Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

[^32]: Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In *ACM SIGGRAPH 2024 Conference Papers*, pages 1–12, 2024a.

[^33]: Yuhang Ma, Wenting Xu, Jiji Tang, Qinfeng Jin, Rongsheng Zhang, Zeng Zhao, Changjie Fan, and Zhipeng Hu. Character-adapter: Prompt-guided region control for high-fidelity character customization, 2024b.

[^34]: Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling. *arXiv preprint arXiv:2501.02487*, 2025.

[^35]: Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. *arXiv preprint arXiv:2108.01073*, 2021.

[^36]: Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. In *Forty-first International Conference on Machine Learning*, 2024.

[^37]: Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 4296–4304, 2024.

[^38]: Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. *arXiv preprint arXiv:2310.02992*, 2023.

[^39]: Zexu Pan, Zhaojie Luo, Jichen Yang, and Haizhou Li. Multi-modal attention for speech emotion recognition. *arXiv preprint arXiv:2009.04107*, 2020.

[^40]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4195–4205, 2023.

[^41]: Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023.

[^42]: Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. *arXiv preprint arXiv:2305.11147*, 2023.

[^43]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021.

[^44]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021.

[^45]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pages 234–241. Springer, 2015.

[^46]: Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. *arXiv preprint arXiv:2410.10792*, 2024.

[^47]: Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 22500–22510, 2023.

[^48]: Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In *ACM SIGGRAPH 2022 conference proceedings*, pages 1–10, 2022a.

[^49]: Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022b.

[^50]: Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8543–8552, 2024.

[^51]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

[^52]: Yanan Sun, Yanchen Liu, Yinhao Tang, Wenjie Pei, and Kai Chen. Anycontrol: Create your artwork with versatile control on text-to-image generation, 2024.

[^53]: Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images, 2022.

[^54]: Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. *arXiv preprint arXiv:2401.07519*, 2024.

[^55]: Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE transactions on image processing*, 13(4):600–612, 2004.

[^56]: Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 15943–15953, 2023.

[^57]: Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. *International Journal of Computer Vision*, pages 1–20, 2024.

[^58]: XLabs-AI. Flux-ip-adapter. [https://huggingface.co/XLabs-AI/flux-ip-adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter), 2024.

[^59]: Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1191–1200, 2022.

[^60]: Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv preprint arXiv:2308.06721*, 2023.

[^61]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 3836–3847, 2023.

[^62]: Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8069–8078, 2024.

[^63]: Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. *Advances in Neural Information Processing Systems*, 36, 2024.