---
title: "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space"
source: "https://ar5iv.labs.arxiv.org/html/2506.15742"
author:
published:
created: 2026-04-30
description: "We present evaluation results for FLUX.1 Kontext, a generative flow matching model that unifies image generation and editing. The model generates novel output views by incorporating semantic context from text and image…"
tags:
  - "clippings"
---
Black Forest Labs  
Stephen Batifol  Andreas Blattmann  Frederic Boesel  Saksham Consul  Cyril Diagne  
Tim Dockhorn  Jack English  Zion English  Patrick Esser  Sumith Kulal  
Kyle Lacey  Yam Levi  Cheng Li  Dominik Lorenz  Jonas Müller  
Dustin Podell  Robin Rombach  Harry Saini  Axel Sauer  Luke Smith  
Please cite this works as "Black Forest Labs (2025)"

###### Abstract

We present evaluation results for *FLUX.1 Kontext*, a generative flow matching model that unifies image generation and editing. The model generates novel output views by incorporating semantic context from text and image inputs. Using a simple sequence concatenation approach, *FLUX.1 Kontext* handles both local editing and generative in-context tasks within a single unified architecture. Compared to current editing models that exhibit degradation in character consistency and stability across multiple turns, we observe that *FLUX.1 Kontext* improved preservation of objects and characters, leading to greater robustness in iterative workflows. The model achieves competitive performance with current state-of-the-art systems while delivering significantly faster generation times, enabling interactive applications and rapid prototyping workflows. To validate these improvements, we introduce *KontextBench*, a comprehensive benchmark with 1026 image-prompt pairs covering five task categories: local editing, global editing, character reference, style reference and text editing. Detailed evaluations show the superior performance of *FLUX.1 Kontext* in terms of both single-turn quality and multi-turn consistency, setting new standards for unified image processing models.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/gull/input.jpg)

(a) Context image generated with FLUX.1.

## 1 Introduction

Images are a foundation of modern communication and form the basis for areas as diverse as social media, e-commerce, scientific visualization, entertainment, and memes. As the volume and speed of visual content increases, so does the demand for intuitive but faithful and accurate image editing. Professional and casual users expect tools that preserve fine detail, maintain semantic coherence, and respond to increasingly natural language commands. The advent of large-scale generative models has changed this landscape, enabling purely text-driven image synthesis and modifications that were previously impractical or impossible [^11] [^17] [^41] [^42] [^40] [^2] [^12] [^49] [^21].

Traditional image processing pipelines work by directly manipulating pixel values or by applying geometric and photometric transformations under explicit user control [^14] [^55]. In contrast, generative processing uses deep learning models and their learned representations to synthesize content that seamlessly fits into the new scene. Two complementary capabilities are central to this paradigm

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/cc/img1.jpg)

(a) Input image

Recent Advances. InstructPix2Pix [^5] and subsequent work [^4] demonstrated the promise of synthetic instruction-response pairs for fine-tuning a diffusion model for image editing, while learning-free methods for personalized text-to-image synthesis [^13] [^43] [^24] enable image modification with off-the-shelf, high-performance image generation models [^42] [^40]. Subsequent instruction-driven editors such as Emu Edit [^51], OmniGen [^56], HiDream-E1 [^15] and ICEdit [^62] – extend these ideas to refined datasets and model architectures. [^20] introduce in-context LoRAs for diffusion transformers on specific tasks, where each task needs to train dedicated LoRA weights. Novel proprietary systems embedded in multimodal LLMs (e.g., GPT-Image [^36] and Gemini Native Image Gen [^23]) further blur the line between dialog and editing. Generative platforms such as Midjourney [^35] and RunwayML [^44] integrate these advances into end-to-end creative workflows.

Shortcomings of recent approaches. In terms of results, current approaches struggle with three major shortcomings: (i) instruction-based methods trained on synthetic pairs inherit the shortcomings of their generation pipelines, limiting the variety and realism of achievable edits; (ii) maintaining the accurate appearance of characters and objects across multiple edits remains an open problem, hindering story-telling and brand-sensitive applications; (iii) in addition to lower quality compared to denoising-based approaches, autoregressive editing models integrated into large multimodal systems often come with long runtimes that are incompatible with interactive use.

Our Solution. We introduce *FLUX.1 Kontext*, a flow-based generative image processing model that matches or exceeds the quality of state-of-the-art black-box systems while overcoming the above limitations. *FLUX.1 Kontext* is a simple flow matching model trained using only a velocity prediction target on a concatenated sequence of context and instruction tokens.

In particular, *FLUX.1 Kontext* offers:

- Character consistency: *FLUX.1 Kontext* excels at character preservation, including multiple, iterative edit turns.
- Interactive speed: *FLUX.1 Kontext* is *fast*. Both text-to-image and image-to-image application reach speeds for synthesising an image at $1024\times 1024$ of 3–5 seconds.
- Iterative application: Fast inference and robust consistency allow users to refine an image through multiple successive edits with minimal visual drift.

## 2 FLUX.1

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/fusedditblock.jpg)

Figure 3: A fused DiT block equipped with rotary positional embeddings

FLUX.1 is a rectified flow transformer [^12] [^30] [^32] trained in the latent space of an image autoencoder [^42]. We follow [^42] and train a convolutional autoencoder with an adversarial objective from scratch. By scaling up the training compute and using 16 latent channels, we improve the reconstruction capabilities compared to related models; see Table˜1. Furthermore, FLUX.1 is built from a mix of double stream and single stream [^38] blocks. Double stream blocks employ separate weights for image and text tokens, and mixing is done by applying the attention operation over the concatenation of tokens. After passing the sequences through the double stream blocks, we concatenate them and apply 38 single stream blocks to the image and text tokens. Finally, we discard the text tokens and decode the image tokens.

To improve GPU utilization of single stream blocks, we leverage fused feed-forward blocks inspired by [^8], which i) reduce the number of modulation parameters in a feedforward block by a factor of 2 and ii) fuse the attention input- and output linear layers with that of the MLP, leading to larger matrix-vector multiplications and thus more efficient training and inference. We utilize factorized three–dimensional Rotary Positional Embeddings (3D RoPE) [^53]. Every latent token is indexed by its space-time coordinates $(t,h,w)$ (with $t\equiv 0$ for single image inputs). See Figure˜3 for a visualization.

| Model | PDist $\downarrow$ | SSIM $\uparrow$ | PSNR $\uparrow$ |
| --- | --- | --- | --- |
| Flux-VAE | 0.332 $\pm$ 0.003 | 0.896 $\pm$ 0.004 | 31.1 $\pm$ 0.08 |
| SD3-VAE [^12] | 0.452 $\pm$ 0.004 | 0.858 $\pm$ 0.005 | 29.6 $\pm$ 0.07 |
| SD3-TAE <sup>3</sup> | 0.746 $\pm$ 0.004 | 0.774 $\pm$ 0.014 | 27.9 $\pm$ 0.06 |
| SDXL-VAE [^40] | 0.890 $\pm$ 0.005 | 0.748 $\pm$ 0.006 | 25.9 $\pm$ 0.07 |
| SD-VAE <sup>4</sup> | 0.949 $\pm$ 0.005 | 0.720 $\pm$ 0.004 | 25.0 $\pm$ 0.07 |

Table 1: Reconstruction quality comparison across different VAE architectures. All metrics computed on 4096 ImageNet images. Values are mean ± standard error (rounded). See also Appendix˜B.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/kontext_v2.jpg)

Figure 4: High-level overview of FLUX.1 Kontext, with input and context image on the left. Details in Section ˜ 3.

## 3 FLUX.1 Kontext

Our goal is to learn a model that can generate images conditioned jointly on a text prompt and a reference images. More formally, we aim to approximate the conditional distribution

$$
p(x\mid y,c)
$$

where $x$ is the target image, $y$ is a context image (or $\varnothing$), and $c$ is a natural-language instruction. Unlike classic text-to-image generation, this objective entails learning *relations between images themselves*, mediated by $c$, so that the same network can >i) perform image-driven edits when $y\neq\varnothing$, and (ii) create new content from scratch when $y=\varnothing$.

To that end, let $x\in\mathcal{X}$ be an output (target) image, $y\in\mathcal{X}\cup\{\varnothing\}$ an optional *context* image, and $c\in\mathcal{C}$ a text prompt. We model the conditional distribution $p_{\theta}(x\mid y,c)$ such that the same network handles *in-context and local edits* when $y\neq\varnothing$ and free *text-to-image generation* when $y=\varnothing$. Training starts from a FLUX.1 text-to-image checkpoint, and we collect and curate millions of relational pairs $(x\,|\,y,c)$ for optimization. In practice, we do not model images in pixel space but instead encode them into a token sequence as discussed in the following paragraph.

#### Token sequence construction.

Images are encoded into latent tokens by the frozen FLUX auto-encoder. These context image tokens $y$ are then appended to the image tokens $x$ and fed into the visual stream of the model. This simple *sequence concatenation* (i) supports different input/output resolutions and aspect ratios, and (ii) readily extends to multiple images $y_{1},y_{2},\dots,y_{N}$. Channel-wise concatenation of $x$ and $y$ was also tested but in initial experiments we found this design choice to perform worse.

We encode positional information via 3D RoPE embeddings, where the embeddings for the context $y$ receive a constant offset for all context tokens. We treat the offset as a *virtual time step* that cleanly separates the context and target blocks while leaving their internal spatial structure intact. Concretely, if a token position is denoted by the triplet $\mathbf{u}=(t,h,w)$, then we set $\mathbf{u}_{x}=(0,h,w)$ for the target tokens and for context tokens we set

$$
\mathbf{u}_{y_{i}}\;=\;(\,i,\,h,\,w\,),\qquad i=1,\dots,N,
$$

#### Rectified-flow objective.

We train with a rectified flow–matching loss

$$
\mathcal{L}_{\theta}=\mathbb{E}_{\,t\sim p(t),\,x,y,c}\bigl{[}\,\lVert v_{\theta}(z_{t},t,y,c)-(\varepsilon-x)\rVert_{2}^{2}\bigr{]},
$$

where $z_{t}$ is the linearly interpolated latent between $x$ and noise $\varepsilon\sim\mathcal{N}(0,1)$; $z_{t}=(1-t)x+t\varepsilon$. We use a logit normal shift schedule (see Section˜A.2) for $p(t;\mu,\sigma=1.0)$, where we change the mode $\mu$ depending on the resolution of the data during training. When sampling pure text–image pairs ($y=\varnothing$) we omit all tokens $y$, preserving the text-to-image generation capability of the model.

#### Adversarial Diffusion Distillation

Sampling of a flow matching model obtained by optimizing Equation˜3 typically involves solving an ordinary or stochastic differential equation [^30] [^1], using 50–250 guided [^16] network evaluations. While samples obtained through such a procedure are of good quality for a well-trained model $v_{\Theta}$, this comes with a few potential drawbacks: First, such multi-step sampling is slow, rendering model-serving at scale expensive and hindering low-latency, interactive applications. Moreover, guidance may occasionally introduce visual artifacts such as over-saturated samples. We tackle both challenges using latent *adversarial diffusion distillation* (LADD) [^47] [^48] [^49], reducing the number of sampling steps while increasing the quality of the samples through adversarial training.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/cc/sref1/1.jpg)

(a) Input image

#### Implementation details.

Starting from a pure text-to-image checkpoint, we jointly fine-tune the model on image-to-image and text-to-image tasks following Equation˜3. While our formulation naturally covers multiple input images, we focus on single context images for conditioning at this time. *FLUX.1 Kontext* \[pro\] is trained with the flow objective followed by LADD [^48]. We obtain *FLUX.1 Kontext* \[dev\] through guidance-distillation into a 12B diffusion transformer following the techniques outlined in [^34]. To optimize *FLUX.1 Kontext* \[dev\] performance on edit tasks, we focus exclusively on image-to-image training, i.e. do not train on the pure text-to-image task for *FLUX.1 Kontext* \[dev\]. We incorporate safety training measures including classifier-based filtering and adversarial training to prevent the generation of non-consensual intimate imagery (NCII) and child sexual abuse material (CSAM).

We use FSDP2 [^29] with mixed precision: all-gather operations are performed in bfloat16 while gradient reduce-scatter uses float32 for improved numerical stability. We use selective activation checkpointing [^26] to reduce maximum VRAM usage. To improve throughput, we use *Flash Attention 3* [^50] and regional compilation of individual Transformer blocks.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/cc/skirt/1.png)

(a) Input image

## 4 Evaluations & Applications

In this section, we evaluate *FLUX.1 Kontext* ’s performance and demonstrate its capabilities. We first introduce KontextBench, a novel benchmark featuring real-world image editing challenges crowd-sourced from users. We then present our primary evaluation: a systematic comparison of *FLUX.1 Kontext* against state-of-the-art text-to-image and image-to-image synthesis methods, where we demonstrate competitive performance across diverse editing tasks. Finally, we explore *FLUX.1 Kontext* ’s practical applications, including iterative editing workflows, style transfer, visual cue editing, and text editing.

### 4.1 KontextBench – Crowd-sourced Real-World Benchmark for In-Context Tasks

Existing benchmarks for editing models are often limited when it comes to capturing real-world usage. InstructPix2Pix [^5] relies on synthetic Stable Diffusion samples and GPT-generated instructions, creating inherent bias. MagicBrush [^60], while using authentic MS-COCO images, is constrained by DALLE-2’s [^41] capabilities during data collection. Other benchmarks like Emu-Edit [^51] use lower-resolution images with unrealistic distributions and focus solely on editing tasks, while DreamBench [^39] lacks broad coverage and GEdit-bench [^31] does not represent the full scope of modern multimodal models. IntelligentBench [^9] remains unavailable with only 300 examples of uncertain task coverage.

To address these gaps, we compile *KontextBench* from crowd-sourced real-world use cases. The benchmark comprises $1026$ unique image-prompt pairs derived from 108 base images including personal photos, CC-licensed art, public domain images, and AI-generated content. It spans five core tasks: local instruction editing (416 examples), global instruction editing (262), text editing (92), style reference (63), and character reference (193). We found that the scale of the benchmark provides a good balance between reliable human evaluation and comprehensive coverage of real-world applications. We will publish this benchmark including the samples of *FLUX.1 Kontext* and all reported baselines.

### 4.2 State-of-the-Art Comparison

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/x1.png)

(a) Text-to-image inference latency

*FLUX.1 Kontext* is designed to perform both text-to-image (T2I) and image-to-image (I2I) synthesis. We evaluate our approach against the strongest proprietary and open-weight models in both domains. We evaluate FLUX.1 Kontext \[pro\] and \[dev\]. As stated above, for \[dev\] we exclusively focus on image-to-image tasks. Additionally, we introduce FLUX.1 Kontext \[max\], which uses more compute to improve generative performance.

Image-to-Image Results. For image editing evaluation, we assess performance across multiple editing tasks: image quality, local editing, *character reference* (CREF), *style reference* (SREF), text editing, and computational efficiency. CREF enables consistent generation of specific characters or objects across novel settings, whereas SREF allows style transfer from reference images while maintaining semantic control. We compare different APIs and find that our models offer the fastest latency (cf. Figure˜7(b)), outperforming related models by up to an order of magnitude in speed difference. In our human evaluation (Figure˜8), we find that FLUX.1 Kontext \[max\] and \[pro\] are the best solution in the categories local and text editing, and for general CREF. We also calculate quantitative scores for CREF, to asses changes in facial characteristics between input and output images we use AuraFace <sup>5</sup> to extract facial embeddings before and after and edit and compare both, see Figure˜8(f). In alignment with our human evaluations, FLUX.1 Kontext outperforms all other models. For global editing and SREF, FLUX.1 Kontext is second only to gpt-image-1, and Gen-4 References, respectively.

Overall, *FLUX.1 Kontext* offers state-of-the-art character consistency, and editing capabilities, while outperforming competing models such as GPT-Image-1 by up to an order of magnitude in speed.

Text-to-Image Results. Current T2I benchmarks predominantly focus on general preference, typically asking questions like *“which image do you prefer?”*. We observe that this broad evaluation criterion often favors a characteristic “AI aesthetic” meaning over-saturated colors, excessive focus on central subjects, pronounced bokeh effects, and convergence toward homogeneous styles. We term this phenomenon bakeyness. To address this limitation, we decompose T2I evaluation into five distinct dimensions: prompt following, aesthetic (*“which image do you find more aesthetically pleasing”*), realism (*“which image looks more real”*), typography accuracy, and inference speed. We evaluate on 1 000 diverse test prompts compiled from academic benchmarks (DrawBench [^46], PartiPrompts [^59]) and real user queries. We refer to this benchmark as Internal-T2I-Bench in the following. In addition, we complement this benchmark with additional evaluations on GenAI bench [^28].

In T2I, FLUX.1 Kontext demonstrates balanced performance across evaluation categories (see Figure˜9). Although competing models excel in certain domains, this often comes at the expense of other categories. For instance, Recraft delivers strong aesthetic quality but limited prompt adherence, whereas GPT-Image-1 shows the inverse overall performance pattern. *FLUX.1 Kontext* consistently improves performance across categories over its predecessor FLUX1.1 \[pro\]. We also observe progressive gains from FLUX.1 Kontext \[pro\] to FLUX.1 Kontext \[max\]. We highlight samples in Fig. 14.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/x3.png)

(a) Text Editing

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/x9.png)

(a) Aesthetics (Internal-T2I-Bench)

### 4.3 Iterative Workflows

Maintaining character and object consistency across multiple edits is crucial for brand-sensitive and storytelling applications. Current state-of-the-art approaches suffer from noticeable visual drift: characters lose identity and objects lose defining features with each edit. In Fig. 12, we demonstrate character identity drift across edit sequences produced by *FLUX.1 Kontext*, Gen-4, and GPT-Image-high. We additionally compute the cosine similarity of AuraFace [^10] [^22] embeddings between the input and images generated via successive edits, highlighting the slower drift of *FLUX.1 Kontext* relative to competing methods. Consistency is essential: marketing needs stable brand characters, media production demands asset continuity, and e-commerce must preserve product details. Applications enabled by *FLUX.1 Kontext* ’s reliable consistency are shown in Fig. 10 and Fig. 11.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/cc/vase/img1.png)

(a) Input image

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/cc/laugh/img1.jpg)

(a) Input image

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/evals/qualitative/dustin_kontext_montage_final.jpg)

Figure 12: Iterative editing based on the same starting image with the same prompts and different models (top: FLUX.1 Kontext, middle: gpt-image-1, bottom: Runway Gen4). Below are face similarity scores between the edited images and the edited images at different steps. For the last edit ("Add sunglasses"), a relative drop is expected due to partial exclusion of the face.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/cc/text_edit/v1.jpg)

(a) Input image

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/t2icherries.jpg)

Figure 14: Text-to-image samples by FLUX.1 Kontext with low bakeyness, diverse styles, and accurate typography.

### 4.4 Specialized Applications

*FLUX.1 Kontext* supports several applications beyond standard generation. Style reference (SREF), first popularized by Midjourney [^35] and commonly implemented via IP-Adapters [^58], enables style transfer from reference images while maintaining semantic control (see Section 4.2). Additionally, the model supports intuitive editing through visual cues, responding to geometric markers like red ellipses to guide targeted modifications. It also provides sophisticated text editing capabilities, including logo refinement, spelling corrections, and style adaptations while preserving surrounding context. We demonstrate style reference in Fig. 5 and visual cue-based editing in Fig. 13.

## 5 Discussion

We introduced *FLUX.1 Kontext*, a flow matching model that combines in-context image generation and editing in a single framework. Through simple sequence concatenation and training recipes, *FLUX.1 Kontext* achieves state-of-the-art performance while addressing key limitations such as character drift during multi-turn edits, slow inference, and low output quality. Our contributions include a unified architecture that handles multiple processing tasks, superior character consistency across iterations, interactive speed, and KontextBench: A real-world benchmark with 1 026 image-prompt pairs. Our extensive evaluations reveal that *FLUX.1 Kontext* is comparable to proprietary systems while enabling fast, multi-turn creative workflows.

Limitations. *FLUX.1 Kontext* exhibits a few limitations in its current implementation. Excessive multi-turn editing can introduce visual artifacts that degrade image quality, see Figure˜15. The model occasionally fails to follow instructions accurately, ignoring specific prompt requirements. In addition, the distillation process can introduce visual artifacts that impact the fidelity of the output.

Future work should focus on extending to multiple image inputs, further scaling, and reducing inference latency to unlock real-time applications. A natural extension of our approach is to include edits in the video domain. Most importantly, reducing degradation during multi-turn editing would enable infinitely fluid content creation. The release of *FLUX.1 Kontext* and KontextBench provides a solid foundation and a comprehensive evaluation framework to drive unified image generation and editing.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2506.15742/assets/img/failure/one/1.jpg)

(a) Input image

## Appendix A Image Generation using Flow Matching

### A.1 Primer on Rectified Flow Matching

For training our models, we construct forward noising processes in the latent space of an image autoencoder as

$$
z_{t}=a_{t}x_{0}+b_{t}\varepsilon,
$$

with $x_{0}\sim p_{data}$, $\varepsilon\sim\mathcal{N}(0,1)$, and the coefficients $a_{t}$ and $b_{t}$ define the log signal-to-noise ratio (log-SNR) [^25]

$$
\lambda_{t}=\log\frac{a_{t}^{2}}{b_{t}^{2}}
$$

Further, we use the conditional flow matching loss [^30]

$$
\mathcal{L}_{\text{CFM}}=\mathbb{E}_{t\sim p(t),\varepsilon\sim\mathcal{N}(0,1)}||v_{\Theta}(z_{t},t)-\frac{a_{t}^{\prime}}{a_{t}}z_{t}+\frac{b_{t}}{2}\lambda_{t}^{\prime}\varepsilon||_{2}^{2}
$$

For rectified flow models [^32], $a_{t}=1-t$ and $b_{t}=t$, and thus

$$
\mathcal{L}_{\text{CFM}}=\mathbb{E}_{t\sim p(t),\varepsilon\sim\mathcal{N}(0,1),x_{0}\sim p_{data}}||v_{\Theta}(z_{t},t)+x_{0}-\varepsilon||_{2}^{2}
$$

and we sample $t$ from a *Logit-Normal Distribution* [^12]: $p(t)=\frac{\exp{(-0.5\cdot(\mathrm{logit}(t)-\mu)^{2}/\sigma^{2})}}{\sigma\sqrt{2\pi}\cdot(1-t)\cdot t}$, where $\mathrm{logit}(t)=\log\frac{t}{1-t}$. From the definition of the Logit-Normal Distribution, it follows that a random variable $Y=\mathrm{logit}(t)\sim\mathcal{N}(\mu,\sigma)$.

### A.2 Expressing shifting of the timestep schedule via the Logit-Normal Distribution

Previous work on high-resolution image synthesis introduced an additional shift of the timestep sampling (and, equivalently, the log-SNR schedule) via a parameter $\alpha$ [^12] [^18]. [^12] empirically demonstrated that $\alpha=3.0$ worked best when increasing the image resolution from $256^{2}$ to $1024^{2}$. In the following, we show that this shifting can be expressed via the Logit-Normal Distribution.

Consider the log-SNR of a rectified flow forward process with $\mu=0$ and $\sigma=1$:

$$
\lambda_{t}^{0,1}=2\log\frac{1-t}{t}=-2\mathrm{logit}(t),
$$

where $\mathrm{logit}(t)\sim\mathcal{N}(0,1)$. Expressing the log-SNR for arbitrary $\mu$ and $\sigma$ gives

$$
\lambda_{t}^{\mu,\sigma}=-2(\sigma\cdot\text{logit}(t)+\mu)=\sigma\cdot\lambda_{t}^{0,1}-2\mu\,.
$$

The $\alpha$ -shifted log-SNR [^12] [^18] is obtained as

$$
\lambda_{t}^{\alpha}=\lambda_{t}^{0,1}-2\log\alpha\,.
$$

Comparing Equation˜9 and Equation˜10, we identify $\mu=\log\alpha$ for $\sigma=1.0$, i.e. a shift of $\alpha=3.0$ would correspond to a logit-normal distribution with $\mu=\log 3.0=1.0986$ and $\sigma=1.0$.

We can further express the shifted log-SNR as a function of shifted timesteps $t^{\prime}$

$$
\lambda_{t^{\prime}}=2\log\frac{1-t^{\prime}}{t^{\prime}}=\sigma\lambda_{t}^{0,1}-2\mu=2\sigma\log\frac{1-t}{t}-2\mu
$$

and solve for $t^{\prime}$:

$$
t^{\prime}=\frac{e^{\mu}}{e^{\mu}+(1/t-1)^{\sigma}}
$$

For $\sigma=1.0$ and $\mu=\log\alpha$ this recovers the redistribution function for the timesteps proposed in [^12] $t^{\prime}=\frac{\alpha t}{1+(\alpha-1)t}$, as expected. This generalized shifting formula 9 can be useful both for training and via 12 for inference.

## Appendix B VAE Evaluation Details

We compare our VAE with related models using three reconstruction metrics, namely, SSIM, PSNR, and the *Perceptual Distance* (PDist) in VGG [^52] feature space. All metrics are computed over 4 096 random ImageNet evaluation images at resolution $256\times 256$. Table˜1 shows the mean and the standard deviation over the 4 096 inputs.

[^1]: Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants, 2022.

[^2]: James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. *Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf*, 2(3), 2023.

[^3]: Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas Müller, and Björn Ommer. Retrieval-augmented diffusion models. *Advances in Neural Information Processing Systems*, 35:15309–15324, 2022.

[^4]: Frederic Boesel and Robin Rombach. Improving image editing models with generative data refinement. In *Tiny Papers @ ICLR*, 2024. URL [https://api.semanticscholar.org/CorpusID:271461432](https://api.semanticscholar.org/CorpusID:271461432).

[^5]: Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18392–18402, 2023.

[^6]: Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

[^7]: Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. *arXiv preprint arXiv:2209.14491*, 2022.

[^8]: Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In *International Conference on Machine Learning*, pages 7480–7512. PMLR, 2023.

[^9]: Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. *arXiv preprint arXiv:2505.14683*, 2025.

[^10]: Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4690–4699, 2019.

[^11]: Patrick Esser, Robin Rombach, Andreas Blattmann, and Bjorn Ommer. Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis. *Advances in neural information processing systems*, 34:3518–3532, 2021.

[^12]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL [https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206).

[^13]: Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. *arXiv preprint arXiv:2208.01618*, 2022.

[^14]: Rafael C Gonzalez. *Digital image processing*. Pearson education india, 2009.

[^15]: HiDream-ai. Hidream-e1: Instruction-based image editing model, 2025. URL [https://github.com/HiDream-ai/HiDream-E1](https://github.com/HiDream-ai/HiDream-E1).

[^16]: Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022.

[^17]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.

[^18]: Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. Simple diffusion: End-to-end diffusion for high resolution images, 2023.

[^19]: Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

[^20]: Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. *arXiv preprint arXiv:2410.23775*, 2024.

[^21]: Imagen-Team-Google,:, Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Lluis Castrejon, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, Zach Eaton-Rosen, Hongliang Fei, Nando de Freitas, Yilin Gao, Evgeny Gladchenko, Sergio Gómez Colmenarejo, Mandy Guo, Alex Haig, Will Hawkins, Hexiang Hu, Huilian Huang, Tobenna Peter Igwe, Christos Kaplanis, Siavash Khodadadeh, Yelin Kim, Ksenia Konyushkova, Karol Langner, Eric Lau, Rory Lawton, Shixin Luo, Soňa Mokrá, Henna Nandwani, Yasumasa Onoe, Aäron van den Oord, Zarana Parekh, Jordi Pont-Tuset, Hang Qi, Rui Qian, Deepak Ramachandran, Poorva Rane, Abdullah Rashwan, Ali Razavi, Robert Riachi, Hansa Srinivasan, Srivatsan Srinivasan, Robin Strudel, Benigno Uria, Oliver Wang, Su Wang, Austin Waters, Chris Wolff, Auriel Wright, Zhisheng Xiao, Hao Xiong, Keyang Xu, Marc van Zee, Junlin Zhang, Katie Zhang, Wenlei Zhou, Konrad Zolna, Ola Aboubakar, Canfer Akbulut, Oscar Akerlund, Isabela Albuquerque, Nina Anderson, Marco Andreetto, Lora Aroyo, Ben Bariach, David Barker, Sherry Ben, Dana Berman, Courtney Biles, Irina Blok, Pankil Botadra, Jenny Brennan, Karla Brown, John Buckley, Rudy Bunel, Elie Bursztein, Christina Butterfield, Ben Caine, Viral Carpenter, Norman Casagrande, Ming-Wei Chang, Solomon Chang, Shamik Chaudhuri, Tony Chen, John Choi, Dmitry Churbanau, Nathan Clement, Matan Cohen, Forrester Cole, Mikhail Dektiarev, Vincent Du, Praneet Dutta, Tom Eccles, Ndidi Elue, Ashley Feden, Shlomi Fruchter, Frankie Garcia, Roopal Garg, Weina Ge, Ahmed Ghazy, Bryant Gipson, Andrew Goodman, Dawid Górny, Sven Gowal, Khyatti Gupta, Yoni Halpern, Yena Han, Susan Hao, Jamie Hayes, Jonathan Heek, Amir Hertz, Ed Hirst, Emiel Hoogeboom, Tingbo Hou, Heidi Howard, Mohamed Ibrahim, Dirichi Ike-Njoku, Joana Iljazi, Vlad Ionescu, William Isaac, Reena Jana, Gemma Jennings, Donovon Jenson, Xuhui Jia, Kerry Jones, Xiaoen Ju, Ivana Kajic, Christos Kaplanis, Burcu Karagol Ayan, Jacob Kelly, Suraj Kothawade, Christina Kouridi, Ira Ktena, Jolanda Kumakaw, Dana Kurniawan, Dmitry Lagun, Lily Lavitas, Jason Lee, Tao Li, Marco Liang, Maggie Li-Calis, Yuchi Liu, Javier Lopez Alberca, Matthieu Kim Lorrain, Peggy Lu, Kristian Lum, Yukun Ma, Chase Malik, John Mellor, Thomas Mensink, Inbar Mosseri, Tom Murray, Aida Nematzadeh, Paul Nicholas, Signe Nørly, João Gabriel Oliveira, Guillermo Ortiz-Jimenez, Michela Paganini, Tom Le Paine, Roni Paiss, Alicia Parrish, Anne Peckham, Vikas Peswani, Igor Petrovski, Tobias Pfaff, Alex Pirozhenko, Ryan Poplin, Utsav Prabhu, Yuan Qi, Matthew Rahtz, Cyrus Rashtchian, Charvi Rastogi, Amit Raul, Ali Razavi, Sylvestre-Alvise Rebuffi, Susanna Ricco, Felix Riedel, Dirk Robinson, Pankaj Rohatgi, Bill Rosgen, Sarah Rumbley, Moonkyung Ryu, Anthony Salgado, Tim Salimans, Sahil Singla, Florian Schroff, Candice Schumann, Tanmay Shah, Eleni Shaw, Gregory Shaw, Brendan Shillingford, Kaushik Shivakumar, Dennis Shtatnov, Zach Singer, Evgeny Sluzhaev, Valerii Sokolov, Thibault Sottiaux, Florian Stimberg, Brad Stone, David Stutz, Yu-Chuan Su, Eric Tabellion, Shuai Tang, David Tao, Kurt Thomas, Gregory Thornton, Andeep Toor, Cristian Udrescu, Aayush Upadhyay, Cristina Vasconcelos, Alex Vasiloff, Andrey Voynov, Amanda Walker, Luyu Wang, Miaosen Wang, Simon Wang, Stanley Wang, Qifei Wang, Yuxiao Wang, Ágoston Weisz, Olivia Wiles, Chenxia Wu, Xingyu Federico Xu, Andrew Xue, Jianbo Yang, Luo Yu, Mete Yurtoglu, Ali Zand, Han Zhang, Jiageng Zhang, Catherine Zhao, Adilet Zhaxybay, Miao Zhou, Shengqi Zhu, Zhenkai Zhu, Dawn Bloxwich, Mahyar Bordbar, Luis C. Cobo, Eli Collins, Shengyang Dai, Tulsee Doshi, Anca Dragan, Douglas Eck, Demis Hassabis, Sissie Hsiao, Tom Hume, Koray Kavukcuoglu, Helen King, Jack Krawczyk, Yeqing Li, Kathy Meier-Hellstern, Andras Orban, Yury Pinsky, Amar Subramanya, Oriol Vinyals, Ting Yu, and Yori Zwols. Imagen 3, 2024. URL [https://arxiv.org/abs/2408.07009](https://arxiv.org/abs/2408.07009).

[^22]: isidentical. Introducing auraface: Open-source face recognition and identity preservation models. [https://huggingface.co/blog/isidentical/auraface](https://huggingface.co/blog/isidentical/auraface), 2024. Accessed: 2025-05-26.

[^23]: Kat Kampf and Nicole Brichtova. Experiment with gemini 2.0 flash native image generation, 2025. URL [https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/).

[^24]: Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models, 2023. URL [https://arxiv.org/abs/2210.09276](https://arxiv.org/abs/2210.09276).

[^25]: Diederik P Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

[^26]: Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. *Proceedings of Machine Learning and Systems*, 5:341–353, 2023.

[^27]: Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 1931–1941, 2023.

[^28]: Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, et al. Genai-bench: Evaluating and improving compositional text-to-visual generation. *arXiv preprint arXiv:2406.13743*, 2024.

[^29]: Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, et al. Torchtitan: One-stop pytorch native solution for production ready llm pre-training. *arXiv preprint arXiv:2410.06511*, 2024.

[^30]: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?id=PqvMRDCJT9t](https://openreview.net/forum?id=PqvMRDCJT9t).

[^31]: Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. *arXiv preprint arXiv:2504.17761*, 2025.

[^32]: Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022.

[^33]: Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11461–11471, 2022.

[^34]: Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14297–14306, 2023.

[^35]: Midjourney. Midjourney, 2025. URL [https://www.midjourney.com/home](https://www.midjourney.com/home).

[^36]: OpenAI. Introducing 4o image generation, 2025. URL [https://openai.com/index/introducing-4o-image-generation/](https://openai.com/index/introducing-4o-image-generation/).

[^37]: Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In *ACM SIGGRAPH 2023 conference proceedings*, pages 1–11, 2023.

[^38]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*. IEEE, 2023. doi: 10.1109/iccv51070.2023.00387. URL [http://dx.doi.org/10.1109/ICCV51070.2023.00387](http://dx.doi.org/10.1109/ICCV51070.2023.00387).

[^39]: Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation, 2025. URL [https://arxiv.org/abs/2406.16855](https://arxiv.org/abs/2406.16855).

[^40]: Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.

[^41]: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022.

[^42]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, 2022. doi: 10.1109/cvpr52688.2022.01042. URL [http://dx.doi.org/10.1109/CVPR52688.2022.01042](http://dx.doi.org/10.1109/CVPR52688.2022.01042).

[^43]: Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. URL [https://arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242).

[^44]: Inc. Runway AI. Runway | tools for human imagination, 2025. URL [https://runwayml.com/](https://runwayml.com/).

[^45]: Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In *ACM SIGGRAPH 2022 Conference Proceedings*, pages 1–10, 2022a.

[^46]: Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022b.

[^47]: Axel Sauer, Kashyap Chitta, Jens Müller, and Andreas Geiger. Projected gans converge faster. *Advances in Neural Information Processing Systems*, 2021.

[^48]: Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. *arXiv preprint arXiv:2311.17042*, 2023.

[^49]: Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation, 2024. URL [https://arxiv.org/abs/2403.12015](https://arxiv.org/abs/2403.12015).

[^50]: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. *Advances in Neural Information Processing Systems*, 37:68658–68685, 2024.

[^51]: Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. *arXiv preprint arXiv:2311.10089*, 2023.

[^52]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.

[^53]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

[^54]: Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In *Proceedings of the IEEE/CVF winter conference on applications of computer vision*, pages 2149–2159, 2022.

[^55]: Richard Szeliski. *Computer vision: algorithms and applications*. Springer Nature, 2022.

[^56]: Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. *arXiv preprint arXiv:2409.11340*, 2024.

[^57]: Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 18381–18391, 2023.

[^58]: Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv preprint arXiv:2308.06721*, 2023.

[^59]: Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation, 2022. URL [https://arxiv.org/abs/2206.10789](https://arxiv.org/abs/2206.10789).

[^60]: Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing, 2024. URL [https://arxiv.org/abs/2306.10012](https://arxiv.org/abs/2306.10012).

[^61]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 3836–3847, 2023.

[^62]: Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. *arXiv preprint arXiv:2504.20690*, 2025.