---
title: "Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation"
source: "https://arxiv.org/html/2604.24763v1"
author:
published: 2026-04-27
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/02_%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B/Tuna-2%2C%20Pixel%20Embeddings%20Beat%20Vision%20Encoders%20for%20Multimodal%20Understanding%20and%20Generation%2C%202026.no_watermark.zh-CN.dual.pdf"
---
1\]Meta AI 2\]The University of Hong Kong 3\]University of Waterloo \[\*\]Joint first authors, listed alphabetically by last name

Zhiheng Liu    Weiming Ren    Xiaoke Huang    Shoufa Chen    Tianhong Li    Mengzhao Chen    Yatai Ji    Sen He    Jonas Schult    Belinda Zeng    Tao Xiang    Wenhu Chen    Ping Luo    Luke Zettlemoyer    Yuren Cong \[ \[ \[

###### Abstract

Unified multimodal models typically rely on pretrained vision encoders and use separate visual representations for understanding and generation, creating misalignment between the two tasks and preventing fully end-to-end optimization from raw pixels. We introduce Tuna-2, a native unified multimodal model that performs visual understanding and generation directly based on pixel embeddings. Tuna-2 drastically simplifies the model architecture by employing simple patch embedding layers to encode visual input, completely discarding the modular vision encoder designs such as the VAE or the representation encoder. Experiments show that Tuna-2 achieves state-of-the-art performance in multimodal benchmarks, demonstrating that unified pixel-space modelling can fully compete with latent-space approaches for high-quality image generation. Moreover, while the encoder-based variant converges faster in early pretraining, Tuna-2’s encoder-free design achieves stronger multimodal understanding at scale, particularly on tasks requiring fine-grained visual perception. These results show that pretrained vision encoders are not necessary for multimodal modelling, and end-to-end pixel-space learning offers a scalable path toward stronger visual representations for both generation and perception.

\[Project page\] [https://tuna-ai.org/tuna-2](https://tuna-ai.org/tuna-2)

![Refer to caption](https://arxiv.org/html/2604.24763v1/x1.png)

Figure 1: Evolution of Tuna-2 architecture and multimodal performance comparison. We simplify Tuna ( liu2025tuna ) by progressively stripping away its visual encoding components. By removing the VAE, we first derive Tuna-R, a pixel-space UMM that relies solely on a representation encoder. further streamlines the design by bypassing the representation encoder entirely, utilizing direct patch embedding layers for raw image inputs. using pixel embeddings outperforms both and across a diverse suite of multimodal benchmarks.

## 1 Introduction

Visual understanding and generation are two core capabilities in multimodal AI. Recent work has increasingly focused on native unified multimodal models (UMMs) (zhou2024transfusion; deng2025emerging; liu2025tuna), which aim to integrate both capabilities within a single framework. A central challenge in building such models is encoding input images into visual representations that effectively support both understanding and generation. Early approaches (deng2025emerging; chen2025janus) adopted decoupled representations, using representation encoders such as CLIP (radford2021learning) for understanding and reconstruction-oriented encoders such as VQ-VAE (esser2021taming) for generation. To address the representation mismatch introduced by this design, more recent UMMs (xie2025show; liu2025tuna) have moved toward modelling both tasks using unified visual representations through a shared vision encoder.

Despite the significant progress, both decoupled and unified visual representation designs still rely heavily on pretrained vision encoders (wan2025wan; tschannen2025siglip) for visual feature extraction. In parallel, recent research on multimodal understanding and generation has begun to move away from encoder-based modular designs toward simpler monolithic, encoder-free architectures. In multimodal understanding, newer native vision-language models (diao2025pixels) remove the pretrained representation encoder and instead align images and natural language within a unified, end-to-end architecture. In visual generation, pixel-space diffusion models (hoogeboom2023simple; chen2025pixelflow; li2025back) have shown increasing flexibility, stronger scalability, and state-of-the-art performance on a wide range of tasks, suggesting that pretrained VAE encoders may no longer be essential even for high-fidelity image synthesis.

Motivated by these observations, we ask a natural but largely unexplored question: can we move beyond pretrained vision encoders altogether, and build unified multimodal models through end-to-end native modelling directly from raw pixels?

We answer this question affirmatively by introducing Tuna-2, a native unified multimodal model that attempts to progressively simplify the encoder modules, and ultimately remove vision encoders completely. We first introduce Tuna-R, which eliminates the VAE model while keeping a representation encoder in the model architecture. Tuna-R performs multimodal understanding similar to standard encoder-based LMMs, and supports visual generation through pixel-space flow matching with an $x$ -prediction objective. We then propose Tuna-2, which further simplifies the architecture by removing the encoder entirely and using only a single transformer decoder to process image and video tokens. As a result, Tuna-2 enables end-to-end native unified modelling directly from raw pixels, without relying on any pretrained encoder modules.

Since learning unified representations directly in high-dimensional pixel space is substantially more challenging than learning them in a compact latent space, we further introduce a masking-based visual feature learning scheme to stabilize training and encourage the learning of more robust pixel-space representations. Together, these designs enable Tuna-2 to achieve state-of-the-art performance across a diverse set of multimodal understanding and generation benchmarks. More importantly, our controlled comparison reveals a clear design insight: after sufficient visual pretraining, the encoder-free Tuna-2 becomes competitive with the encoder-based Tuna-R on visual generation, while consistently outperforming it on multimodal understanding, especially on benchmarks that require fine-grained visual perception. These findings suggest that removing pretrained vision encoders can be advantageous for learning stronger fine-grained visual representations in end-to-end pretraining. As shown in Figures 1 and 2, this leads to highly competitive performances in both multimodal understanding and generation.

Our main contributions are summarized as follows:

- We propose Tuna-2, a native unified multimodal model that supports multimodal understanding and generation with encoder-free designs, achieving state-of-the-art performance across a wide range of understanding and generation benchmarks.
- We conduct controlled comparisons between Tuna-2 and an encoder-based pixel-space UMM variant Tuna-R, and show that after sufficient multimodal pretraining, Tuna-2 and its encoder-free design are competitive on generation and advantageous for understanding, especially on fine-grained, perception-intensive tasks.
- We conduct comprehensive ablations and analyses on pixel-space UMMs to study their training dynamics and model behaviours, offering useful insights for the development of future native unified multimodal models.

![Refer to caption](https://arxiv.org/html/2604.24763v1/x2.png)

Figure 2: While being completely encoder-free, Tuna-2 is capable of performing high-fidelity text-to-image generation and image editing.

## 2 Method

In this section, we present Tuna-2, a native unified multimodal model that performs visual understanding and generation both in pixel space. We start by detailing our approach to progressively remove vision encoder components to derive Tuna-2 in Section 2.1. We then describe our masked feature learning scheme in Section 2.2 and our model training pipeline in Section 2.3.

### 2.1 Towards Encoder-Free Unified Models

As shown in Figure 1, existing UMMs with unified visual representations, such as Tuna (liu2025tuna), typically consist of a vision encoder and an LLM decoder for joint vision-language modeling, followed by modality-specific heads, including a language modelling head for autoregressive text generation and a flow matching head for image generation. In this work, we propose Tuna-2 as an encoder-free UMM formulation by progressively simplifying the vision encoder components in existing architectures. Our design process for this architectural simplification is as follows:

Representation encoder-based architecture. First, we attempt to remove the VAE model and only employ a pretrained representation encoder in the vision encoder. As shown in Figure 1, this resonates a standard paradigm for vision-language modelling: the representation encoder first encodes input images into visual tokens, which are then combined with the text tokens in the LLM decoder for joint vision-language modelling. Originally proposed in LLaVA (liu2023visual), this paradigm has been verified and scaled up by later works such as Qwen3-VL (bai2025qwen3) and InternVL3.5 (wang2025internvl3), and remains the most popular framework for multimodal understanding. We refer to this intermediate design as Tuna-R. Although our ultimate goal is to move beyond encoder-based architectures, we view Tuna-R as an important intermediate step that enables a controlled comparison with Tuna-2.

Encoder-free (non-encoder) architecture. Second, we consider a further simplified architecture that removes the representation encoder entirely, which becomes our main design for Tuna-2. As shown in Figure 1, this design replaces pretrained vision encoders with simple patch embedding layers that convert images into visual tokens, which are then processed jointly with text tokens by the LLM decoder. Similar encoder-free designs have recently been explored in models such as Mono-InternVL (luo2025mono) and NEO (diao2025pixels). By removing the pretrained representation encoder, this design avoids its built-in inductive biases, such as fixed input resolutions and limited access to fine-grained low-level visual details. It also simplifies the model architecture into a single unified transformer. In Section 3, we present a series of in-depth analyses comparing Tuna-2 with Tuna-R, and demonstrate the effectiveness and scalability of Tuna-2.

Pixel-space image generation. Our VAE-free design allows us to directly perform multimodal understanding and text generation using the LLM decoder and the language modelling head. However, discarding the VAE also means that we can no longer adopt the designs from existing UMMs and generation-only models that follow the latent diffusion architecture. To effectively perform pixel-space image generation, we adopt the $x$ -prediction and $v$ -loss paradigm from JiT (li2025back) for pixel-space flow matching. Specifically, given the source image $x_{1}$, the sampled noise $x_{0}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and the timestamp $t$, we employ rectified flow and its linear schedule to construct a noisy sample in pixel space:

$$
x_{t}=tx_{1}+(1-t)x_{0},t\in[0,1].
$$

Tuna-2 is then formulated to directly predict the clean image from the noisy image in pixel space:

$$
x_{\theta}=\pi_{\theta}(x_{t},c,t),
$$

where $\pi_{\theta}$ is our unified model (vision-language backbone and flow matching head) and $c$ is the conditioning signals (text for text-to-image generation and text+image for image editing). As suggested in JiT, while our model directly predicts $x_{\theta}$, we still transform it into the velocity term $v_{\theta}$ and regress $v_{\theta}$ as our learning objective:

$$
\displaystyle v_{\theta}
$$
 
$$
\displaystyle=\frac{x_{\theta}-x_{t}}{1-t},
$$
$$
\displaystyle\mathcal{L}_{\mathrm{flow}}
$$
 
$$
\displaystyle=\mathbb{E}_{t,c,x_{1},x_{0}}||v_{\theta}-v||^{2}_{2},
$$

where $v$ is the ground truth velocity defined by $v=x_{1}-x_{0}$. During inference, we employ the Euler solver and predict the denoised image at $t^{\prime}$ from the noisier image at $t<t^{\prime}$ based on the velocity term $v_{\theta}$, such that $x_{t^{\prime}}=x_{t}+(t^{\prime}-t)v_{\theta}$, where $v_{\theta}$ is transformed from our model prediction $x_{\theta}$, based on Equation 3.

![Refer to caption](https://arxiv.org/html/2604.24763v1/x3.png)

Figure 3: Illustration of our proposed masking-based feature learning scheme. During training, we use the learnable mask token to regularize multimodal understanding and perform masked prediction for visual generation.

### 2.2 Learning Robust Visual Representations via Masking

While removing the VAE simplifies our model architecture and enables fully end-to-end unified multimodal training, it also shifts visual modelling from a compact latent space to the much higher-dimensional pixel space. As a result, learning a unified visual representation becomes more challenging: the increased redundancy in pixel-space inputs makes it easier for the model to rely on superficial shortcuts, rather than learning visual cues that are genuinely informative for both understanding and generation. To learn more robust visual representations in pixel space, we introduce a masking-based visual feature learning scheme. As shown in Figure 3, during training, we (optionally) randomly select a subset of image patches according to a masking ratio and replace the masked visual tokens with a learnable mask token before feeding them into the LLM decoder. The same masking operation is applied to both generation and understanding examples, but plays different roles in the two settings:

- For generation examples, we let the model predict the clean image patches in both the masked and the unmasked regions, such that (1) we create a harder denoising problem for the model to predict clean images from partially observed noisy images; and (2) it encourages the learnable mask token to absorb useful information for image reconstruction conditioned on the visible context.
- For understanding examples, our model predicts the ground truth text response based on the masked visual input. In this case, masking serves as a regularization mechanism that forces the model to perform multimodal reasoning under partial visual observation, leading to more robust visual representations.

Our masking-based feature learning scheme resembles masked modelling methods in visual understanding and generation, such as MAE (he2022masked) and SigLIP 2 (tschannen2025siglip) for semantic learning and MaskGIT (chang2022maskgit) and DeTok (yang2025latent) for visual generation. Empirically, we find that applying masking leads to enhanced model performance during pretraining stages.

### 2.3 Training Pipeline

Our encoder-free design enables fully end-to-end training of Tuna-2, without requiring separate stages to train connector layers, which is a common design in encoder-based modular approaches. As described below, our training pipeline consists of two stages, both of which are carried out in a fully end-to-end manner:

Stage 1: full model pretraining. In the first stage, we aim to establish a strong initialization for the flow matching head, and adapt pixel-space visual inputs for unified multimodal understanding and generation. To achieve this, we train the full model jointly on two tasks: image captioning and text-to-image generation.

Stage 2: supervised finetuning (SFT). Next, we perform supervised fine-tuning (SFT) of the full model with a lower learning rate. We use datasets for image editing, image instruction-following, and high-quality image generation. This step refines Tuna-2’s abilities, boosting performance and generalization across various multimodal tasks.

For Tuna-R, which includes a connector layer between the representation encoder and the LLM decoder, we add an extra alignment stage before Stage 1. In this stage, we train only the connector layer for a short period using image captioning and text-to-image generation data. As noted above, Tuna-2 does not require this additional stage because of its encoder-free design.

## 3 Experiments

### 3.1 Experiment Setup

We employ Qwen2.5-7B-Instruct (qwen2024qwen2) as the LLM decoder for Tuna-2. For Stage 1 pretraining, we use 550M in-house image-text pairs, consisting of 70% image captioning data for multimodal understanding and 30% text-to-image generation data. In addition, we include text-only data from Nemotron (bercovich2025llama), which accounts for 20% of the total pretraining data. The full model is trained end-to-end for 300k steps on 64 nodes with the AdamW optimizer (loshchilov2017decoupled) and a learning rate of $1\times 10^{-4}$. For Stage 2 supervised finetuning, we use a curated SFT corpus covering image instruction-following, image editing, and high-quality image generation. Specifically, for image instruction-following, we include 13M conversational examples from the open-source FineVision (wiedmann2025finevision) dataset. For image editing, we use approximately 2M examples from OmniEdit (wei2024omniedit). This stage is trained for 50k steps with AdamW and a learning rate of $2\times 10^{-5}$. For all training stages, we pad the input sequence length to 16k tokens per GPU.

For Tuna-R, we use the same Qwen2.5-7B-Instruct as the LLM decoder. We follow Tuna and adopt SigLIP 2 So400M (tschannen2025siglip) as the representation encoder. For the connector-alignment stage in Tuna-R, we train the model for 3k steps with AdamW and a learning rate of $5\times 10^{-4}$.

Table 1: Comparisons between Tuna-2 and baseline models on multimodal understanding benchmarks. Results with model size greater than 13B are grayed. Bold: best results among all UMMs. Underline: second-best among all UMMs.

<table><tbody><tr><th rowspan="2">Models</th><th rowspan="3">Size</th><td colspan="9">General Benchmarks</td><td colspan="3">Pixel-centric Benchmarks</td></tr><tr><td>GQA</td><td>RealWorldQA</td><td>MMVet</td><td>MMMU</td><td>MMVP</td><td>SEED-Bench2+</td><td>AI2D</td><td>ChartQA</td><td>OCRBench</td><td>V*</td><td>CountBench</td><td>VisuLogic</td></tr><tr><th colspan="14">[HTML]EFEFEFUnderstanding-only Models (LMMs)</th></tr><tr><th>LLaVA-1.5 <cite>(liu2023visual)</cite></th><th>7B</th><td>62.0</td><td>54.8</td><td>32.9</td><td>35.7</td><td>-</td><td>-</td><td>55.5</td><td>17.8</td><td>31.8</td><td>-</td><td>-</td><td>-</td></tr><tr><th>Qwen-VL-Chat <cite>(bai2023qwen)</cite></th><th>7B</th><td>57.5</td><td>49.3</td><td>47.3</td><td>37.0</td><td>-</td><td>-</td><td>57.7</td><td>49.8</td><td>48.8</td><td>-</td><td>-</td><td>-</td></tr><tr><th>LLaVA-OV <cite>(li2024llava)</cite></th><th>7B</th><td>-</td><td>69.9</td><td>51.9</td><td>48.8</td><td>77.3</td><td>62.2</td><td>81.4</td><td>80.9</td><td>62.2</td><td>72.7</td><td>76.2</td><td>24.8</td></tr><tr><th>Qwen2.5-VL <cite>(li2024llava)</cite></th><th>7B</th><td>60.7</td><td>69.9</td><td>61.7</td><td>58.6</td><td>78.0</td><td>70.5</td><td>82.7</td><td>83.0</td><td>83.7</td><td>71.2</td><td>74.1</td><td>20.0</td></tr><tr><th colspan="14">[HTML]EFEFEFComposite UMMs</th></tr><tr><th>TokenFlow-XL <cite>(qu2025tokenflow)</cite></th><th>14B</th><td>62.5</td><td>56.6</td><td>-</td><td>43.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>BLIP3-o <cite>(chen2025blip3)</cite></th><th>4B</th><td>-</td><td>60.4</td><td>-</td><td>46.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><th>Tar <cite>(han2025vision)</cite></th><th>7B</th><td>61.3</td><td>-</td><td>-</td><td>39.0</td><td>74.3</td><td>46.2</td><td>-</td><td>-</td><td>-</td><td>41.4</td><td>64.2</td><td>24.3</td></tr><tr><th>X-Omni <cite>(geng2025x)</cite></th><th>7B</th><td>62.8</td><td>62.6</td><td>-</td><td>47.2</td><td>-</td><td>-</td><td>76.8</td><td>81.5</td><td>70.4</td><td>-</td><td>-</td><td>-</td></tr><tr><th colspan="14">[HTML]EFEFEFNative UMMs</th></tr><tr><th>BAGEL <cite>(deng2025emerging)</cite></th><th>14B</th><td>66.4</td><td>72.8</td><td>67.2</td><td>55.3</td><td>85.0</td><td>71.9</td><td>89.2</td><td>78.5</td><td>73.3</td><td>70.2</td><td>82.5</td><td>41.7</td></tr><tr><th>Ming-UniVision <cite>(huang2025mingunivision)</cite></th><th>16B</th><td>59.4</td><td>59.1</td><td>64.2</td><td>40.3</td><td>71.0</td><td>56.8</td><td>82.8</td><td>76.7</td><td>72.4</td><td>48.2</td><td>76.8</td><td>26.7</td></tr><tr><th>Harmon <cite>(wu2025harmonizing)</cite></th><th>1.5B</th><td>58.9</td><td>49.8</td><td>-</td><td>38.9</td><td>61.7</td><td>41.6</td><td>57.0</td><td>29.8</td><td>11.2</td><td>41.9</td><td>67.0</td><td>25.1</td></tr><tr><th>JanusFlow <cite>(ma2025janusflow)</cite></th><th>1.3B</th><td>60.3</td><td>41.2</td><td>36.2</td><td>29.3</td><td>67.7</td><td>39.8</td><td>54.2</td><td>42.4</td><td>53.2</td><td>42.9</td><td>78.6</td><td>22.0</td></tr><tr><th>Emu3 <cite>(wang2024emu3)</cite></th><th>8B</th><td>60.3</td><td>57.4</td><td>23.5</td><td>31.6</td><td>71.0</td><td>44.6</td><td>70.0</td><td>69.4</td><td>68.7</td><td>53.4</td><td>65.2</td><td>24.7</td></tr><tr><th>VILA-U <cite>(wu2024vila)</cite></th><th>7B</th><td>60.8</td><td>46.8</td><td>26.3</td><td>31.2</td><td>62.7</td><td>31.9</td><td>49.0</td><td>11.4</td><td>23.3</td><td>38.7</td><td>55.2</td><td>25.4</td></tr><tr><th>Janus-Pro <cite>(chen2025janus)</cite></th><th>7B</th><td>62.0</td><td>58.0</td><td>41.1</td><td>41.0</td><td>73.3</td><td>56.3</td><td>71.3</td><td>25.8</td><td>59.0</td><td>47.6</td><td>53.2</td><td>23.8</td></tr><tr><th>Show-o2 <cite>(xie2025show)</cite></th><th>7B</th><td>63.1</td><td>64.7</td><td>39.6</td><td>48.9</td><td>76.7</td><td>61.3</td><td>78.6</td><td>52.3</td><td>32.4</td><td>44.5</td><td>63.5</td><td>26.9</td></tr><tr><th>OneCat <cite>(li2025onecat)</cite></th><th>9B</th><td>63.1</td><td>65.2</td><td>52.2</td><td>41.9</td><td>71.3</td><td>61.6</td><td>77.8</td><td>81.2</td><td>79.0</td><td>63.4</td><td>34.2</td><td>24.9</td></tr><tr><th>Tuna <cite>(liu2025tuna)</cite></th><th>7B</th><td>63.9</td><td>66.1</td><td>42.9</td><td>49.8</td><td>70.7</td><td>52.7</td><td>79.3</td><td>85.8</td><td>74.3</td><td>52.4</td><td>73.5</td><td>22.4</td></tr><tr><th>[HTML]ECF4FF Tuna-R</th><th>7B</th><td>63.5</td><td>67.9</td><td>46.7</td><td>51.1</td><td>74.7</td><td>58.4</td><td>79.4</td><td>85.6</td><td>78.3</td><td>57.6</td><td>77.8</td><td>26.2</td></tr><tr><th>[HTML]ECF4FF Tuna-2</th><th>7B</th><td>65.0</td><td>67.7</td><td>51.7</td><td>50.7</td><td>77.3</td><td>61.1</td><td>79.6</td><td>85.6</td><td>79.7</td><td>59.2</td><td>81.7</td><td>28.8</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2604.24763v1/x4.png)

Table 2: Image generation results on GenEval and DPG-Bench. “Col. Attr.” means “Color Attribute”. † refers to methods using LLM rewriters in GenEval. Bold: best results among native UMMs. Underline: second-best.