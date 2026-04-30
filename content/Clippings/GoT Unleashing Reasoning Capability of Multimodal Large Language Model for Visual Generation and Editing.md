---
title: "GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing"
source: "https://arxiv.org/html/2503.10639v1"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Rongyao Fang <sup>1</sup>   Chengqi Duan <sup>2</sup> <sup>∗</sup> Kun Wang <sup>3</sup> Linjiang Huang <sup>6</sup> Hao Li <sup>1,4</sup> Shilin Yan  
Hao Tian <sup>3</sup> Xingyu Zeng <sup>3</sup> Rui Zhao <sup>3</sup> Jifeng Dai <sup>4,5</sup> Xihui Liu <sup>2</sup> Hongsheng Li <sup>1</sup> <sup>†</sup>  
<sup>1</sup> CUHK MMLab <sup>2</sup> HKU <sup>3</sup> SenseTime <sup>4</sup> Shanghai AI Laboratory <sup>5</sup> THU <sup>6</sup> BUAA Equal ContributionCorresponding Authors

###### Abstract

Current image generation and editing methods primarily process textual prompts as direct inputs without reasoning about visual composition and explicit operations. We present Generation Chain-of-Thought (GoT), a novel paradigm that enables generation and editing through an explicit language reasoning process before outputting images. This approach transforms conventional text-to-image generation and editing into a reasoning-guided framework that analyzes semantic relationships and spatial arrangements. We define the formulation of GoT and construct large-scale GoT datasets containing over 9M samples with detailed reasoning chains capturing semantic-spatial relationships. To leverage the advantages of GoT, we implement a unified framework that integrates Qwen2.5-VL for reasoning chain generation with an end-to-end diffusion model enhanced by our novel Semantic-Spatial Guidance Module. Experiments show our GoT framework achieves excellent performance on both generation and editing tasks, with significant improvements over baselines. Additionally, our approach enables interactive visual generation, allowing users to explicitly modify reasoning steps for precise image adjustments. GoT pioneers a new direction for reasoning-driven visual generation and editing, producing images that better align with human intent. To facilitate future research, we make our datasets, code, and pretrained models publicly available at [https://github.com/rongyaofang/GoT](https://github.com/rongyaofang/GoT).

![[Uncaptioned image]](https://arxiv.org/html/2503.10639v1/x1.png)

Figure 1: Generation Chain-of-Thought (GoT) with Semantic-Spatial Reasoning. Our approach transforms input prompts into explicit reasoning chains with coordinates (middle), which guides vivid image generation and precise editing (right). This reasoning-based generation paradigm unifies spatial understanding across visual tasks: semantically-grounded visual generation (top), controllable interactive generation (middle), and localized image editing (bottom).

## 1 Introduction

Language provides the primary interface for expressing human intent in visual content generation. Traditional image generation systems [^37] [^6] [^21], particularly diffusion models, process textual prompts by mapping semantic concepts to visual elements without explicit reasoning. These approaches struggle with complex scenes requiring precise spatial arrangements and object interactions that humans naturally consider when constructing scenes. Meanwhile, multimodal large language models (MLLMs) [^3] [^2] [^25] excel at sophisticated reasoning tasks, including analyzing semantic structures, inferring relationships, grounding visual concepts, and processing detailed contexts through explicit reasoning chains. This gap between MLLMs’ advanced reasoning capabilities and the limited reasoning in current generation systems raises a key question: How can we integrate the reasoning mechanisms that have revolutionized language understanding into visual generation and editing?

Prior work attempted to leverage LLMs for image generation from different perspectives. One line of research [^23] [^55] leverages LLMs as text encoders for better prompt interpretation. However, the reasoning capabilities of LLMs are not introduced. Another line of work develops multimodal LLMs to unify understanding and generation [^44] [^47] [^50] [^8]. Although they present unified models for different tasks, there is no evidence that generation benefits from strong understanding and reasoning abilities of the models. They merely combine independent tasks rather than truly fusing language reasoning with visual generation. Additionally, layout-based methods like GLIGEN [^22], LayoutGPT [^9], and RPG [^52] incorporate LLMs for layout planning and diffusion models for layout-guided generation. However, these methods treat planning and generation as separate stages rather than integrating reasoning throughout the end-to-end process. Consequently, current image generation methods lack reasoning capabilities, emphasizing the need for a framework that seamlessly combines reasoning with visual generation and editing.

Inspired by chain-of-thought (CoT) reasoning of the LLMs, we introduce Generation Chain-of-Thought (GoT), a novel paradigm that enables visual generation to first output step-by-step reasoning in natural language before producing images. However, implementing GoT poses two significant challenges. First, different from CoT in LLMs, the reasoning chain for visual generation and editing requires both semantic and spatial information. It requires a new formulation and collecting training data in this new format. Second, existing diffusion-based models cannot leverage explicit language reasoning chains during visual generation. We need to design a framework supporting end-to-end language reasoning and visual generation.

To address the first challenge, we formulate GoT as a multimodal reasoning chain that integrates semantic and spatial analyses to enhance image generation and editing tasks. For visual generation, GoT provides precise control over object layout, relationships, and attributes, while for editing, it leverages semantic and spatial understanding to decompose user requests into coherent grounding and modification steps. We utilize advanced MLLMs and LLMs to construct complex annotation pipelines, which capture semantic-spatial interactions across diverse visual contexts. We assembled extensive datasets comprising 8.4M images for text-to-image generation (from Laion-Aesthetics [^39], JourneyDB [^41], and FLUX [^21]) and 920K examples for image editing (from OmniEdit [^48] and SEED-Edit-Multiturn [^12]). This computationally intensive effort produced the first large-scale dataset of reasoning chains for image generation and editing.

To tackle the second challenge of architecture design supporting reasoning and generation, we construct a unified end-to-end framework. Our GoT framework integrates the reasoning capabilities of MLLMs with the high-fidelity generation qualities of diffusion models. The proposed framework leverages an MLLM to generate reasoning steps and visual tokens, providing explicit guidance that incorporates semantic relationships and spatial configurations. This guidance flows into our novel Semantic-Spatial Guidance Module (SSGM), which conditions the diffusion process to ensure that generated images are closely guided by the reasoning process. This design supports end-to-end training and inference for visual generation and editing guided by explicit reasoning chains.

By effectively integrating reasoning into visual generation, our GoT framework demonstrates significant improvements in both text-to-image generation quality and image editing accuracy. Additionally, GoT enables interactive generation, allowing users to control the generated image by directly modifying the explicit reasoning process according to their preferences. These advantages represent a substantial advancement in reasoning-guided visual synthesis.

The main contributions can be summarized as follows:

- We propose Generation Chain-of-Thought (GoT), a paradigm that integrates reasoning into visual generation and editing tasks, enabling explicit semantic and spatial reasoning for these tasks.
- We define the formulation of semantic and spatial reasoning chains for visual generation and editing, and collect the first large-scale GoT datasets comprising 8.4M image generation samples and 920K image editing samples.
- We develop a unified end-to-end framework that leverages multimodal language models and diffusion models, with a novel Semantic-Spatial Guidance Module that ensures generated images follow the reasoning process.
- Our experimental results demonstrate significant improvements in both text-to-image generation and editing.

## 2 Related Work

![Refer to caption](https://arxiv.org/html/2503.10639v1/x2.png)

Figure 2: GoT Dataset Construction Process. Left: Text-to-image GoT annotation pipeline that labels detailed GoT with semantic content and spatial coordinates. Right: Editing GoT annotation pipeline that processes source image, target image, and instruction to generate entity-aware reasoning GoT with precise spatial grounding. Both pipelines leverage Qwen2-VL 46 and Qwen2.5 51 models for various stages of the annotation process.

### 2.1 Diffusion Models

Diffusion models have revolutionized visual content creation. Early approaches [^35] [^30] demonstrated this paradigm’s potential, while Stable Diffusion [^37] improved efficiency through latent space compression. Recent models [^36] [^38] [^32] [^6] [^21] [^16] have further advanced photorealism through architectural innovations and larger-scale training. Various efforts to extend diffusion models’ capabilities include controllable generation methods [^54] [^28] and instruction-based editing frameworks [^5] [^40]. While some researchers have explored unifying vision tasks [^11] [^7], these primarily focus on traditional computer vision tasks rather than general image generation. Despite these advances, current models typically process prompts through direct mapping, using text encoders like CLIP [^33] or T5 [^34] to condition the diffusion process via cross-attention [^45]. This approach treats text as a static representation without explicit reasoning about scene composition or object relationships. The fundamental limitation becomes evident when generating complex scenes with multiple objects and specific spatial arrangements, necessitating more sophisticated reasoning-based approaches.

### 2.2 Large Language Models and Reasoning

Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities through chain-of-thought (CoT) [^49], enabling complex problem decomposition. This paradigm extends to MLLMs [^1] [^2], which integrate visual and textual understanding. Some advanced works [^25] [^31] have enhanced spatial understanding by grounding textual concepts to image regions, enabling analysis of object relationships. Despite these capabilities, MLLMs remain underutilized for visual generation. While models like Chameleon [^44] and Emu2 [^43] incorporate image generation, they lack mechanisms to decompose user intent into semantic-spatial reasoning steps.

### 2.3 Layout-guided Image Generation and Editing

Recent research has explored layout-guided approaches for spatial control in visual synthesis. GLIGEN [^22] incorporated bounding boxes through gated cross-attention layers, enhancing object placement. LayoutGPT [^9] proposed a two-stage pipeline converting text into scene layouts before generation. RPG [^52] advanced this through recurrent planning, alternating between layout refinement and synthesis. SmartEdit [^17] adapts the LLaVA [^24] model to specialize in image editing tasks. FlexEdit [^29] employs an MLLM to comprehend the image content, mask, and user instructions. Despite these advances, existing approaches treat layouts as static constraints or sequential plans generated before synthesis, disconnecting spatial planning from generation.

## 3 Generation Chain-of-Thought (GoT)

During visual generation and editing, humans naturally reason about object relationships and spatial arrangements. In contrast, most current models process prompts without explicit reasoning, making it difficult to interpret complex human intentions for generating scenes with detailed object relationships and spatial configurations.

Motivated by chain-of-thought (CoT) in language models, we propose Generation Chain-of-Thought (GoT), shifting the visual generation from direct mapping to a reasoning-guided process. Unlike language generation, which operates primarily within a semantic space, visual generation requires an integrated understanding of both semantic relationships and spatial configurations. To address this complexity, GoT employs a multi-modal reasoning formulation that bridges conceptual understanding and spatial reasoning. This formulation incorporates explicit coordinate information in format (x1,y1),(x2,y2) with range \[0,1000), ensuring precise management over the placement of visual elements. This unified semantic-spatial reasoning chain enables fine-grained control of object placement, attributes, and inter-object relationships, ultimately supporting robust and coherent visual generation.

To illustrate the formulation of GoT, Fig. 1 presents examples of both text-to-image generation and editing tasks. For text-to-image, GoT generates a detailed reasoning chain specifying precise coordinates of elements. This explicit spatial reasoning enables a proper arrangement of all constituents while maintaining their semantic relationships, resulting in a coherent and visually appealing composition.

The image editing example in Fig. 1 demonstrates how GoT handles manipulation tasks through structured reasoning. When tasked with *replace the giant leaf with an umbrella*, GoT first analyzes the scene and then plans edits with precise coordinates. Finally, GoT describes what the image shows after editing. This decomposition into sequential steps with explicit spatial reasoning streamlines complex manipulations, contrasting with traditional editing methods that lack spatial awareness and reasoning.

GoT endows image generation and editing with reasoning benefits. By decomposing complex instructions into clearly defined, sequential steps, GoT delivers results that more accurately fulfill human requests. Its transparent process explains the intermediate reasoning behind each change and enables both image generation and editing within a unified system.

Implementing GoT requires two key components:

- A Comprehensive Dataset: This dataset must consist of detailed reasoning chains that align with visual content, capturing both semantic relationships and spatial configurations. Such data provide the necessary foundation for the reasoning process.
- A Compatible Visual Generation Model: The model needs to accommodate chain input to integrate semantic analysis and spatial reasoning, ensuring effective execution of the reasoning steps derived from the dataset.

In the following sections, we elaborate on these components and discuss how they contribute to the robust performance of the GoT framework.

## 4 GoT Dataset: Semantic-Spatial Reasoning Chains for Visual Generation and Editing

![Refer to caption](https://arxiv.org/html/2503.10639v1/x3.png)

Figure 3: GoT Framework with Semantic-Spatial Guidance. Left: Our dual-task framework handling both text-to-image generation (T2I) and image editing. Right: The SSGM Diffusion Module, which combines spatial layouts guidance G s subscript 𝐺 𝑠 G\_{s} italic\_G start\_POSTSUBSCRIPT italic\_s end\_POSTSUBSCRIPT, reference image guidance r 𝑟 G\_{r} italic\_G start\_POSTSUBSCRIPT italic\_r end\_POSTSUBSCRIPT, and semantic guidance t 𝑡 G\_{t} italic\_G start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT to generate the final image with precise content and spatial control.

Based on the formulation presented previously, we construct large-scale training datasets using advanced LLMs and MLLMs. Our GoT dataset features meticulously crafted semantic-spatial reasoning chains for both generation and editing tasks, with each sample containing instructions, reasoning chain annotations, and corresponding images. The construction requires careful design of task-specific annotation pipelines to ensure quality. The prompts used in the pipelines are attached in Appendix Sec. 11.

### 4.1 Automated Data Creation Pipeline

As illustrated in Fig. 2, our annotation pipeline demonstrates the multiple stages of processing required to generate these high-quality annotations. For text-to-image, we utilize Qwen2-VL [^46] to generate concise prompts that serve as text-to-image generation prompts and detailed visual descriptions that form the semantic component of GoT. Qwen2.5 [^51] then performs object entity extraction, followed by Qwen2-VL establishing spatial relationships through object grounding. The detailed visual descriptions merged with precise object groundings together constitute the complete GoT annotation for text-to-image generation.

For the image editing pipeline, we employ Qwen2-VL to generate comprehensive descriptions of source and target images, precisely localize editing regions through bounding boxes, and generate detailed descriptions of edited objects after cropping. We then leverage Qwen2.5 with carefully designed in-context prompting to synthesize coherent GoT reasoning chains, ensuring logical flow and completeness of the editing process. From this pipeline, we derive concise editing instructions as editing inputs while using the detailed semantic-spatial reasoning steps as GoT annotations. For the complex multi-turn editing dataset, we developed a related but more sophisticated protocol with Qwen2-VL and Qwen2.5 to obtain intricate step-by-step reasoning chains with multiple spatial coordinates and transformation descriptions, capturing complex editing sequences.

### 4.2 Dataset Construction

For text-to-image generation, we construct dataset from three sources: Laion-Aesthetics-High-Resolution (LAHR) [^39] with 3.77M samples filtered for images larger than 512 pixels, JourneyDB [^41] with 4.09M samples, and 600K FLUX.1-generated [^21] images using LAHR prompts. The final datasets yield rich annotations: LAHR-GoT samples with prompts averaging 110.81 characters, GoT descriptions averaging 811.56 characters, and 3.78 bounding boxes per image. Similarly, JourneyDB-GoT annotations average 149.78 characters for prompts, 906.01 characters for GoT descriptions, and 4.09 boxes image.

For the single-turn image editing dataset, we build on OmniEdit [^48], a premier open-source image editing dataset with high-fidelity images, processing 736,691 samples covering editing operations (addition, removal, swap, changing expression/color/weather/lighting, and style transfer). The multi-turn image editing dataset is built upon SEED-Edit-Multiturn [^12], resulting in 180,190 samples.

The entire data creation process demanded substantial computational resources, requiring 100 NVIDIA A100 GPUs for over a month. This comprehensive approach ensures our dataset provides the robust foundation necessary for training models capable of sophisticated image generation and editing tasks.

## 5 GoT Framework: Reasoning-guided Visual Generation and Editing

We present the GoT framework, a unified end-to-end approach embedding reasoning-guided processes into visual generation and editing tasks. GoT integrates two primary components: a semantic-spatial aware MLLM generating structured reasoning chains with spatial information, and a multi-guided diffusion model leveraging these reasoning outputs through our proposed Semantic-Spatial Guidance Module (SSGM) in an end-to-end manner. This design ensures that generated images precisely follow logical reasoning steps, allowing detailed control over both semantic content and spatial relationships.

### 5.1 Semantic-Spatial MLLM Design

Our framework utilizes a state-of-the-art MLLM Qwen2.5-VL-3B as the backbone, chosen for its outstanding visual understanding and grounding capabilities. This MLLM functions as a reasoning engine, handling both generation and editing tasks through a unified architecture.

<table><tbody><tr><td>Method</td><td>Architecture</td><td>Overall</td><td>Single Obj.</td><td>Two Obj.</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attr. Binding</td></tr><tr><td colspan="9">Frozen Text Encoder Mapping Methods</td></tr><tr><td>SDv1.5 <sup><a href="#fn:37">37</a></sup></td><td>Unet+CLIP</td><td>0.43</td><td>0.97</td><td>0.38</td><td>0.35</td><td>0.76</td><td>0.04</td><td>0.06</td></tr><tr><td>SDv2.1 <sup><a href="#fn:37">37</a></sup></td><td>Unet+CLIP</td><td>0.50</td><td>0.98</td><td>0.51</td><td>0.44</td><td>0.85</td><td>0.07</td><td>0.17</td></tr><tr><td>SD-XL <sup><a href="#fn:32">32</a></sup></td><td>Unet+CLIP</td><td>0.55</td><td>0.98</td><td>0.74</td><td>0.39</td><td>0.85</td><td>0.15</td><td>0.23</td></tr><tr><td>DALLE-2 <sup><a href="#fn:36">36</a></sup></td><td>Unet+CLIP</td><td>0.52</td><td>0.94</td><td>0.66</td><td>0.49</td><td>0.77</td><td>0.10</td><td>0.19</td></tr><tr><td>SD3 (d=24) <sup><a href="#fn:6">6</a></sup></td><td>MMDIT+CLIP+T5</td><td>0.62</td><td>0.98</td><td>0.74</td><td>0.63</td><td>0.67</td><td>0.34</td><td>0.36</td></tr><tr><td colspan="9">LLMs/MLLMs Enhanced Methods</td></tr><tr><td>LlamaGen <sup><a href="#fn:42">42</a></sup></td><td>Autoregressive</td><td>0.32</td><td>0.71</td><td>0.34</td><td>0.21</td><td>0.58</td><td>0.07</td><td>0.04</td></tr><tr><td>Chameleon <sup><a href="#fn:44">44</a></sup></td><td>Autoregressive</td><td>0.39</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LWM <sup><a href="#fn:26">26</a></sup></td><td>Autoregressive</td><td>0.47</td><td>0.93</td><td>0.41</td><td>0.46</td><td>0.79</td><td>0.09</td><td>0.15</td></tr><tr><td>SEED-X <sup><a href="#fn:13">13</a></sup></td><td>Unet+Llama</td><td>0.49</td><td>0.97</td><td>0.58</td><td>0.26</td><td>0.80</td><td>0.19</td><td>0.14</td></tr><tr><td>Emu3-Gen <sup><a href="#fn:47">47</a></sup></td><td>Autoregressive</td><td>0.54</td><td>0.98</td><td>0.71</td><td>0.34</td><td>0.81</td><td>0.17</td><td>0.21</td></tr><tr><td>Janus <sup><a href="#fn:50">50</a></sup></td><td>Autoregressive</td><td>0.61</td><td>0.97</td><td>0.68</td><td>0.30</td><td>0.84</td><td>0.46</td><td>0.42</td></tr><tr><td>JanusFlow <sup><a href="#fn:27">27</a></sup></td><td>Autoregressive</td><td>0.63</td><td>0.97</td><td>0.59</td><td>0.45</td><td>0.83</td><td>0.53</td><td>0.42</td></tr><tr><td>GoT Framework</td><td>Unet+Qwen2.5-VL</td><td>0.64</td><td>0.99</td><td>0.69</td><td>0.67</td><td>0.85</td><td>0.34</td><td>0.27</td></tr></tbody></table>

Table 1: Evaluation of text-to-image generation on GenEval benchmark [^14]. Obj.: Object. Attr.: Attribution.

As shown in Fig. 3, the MLLM’s pipeline begins with task-specific input handling. For editing tasks, it processes reference images through the vision encoder to understand the source content. For both generation and editing, the MLLM produces GoT reasoning chains, capturing object attributes, relationships, modifications, and bounding box information. Following reasoning chain generation, the model processes an image start token followed by $N=64$ special \[IMG\] tokens, generating semantic guidance embeddings $G_{t}$ that encapsulate information from the previous reasoning chains. Additionally, the spatial guidance $G_{s}$ is derived by parsing and converting the explicit spatial information in the generated reasoning chains.

This semantic-spatial aware design enables the MLLM to direct the SSGM Diffusion Module with precise control over content and layout. During training, the MLLM receives supervision through two pathways: cross-entropy loss on GoT reasoning tokens and gradient signals backpropagated from the end-to-end SSGM diffusion module through semantic guidance $G_{t}$.

### 5.2 Semantic-spatial Guided Diffusion Generation

The end-to-end diffusion module builds upon SDXL’s [^32] architecture, incorporating an innovative triple-guidance mechanism that integrates semantic understanding, spatial awareness, and reference knowledge through our Semantic-Spatial Guidance Module (SSGM). In SSGM, the semantic guidance pathway enhances the diffusion model by channeling $N=64$ MLLM-generated embeddings $G_{t}$ through cross-attention layers, replacing conventional CLIP embeddings for more precise semantic control.

For spatial guidance in SSGM, we extract coordinate information from the generated GoT to create color-coded masks where each object or editing region receives a distinct color based on a predefined order in the GoT sequence. These colored masks are processed through a VAE encoder [^18] and averaged to produce spatial latent features $G_{s}$, which are concatenated with the diffusion model’s latent representations, enabling precise spatial control during both generation and editing tasks.

Following InstructPix2Pix [^5], we incorporate reference image guidance as the third SSGM pathway. For editing tasks, the source image serves as a reference, while for text-to-image generation, we use a black reference image for architectural consistency. This design enables a seamless transition between generation and editing tasks without architectural modifications. All references are processed through the VAE encoder to extract visual features $G_{r}$.

![Refer to caption](https://arxiv.org/html/2503.10639v1/x4.png)

Figure 4: Text-to-Image samples generated by our model. The GoT framework can plan object placement based on the input caption and generate highly aligned and aesthetic images accordingly.

### 5.3 Multi-Guidance Strategy

We employ a classifier-free guidance strategy integrating semantic, spatial, and reference image guidance. During diffusion, the score estimation $\varepsilon_{\theta}$ is calculated through a weighted combination:

$$
\displaystyle\varepsilon_{\theta}=
$$
 
$$
\displaystyle\varepsilon_{\theta}(z_{t},\varnothing,\varnothing,\varnothing)+
$$
 
$$
\displaystyle\alpha_{t}\cdot[\varepsilon_{\theta}(z_{t},G_{t},\varnothing,G_{r%
})-\varepsilon_{\theta}(z_{t},\varnothing,\varnothing,G_{r})]+
$$
 
$$
\displaystyle\alpha_{s}\cdot[\varepsilon_{\theta}(z_{t},G_{t},G_{s},G_{r})-%
\varepsilon_{\theta}(z_{t},G_{t},\varnothing,G_{r})]+
$$
 
$$
\displaystyle\alpha_{r}\cdot[\varepsilon_{\theta}(z_{t},\varnothing,%
\varnothing,G_{r})-\varepsilon_{\theta}(z_{t},\varnothing,\varnothing,%
\varnothing)]
$$

where $z_{t}$ is the noisy latent, $G_{t}$ denotes semantic guidance embeddings, $G_{s}$ indicates spatial guidance features, and $G_{r}$ represents reference image features. Guidance scales $\alpha_{t}$, $\alpha_{s}$, and $\alpha_{r}$ control the strength of each guidance type, while $\varnothing$ denotes null conditioning. During training, we randomly sample conditioning combinations with a probability of 5%, excluding the fully-conditioned case $\varepsilon_{\theta}(z_{t},G_{t},G_{s},G_{r})$, to enhance robustness. Optimal guidance parameters are introduced in Sec. 6.

### 5.4 Training Procedure

Our training process implements a two-phase approach: pretraining using LAHR-GoT, JourneyDB-GoT, and OmniEdit-GoT datasets (60,000 steps), followed by finetuning with FLUX-GoT, OmniEdit-GoT, and SEED-Edit-MultiTurn-GoT (10,000 steps). We employ low-rank adaptation (LoRA) [^15] to efficiently update the Qwen2.5-VL decoder’s parameters while fully optimizing the SDXL-based diffusion module. The process operates end-to-end, jointly optimizing the MLLM GoT cross-entropy token loss and diffusion MSE loss with equal weighting $1.0$, demonstrating robustness without complex hyperparameter tuning.

## 6 Experiments

We evaluate GoT framework on text-to-image generation, interactive image generation, and image editing. Experiments show quantitative improvements and qualitative benefits of our reasoning-guided approach, with ablation studies validating our design choices.

### 6.1 Text-to-Image Generation

#### 6.1.1 Quantitative Results

![Refer to caption](https://arxiv.org/html/2503.10639v1/x5.png)

Figure 5: Samples on interactive generation with GoT framework. By modifying GoT content (description and bounding box position), user can customize their text-to-image process with: 1. Object replacement 2. Object position adjustment 3. Object attribute modification.

Tab. 1 presents a evaluation of text-to-image generation (T2I) on GenEval [^14]. The comparison spans two main categories of models: those employing frozen text encoders for direct prompt-to-image generation (primarily diffusion-based approaches) and those leveraging LLMs or MLLMs to enhance the generation process. On T2I task, GoT framework adopts $\alpha_{t}=7.5$ and $\alpha_{s}=4.0$, and more discussions on $\alpha$ tuning are shown in Appendix Sec. 9.2.

As shown in Tab. 1, our framework achieves the highest overall score of 0.64, outperforming both frozen text encoder methods and LLM/MLLM-enhanced approaches. GoT framework excels particularly in single object (0.99), counting tasks (0.67), and color tasks (0.85), demonstrating the effectiveness of our reasoning-guided generation paradigm. While methods like JanusFlow [^27] perform better in position and attribute binding tasks, GoT framework’s balanced performance across all metrics validates that incorporating explicit reasoning mechanisms enhances compositional generation abilities.

Among the LLM/MLLM-enhanced methods, our approach outperforms recent systems like Janus [^50] and JanusFlow [^27] in overall performance despite their advantages in specific areas. This suggests that while autoregressive models excel in certain spatial tasks, our GoT framework’s structured reasoning provides more consistent performance across diverse generation requirements.

#### 6.1.2 Qualitative Results

In addition to the outstanding compositional text-to-image generation capability, GoT framework also exhibits high generation quality. In Fig. 4, we showcase the generation results of our model across a diverse set of prompts. We present samples from compositional prompts containing multiple objects, incorporating object attributes, relationships, and relative spatial positions. Our model effectively plans the placement of different objects, producing coherent and aesthetically pleasing images.

### 6.2 Interactive Generation

In our experiments, we further demonstrate the interactive capabilities of the GoT framework, as illustrated in Fig. 5. This approach enables user control over the generation process by modifying the GoT content, including both textual descriptions and bounding box positions. Users can customize their text-to-image generation through three primary interaction types: object replacement, object position adjustment, and object attribute modification. The examples showcase how the framework maintains overall scene coherence while precisely implementing the requested changes. This interactive flexibility provides an interpretable and manipulable interface for text-to-image generation that traditional black-box systems lack, allowing for precise control over the output without requiring expertise.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Params.</th><th colspan="2">Emu-Edit</th><th>ImagenHub</th><th>Reason-Edit</th></tr><tr><th>CLIP-I</th><th>CLIP-T</th><th>GPT-4o Eval.</th><th>GPT-4o Eval.</th></tr></thead><tbody><tr><th>IP2P <sup><a href="#fn:5">5</a></sup></th><th>0.9B+0.1B</th><td>0.834</td><td>0.219</td><td>0.308</td><td>0.286</td></tr><tr><th>MagicBrush <sup><a href="#fn:53">53</a></sup></th><th>0.9B+0.1B</th><td>0.838</td><td>0.222</td><td>0.513</td><td>0.334</td></tr><tr><th>MGIE <sup><a href="#fn:10">10</a></sup></th><th>0.9B+7B</th><td>0.783</td><td>0.253</td><td>0.392</td><td>0.264</td></tr><tr><th>Emu-Edit <sup><a href="#fn:40">40</a></sup></th><th>-</th><td>0.859</td><td>0.231</td><td>-</td><td>-</td></tr><tr><th>SEED-X <sup><a href="#fn:13">13</a></sup></th><th>2.8B+14B</th><td>0.825</td><td>0.272</td><td>0.166</td><td>0.239</td></tr><tr><th>SmartEdit <sup>†</sup> <sup><a href="#fn:17">17</a></sup></th><th>0.9B+7B</th><td>-</td><td>-</td><td>-</td><td>0.572</td></tr><tr><th>CosXL-Edit <sup><a href="#fn:4">4</a></sup></th><th>-</th><td>0.860</td><td>0.274</td><td>0.464</td><td>0.325</td></tr><tr><th>GoT Framework</th><th>2.8B+3B</th><td>0.864</td><td>0.276</td><td>0.533</td><td>0.561</td></tr></tbody></table>

Table 2: Quantitative comparison on image editing benchmarks. <sup>†</sup> denotes that SmartEdit mainly supports removing and replacing operation and is not designed for general editing operations.

### 6.3 Image Editing

![Refer to caption](https://arxiv.org/html/2503.10639v1/x6.png)

Figure 6: Qualitative results of image editing. Our GoT framework demonstrates superior performance in settings that require semantic-spatial reasoning. Red bounding boxes indicate the coordinates predicted by MLLM within the GoT framework.

#### 6.3.1 Quantitative Results

As shown in Tab. 2, we evaluate our GoT framework against state-of-the-art image editing methods across multiple benchmarks. On Emu-Edit benchmark [^40], GoT framework achieves the highest scores for both CLIP-I (0.864) and CLIP-T (0.276) metrics, outperforming previous methods including CosXL-Edit [^4] and Emu-Edit [^40]. Since CLIP-I and CLIP-T cannot fully reflect editing accuracy, we also evaluated using GPT-4o [^1], which aligns better with human evaluation [^19]. On ImagenHub [^20], our approach attains the highest score of 0.533. On the reasoning-based Reason-Edit benchmark [^17], our model achieves a strong score of 0.561, second only to SmartEdit (0.572) [^17], which is specially designed for reasoning removing and replacing operations. This demonstrates our method’s strong editing ability, especially in complex reasoning settings. GoT framework shows consistently superior performance while maintaining competitive parameter efficiency (2.8B+3B) compared to approaches like SEED-X (2.8B+14B) [^13]. In the editing task, GoT framework adopts $\alpha_{t}=4.0$, $\alpha_{s}=3.0$, and $\alpha_{r}=1.5$. The evaluation prompt of GPT-4o is shown in Appendix Sec. 11.1.

#### 6.3.2 Qualitative Results

We present qualitative comparison of image editing with other models in Fig. 6. Our approach demonstrates superior performance across diverse editing scenarios that require semantic-spatial reasoning. The examples highlight our framework’s distinctive capabilities: First, our model accurately identifies and localizes objects referenced through indirect descriptions. Second, our approach handles complex spatial instructions effectively, such as removing specific signage or adding delicate elements to precise locations. Third, our framework excels at multi-step editing operations, as demonstrated in the bottom example. The red bounding boxes visible in our results indicate the coordinates predicted by the MLLM within the GoT framework, providing interpretable insight into how our system reasons about spatial relationships during the editing process.

### 6.4 Ablation Study on Framework Design

We conduct an ablation study to analyze the impact of different components in our framework. Table 3 presents the results of our study, where we progressively integrate different components into the baseline and evaluate their effects on GenEval and ImagenHub benchmarks.

The baseline model leverages Qwen2.5-VL-3B and SDXL but does not incorporate GoT reasoning chains. It is trained with FLUX-GoT and OmniEdit-GoT for 10,000 steps. Adding GoT reasoning chains to the baseline model enables the LLM to achieve stronger semantic guidance capabilities. The reasoning process helps LLM plan for guidance in generation.

Introducing the Semantic-Spatial Guidance Module (SSGM) further enhances model performance, particularly in image editing. SSGM provides spatial control over the diffusion model, ensuring that object placement aligns more accurately with the reasoning process. This enables fine-grained editing, as reflected by the significant improvement in the ImagenHub evaluation. However, in GenEval, only the position category is notably affected by SSGM, which explains the relatively minor performance gain.

Our final framework, which includes GoT reasoning, SSGM, and an extensive 60,000-step pretraining phase, achieves the highest scores, demonstrating the significant benefits of prolonged pretraining and the full model design. The ablation study confirms that each added component contributes positively to the overall performance, validating our framework design choices.

| Method | GoT | SSGM | Pretrain | GenEval | ImagenHub |
| --- | --- | --- | --- | --- | --- |
| Baseline | $\times$ | $\times$ | $\times$ | 0.38 | 0.176 |
| \+ GoT | ✓ | $\times$ | $\times$ | 0.40 | 0.181 |
| \+ SSGM | ✓ | ✓ | $\times$ | 0.42 | 0.370 |
| GoT Framework | ✓ | ✓ | ✓ | 0.64 | 0.533 |

Table 3: Ablation study of our GoT framework on GenEval overall and ImagenHub GPT-4o eval.

## 7 Conclusion

We presented Generation Chain-of-Thought (GoT), a paradigm that integrates MLLM reasoning capabilities into visual generation through explicit semantic-spatial reasoning chains. Our approach transforms visual generation from direct mapping into a reasoning-guided process with precise spatial control, addressing limitations in existing methods that lack explicit understanding of object relationships and arrangements. Through large-scale dataset construction (9M+ examples), a novel Semantic-Spatial Guidance Module, and an end-to-end training framework, GoT achieves state-of-the-art performance on text-to-image generation and editing benchmarks while enabling unprecedented interactive control through modifiable reasoning chains. By bridging the gap between human reasoning and visual creation, GoT introduces a more intuitive and powerful approach to visual synthesis that aligns with natural cognitive processes.

## References

Supplementary Material

## 8 Training Details

We pretrain our model for 60,000 steps on LAHR-GoT, JourneyDB-GoT, and OmniEdit-GoT. We adopt a cosine learning rate scheduler with 500 warmup steps and a maximum learning rate of $1\times 10^{-4}$.

During the fine-tuning stage, we train the model on FLUX-GoT, OmniEdit-GoT, and SEED-Edit-MultiTurn-GoT for 10,000 steps. In this phase, we set the warmup steps to 200 and the maximum learning rate to $5\times 10^{-5}$.

For both stages, we use the Adam optimizer with $\beta_{1}=0.9$, $\beta_{2}=0.98$, and $\epsilon=1\times 10^{-6}$. We also apply a weight decay of 0.05 during training. The number of batch size is set to 128.

The LLM is fine-tuned using LoRA with $r=32$, LoRA alpha set to 32, and a LoRA dropout rate of 0.05. For diffusion, we introduce a noise offset of 0.1.

## 9 Visualization Results

### 9.1 Qualitative Analysis of Image Editing and Interactive Generation

We provide additional examples to demonstrate the capabilities of the GoT framework. Figure 7 illustrates the image editing performance of our model. Additionally, we present the corresponding GoT content generated alongside each sample. Further examples of interactive generation using our model are shown in Figure 8.

### 9.2 Visualization of Multi-Guidance Strategy Hyperparameter Selection

We analyze the effect of hyperparameter selection in the Multi-Guidance Strategy on the generated images, as depicted in Figure 9. The definitions of these hyperparameters are provided in Section 5.3.

## 10 GoT Format and Examples

This section presents examples of the GoT format in our dataset. The GoT structure varies across different tasks, including text-to-image (T2I) generation, single-turn editing, and multi-turn editing.

For text-to-image generation, Figure 10 showcases examples from FLUX-GoT, JourneyDB-GoT, and LAHR-GoT. Our GoT format represents the structured planning process of the upstream model in generating image content. It provides a detailed breakdown of the various components within an image and their spatial relationships. To enhance spatial understanding, we append location information to key objects within the GoT representation.

Figure 11 illustrates the GoT format for image editing within our dataset. For single-turn editing, GoT represents the reasoning plan of the upstream model for a specific editing action. It consists of a description of the source image, the object to be modified, the specific editing operation, and the resulting edited image. This structured process ensures a step-by-step transformation, beginning with the original image, identifying the target object, applying the specified modification, and generating the edited image.

For multi-turn editing, GoT follows a more complex structure, as it must encapsulate the breakdown of an instruction into a sequence of consecutive steps. In practice, we first generate a description of the source image, then decompose the multi-turn instruction into a series of step-by-step editing commands. At each step, GoT operates as a single-turn editing process, specifying the object to be modified along with the corresponding transformation. Finally, the process concludes with a description of the fully edited image.

Furthermore, for image editing tasks, positional information is appended to each object to enhance spatial comprehension.

## 11 Prompts for Evaluation and Dataset Construction

### 11.1 Prompts for Evaluating Image Editing Performance

We provide the prompts used for evaluating image editing performance with GPT-4o in Figure 12. We are using GPT-4o-2024-11-20. The final score is the average of the minimum value of the two scores for each sample.

### 11.2 Prompts for Text-to-Image Data Construction

Figures 13, 14, and 16 present the key prompts utilized in text-to-image data preparation.

### 11.3 Prompts for Image Editing Data Construction

Figures 15–20 illustrate the key-step prompts employed in image editing data preparation.

![Refer to caption](https://arxiv.org/html/2503.10639v1/x7.png)

Figure 7: More samples on image editing with the GoT content generated by our model.

![Refer to caption](https://arxiv.org/html/2503.10639v1/x8.png)

Figure 8: More examples on interactive generation.

![Refer to caption](https://arxiv.org/html/2503.10639v1/x9.png)

Figure 9: Visualization on Multi-Guidance Strategy Hyper-parameter Selection. The above are text-to-image samples generated by GoT framework under different hyper-parameters.

![Refer to caption](https://arxiv.org/html/2503.10639v1/x10.png)

Figure 10: Examples of GoT dataset for text-to-image generation, including FLUX-GoT, JourneyDB-GoT, and Laion-Aesthetics-High-Resolution-GoT.

![Refer to caption](https://arxiv.org/html/2503.10639v1/x11.png)

Figure 11: Examples of GoT dataset for image editing, including OmniEdit-GoT for single-turn editing and SEED-Edit-Multiturn-GoT for multi-turn editing.

<svg height="440.45" id="S11.F12.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,440.45) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 426.68 C 0 434.29 6.17 440.45 13.78 440.45 L 673.72 440.45 C 681.33 440.45 687.5 434.29 687.5 426.68 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 426.68 C 1.97 433.2 7.26 438.49 13.78 438.49 L 673.72 438.49 C 680.24 438.49 685.53 433.2 685.53 426.68 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="412.9" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F12.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F12.pic1.1.1.1.1.1.1"><span id="S11.F12.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F12.pic1.1.1.1.1.1.2"><span id="S11.F12.pic1.1.1.1.1.1.2.1" style="font-size:144%;">You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on the given rules. You will have to give your output in this way (Keep your reasoning concise and short.): ”score”: […], ”reasoning”: ”…” and don’t output anything else.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.3"><span id="S11.F12.pic1.1.1.1.1.1.3.1" style="font-size:144%;">Two images will be provided:</span></span> <span id="S11.F12.pic1.1.1.1.1.1.4"><span id="S11.F12.pic1.1.1.1.1.1.4.1" style="font-size:144%;">The first being the original AI-generated image and the second being an edited version of the first. The objective is to evaluate how successfully the editing instruction has been executed in the second image. Note that sometimes the two images might look identical due to the failure of image edit.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.5"><span id="S11.F12.pic1.1.1.1.1.1.5.1" style="font-size:144%;">From a scale 0 to 10:</span></span> <span id="S11.F12.pic1.1.1.1.1.1.6"><span id="S11.F12.pic1.1.1.1.1.1.6.1" style="font-size:144%;">A score from 0 to 10 will be given based on the success of the editing.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.7"><span id="S11.F12.pic1.1.1.1.1.1.7.1" style="font-size:144%;">- 0 indicates that the scene in the edited image does not follow the editing instruction at all.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.8"><span id="S11.F12.pic1.1.1.1.1.1.8.1" style="font-size:144%;">- 10 indicates that the scene in the edited image follow the editing instruction text perfectly.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.9"><span id="S11.F12.pic1.1.1.1.1.1.9.1" style="font-size:144%;">- If the object in the instruction is not present in the original image at all, the score will be 0.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.10"><span id="S11.F12.pic1.1.1.1.1.1.10.1" style="font-size:144%;">A second score from 0 to 10 will rate the degree of overediting in the second image.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.11"><span id="S11.F12.pic1.1.1.1.1.1.11.1" style="font-size:144%;">- 0 indicates that the scene in the edited image is completely different from the original.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.12"><span id="S11.F12.pic1.1.1.1.1.1.12.1" style="font-size:144%;">- 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.13"><span id="S11.F12.pic1.1.1.1.1.1.13.1" style="font-size:144%;">Put the score in a list such that output score = [score1, score2], where ’score1’ evaluates the editing success and ’score2’ evaluates the degree of overediting.</span></span> <span id="S11.F12.pic1.1.1.1.1.1.14"><span id="S11.F12.pic1.1.1.1.1.1.14.1" style="font-size:144%;">Editing instruction:</span> <span id="S11.F12.pic1.1.1.1.1.1.14.2" style="font-size:144%;">&lt;instruction&gt;</span></span> <span id="S11.F12.pic1.1.1.1.1.1.15"><span id="S11.F12.pic1.1.1.1.1.1.15.1" style="font-size:144%;">&lt;Image&gt;</span> <span id="S11.F12.pic1.1.1.1.1.1.15.2" style="font-size:144%;">Source Image</span> <span id="S11.F12.pic1.1.1.1.1.1.15.3" style="font-size:144%;">&lt;/Image&gt;</span></span> <span id="S11.F12.pic1.1.1.1.1.1.16"><span id="S11.F12.pic1.1.1.1.1.1.16.1" style="font-size:144%;">&lt;Image&gt;</span> <span id="S11.F12.pic1.1.1.1.1.1.16.2" style="font-size:144%;">Edited Image</span> <span id="S11.F12.pic1.1.1.1.1.1.16.3" style="font-size:144%;">&lt;/Image&gt;</span></span> <span id="S11.F12.pic1.1.1.1.1.1.17"><span id="S11.F12.pic1.1.1.1.1.1.17.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 12: Prompt for GPT4-o image editing evaluation. We are using GPT-4o-2024-11-20. The final score is the average of the minimum value of the two scores for each sample.

<svg height="376.53" id="S11.F13.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,376.53) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 362.75 C 0 370.36 6.17 376.53 13.78 376.53 L 673.72 376.53 C 681.33 376.53 687.5 370.36 687.5 362.75 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 362.75 C 1.97 369.27 7.26 374.56 13.78 374.56 L 673.72 374.56 C 680.24 374.56 685.53 369.27 685.53 362.75 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="348.97" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F13.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F13.pic1.1.1.1.1.1.1"><span id="S11.F13.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F13.pic1.1.1.1.1.1.2"><span id="S11.F13.pic1.1.1.1.1.1.2.1" style="font-size:144%;">&lt;Image&gt;</span> <span id="S11.F13.pic1.1.1.1.1.1.2.2" style="font-size:144%;">Image</span> <span id="S11.F13.pic1.1.1.1.1.1.2.3" style="font-size:144%;">&lt;/Image&gt;</span></span> <span id="S11.F13.pic1.1.1.1.1.1.3"><span id="S11.F13.pic1.1.1.1.1.1.3.1" style="font-size:144%;">You are an advanced AI visual assistant specializing in highly detailed and comprehensive visual analysis for one image. Your role is to generate a single, descriptive paragraph that encapsulates all relevant details about an image. Here is the provided image prompt for this image:</span> <span id="S11.F13.pic1.1.1.1.1.1.3.2" style="font-size:144%;">&lt;prompt&gt;</span><span id="S11.F13.pic1.1.1.1.1.1.3.3" style="font-size:144%;">.</span></span> <span id="S11.F13.pic1.1.1.1.1.1.4"><span id="S11.F13.pic1.1.1.1.1.1.4.1" style="font-size:144%;">If the provided prompt aligns with the image, enhance it by adding detailed observations about the objects, their colors, shapes, textures, numeracy, and spatial relationships. If the provided prompt does not match the image content, disregard it and craft a complete description based solely on the visual elements you observe. Consider the 2D-spatial relationships (e.g., ”to the left of,” ”near,” ”aligned with”) and 3D-spatial relationships (e.g., ”in front of,” ”above,” ”at a distance from”) when describing the scene. Include details about the overall composition, highlighting how elements are arranged relative to each other, their groupings, and any complex interactions or dynamic elements within the scene. Pay close attention to the interplay of colors, textures, and shapes, ensuring that the description reflects both the visual richness and structural composition of the image. Ensure to provide the description as one single paragraph, without preamble or additional explanation.</span></span> <span id="S11.F13.pic1.1.1.1.1.1.5"><span id="S11.F13.pic1.1.1.1.1.1.5.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 13: Prompt for detailed recaption for text-to-image data.

<svg height="681.83" id="S11.F14.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,681.83) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 668.05 C 0 675.66 6.17 681.83 13.78 681.83 L 673.72 681.83 C 681.33 681.83 687.5 675.66 687.5 668.05 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 668.05 C 1.97 674.57 7.26 679.86 13.78 679.86 L 673.72 679.86 C 680.24 679.86 685.53 674.57 685.53 668.05 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="654.27" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F14.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F14.pic1.1.1.1.1.1.1"><span id="S11.F14.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.2"><span id="S11.F14.pic1.1.1.1.1.1.2.1" style="font-size:144%;">You are tasked with identifying and extracting all the real object names from a detailed caption.</span></span> <span id="S11.F14.pic1.1.1.1.1.1.3"><span id="S11.F14.pic1.1.1.1.1.1.3.1" style="font-size:144%;">An object name refers to any tangible or physical entity mentioned in the caption that can be visually grounded in the image. Ensure not to include any adjectives or single-word descriptions that do not refer to a specific object, such as ”background.”</span></span> <span id="S11.F14.pic1.1.1.1.1.1.4"><span id="S11.F14.pic1.1.1.1.1.1.4.1" style="font-size:144%;">Please follow these instructions:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.5"><span id="S11.F14.pic1.1.1.1.1.1.5.1" style="font-size:144%;">Identify all object names in the caption in the order they appear. Maintain the exact wording of each object name as it is in the caption, including case consistency. Output the object names in a Python list format. For example, consider the following caption:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.6"><span id="S11.F14.pic1.1.1.1.1.1.6.1" style="font-size:144%;">Example 1:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.7"><span id="S11.F14.pic1.1.1.1.1.1.7.1" style="font-size:144%;">”In the image, a person is prominently featured at a vibrant pride parade, exuding confidence and pride. They are adorned in an extravagant outfit that mirrors the rainbow flag, with a deep V-neck top in bold, colorful stripes of red, orange, yellow, green, blue, and purple. The person’s hair is styled in a striking rainbow color, complementing their outfit. They are surrounded by a lively crowd, with individuals wearing various colors and accessories, adding to the festive atmosphere. The background reveals a bustling street scene with buildings and trees, suggesting an urban setting. The overall composition is dynamic, with the person at the center, drawing attention to their vibrant attire and the energetic parade around them.”</span></span> <span id="S11.F14.pic1.1.1.1.1.1.8"><span id="S11.F14.pic1.1.1.1.1.1.8.1" style="font-size:144%;">Your output should be a list of object names like this:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.9"><span id="S11.F14.pic1.1.1.1.1.1.9.1" style="font-size:144%;">[’person’, ’pride parade’, ’outfit’, ’V-neck top’, ’The person’s hair’, ’a lively crowd’, ’individuals’, ’street’, ’buildings’, ’trees’]</span></span> <span id="S11.F14.pic1.1.1.1.1.1.10"><span id="S11.F14.pic1.1.1.1.1.1.10.1" style="font-size:144%;">Example 2:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.11"><span id="S11.F14.pic1.1.1.1.1.1.11.1" style="font-size:144%;">”The image depicts a young boy with slender features and a pale complexion, exuding an air of arrogance and coldness. His white-blonde hair is slicked back, adding to his composed demeanor. The boy’s eyes are a striking shade of cold grey, reflecting a sense of detachment and intelligence. He is dressed in a white shirt with a blue and white patterned collar, which contrasts with his pale skin and adds a touch of elegance to his appearance. The overall composition is balanced, with the boy centrally positioned against a dark background that accentuates his features and the sharpness of his expression. The interplay of colors, textures, and shapes creates a visually striking and emotionally charged image.”</span></span> <span id="S11.F14.pic1.1.1.1.1.1.12"><span id="S11.F14.pic1.1.1.1.1.1.12.1" style="font-size:144%;">Your output should be a list of object names like this:</span></span> <span id="S11.F14.pic1.1.1.1.1.1.13"><span id="S11.F14.pic1.1.1.1.1.1.13.1" style="font-size:144%;">[’young boy’, ’white-blonde hair’, "The boy’s eyes", ’white shirt’]</span></span> <span id="S11.F14.pic1.1.1.1.1.1.14"><span id="S11.F14.pic1.1.1.1.1.1.14.1" style="font-size:144%;">Now, given the following caption, extract the object names in the same format:</span> <span id="S11.F14.pic1.1.1.1.1.1.14.2" style="font-size:144%;">&lt;caption&gt;</span></span> <span id="S11.F14.pic1.1.1.1.1.1.15"><span id="S11.F14.pic1.1.1.1.1.1.15.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 14: Prompt for identifying objects in text-to-image caption.

<svg height="121.87" id="S11.F15.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,121.87) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 108.09 C 0 115.7 6.17 121.87 13.78 121.87 L 673.72 121.87 C 681.33 121.87 687.5 115.7 687.5 108.09 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 108.09 C 1.97 114.62 7.26 119.9 13.78 119.9 L 673.72 119.9 C 680.24 119.9 685.53 114.62 685.53 108.09 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="94.31" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F15.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F15.pic1.1.1.1.1.1.1"><span id="S11.F15.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F15.pic1.1.1.1.1.1.2"><span id="S11.F15.pic1.1.1.1.1.1.2.1" style="font-size:144%;">Please tell me according to the instruction:</span> <span id="S11.F15.pic1.1.1.1.1.1.2.2" style="font-size:144%;">&lt;instruction&gt;</span><span id="S11.F15.pic1.1.1.1.1.1.2.3" style="font-size:144%;">. Which object is being replaced with another object? Please only answer the exact name of the two objects using the same words from the instruction. Use the format of a Python list including the two object names. The first is the ’object’ and the second is the ’another_object’.</span></span> <span id="S11.F15.pic1.1.1.1.1.1.3"><span id="S11.F15.pic1.1.1.1.1.1.3.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 15: An example of prompt for parsing the edited object. This is used when the task type is ’replace’.

<svg height="66.8" id="S11.F16.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,66.8) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 53.02 C 0 60.63 6.17 66.8 13.78 66.8 L 673.72 66.8 C 681.33 66.8 687.5 60.63 687.5 53.02 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 53.02 C 1.97 59.54 7.26 64.83 13.78 64.83 L 673.72 64.83 C 680.24 64.83 685.53 59.54 685.53 53.02 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="39.24" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F16.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F16.pic1.1.1.1.1.1.1"><span id="S11.F16.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F16.pic1.1.1.1.1.1.2"><span id="S11.F16.pic1.1.1.1.1.1.2.1" style="font-size:144%;">&lt;Image&gt;</span> <span id="S11.F16.pic1.1.1.1.1.1.2.2" style="font-size:144%;">Image</span> <span id="S11.F16.pic1.1.1.1.1.1.2.3" style="font-size:144%;">&lt;/Image&gt;</span></span> <span id="S11.F16.pic1.1.1.1.1.1.3"><span id="S11.F16.pic1.1.1.1.1.1.3.1" style="font-size:144%;">Please provide the bounding box coordinates of this sentence describes:</span> <span id="S11.F16.pic1.1.1.1.1.1.3.2" style="font-size:144%;">&lt;object_name&gt;</span></span> <span id="S11.F16.pic1.1.1.1.1.1.4"><span id="S11.F16.pic1.1.1.1.1.1.4.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 16: Prompt for grounding object. This works for both text-to-image and image editing data.

<svg height="104.99" id="S11.F17.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,104.99) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 91.21 C 0 98.82 6.17 104.99 13.78 104.99 L 673.72 104.99 C 681.33 104.99 687.5 98.82 687.5 91.21 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 91.21 C 1.97 97.73 7.26 103.02 13.78 103.02 L 673.72 103.02 C 680.24 103.02 685.53 97.73 685.53 91.21 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="77.43" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F17.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F17.pic1.1.1.1.1.1.1"><span id="S11.F17.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F17.pic1.1.1.1.1.1.2"><span id="S11.F17.pic1.1.1.1.1.1.2.1" style="font-size:144%;">&lt;Image&gt;</span> <span id="S11.F17.pic1.1.1.1.1.1.2.2" style="font-size:144%;">Image</span> <span id="S11.F17.pic1.1.1.1.1.1.2.3" style="font-size:144%;">&lt;/Image&gt;</span></span> <span id="S11.F17.pic1.1.1.1.1.1.3"><span id="S11.F17.pic1.1.1.1.1.1.3.1" style="font-size:144%;">You are an AI visual assistant, and you are seeing a single image. Please describe this image in one paragraph using no more than two sentences. Always remember to include describing</span> <span id="S11.F17.pic1.1.1.1.1.1.3.2" style="font-size:144%;">&lt;object_name&gt;</span> <span id="S11.F17.pic1.1.1.1.1.1.3.3" style="font-size:144%;">in the image.</span></span> <span id="S11.F17.pic1.1.1.1.1.1.4"><span id="S11.F17.pic1.1.1.1.1.1.4.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 17: Prompt for image description for image editing data.

<svg height="66.58" id="S11.F18.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,66.58) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 52.8 C 0 60.41 6.17 66.58 13.78 66.58 L 673.72 66.58 C 681.33 66.58 687.5 60.41 687.5 52.8 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 52.8 C 1.97 59.32 7.26 64.61 13.78 64.61 L 673.72 64.61 C 680.24 64.61 685.53 59.32 685.53 52.8 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="39.02" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F18.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F18.pic1.1.1.1.1.1.1"><span id="S11.F18.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F18.pic1.1.1.1.1.1.2"><span id="S11.F18.pic1.1.1.1.1.1.2.1" style="font-size:144%;">&lt;Image&gt;</span> <span id="S11.F18.pic1.1.1.1.1.1.2.2" style="font-size:144%;">Cropped Image</span> <span id="S11.F18.pic1.1.1.1.1.1.2.3" style="font-size:144%;">&lt;/Image&gt;</span></span> <span id="S11.F18.pic1.1.1.1.1.1.3"><span id="S11.F18.pic1.1.1.1.1.1.3.1" style="font-size:144%;">Please describe</span> <span id="S11.F18.pic1.1.1.1.1.1.3.2" style="font-size:144%;">&lt;object_name&gt;</span> <span id="S11.F18.pic1.1.1.1.1.1.3.3" style="font-size:144%;">briefly in several words no more than one sentence.</span></span> <span id="S11.F18.pic1.1.1.1.1.1.4"><span id="S11.F18.pic1.1.1.1.1.1.4.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 18: Prompt for cropped image object description for image editing.

<svg height="246.18" id="S11.F19.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,246.18) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 232.4 C 0 240.01 6.17 246.18 13.78 246.18 L 673.72 246.18 C 681.33 246.18 687.5 240.01 687.5 232.4 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 232.4 C 1.97 238.93 7.26 244.22 13.78 244.22 L 673.72 244.22 C 680.24 244.22 685.53 238.93 685.53 232.4 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="218.62" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F19.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F19.pic1.1.1.1.1.1.1"><span id="S11.F19.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F19.pic1.1.1.1.1.1.2"><span id="S11.F19.pic1.1.1.1.1.1.2.1" style="font-size:144%;">You are a helpful visual assistant. I have an image editing data with the original instruction:</span> <span id="S11.F19.pic1.1.1.1.1.1.2.2" style="font-size:144%;">&lt;instructions&gt;</span><span id="S11.F19.pic1.1.1.1.1.1.2.3" style="font-size:144%;">. I want to augment the instruction to obtain more free-language format instructions.</span></span> <span id="S11.F19.pic1.1.1.1.1.1.3"><span id="S11.F19.pic1.1.1.1.1.1.3.1" style="font-size:144%;">Your task is to rewrite this original instruction in English into three distinct, human-like, free-form instructions that convey the same meaning but use varied language and phrasing. The new instructions should reflect how humans might naturally request image edits.</span></span> <span id="S11.F19.pic1.1.1.1.1.1.4"><span id="S11.F19.pic1.1.1.1.1.1.4.1" style="font-size:144%;">Please provide me with three more instructions that have the same meaning as the original instruction but in a more free-language format. The new instruction can be in any format that a human might input as an editing instruction. The first instruction should be relatively concise.</span></span> <span id="S11.F19.pic1.1.1.1.1.1.5"><span id="S11.F19.pic1.1.1.1.1.1.5.1" style="font-size:144%;">Use the format of a Python list which includes three items as strings.</span></span> <span id="S11.F19.pic1.1.1.1.1.1.6"><span id="S11.F19.pic1.1.1.1.1.1.6.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 19: Prompt for reinstruction for image editing data.

<svg height="1197.56" id="S11.F20.pic1" overflow="visible" version="1.1" width="687.5"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,1197.56) matrix(1 0 0 -1 0 0)"><g fill="#000000" fill-opacity="1.0"><path d="M 0 13.78 L 0 1183.78 C 0 1191.39 6.17 1197.56 13.78 1197.56 L 673.72 1197.56 C 681.33 1197.56 687.5 1191.39 687.5 1183.78 L 687.5 13.78 C 687.5 6.17 681.33 0 673.72 0 L 13.78 0 C 6.17 0 0 6.17 0 13.78 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0"><path d="M 1.97 13.78 L 1.97 1183.78 C 1.97 1190.3 7.26 1195.59 13.78 1195.59 L 673.72 1195.59 C 680.24 1195.59 685.53 1190.3 685.53 1183.78 L 685.53 13.78 C 685.53 7.26 680.24 1.97 673.72 1.97 L 13.78 1.97 C 7.26 1.97 1.97 7.26 1.97 13.78 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 13.78 13.78)"><foreignObject color="#000000" height="1170" overflow="visible" transform="matrix(1 0 0 -1 0 16.6)" width="659.94"><span id="S11.F20.pic1.1.1.1.1.1" style="width:476.9pt;"><span id="S11.F20.pic1.1.1.1.1.1.1"><span id="S11.F20.pic1.1.1.1.1.1.1.1" style="font-size:144%;">Human:</span></span> <span id="S11.F20.pic1.1.1.1.1.1.2"><span id="S11.F20.pic1.1.1.1.1.1.2.1" style="font-size:144%;">You are a helpful assistant for a designer. I have image editing data with the following information: instruction:</span> <span id="S11.F20.pic1.1.1.1.1.1.2.2" style="font-size:144%;">&lt;instructions&gt;</span><span id="S11.F20.pic1.1.1.1.1.1.2.3" style="font-size:144%;">, description of source image:</span> <span id="S11.F20.pic1.1.1.1.1.1.2.4" style="font-size:144%;">&lt;source_desc&gt;</span><span id="S11.F20.pic1.1.1.1.1.1.2.5" style="font-size:144%;">, description of target image:</span> <span id="S11.F20.pic1.1.1.1.1.1.2.6" style="font-size:144%;">&lt;target_desc&gt;</span><span id="S11.F20.pic1.1.1.1.1.1.2.7" style="font-size:144%;">,</span> <span id="S11.F20.pic1.1.1.1.1.1.2.8" style="font-size:144%;">&lt;coord&gt;</span> <span id="S11.F20.pic1.1.1.1.1.1.2.10" style="font-size:144%;">&lt;object_desc&gt;</span> <span id="S11.F20.pic1.1.1.1.1.1.2.11" style="font-size:144%;">Assume you are a visual assistant with access to the edit instruction and the source image. Your task is to provide a step-by-step chain of thought for the image editing process which only includes the image editing processes. The chain of thought can includes the following several type steps (can not in this order, not includes these words in the answer): Describe the source image; the object to be edited; the specific area to be edited; Identify the specific changes to be made; Describe the image after the edit. All information besides the instruction should be considered as derived from the source image. The output is meant to train a multi-modal large language model that takes the source image and instruction as input, generates the editing chain of thought, and then outputs the edited image. Therefore, your response should consider this application and provide clear, concise reasoning in numbered steps (1. 2. 3. … etc). The response should be purely reasoning text and formatted succinctly. Ensure your answer be brief and few steps.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.3"><span id="S11.F20.pic1.1.1.1.1.1.3.1" style="font-size:144%;">In context learning, example 1:</span></span> <span id="S11.F20.pic1.1.1.1.1.1.4"><span id="S11.F20.pic1.1.1.1.1.1.4.1" style="font-size:144%;">1. The source image shows a grand, classical building with intricate stone carvings and statues. One prominent statue, a female figure, stands on a pedestal, holding a torch and a book. The building features arched windows and a sign that reads ”Learning Center.”</span></span> <span id="S11.F20.pic1.1.1.1.1.1.5"><span id="S11.F20.pic1.1.1.1.1.1.5.1" style="font-size:144%;">2. The object to be edited is a statue of a woman holding a torch and a book.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.6"><span id="S11.F20.pic1.1.1.1.1.1.6.1" style="font-size:144%;">3. The specific area to be edited is defined by the bounding box coordinates</span> <span id="S11.F20.pic1.1.1.1.1.1.6.2" style="font-size:144%;">((554, 166), (768, 711))</span><span id="S11.F20.pic1.1.1.1.1.1.6.3" style="font-size:144%;">, which encompasses the statue.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.7"><span id="S11.F20.pic1.1.1.1.1.1.7.1" style="font-size:144%;">4. Remove the statue completely from the image while maintaining the surrounding architectural details and other elements like the building’s facade, arched windows, and the ”Learning Center” sign. 5. The edited image will show the grand, classical building with intricate stone carvings and the ”Learning Center” sign. The statue, a female figure holding a torch and a book, will no longer be present, and the area where the statue was will appear seamless with the surrounding architecture. The building’s arched windows and stone facade will remain intact.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.8"><span id="S11.F20.pic1.1.1.1.1.1.8.1" style="font-size:144%;">In context learning, example 2:</span></span> <span id="S11.F20.pic1.1.1.1.1.1.9"><span id="S11.F20.pic1.1.1.1.1.1.9.1" style="font-size:144%;">1. The source image depicts a snowy mountain slope with a ski board in the foreground, indicating a skiing or snowboarding activity. The background features a clear blue sky and rocky terrain, suggesting a high-altitude or alpine setting.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.10"><span id="S11.F20.pic1.1.1.1.1.1.10.1" style="font-size:144%;">2. The inserted object is a skier in a black jacket, complete with goggles, sitting on a snowboard. This skier will be positioned in the center of the slope, facing downhill, sitting on a snowboard.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.11"><span id="S11.F20.pic1.1.1.1.1.1.11.1" style="font-size:144%;">3. The specific area to be edited is within the bounding box</span> <span id="S11.F20.pic1.1.1.1.1.1.11.2" style="font-size:144%;">((382, 303), (782, 813))</span><span id="S11.F20.pic1.1.1.1.1.1.11.3" style="font-size:144%;">, where the current object (a ski board) is located. This area needs to be replaced with the new skier.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.12"><span id="S11.F20.pic1.1.1.1.1.1.12.1" style="font-size:144%;">4. The image now shows a skier dressed in a black jacket and goggles, sitting on a snowboard on a snowy slope. The background features a clear blue sky and rocky terrain, with other skiers and equipment visible in the distance. The skier is positioned in the middle of the slope, looking downhill, seamlessly blending with the existing scene.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.13"><span id="S11.F20.pic1.1.1.1.1.1.13.1" style="font-size:144%;">In context learning, example 3:</span></span> <span id="S11.F20.pic1.1.1.1.1.1.14"><span id="S11.F20.pic1.1.1.1.1.1.14.1" style="font-size:144%;">1. The source image depicts a group of women and a child standing on a beach, all dressed in vibrant, summery outfits. The scene is bright and cheerful, with the ocean and sky forming a picturesque backdrop. The style of the image is casual and candid, capturing a moment of joy and togetherness.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.15"><span id="S11.F20.pic1.1.1.1.1.1.15.1" style="font-size:144%;">2. The edited area is</span> <span id="S11.F20.pic1.1.1.1.1.1.15.2" style="font-size:144%;">((0, 0), (999, 999))</span><span id="S11.F20.pic1.1.1.1.1.1.15.3" style="font-size:144%;">, which is the whole image. The object to be edited is the group of women and the child, along with the beach and the background elements. These need to be transformed into a traditional Chinese ink painting style.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.16"><span id="S11.F20.pic1.1.1.1.1.1.16.1" style="font-size:144%;">3. After the edit, the image will depict a group of women and a child standing in a traditional Chinese ink painting style, dressed in elegant, flowing garments. They will be positioned against a backdrop of serene mountains and a tranquil sea, with the overall composition reflecting the classical and detailed style of traditional Chinese ink paintings.</span></span> <span id="S11.F20.pic1.1.1.1.1.1.17"><span id="S11.F20.pic1.1.1.1.1.1.17.1" style="font-size:144%;">Assistant:</span></span></span></foreignObject></g></g></svg>

Figure 20: In-context assembling GoT prompt for image editing data.

[^1]: Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

[^2]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv preprint arXiv:2308.12966*, 2023.

[^3]: Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

[^4]: Frederic Boesel and Robin Rombach. Improving image editing models with generative data refinement. In *The Second Tiny Papers Track at ICLR 2024*, 2024.

[^5]: Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 18392–18402, 2023.

[^6]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.

[^7]: Rongyao Fang, Shilin Yan, Zhaoyang Huang, Jingqiu Zhou, Hao Tian, Jifeng Dai, and Hongsheng Li. Instructseq: Unifying vision tasks with instruction-conditioned multi-modal sequence generation. *arXiv preprint arXiv:2311.18835*, 2023.

[^8]: Rongyao Fang, Chengqi Duan, Kun Wang, Hao Li, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Hongsheng Li, and Xihui Liu. Puma: Empowering unified mllm with multi-granular visual generation. *arXiv preprint arXiv:2410.13861*, 2024.

[^9]: Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. *Advances in Neural Information Processing Systems*, 36:18225–18250, 2023.

[^10]: Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. *arXiv preprint arXiv:2309.17102*, 2023.

[^11]: Yulu Gan, Sungwoo Park, Alexander Schubert, Anthony Philippakis, and Ahmed M Alaa. Instructcv: Instruction-tuned text-to-image diffusion models as vision generalists. *arXiv preprint arXiv:2310.00390*, 2023.

[^12]: Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. *arXiv preprint arXiv:2405.04007*, 2024a.

[^13]: Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. *arXiv preprint arXiv:2404.14396*, 2024b.

[^14]: Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. *Advances in Neural Information Processing Systems*, 36:52132–52152, 2023.

[^15]: Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

[^16]: Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2025.

[^17]: Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8362–8371, 2024.

[^18]: Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.

[^19]: Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. *arXiv preprint arXiv:2312.14867*, 2023a.

[^20]: Max Ku, Tianle Li, Kai Zhang, Yujie Lu, Xingyu Fu, Wenwen Zhuang, and Wenhu Chen. Imagenhub: Standardizing the evaluation of conditional image generation models. *arXiv preprint arXiv:2310.01596*, 2023b.

[^21]: Black Forest Labs. Flux. [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux), 2024.

[^22]: Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 22511–22521, 2023.

[^23]: Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. *arXiv preprint arXiv:2305.13655*, 2023.

[^24]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in neural information processing systems*, 36:34892–34916, 2023.

[^25]: Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024a.

[^26]: Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. *arXiv e-prints*, pages arXiv–2402, 2024b.

[^27]: Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. *arXiv preprint arXiv:2411.07975*, 2024.

[^28]: Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In *Proceedings of the AAAI conference on artificial intelligence*, pages 4296–4304, 2024.

[^29]: Trong-Tung Nguyen, Duc-Anh Nguyen, Anh Tran, and Cuong Pham. Flexedit: Flexible and controllable diffusion-based object-centric image editing. *arXiv preprint arXiv:2403.18605*, 2024.

[^30]: Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. *arXiv preprint arXiv:2112.10741*, 2021.

[^31]: Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. *arXiv preprint arXiv:2306.14824*, 2023.

[^32]: Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023.

[^33]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PmLR, 2021.

[^34]: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140):1–67, 2020.

[^35]: Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In *International conference on machine learning*, pages 8821–8831. Pmlr, 2021.

[^36]: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.

[^37]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022.

[^38]: Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022.

[^39]: Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. *Advances in neural information processing systems*, 35:25278–25294, 2022.

[^40]: Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8871–8879, 2024.

[^41]: Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. *Advances in neural information processing systems*, 36:49659–49678, 2023.

[^42]: Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. *arXiv preprint arXiv:2406.06525*, 2024a.

[^43]: Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14398–14409, 2024b.

[^44]: Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. *arXiv preprint arXiv:2405.09818*, 2024.

[^45]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[^46]: Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024a.

[^47]: Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. *arXiv preprint arXiv:2409.18869*, 2024b.

[^48]: Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. *arXiv preprint arXiv:2411.07199*, 2024.

[^49]: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022.

[^50]: Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation, 2024a. *URL https://arxiv. org/abs/2410.13848*, 2024.

[^51]: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. *arXiv preprint arXiv:2412.15115*, 2024a.

[^52]: Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In *Forty-first International Conference on Machine Learning*, 2024b.

[^53]: Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. *Advances in Neural Information Processing Systems*, 36:31428–31449, 2023a.

[^54]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 3836–3847, 2023b.

[^55]: Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. *arXiv preprint arXiv:2406.18583*, 2024.