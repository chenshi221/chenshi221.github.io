---
title: "AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea"
source: "https://arxiv.org/html/2411.15738v3"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/04_%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E4%B8%8E%E7%BC%96%E8%BE%91/AnyEdit%2C%20Mastering%20Unified%20High-Quality%20Image%20Editing%20for%20Any%20Idea%2C%20Qifan%20Yu%20et%20al.%2C%202024.no_watermark.zh-CN.dual.pdf"
---
Qifan Yu <sup>1</sup> <sup>1</sup>  Wei Chow <sup>1</sup> <sup>1</sup>  Zhongqi Yue <sup>2</sup> <sup>1</sup>  Kaihang Pan <sup>1</sup>  Yang Wu <sup>3</sup>  Xiaoyang Wan <sup>1</sup>  
Juncheng Li <sup>1</sup> <sup>2</sup>  Siliang Tang <sup>1</sup>  Hanwang Zhang <sup>2</sup>  Yueting Zhuang <sup>1</sup> <sup>2</sup>  
<sup>1</sup> Zhejiang University, <sup>2</sup> Nanyang Technological University, <sup>3</sup> Ant Group  
{yuqifan, 3210103790, kaihangpan, junchengli, siliang, yzhuang}@zju.edu.cn  
nickyuezhongqi@gmail.com, wy306396@antgroup.com  
  
[https://dcd-anyedit.github.io/](https://dcd-anyedit.github.io/)

###### Abstract

Instruction-based image editing aims to modify specific image elements with natural language instructions. However, current models in this domain often struggle to execute complex user instructions accurately, as they are trained on low-quality data with limited editing types. We present AnyEdit, a comprehensive multi-modal instruction editing dataset, comprising 2.5 million high-quality editing pairs spanning over 20 editing types and five domains. We ensure the diversity and quality of the AnyEdit collection through three aspects: initial data diversity, adaptive editing process, and automated selection of editing results. Using the dataset, we further train a novel AnyEdit Stable Diffusion with task-aware routing and learnable task embedding for unified image editing. Comprehensive experiments on three benchmark datasets show that AnyEdit consistently boosts the performance of diffusion-based editing models. This presents prospects for developing instruction-driven image editing models that support human creativity.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x1.png)

Figure 1: Examples of AnyEdit at scale. We comprehensively categorize image editing tasks into 5 groups based on different editing capabilities: (a) Local Editing which focuses on region-based editing ( green area ); (b) Global Editing which focuses on the full range of image rendering ( yellow area ); (c) Camera Move Editing which focuses on viewpoints changing instead of scenes ( gray area ); (d) Implicit Editing which requires commonsense knowledge to complete complex editing ( orange area ); (e) Visual Editing which encompasses additional visual inputs, addressing the requirements for multi-modal editing ( blue area ).

## 1 Introduction

With the development of multi-modal datasets [^36] [^63] and generative frameworks [^20] [^14], recent text-to-image generation (T2I) models [^3] [^49] [^55] [^58] [^61] [^79] [^75] [^2] have shown promising results in generating high-fidelity photo-realistic images. To enhance the controllability of generated images, image editing [^89] [^8] [^37] [^13] [^65] [^22] aims to modify desired target elements and retain unrelated contents, which has achieved significant improvements for T2I models. More recently, instruction-based image editing [^22] [^5] [^6] [^13] [^75] pave the way to conveniently edit images using natural language instructions without complex descriptions or region-specific masks. However, the scarcity of high-quality instruction editing datasets makes it difficult to develop powerful instruction-guided image editing models.

Despite several prior works [^5] [^18] [^64] [^85] [^82] [^76] [^17] showcasing promising instruction-following editing datasets and achieving effective results, they still struggle with drawbacks due to limited editing types and low data quality. These methods lack support for editing instructions based on complex multi-modal perception (e.g., spatial composition [^87], viewpoint [^81], commonsense understanding [^4]) and reference-based editing instructions (e.g., customization [^9] [^74], image transfer [^83]), limiting the model’s ability to understand interleaved editing instructions and robustly execute a wide variety of editing tasks. To address the above limitations, this work, AnyEdit, first provides a comprehensive catalog of editing instructions and systematically introduces a unified editing framework for various editing tasks. As shown in figure 1, AnyEdit is divided into five classes based on editing capabilities, with each class containing multiple task types. Regarding the general editing scenarios in previous work [^82] [^85] [^18], AnyEdit categorizes them into Local Editing and Global Editing based on their granularity to construct more precise instructions. To accommodate the editing intentions for spatial perception, we further include Camera Movement Editing. For commonsense understanding, we introduce Implicit Editing, which replaces direct intentions with more complex instructions that require imaginative thinking. In addition, to accommodate reference inputs from the visual modality, we incorporate Visual Editing to construct multi-modal editing instructions.

Although AnyEdit has enriched editing types, there exist three principal challenges for high-quality dataset collection: (1) First, we observe that current editing datasets overlook inherent data biases when preparing initial image-text pairs [^16] [^78] [^48], leading to an imbalance of concepts within the dataset. For instance, “change the tree in the park to a camel” may bring undesired sand because “sand” and “camel” frequently co-occur in the training data. Thus, we introduce counterfactual synthetic scenes to enrich real-world data with tail concepts and their diverse combinations, achieving a balanced distribution of concepts in the dataset and enhancing generalization in editing tasks. (2) Secondly, current data collection methods struggle to accommodate diverse input formats and editing requirements due to the shared pipeline for all instructions. Therefore, we propose an adaptive editing pipeline that automatically selects the appropriate data pipeline for each task. (3) Third, there exist inherent drawbacks in the collected triplets of original images, edit instructions, and edited images due to the randomness of generation (i.e., low resolution [^63], high noise [^21], and misalignment between text and images [^67] [^70]). Thus, we develop an instruction validation pre-filter and image assessment post-filter strategy to filter out unsatisfactory results. In this way, AnyEdit automatically comprises 2.5M high-quality editing data across 25 distinct editing types (c.f., Tab. 1), showing a 28.9% improvement in visual similarity and 18.8% in semantic similarity over SOTA datasets (c.f., Tab. 3). This automated process opens up the possibility of scaling the high-quality editing dataset in a low-resource manner.

<table><tbody><tr><td rowspan="2">Editing Dataset</td><td rowspan="2">#Size</td><td rowspan="2">#Types</td><td colspan="5">Instruction Type</td><td colspan="3">Target Scenario</td><td rowspan="2">Open Source</td></tr><tr><td>Local</td><td>Global</td><td>Camera Movement</td><td>Implicit</td><td>Visual</td><td>Real Image</td><td>Synthetic Image</td><td>Synthetic Scene</td></tr><tr><td>MagicBrush <sup><a href="#fn:82">82</a></sup></td><td>10K</td><td>5</td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td></tr><tr><td>InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td><td>313K</td><td>4</td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td></tr><tr><td>EMU-Edit <sup><a href="#fn:64">64</a></sup></td><td>-</td><td>8</td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td></tr><tr><td>HQ-Edit <sup><a href="#fn:24">24</a></sup></td><td>197K</td><td>6</td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td></tr><tr><td>SEED-Data-Edit <sup><a href="#fn:18">18</a></sup></td><td>3.7M</td><td>6</td><td>✓</td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td></tr><tr><td>EditWorld <sup><a href="#fn:76">76</a></sup></td><td>8.6K</td><td>1</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td></tr><tr><td>UltraEdit <sup><a href="#fn:85">85</a></sup></td><td>4M</td><td>9</td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td><td>✓</td><td><math><semantics><mo>×</mo> <annotation>\times</annotation> <annotation>×</annotation></semantics></math></td><td>✓</td></tr><tr><td>AnyEdit</td><td>2.5M</td><td>25</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>

Table 1: Comparison of existing image editing datasets. “Real Image” means the original images are from real world, “Synthetic Image" means they are from T2I models, “Synthetic Scene” indicates both images and captions are generated to address the inherent data bias.

To fully harness the potential of AnyEdit’s high-quality editing data, we propose a novel AnyEdit Stable Diffusion approach (AnySD) that employs task-aware routing and learnable task embeddings to support various editing types within AnyEdit, thereby constructing a robust instruction-based editing model for handling diverse editing requests. The task-aware routing mechanism leverages AnyEdit’s diverse data to adjust the granularity of editing (e.g., local object or global style), while the learnable task embedding coordinates the inherent complexity across tasks within AnyEdit. In this way, AnySD fully harnesses the comprehensive and diverse data from AnyEdit to set new records on the MagicBrush and Emu-Edit benchmarks.

Furthermore, to facilitate a comprehensive evaluation of instruction-based image editing, we manually curate a challenging benchmark, AnyEdit-Test, from our automatically collected AnyEdit dataset, comprising 1,250 high-quality editing pairs in diverse scenarios, across 25 distinct categories (50 per category). We notice that (1) Existing models often fail to properly modify or cause distortions in complex tasks (e.g., action change shown in Fig. 5), exposing the limitations of current benchmarks for complex tasks. (2) Even for common tasks in AnyEdit-Test, some previous SOTA models show a notable performance drop compared to existing benchmarks (*e.g*., DINO’s decline in global editing in Tab. 5), revealing the limitations of current benchmarks in multi-scene editing. This highlights the critical role of AnyEdit-Test in thoroughly assessing the capabilities of editing models. We will release it for public research.

Our main contributions are summarized as follows:

- We systematically categorize various editing instructions in a novel perspective and innovatively introduce a unified editing framework that leverages an adaptive editing pipeline to automatically collect diverse high-quality editing data across different scenarios in a scalable way.
- We construct a multi-type, multi-scene dataset, AnyEdit, and its corresponding benchmark AnyEdit-Test for instruction-based editing in various scenarios, comprising 25 distinct and complex editing types to accommodate broader requirements of real-world editing.
- We use our proposed AnySD to fully unlock the potential of AnyEdit, achieving SOTA improvements in instruction adherence and image fidelity across diverse editing types.

## 2 Related Work

Table 1 compares the existing Instruction-based image editing datasets including InstructPix2Pix [^5], MagicBrush [^82], EMU-Edit [^64], HQ-Edit [^24], SEED-Data-Edit [^18], EDITWORLD [^76], and UltraEdit [^85]. Among these, the first four datasets use automatic methods to collect data, while the latter three incorporate human curation. InstructPix2Pix [^5] utilizes P2P [^22] for image editing, which makes it only suitable for synthetic images. MagicBrush [^82] hires crowd workers to annotate images from the MSCOCO [^36] dataset manually but only includes 10K editing pairs due to expensive labor expenses. HQ-Edit [^24] also lacks fine-grained details and realism due to its diptych generation though it exploits GPT-4V [^1] and DALL-E [^3] to enhance descriptions. Both of them are lack of generalization. With the advancements in multi-modal understanding [^34] [^33] [^35] [^31] [^51] [^80] [^52] [^50] and generation [^53] [^11] [^19] [^71], SEED-Data-Edit [^18] and UltraEdit [^85] have been recently introduced to enrich the types and turns of image editing. However, they only focus on traditional editing tasks, neglecting more complex and customized editing needs. Although EMU-Edit [^64] and EditWorld [^76] endeavor to simulate physical world editing, they lack a unified definition of implicit editing. Moreover, these datasets fail to incorporate viewpoint changes and support multi-modal inputs. In AnyEdit, we combine five distinct groups of data, covering 25 editing types, which will be released to help the community. It is worth noting that AnyEdit is the only dataset that considers the data bias and introduces counterfactual synthetic scenes to balance the distribution of the dataset.

<table><tbody><tr><td>Dataset</td><td>#Samples</td><td>#Concepts</td><td>License</td><td>Annotator</td></tr><tr><td colspan="5">Real World Image Caption Paired Dataset</td></tr><tr><td>MS COCO <sup><a href="#fn:36">36</a></sup></td><td>123,287</td><td>80</td><td>CC BY 4.0</td><td>Human</td></tr><tr><td>MVImgNet <sup><a href="#fn:81">81</a></sup></td><td>31,783</td><td>238</td><td>CC BY 4.0</td><td>Human</td></tr><tr><td>LLaVA-CC3M-Pretrain <sup><a href="#fn:38">38</a></sup></td><td>519,176</td><td>31423</td><td>Custom</td><td>GPT-4V</td></tr><tr><td colspan="5">Counterfactual Synthetic Scene Pair Dataset</td></tr><tr><td>AnyEdit-Composition</td><td>22,936</td><td>500</td><td>Custom</td><td>Composition</td></tr></tbody></table>

Table 2: Data preparation details for AnyEdit dataset collection.

## 3 AnyEdit

![Refer to caption](https://arxiv.org/html/2411.15738v3/x2.png)

Figure 2: The comprehensive construction pipeline of AnyEdit. We summarize the general pipeline into five steps: (1) General data preparation from real-world image-text pairs and synthetic scenes. (2) Diverse instruction generation using LLM to produce high-quality editing instructions. (3) Pre-filtering for instruction validation. (4) Adaptive editing pipeline tailors specific editing methods for each edit type to generate high-quality edited images. (5) Image quality assessment ensures high-quality editing pairs for the AnyEdit Dataset.

### 3.1 Editing Type Definition

To equip instruction-based editing models with comprehensive capabilities to follow any creative ideas, we compiled a multi-modal image editing dataset AnyEdit for instruction-based image editing, consisting of 2.5M high-quality editing pairs across five primary domains, as illustrated in Fig. 1. The dataset comprises editing tasks divided into five main categories, each containing various editing types: (1) Local editing: add, remove, replace, color alter, appearance alter, material change, action change, textual change, and counting. (2) Global editing: background change, tone transfer, and style change. (3) Camera movement editing: rotation change, out-painting, movement, and resize. (4) Implicit editing: implicit change and relation change. (5) Visual editing: visual reference, material transfer, and visual conditions (i.e., depth, segmentation, scribble, sketch, mask). Specifically, local editing targets specific areas of an image without altering unrelated semantic content, while global editing affects the entire image. Camera movement editing extends this concept by manipulating the viewpoint of specific objects or the whole content within the scene. Furthermore, implicit editing involves hidden changes in state or interaction patterns that require comprehension. Visual editing, on the other hand, incorporates additional visual inputs as references alongside editing instructions. Figure 1(a) shows examples of various editing types in each category and their detailed definitions are in Appendix B.1.

### 3.2 Automatic Dataset Collection

General Data Preparation. Previous studies have demonstrated that high-quality initial images facilitate the diversity of editing image creation [^18] [^85]. To address real-world user demands for image editing in complex scenarios, we collect $\sim$ 680K real-world image-caption pairs from annotated datasets (i.e., MSCOCO [^36], LLaVA-CC3M-Pretrain [^38]) and multi-view image datasets (i.e., MVImgNet [^81]). Then, we enrich those brief captions by MLLMs (e.g., VILA [^35]) for the completeness of the descriptions. However, these image-text pairs suffer from inherent data bias, leading to the model falling short in domains not well covered by general knowledge [^16] [^48]. Thus, we introduce the Counterfactual Synthetic Scene Pair Dataset to balance the data distribution of the initial image-text pairs. Specifically, we collect infrequent tail concepts from internet data and combine multiple concepts and contexts to generate a description by LLaMA-3B [^15]. Subsequently, we invoke off-the-shelf T2I models to produce the initial images. In this manner, we enrich the original dataset by incorporating rare concept combinations, resulting in $\sim$ 700K high-quality and diverse image-caption pairs for the AnyEdit dataset collection, as illustrated in Table 2.

Diverse Instruction Generation. It aims to create varied editing instructions and corresponding edited caption outputs based on the caption of the initial image. As illustrated in Figure 2, we leverage the public Llama3-8b [^15] model to convert original captions to diverse editing instructions. To address the limitations in instruction diversity and consistency when generating editing instructions, we integrate intuitive type constraints with LLM generation and employ in-context examples to develop a task-specific agent tailored to each editing type. Furthermore, we integrate the generated editing instructions with the original captions to create instruction pairs, which serve as in-context examples for iterative self-enhancement, thereby gradually increasing the diversity and complexity of the instructions. Details of the prompt constraints and examples are in Appendix B.2.

Adaptive Editing Pipelines. Traditional instruction-editing datasets [^64] [^85] rely on fixed pipelines [^22] or time-consuming manual filtering, making it difficult to efficiently generate high-quality edited images for complex editing types and various input formats. Here we propose an adaptive view of editing pipelines that customizes the edited image according to the specific type of editing. Specifically, we design 9 core pipelines for generating local, global, camera movement, implicit, and visual editing data. During image editing generation, we input editing instruction pairs along with the original images and their variants into the adaptive editing pipeline. The pipeline dynamically selects tailored solutions based on the editing type to generate images that align with the intended edits. Additionally, we incorporate extra constraints (e.g., dilated masks, layouts, and geometry guidance) into the U-Net layers within the diffusion process to achieve more precise semantic alignment and artifact reduction. Full details of the image construction process for each editing type can be found in Appendix B.3.

Data Quality Enhancement. Since the quality of editing data is critical for robust editing model training in AnyEdit, we further introduce a comprehensive filtering strategy for data quality enhancement. It consists of two steps: instruction validation pre-filter and image quality post-filter.

(i) Instruction Validation Pre-filter. We notice that partial editing instructions from LLMs sometimes introduce ambiguities that adversely impact the edited image (e.g., changing appearance in “color alter" editing or changing action of the static desk in “action change" editing). Simultaneously, low-quality initial images (e.g., low resolution, bad aspect ratio, lack of aesthetics) consistently lead to unsatisfactory editing results, even after time-consuming rounds of selection. Thus, we employ task-specific heuristic rules to validate various instructions, ensuring consistency and performing aesthetic evaluations to guarantee that aesthetically balanced images are used for the editing process.

(ii) Image Quality Post-filter. After collecting all the editing results, we conducted an automatic data evaluation to filter out unsatisfactory images that did not meet the criteria for data generation. First, we utilize CLIP filtering metrics [^64] [^85] to assess the alignment between edited images $I_{e}$ and their output captions $T_{e}$, ensuring consistency with editing instructions and desired modifications in target regions. Subsequently, we utilize CLIP image similarity to compare the original image $I_{o}$ and the edited image $I_{e}$, ensuring the fidelity of unedited image elements at the pixel level. Additionally, we compute the L1 distance between $I_{o}$ and $I_{e}$ to maintain the global structure of the original images. Finally, we employ CLIP directional similarity [^82], which quantifies the alignment between modifications in images and the associated changes in captions, to verify the AnyEdit’s instruction-following capability. We further apply the object detector to validate the presence of edited objects (in Local Editing) and the VLM to assess the modification of global images (in Global Editing).

<table><tbody><tr><td rowspan="2">Dataset</td><td colspan="2">Semantic Similarity</td><td colspan="2">Visual Similarity</td></tr><tr><td>CLIPin</td><td>CLIPout</td><td>SSIM</td><td>DINOv2</td></tr><tr><td>IP2P <sup><a href="#fn:5">5</a></sup></td><td>0.2650</td><td>0.2694</td><td>0.5826</td><td>0.6859</td></tr><tr><td>UltraEdit <sup><a href="#fn:85">85</a></sup></td><td>0.2834</td><td>0.3049</td><td>0.6401</td><td>0.7231</td></tr><tr><td>AnyEdit (Ours)</td><td>0.3288</td><td>0.3133</td><td>0.6638</td><td>0.9053</td></tr></tbody></table>

Table 3: Overall dataset quality comparison between AnyEdit and other datasets using automatic semantic and visual metrics.

### 3.3 Characteristics and Statistics

Benefiting from our promising automated dataset collection approach, AnyEdit comprises 2.5 million high-quality editing pairs across 25 distinct editing types. AnyEdit encompasses a broader range of domains, including complex editing tasks such as viewpoint editing, implicit editing, and visual editing, and it incorporates a richer variety of scenes, including conceptually rich synthetic scenes (c.f., Tab. 1). Moreover, the data distribution of AnyEdit in Fig. 1(b) reflects a broad variety of edit types, with coverage across domains. Quantitatively, we assess AnyEdit’s data quality based on semantic similarity and visual similarity metrics (c.f., Tab. 3). The significant improvement (+25.2% in DINOv2 and +16.0% in CLIPin compared to the UltraEdit) shows AnyEdit’s effectiveness in maintaining pixel-level consistency and accurately reflecting editing instructions.

## 4 Method

### 4.1 Architecture

Since AnyEdit contains a wide range of editing instructions across various domains, it holds promising potential for developing a powerful editing model to address high-quality editing tasks. However, training such a model has three extra challenges: (a) aligning the semantics of various multi-modal inputs; (b) identifying the semantic edits within each domain to control the granularity and scope of the edits; (c) coordinating the complexity of various editing tasks to prevent catastrophic forgetting. To this end, we propose a novel AnyEdit Stable Diffusion approach (AnySD) to cope with various editing tasks in the real world. As illustrated in Figure 3, AnySD includes three designs: visual prompt projector, task-aware routing, and learnable task embedding. Then, we will introduce each AnySD design.

Visual Prompt Projector. To align the semantics of multi-modal inputs, we first use a visual prompt projector to align these image features $z_{V}$ from the frozen CLIP image encoder with the instruction embeddings to obtain visual prompt $c_{V}$. We then integrate these embeddings into the timestep embeddings of the UNet through cross-attention interactions to provide visual prompt conditioning during general image editing diffusion [^5]. The diffusion objective is then adjusted as follows:

$$
L=\mathbb{E}_{\mathcal{E}(x),\mathcal{E}(c_{I}),c_{T},c_{V},\epsilon\sim%
\mathcal{N}(0,1),t}\Big{[}\|\epsilon-\epsilon_{\theta}(z_{t},t,\mathcal{E}(c_{%
I}),c_{T},c_{V}))\|_{2}^{2}\Big{]}
$$

where $\epsilon\in N(0,1)$ is the noise added by the diffusion process and $[x,c_{I},c_{T},(c_{V})]$ is a quadruple of editing instruction, with $c_{V}$ only being applicable for visual editing.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x3.png)

Figure 3: Architecture of AnySD. AnySD is a novel architecture to supports three conditions (original image, editing instruction, visual prompt) for various editing tasks.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x4.png)

Table 4: Comparison of methods on EMU-Edit 64 and MagicBrush 82 benchmark. We show performance improvements over SOTA models of the same architecture, with only training data differences.

Task-aware Routing. Since each editing type exhibits different granularities and scopes of edits, we propose a task-aware routing strategy with Mixture of Experts (MoE) [^42] blocks to meet task-specific editing requirements. Specifically, the visual condition $c_{V}$ is integrated into the frozen UNet layers with decoupled cross-attention to avoid disrupting the edit instruction condition. Each MoE block shares the same text attention layer but has diverse visual attention weights for $c_{V}$ by the router based on the task embedding. Therefore, we introduce a cross-attention layer for each expert and add the distributed results from all experts to the output of text cross-attention as follows,

$$
\displaystyle\mathbf{Z}^{new}
$$
 
$$
\displaystyle=\text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}%
\right)\mathbf{V}+\text{Softmax}\left(\frac{\mathbf{Q}(\mathbf{K}^{\prime})^{%
\top}}{\sqrt{d}}\right)\mathbf{V}^{\prime}
$$

where $\mathbf{K}^{\prime}=\bm{c}_{v}\mathbf{W}^{\prime}_{k}$ is the query matrix from the visual prompt and $\mathbf{W}^{\prime}_{k}$ is the corresponding weight matrix.

Learnable Task Embeddings. To guide the generation process toward the appropriate granularity for each editing task, we learn a task embedding $v_{i}$ during training. Unlike previous work [^64], we insert task embeddings before the MoE block rather than integrate them into the whole denoising process to alleviate query confusion between instructions and editing types. The task embedding $v_{i}$ has two main roles: (1) It concatenates with $z_{v}$ (set to zero when no visual prompt is provided) to form $c_{V}$ for MoE blocks; (2) it inputs to the router [^88], distributing weights across MoE experts. More details of architecture are in Appendix G.1

### 4.2 Training and Inference

To enhance AnySD’s ability to handle diverse conditioning for editing, we introduce CFG [^23] extending from InstructPix2Pix [^5] for three conditioning. Furthermore, we structure our training into two stages for AnySD, ensuring that the diffusion model can thoroughly understand general editing knowledge and develop fine-grained task-specific skills.

Stage I: Instruction Understanding. In this phase, we freeze the task-aware router and only pre-train the UNet layer to align with editing instructions. Additionally, We set additional conditions to zero tensors by CFG to enhance the model’s instruction-following capability.

Stage II: Task Tuning. In this phase, we further fine-tune $\mathbf{W}^{\prime}_{k}$ and $\mathbf{W}^{\prime}_{v}$ within the MoE block, task-aware router, visual prompt projector, and task embeddings to adapt the model to the task-specific editing granularity.

Inference. In the inference stage, we predict the editing type with LLMs (e.g., LLaMA-3) when given the input instruction. Then, we apply our AnySD for editing.

<table><tbody><tr><td></td><td></td><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td></tr><tr><td rowspan="7">Local</td><td>Null-Text <sup><a href="#fn:45">45</a></sup></td><td>0.773</td><td>0.270</td><td>0.245</td><td>0.641</td></tr><tr><td>InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td><td>0.753</td><td>0.274</td><td>0.164</td><td>0.562</td></tr><tr><td>MagicBrush <sup><a href="#fn:82">82</a></sup></td><td>0.823</td><td>0.293</td><td>0.120</td><td>0.698</td></tr><tr><td>HIVE <sup>w</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.814</td><td>0.294</td><td>0.184</td><td>0.651</td></tr><tr><td>HIVE <sup>c</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.845</td><td>0.299</td><td>0.114</td><td>0.766</td></tr><tr><td>UltraEdit (SD3) <sup><a href="#fn:85">85</a></sup></td><td>0.831</td><td>0.308</td><td>0.112</td><td>0.731</td></tr><tr><td>AnyEdit (Ours)</td><td>0.863</td><td>0.297</td><td>0.094</td><td>0.788</td></tr><tr><td rowspan="7">Global</td><td>Null-Text <sup><a href="#fn:45">45</a></sup></td><td>0.753</td><td>0.277</td><td>0.270</td><td>0.613</td></tr><tr><td>InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td><td>0.747</td><td>0.261</td><td>0.180</td><td>0.523</td></tr><tr><td>MagicBrush <sup><a href="#fn:82">82</a></sup></td><td>0.731</td><td>0.278</td><td>0.233</td><td>0.493</td></tr><tr><td>HIVE <sup>w</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.762</td><td>0.287</td><td>0.196</td><td>0.579</td></tr><tr><td>HIVE <sup>c</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.787</td><td>0.294</td><td>0.253</td><td>0.576</td></tr><tr><td>UltraEdit (SD3) <sup><a href="#fn:85">85</a></sup></td><td>0.772</td><td>0.297</td><td>0.191</td><td>0.619</td></tr><tr><td>AnyEdit (Ours)</td><td>0.788</td><td>0.301</td><td>0.159</td><td>0.647</td></tr><tr><td rowspan="6">Camera Movement</td><td>InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td><td>0.700</td><td>-</td><td>0.178</td><td>0.477</td></tr><tr><td>MagicBrush <sup><a href="#fn:82">82</a></sup></td><td>0.765</td><td>-</td><td>0.170</td><td>0.589</td></tr><tr><td>HIVE <sup>w</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.779</td><td>-</td><td>0.175</td><td>0.619</td></tr><tr><td>HIVE <sup>c</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.838</td><td>-</td><td>0.171</td><td>0.739</td></tr><tr><td>UltraEdit (SD3) <sup><a href="#fn:85">85</a></sup></td><td>0.802</td><td>-</td><td>0.340</td><td>0.514</td></tr><tr><td>AnyEdit (Ours)</td><td>0.833</td><td>-</td><td>0.110</td><td>0.745</td></tr><tr><td rowspan="6">Implicit</td><td>InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td><td>0.794</td><td>0.288</td><td>0.190</td><td>0.558</td></tr><tr><td>MagicBrush <sup><a href="#fn:82">82</a></sup></td><td>0.865</td><td>0.280</td><td>0.149</td><td>0.711</td></tr><tr><td>HIVE <sup>w</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.821</td><td>0.284</td><td>0.161</td><td>0.635</td></tr><tr><td>HIVE <sup>c</sup> <sup><a href="#fn:84">84</a></sup></td><td>0.862</td><td>0.284</td><td>0.137</td><td>0.728</td></tr><tr><td>UltraEdit (SD3) <sup><a href="#fn:85">85</a></sup></td><td>0.856</td><td>0.281</td><td>0.135</td><td>0.703</td></tr><tr><td>AnyEdit (Ours)</td><td>0.867</td><td>0.289</td><td>0.130</td><td>0.733</td></tr><tr><td rowspan="2">Visual</td><td>Uni-controlnet <sup><a href="#fn:86">86</a></sup></td><td>0.717</td><td>0.249</td><td>0.260</td><td>0.442</td></tr><tr><td>AnyEdit (Ours)</td><td>0.801</td><td>0.258</td><td>0.145</td><td>0.628</td></tr></tbody></table>

Table 5: Comparison of methods on AnyEdit-Test benchmark.

## 5 Experiments

In this section, we first assess AnyEdit and AnySD on popular standard editing benchmarks (§ 5.2), demonstrating the high quality of the AnyEdit dataset and the superiority of the AnySD architecture. Additionally, we extend the evaluation to the more challenging AnyEdit-Test benchmark (§ 5.3) to show the promising expandability of our approach, better aligning with the creative editing demands in real-world scenarios. We further present qualitative results (§ 5.4) and conduct in-depth analysis (§ 5.5) to illustrate the scalability and broader applicability of AnyEdit.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x5.png)

Figure 5: Qualitative evaluation of 10 distinct and complex tasks on AnyEdit-Test, such as action change (i), appearance alter (iv), rotation change (viii), counting (ix), and outpainting (x), demonstrates that our method yields promising results across these editing tasks.

### 5.1 Experimental Setup

Settings. For a fair comparison, we adopt Stable-Diffusion 1.5 [^58] as the backbone and follow the settings of InstructPix2Pix [^5] to train our AnySD. Notably, we only use data from AnyEdit for training, without incorporating any additional datasets. More details are in Appendix G.5.

Benchmarks & Metrics. We access our method across two popular benchmarks: Emu Edit Test [^64] and MagicBrush [^82]. These standard benchmarks evaluate editing models by comparing edited results with ground truths. Additionally, we manually selected 50 high-quality editing data from each editing type in AnyEdit, creating AnyEdit-Test for a more challenging and comprehensive evaluation. Notably, AnyEdit-Test is not visible during training. See Appendix E for details of AnyEdit-Test. Following prior work [^82] [^64] [^17], we adopt semantic similarity (i.e., CLIPim and CLIPout) and visual similarity (i.e., DINO and L1 distance) metrics to evaluate the effectiveness of AnySD trained on AnyEdit for instruction-based image editing.

Baselines. We use the following baselines: 1) specialized image editing methods: PnP [^68], Null-Text [^45]. 2) instruction-based methods: it directly edits images with natural language, including InstructPix2Pix [^5], MagicBrush [^82], HIVE [^84], EMU-Edit [^64], UltraEdit [^85]. 3) visual condition methods: it targets visual editing, including Uni-controlnet [^86]. More details are in Appendix G.6.

### 5.2 Main Results on Standard Image Editing

We report the standard image editing results of AnyEdit and other baselines on EMU-Edit Test and MagicBrush benchmarks in Table 4. Based on the experimental results, we have summarized the following conclusions: (i) Our SD-1.5 with AnyEdit, which only changes the training data to AnyEdit, consistently demonstrates superior semantic performance in both edit alignment and content preservation compared to SOTA methods, even without additional mask supervision (0.872 for CLIPim and 0.285 for CLIPout on the EMU-Edit Test). It highlights AnyEdit’s effectiveness in mastering high-quality image editing, validating its high-quality editing data with significant semantic alignment and underlying clear editing structure. (ii) Our AnySD model, trained on AnyEdit using the AnySD architecture, further surpasses SOTA methods in both semantic and visual similarity (0.872 of CLIPim on EMU-Edit Test and 0.881 of DINO on MagicBrush Test), setting new records on MagicBrush and Emu-Edit benchmarks. This demonstrates the superiority of AnySD in following editing instructions while preserving unchanged image elements, thanks to its task-aware architecture that learns task-specific knowledge from the diverse editing types in AnyEdit, enhancing the model’s cross-task editing capabilities.

### 5.3 Comparison on AnyEdit-Test Benchmark

Table 5 presents the results of the AnyEdit-Test benchmark, where each instruction is designed to rigorously evaluate AnyEdit’s adaptability across a wider range of challenging editing scenarios. We provide further results of each editing category in Appendix F. It can be observed that (i) most baselines struggle to effectively handle more complex editing tasks that are rarely in standard benchmarks (0.190 v.s. 0.121 on average L1), especially for implicit editing that requires reasoning abilities. This illustrates the importance of AnyEdit-Test for evaluating the performance of editing models on complex tasks. (ii) Even for common editing tasks, state-of-the-art models show a significant decline in consistency performance on AnyEdit-Test (-3.5% on CLIPim and -19.2% on DINO of UltraEdit). This underscores the limitations of existing benchmarks in evaluating multi-scene editing. (iii) In contrast, AnyEdit significantly outperforms SOTA methods across all editing categories, demonstrating its scalability and robustness in handling complex tasks across diverse scenarios. (iv) Traditional methods often struggle to handle visual editing effectively due to additional visual inputs. In such cases, even when compared to Uni-ControlNet, which is pre-trained with diverse visual conditions, AnyEdit consistently performs better in visual editing tasks. It shows the efficacy of AnyEdit in handling vision-conditioned editing instructions. We provide more qualitative evidence in Sec. 5.4.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x6.png)

Figure 6: Comparison of our image editing method against Uni-controlnet 86 for visual editing tasks on AnyEdit-Test.

### 5.4 Qualitative Evaluation

Due to the limitations of quantitative metrics in evaluating editing tasks, we perform qualitative evaluations to further assess the effectiveness of our approach, as shown in Figure 5. Our key observations are: 1) Most baseline models, including the human-tuned HIVE-c [^84] and the SOTA UltraEdit with its extensive training data [^85], still suffer from over-editing or misalignment when handling complex fine-grained instructions (e.g., “facial distortion" and “missing glasses" in Fig. 5(ii)). 2) Due to the limited diversity and quality of current datasets, previous methods (i.e., ip2p [^5], MagicBrush [^82], and UltraEdit) struggle to generalize to novel editing types in diverse scenarios (e.g., failing to follow instructions in rotation change and counting tasks, or roughly altering objects instead of its fine-grained appearance in appearance alter task). 3) In contrast, our method can effectively ensure editing accuracy in target regions and maintain consistency in irrelevant areas even without any mask guidance (Fig. 5(i), (vii)). Also, our method can automatically distinguish between foreground and background to modify the background (Fig. 5(v)). Moreover, our method successfully executes more complex editing instructions (e.g., style change in Fig. 5(vi) and outpainting in Fig. 5(x)).

Additionally, we visualize AnyEdit results on visual editing in Figure 6. In this challenging setting, UniControl can either reflect only the pixel information from the visual condition or retain the semantics of the original image without performing any edits. In contrast, for various visual instructions, AnyEdit consistently comprehends the pixel information in visual conditions and achieves reliable editing. These promising visualization results confirm the effectiveness and high quality of the diverse editing data in AnyEdit. More qualitative results are shown in Appendix H.2.

|  |  | CLIPim $\uparrow$ | CLIPout $\uparrow$ | L1 $\downarrow$ | DINO $\uparrow$ |
| --- | --- | --- | --- | --- | --- |
|  | AnySD w/ AnyEdit | 0.872 | 0.285 | 0.070 | 0.821 |
| 1 | w/o task-aware routing | 0.838 | 0.275 | 0.154 | 0.757 |
| 2 | w/o task emb | 0.859 | 0.282 | 0.107 | 0.809 |
| 3 | AnyEdit (w/o compsn.) | 0.868 | 0.271 | 0.099 | 0.785 |

Table 6: Ablation study of our method on EMU-Edit Test [^64].

### 5.5 In-depth Analysis

AnySD Architecture. We investigate the effectiveness of each component and conduct the following experiments on EMU-Edit Test benchmark: (1) We remove the text-aware routing strategy in AnySD (c.f. Rows 1 of Tab. 6) and find that it leads to significant performance degradation (0.838 v.s. 0.872 in CLIPim and 0.154 v.s. 0.070 in L1), demonstrating its crucial role for adapting diverse image editing tasks. (2) We remove task embeddings in AnySD and observe that it has little impact on semantic alignment but significantly affects visual consistency (c.f. Rows 2 of Tab. 6), suggesting that task embeddings control the perceptual granularity of pixel information during cross-attention.

Analysis of Data Scaling in AnyEdit. In Figure 4, we analyze the data scaling effect of AnyEdit on image editing capability. We observe: (1) For consistency metrics (i.e., CLIPim and DINO), performance improves progressively as data scale increases; (2) For editing accuracy metrics (i.e., CLIPout), we can achieve promising performance even with a small amount of data, indicating that AnyEdit excels in semantic alignment. (3) Furthermore, we remove the AnyEdit-Composition editing data in counterfactual synthesis scenarios, as shown in row 3 of Tab. 6, the lack of conceptual balance in AnySD hinders its ability to generalize, leading to a decline in semantic performance (-4.9% of CLIPout). This confirms the efficacy of counterfactual synthetic scenes to generalization in editing tasks.

## 6 Conclusion

In this work, we present a novel perspective for categorizing editing tasks and introduce a unified framework that exploits adaptive pipelines to construct high-quality data for diverse editing tasks in a low-resource manner. Building on this, we propose AnyEdit, a multi-type, multi-scene instruction-based editing dataset comprising 2.5M editing samples across 25 distinct types, along with its benchmark, AnyEdit-Test, enabling a more comprehensive paradigm for unified image editing. Furthermore, we develop the powerful AnySD, unlocking the full potential of AnyEdit. Extensive experiments on standard benchmarks and the challenging AnyEdit-Test demonstrate that our method excels in high-quality image editing across diverse tasks and scenarios, accurately executing complex instructions while preserving image consistency in unmodified elements.

Acknowledgment. This work was supported by the National Natural Science Foundation of China (62436007), the Key R&D Projects in Zhejiang Province (No. 2024C01106, 2025C01030), and the Zhejiang NSF (LRG25F020001). We thank all the reviewers for their valuable comments.

## References

Supplementary Material

## Appendix A Overview

In this supplementary material, we present:

- More detailed dataset collection process of AnyEdit (Section B).
- Statistic information of AnyEdit (Section C).
- Additional examples of AnyEdit (Section D).
- Detailed description of AnyEdit-Test Benchmark (Section E).
- Detailed experimental results of various editing types on AnyEdit-Test Benchmark (Section F).
- Implementation details (Section G).
- More qualitative results on various benchmarks and human evaluations (Section H).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x7.png)

Figure 7: The illustration of detailed pipelines for each editing type in the AnyEdit dataset collection.

## Appendix B Detailed Dataset Collection Process

### B.1 Editing Type Definition

Here, we explain the detailed definition of each editing task in AnyEdit, which comprises five primary categories with 25 distinct editing types, as shown in Table 13.

### B.2 Diverse Instruction Generation.

#### B.2.1 Prompt Constraints

To address the limitations in instruction diversity and consistency during the process of Instruction Generation, we use prompt constraints to guide the LLM as a task-specific agent that responds in JSON format with diverse editing types. A key innovation is incorporating a task-specific user-LLM conversational history into the prompt, where we replace direct constraints with high-quality, hand-crafted examples. This approach enables the LLM to learn from ideal responses and improve subsequent generations. Specifically, we design templates for each task instructing the LLM to respond in the required format. These templates are tailored for each task and include four core elements: input, output, editing type, and edit instruction. We also define a set of action verbs for each task, ensuring the LLM’s response aligns with our guidelines and improving the quality and consistency of the generated output. The detailed prompt constraints are shown in Table 14.

#### B.2.2 In-context Examples

As mentioned in Section 3.2, we employ in-context examples in the conversation history to develop a task-specific agent tailored to each editing type. For each new task, we initially generate a set of five in-context example instructions into prompt templates for instruction generation. After generating an editing instruction, we integrate it with its original caption to create instruction pairs, which are used to expand the in-context example pool. For each subsequent generation process, five in-context examples are randomly selected. This iterative self-enhancement mechanism exposes the generation process to a diverse range of examples, encouraging more varied and robust output responses. In this way, we maintain a cohesive conversational flow while progressively increasing the diversity and complexity of the generated instructions, facilitating the construction of the AnyEdit dataset.

### B.3 Adaptive Editing Pipelines

This section will elaborate on the pipeline implementation details for various editing tasks in the AnyEdit datasets collection. Figure 7 shows the illustrations of the main specific pipelines in our Adaptive Editing Pipeline module to construct these various high-quality editing instructions adaptively.

#### B.3.1 Local Editing

Remove. As shown in Pipeline1 of Figure 7, we first extract the mask of the edited object in the editing instruction by GroundingDINO [^40] and Segment Anything [^30]. We then generate the target image using SD-Inpaint [^58], with the original image and the mask produced during the process. To remove the target object, we set the prompt word to empty and the negative prompt word to edited object. Notably, we also apply dilation to the mask and perform Gaussian filtering to smooth it, ensuring a more natural blend with the surrounding contents. In addition, we merge the edited image, mask, and original image to retain image elements outside the content of the editing instructions.

Replace. As shown in Pipeline1 of Figure 7, the process of generating data for replace type is similar to the removal. The only difference is that we set the prompt word to the new object. In this way, our pipeline tends to produce a new object to replace the edited object instead of removing it.

Add. As shown in Pipeline1 of Figure 7, the process of generating data for add type is similar to the removal. However, since the add edit is to add a new object to the original image, its correct placement is unknown. Thus, we reverse the process by first generating the image of the output caption as the edited image and then using the ‘remove’ editing instruction to obtain the original image that does not include the newly added object. In this way, we obtain the original image and the edited image with the newly added object seamlessly integrated.

Counting. The counting-type editing introduces the concept of object quantity, performing the corresponding number of removal or additions iteratively based on the specified count in the instructions. Each step in the process is illustrated in the Pipeline1 of Figure 7.

Color Alter & Appearance Alter. As shown in Pipeline2 of Figure 7, the key aspect of editing types like color alteration and appearance alteration lies in modifying only the object’s attributes or appearance instead of altering or removing whole objects. Therefore, we introduce a Normalized Attention Difference [^43] based on input-output caption discrepancies to identify the target editing mask. Based on this, we apply InstructPix2Pix [^5] for instruction-based editing, blending the original and edited images within the masked region to produce the final result, thereby minimizing element confusion.

Action Change. To achieve complex non-rigid image editing, we introduce a joint intervention mechanism involving mutual self-attention and masked cross-attention, as shown in Pipeline3 of Figure 7. This approach addresses the limitations in the action change editing instructions, which sometimes fail to accurately execute editing intentions due to the need for fine-grained modifications.

Textual Change. To meet the demands for textual change, we collected captions containing text from the AnyWord-3M dataset [^69], namely, ArT [^10], COCO-Text [^12], RCTW [^56], LSVT [^41], MLT [^44], MTWI [^47], ReCTS [^57]. Following this, we generate editing instructions that alter only the text within the image, guided by specific type constraints and in-context examples in our diverse instruction generation. We ultimately generate corresponding images as the final result by using a text-specialized T2I model (i.e., FLUX), with both the original caption and the edited caption maintained under the same seed.

Material Change. We reuse the original and edited images from Material Transfer in Visual Editing. However, we only utilize editing instructions to convey the editing intent without using material images as references. Specifically, we will change “the material of \[object\] like the image" to “change the material of \[object\] to \[material category\]".

#### B.3.2 Global Editing

Background Change. As shown in Pipeline1 of Figure 7, we define background changes as modifications to the edited object “background". To avoid unnecessary foreground modifications, we extract and invert all foreground masks from captions, then merge them with the background mask. Similar to the replacement instructions, we also apply dilation to the merged mask and perform Gaussian filtering to eliminate artifacts in the contour.

Tone Transfer. We define three types of changing scenes (i.e., season, time, weather) involving the overall tone of the image. According to this, we generated editing instructions tailored to tone transfer and used InstructPix2Pix [^5] to edit the whole image, as shown in Pipeline1 of Figure 7.

Style Change. We collect 50 desired style images and extract 2,500 images from the MSCOCO validation set as original images. Using an API of Prisma Art, we applied style transfer to obtain the edited results. Ultimately, we only retain the intuitive style as the style changes editing instructions, such as “animated".

#### B.3.3 Camera Movement Editing

Movement & Resize. As shown in Pipeline4 of Figure 7, we first extract the foreground object and backgrounds separately according to the edited object. Here we use the “remove" operation to ensure the pixel integrity of the background after removing the foreground. Then, we utilize the crop-and-paste operation to change the size of the edited object and the position of it in the edited image.

Outpainting. To reduce the complexity of constructing data, we inversely designate the images from the initial dataset as extended images. Given the input caption, we randomly select an object within it and use GroundingDINO to extract its bounding box in the image. Then, we apply a mask to the areas outside the bounding box and obtain the original image that contains only selected elements. The extended and original images are then used to construct editing instructions for the outpainting type.

Rotation Change. Since direct perspective rotation change of images is challenging, we extract related image pairs directly from MVImgNet [^81], the Large-scale Dataset of Multi-view Images, to construct original images and edited images for rotation changes. Then, we categorize the editing instructions as “rotate the object clockwise" and “rotate the object counterclockwise" according to changes in the camera’s viewpoint, thereby constructing corresponding pairs of editing data.

#### B.3.4 Implicit Editing

Implicit Change. As shown in Pipeline5 of Figure 7, we first elicit the world knowledge from LLMs to transform implicit instructions into explicit instructions, which directly convey executable editing intentions (e.g., “Flatten the clay" directly conveys the alteration in the clay’s appearance without requiring additional interpretation). In this way, we use the instruction-based editing method to complete the explicit instructions step-by-step, thereby constructing edited images with implicit changes. We also enrich the dataset by using existing dynamic world editing datasets [^76].

Relation Change. To adjust the positional relationships of objects within images, our pipeline first generates layouts based on the original captions, as shown in Pipeline6 of Figure 7. We can swap the positional relationship between two objects in the layout space to construct the edited layout. Subsequently, we adopt attention manipulation to the layout-to-image models [^87] to generate original and edited images that alter only relative positioning without changing other content.

#### B.3.5 Visual Editing

Image Reference. We are the first to incorporate additional visual input into instruction-based image editing. To reduce the cost of automated synthetic data generation, we leverage zero-shot image customization [^9] to synthesize images containing visual concepts. We repurpose the edited objects from the remove or replace steps and the corresponding masks to guide the target positioning within the edited image. Additionally, we introduce an ID extractor to embed visual concepts into the target image and a detail extractor to preserve fine content details. Finally, We construct edited images containing the visual concepts in the image reference, as shown in the pipeline7 of Figure 7.

Material Transfer. Similar to the image reference, the material transfer requires injecting the material reference into the target image to achieve the material transfer effect. Considering the compatibility between materials and target objects, we further introduce depth estimation and latent illumination guidance for seamless material fusion. The total process is shown in the Pipeline8 of the Figure 7.

Visual Condition. To support a broader range of visual editing types, we incorporate additional condition images as reference images through tool pockets from ControlNet [^83]. We use tools to generate the corresponding conditional images and construct the corresponding editing instructions by templates. Notably, the edited images originate from other editing instructions, where the visual condition constructs new instruction pairs without generating additional edited images.

## Appendix C Statistics of AnyEdit

We present detailed dataset statistics for all editing types in AnyEdit in Table 7.

| Editing Type | #Instruction | #Image |
| --- | --- | --- |
| Local Editing |  |  |
| Remove | 116013 | 116013 |
| Replace | 97219 | 97219 |
| Add | 390049 | 390049 |
| Color Alter | 337078 | 337078 |
| Appearance Alter | 79720 | 79720 |
| Material Change | 21646 | \- |
| Action Change | 47210 | 47210 |
| Textual Change | 2500 | 2500 |
| Counting | 698 | 698 |
| Global Editing |  |  |
| Background Change | 413570 | 413570 |
| Tone Transfer | 553919 | 553919 |
| Style Change | 27488 | \- |
| Camera Movement Editing |  |  |
| Movement | 7724 | 7724 |
| Outpaint | 57462 | 57462 |
| Rotation Change | 17022 | 17022 |
| Resize | 10219 | 10219 |
| Implicit Editing |  |  |
| Implicit Change | 10000 | 10000 |
| Relation Change | 410 | 410 |
| Visual Editing |  |  |
| Visual Sketch | 55385 | \- |
| Visual Scribble | 55385 | \- |
| Visual Segmentation | 55385 | \- |
| Visual Depth | 55385 | \- |
| Visual Layout | 55385 | \- |
| Material Transfer | 21646 | 21646 |
| Image Reference | 17885 | 17885 |
| Total | 2506403 | 2180350 |

Table 7: The detailed statistics of the AnyEdit dataset.

## Appendix D More Examples of AnyEdit

See figure 8 and 9 for more data examples with various editing types in AnyEdit.

## Appendix E AnyEdit-Test Benchmark

To comprehensively evaluate AnyEdit’s capabilities across a broader range of editing tasks, we carefully selected 50 example pairs from each type of editing task supported by AnyEdit. This selection process allowed us to construct a new test set, named AnyEdit-Test, designed specifically to provide a more rigorous assessment. The resulting dataset includes diverse and representative editing challenges, offering a well-rounded evaluation benchmark better to understand AnyEdit’s performance across different task types. In this way, AnyEdit-Test not only broadens the scope of evaluation but also ensures that the test set includes a variety of editing complexities, thereby making the evaluation both more challenging and more insightful. In figures 10 to 14, the editing examples encompass all types from our AnyEdit dataset, all of which demonstrate excellent adherence to instructions and high visual fidelity. This further attests to the high quality and diversity of the editing data in AnyEdit. We will release this high-quality dataset and benchmark for community research.

## Appendix F Detailed Experiments of AnyEdit-Test

We conduct detailed quantitative evaluations of 25 editing types in AnyEdit-Test, focusing on editing accuracy and content consistency. Detailed results for the AnyEdit-Test benchmark can be seen in Table 15, 16, 17. We have the following observations: (1) Existing models often fail to ensure editing accuracy in complex tasks (*e.g*., significant reduction of CLIPim in action change, rotation, outpainting shown in Tab. 15 & 16, exposing the limitations of current benchmarks for complex tasks. For fine-grained editing tasks, models frequently struggle to maintain the integrity of image content while making precise modifications (*e.g*., L1 nearly doubled performance degradation in action change and textual change). These tasks demand both a high level of fine-grained control and the ability to preserve the original context. These limitations highlight a fundamental gap in existing benchmarks and AnyEdit-Test, which is more comprehensive for complex, real-world editing demands. (2) Even for common tasks in AnyEdit-Test, some previous SOTA models show a notable performance drop compared to existing benchmarks, revealing the limitations of current benchmarks in multi-scene editing. While many state-of-the-art models have achieved impressive results on conventional benchmarks, they struggle to generalize to more diverse, multi-scene editing tasks that are present in AnyEdit-Test. This performance drop highlights the limitations of traditional benchmarks when adapting to the increased diversity of multi-scene editing. In contrast, AnyEdit-Test introduces a broader range of editing scenes, making it a more accurate reflection of real-world scenarios.

## Appendix G Implementation Details

### G.1 AnySD Architecture

AnySD is a diffusion model designed to handle a broad range of editing tasks through language-based instructions. Given the distinct demands of each edit type, which require the model to selectively focus on different elements—such as faithfully preserving visual likeness in visual instructions or altering style while retaining the original image composition in style transfer—we adopt a Mixture of Experts (MoE) architecture [^42].

The visual condition $c_{V}$ is integrated into the pretrained UNet [^60] by the adapted modules with decoupled cross-attention to avoid disrupt the edit instruction condition. Each MoE block share the same language attention layer but diverse in the attention for $c_{V}$ and the weights are distributed by the router based on the task embedding.

In the original SD model, given the query features $\mathbf{Z}$ and the text features $\bm{z}_{t}$, the output of cross-attention $\mathbf{Z}^{\prime}$ can be defined by the following equation:

$$
\begin{split}\mathbf{Z}^{\prime}=\text{Attention}(\mathbf{Q},\mathbf{K},%
\mathbf{V})=\text{Softmax}(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}})%
\mathbf{V},\\
\end{split}
$$

where $\mathbf{Q}=\mathbf{Z}\mathbf{W}_{q}$, $\mathbf{K}=\bm{z}_{t}\mathbf{W}_{k}$, $\mathbf{V}=\bm{z}_{t}\mathbf{W}_{v}$ are the query, key, and values matrices of the attention operation respectively, and $\mathbf{W}_{q}$, $\mathbf{W}_{k}$, $\mathbf{W}_{v}$ are the weight matrices of the trainable linear projection layers.

To achieve separate attention mechinism, we add a new cross-attention layer for each cross-attention layer in the original UNet model to insert image features. Given the $\bm{c}_{V}$, the output of new cross-attention $\mathbf{Z}^{\prime\prime}$ is computed as follows:

$$
\begin{split}\mathbf{Z}^{\prime\prime}=\text{Attention}(\mathbf{Q},\mathbf{K}^%
{\prime},\mathbf{V}^{\prime})=\text{Softmax}(\frac{\mathbf{Q}(\mathbf{K}^{%
\prime})^{\top}}{\sqrt{d}})\mathbf{V}^{\prime},\\
\end{split}
$$

where, $\mathbf{K}^{\prime}=\bm{c}_{v}\mathbf{W}^{\prime}_{k}$ and $\mathbf{V}^{\prime}=\bm{c}_{v}\mathbf{W}^{\prime}_{v}$ are the query, key, and values matrices from the image features. $\mathbf{W}^{\prime}_{k}$ and $\mathbf{W}^{\prime}_{v}$ are the corresponding weight matrices. In order to speed up the convergence, $\mathbf{W}^{\prime}_{k}$ and $\mathbf{W}^{\prime}_{v}$ are initialized from $\mathbf{W}_{k}$ and $\mathbf{W}_{v}$. Then, we simply add the output of image cross-attention to the output of text cross-attention:

$$
\displaystyle\mathbf{Z}^{new}
$$
 
$$
\displaystyle=\text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}%
\right)\mathbf{V}+\text{Softmax}\left(\frac{\mathbf{Q}(\mathbf{K}^{\prime})^{%
\top}}{\sqrt{d}}\right)\mathbf{V}^{\prime}
$$

### G.2 CFG for Three Conditionings

AnySD is based on the latent diffusion model architecture [^58] [^66] [^59] to support high-resolution image generation and incorporated variational autoencoder [^29] with encoder $\mathcal{E}$ and decoder $\mathcal{D}$, with estimating the score [^26] of a data distribution. To support image conditioning, we add additional input channels to the first convolutional layer on the simple text to image diffusion model [^59], concatenating $z_{t}$ and $\mathcal{E}(c_{I})$, following InstructPix2Pix [^5].

For an image $x$, the diffusion process adds noise to the encoded latent $z=\mathcal{E}(x)$ producing a noisy latent $z_{t}$ where the noise level increases over timesteps $t\in T$. We learn a network $\epsilon_{\theta}$ that predicts the noise added to the noisy latent $z_{t}$ given original image conditioning $c_{I}$, text edit instruction conditioning $c_{T}$ and visual prompt conditioning $c_{V}$. We minimize the following latent diffusion objective:

$$
L=\mathbb{E}_{\mathcal{E}(x),\mathcal{E}(c_{I}),c_{T},\epsilon\sim\mathcal{N}(%
0,1),t}\Big{[}\|\epsilon-\epsilon_{\theta}(z_{t},t,\mathcal{E}(c_{I}),c_{T},c_%
{V}))\|_{2}^{2}\Big{]}
$$

Classifier-free diffusion guidance (CFG) [^23] [^39] effectively shifts probability mass toward data where an implicit classifier $p_{\theta}(c|z_{t})$ assigns high likelihood to the conditioning $c$. Training for unconditional denoising is done by simply setting the conditioning to a fixed null value $c=\varnothing$ at some frequency during training. At inference time, with a guidance scale $s\geq 1$, the modified score estimate $\tilde{e_{\theta}}(z_{t},c)$ is extrapolated in the direction toward the conditional $e_{\theta}(z_{t},c)$ and away from the unconditional $e_{\theta}(z_{t},\varnothing)$.

$$
\tilde{e_{\theta}}(z_{t},c)=e_{\theta}(z_{t},\varnothing)+s\cdot(e_{\theta}(z_%
{t},c)-e_{\theta}(z_{t},\varnothing))
$$

For our task, the score network $e_{\theta}(z_{t},c_{I},c_{T},c_{V})$ has three conditionings: the input image $c_{I}$, text instruction $c_{T}$ and visual prompt $c_{V}$. We introduce two guidance scales, $s_{I}$, $s_{T}$ and $s_{V}$ which can be adjusted to trade off how strongly each condition. Our modified score estimate is as follows:

$$
\begin{split}\tilde{e_{\theta}}&(z_{t},c_{I},c_{T},c_{V})=\>e_{\theta}(z_{t},%
\varnothing,\varnothing,\varnothing)\\
&+s_{I}\cdot(e_{\theta}(z_{t},c_{I},\varnothing,\varnothing)-e_{\theta}(z_{t},%
\varnothing,\varnothing,\varnothing))\\
&+s_{T}\cdot(e_{\theta}(z_{t},c_{I},c_{T},\varnothing)-e_{\theta}(z_{t},c_{I},%
\varnothing,\varnothing))\\
&+s_{V}\cdot(e_{\theta}(z_{t},c_{I},c_{T},c_{V})-e_{\theta}(z_{t},c_{I},c_{T},%
\varnothing))\end{split}
$$

### G.3 Classifier-free Guidance Details

As discussed in Section G.2, we apply classifier-free guidance with respect to three conditionings: the input image $c_{I}$, the text instruction $c_{T}$ and the visual prompt with task embedding $c_{V}$. We introduce separate guidance scales $s_{I}$, $s_{T}$ and $s_{V}$ that enable separately trading off the strength of each conditioning.

When ignoring $c_{V}$, we can have the modified score estimate as InstructPix2Pix [^5]:

$$
\begin{split}\tilde{e_{\theta}}(z_{t},c_{I},c_{T})=&\>e_{\theta}(z_{t},%
\varnothing,\varnothing)\\
&+s_{I}\cdot(e_{\theta}(z_{t},c_{I},\varnothing)-e_{\theta}(z_{t},\varnothing,%
\varnothing))\\
&+s_{T}\cdot(e_{\theta}(z_{t},c_{I},c_{T})-e_{\theta}(z_{t},c_{I},\varnothing)%
)\end{split}
$$

Below is the modified score estimate for our model with classifier-free guidance on three conditions (copied from Equation 8):

$$
\begin{split}\tilde{e_{\theta}}&(z_{t},c_{I},c_{T},c_{V})=\>e_{\theta}(z_{t},%
\varnothing,\varnothing,\varnothing)\\
&+s_{I}\cdot(e_{\theta}(z_{t},c_{I},\varnothing,\varnothing)-e_{\theta}(z_{t},%
\varnothing,\varnothing,\varnothing))\\
&+s_{T}\cdot(e_{\theta}(z_{t},c_{I},c_{T},\varnothing)-e_{\theta}(z_{t},c_{I},%
\varnothing,\varnothing))\\
&+s_{V}\cdot(e_{\theta}(z_{t},c_{I},c_{T},c_{V})-e_{\theta}(z_{t},c_{I},c_{T},%
\varnothing))\end{split}
$$

Our generative model learns $P(z|c_{I},c_{T})$, the probability distribution of image latents $z=\mathcal{E}(x)$ conditioned on an input image $c_{I}$, a text instruction $c_{T}$ and the visual prompt with task embedding $c_{V}$. We arrive at our particular classifier-free guidance formulation by expressing the conditional probability as follows:

$$
\begin{split}P(z|c_{T},c_{I},c_{V})&=\frac{P(z,c_{T},c_{I},c_{V})}{P(c_{T},c_{%
I},c_{V})}\\
&=\frac{P(c_{T}|c_{I},c_{V},z)P(c_{I}|c_{V},z)P(c_{V}|z)P(z)}{P(c_{T},c_{I},c_%
{V})}\end{split}
$$

Diffusion models estimate the score [^26] of the data distribution, i.e., the derivative of the log probability. Taking the logarithm gives us the following expression:

$$
\begin{split}\log(P(z|c_{T},c_{I},c_{V}))=&\>\log(P(c_{T}|c_{I},c_{V},z))\\
&+\log(P(c_{I}|c_{V},z))\\
&+\log(P(c_{V}|z))+\log(P(z)\\
&-\log(P(c_{T},c_{I},c_{V}))\end{split}
$$

Taking the derivative and rearranging we attain:

$$
\begin{split}\nabla_{z}\log(P(z|c_{T},c_{I},c_{V}))=&\>\nabla_{z}\log(P(z))\\
&+\nabla_{z}\log(P(c_{V}|z))\\
&+\nabla_{z}\log(P(c_{i}|c_{V},z))\\
&+\nabla_{z}\log(P(c_{T}|c_{I},c_{V},z))\end{split}
$$

This corresponds with the terms in our classifier-free guidance formulation in Equation 8.

### G.4 Supporting Tasks for AnySD

In general, the editing tasks supported by AngSD align with those listed for AnyEdit-Test. For each task, we utilize a distinct learned task embedding of size $N$ (where $N$ matches the dimensionality of CLIP).

Additionally, there are substantial differences between various types of tasks. Consequently, we employ a Mixture of Experts (MoE) framework. Specifically, our expert categorization is detailed in Table 8.

| Expert | Supporting tasks |
| --- | --- |
| Expert 1 | tone transfer, background change, style transfer, style change |
| Expert 2 | movement, outpaint, resize, rotation |
| Expert 3 | visual bbox |
| Expert 4 | visual depth |
| Expert 5 | visual material transfer |
| Expert 6 | visual reference |
| Expert 7 | visual scribble |
| Expert 8 | visual segment |
| Expert 9 | visual sketch |

Table 8: Expert division for various editing tasks of AnySD.

### G.5 Training Details

Stage I: Instruction Understanding. In this stage, we use the dataset type of background change, tone transfer, remove, replace, add, color, and appearance change in AnyEdit to enhance the model’s instruction-following capability. Following prior works [^5] [^82] [^64], we train our image editing model for 110,000 steps using four 48GB NVIDIA A6000 GPUs for 280 hours. Specifically, the training is conducted at a resolution of 256 × 256 with a total batch size of 1024 (gradient\_accumulation\_steps=2, batch\_size=128 per GPU). We apply random horizontal flip augmentation and crop augmentation, where images are first randomly resized between 256 and 288 pixels, followed by cropping to 256 × 256. The model is trained with a learning rate of $10^{-4}$, without any learning rate warm-up. We initialize our model using the EMA weights from the Stable Diffusion 1.5 checkpoint [^59] and adopt other training configurations from the publicly available Stable Diffusion codebase. Although the model is trained at a resolution of 256 × 256, it generalizes well to a resolution of 512 × 512 during inference. All results presented in this paper are generated at 512 × 512 resolution, with an Euler ancestral sampler and the denoising variance schedule proposed by [^27].

Stage II: Task Tuning. In the second stage, we train our model on the entire AnyEdit dataset to adapt the model to the task-specific editing granularity. We utilize the task embedding and each expert is described in Appendix G.4. Unlike the first stage, we do not use EMA (Exponential Moving Average) for training [^25]. Additionally, we set the training resolution to 512 × 512, compared to 256 × 256 in the first stage, to achieve better editing results for specific tasks. The model is trained with a learning rate of $10^{-4}$, without any learning rate warm-up. The second stage is trained for 400,000 steps using eight 48GB NVIDIA A6000 GPUs over approximately 150 hours.

### G.6 Baselines Details

We establish the models in Table 9 as baselines, organized into two categories: instruction-based and specific image editing methods. The former utilizes natural instructions to guide the editing process, while the latter relies on global descriptions of the target image to enable editing. Instruction-based image editing methods include InstructPix2Pix [^5], HIVE [^84], UltraEdit [^85], EMU-Edit [^64], and MagicBrush [^28]. The specific image editing methods include Null Text Inversion [^45], while the visual condition editing methods include Uni-ControlNet [^86].

Instruction-Based Editing Methods:

- InstructPix2Pix [^5]: Utilizes automatically generated instruction-based image editing data to fine-tune Stable Diffusion [^58], enabling instruction-based image editing during inference without requiring any test-time tuning. We use the official Hugging Face to implement it.
- HIVE [^84]: Trained with supplementary data akin to InstructPix2Pix, HIVE undergoes further fine-tuning using a reward model derived from human-ranked data. Notably, the edited output of HIVE is not square; instead, it is scaled to preserve the original aspect ratio, with the longer side resized to 512 pixels. We utilized two models, the weighted reward (SD1.5) and the conditional reward (SD1.5), referred to as HIVE <sup>w</sup> and HIVE <sup>c</sup>, respectively.
- UltraEdit [^85]. It is trained on nearly 4 million instruction-based editing samples based on the Stable Diffusion 3 [^62] and supports free-form and mask-form inputs to enhance editing performance. To ensure comparison fairness, we utilize its freeform model for all experiments. Notably, since it is trained on SD3, its performance cannot accurately reflect the improvements brought by its editing data.
- EMU-Edit [^64]. It is a fine-tuned editing model that integrates recognition and generation tasks. Although it provides promising results, the model is not open-sourced. Therefore, we only conduct comparisons of it and AnyEdit on public benchmarks to demonstrate the superiority of our approach.
- MagicBrush [^28]: MagicBrush curates a well-structured editing dataset with detailed human annotations and fine-tunes its model on this dataset using the InstructPix2Pix [^5] framework. Therefore, we use this as a baseline to fairly compare the improvement in editing capabilities brought by the AnyEdit dataset in our experiments.

Specific Editing Methods:

- Null Text Inversion [^45]: This method inverts the source image using the DDIM [^66] trajectory and performs edits during the denoising process by controlling cross-attention between text and image. Notably, Null Text Inversion requires that "attention replacement editing can only be applied to prompts of the same length." Therefore, if the input and output captions differ in length, we align the word count by truncating the longer caption. Additionally, it is worth mentioning that the official repository performs a center crop when processing non-square images, and we adhered to this setting.

Visual Condition Methods:

- Uni-controlnet [^86]. Uni-ControlNet categorizes conditions into two groups: local and global. By adding only two additional adapters, the cost of fine-tuning and the model size are significantly reduced. For local controls, we introduce a multi-scale conditional injection strategy, while for global controls, a global condition encoder is used to convert them into conditional tokens, which then interact with the incoming features. To let it support visual reference editing, we use the HED condition as the channel of reference image input.

| Method | Configuration |
| --- | --- |
| InstructPix2Pix [^5] | num\_inference\_steps=10,   image\_guidance\_scale=1 |
| MagicBrush [^82] | seed=42, guidance\_scale=7   num\_inference\_steps=20,   image\_guidance\_scale=1.5 |
| UltraEdit [^85] | negative\_prompt="",   num\_inference\_steps=50,   image\_guidance\_scale=1.5,   guidance\_scale=7.5 |
| HIVE <sup>w</sup> [^84] | steps=100   text\_cfg\_scale=7.5,   image\_cfg\_scale=1.5 |
| HIVE <sup>c</sup> [^84] | steps=100   text\_cfg\_scale=7.5,   image\_cfg\_scale=1.5 |
| Null-Text [^45] | cross\_replace\_steps.default=0.8,   self\_replace\_steps= 0.5,   blend\_words=None,   equilizer\_params=None |
| Uni-controlnet [^86] | num\_samples = 1   image\_resolution = 512   strength = 1   global\_strength = 1   low\_threshold = 100   high\_threshold = 200   value\_threshold = 0.1   distance\_threshold = 0.1   alpha = 6.2   ddim\_steps = 50   scale = 7.5   seed = 42   eta = 0.0   a\_prompt = ’best quality, extremely detailed’   n\_prompt = ’longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality’ |

Table 9: Configuration of the baselines for AnyEdit-Test. We strictly adhered to the default hyperparameters provided in the official repositories or Huggingface implementations of these baseline models.

### G.7 Details on Benchmarks and Metrics

Metrics and code. For metrics evaluation, we closely follow the MagicBrush evaluation script without any modifications. Following previous works [^2] [^82] [^85], we employ L1 metrics to measure pixel-level differences between the generated and ground truth images. Additionally, CLIP and DINO similarities are used to assess the overall similarity with the ground truth, while CLIP-T measures text-image alignment based on local descriptions and the CLIP embedding of generated images. Furthermore, CLIP text-image similarity between the edited image and the output caption, as well as CLIP text-image direction similarity (CLIPdir), are employed to evaluate the model’s instruction-following ability. Specifically, CLIPdir measures the agreement between changes in caption embedding and changes in image embedding. While the Emu Edit Test eliminates bias and overfitting at the image level by not providing ground truth images, the evaluation metrics still implicitly assess the model’s editing capabilities.

EMU-Edit-Test. We observe that the original EMU-Edit [^64] paper and dataset don’t specify the versions of CLIP [^54] and DINO [^7] used. To maintain consistency with other benchmarks, we follow the settings from the MagicBrush repository [^82], modifying only the evaluation dataset to EMU-Edit-Test.

MagicBrush-Test. MagicBrush is designed to evaluate both the single-turn and multi-turn image editing capabilities of models. It provides annotator-defined instructions and editing masks, along with ground truth images generated by DALLE-2 [^55], facilitating a more effective metric-based assessment of the model’s editing performance. However, the dataset suffers from inherent biases. During data collection, annotators are instructed to use the DALLE-2 image editing platform to generate the edited images, making the benchmark biased towards images and editing instructions that the DALLE-2 editor can successfully follow. This bias may limit the dataset’s diversity and complexity. The baseline results in Table 4 of the main paper correspond to EMU-Edit [^64].

## Appendix H Qualitative and Human Evaluations

### H.1 Human Evaluation

We conduct comprehensive human evaluations to assess both the consistency and image quality of generated images across three tasks: multiple-choice comparison, pairwise comparison, and individual image assessment. For each task, we randomly sample 100 images from AnyEdit-Test (excluding the visual instruction component). These images are evenly distributed among evaluators, and where applicable, we report averaged scores. Specifically, we evaluate three methods, comparing our approach against four SOTA editing methods: InstructPix2Pix [^5], MagicBrush [^82], HIVE <sup>w</sup> [^84], HIVE <sup>c</sup> [^84], UltraEdit (SD3) [^85] and our method.

Multiple-Choice Comparison. In this task, evaluators select the best-edited image based on consistency and image quality. As shown in Table 10, our method demonstrates superior performance, significantly surpassing the other methods, which emphasizes the effectiveness of training on our AnyEdit dataset. Notably, while MagicBrush and UltraEdit score highly in automated evaluations, their performance in human assessments is comparatively lower, especially in instruction consistency. This discrepancy highlights the limitations of current automatic metrics, which focus primarily on image quality and may not fully capture human preferences, underscoring the need for future research to develop more robust and aligned evaluation metrics.

|  | Consistency | Image Quality |
| --- | --- | --- |
| MagicBrush [^82] | 10 | 9 |
| HIVE <sup>w</sup> [^84] | 15 | 17 |
| HIVE <sup>c</sup> [^84] | 20 | 21 |
| UltraEdit (SD3) [^85] | 13 | 17 |
| AnySD | 42 | 36 |

Table 10: Multi-choice comparison of four methods. The numbers represent the frequency of each method being chosen as the best for each aspect.

One-on-One Comparison.The one-on-one comparison provides a detailed and nuanced assessment of the edited results by juxtaposing them with robust baselines. Evaluators are instructed to select the preferred option based on both consistency and image quality. As shown in Table 11, AnySD consistently outperforms the alternatives in both aspects, with a majority of evaluators favoring AnySD’s results in these direct comparisons.

|  | Consistency | Image Quality |
| --- | --- | --- |
| MagicBrush [^82] | 0.35 | 0.27 |
| HIVE <sup>w</sup> [^84] | 0.42 | 0.41 |
| HIVE <sup>c</sup> [^84] | 0.47 | 0.48 |
| UltraEdit (SD3) [^85] | 0.35 | 0.37 |

Table 11: One-on-one comparisons. The numbers in the table indicate the percent of each method being chosen as the better option compared with the AnySD’s results.

Individual Evaluation. The individual evaluation utilizes a 5-point Likert scale to gather subjective feedback on image quality generated by four distinct models. Evaluators rate each image from 1 to 5, focusing on both consistency and overall quality. As shown in Table 12, the results clearly indicate that AnySD outperforms the other baselines, underscoring the advantages of training or fine-tuning models on the AnyEdit dataset.

|  | Consistency | Image Quality |
| --- | --- | --- |
| MagicBrush [^82] | 3.3 | 3.1 |
| HIVE <sup>w</sup> [^84] | 4.1 | 3.8 |
| HIVE <sup>c</sup> [^84] | 4.2 | 4.2 |
| UltraEdit (SD3) [^85] | 3.7 | 4.0 |
| AnySD | 4.3 | 4.4 |

Table 12: Individual evaluation using a 5-point Likert scale. The numbers in the table represent the average scores calculated for each aspect.

### H.2 Qualitative Evaluation on Different Benchmarks

Detailed Results for EMU-Edit Test. More qualitative results of the EMU-Edit Test are shown in Figure 15. We observe that AnySD can effectively distinguish between the foreground and background of an image solely based on editing instructions, accurately modifying the background while preserving the content of the foreground.

Detailed Results for MagicBrush Benchmark. More qualitative results of the MagicBrush Test are shown in Figure 16. We visually compare the performance of our method on local editing with the SOTA mask-based Editing model (i.e., DALLE-2 [^55]). We notice that even without masks as supervision signals, our method accurately performs edits in specific regions, benefiting from the well-aligned editing data provided by AnyEdit.

Detailed Results for AnyEdit-Test. More qualitative results of the AnyEdit-Test are shown in Figure 17, 18, 19.

More qualitative results of high-quality image editing from AnySD. In Figure 20, we visualize AnySD editing results on a wide variety of images. We provide different editing instructions for the same image and observe that our method consistently achieves high-quality and fine-grained editing. For example, it successfully modifies underwater reflections and performs appearance modifications involving world knowledge. It effectively demonstrates the high quality of the AnyEdit dataset and the superiority of the AnySD architecture.

Multi-turn in MagicBrush. Figures 21 and 22 illustrate the performance of our AnySD model in multi-turn editing. Compared to Text2LIVE [^72], GLIDE [^49], InstructPix2Pix, and MagicBrush, our model demonstrates stronger consistency, maintaining greater similarity to the original image even in the final editing rounds. Our results even surpass the ground truth provided by MagicBrush, further affirming the high quality of the AnyEdit dataset.

Additional Image Editing Methods. We also evaluate our image editing model in comparison with other approaches, including Versatile Diffusion [^73], BLIP Diffusion [^32], Uni-ControlNet [^86], T2I-Adapter [^46], ControlNet Shuffle [^83], ControlNet Reference-only [^83], and IP-Adapter [^77]. The comparison results are presented in Figure 23, 24. Compared to other methods, our approach consistently produces superior results in terms of image quality and alignment with multimodal prompts.

| Type | Description |
| --- | --- |
| Local Editing |  |
| Remove | Remove a specific object in the image and fill it with a background. |
| Replace | Replace a specific object in the image with a new object. |
| Add | Inserting a new object in the image. |
| Counting | Removing a specified number of objects to satisfy the number requirement. |
| Color Alter | Altering the color of specific objects in the image. |
| Appearance Alter | Altering the appearance (e.g., decoration, texture, illumination) of specific objects in the image. |
| Action Change | Change the action of the specific object in the image. |
| Textual Change | Change the specific textual contents in the image to new textual contents |
| Material Change | Change the material of the specific object in the image. |
| Global Editing |  |
| Background Change | Modifying the background of the entire image but not affecting the foreground objects. |
| Tone Transfer | Modifying the overall tone of the image, including changes in time, weather, and seasons. |
| Style Change | Modifying the overall style of the image according to the given style word. |
| Camera Movement Editing |  |
| Movement | Move the position viewpoint of a specific object in the image to the left or right or up or down. |
| Resize | Zoom in or zoom out to a specific object in the image. |
| Outpainting | Expanding the overall viewpoint by imagining the surroundings of the image elements of the mask. |
| Rotation Change | Rotate the overall viewpoint of the image to obtain images from different perspectives. |
| Implicit Editing |  |
| Implicit Change | Implicitly altering the contents of an image necessitates comprehension rather than explicit instructions. |
| Relation Change | Change the position relationship between two objects (i.e., swap their positions) in the image. |
| Visual Editing |  |
| Image Reference | Replace a specific object in the image with the object in the reference image instead of any word. |
| Material Transfer | Transfer the material in the reference image to the specific object in the image. |
| Style Transfer | Transfer the style in the reference image to the specific image. |
| Visual Bounding Box | Utilizing bounding box images as visual conditions images to guide the removal or replacement. |
| Visual Scribble | Utilizing scribble images as visual conditions images to guide the removal or replacement. |
| Visual Segmentation | Utilizing segmentation images as visual conditions images to guide the removal or replacement. |
| Visual Sketch | Utilizing sketch images as visual conditions images to guide the removal or replacement. |
| Visual Depth | Utilizing depth images as visual conditions images to guide the removal or replacement. |

Table 13: The Definition of Editing Instruction Types in AnyEdit.

<table><tbody><tr><td>Element</td><td>Content</td></tr><tr><td>System Prompt</td><td>You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: edit, edited object, output. Here is the latest conversation between the Assistant and the User.</td></tr><tr><td>Task Description (Model)</td><td>Hi, My job is to take a given caption (input) and to output the following: an instruction for adding an object to the image (edit), the object to add (edited object), and the caption with the object (output). Please help me do it. I will give you the input, and you will help.</td></tr><tr><td>Output Format</td><td>When you reply, use the following format: {’edit’: ’<instruction>’, ’edited object’: ’<object>’, ’output’: ’<caption>’}. Construct the instruction with one of the following instruction words: [place, add, include]. Don’t include any \ or edit any actions in the instruction.</td></tr><tr><td>Initial Message from LLM</td><td>Sure, I’d be happy to help! Just provide me with the input (the original caption), and I’ll generate the instruction, edited object, and output caption for you. Let’s get started!</td></tr><tr><td>Example Input</td><td>User input: Beautiful cat with mojito sitting in a cafe on the street.</td></tr><tr><td>Example Output from LLM</td><td>{’edit’: ’add a hat to the cat’, ’edited object’: ’hat’, ’output’: ’Beautiful cat wearing a hat with mojito sitting in a cafe on the street.’}</td></tr><tr><td colspan="2">Constrained Editing Instruction Generation</td></tr><tr><td>Input</td><td>{caption} from real-world image-text pairs or counterfactual synthetic datasets with {editing type} requirements from users</td></tr><tr><td>Response</td><td>{model generated edit instructions in JSON format}. An example of editing data is: { "edit": "change the airplane to green", "edited object": "airplane", "input": "a small airplane sits stationary on a piece of concrete.", "output": "A green small airplane sits stationary on a piece of concrete.", "edit type": "color alter", "visual input": "None", "image file": "COCO-train2014-000000521165.jpg", "edited file": "xxxxx.jpg" }"</td></tr></tbody></table>

Table 14: Prompts constraints for LLMs to write edit instructions and captions.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x8.png)

Figure 8: More Examples of AnyEdit dataset (Part 1). textual instruction-based (first three columns) and visual instruction-based (last column) image editing.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x9.png)

Figure 9: More Examples of AnyEdit dataset (Part 2). textual instruction-based (first three columns) and visual instruction-based (last column) image editing.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x10.png)

Figure 10: More Examples of AnyEdit-Test with local editing categories.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x11.png)

Figure 11: More Examples of AnyEdit-Test with global editing categories.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x12.png)

Figure 12: More Examples of AnyEdit-Test with camera movement editing categories.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x13.png)

Figure 13: More Examples of AnyEdit-Test with visual editing categories.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x14.png)

Figure 14: More Examples of AnyEdit-Test with implicit editing categories.

<table><tbody><tr><td rowspan="2"></td><td colspan="9">local</td></tr><tr><td>remove</td><td>replace</td><td>add</td><td>color</td><td>appearance</td><td>material change</td><td>action</td><td>textual</td><td>counting</td></tr><tr><td colspan="10">InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.664</td><td>0.779</td><td>0.832</td><td>0.862</td><td>0.770</td><td>0.700</td><td>0.674</td><td>0.744</td><td>0.803</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.227</td><td>0.276</td><td>0.302</td><td>0.318</td><td>0.308</td><td>-</td><td>0.228</td><td>0.298</td><td>0.272</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.146</td><td>0.188</td><td>0.134</td><td>0.162</td><td>0.160</td><td>0.168</td><td>0.167</td><td>0.190</td><td>0.149</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.408</td><td>0.537</td><td>0.706</td><td>0.773</td><td>0.593</td><td>0.369</td><td>0.413</td><td>0.694</td><td>0.590</td></tr><tr><td colspan="10">MagicBrush <sup><a href="#fn:82">82</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.849</td><td>0.814</td><td>0.930</td><td>0.826</td><td>0.843</td><td>0.809</td><td>0.754</td><td>0.759</td><td>0.875</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.264</td><td>0.289</td><td>0.321</td><td>0.305</td><td>0.319</td><td>-</td><td>0.272</td><td>0.312</td><td>0.264</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.076</td><td>0.143</td><td>0.071</td><td>0.112</td><td>0.084</td><td>0.111</td><td>0.203</td><td>0.157</td><td>0.100</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.783</td><td>0.604</td><td>0.897</td><td>0.667</td><td>0.739</td><td>0.570</td><td>0.548</td><td>0.774</td><td>0.731</td></tr><tr><td colspan="10">HIVE <sup>w</sup> <sup><a href="#fn:84">84</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.750</td><td>0.788</td><td>0.914</td><td>0.853</td><td>0.819</td><td>0.764</td><td>0.826</td><td>0.801</td><td>0.866</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.237</td><td>0.282</td><td>0.312</td><td>0.307</td><td>0.313</td><td>-</td><td>0.291</td><td>0.318</td><td>0.266</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.118</td><td>0.184</td><td>0.079</td><td>0.114</td><td>0.147</td><td>0.126</td><td>0.155</td><td>0.139</td><td>0.122</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.586</td><td>0.600</td><td>0.857</td><td>0.779</td><td>0.690</td><td>0.536</td><td>0.735</td><td>0.838</td><td>0.738</td></tr><tr><td colspan="10">HIVE <sup>c</sup> <sup><a href="#fn:84">84</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.823</td><td>0.778</td><td>0.932</td><td>0.894</td><td>0.864</td><td>0.785</td><td>0.874</td><td>0.807</td><td>0.899</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.254</td><td>0.284</td><td>0.312</td><td>0.309</td><td>0.309</td><td>-</td><td>0.308</td><td>0.319</td><td>0.267</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.099</td><td>0.167</td><td>0.066</td><td>0.097</td><td>0.105</td><td>0.103</td><td>0.147</td><td>0.129</td><td>0.100</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.728</td><td>0.584</td><td>0.891</td><td>0.850</td><td>0.795</td><td>0.594</td><td>0.811</td><td>0.871</td><td>0.800</td></tr><tr><td colspan="10">UltraEdit (SD3) <sup><a href="#fn:85">85</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.806</td><td>0.805</td><td>0.925</td><td>0.851</td><td>0.817</td><td>0.764</td><td>0.827</td><td>0.854</td><td>0.880</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.262</td><td>0.295</td><td>0.323</td><td>0.320</td><td>0.320</td><td>-</td><td>0.292</td><td>0.344</td><td>0.273</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.087</td><td>0.151</td><td>0.072</td><td>0.091</td><td>0.100</td><td>0.108</td><td>0.158</td><td>0.127</td><td>0.089</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.709</td><td>0.615</td><td>0.867</td><td>0.791</td><td>0.729</td><td>0.522</td><td>0.724</td><td>0.890</td><td>0.764</td></tr><tr><td colspan="10">Null-Text <sup><a href="#fn:45">45</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.752</td><td>0.710</td><td>-</td><td>0.814</td><td>0.785</td><td>-</td><td>0.838</td><td>0.764</td><td>-</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.250</td><td>0.247</td><td>-</td><td>0.274</td><td>0.285</td><td>-</td><td>0.298</td><td>0.305</td><td>-</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.235</td><td>0.253</td><td>-</td><td>0.227</td><td>0.239</td><td>-</td><td>0.243</td><td>0.275</td><td>-</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.598</td><td>0.384</td><td>-</td><td>0.695</td><td>0.675</td><td>-</td><td>0.732</td><td>0.764</td><td>-</td></tr><tr><td colspan="10">AnySD w/ AnyEdit (Ours)</td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.851</td><td>0.853</td><td>0.946</td><td>0.896</td><td>0.877</td><td>0.811</td><td>0.873</td><td>0.763</td><td>0.898</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.265</td><td>0.292</td><td>0.322</td><td>0.313</td><td>0.309</td><td>-</td><td>0.306</td><td>0.303</td><td>0.263</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.103</td><td>0.123</td><td>0.052</td><td>0.061</td><td>0.051</td><td>0.084</td><td>0.145</td><td>0.136</td><td>0.088</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.785</td><td>0.688</td><td>0.921</td><td>0.855</td><td>0.840</td><td>0.602</td><td>0.782</td><td>0.800</td><td>0.819</td></tr></tbody></table>

Table 15: Comparison of Methods on AnyEdit-Test (Part 1). ’-’ indicates ’not applicable’.

<table><tbody><tr><td rowspan="2"></td><td colspan="3">global</td><td colspan="4">camera</td><td colspan="2">implicit</td></tr><tr><td>background</td><td>tone transfer</td><td>style change</td><td>movement</td><td>outpaint</td><td>rotation</td><td>resize</td><td>implicit</td><td>relation</td></tr><tr><td colspan="10">InstructPix2Pix <sup><a href="#fn:5">5</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.680</td><td>0.860</td><td>0.702</td><td>0.805</td><td>0.563</td><td>0.675</td><td>0.755</td><td>0.762</td><td>0.826</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.259</td><td>0.304</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.288</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.221</td><td>0.098</td><td>0.221</td><td>0.131</td><td>0.290</td><td>0.148</td><td>0.141</td><td>0.212</td><td>0.167</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.411</td><td>0.804</td><td>0.354</td><td>0.639</td><td>0.341</td><td>0.361</td><td>0.566</td><td>0.538</td><td>0.577</td></tr><tr><td colspan="10">MagicBrush <sup><a href="#fn:82">82</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.739</td><td>0.789</td><td>0.664</td><td>0.863</td><td>0.561</td><td>0.791</td><td>0.845</td><td>0.819</td><td>0.910</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.268</td><td>0.287</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.280</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.233</td><td>0.213</td><td>0.252</td><td>0.093</td><td>0.353</td><td>0.134</td><td>0.101</td><td>0.189</td><td>0.109</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.529</td><td>0.657</td><td>0.292</td><td>0.710</td><td>0.344</td><td>0.575</td><td>0.725</td><td>0.622</td><td>0.800</td></tr><tr><td colspan="10">HIVE <sup>w</sup> <sup><a href="#fn:84">84</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.764</td><td>0.816</td><td>0.706</td><td>0.872</td><td>0.582</td><td>0.774</td><td>0.888</td><td>0.784</td><td>0.858</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.280</td><td>0.293</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.284</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.202</td><td>0.175</td><td>0.212</td><td>0.131</td><td>0.328</td><td>0.135</td><td>0.107</td><td>0.202</td><td>0.119</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.635</td><td>0.719</td><td>0.383</td><td>0.732</td><td>0.328</td><td>0.620</td><td>0.796</td><td>0.572</td><td>0.697</td></tr><tr><td colspan="10">HIVE <sup>c</sup> <sup><a href="#fn:84">84</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.822</td><td>0.833</td><td>0.705</td><td>0.926</td><td>0.665</td><td>0.848</td><td>0.912</td><td>0.809</td><td>0.914</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.294</td><td>0.293</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.284</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.177</td><td>0.182</td><td>0.401</td><td>0.112</td><td>0.349</td><td>0.129</td><td>0.093</td><td>0.180</td><td>0.093</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.777</td><td>0.748</td><td>0.202</td><td>0.866</td><td>0.428</td><td>0.739</td><td>0.861</td><td>0.627</td><td>0.829</td></tr><tr><td colspan="10">UltraEdit (SD3) <sup><a href="#fn:85">85</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.790</td><td>0.795</td><td>0.730</td><td>0.867</td><td>0.705</td><td>0.765</td><td>0.872</td><td>0.825</td><td>0.887</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.293</td><td>0.301</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.281</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.181</td><td>0.184</td><td>0.208</td><td>0.106</td><td>0.372</td><td>0.139</td><td>0.086</td><td>0.176</td><td>0.093</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.701</td><td>0.709</td><td>0.448</td><td>0.762</td><td>0.612</td><td>0.523</td><td>0.813</td><td>0.642</td><td>0.764</td></tr><tr><td colspan="10">Null-Text <sup><a href="#fn:45">45</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.755</td><td>0.750</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.285</td><td>0.269</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.251</td><td>0.289</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.617</td><td>0.608</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">AnySD w/ AnyEdit (Ours)</td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.819</td><td>0.836</td><td>0.710</td><td>0.870</td><td>0.738</td><td>0.826</td><td>0.898</td><td>0.825</td><td>0.908</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.300</td><td>0.302</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.289</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.169</td><td>0.115</td><td>0.192</td><td>0.069</td><td>0.189</td><td>0.122</td><td>0.060</td><td>0.169</td><td>0.091</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.744</td><td>0.811</td><td>0.385</td><td>0.782</td><td>0.682</td><td>0.685</td><td>0.832</td><td>0.643</td><td>0.822</td></tr></tbody></table>

Table 16: Comparison of Methods on AnyEdit-Test (Part 2). ’-’ indicates ’not applicable’.

<table><tbody><tr><td rowspan="2"></td><td colspan="7">Visual</td></tr><tr><td>visual depth</td><td>visual sketch</td><td>visual scribble</td><td>visual segment</td><td>visual bbox</td><td>material transfer</td><td>visual reference</td></tr><tr><td colspan="8">Uni-controlnet <sup><a href="#fn:86">86</a></sup></td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.741</td><td>0.763</td><td>0.770</td><td>0.716</td><td>0.734</td><td>0.642</td><td>0.652</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.246</td><td>0.259</td><td>0.253</td><td>0.246</td><td>0.253</td><td>-</td><td>0.234</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.271</td><td>0.247</td><td>0.254</td><td>0.281</td><td>0.214</td><td>0.278</td><td>0.275</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.503</td><td>0.576</td><td>0.531</td><td>0.421</td><td>0.512</td><td>0.241</td><td>0.308</td></tr><tr><td colspan="8">AnySD w/ AnyEdit (Ours)</td></tr><tr><td>CLIPim <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.780</td><td>0.803</td><td>0.805</td><td>0.770</td><td>0.811</td><td>0.849</td><td>0.714</td></tr><tr><td>CLIPout <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.250</td><td>0.268</td><td>0.258</td><td>0.252</td><td>0.258</td><td>-</td><td>0.260</td></tr><tr><td>L1 <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td>0.177</td><td>0.164</td><td>0.158</td><td>0.181</td><td>0.125</td><td>0.090</td><td>0.121</td></tr><tr><td>DINO <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td>0.612</td><td>0.663</td><td>0.627</td><td>0.607</td><td>0.687</td><td>0.712</td><td>0.488</td></tr></tbody></table>

Table 17: Comparison of Methods on AnyEdit-Test (Part 3). ’-’ indicates ’not applicable’.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x15.png)

Figure 15: More qualitative results of the EMU-Edit Test for the editing of background change.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x16.png)

Figure 16: More qualitative results of the MagicBrush Test for local editing. The mask is used solely to supervise the editing process in DALLE-2 55 and is not provided as input to our method.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x17.png)

Figure 17: More qualitative evaluation of our model trained on AnyEdit across AnyEdit-Test benchmark (Part I).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x18.png)

Figure 18: More qualitative evaluation of our model trained on AnyEdit across AnyEdit-Test benchmark (Part II).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x19.png)

Figure 19: More qualitative evaluation of our model trained on AnyEdit across AnyEdit-Test benchmark (Part III).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x20.png)

Figure 20: Qualitative evaluation of using real images as user inputs for the robustness of our editing model.

![Refer to caption](https://arxiv.org/html/2411.15738v3/x23.png)

Figure 21: Qualitative evaluation of multi-turn editing scenario. We provide all baselines their desired input formats (Part I).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x24.png)

Figure 22: Qualitative evaluation of multi-turn editing scenario. We provide all baselines their desired input formats (Part II).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x25.png)

Figure 23: Comparison with more other image instruction edit methods (Part I).

![Refer to caption](https://arxiv.org/html/2411.15738v3/x27.png)

Figure 24: Comparison with more other image instruction edit methods (Part II).

[^1]: Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

[^2]: Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. *arXiv preprint arXiv:2410.08261*, 2024.

[^3]: James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. *Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf*, 2(3):8, 2023.

[^4]: Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. Breaking common sense: Whoops! a vision-and-language benchmark of synthetic and compositional images. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2616–2627, 2023.

[^5]: Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18392–18402, 2023.

[^6]: Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 22560–22570, 2023.

[^7]: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of the International Conference on Computer Vision (ICCV)*, 2021.

[^8]: Jianbo Chen, Yelong Shen, Jianfeng Gao, Jingjing Liu, and Xiaodong Liu. Language-based image editing with recurrent attentive models. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 8721–8729, 2018.

[^9]: Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6593–6602, 2024.

[^10]: Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In *2019 International Conference on Document Analysis and Recognition (ICDAR)*, pages 1571–1576. IEEE, 2019.

[^11]: Wei Chow, Juncheng Li, Qifan Yu, Kaihang Pan, Hao Fei, Zhiqi Ge, Shuai Yang, Siliang Tang, Hanwang Zhang, and Qianru Sun. Unified generative and discriminative training for multi-modal large language models. *arXiv preprint arXiv:2411.00304*, 2024.

[^12]: COCO-Text. A large-scale scene text dataset based on mscoco. https://bgshih.github.io/cocotext, 2016.

[^13]: Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance. In *European Conference on Computer Vision*, pages 88–105. Springer, 2022.

[^14]: Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. *arXiv preprint arXiv:1410.8516*, 2014.

[^15]: Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[^16]: Felix Friedrich, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Patrick Schramowski, Sasha Luccioni, and Kristian Kersting. Fair diffusion: Instructing text-to-image generation models on fairness. *arXiv preprint arXiv:2302.10893*, 2023.

[^17]: Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. *arXiv preprint arXiv:2309.17102*, 2023.

[^18]: Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. *arXiv preprint arXiv:2405.04007*, 2024a.

[^19]: Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. *arXiv preprint arXiv:2404.14396*, 2024b.

[^20]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. *Advances in neural information processing systems*, 27, 2014.

[^21]: Rashida Hasan and Cheehung Chu. Noise in datasets: What are the impacts on classification performance?\[noise in datasets: What are the impacts on classification performance?\]. In *Proceedings of the 11th International Conference on Pattern Recognition Applications and Methods*, 2022.

[^22]: Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. *arXiv preprint arXiv:2208.01626*, 2022.

[^23]: Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022.

[^24]: Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. *arXiv preprint arXiv:2404.09990*, 2024.

[^25]: J Stuart Hunter. The exponentially weighted moving average. *Journal of quality technology*, 18(4):203–210, 1986.

[^26]: Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research*, 6(4), 2005.

[^27]: Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. *Advances in neural information processing systems*, 35:26565–26577, 2022.

[^28]: Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6007–6017, 2023.

[^29]: Diederik P Kingma. Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*, 2013.

[^30]: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4015–4026, 2023.

[^31]: Juncheng Li, Xin He, Longhui Wei, Long Qian, Linchao Zhu, Lingxi Xie, Yueting Zhuang, Qi Tian, and Siliang Tang. Fine-grained semantically aligned vision-language pre-training. *Advances in neural information processing systems*, 35:7290–7303, 2022.

[^32]: Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference on machine learning*, pages 19730–19742. PMLR, 2023a.

[^33]: Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Fine-tuning multimodal llms to follow zero-shot demonstrative instructions. In *The Twelfth International Conference on Learning Representations*, 2023b.

[^34]: Juncheng Li, Siliang Tang, Linchao Zhu, Wenqiao Zhang, Yi Yang, Tat-Seng Chua, Fei Wu, and Yueting Zhuang. Variational cross-graph reasoning and adaptive structured semantics learning for compositional temporal grounding. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(10):12601–12617, 2023c.

[^35]: Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26689–26699, 2024.

[^36]: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13*, pages 740–755. Springer, 2014.

[^37]: Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, and Sanja Fidler. Editgan: High-precision semantic image editing. *Advances in Neural Information Processing Systems*, 34:16331–16345, 2021.

[^38]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in neural information processing systems*, 36, 2024.

[^39]: Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In *European Conference on Computer Vision*, pages 423–439. Springer, 2022.

[^40]: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. *arXiv preprint arXiv:2303.05499*, 2023.

[^41]: LSVT. Icdar2019 robust reading challenge on large-scale street view text with partial labeling. https://rrc.cvc.uab.es/?ch=16, 2019.

[^42]: Saeed Masoudnia and Reza Ebrahimpour. Mixture of experts: a literature survey. *Artificial Intelligence Review*, 42:275–293, 2014.

[^43]: Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G Derpanis, and Igor Gilitschenski. Watch your steps: Local image and scene editing by text instructions. In *European Conference on Computer Vision*, pages 111–129. Springer, 2025.

[^44]: MLT. Icdar 2019 robust reading challenge on multi-lingual scene text detection and recognition. https://rrc.cvc.uab.es/?ch=15, 2019.

[^45]: Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6038–6047, 2023.

[^46]: Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 4296–4304, 2024.

[^47]: MTWI. Icpr 2018 challenge on multi-type web images. https://tianchi.aliyun.com/dataset/137084, 2018.

[^48]: Ranjita Naik and Besmira Nushi. Social biases through the text-to-image generation lens. In *Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society*, pages 786–808, 2023.

[^49]: Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. *arXiv preprint arXiv:2112.10741*, 2021.

[^50]: Kaihang Pan, Juncheng Li, Hongye Song, Jun Lin, Xiaozhong Liu, and Siliang Tang. Self-supervised meta-prompt learning with meta-gradient regularization for few-shot generalization. *arXiv preprint arXiv:2303.12314*, 2023.

[^51]: Kaihang Pan, Zhaoyu Fan, Juncheng Li, Qifan Yu, Hao Fei, Siliang Tang, Richang Hong, Hanwang Zhang, and Qianru Sun. Towards unified multimodal editing with enhanced knowledge collaboration. *arXiv preprint arXiv:2409.19872*, 2024a.

[^52]: Kaihang Pan, Juncheng Li, Wenjie Wang, Hao Fei, Hongye Song, Wei Ji, Jun Lin, Xiaozhong Liu, Tat-Seng Chua, and Siliang Tang. I3: I ntent-i ntrospective retrieval conditioned on i nstructions. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1839–1849, 2024b.

[^53]: Kaihang Pan, Siliang Tang, Juncheng Li, Zhaoyu Fan, Wei Chow, Shuicheng Yan, Tat-Seng Chua, Yueting Zhuang, and Hanwang Zhang. Auto-encoding morph-tokens for multimodal llm. *arXiv preprint arXiv:2405.01926*, 2024c.

[^54]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021.

[^55]: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.

[^56]: RCTW. Icdar2017 competition on reading chinese text in the wild. https://rctw.vlrlab.net/dataset, 2017.

[^57]: ReCTS. Icdar 2019 robust reading challenge on reading chinese text on signboard. https://rrc.cvc.uab.es/?ch=12, 2019.

[^58]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022a.

[^59]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10684–10695, 2022b.

[^60]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pages 234–241. Springer, 2015.

[^61]: Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022.

[^62]: Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. *arXiv preprint arXiv:2403.12015*, 2024.

[^63]: Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. *Advances in Neural Information Processing Systems*, 35:25278–25294, 2022.

[^64]: Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8871–8879, 2024.

[^65]: Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. *arXiv preprint arXiv:2306.00983*, 2023.

[^66]: Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *arXiv preprint arXiv:2010.02502*, 2020.

[^67]: Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5238–5248, 2022.

[^68]: Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1921–1930, 2023.

[^69]: Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. 2023.

[^70]: Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2096–2105, 2023.

[^71]: Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. *arXiv preprint arXiv:2409.11340*, 2024.

[^72]: Chao Xu, Jiangning Zhang, Yue Han, Guanzhong Tian, Xianfang Zeng, Ying Tai, Yabiao Wang, Chengjie Wang, and Yong Liu. Designing one unified framework for high-fidelity face reenactment and swapping. In *European conference on computer vision*, pages 54–71. Springer, 2022.

[^73]: Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi. Versatile diffusion: Text, images and variations all in one diffusion model. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 7754–7765, 2023.

[^74]: Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18381–18391, 2023.

[^75]: Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In *Forty-first International Conference on Machine Learning*, 2024a.

[^76]: Ling Yang, Bohan Zeng, Jiaming Liu, Hong Li, Minghao Xu, Wentao Zhang, and Shuicheng Yan. Editworld: Simulating world dynamics for instruction-following image editing. *arXiv preprint arXiv:2405.14785*, 2024b.

[^77]: Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. 2023.

[^78]: Qifan Yu, Juncheng Li, Yu Wu, Siliang Tang, Wei Ji, and Yueting Zhuang. Visually-prompted language model for fine-grained scene graph generation in an open world. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 21560–21571, 2023a.

[^79]: Qifan Yu, Juncheng Li, Wentao Ye, Siliang Tang, and Yueting Zhuang. Interactive data synthesis for systematic vision adaptation via llms-aigcs collaboration. *arXiv preprint arXiv:2305.12799*, 2023b.

[^80]: Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, and Yueting Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12944–12953, 2024.

[^81]: Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset of multi-view images. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9150–9161, 2023c.

[^82]: Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. *Advances in Neural Information Processing Systems*, 36, 2024.

[^83]: Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 3836–3847, 2023a.

[^84]: Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, Caiming Xiong, and Ran Xu. Hive: Harnessing human feedback for instructional visual editing. *arXiv preprint arXiv:2303.09618*, 2023b.

[^85]: Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. *arXiv preprint arXiv:2407.05282*, 2024.

[^86]: Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K. Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. *Advances in Neural Information Processing Systems*, 2023.

[^87]: Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6818–6828, 2024.

[^88]: Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. *Advances in Neural Information Processing Systems*, 35:7103–7114, 2022.

[^89]: Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In *Proceedings of the IEEE international conference on computer vision*, pages 2223–2232, 2017.