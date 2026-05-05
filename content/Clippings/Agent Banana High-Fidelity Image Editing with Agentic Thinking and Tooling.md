---
title: "Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling"
source: "https://arxiv.org/html/2602.09084v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/04_%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E4%B8%8E%E7%BC%96%E8%BE%91/Agent%20Banana%2C%20High-Fidelity%20Image%20Editing%20with%20Agentic%20Thinking%20and%20Tooling%2C%202026.no_watermark.zh-CN.dual.pdf"
---
Ruijie Ye <sup>1,2†</sup>, Jiayi Zhang <sup>3†</sup>, Zhuoxin Liu <sup>4†</sup>, Zihao Zhu <sup>1</sup>, Siyuan Yang <sup>1</sup>,  
Li Li <sup>5</sup>, Tianfu Fu <sup>6</sup>, Franck Dernoncourt <sup>7‡</sup>, Yue Zhao <sup>5</sup>, Jiacheng Zhu <sup>8§</sup>,  
Ryan Rossi <sup>7‡</sup>, Wenhao Chai <sup>9</sup>, Zhengzhong Tu <sup>1⋆</sup>  
<sup>1</sup> TAMU   <sup>2</sup> Brown University   <sup>3</sup> UW-Madison   <sup>4</sup> UCSD   <sup>5</sup> USC   <sup>6</sup> xAI  
   <sup>7</sup> Adobe Research   <sup>8</sup> Meta AI    <sup>9</sup> Princeton University  
<sup>⋆</sup> Corresponding Author: tzz@tamu.edu. <sup>†</sup> Equal contributions.  
<sup>‡</sup> Work not done at Adobe. <sup>§</sup> Work done outside of Meta.  
Project Website: [agent-banana.github.io](https://agent-banana.github.io/)

###### Abstract

We study instruction-based image editing under professional workflows and identify three persistent challenges: (i) editors often over-edit, modifying content beyond the user’s intent; (ii) existing models are largely single-turn, while multi-turn edits can alter object faithfulness; and (iii) evaluation at around 1K resolution is misaligned with real workflows that often operate on ultra high-definition images (e.g., 4K). We propose Agent Banana, a hierarchical agentic planner–executor framework for high-fidelity, object-aware, thinking with editing. Agent Banana introduces two key mechanisms: ❶ Context Folding, which compresses long interaction histories into structured memory for stable long-horizon control, and ❷ Image Layer Decomposition, which performs localized layer-based edits to preserve non-target regions while enabling native-resolution outputs. To support rigorous evaluation, we build HDD-Bench, a high-definition, dialogue-based benchmark featuring verifiable stepwise targets and native 4K images (11.8M pixels) for diagnosing long-horizon failures. On HDD-Bench, Agent Banana achieves the best multi-turn consistency and background fidelity (e.g., IC 0.871, SSIM ${}_{\text{OM}}$ 0.84, LPIPS ${}_{\text{OM}}$ 0.12) while remaining competitive on instruction following, and also attains strong performance on standard single-turn editing benchmarks. We hope this work advances reliable, professional-grade agentic image editing and its integration into real workflows.

Figure 1: We present Agent Banana, an agentic editing system that enables high-fidelity, native-resolution image editing through reasoning-based natural-language interaction, where each edit is context-aware, logically dependent, and locally precise. In this example, the user provides a vague yet complex editing prompt, and Agent Banana iteratively refines a scene in native high resolution ($5460\times 3640$)—from a stylistic replacement (Turn 1), to attribute decoupling that preserves non-target dynamics (changing the bottle color without affecting the pouring liquid; Turn 2), and finally to retrieving prior state and adding fine details (Turn 3). The result is a professional-style workflow that resists over-editing and background drift, while faithfully preserving what should remain unchanged.

## 1 Introduction

Instruction-based image editing [^3] [^54] [^40] [^10] [^19] [^4] [^33] [^45] [^27] enables users to modify images via natural-language commands and has become a core capability of modern generative vision systems. Recent advances in foundation models—particularly diffusion [^14] [^26] and autoregressive transformers [^42] —have substantially improved both photorealism and instruction following, powering practical editing experiences in commercial systems (e.g., GPT-4o [^33], Gemini 2.5 Flash Image [^8]) and strong open-source models (e.g., Flux-1 [^18], Qwen-Image-Edit [^44]).

Despite this rapid progress, a substantial gap remains between current generative editors [^44] [^27] [^17] and the requirements of *professional* workflows. In high-stakes settings such as photography [^16], graphic design [^28], visual effects (VFX), and filmmaking [^56], users typically work on native high-resolution assets (often 4K or higher) and demand precise, localized modifications that preserve all non-target content [^17]. By contrast, today’s models often operate at reduced resolution or rely on downsampling, making it difficult to maintain fine textures and sharp boundaries. Moreover, they frequently exhibit over-editing effects, unintentionally altering regions outside the user’s intent or degrading global semantic coherence. Lastly, they struggle with complex requests that are multi-goal or sequential [^59], where success requires decomposing the instruction, verifying intermediate results, and revising earlier decisions across turns.

We argue that to bridge this gap, next-generation editing tools must satisfy four core capabilities: Intent understanding and decomposition of complex requests into atomic sub-edits; Accurate localized editing to ensure edits are precisely applied while maintaining the rest of the content unchanged, on native resolution; State tracking and rollback to retain intermediate steps across multi-turn interactions so that users (or intelligent agents) can easily revert to a previous step and re-plan the remaining steps; and High-resolution native editing to operate directly on native 4K images, preserving fine-grained textures and sharp boundaries while avoiding downsampling.

To this end, we introduce Agent Banana, an agentic, layer-aware image editing framework that couples high-level reasoning and planning with tool-use capabilities, benefiting from the rapid progress of Vision-Language Models (VLMs) in image understanding, reasoning, and tool invocation [^15] [^36] [^46] [^47] [^34] [^1]. Agent Banana decomposes “vibe”-type prompts into discrete, single-goal steps, executes these steps using a ‘Photoshop-style‘ layer isolation, masking, and local edits. Agent Banana also includes a self-reflection mechanism [^50] [^38], allowing it to retry, rollback, and replan at inference time. Crucially, Agent Banana is built around two mechanisms tailored for long-horizon, high-resolution editing: Context Folding, which compresses long interaction histories into structured memory for stable state tracking across turns, and Image Layer Decomposition, which performs edits on isolated high-resolution layers to preserve non-target content and prevent drift across iterations.

To evaluate multi-turn, high-definition editing under realistic stepwise dependencies, we build HDD-Bench, a High-Definition and Dialogue-based benchmark designed to simulate professional editing workflows. Unlike prior benchmarks that are predominantly single-turn or weakly dependent across turns [^10] [^19] [^4] [^45] [^27] [^51], HDD-Bench features logically dependent instruction chains where each turn induces a well-defined state transition and can be verified step by step. HDD-Bench benchmarks instruction adherence, edit locality, multi-turn consistency, and overall visual fidelity at native resolution. To reduce evaluation ambiguity, we further introduce a graph-based evaluation protocol that tracks object-state transitions across turns, complementing global perceptual metrics with localized, turn-level checks of whether the intended edits are applied and whether non-target regions remain preserved.

![Refer to caption](https://arxiv.org/html/2602.09084v2/x1.png)

Figure 2: Overview of the Agent Banana Framework. The system operates in a multi-turn loop (Left), comprising two core agents: a Planner that decomposes user queries into executable editing plans, and an Executor that selects tools via the MCP Server. Crucially, the Executor incorporates a self-correction mechanism (Quality Test), reiterating the editing process if the quality check fails before presenting the result to the user. (Right) Our Evaluator assesses performance by analyzing the transition between Turn n − 1 n-1 and Turn, utilizing instruction adherence checks and state tracking (JSON) to derive the final score.

## 2 Agent Banana Framework

### 2.1 Problem Setup and Motivation

We consider a multi-turn instruction-based image editing task, where the user provides a sequence of natural-language instructions $q=\{q_{1},q_{2},\dots,q_{T}\}$ and an initial image $I_{0}$. The system responds by executing a trajectory of editing steps $\tau=\{(a_{1},o_{1}),(a_{2},o_{2}),\dots,(a_{T},o_{T})\}$, where each $a_{i}$ denotes the $i$ -th action (comprising reasoning and tool invocation), and $o_{i}$ is the resulting image state. The environment dynamics can be abstracted as a transition operator $\mathcal{E}$ such that $o_{i}=\mathcal{E}(o_{i-1},a_{i})$. Following the ReAct-style paradigm [^50], the agent incrementally selects actions based on the full interaction history:

$$
P_{\theta}(\tau\mid q)\propto\prod_{i=1}^{T}\pi_{\theta}\left(a_{i}\mid q,a_{<i},o_{<i}\right).
$$

While conceptually simple, this formulation introduces two major challenges in practice: ❶ Long-horizon context overflow. As the number of editing steps increases, the agent must repeatedly condition on the entire interaction history, both textual and visual. This leads to severe token inefficiency, quickly exceeding the LLM’s context length, and introduces irrelevant noise that impairs reasoning and planning in later steps. ❷ Full-image detail degradation Existing editing tools often operate by resampling the entire image at each step, regardless of the locality of the edit. This not only wastes computation on unchanged regions, but also causes subtle degradation of fine details over time—especially in backgrounds or fixed objects—leading to accumulation of visual artifacts across turns.

### 2.2 Overview of Agent Banana

To address the challenges of context overflow and iterative degradation, we introduce Agent Banana, a hierarchical multi-agent editing framework designed for high-fidelity, multi-turn image editing at native resolution, as shown in Figure 2. The framework explicitly separates global task reasoning from low-level execution via two specialized agents:

- Planner: Performs global intent interpretation, decomposes complex user instructions into executable sub-goals, and monitors overall progress.
- Executor: Carries out atomic editing operations, invokes tools on localized image regions, and handles intermediate validation and error recovery.

This division of roles enables the system to both reason over long-horizon objectives and execute fine-grained visual edits in a scalable and interpretable manner.

Agent Banana is built around two key mechanisms that mitigate the core bottlenecks identified earlier:

- Context Folding: A hierarchical memory abstraction that compresses the growing interaction history into structured representations, enabling long-horizon planning without exceeding context limits.
- Image Layer Decomposition (ILD): A localized execution strategy that performs edits on cropped high-resolution patches (layers), preserving pixel-level fidelity in unedited regions and naturally supporting ultra-HD editing workflows.

During each interaction round, the Planner receives the user instruction and current image state, decomposes the task into sub-goals, and delegates them to the Executor. The Executor generates intermediate candidates via ILD-based editing and returns feedback. The Planner verifies whether the updated image meets the instruction goal; if not, it can replan or rollback using the maintained image state graph. This closed-loop process continues until the user’s intent is satisfied or a predefined turn limit is reached.

### 2.3 Context Folding

To effectively mitigate the exponential explosion of context in long-horizon tasks, we introduce the Context Folding mechanism. The core idea is to "fold" the raw, high-dimensional interaction history into a compact semantic representation through hierarchical abstraction and selective memory. Specifically, we decouple context information into three schemas of varying granularity: the Asset Level, the Execution Level, and the Planning Level.

#### Asset Level: ImageContext.

This is the fundamental data unit of the system, constructed by the Executor after each image generation. Instead of directly embedding high-dimensional image tokens, ImageContext abstracts the image into a lightweight semantic node, containing a unique identifier (URI), VLM-generated textual description of the content, its parent URI, and the transformation type leading to this state change. Through this text-based graph representation, the agent can track the full image evolution history with minimal context overhead while preserving the topological relationships between image states.

#### Execution Level: ToolContext.

This serves as the Transient Working Memory used by the Executor during single-step reasoning. It details the microscopic operations required to complete an atomic instruction, including tool selection, parameter configuration, the intermediate reasoning process (Thought), and references to relevant ImageContexts. ToolContext primarily facilitates error recovery and state backtracking within the current step. Once the current sub-task is completed, these trivial trial-and-error details are "folded" and do not enter the long-term global memory, thereby preventing irrelevant execution noise from interfering with the Planner.

#### Planning Level: ActionContext.

This forms the Persistent Memory established after each round of user interaction. When the Planner confirms that a series of operations successfully meets the user’s requirements, it constructs an ActionContext. This context retains only the verified effective editing path: the final intention determined by the Planner and the corresponding sequence of key ImageContexts. ActionContext essentially acts as a semantic compression of ToolContext, discarding procedural tool invocation details and preserving only high-level task semantics and result states. This ensures that the agent maintains a clear cognitive grasp of the global task state even after dozens of interaction turns, without being overwhelmed by excessive token sequences.

### 2.4 Image Layer Decomposition

To resolve the issues of detail loss and resolution limitations inherent in full-image generation, we propose the Image Layer Decomposition (ILD) mechanism. Traditional end-to-end editing models often resample the entire image, causing unintended pixel drift in unedited regions (such as the background or irrelevant objects). The ILD mechanism abandons this global operation in favor of a "decompose-edit-fuse" local processing paradigm.

Specifically, this mechanism utilizes a dynamic object-aware mask to precisely localize the target region, losslessly cropping it from the original high-resolution image into an independent layer patch. All generative editing is performed solely within the local coordinate system of this patch, thereby freezing the pixel state of the background region and substantially reducing degradation in non-target regions by avoiding full-image resampling. Upon completion of editing, the system seamlessly blends the updated patch back into the original image using Gaussian blending algorithms. Furthermore, since it only processes local patches, this mechanism naturally supports ultra-high-definition image editing beyond the model’s native resolution limits.

Based on the ILD mechanism, we define an Action Space of five atomic instructions that cover common editing needs:

- replace: Substitutes the content of the target layer with a new object using inpainting techniques while maintaining edge consistency.
- remove: Eliminates the target layer and fills the void using background completion algorithms.
- add: Generates a new layer at a specified location and performs layer superposition.
- adjust: Applies attribute transformations (e.g., color correction, style transfer) to the target layer without altering its geometry.
- undo: Rapidly rolls back to the previous image state node based on the state graph maintained in Context Folding.

These five atomic operations form the foundational capability set for Agent Banana, enabling the Planner to execute complex, composite edits by composing these primitives.

Table 1: Comparison of existing image editing datasets vs. our HDD-Bench. We compare key features including support for multi-turn interaction, high-resolution images, object-level editing granularity, reasoning capabilities, and ground-truth verification. HDD-Bench is the only benchmark encompassing all these capabilities, bridging the gap for professional-grade editing evaluation.

| Dataset | #Turn | Res. (pxs) | Obj Scale | Reason. | Verifi. |
| --- | --- | --- | --- | --- | --- |
| AnyEdit [^52] | 1.0 | 0.4M | Large | – | – |
| GEdit-Bench [^27] | 1.0 | 3.4M | Large | – | – |
| SEED-DataEdit [^12] | 3.8 | 1.1M | Medium | ✓ | – |
| ImgEdit-Bench [^51] | 1.0 | 1.1M | Large | – | – |
| HDD-Bench (ours) | 3.0 | 11.8M | Small | ✓ | ✓ |

## 3 HDD-Bench: High-Definition, Dialogue-based image editing benchmark

Recent generative editors are increasingly interactive and agentic, yet rigorous evaluation for *professional-grade* editing remains underdeveloped. Existing benchmarks typically fall short along at least one key dimension: (i) *single-turn interactions* that fail to capture the stepwise dependencies inherent to real editing sessions; (ii) *low-resolution formats* that cannot meet the fidelity and locality requirements of native 4K workflows; and (iii) *human-in-the-loop processes* that, while enabling richer interactions, act as a significant bottleneck restricting dataset scale and diversity. More importantly, most benchmarks provide only an end result, without a *verifiable intermediate interface*. Without turn-by-turn targets, it is hard to diagnose long-horizon failures such as error accumulation, over-editing of non-target regions, or semantic drift across turns. This motivates a benchmark that (i) supports *multi-turn*, logically dependent instruction chains; (ii) evaluates at *native high resolution* to ensure fidelity; and (iii) provides *structured intermediate supervision*, enabling precise failure attribution without the scalability constraints of human oversight.

### 3.1 A Scalable Data Pipeline for Multi-turn Editing

![Refer to caption](https://arxiv.org/html/2602.09084v2/x2.png)

Figure 3: Scalable Data Pipeline for Multi-turn Editing. This diagram illustrates the process of generating aligned (State, Instruction) pairs from HD images.

To enable verifiable multi-turn supervision without expensive pixel-level annotation, we propose a scalable symbolic data engine that synthesizes editing trajectories in an attribute-level state space. For each input image, we construct an initial scene state $s_{0}$ that represents salient objects and their attributes, including name, color, size, material, and shape. Each editing turn is specified by a set of *canonical* edit commands $\mathbf{c}_{t}$, which are applied deterministically to update the state:

$$
s_{t+1}=\mathcal{T}(s_{t},\mathbf{c}_{t}),
$$

where $\mathcal{T}$ is a deterministic transition operator that modifies only the targeted object attributes. This design decouples *interaction synthesis* from *image generation*: we can generate consistent and checkable intermediate targets $\{s_{1},s_{2},\dots\}$ without rendering images during data construction.

To mimic real user behavior, a language agent paraphrases the canonical command set $\mathbf{c}_{t}$ into a single natural-language instruction $q_{t}$, optionally mixing multiple intents (e.g., adding an object while changing another object’s color). Importantly, any ambiguity is introduced only in the surface phrasing $q_{t}$, while the underlying $\mathbf{c}_{t}$ and target state $s_{t+1}$ are preserved as the internal ground truth. To ensure reliability, we incorporate human verification at the entry point of the pipeline: the initial scene graph and extracted attributes used to form $s_{0}$ are manually inspected and corrected. Since subsequent turns are produced by deterministic transitions, this guarantees the correctness of the entire multi-turn state chain and provides a principled, verifiable interface for evaluation.

### 3.2 Constructing the HDD-Bench

Built on top of the data engine, we construct HDD-Bench, a High-Definition, Dialogue-based benchmark that targets professional editing requirements. HDD-Bench is designed to jointly stress (i) *multi-turn dependency*, where later instructions build on earlier edits; (ii) *high-resolution fidelity*, where fine textures and sharp boundaries must be preserved at native resolution; and (iii) *object-level compositionality*, where instructions may involve multiple objects and mixed intents.

Each sample in HDD-Bench is a three-turn editing session. At turn $t$, the benchmark provides a natural-language instruction $q_{t}$ (often combining multiple edit intents into a single request) and a corresponding target symbolic state $s_{t}$ for verifiable evaluation. We adopt three-turn interactions to control difficulty and simplify comparisons across methods, while still capturing stepwise dependency and error accumulation; notably, our engine can generate longer sessions without changing the evaluation interface.

HDD-Bench contains 96 curated sessions selected from the synthesis pipeline. The selected samples emphasize scenes with multiple salient objects and non-trivial edit chains, and cover a diverse set of atomic actions (e.g., add, remove, replace, adjust, undo) as well as hybrid instructions that require composing multiple actions within a turn.

### 3.3 Evaluation Protocol

HDD-Bench evaluates editing quality from two complementary perspectives: (i) semantic correctness of the intended edits, and (ii) visual preservation of non-target regions. The first aspect is assessed in a verifiable, object-centric manner using the symbolic state representation; the second is assessed at the pixel/perceptual level to quantify background fidelity.

#### State-based metrics: Instruction Following and Image Consistency.

Given a generated image at turn $t$, we map it to a predicted post-edit state $\hat{s}_{t}$ using the same perception pipeline used to construct $s_{0}$. We then compare $\hat{s}_{t}$ against the ground-truth target state $s_{t}$ to compute two scores: Instruction Following (IF) measures whether the attributes of *targeted* objects match the requested edits, while Image Consistency (IC) measures whether *non-target* objects remain unchanged across turns. Both scores are computed by averaging attribute-level correctness over objects:

$$
s_{\text{IF}}\ \text{or}\ s_{\text{IC}}=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1}{M_{i}}\sum_{j=1}^{M_{i}}s_{i,j}\right),
$$

where $N$ is the number of evaluated objects (edited or preserved), $M_{i}$ is the number of attributes for object $i$, and $s_{i,j}$ is the correctness score for the $j$ -th attribute.

#### Otsu-masked background fidelity.

Global full-reference metrics such as PSNR, SSIM, and LPIPS can be misleading for editing, since they penalize valid foreground changes and unwanted background corruption equally. To isolate preservation quality, we compute Otsu-Masked PSNR/SSIM/LPIPS [^35] [^43] [^55], denoted as $\mathrm{PSNR}_{\mathrm{OM}}$, $\mathrm{SSIM}_{\mathrm{OM}}$, and $\mathrm{LPIPS}_{\mathrm{OM}}$. Concretely, we form a pixel-wise difference map between the pre-edit and post-edit images, apply Otsu’s method to obtain an adaptive threshold $k^{*}$ by maximizing inter-class variance,

$$
k^{*}=\arg\max_{1\leq k<L}\sigma_{B}^{2}(k),
$$

and construct a background mask $M_{\text{bg}}$ by selecting pixels whose differences fall below $k^{*}$. We then compute metrics only on the masked background region. This provides a targeted measure of whether the model preserves non-edited context while performing the intended semantic edits.

![Refer to caption](https://arxiv.org/html/2602.09084v2/x3.png)

Figure 4: Qualitative Comparison of Editing Fidelity. We utilize the instruction "…And change that little bright blue cooler under the shelter to a softer sea‑foam green with a creamy top …" to guide the editing process. While the prompt solely targets color modification, baseline models exhibit significant limitations: they often suffer from reduced resolution, introduce unwanted structural changes (modifying shape or position), or fail to apply the target color change. By leveraging our agent’s superior interpretation capabilities, our method accurately captures the instruction’s focus while preserving the integrity of the original image.

Table 2: Quantitative Comparison of Image Editing Performance. We evaluate models on HDD-Bench focusing on image fidelity (PSNR ${}_{\text{OM}}$, SSIM ${}_{\text{OM}}$, LPIPS ${}_{\text{OM}}$), instruction adherence (Instruct-Following, Image Consistency), and support for high-resolution (4K) editing. Agent Banana achieves state-of-the-art performance, balancing precise instruction execution with high visual fidelity, and is natively capable of processing at 4K resolution.

<table><tbody><tr><th>Model</th><td colspan="5">HDD Bench</td><td colspan="4">ImgEdit</td><td>4K</td></tr><tr><th></th><td>PSNR <math><semantics><msub><mtext>OM</mtext></msub> <annotation>{}_{\text{OM}}</annotation></semantics></math> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>SSIM <math><semantics><msub><mtext>OM</mtext></msub> <annotation>{}_{\text{OM}}</annotation></semantics></math> <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>LPIPS <math><semantics><msub><mtext>OM</mtext></msub> <annotation>{}_{\text{OM}}</annotation></semantics></math> <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>IF <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>IC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Add <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Adj. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Repl. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Rem. <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Res</td></tr><tr><th>ICEdit <sup><a href="#fn:57">57</a></sup></th><td>29.21</td><td>0.80</td><td>0.14</td><td>0.595</td><td>0.687</td><td>3.58</td><td>3.39</td><td>3.15</td><td>2.93</td><td>–</td></tr><tr><th>Qwen-Image <sup><a href="#fn:44">44</a></sup></th><td>23.62</td><td>0.80</td><td>0.14</td><td>0.845</td><td>0.807</td><td>4.38</td><td>4.16</td><td>4.66</td><td>4.14</td><td>–</td></tr><tr><th>OmniGen2 <sup><a href="#fn:45">45</a></sup></th><td>23.59</td><td>0.72</td><td>0.23</td><td>0.545</td><td>0.655</td><td>3.57</td><td>3.06</td><td>3.74</td><td>3.20</td><td>–</td></tr><tr><th>BAGEL <sup><a href="#fn:11">11</a></sup></th><td>26.93</td><td>0.79</td><td>0.17</td><td>0.676</td><td>0.723</td><td>3.56</td><td>3.31</td><td>3.3</td><td>2.62</td><td>–</td></tr><tr><th>Step1X-Edit <sup><a href="#fn:27">27</a></sup></th><td>25.82</td><td>0.77</td><td>0.19</td><td>0.808</td><td>0.797</td><td>3.88</td><td>3.14</td><td>3.40</td><td>2.41</td><td>✓</td></tr><tr><th>FLUX.1 Kontext [Pro] <sup><a href="#fn:20">20</a></sup></th><td>25.98</td><td>0.74</td><td>0.17</td><td>0.845</td><td>0.702</td><td>4.25</td><td>4.15</td><td>4.56</td><td>3.57</td><td>–</td></tr><tr><th>GPT Image 1 [High] <sup><a href="#fn:32">32</a></sup></th><td>19.20</td><td>0.54</td><td>0.33</td><td>0.882</td><td>0.727</td><td>4.61</td><td>4.33</td><td>4.35</td><td>3.66</td><td>–</td></tr><tr><th>Nano Banana Pro <sup><a href="#fn:9">9</a></sup></th><td>26.62</td><td>0.72</td><td>0.14</td><td>0.911</td><td>0.861</td><td>4.58</td><td>4.56</td><td>4.55</td><td>4.39</td><td>✓</td></tr><tr><th>Agent Banana (ours)</th><td>28.40</td><td>0.84</td><td>0.12</td><td>0.849</td><td>0.871</td><td>4.58</td><td>4.59</td><td>4.62</td><td>4.60</td><td>✓</td></tr></tbody></table>

## 4 Experiments

### 4.1 Experimental Setup

In our experiments, we employ GPT-5-mini as the foundational Large Language Model (LLM) powering both the Planner and Executor agents. To endow the agents with robust visual generation and editing capabilities, we construct a comprehensive toolset integrating state-of-the-art visual models, including both open-source and private models for high-quality generation and editing, complemented by GPT-5-mini for visual verification. To ensure a fair comparison, when Nano Banana Pro is used as the underlying image model, our gains reflect agentic scaffolding (decomposition, masking, verification) rather than changes to the generator weights; instead, we compare against the Nano Banana Pro and other baseline models operating without our multi-step workflow.

### 4.2 Performance on Multi-turn Editing

To comprehensively evaluate the performance of Agent Banana, we benchmark it against representative image editing models, including the closed-source commercial model Flux.1 Kontext [^20], Nano Banana Pro [^9], and GPT-Image-1 \[High\] [^32]. We adopt the standard metrics defined by HDD-Bench, covering editing accuracy (${s}_{\text{instruction following}}$), Otsu-Masked PSNR (${s}_{\text{$\text{PSNR}_{\text{OM}}$}}$), and the final composite score ($s_{\text{final}}$). Detailed quantitative comparisons are presented in Table 2.

Given that this benchmark focuses on multi-turn sequential editing tasks, we report the average score across all interaction turns as the final performance metric. Notably, our dataset consists entirely of 4K-resolution images, posing a significant challenge to the high-resolution processing capabilities of the models. For baselines that downsample inputs during processing, we explicitly denote their maximum supported resolution in the results table and evaluate them after upsampling the output back to 4K.

The results indicate that Agent Banana not only achieves competitive scores against the baselines but, crucially, is one of only two models capable of maintaining high fidelity at 4K native resolution. This validates the effectiveness of our proposed Image Layer Decomposition mechanism in preventing detail loss during high-resolution editing.

### 4.3 Performance on Single-turn Editing

In addition to evaluating long-horizon multi-turn capabilities, we assess the foundational performance of Agent Banana on single-turn editing tasks using the ImgEdit-Bench. This experiment aims to verify that our agent architecture, despite being designed for complex planning, maintains SOTA precision when handling atomic editing instructions.

We compare Agent Banana with mainstream single-turn editing models. As shown in Table 2, our method achieves leading or comparable results across all metrics. This is primarily attributed to the Executor’s precise control over tool parameters and the self-verification mechanism provided by the Quality Test modules.

### 4.4 Ablation Study

Impact of Foundational LLM Capabilities. We first investigate the sensitivity of system performance to the capabilities of the base model. Given that the base model directly dictates instruction understanding, task planning, and the accuracy of tool invocation, we experimented by replacing the kernels of the Planner and Executor with the smaller-scale Qwen-3-8B. Observations reveal that a weaker base model exhibits significant performance degradation when handling ambiguous instructions and long-sequence planning, frequently generating unparseable tool parameters or erroneous dependencies, leading to workflow interruptions. This confirms that robust reasoning capability is a prerequisite for agents handling complex multi-turn editing tasks.

![Refer to caption](https://arxiv.org/html/2602.09084v2/x4.png)

Figure 5: Qualitative Comparison of Unedited Region Consistency. Although the editing instruction does not target the sofa cushion, Nano Banana Pro distorts the original details due to global editing. In contrast, our method successfully maintains the visual consistency of the unedited regions.

![Refer to caption](https://arxiv.org/html/2602.09084v2/x5.png)

Figure 6: Metric Comparison across Sequential Turns. Agent Banana (red line) exhibits relatively better performance and consistent stability across all evaluated metrics. Compared to several other models, demonstrating its effectiveness in preserving image quality throughout the multi-turn process.

### 4.5 Native-Resolution Editing Analysis

A significant advantage of Agent Banana is its capability for native-resolution editing. Unlike existing baselines (e.g., FLUX.1 Kontext or Qwen-Image) that typically force inputs to be downsampled to 1k resolution, our method avoids this loss through a layered processing mechanism. As illustrated in Figure 5, for a high-resolution input of $2716\times 4060$, baseline models lose substantial texture detail during the downsampling-upsampling process, whereas our method perfectly preserves the high-frequency information of the original image. However, baseline models exhibit significant limitations: they often suffer from reduced resolution, introduce unwanted structural changes (e.g., modifying object shape or position), or fail to apply the target color change. By leveraging our agent’s superior interpretation capabilities, our method accurately captures the instruction’s focus while preserving the integrity of the original image. This minimal-loss characteristic positions Agent Banana as a viable solution for professional-grade image editing tasks.

### 4.6 On the Prior-Induced Editing Drift (PIED)

We observe a subtle but important failure mode in multi-turn editing using generative editors: even when each turn appears highly realistic—sometimes indistinguishable to the eye—the purported “non-edited” regions (which are, in practice, repeatedly re-generated) can gradually drift toward the generator’s preferred texture and style statistics as turns accumulate. We term this effect Prior-Induced Editing Drift (PIED). Figure 6 shows that several baselines exhibit a steady *increase* in $\text{PSNR}_{\text{OM}}$ on non-edited regions across turns, which can be misleading. We hypothesize that PIED “games” this metric: repeated re-synthesis slightly re-renders the whole image, shrinking Otsu-partitioned background changes (thus inflating $\text{PSNR}_{\text{OM}}$) while faithfulness to the original input still degrades. In contrast, Agent Banana keeps $\text{PSNR}_{\text{OM}}$ nearly constant across turns, matching qualitative observations of reduced accumulated artifacts and better preservation of high-frequency details and style in non-edited regions. Overall, PIED suggests that per-turn visual fidelity can decouple from long-horizon faithfulness, and drift accumulation should be explicitly measured in evaluating multi-turn editors.

## 5 Related Work

### 5.1 Instruction-based Image Editing

Recent progress in instruction-based image editing has been driven by diffusion and autoregressive foundation models, such as GLIDE [^31], InstructPix2Pix [^3], MagicBrush [^53], Prompt-to-Prompt [^13], and UltraEdit [^58]. Beyond these single-turn editors, emerging interactive systems (e.g., GPT-Image-1 [^33] and Nano Banana [^8]) indicate a shift toward multi-turn, context-aware interaction. To further strengthen fine-grained control, follow-up work explores attention manipulation [^13], mask-based inpainting [^2], and automatic region detection [^6]; additionally, several methods decompose scenes into object-specific layers for more precise localized editing [^30] [^48] [^41].

### 5.2 Agentic Systems for Image Editing

The exceptional reasoning and language capabilities of large language models (LLMs) have catalyzed rapid advances in agentic systems for interaction and task solving in complex environments. Paradigms exemplified by ReAct [^49] establish a foundational framework by alternating reasoning and atomic actions within an iterative think–act loop. Meanwhile, Anthropic’s Model Context Protocol (MCP) [^29] unifies the communication interface between LLMs and external tools, substantially improving the standardization and scalability of tool orchestration. Agentic perception–decision–action paradigms have long been explored in vision and learning via closed-loop or adaptive frameworks [^37] [^7] [^39] [^61] [^24] [^5], with VLMs increasingly serving as planners. For image/video restoration, AgenticIR and MoA-VR independently introduce VLM-integrated multi-agent repair paradigms [^60] [^25]. In creative photo retouching and task-oriented restoration, intelligent tool-invocation workflows such as JarvisIR, JarvisArt, 4KAgent, and JarvisEvo further demonstrate the effectiveness of agentic pipelines for restoration and editing [^21] [^22] [^62] [^23].

## 6 Conclusion

We introduce Agent Banana, a multi-agent, layer-aware framework for instruction-based image editing, together with HDD-Bench, a high-resolution multi-turn benchmark aligned with professional workflows. By coupling LLM planning, VLM perception, and layer-aware tool use, Agent Banana performs precise, rollback-safe edits on 4K images while preserving non-target regions, and consistently improves instruction following, edit locality, and multi-turn stability over strong non-agentic baselines. Beyond a single system and benchmark, we model editing as explicit state transitions on object-level graphs, enabling stepwise, verifiable evaluation and natural support for undo, branching, and long-horizon correction. Our scalable data engine further decouples state transitions from pixel rendering, making it practical to synthesize large-scale vision–language reasoning traces and edit histories.

#### Impact Statement.

This work advances instruction-based image editing toward professional workflows by emphasizing two properties that matter in real deployments: native high-resolution fidelity and multi-turn reliability. In particular, our benchmark and evaluation protocol provide stepwise, verifiable checks of what changed and what must remain invariant across turns, helping the community move beyond single-turn demos and toward diagnosing long-horizon failure modes such as over-editing, drift, and irreversible degradation. At the same time, stronger editing capabilities can be misused to create misleading visual content or facilitate non-consensual manipulation. We therefore emphasize evaluation and auditing: our contributions are designed to measure controllability and detect failure accumulation rather than to optimize for unconstrained manipulation, and we encourage future systems built on this line of work to adopt provenance, consent, and disclosure mechanisms when applied to real-world media.

[^1]: Anthropic (2025) Claude 3.7 sonnet system card. Note: [https://www.anthropic.com/news/claude-3-7-sonnet-system-card](https://www.anthropic.com/news/claude-3-7-sonnet-system-card) Accessed: 2025-10-29 Cited by: §1.

[^2]: O. Avrahami, D. Lischinski, and O. Fried (2022) Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18208–18218. Cited by: §5.1.

[^3]: T. Brooks, A. Holynski, and A. A. Efros (2023) Instructpix2pix: learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18392–18402. Cited by: §1, §5.1.

[^4]: Q. Cai, J. Chen, Y. Chen, Y. Li, F. Long, Y. Pan, Z. Qiu, Y. Zhang, F. Gao, P. Xu, et al. (2025) HiDream-i1: a high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705. Cited by: §1, §1.

[^5]: H. Chen, W. Li, J. Gu, J. Ren, S. Chen, T. Ye, R. Pei, K. Zhou, F. Song, and L. Zhu (2024) RestoreAgent: autonomous image restoration agent via multimodal large language models. External Links: 2407.18035 Cited by: §5.2.

[^6]: G. Couairon, J. Verbeek, H. Schwenk, and M. Cord (2022) Diffedit: diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427. Cited by: §5.1.

[^7]: E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le (2019) Autoaugment: learning augmentation strategies from data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 113–123. Cited by: §5.2.

[^8]: G. DeepMind (2025) Gemini 2.5 flash image (n̈ano banana)̈. Note: [https://aistudio.google.com/models/gemini-2-5-flash-image](https://aistudio.google.com/models/gemini-2-5-flash-image) Accessed: 2025-10-29 Cited by: §1, §5.1.

[^9]: G. DeepMind (2025) Gemini 3 image preview (n̈ano banana pro)̈. Note: [https://deepmind.google/models/gemini-image/pro/](https://deepmind.google/models/gemini-image/pro/) Accessed: 2026-1-28 Cited by: Table 2, §4.2.

[^10]: C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. (2025) Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683. Cited by: §1, §1.

[^11]: C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, G. Shi, and H. Fan (2025) Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683. Cited by: Table 2.

[^12]: Y. Ge, S. Zhao, C. Li, Y. Ge, and Y. Shan (2024) Seed-data-edit technical report: a hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007. Cited by: Table 1.

[^13]: A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or (2022) Prompt-to-prompt image editing with cross attention control.(2022). URL https://arxiv. org/abs/2208.01626 3. Cited by: §5.1.

[^14]: J. Ho, A. Jain, and P. Abbeel (2020) Denoising diffusion probabilistic models. Advances in neural information processing systems 33, pp. 6840–6851. Cited by: §1.

[^15]: W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Dong, M. Ding, et al. (2024) Cogagent: a visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14281–14290. Cited by: §1.

[^16]: W. Hu, Z. Lin, M. Clark, J. Henwood, X. Shang, R. Chen, K. Kiburg, L. Zhang, Z. Ge, P. van Wijngaarden, et al. (2025) Real-world feasibility, accuracy and acceptability of automated retinal photography and ai-based cardiovascular disease risk assessment in australian primary care settings: a pragmatic trial. NPJ Digital Medicine 8 (1), pp. 122. Cited by: §1.

[^17]: M. Hui, S. Yang, B. Zhao, Y. Shi, H. Wang, P. Wang, Y. Zhou, and C. Xie (2024) HQ-edit: a high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990. Cited by: §1.

[^18]: B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, S. Kulal, K. Lacey, Y. Levi, C. Li, D. Lorenz, J. Müller, D. Podell, R. Rombach, H. Saini, A. Sauer, and L. Smith (2025) FLUX.1 kontext: flow matching for in-context image generation and editing in latent space. External Links: 2506.15742, [Link](https://arxiv.org/abs/2506.15742) Cited by: §1.

[^19]: B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, et al. (2025) FLUX. 1 kontext: flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742. Cited by: §1, §1.

[^20]: B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, et al. (2025) FLUX. 1 kontext: flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742. Cited by: Table 2, §4.2.

[^21]: Y. Lin, Z. Lin, H. Chen, P. Pan, C. Li, S. Chen, K. Wen, Y. Jin, W. Li, and X. Ding (2025) Jarvisir: elevating autonomous driving perception with intelligent image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22369–22380. Cited by: §5.2.

[^22]: Y. Lin, Z. Lin, K. Lin, J. Bai, P. Pan, C. Li, H. Chen, Z. Wang, X. Ding, W. Li, et al. (2025) JarvisArt: liberating human artistic creativity via an intelligent photo retouching agent. arXiv preprint arXiv:2506.17612. Cited by: §5.2.

[^23]: Y. Lin, L. Wang, K. Lin, Z. Lin, K. Gong, W. Li, B. Lin, Z. Li, S. Zhang, Y. Peng, et al. (2025) JarvisEvo: towards a self-evolving photo editing agent with synergistic editor-evaluator optimization. arXiv preprint arXiv:2511.23002. Cited by: §5.2.

[^24]: H. Liu, X. Li, P. Li, M. Liu, D. Wang, J. Liu, B. Kang, X. Ma, T. Kong, and H. Zhang (2025) Towards generalist robot policies: what matters in building vision-language-action models. Cited by: §5.2.

[^25]: L. Liu, C. Cai, S. Shen, J. Liang, W. Ouyang, T. Ye, J. Mao, H. Duan, J. Yao, X. Zhang, Q. Hu, and G. Zhai (2025) MoA-vr: a mixture-of-agents system towards all-in-one video restoration. External Links: 2510.08508, [Link](https://arxiv.org/abs/2510.08508) Cited by: §5.2.

[^26]: N. Liu, S. Li, Y. Du, A. Torralba, and J. B. Tenenbaum (2022) Compositional visual generation with composable diffusion models. In European conference on computer vision, pp. 423–439. Cited by: §1.

[^27]: S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. (2025) Step1x-edit: a practical framework for general image editing. arXiv preprint arXiv:2504.17761. Cited by: §1, §1, §1, Table 1, Table 2.

[^28]: V. Mirzaei (2025) The impact of artificial intelligence on creativity in graphic design. Available at SSRN 5292032. Cited by: §1.

[^29]: Model Context Protocol (2025-11) Model context protocol (mcp) specification (version 2025-11-25). Note: [https://modelcontextprotocol.io/specification/2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) Accessed: 2026-01-29 Cited by: §5.2.

[^30]: T. Monnier, E. Vincent, J. Ponce, and M. Aubry (2021) Unsupervised layered image decomposition into object prototypes. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 8640–8650. Cited by: §5.1.

[^31]: A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen (2021) Glide: towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741. Cited by: §5.1.

[^32]: OpenAI (2025) GPT-image-1. External Links: [Link](https://openai.com/index/introducing-4o-image-generation/) Cited by: Table 2, §4.2.

[^33]: OpenAI (2025) Introducing 4o image generation. Note: [https://openai.com/index/introducing-4o-image-generation/](https://openai.com/index/introducing-4o-image-generation/) Accessed: 2025-10-29 Cited by: §1, §5.1.

[^34]: OpenAI (2025) Operator. Note: [https://openai.com/index/introducing-operator/](https://openai.com/index/introducing-operator/) Accessed: 2025-10-29 Cited by: §1.

[^35]: N. Otsu et al. (1975) A threshold selection method from gray-level histograms. Automatica 11 (285-296), pp. 23–27. Cited by: §3.3.

[^36]: Y. Qin, Y. Ye, J. Fang, H. Wang, S. Liang, S. Tian, J. Zhang, J. Li, Y. Li, S. Huang, et al. (2025) Ui-tars: pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326. Cited by: §1.

[^37]: C. Rother, V. Kolmogorov, and A. Blake (2004) " GrabCut" interactive foreground extraction using iterated graph cuts. ACM transactions on graphics (TOG) 23 (3), pp. 309–314. Cited by: §5.2.

[^38]: N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao (2023) Reflexion: language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems 36, pp. 8634–8652. Cited by: §1.

[^39]: D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell (2021) Tent: fully test-time adaptation by entropy minimization. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=uXl3bZLkr3c) Cited by: §5.2.

[^40]: P. Wang, Y. Shi, X. Lian, Z. Zhai, X. Xia, X. Xiao, W. Huang, and J. Yang (2025) SeedEdit 3.0: fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083. Cited by: §1.

[^41]: Z. Wang, J. Huang, Z. Sun, Y. Gong, D. Cohen-Or, and M. Lu (2025) Layered image vectorization via semantic simplification. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7728–7738. Cited by: §5.1.

[^42]: Z. Wang, A. Li, Z. Li, and X. Liu (2024) Genartist: multimodal llm as an agent for unified image generation and editing. Advances in Neural Information Processing Systems 37, pp. 128374–128395. Cited by: §1.

[^43]: Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli (2004) Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13 (4), pp. 600–612. Cited by: §3.3.

[^44]: C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. Yin, S. Bai, X. Xu, Y. Chen, et al. (2025) Qwen-image technical report. arXiv preprint arXiv:2508.02324. Cited by: §1, §1, Table 2.

[^45]: C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou, et al. (2025) OmniGen2: exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871. Cited by: §1, §1, Table 2.

[^46]: Z. Wu, Z. Wu, F. Xu, Y. Wang, Q. Sun, C. Jia, K. Cheng, Z. Ding, L. Chen, P. P. Liang, et al. (2024) Os-atlas: a foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218. Cited by: §1.

[^47]: Y. Xu, Z. Wang, J. Wang, D. Lu, T. Xie, A. Saha, D. Sahoo, T. Yu, and C. Xiong (2024) Aguvis: unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454. Cited by: §1.

[^48]: J. Yang, Q. Liu, Y. Li, S. Y. Kim, D. Pakhomov, M. Ren, J. Zhang, Z. Lin, C. Xie, and Y. Zhou (2025) Generative image layer decomposition with visual effects. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7643–7653. Cited by: §5.1.

[^49]: S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023) ReAct: synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), Cited by: §5.2.

[^50]: S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao (2022) React: synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, Cited by: §1, §2.1.

[^51]: Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan (2025) Imgedit: a unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275. Cited by: §1, Table 1.

[^52]: Q. Yu, J. Li, Y. Wu, S. Tang, W. Ji, and Y. Zhuang (2025) AnyEdit: mastering unified high-quality image editing for any idea. In CVPR, Cited by: Table 1.

[^53]: K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su (2023) Magicbrush: a manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36, pp. 31428–31449. Cited by: §5.1.

[^54]: P. Zhang, X. Dong, B. Wang, Y. Cao, C. Xu, L. Ouyang, Z. Zhao, H. Duan, S. Zhang, S. Ding, et al. (2023) Internlm-xcomposer: a vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112. Cited by: §1.

[^55]: R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang (2018) The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, Cited by: §3.3.

[^56]: R. Zhang, B. Yu, J. Min, Y. Xin, Z. Wei, J. N. Shi, M. Huang, X. Kong, N. L. Xin, S. Jiang, et al. (2025) Generative ai for film creation: a survey of recent advances. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6267–6279. Cited by: §1.

[^57]: Z. Zhang, J. Xie, Y. Lu, Z. Yang, and Y. Yang (2025) In-context edit: enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690. Cited by: Table 2.

[^58]: H. Zhao, X. S. Ma, L. Chen, S. Si, R. Wu, K. An, P. Yu, M. Zhang, Q. Li, and B. Chang (2024) Ultraedit: instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems 37, pp. 3058–3093. Cited by: §5.1.

[^59]: Z. Zhou, Y. Deng, X. He, W. Dong, and F. Tang (2025) Multi-turn consistent image editing. arXiv preprint arXiv:2505.04320. Cited by: §1.

[^60]: K. Zhu, J. Gu, Z. You, Y. Qiao, and C. Dong (2025) An intelligent agentic system for complex image restoration problems. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=3RLxccFPHz) Cited by: §5.2.

[^61]: X. Zhu, Y. Chen, H. Tian, C. Tao, W. Su, C. Yang, G. Huang, B. Li, L. Lu, X. Wang, et al. (2023) Ghost in the minecraft: generally capable agents for open-world environments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144. Cited by: §5.2.

[^62]: Y. Zuo, Q. Zheng, M. Wu, X. Jiang, R. Li, J. Wang, Y. Zhang, G. Mai, L. V. Wang, J. Zou, et al. (2025) 4kagent: agentic any image to 4k super-resolution. arXiv preprint arXiv:2507.07105. Cited by: §5.2.