---
title: "Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation"
source: "https://arxiv.org/html/2602.01756v1"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/04_%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E4%B8%8E%E7%BC%96%E8%BE%91/Mind-Brush%2C%20Integrating%20Agentic%20Cognitive%20Search%20and%20Reasoning%20into%20Image%20Generation%2C%20Jun%20He%20et%20al.%2C%202026.no_watermark.zh-CN.dual.pdf"
---
Jun He    Junyan Ye    Zilong Huang    Dongzhi Jiang    Chenjue Zhang    Leqi Zhu    Renrui Zhang    Xiang Zhang    Weijia Li

###### Abstract

While text-to-image generation has achieved unprecedented fidelity, the vast majority of existing models function fundamentally as static text-to-pixel decoders. Consequently, they often fail to grasp implicit user intentions. Although emerging unified understanding-generation models have improved intent comprehension, they still struggle to accomplish tasks involving complex knowledge reasoning within a single model. Moreover, constrained by static internal priors, these models remain unable to adapt to the evolving dynamics of the real world. To bridge these gaps, we introduce Mind-Brush, a unified agentic framework that transforms generation into a dynamic, knowledge-driven workflow. Simulating a human-like ’think-research-create’ paradigm, Mind-Brush actively retrieves multimodal evidence to ground out-of-distribution concepts and employs reasoning tools to resolve implicit visual constraints. To rigorously evaluate these capabilities, we propose Mind-Bench, a comprehensive benchmark comprising 500 distinct samples spanning real-time news, emerging concepts, and domains such as mathematical and Geo-Reasoning. Extensive experiments demonstrate that Mind-Brush significantly enhances the capabilities of unified models, realizing a zero-to-one capability leap for the Qwen-Image baseline on Mind-Bench, while achieving superior results on established benchmarks like WISE and RISE.

Machine Learning, ICML

| Github: | [https://github.com/PicoTrex/Mind-Brush](https://github.com/PicoTrex/Mind-Brush) |
| --- | --- |
| Dataset: | [https://huggingface.co/datasets/PicoTrex/Mind-Brush](https://huggingface.co/datasets/PicoTrex/Mind-Brush) |

![[Uncaptioned image]](https://arxiv.org/html/2602.01756v1/images/flag.jpg)

Figure 1: We introduce Mind-Brush, an agentic framework that synergizes active search with explicit reasoning for image generation. By decomposing user intent, retrieving multimodal evidence, and inferring latent requirements, our agent effectively bridges the cognitive gaps and interpretative biases prevalent in existing models. Furthermore, we propose Mind-Bench, a comprehensive benchmark designed to evaluate model performance on up-to-date long-tail concepts and multimodal reasoning tasks, thereby probing the boundaries of unified understanding and generation capabilities.

## 1 Introduction

Recent advancements in image generation have democratized visual creation, enabling the seamless translation of imagination into high-fidelity imagery [^31] [^12] [^14]. However, fundamentally, the vast majority of existing models function primarily as static text-to-pixel decoders [^30] [^4]. Confined to mapping explicit user instructions to pixels, these models often fail to grasp implicit, high-level user intentions, thereby diverging significantly from the human artistic creation process. While emerging unified multimodal understanding-generation models, such as GPT-Image [^28] and Bagel [^11], demonstrate promising capabilities in comprehending user intent and incorporating world knowledge, their performance remains constrained in tasks demanding complex mathematical or knowledge-intensive reasoning. This limitation suggests that monolithic architectures may struggle to encompass the full spectrum of capabilities required for such intricate, end-to-end tasks.

Moreover, constrained by the temporal knowledge cutoff inherent in pre-training data, the cognitive boundaries of current image generation models remain static. Consequently, they struggle to adapt to the evolving dynamics of the real world, resulting in significant capability gaps when handling real-time news or novel IP concepts [^32] [^23]. In the realm of Large Language Models (LLMs), researchers have successfully transcended these boundaries by integrating retrieval capabilities through agentic designs [^9] [^48], as exemplified by Search-o1 [^25]. While recent proprietary models such as Nano Banana Pro [^10] and FLUX-2 Max [^3] have demonstrated integrated search and reasoning capabilities, a significant gap persists within the open-source community. Specifically, there is a notable absence of open-source models capable of interacting with the open world, performing complex reasoning, and executing active planning.

Recently, agentic approaches for image generation have emerged [^19], such as T2I-Copilot [^6] and PromptSculptor [^43], which focus on elaborating concise instructions into detail-rich descriptions. Think-Then-Gen [^20] advances this further by leveraging LLMs to decompose user queries into sequential drawing steps. Nevertheless, these efforts remain largely confined to standard T2I benchmarks like GenEval++ [^46], prioritizing prompt refinement over intricate cognitive tasks such as mathematical derivation or commonsense reasoning. Crucially, due to the absence of external tools, their reasoning relies solely on internalized knowledge. Consequently, these methods falter when tasks demand factual verification of real-time events or evolving contexts.

To bridge this gap, we introduce Mind-Brush, a unified agentic framework that shifts image generation from a static mapping paradigm to a dynamic, knowledge-driven workflow. Rather than treating generation as single-step inference, Mind-Brush orchestrates a cognitive process: it actively retrieves multimodal evidence to ground out-of-distribution concepts and employs logical reasoning to deduce implicit visual constraints, thereby realizing the unification of agentic understanding and generation. By effectively simulating the human artist’s “Think-Research-Create” workflow, Mind-Brush enables high-fidelity generation requiring real-time knowledge and handles tasks involving complex reasoning.

Furthermore, existing image generation benchmarks, such as GenEval [^13] and ImgEdit [^47], primarily prioritize the evaluation of instruction following. While datasets like WISE [^27] and RISE [^51] extend this scope to probe internal knowledge recall and rudimentary reasoning, they fall short of evaluating capabilities that require active information retrieval and complex reasoning. To address this limitation, we introduce Mind-Bench, a comprehensive benchmark specifically designed to assess generative performance under conditions necessitating complex reasoning and external knowledge acquisition. As illustrated in Figure 1, Mind-Bench encompasses 500 distinct samples across 10 diverse categories, spanning challenging scenarios from real-time news and emerging IP concepts to complex mathematical and geographical reasoning. This benchmark fills a critical void in evaluating image synthesis tasks that demand real-time knowledge and deep reasoning, particularly for unified understanding-generation models.

Our main contributions are summarized as follows:

1. We propose Mind-Brush, a novel agentic framework that unifies intent analysis, multi-modal search, and knowledge reasoning to enable a ’think-research-create’ paradigm for image generation.
2. We propose Mind-Bench, a benchmark tailored to evaluate generative capabilities involving dynamic external knowledge and complex reasoning. Experimental results reveal critical limitations in current unified multimodal models regarding real-time awareness and logical deduction.
3. Mind-Brush substantially elevates the accuracy of the Qwen-Image baseline from 0.02 to 0.31 on Mind-Bench, while significantly outperforming existing baselines on established benchmarks including the knowledge-driven WISE (+25.8 % WiScore) and reasoning-driven RISEBench (+27.3 % Accuracy).

## 2 Related Work

### 2.1 Agent for Image Generation

Multimodal LLMs are increasingly serving as agentic decision-makers to align vague user intents with precise image synthesis [^45] [^50] [^35]. A primary stream focuses on prompt optimization: T2I-Copilot [^6] and PromptSculptor [^43] employ multi-agent collaboration to refine concise instructions into detailed descriptions, while ImAgent introduces test-time policy scaling for semantic alignment. Parallel works target precise control: MCCD [^24] utilizes agents to decouple multi-object attributes, while AgentStory [^53] and CREA [^39] ensure narrative consistency and creative editing, respectively. To transcend static knowledge boundaries, Think-Then-Generate introduces a ”think-then-generate” paradigm to explicitize internal visual logic. Concurrently, World-to-Image [^32] and IA-T2I [^23] attempt to employ simple image retrieval as visual cues to supplement out-of-distribution (OOD) concepts. However, current solutions remain fragmented. Reasoning-based methods are confined by closed training data, failing on real-time events; conversely, retrieval-based approaches often treat external evidence as shallow visual cues without deep logical integration. A unified workflow synergizing active multimodal search with explicit reasoning remains absent, limiting performance on complex, knowledge-intensive tasks.

### 2.2 Image Generation Model

The ultimate goal of Unified Multi-Modal Models (UMMs) is to consolidate cross-modal understanding and generation within a single architecture. Pioneering works (e.g., Chameleon [^37], Emu3 [^40]) integrated image generation into the LLM paradigm via visual signal discretization. However, their reliance on VQ-VAE [^38] introduced lossy compression, which fundamentally restricted generation fidelity. Subsequent research (e.g., Transfusion [^52], Show-o [^44]) attempted to mitigate this by unifying autoregressive text prediction and bidirectional image diffusion within a shared Transformer backbone. Yet, this approach triggered irreconcilable modal conflicts. Even with recent architectural innovations—such as Mixture-of-Tokens (MoT) or Mixture-of-Experts (MoE) employed in Bagel and OneCat—balancing the dual objectives of understanding and generation remains a formidable challenge. Consequently, state-of-the-art methods (e.g., OmniGen2 [^42], BLIP-o3 [^7]) have shifted toward a decoupled strategy, utilizing powerful multimodal large language models (MLLMs) to guide external diffusion heads, thereby achieving superior performance.

### 2.3 Image Generation Benchmarks

Evaluating Unified Multi-Modal Models (UMMs) necessitates a comprehensive assessment of both comprehension and synthesis. Mainstream benchmarks primarily assess text-image alignment and instruction following capabilities. For instance, GenEval [^13] evaluates compositional integrity, quantifying the model’s ability to bind attributes (e.g., counts, positions) explicitly stated in prompts. However, these benchmarks are largely confined to shallow, explicit text comprehension, neglecting deeper conceptual or reasoning capabilities. To address this, subsequent benchmarks incorporate extensive world knowledge. WISE [^27] and PhyBench [^26] assess domain-specific knowledge (e.g., culture, physics), while RISEBench [^51] focuses on logical reasoning, evaluating the translation of causal and spatio-temporal semantics into visual representations. Nevertheless, existing methods predominantly assess Internalized Parametric Memory and there is a notable absence of benchmarks evaluating multimodal reasoning or Out-of-Distribution (OOD) concepts, failing to distinguish whether a model is merely retrieving stored static knowledge or actively performing reasoning for real-time scenarios.

![Refer to caption](https://arxiv.org/html/2602.01756v1/images/pipeline_v1.jpg)

Figure 2: The overall framework of Mind-Brush. The user input first undergoes intent decomposition to identify potential knowledge deficits and formulate a question list. Based on specific requirements, the system dynamically executes specialized tools—such as utilizing active search or logical reasoning—to effectively bridge cognitive gaps. Finally, the consolidated evidence is organized into a final instruction via a concept review process to guide precise image generation, ensuring alignment with the user’s authentic intent.

## 3 Mind-Brush

### 3.1 Problem Formulation

We formalize the inference workflow of Mind-Brush as a Hierarchical Sequential Decision-Making Process, defined by the tuple $\mathcal{M}=\langle\mathcal{S},\mathcal{A},\pi,\mathcal{E}\rangle$. This framework generates a structured cognitive trajectory to bridge the gap between abstract intent and visual realization.

- Cognitive State ($\mathcal{S}$): Let $s_{t}=\{I,I_{img},\mathcal{E}_{t}\}$ denote the state at step $t$. It encapsulates the original user inputs (instruction $I$ and optional reference image $I_{img}$) and the dynamic evidence buffer $\mathcal{E}_{t}$, which accumulates retrieved knowledge and reasoning chains.
- Action Space ($\mathcal{A}$): The set of operators available to the agent. We distinguish between the Meta-Action $a_{plan}$ (Cognitive Gap Detection), which identifies cognitive gaps $Q_{gap}$, and Execution Actions $a_{exec}\in\{a_{search},a_{reason}\}$, which actively acquire multimodal evidence.
- Execution Policy ($\pi$): The Intent Analysis module functions as a high-level policy $\pi(a_{plan}|s_{0})$. It assesses the initial state to formulate a deterministic execution path based on the identified $Q_{gap}$.

The inference process evolves as a context-aware trajectory. As shown in Figure 2, the system does not follow a rigid workflow; instead, it dynamically adapts to the user’s request. By evaluating the specific nature of cognitive gaps in the initial state such as factual deficits or logical conflicts, the planner infers the optimal structure for evidence accumulation, routing the execution through specialized Search or Reasoning branches. This effectively aligns the inference computation with the intrinsic complexity of the user intent.

Ultimately, our objective is to generate the optimal target image $x^{*}$ based on the final converged state $s_{T}$. This state contains the consolidated Master Prompt $P_{master}$ and verified visual references $\mathcal{I}_{ref}$, transforming static generation into a dynamic explicit evidence accumulation process.

### 3.2 Cognitive Gap Detection

User instructions often contain implicit constraints and long-tail concepts exceeding the model’s parametric knowledge boundaries. To address this, we introduce the Cognitive Gap Detection strategy, integrated within the Intent Analysis Agent ($\mathcal{A}_{intent}$) as a meta-planner, to bridge this cognitive divide. Specifically, it maps the text instruction $I$ and optional image $I_{img}$ into a structured semantic space via the 5W1H (What, When, Where, Why, Who, and How) paradigm [^5], establishing a multimodal “Ground Truth” to determine signal dominance. Subsequently, the module executes a rigorous gap analysis by detecting specific entities or logical dependencies that require external verification. Information absent from internal knowledge is formalized into a set of explicit atomic questions, denoted as $Q_{gap}$. Based on the composition of $Q_{gap}$, the system instantiates a dynamic execution policy $\pi$, routing the workflow to the appropriate factual grounding or logical reasoning branch defined in the action space.

### 3.3 Adaptive Knowledge Completion

To bridge the identified cognitive gaps, Mind-Brush employs a Internal Logical Derivation mechanism. Unlike rigid single-pathway systems, the execution policy $\pi$ flexibly composes the retrieval and reasoning tools based on the complexity of $Q_{gap}$.

External Knowledge Anchoring. For gaps involving OOD entities or dynamic events, the framework activates the Cognition Search Agent ($\mathcal{A}_{search}$). It first utilizes a Keyword Generator to synthesize the user’s multimodal inputs ($I,I_{img}$) and the identified gaps $Q_{gap}$, producing precise textual queries $Q_{txt}$ and initial visual queries $Q_{img}$. Upon retrieving factual documents $\mathcal{T}_{ref}$ from open-world knowledge bases, the system performs a dual-update operation:

$$
I^{\prime}=\text{Inject}(I,\mathcal{T}_{ref}),Q^{\prime}_{img}=\text{Calibrate}(Q_{img},\mathcal{T}_{ref})
$$

where the retrieved concepts are injected back into the user instruction ($I^{\prime}$) to update the textual context, while simultaneously calibrating visual queries ($Q^{\prime}_{img}$) to ensure that subsequently retrieved reference images $\mathcal{I}_{ref}$ align with validated facts.

Internal Logical Derivation. For gaps requiring complex deduction—such as solving mathematical problems in $I_{img}$ or inferring spatial relations from retrieved data—the system triggers the CoT Knowledge Reasoning Agent ($\mathcal{A}_{reasoning}$). This engine functions as a logic processor that ingests the user instruction, the input image, and crucially, the accumulated search evidence ($\mathcal{E}_{search}=\mathcal{T}_{ref}\cup\mathcal{I}_{ref}$). It performs multi-step reasoning to resolve implicit conflicts or interpret retrieved visual data, producing explicit conclusions $\mathcal{R}_{cot}$.

The final evidence set $\mathcal{E}=\mathcal{E}_{search}\cup\mathcal{R}_{cot}$ forms a comprehensive, logically consistent cognitive context for generation.

### 3.4 Constrained Generation

The accumulation of external information introduces the risk of redundancy or irrelevance. Therefore, the final phase focuses on Information Consolidation and Conditional Synthesis. First, the Concept Review Agent ($\mathcal{A}_{review}$) serves as a consolidation mechanism to filter noise from the disjointed evidence stream $\mathcal{E}$. It synthesizes the verified facts and logical conclusions with the user’s original creative intent, rewriting them into a structured Master Prompt $P_{master}$. This prompt explicitly articulates visual attributes that were previously implicit or unknown. Subsequently, the Unified Image Generation Agent ($\mathcal{A}_{generation}$) executes the visual synthesis. Distinct from standard T2I models, $\mathcal{A}_{generation}$ is conditioned on both the text-aligned $P_{master}$ and adaptive visual cues $V_{in}$. Specifically, based on user intent, the mechanism dynamically selects between generation and editing modes to determine the visual conditioning source $V_{in}$ (i.e., from $\mathcal{I}_{ref}$ or $I_{img}$). These constraints effectively guide the model to achieve high fidelity to the user’s creative vision while strictly adhering to the factual and logical boundaries established during the knowledge acquisition phase.

## 4 Mind-Bench

### 4.1 Motivation and Task Definition

Complex image generation transcends simple text-to-pixel translation, necessitating a “Research-then-Create” paradigm akin to human artistry. However, current evaluation benchmarks often prioritize direct generation capabilities based on static knowledge. While some extend to general world knowledge, they remain relatively simplistic, suffering from a lack of temporal sensitivity and depth in multimodal reasoning. To probe the boundaries of “cognitive generation,” we propose Mind-Bench, a comprehensive benchmark comprising 500 samples designed to objectively evaluate generation capabilities dependent on dynamic external knowledge and user intent reasoning.

To systematically assess these capabilities, we categorize Mind-Bench into two primary clusters covering 10 diverse sub-domains, as shown in Figure 1.

Knowledge-Driven Tasks: This category includes five sub-domains: Special Events (e.g., breaking news scenes), Weather (real-time meteorological conditions), Character (specific IP character), Object (long-tail artifacts), and World Knowledge (Common sense in general situations). These tasks comprehensively evaluate the model’s ability to retrieve and integrate external information for precise visual grounding. The core challenge lies in mitigating hallucinations regarding Out-of-Distribution (OOD) entities.

Reasoning-Driven Tasks: This category encompasses: Life Reasoning (common sense inference), Geo Reasoning (spatial and map understanding), Math (geometric and algebraic visualization), Science & Logic (physical states and abstract logic), and Poem (imagery derived from literary metaphor). The core challenge lies in the model’s capacity to deduce implicit constraints from ostensibly simple instructions—determining whether the model can genuinely comprehend the latent reasoning results required for accurate generation. More details of task description can be found in the appendix B.1.

![Refer to caption](https://arxiv.org/html/2602.01756v1/images/mind-bench4.jpg)

Figure 3: Overview of Checklist-based Strict Accuracy (CSA) evaluation pipeline in Mind-Bench.

Table 1: Quantitative comparison of different models on Mind-Bench. The table is divided into proprietary (top) and open-source (bottom) models. The best performing model is highlighted in bold. The symbol ‘-’ indicates that the model is not applicable to I2I tasks.

<table><tbody><tr><td rowspan="2">Model Name</td><td colspan="5">Knowledge-Driven</td><td colspan="5">Reasoning-Driven</td><td rowspan="2">Overall</td></tr><tr><td>SE</td><td>Weather</td><td>MC</td><td>IP</td><td>WK</td><td>SL</td><td>Poem</td><td>Life Reason</td><td>GU</td><td>Math</td></tr><tr><td>GPT-Image-1 <sup><a href="#fn:28">28</a></sup></td><td>0.32</td><td>0.06</td><td>0.22</td><td>0.02</td><td>0.16</td><td>0.32</td><td>0.10</td><td>0.24</td><td>0.10</td><td>0.12</td><td>0.17</td></tr><tr><td>GPT-Image-1.5 <sup><a href="#fn:29">29</a></sup></td><td>0.36</td><td>0.18</td><td>0.22</td><td>0.04</td><td>0.30</td><td>0.34</td><td>0.08</td><td>0.34</td><td>0.10</td><td>0.02</td><td>0.21</td></tr><tr><td>FLUX 2 Pro <sup><a href="#fn:3">3</a></sup></td><td>0.38</td><td>0.12</td><td>0.08</td><td>0.00</td><td>0.20</td><td>0.44</td><td>0.64</td><td>0.18</td><td>0.04</td><td>0.02</td><td>0.21</td></tr><tr><td>FLUX 2 Max <sup><a href="#fn:2">2</a></sup></td><td>0.44</td><td>0.12</td><td>0.10</td><td>0.04</td><td>0.38</td><td>0.40</td><td>0.50</td><td>0.20</td><td>0.02</td><td>0.06</td><td>0.23</td></tr><tr><td>Nano Banana <sup><a href="#fn:16">16</a></sup></td><td>0.30</td><td>0.10</td><td>0.12</td><td>0.00</td><td>0.30</td><td>0.32</td><td>0.36</td><td>0.20</td><td>0.04</td><td>0.08</td><td>0.18</td></tr><tr><td>Nano Banana Pro <sup><a href="#fn:15">15</a></sup></td><td>0.50</td><td>0.36</td><td>0.40</td><td>0.16</td><td>0.56</td><td>0.62</td><td>0.68</td><td>0.30</td><td>0.16</td><td>0.46</td><td>0.41</td></tr><tr><td>SDXL <sup><a href="#fn:30">30</a></sup></td><td>0.04</td><td>0.00</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>-</td><td>-</td><td>-</td><td>0.01</td></tr><tr><td>SD-3.5 M <sup><a href="#fn:34">34</a></sup></td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>-</td><td>-</td><td>-</td><td>0.01</td></tr><tr><td>SD-3.5 L <sup><a href="#fn:33">33</a></sup></td><td>0.04</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.06</td><td>-</td><td>-</td><td>-</td><td>0.01</td></tr><tr><td>FLUX 1 dev <sup><a href="#fn:22">22</a></sup></td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.02</td><td>0.04</td><td>-</td><td>-</td><td>-</td><td>0.02</td></tr><tr><td>FLUX 1 Kontext <sup><a href="#fn:21">21</a></sup></td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>-</td><td>-</td><td>-</td><td>0.01</td></tr><tr><td>FLUX 1 Krea <sup><a href="#fn:22">22</a></sup></td><td>0.04</td><td>0.00</td><td>0.04</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.02</td><td>-</td><td>-</td><td>-</td><td>0.02</td></tr><tr><td>Bagel <sup><a href="#fn:11">11</a></sup></td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.00</td><td>0.08</td><td>0.02</td></tr><tr><td>Echo-4o <sup><a href="#fn:46">46</a></sup></td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.06</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td></tr><tr><td>DraCo <sup><a href="#fn:18">18</a></sup></td><td>0.02</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.02</td><td>0.04</td><td>0.02</td><td>0.06</td><td>0.02</td></tr><tr><td>Z-Image <sup><a href="#fn:4">4</a></sup></td><td>0.02</td><td>0.00</td><td>0.08</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>-</td><td>-</td><td>-</td><td>0.02</td></tr><tr><td>Qwen-Image <sup><a href="#fn:41">41</a></sup></td><td>0.08</td><td>0.00</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.04</td><td>0.00</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.02</td></tr><tr><td>Mind-Brush (Ours)</td><td>0.54</td><td>0.16</td><td>0.62</td><td>0.18</td><td>0.40</td><td>0.26</td><td>0.54</td><td>0.10</td><td>0.16</td><td>0.14</td><td>0.31</td></tr></tbody></table>

### 4.2 Benchmark Construction

Mind-Bench is constructed through a rigorous Human-Machine Collaborative Pipeline to ensure multidimensional complexity and factual reliability.

First, distinct from random web-crawling, we recruited 6 graduate students in AI to carefully curate high-difficulty prompts. For each prompt, annotators manually collected strongly correlated multimodal evidence (e.g., official news reports, authoritative reference images) to establish objective factual anchors.

Subsequently, annotators utilized LLMs to generate candidate fine-grained evaluation checklist items based on the collected evidence. These items underwent strict human verification to eliminate redundancies and ensure executability. This process yields a final set of samples, each equipped with the input instruction, multimodal reference evidence, and a rigorous evaluation checklist.

### 4.3 Evaluation Criterion

Previous image generation benchmarks typically employ CLIP scores or MLLM ratings to assess generation quality across abstract perceptual dimensions. However, these metrics exhibit significant limitations in accuracy: they predominantly operate at a coarse semantic level (e.g., verifying the presence of a generic object) but fail to distinguish identity-level details or specific factual attributes.

To accurately reflect model usability in complex cognitive tasks, we propose Checklist-based Strict Accuracy (CSA) as the core metric, as illustrated in Figure 3. This criterion employs an MLLM judge to scrutinize the generated image against the checklist under a Holistic Pass Criterion: a sample is deemed correct only if all sub-items are verified as “Pass”. For a dataset with $N$ samples, the accuracy is defined as:

$$
Acc_{\text{CSA}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\prod_{j=1}^{|C_{i}|}\text{VQA}(I_{gen}^{(i)},c_{j}^{(i)})=1\right)
$$

where $\mathbb{I}(\cdot)$ is the indicator function and $\text{VQA}(I,c)$ returns 1 if image $I$ satisfies item $c$. This metric offers a rigorous evaluation, effectively penalizing generations that are partially correct but logically flawed.

## 5 Experiments

Table 2: Quantitative comparison of different models on WISE and RISEBench. The table is divided into proprietary (top) and open-source (bottom) models. Within each group, the best performing model is highlighted in bold. The symbol ‘-’ indicates that the model is not applicable to I2I tasks.

<table><tbody><tr><td rowspan="2">Model Name</td><td colspan="7">WISE</td><td colspan="4">RISEBench</td></tr><tr><td>Cultural</td><td>Time</td><td>Space</td><td>Bio</td><td>Phys</td><td>Chem</td><td>Overall</td><td>Instr. Reas.</td><td>App. Consis.</td><td>Vis. Plaus.</td><td>Accuracy</td></tr><tr><td>GPT-Image-1 <sup><a href="#fn:28">28</a></sup></td><td>0.81</td><td>0.71</td><td>0.89</td><td>0.83</td><td>0.79</td><td>0.74</td><td>0.80</td><td>62.8</td><td>80.2</td><td>94.9</td><td>28.9</td></tr><tr><td>Nano Banana <sup><a href="#fn:16">16</a></sup></td><td>0.89</td><td>0.87</td><td>0.95</td><td>0.89</td><td>0.89</td><td>0.79</td><td>0.89</td><td>61.2</td><td>86.0</td><td>91.3</td><td>32.8</td></tr><tr><td>Nano Banana Pro <sup><a href="#fn:15">15</a></sup></td><td>0.89</td><td>0.80</td><td>0.89</td><td>0.88</td><td>0.86</td><td>0.85</td><td>0.87</td><td>77.0</td><td>85.5</td><td>94.4</td><td>47.2</td></tr><tr><td>FLUX.1-dev <sup><a href="#fn:22">22</a></sup></td><td>0.48</td><td>0.58</td><td>0.62</td><td>0.42</td><td>0.51</td><td>0.35</td><td>0.50</td><td>26.0</td><td>71.6</td><td>85.2</td><td>1.9</td></tr><tr><td>FLUX.1-Canny <sup><a href="#fn:22">22</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>20.2</td><td>13.1</td><td>77.5</td><td>0.0</td></tr><tr><td>SD-XL-base <sup><a href="#fn:30">30</a></sup></td><td>0.43</td><td>0.48</td><td>0.47</td><td>0.44</td><td>0.45</td><td>0.27</td><td>0.43</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SD-3.5-medium <sup><a href="#fn:34">34</a></sup></td><td>0.43</td><td>0.50</td><td>0.52</td><td>0.41</td><td>0.53</td><td>0.33</td><td>0.45</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SD-3.5-large <sup><a href="#fn:33">33</a></sup></td><td>0.44</td><td>0.50</td><td>0.58</td><td>0.44</td><td>0.52</td><td>0.31</td><td>0.46</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BAGEL (w/ CoT) <sup><a href="#fn:11">11</a></sup></td><td>0.76</td><td>0.69</td><td>0.75</td><td>0.65</td><td>0.75</td><td>0.58</td><td>0.70</td><td>45.9</td><td>73.8</td><td>80.1</td><td>11.9</td></tr><tr><td>BAGEL <sup><a href="#fn:11">11</a></sup></td><td>0.44</td><td>0.55</td><td>0.68</td><td>0.44</td><td>0.60</td><td>0.39</td><td>0.52</td><td>36.5</td><td>53.5</td><td>73.0</td><td>6.1</td></tr><tr><td>Qwen-Image <sup><a href="#fn:41">41</a></sup></td><td>0.62</td><td>0.63</td><td>0.77</td><td>0.57</td><td>0.75</td><td>0.40</td><td>0.62</td><td>49.9</td><td>71.0</td><td>91.5</td><td>19.4</td></tr><tr><td>GenAgent <sup><a href="#fn:19">19</a></sup></td><td>0.78</td><td>0.67</td><td>0.78</td><td>0.72</td><td>0.77</td><td>0.55</td><td>0.72</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Mind-Brush (Ours)</td><td>0.83</td><td>0.69</td><td>0.84</td><td>0.71</td><td>0.85</td><td>0.68</td><td>0.78</td><td>61.5</td><td>79.4</td><td>86.5</td><td>24.7</td></tr></tbody></table>

### 5.1 Benchmarks and Evaluation Protocols

To comprehensively evaluate the capabilities of current methods in understanding user intent and generating long-tail concepts, we selected three benchmarks with distinct focuses. First, our proposed Mind-Bench assesses generation capabilities dependent on dynamic external knowledge and multi-step reasoning using Checklist-based Strict Accuracy (CSA). We also include WISE, which focuses on complex semantic understanding and world knowledge integration evaluated via WiScore, and RISEBench, which evaluates joint text-image analysis capabilities across four reasoning dimensions (Instruction Reasoning, Appearance Consistency, Visual Plausibility, and Accuracy). Further details regarding the evaluation protocols and implementation can be found in the Appendix A.4.

### 5.2 Experimental Settings

Baselines. We compare Mind-Brush against current mainstream proprietary UMMs, including GPT-Image-1 [^28], GPT-Image-1.5 [^29], Nano Banana [^16], Nano Banana Pro [^15], FLUX-2 Pro [^3], and FLUX-2 Max [^2]. Additionally, we compare it with state-of-the-art (SOTA) open-source T2I models or UMMs, including FLUX.1 dev [^22], FLUX 1 Kontext [^21], FLUX 1 Krea [^22], Stable Diffusion (SD) 3.5 Large [^33], SD 3.5 Medium [^34], SD-XL [^30], Bagel [^11], Echo-4o [^46], DraCo [^18], Z-Image [^4], GenAgent [^19] and Qwen-Image [^41]. All baselines are evaluated using their official default settings.

Implementation Details. To ensure a fair comparison, Mind-Brush maintains consistent experimental settings across all benchmarks. We employ Qwen-Image-Edit-2512 as the image-guided T2I model and Qwen-Image as the prompt-guided T2I model. For the backbone MLLM, we uniformly utilize GPT-5.1 across all agents. Regarding search tools, we use the Google Search API for retrieval, setting the limit for text search results to 2 and truncating web content at 2000 words to ensure sufficient textual evidence without incurring excessive token costs. The limit for image search results is set to 5. All experiments involving open-source models were conducted on 8 NVIDIA A100 80G GPUs.

![Refer to caption](https://arxiv.org/html/2602.01756v1/images/searchcase4.jpg)

Figure 4: Qualitative Comparison of different models on Mind-Bench, including knowledge-driven (upper part) and reasoning-driven (lower part) tasks. The green border indicates that the generated result matches the facts, while the red border indicates the opposite.

### 5.3 Comparison with State of the Art

Table 1 presents the quantitative comparison results on Mind-Bench. Experimental results demonstrate that Mind-Brush achieves a significant improvement in overall generation accuracy compared to open-source T2I and UMM models, surpassing SD-3.5 Large and Qwen-Image by 30.0% and 29.0%, respectively. Notably, while the baseline accuracy of Qwen-Image on Mind-Bench is considerably lower than that of proprietary methods, our Mind-Brush framework empowers these base models to match or even exceed the performance of several proprietary UMMs. Specifically, in tasks heavily reliant on search, such as Special Events(SE) and Character(MC). In Geo Understanding (GU), which demands cross-modal reasoning, Mind-Brush outperforms the vast majority of proprietary models and matches the performance of Nano Banana Pro. Furthermore, our framework achieves a 47.6% improvement in overall accuracy compared to GPT-Image-1.5. This validates the effectiveness of the textual and visual references provided by our retrieval and reasoning modules, achieving a unification of understanding and generation on open-source models through active knowledge acquisition and reasoning. The experimental results also indicate that Mind-Bench poses a significant challenge to current mainstream models.

Table 3: Ablation Study on Knowledge and Reasoning Modules

| Ablation Setting | Knowledge- Driven | Reasoning- Driven | Overall |
| --- | --- | --- | --- |
| Baseline | 0.02 | 0.02 | 0.02 |
| \+ $\mathcal{A}_{reasoning}$ | 0.11 | 0.21 | 0.17 |
| \+ $\mathcal{A}_{search}$ | 0.30 | 0.20 | 0.25 |
| Mind-Brush (Ours) | 0.38 | 0.24 | 0.31 |

Mind-Brush also delivers outstanding performance on WISE, which emphasizes world knowledge, and RISEBench, which focuses on reasoning logic. As shown in Table 2, on WISE, Mind-Brush outperforms all proprietary image generation models, boosting the overall WiScore by 25.8% compared to Qwen-Image and reaching 0.78, matching the performance of GPT-Image-1. On RISEBench, our method achieves a score of 61.5 in Instruction Reasoning, surpassing Nano Banana and exceeding Bagel by 68.5%. Furthermore, our overall accuracy approaches that of GPT-Image-1. These results further corroborate that Mind-Brush effectively integrates multimodal inputs to infer implicit user intents, thereby realizing accurate image generation.

Figure 4 illustrates the qualitative comparison between Mind-Brush and competing baselines on Mind-Bench. In the context of Knowledge-Driven tasks, Mind-Brush effectively leverages search tools to retrieve pertinent visual references, thereby achieving accurate synthesis of Out-of-Distribution concepts that baseline models fail to recognize. Regarding Reasoning-Driven tasks, the Knowledge Reasoning Agent dissects implicit attribute features within concise user instructions—such as spatial relationships in the output or the mathematical logic underlying input questions. This capability allows the system to effectively manifest the user’s intricate intent in the generated imagery, ensuring logical coherence alongside visual fidelity.

### 5.4 Ablation Study

To validate the efficacy of the individual components within Mind-Brush, we conducted comprehensive ablation studies on the core Reasoning and Search tools using the Mind-Bench dataset. As reported in Table 3, the incorporation of the Knowledge Reasoning Agent yielded significant accuracy gains on Reasoning-Driven tasks compared to the baseline. Conversely, the Cognition Search Agent effectively compensated for the model’s cognitive deficits regarding unknown concepts, resulting in an accuracy improvement of 0.28 on Knowledge-Driven tasks over the baseline. Furthermore, the simultaneous deployment of both agents demonstrates a strong synergistic effect. This combination not only bridges cognitive gaps regarding OOD concepts but also accurately deciphers the user’s authentic generative intent. Consequently, the full framework achieves accuracy improvements of 0.17 and 0.06 compared to configurations using solely the Reasoning Agent or the Search Agent, respectively. More ablation study of can be found in the appendix A.3.

## 6 Conclusion

We introduced Mind-Brush, a training-free agentic framework that transforms text-to-image generation from passive decoding into an active cognitive workflow. By orchestrating intent analysis, multimodal grounding, and explicit chain-of-thought reasoning, Mind-Brush effectively bridges the gap between vague user intents and precise, factually grounded visual synthesis. To rigorously evaluate this, we established Mind-Bench, a benchmark designed to stress-test models on knowledge-intensive and reasoning-dependent tasks. Empirical results demonstrate that our framework significantly outperforms state-of-the-art models, validating the synergy of active retrieval and logical deduction. We believe this shift towards an Agentic Generative Paradigm paves the way for next-generation systems capable of complex problem-solving in visual synthesis.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

## Appendix

## Appendix A Additional Implementary Details of Mind-Brush

### A.1 Workflow of Mind-Brush

Algorithm 1 formally delineates the inference workflow of the Mind-Brush framework. Unlike static generation paradigms, Mind-Brush operates as a dynamic, agentic cognitive system designed to align vague user intents with objective reality.

The workflow initiates with the reception of a User Instruction ($I_{inst}$) and an optional User Image ($I_{img}$), alongside the initialization of core foundation models and toolsets. In the Intent Analysis phase, the system structurally decomposes the input using the 5W1H paradigm to pinpoint cognitive gaps ($Q_{gap}$) and formulates a dynamic execution strategy ($\mathcal{S}_{plan}$).

Driven by this strategy, the system branches into active knowledge acquisition. If Cognition Search is triggered, the agent performs a two-stage retrieval process: gathering textual evidence ($\mathcal{T}_{ref}$) to ground factual concepts and subsequently refining visual queries to retrieve semantically accurate reference images ($\mathcal{I}_{ref}$). Conversely, for tasks necessitating deep logic or cross-modal understanding, the Knowledge Reasoning agent is activated. Crucially, this agent ingests the user image ($I_{img}$) alongside the instruction and gathered evidence, employing Chain-of-Thought (CoT) reasoning to derive explicit logical conclusions ($\mathcal{R}_{cot}$) and resolve implicit constraints.

Finally, the Concept Review agent acts as a convergence hub, synthesizing the user’s original intent with the disjointed streams of multimodal evidence ($\mathcal{E}$) into a coherent Master Prompt ($P_{master}$). This prompt, enriched with factual and logical precision, guides the Image Generator ($\mathcal{G}_{\theta}$) to synthesize the final high-fidelity image ($\mathcal{I}_{final}$), ensuring rigorous alignment with both user creativity and real-world logic.

Algorithm 1 Inference Workflow of Mind-Brush Framework

 Input: User Instruction $I_{inst}$, User Image $I_{img}$ (Optional)

 Initialization: Large Language Model $\text{LLM}_{\phi}$, Image Generator $\mathcal{G}_{\theta}$, Search Tools; Evidence set $\mathcal{E}\leftarrow\emptyset$, Visual References $\mathcal{I}_{ref}\leftarrow\emptyset$

 Decompose $I_{inst}$ via 5W1H paradigm and identify cognitive gaps $Q_{gap}$ ($\mathcal{A}_{intent}$)

 Formulate execution strategy $\mathcal{S}_{plan}$ based on $Q_{gap}$

 if $\mathcal{S}_{plan}$ requires Cognition Search then

  Generate text queries $Q_{txt}$ and initial visual queries $Q_{v}$

   $\mathcal{T}_{ref}\leftarrow\text{Search}_{txt}(Q_{txt})_{k}$

  Update Evidence $\mathcal{E}\leftarrow\mathcal{E}\cup\mathcal{T}_{ref}$

   $Q^{\prime}_{v}\leftarrow\Phi_{refine}(Q_{v},\mathcal{T}_{ref})$    $\mathcal{I}_{ref}\leftarrow\text{Search}_{img}(Q^{\prime}_{v})_{k}$

 end if

 if $\mathcal{S}_{plan}$ requires Knowledge Reasoning then

  Retrieve relevant context from $\mathcal{E}$, $I_{inst}$ and $I_{img}$

   $\mathcal{R}_{cot}\leftarrow\Phi_{reason}(I_{inst},I_{img},Q_{gap},\mathcal{E})$

  Update Evidence $\mathcal{E}\leftarrow\mathcal{E}\cup\mathcal{R}_{cot}$

 end if

 Synthesize disjointed streams $\{I_{inst},I_{img},\mathcal{E}\}$ and resolve ambiguities ($\mathcal{A}_{review}$)

  $P_{master}\leftarrow\text{Rewrite}(I_{inst},I_{img},\mathcal{E})$ to generate Master Prompt

  $\mathcal{I}_{final}\leftarrow\mathcal{G}_{\theta}(P_{master},\mathcal{I}_{ref})$ conditioned on visual references

 return High-fidelity Generated Image $\mathcal{I}_{final}$

### A.2 Additional Experimental Results

Due to space limitations in the manuscript, we conducted additional evaluations of Mind-Brush on the GenEval++ [^46] and Imagine-Bench [^46] benchmarks. GenEval++ is an enhanced instruction-following benchmark that utilizes Accuracy as its metric. It increases instruction complexity and compositional difficulty across various task categories (e.g., Color, Count, Position) to mitigate metric saturation. Imagine-Bench is a benchmark dedicated to creative generation, evaluating models on task categories such as Attribute Shift and Spatiotemporal Anomalies. It employs a Comprehensive Score to assess semantic understanding and visual quality in surreal fantasy scenarios.

Experimental results demonstrate that Mind-Brush achieves impressive performance on these challenging benchmarks. As shown in Table 4, compared to Agentic methods such as PromptEnhancer and GenAgent, our method outperforms the best performance baseline (GenAgent) on GenEval++. Specifically, Mind-Brush improves Accuracy by 41.7% on the Pos/Count task and 13.3% on the Multi-Count task, reaching performance levels close to GPT-4o. Table 5 presents the performance comparison on Imagine-Bench; our results show improvements in Score of 8.2% and 2.6% on the Spatiotemporal and Hybridization tasks, respectively, compared to GenAgent. Extensive experimental results validate the superior capability of our framework in handling complex long-tail instructions and open-world creative generation tasks.

Table 4: Quantitative Comparison of different methods on GenEval++. The best performing model (excluding close-sourece model GPT4o) is highlighted in bold and underline indicates the second best.

| Method | Color | Count | Color/Count | Color/Pos | Pos/Count | Pos/Size | Multi-Count | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLUX.1-dev [^22] | 0.400 | 0.600 | 0.250 | 0.250 | 0.075 | 0.400 | 0.300 | 0.325 |
| Qwen-Image [^41] | 0.875 | 0.725 | 0.725 | 0.600 | 0.475 | 0.725 | 0.550 | 0.668 |
| GPT4o [^1] | 0.900 | 0.675 | 0.725 | 0.625 | 0.600 | 0.800 | 0.850 | 0.739 |
| Janus Pro 7B [^8] | 0.450 | 0.300 | 0.125 | 0.300 | 0.075 | 0.350 | 0.125 | 0.246 |
| T2I-R1 [^17] | 0.675 | 0.325 | 0.200 | 0.350 | 0.075 | 0.250 | 0.300 | 0.311 |
| Bagel [^11] | 0.325 | 0.600 | 0.250 | 0.325 | 0.250 | 0.475 | 0.375 | 0.371 |
| PromptEnhancer [^43] | 0.500 | 0.625 | 0.225 | 0.375 | 0.125 | 0.450 | 0.375 | 0.382 |
| ReflectionFLow [^54] | 0.400 | 0.625 | 0.275 | 0.275 | 0.200 | 0.425 | 0.325 | 0.361 |
| GenAgent [^19] | 0.775 | 0.775 | 0.650 | 0.800 | 0.600 | 0.725 | 0.750 | 0.725 |
| Mind-Brush (Ours) | 0.775 | 0.700 | 0.775 | 0.750 | 0.850 | 0.775 | 0.850 | 0.782 |

Table 5: Quantitative Comparison of different methods on Imagine. The best performing model (excluding close-sourece model GPT4o) is highlighted in bold and underline indicates the second best.

| Method | Attribute Shift | Spatiotemporal | Hybridization | Multi-Object | Overall |
| --- | --- | --- | --- | --- | --- |
| FLUX.1-dev [^22] | 5.298 | 6.350 | 7.053 | 5.973 | 6.072 |
| Qwen-Image [^41] | 6.771 | 7.193 | 8.130 | 7.500 | 7.329 |
| GPT4o [^1] | 8.540 | 9.180 | 8.570 | 7.980 | 8.560 |
| Janus Pro 7B [^8] | 5.300 | 7.280 | 6.730 | 6.040 | 6.220 |
| T2I-R1 [^17] | 5.850 | 7.700 | 7.360 | 6.680 | 6.780 |
| Bagel [^11] | 5.370 | 6.930 | 6.500 | 6.410 | 6.200 |
| PromptEnhancer [^43] | 5.489 | 6.213 | 7.327 | 6.493 | 6.281 |
| GenAgent [^19] | 7.613 | 7.547 | 8.343 | 7.763 | 7.794 |
| Mind-Brush (Ours) | 7.416 | 8.167 | 8.557 | 7.533 | 7.862 |

### A.3 Additional Ablation Study

To rigorously evaluate the generalizability and robustness of the Mind-Brush framework, we conducted ablation studies involving diverse configurations of the Multimodal Large Language Model (MLLM) backbone and the image generation engine. The quantitative results, presented in Table 6, confirm the framework’s efficacy across distinct computational regimes and disentangle the contributions of the agentic intelligence substrate versus the visual synthesis capabilities.

Quantitative Comparison with Different MLLM Baselines. To isolate the contribution of the multimodal foundation model that drives the entire agentic workflow, we evaluated the framework’s performance by replacing the MLLM backbone across all constituent agents while holding the generation model constant. We first tested a fully open-source configuration utilizing Qwen3-VL-235B as the universal backbone for all agents and Qwen-Image as the visual generator. Despite relying on accessible open-weights models, this configuration achieves an overall CSA score of 0.24, surpassing the proprietary baseline GPT-Image-1.5 (0.21) by a relative margin of 14.3%. This empirical evidence suggests that the collaborative agentic workflow can effectively compensate for the capacity gaps between open-source models and state-of-the-art proprietary baselines. Furthermore, by upgrading the framework’s backbone from Qwen3-VL to GPT-5.1 (while keeping Qwen-Image fixed), the overall performance significantly improves from 0.24 to 0.31, yielding a 29.2% relative gain. This finding is pivotal, demonstrating that the intelligence level of the underlying MLLM is the dominant factor. A more capable backbone enhances the precision of every agentic step—from evidence retrieval to constraint verification—directly translating into higher generation fidelity even without altering the image decoder.

Quantitative Comparison with Different Image Generation Models. We further investigated the framework’s capacity to empower and scale with different visual executors. Integrating the legacy GPT-Image-1 into the Mind-Brush framework (driven by the GPT-5.1 backbone) results in a substantial performance leap, where the overall CSA score increases from the baseline of 0.17 to 0.34, representing a 100% relative improvement. This confirms that Mind-Brush serves as a powerful meta-architecture capable of unlocking the latent potential of weaker generators. Notably, under the same agentic backbone (GPT-5.1), the GPT-Image-1 based configuration outperforms the Qwen-Image variant. This indicates that while the agentic backbone is crucial for intent alignment and factuality, the framework effectively utilizes the superior visual synthesis capabilities of stronger generation engines, ensuring that performance scales consistently with the quality of both the intelligence substrate and the visual decoder.

Table 6: Ablation study of different MLLM backbones and image generation model. The best performing model is highlighted in bold.

<table><tbody><tr><td rowspan="2">MLLM model</td><td rowspan="2">Generation model</td><td colspan="5">Knowledge-Intensive</td><td colspan="5">Reasoning-Intensive</td><td rowspan="2">Overall</td></tr><tr><td>SE</td><td>Weather</td><td>MC</td><td>IP</td><td>WK</td><td>Life Reason</td><td>GU</td><td>Math</td><td>SL</td><td>Poem</td></tr><tr><td>-</td><td>GPT-Image-1</td><td>0.32</td><td>0.06</td><td>0.22</td><td>0.02</td><td>0.16</td><td>0.24</td><td>0.10</td><td>0.12</td><td>0.32</td><td>0.10</td><td>0.17</td></tr><tr><td>-</td><td>GPT-Image-1.5</td><td>0.36</td><td>0.18</td><td>0.22</td><td>0.04</td><td>0.30</td><td>0.34</td><td>0.10</td><td>0.02</td><td>0.34</td><td>0.08</td><td>0.21</td></tr><tr><td>-</td><td>Nano Banana</td><td>0.30</td><td>0.10</td><td>0.12</td><td>0.00</td><td>0.30</td><td>0.20</td><td>0.04</td><td>0.08</td><td>0.32</td><td>0.36</td><td>0.18</td></tr><tr><td>-</td><td>Nano Banana Pro</td><td>0.50</td><td>0.36</td><td>0.40</td><td>0.16</td><td>0.56</td><td>0.30</td><td>0.16</td><td>0.46</td><td>0.62</td><td>0.68</td><td>0.41</td></tr><tr><td colspan="13">Mind-Brush</td></tr><tr><td>Qwen3-VL-235B</td><td>Qwen-Image</td><td>0.42</td><td>0.06</td><td>0.44</td><td>0.10</td><td>0.36</td><td>0.06</td><td>0.14</td><td>0.18</td><td>0.20</td><td>0.44</td><td>0.24</td></tr><tr><td>GPT-5.1</td><td>Qwen-Image</td><td>0.54</td><td>0.16</td><td>0.62</td><td>0.18</td><td>0.40</td><td>0.10</td><td>0.16</td><td>0.14</td><td>0.26</td><td>0.54</td><td>0.31</td></tr><tr><td>GPT-5.1</td><td>GPT-Image-1</td><td>0.64</td><td>0.18</td><td>0.56</td><td>0.10</td><td>0.50</td><td>0.28</td><td>0.10</td><td>0.06</td><td>0.50</td><td>0.48</td><td>0.34</td></tr></tbody></table>

### A.4 Evaluation Protocols

To ensure a rigorous and standardized assessment of model performance across diverse cognitive dimensions, we strictly adhere to the official evaluation protocols defined in the respective benchmarks. All automated evaluations for existing benchmarks utilize the official LLM settings as the core judge to maintain consistency with prior literature.

#### WISE Evaluation.

For the WISE benchmark, which necessitates deep semantic comprehension and world knowledge integration, we report the WiScore following the official definition. This composite metric provides a quantitative measure of knowledge-image alignment by aggregating three sub-dimensions on a discrete 3-point scale ($s\in\{0,1,2\}$): Consistency ($S_{con}$), Realism ($S_{real}$), and Aesthetic Quality ($S_{aes}$). Formally, the WiScore is calculated as a weighted linear combination:

$$
\text{WiScore}=\alpha_{1}\cdot\bar{S}_{con}+\alpha_{2}\cdot\bar{S}_{real}+\alpha_{3}\cdot\bar{S}_{aes}
$$

The weights are rigorously calibrated to $\alpha_{1}=0.7$, $\alpha_{2}=0.2$, and $\alpha_{3}=0.1$. This weighting strategy prioritizes the model’s fidelity to implicit semantic constraints (Consistency) while simultaneously penalizing violations of physical laws (Realism) and low artistic quality (Aesthetic), thereby offering a holistic view of generation quality grounded in world knowledge.

#### RISEBench Evaluation.

RISEBench targets reasoning-informed visual editing, where precision is paramount. Following its official guidelines, we conduct a fine-grained evaluation using GPT-4.1 to score results on a scale of 1 to 5 across three critical dimensions: Instruction Reasoning ($S_{ir}$), which measures the accurate execution of complex logical directives; Appearance Consistency ($S_{ac}$), which assesses the preservation of task-irrelevant visual attributes; and Visual Plausibility ($S_{vp}$), which evaluates the coherence of the edited output. Distinct from standard aggregation methods, RISEBench imposes a strict All-or-Nothing success criterion. A sample is deemed successfully solved if and only if it achieves perfect scores across all applicable dimensions. The final accuracy is defined as:

$$
\text{Acc}_{\text{RISE}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(S_{ir}^{(i)}=5\land S_{ac}^{(i)}=5\land S_{vp}^{(i)}=5\right)
$$

where $\mathbb{I}(\cdot)$ denotes the indicator function. This rigorous metric effectively penalizes partial failures, ensuring that high performance reflects robust reasoning capabilities rather than successful approximations.

#### Mind-Bench Evaluation.

For our proposed Mind-Bench, to rigorously quantify the model’s ability to bridge cognitive gaps without hallucination, we employ the Checklist-based Strict Accuracy (CSA). To mitigate potential self-evaluation biases inherent in GPT-series models, we utilize Gemini-3.0-Pro as the expert evaluator. Each test sample is associated with a human-verified set of atomic factual claims $\mathcal{C}=\{c_{1},\dots,c_{k}\}$. The evaluator performs a binary verification ($\{0,1\}$) for each claim. We adopt a Holistic Pass Criterion, where a generation is considered correct only if it satisfies all constraints in the checklist:

$$
\text{Acc}_{\text{CSA}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\prod_{j=1}^{|\mathcal{C}_{i}|}\text{VQA}_{\mathcal{M}}(I_{gen}^{(i)},c_{j}^{(i)})=1\right)
$$

This stringent protocol ensures that the reported accuracy reflects comprehensive understanding and precise grounding of external knowledge, rather than superficial semantic overlap.

Table 7: Overview of Mind-Bench. Task distribution, types, and per-task definitions across Knowledge and Reasoning domains.

<table><tbody><tr><td>Task</td><td>Number</td><td>Type</td><td>Definition</td></tr><tr><td colspan="4">Knowledge-driven</td></tr><tr><td>News</td><td>50</td><td>T2I</td><td>Retrieve and generate images of specific news events based on the provided temporal and spatial contexts.</td></tr><tr><td>Weather</td><td>50</td><td>T2I</td><td>Retrieve and generate images of meteorological conditions for specific times and locations.</td></tr><tr><td>Character</td><td>50</td><td>T2I</td><td>Retrieve and generate images of specific personas, celebrities, or fictional characters from user input.</td></tr><tr><td>IP</td><td>50</td><td>T2I</td><td>Retrieve and generate images of products and artifacts associated with well-known intellectual properties.</td></tr><tr><td>World Knowledge</td><td>50</td><td>T2I</td><td>Retrieve and generate images corresponding to specific factual and historical information about the world.</td></tr><tr><td colspan="4">Reasoning-driven</td></tr><tr><td>Life Reasoning</td><td>50</td><td>I2I</td><td>Reason and generate images related to daily life tasks and their outcomes based on provided lifestyle imagery.</td></tr><tr><td>Geo Understanding</td><td>50</td><td>I2I</td><td>Reason, retrieve, and generate images of specific locations based on input map imagery and spatial contexts.</td></tr><tr><td>Math</td><td>50</td><td>I2I</td><td>Reason visual mathematical problems and generate images rendering the step-by-step results.</td></tr><tr><td>Science</td><td>50</td><td>T2I</td><td>Reason and generate images depicting scientific phenomena, physical states, or experimental processes.</td></tr><tr><td>Poem</td><td>50</td><td>T2I</td><td>Reason and generate visual scenes that embody the specific imagery and metaphors of poems given the poet and emotion.</td></tr></tbody></table>

Table 8: Comparison between Mind-Bench and existing T2I benchmarks.

| Benchmark | Up-to-date | Reasoning modality | Sample Num | Task Num | Metric Type |
| --- | --- | --- | --- | --- | --- |
| GenEval [^13] | No | Text | $\sim$ 550 | 6 | Scoring |
| GenEval++ [^46] | No | Text | $\sim$ 280 | 7 | Accuracy |
| WISE [^27] | No | Text | 1,000 | 25 | Scoring |
| T2I-ReasonBench [^36] | No | Text + Image | 800 | 4 | Scoring |
| RISEBench [^51] | No | Text + Image | 360 | 4 | Scoring + Accuracy |
| Mind-Bench (Ours) | Yes | Text + Image | 500 | 10 | Accuracy (CSA) |

## Appendix B Additional Details of Mind-Bench

### B.1 Additional task description of Mind-Bench

Due to space limitations in the main manuscript, we provide a comprehensive specification of the task taxonomy within Mind-Bench in this section. Table 7 presents a detailed overview, outlining the sample distribution, input modalities, and precise definitions for each of the 10 distinct task categories, structured across Knowledge-driven and Reasoning-driven domains to fully elucidate the benchmark’s evaluation scope.

### B.2 Comparison with Existing Benchmarks

To contextualize the contributions of Mind-Bench, Table 8 presents a comprehensive comparison with existing T2I benchmarks, highlighting our distinct advantages in temporality, modality, and evaluation rigor. Uniquely, Mind-Bench serves as the sole platform for assessing real-time information retrieval, diverging from traditional benchmarks like GenEval and WISE that rely on static, frozen knowledge distributions to explicitly target dynamic concepts. Beyond simple text alignment, it incorporates multimodal contexts to evaluate complex reasoning, matching the depth of specialized benchmarks like T2I-ReasonBench while offering superior breadth across 10 distinct task categories. Furthermore, by adopting the Checklist-based Strict Accuracy (CSA) metric instead of subjective scalar scoring, Mind-Bench ensures a precise, objective standard for validating the capabilities of agentic generation systems.

## Appendix C Additional Visualization of Inference Process

To provide a more intuitive understanding of our framework, we present additional qualitative examples (from fig. 5 to fig. 24) that visualize the complete step-by-step cognitive trajectory of Mind-Brush, ranging from initial intent analysis to the final image synthesis across diverse scenarios.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/event1.jpg)

Figure 5: A generation process of Mind-Brush in Special Events task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/event2.jpg)

Figure 6: A generation process of Mind-Brush in Special Events task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/weather1.jpg)

Figure 7: A generation process of Mind-Brush in Weather task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/weather2.jpg)

Figure 8: A generation process of Mind-Brush in Weather task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/cha1.jpg)

Figure 9: A generation process of Mind-Brush in Character task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/cha2.jpg)

Figure 10: A generation process of Mind-Brush in Character task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/object1.jpg)

Figure 11: A generation process of Mind-Brush in Object task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/object2.jpg)

Figure 12: A generation process of Mind-Brush in Object task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/world1.jpg)

Figure 13: A generation process of Mind-Brush in World Knowledge task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/world2.jpg)

Figure 14: A generation process of Mind-Brush in World Knowledge task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/life1.jpg)

Figure 15: A generation process of Mind-Brush in Life Reasoning task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/life2.jpg)

Figure 16: A generation process of Mind-Brush in Life Reasoning task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/Geo1.jpg)

Figure 17: A generation process of Mind-Brush in Geo Reasoning task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/Geo2.jpg)

Figure 18: A generation process of Mind-Brush in Geo Reasoning task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/math1.jpg)

Figure 19: A generation process of Mind-Brush in Geo Math task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/math2.jpg)

Figure 20: A generation process of Mind-Brush in Math task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/science1.jpg)

Figure 21: A generation process of Mind-Brush in Science & logic task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/science2.jpg)

Figure 22: A generation process of Mind-Brush in Science & logic task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/poem1.jpg)

Figure 23: A generation process of Mind-Brush in Poem task of Mind-Bench.

![Refer to caption](https://arxiv.org/html/2602.01756v1/supplment_case_resize/poem2.jpg)

Figure 24: A generation process of Mind-Brush in Poem task of Mind-Bench.

[^1]: Gpt-4 technical report. arXiv preprint arXiv:2303.08774. Cited by: Table 4, Table 5.

[^2]: FLUX 2 max: next generation image synthesis. Note: [https://bfl.ai/models/flux-2-max](https://bfl.ai/models/flux-2-max) Accessed: 2026-01-26 Cited by: Table 1, §5.2.

[^3]: FLUX 2 pro: state-of-the-art quality at maximum speed.. Note: [https://bfl.ai/models/flux-2](https://bfl.ai/models/flux-2) Accessed: 2026-01-26 Cited by: §1, Table 1, §5.2.

[^4]: Z-image: an efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699. Cited by: §1, Table 1, §5.2.

[^5]: 5W1H extraction with large language models. External Links: 2405.16150, [Link](https://arxiv.org/abs/2405.16150) Cited by: §3.2.

[^6]: T2i-copilot: a training-free multi-agent text-to-image system for enhanced prompt interpretation and interactive generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19396–19405. Cited by: §1, §2.1.

[^7]: Blip3-o: a family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568. Cited by: §2.2.

[^8]: Janus-pro: unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811. Cited by: Table 4, Table 5.

[^9]: Mindsearch: mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183. Cited by: §1.

[^10]: Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261. Cited by: §1.

[^11]: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683. Cited by: Table 4, Table 5, §1, Table 1, §5.2, Table 2, Table 2.

[^12]: Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, Cited by: §1.

[^13]: Geneval: an object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36, pp. 52132–52152. Cited by: Table 8, §1, §2.3.

[^14]: Seedream 2.0: a native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703. Cited by: §1.

[^15]: Gemini image pro: high-quality image generation. Note: [https://deepmind.google/models/gemini-image/pro/](https://deepmind.google/models/gemini-image/pro/) Accessed: 2026-01-26 Cited by: Table 1, §5.2, Table 2.

[^16]: Gemini image: high-quality image generation. Note: [https://deepmind.google/models/gemini-image/flash/](https://deepmind.google/models/gemini-image/flash/) Accessed: 2026-01-26 Cited by: Table 1, §5.2, Table 2.

[^17]: T2i-r1: reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703. Cited by: Table 4, Table 5.

[^18]: DraCo: draft as cot for text-to-image preview and rare concept generation. arXiv preprint arXiv:2512.05112. Cited by: Table 1, §5.2.

[^19]: GenAgent: scaling text-to-image generation via agentic multimodal reasoning. External Links: 2601.18543, [Link](https://arxiv.org/abs/2601.18543) Cited by: Table 4, Table 5, §1, §5.2, Table 2.

[^20]: Think-then-generate: reasoning-aware text-to-image diffusion with llm encoders. arXiv preprint arXiv:2601.10332. Cited by: §1.

[^21]: FLUX.1 kontext: flow matching for in-context image generation and editing in latent space. External Links: 2506.15742, [Link](https://arxiv.org/abs/2506.15742) Cited by: Table 1, §5.2.

[^22]: FLUX. Note: [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux) Cited by: Table 4, Table 5, Table 1, Table 1, §5.2, Table 2, Table 2.

[^23]: IA-t2i: internet-augmented text-to-image generation. arXiv preprint arXiv:2505.15779. Cited by: §1, §2.1.

[^24]: MCCD: multi-agent collaboration-based compositional diffusion for complex text-to-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13263–13272. Cited by: §2.1.

[^25]: Search-o1: agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366. Cited by: §1.

[^26]: Phybench: a physical commonsense benchmark for evaluating text-to-image models. arXiv preprint arXiv:2406.11802. Cited by: §2.3.

[^27]: Wise: a world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265. Cited by: Table 8, §1, §2.3.

[^28]: GPT-image-1: models and capabilities for image generation. Note: [https://platform.openai.com/docs/models/gpt-image-1](https://platform.openai.com/docs/models/gpt-image-1) Accessed: 2026-01-29 Cited by: §1, Table 1, §5.2, Table 2.

[^29]: GPT-image-1.5: enhanced visual reasoning and creative generation. Note: [https://platform.openai.com/docs/models/gpt-image-1.5](https://platform.openai.com/docs/models/gpt-image-1.5) Accessed: 2026-01-29 Cited by: Table 1, §5.2.

[^30]: Sdxl: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952. Cited by: §1, Table 1, §5.2, Table 2.

[^31]: High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695. Cited by: §1.

[^32]: World-to-image: grounding text-to-image generation with agent-driven world knowledge. arXiv preprint arXiv:2510.04201. Cited by: §1, §2.1.

[^33]: Stable diffusion 3.5 large. Note: [https://huggingface.co/stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) Cited by: Table 1, §5.2, Table 2.

[^34]: Stable diffusion 3.5 medium. Note: [https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/) Cited by: Table 1, §5.2, Table 2.

[^35]: Thinking with images for multimodal reasoning: foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918. Cited by: §2.1.

[^36]: T2i-reasonbench: benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472. Cited by: Table 8.

[^37]: Chameleon: mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818. Cited by: §2.2.

[^38]: Neural discrete representation learning. Advances in neural information processing systems 30. Cited by: §2.2.

[^39]: CREA: a collaborative multi-agent framework for creative image editing and generation. arXiv preprint arXiv:2504.05306. Cited by: §2.1.

[^40]: Emu3: next-token prediction is all you need. arXiv preprint arXiv:2409.18869. Cited by: §2.2.

[^41]: Qwen-image technical report. arXiv preprint arXiv:2508.02324. Cited by: Table 4, Table 5, Table 1, §5.2, Table 2.

[^42]: OmniGen2: exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871. Cited by: §2.2.

[^43]: Promptsculptor: multi-agent based text-to-image prompt optimization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 774–786. Cited by: Table 4, Table 5, §1, §2.1.

[^44]: Show-o: one single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528. Cited by: §2.2.

[^45]: Unified multimodal model as auto-encoder. arXiv preprint arXiv:2509.09666. Cited by: §2.1.

[^46]: Echo-4o: harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987. Cited by: §A.2, Table 8, §1, Table 1, §5.2.

[^47]: Imgedit: a unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275. Cited by: §1.

[^48]: Auto-rag: autonomous retrieval-augmented generation for large language models. arXiv preprint arXiv:2411.19443. Cited by: §1.

[^49]: Mathverse: does your multi-modal llm truly see the diagrams in visual math problems?. In European Conference on Computer Vision, pp. 169–186. Cited by: item 4.

[^50]: Unified multimodal understanding and generation models: advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567. Cited by: §2.1.

[^51]: Envisioning beyond the pixels: benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826. Cited by: Table 8, §1, §2.3.

[^52]: Transfusion: predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039. Cited by: §2.2.

[^53]: AgentStory: a multi-agent system for story visualization with multi-subject consistent text-to-image generation. In Proceedings of the 2025 International Conference on Multimedia Retrieval, pp. 1894–1902. Cited by: §2.1.

[^54]: From reflection to perfection: scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15329–15339. Cited by: Table 4.