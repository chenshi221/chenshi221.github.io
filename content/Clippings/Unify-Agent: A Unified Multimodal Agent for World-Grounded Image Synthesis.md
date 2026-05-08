---
title: "Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis"
source: "https://arxiv.org/abs/2603.29620"
published: "2026"
created: "2026-05-07"
description: null
tags: ["clippings"]
arxiv: "2603.29620"
url: "https://arxiv.org/abs/2603.29620"
---

# Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis

Shawn Chen1,2,*, Quanxin Shou4,*, Hangting Chen2,†, Yucheng Zhou2,†, Kaituo Feng3,†, Wenbo Hu1,†, Yi-Fan Zhang, Yunlong Lin3, Wenxuan Huang3, Mingyang Song3, Dasen Dai3, Bolin Jiang4, Manyuan Zhang3, Shi-Xue Zhang2, Zhengkai Jiang2, Lucas Wang2, Zhao Zhong2, Yu Cheng3, Nanyun Peng1

1University of California, Los Angeles, 2Tencent Hunyuan   
3The Chinese University of Hong Kong, 4The Hong Kong University of Science and Technology

Unified multimodal models provide a natural and promising architecture for understanding diverse and complex real-world knowledge while generating high-quality images. However, they still rely primarily on frozen parametric knowledge, which makes them struggle with real-world image generation involving long-tail and knowledge-intensive concepts. Inspired by the broad success of agents on real-world tasks, we explore agentic modeling to address this limitation. Specifically, we present Unify-Agent, a unified multimodal agent for world-grounded image synthesis, which reframes image generation as an agentic pipeline consisting of prompt understanding, multimodal evidence searching, grounded recaptioning, and final synthesis. To train our model, we construct a tailored multimodal data pipeline and curate 143K high-quality agent trajectories for world-grounded image synthesis, enabling effective supervision over the full agentic generation process. We further introduce FactIP, a benchmark covering 12 categories of culturally significant and long-tail factual concepts that explicitly requires external knowledge grounding. Extensive experiments show that our proposed Unify-Agent substantially improves over its base unified model across diverse benchmarks and real world generation tasks, while approaching the world knowledge capabilities of the strongest closed-source models. As an early exploration of agent-based modeling for world-grounded image synthesis, our work highlights the value of tightly coupling reasoning, searching, and generation for reliable open-world agentic image synthesis.

GitHub: https://github.com/shawn0728/Unify-Agent

# 1 Introduction

Recent advances in Text-to-Image (T2I) generation has significantly improved visual realism, controllability, and stylistic diversity (Esser et al., 2024; Labs, 2025; Rombach et al., 2022; Sun et al., 2024). Despite these developments, visual quality alone remains insufficient for many practical applications. A growing challenge is whether such models can faithfully depict entities and concepts grounded in the real world, including real people, cultural symbols, rare intellectual properties (IPs), historical scenes, scientific phenomena, and other long-tail concepts. In such cases, success requires more than visually plausible outputs, it also requires factual and visual fidelity to the intended entity. Unified multimodal models (UMMs) offer a promising approach to this challenge by unifying visual understanding and image generation within a shared architecture, allowing world knowledge and multimodal reasoning to inform image synthesis (Deng et al., 2025a; Team, 2024a; Wu et al., 2025b; Xie et al., 2024; Zhou et al., 2024). Existing models are limited to parametric knowledge, which is often insufficient to recover the correct appearance and identity-defining visual cues of the target. This limitation reflects a fundamental constraint: the world knowledge is embedded in fixed model parameters and cannot be updated at inference time. Therefore, factual image synthesis may require external world knowledge at inference time rather than sole reliance on parametric memory, as the key challenge is not rendering quality but knowing the target’s correct appearance and identity-defining visual cues.

We argue that addressing this bottleneck requires moving beyond closed-book generation toward open-book, agentic generation, where models can access external world knowledge at inference time instead of relying solely on parametric memory. Yet existing agentic text-to-image systems fall short of this vision. Most are brittle, multi-stage pipelines that loosely connect an LLM planner, retrieval tools, and a standalone image generator (Chen et al., 2025a; He et al.,

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/59001e0df0185175dd296646f238c2d151555011e31f35e7b8f2b322ee42d70a.jpg]]  
Figure 1 | High-quality samples from our Unify-Agent, highlighting its excellence in unified multi-image generation and agentic search enhanced world knowledge integration. It delivers strong cross-image consistency, broad stylistic versatility, and more faithful, knowledge-grounded visual generation across diverse concepts and scenarios.

2026; Son et al., 2025; Wang et al., 2024). Consequently, evidence acquisition, multimodal reasoning, and visual synthesis remain separate stages rather than parts of a unified generative process, making effective evidence integration difficult. Retrieved text can enrich semantic context, but it rarely specifies the fine-grained visual attributes needed for faithful generation. Reference images, although visually informative, often include irrelevant background elements and may conflict with the user-specified composition. Therefore, the key challenge is not simply retrieving external world knowledge. It is turning that knowledge into visual guidance that can both preserve fidelity to the source knowledge and remain aligned with user intent. This capability remains missing in current agentic image generation systems.

To this end, we present Unify-Agent, the first end-to-end unified multimodal agent for world-grounded image synthesis, to the best of our knowledge. Unify-Agent integrates four tightly coupled capabilities within a unified multimodal model: (1) THINK, for prompt understanding and information-gap detection; (2) RESEARCH, for actively acquiring relevant textual and visual knowledge from external sources; (3) RECAPTION, for converting retrieved knowledge into structured textual guidance; and (4) GENERATE, for final image synthesis. Rather than passing raw retrieved content to the generator, Unify-Agent uses recaptioning as an intelligent filtering mechanism to transform external knowledge into structured textual constraints, explicitly disentangling identity-preserving cues from scene-compositional requirements.

To realize this vision in practice, we build the full agentic workflow on a unified model instead of a decoupled pipeline. This architectural choice is motivated by a key insight: generative priors may offer a promising way to enhance multimodal understanding. In decoupled systems, language-based reasoning often overlooks fine-grained visual attributes, limiting the interpretation of retrieved visual references. In contrast, Unify-Agent leverages recaptioning as a natural bridge between understanding and generation. The Vision Transformer (ViT) captures high-level semantics, including global context and entity identity, while the Variational Autoencoder (VAE), although originally designed for image synthesis, provides low-level perceptual latents that retain details such as texture, material, and structural geometry. Together, these components enable Unify-Agent to interpret retrieved visual references with higher fidelity and convert them into precise textual specifications for generation. To evaluate this capability, we introduce FactIP, a comprehensive benchmark of 2,462 curated prompts centered on rare identities and culturally significant long-tail concepts. Experimental results show that Unify-Agent significantly improves factual visual synthesis. On FactIP, it attains an overall score of 73.2, surpassing its base model (Bagel) by more than 22 points and outperforming strong generation-only baselines such as FLUX.1-dev and SD-3.5-large. Moreover, Unify-Agent achieves superior performance among open-source unified models on several widely used world-knowledge benchmarks, including WiSE, KiTTEN, and T2I-FactualBench, while also providing a promising direction for enhancing world knowledge in unified models.

In summary, our main contributions are as follows:

$\diamond$ A Novel Agentic Paradigm. We introduce Unify-Agent, the first end-to-end unified multimodal agent that reformulates text-to-image generation from a passive mapping into an active, inference-time sequential decision process (Think, Research, Recaption, Generate).   
$\diamond$ Architectural Insights on UMMs. We reveal that unifying understanding and generation creates a mutually reinforcing synergy. Specifically, we demonstrate that the joint availability of low-level generative latents (VAE) and high-level semantic tokens (ViT) enables vastly superior multimodal reasoning during evidence recaptioning.   
$\diamond$ A Comprehensive Benchmark. We present FactIP, a highly curated benchmark designed specifically to evaluate the identity consistency and factual faithfulness of world-grounded generation for rare and long-tail concepts.   
$\diamond$ Superior Performance. Unify-Agent sets superior records among open-source unified models across several established factual benchmarks (FactIP, WiSE, KiTTEN, and T2I-FactualBench), seamlessly blending real-world knowledge with creative visual synthesis, and demonstrates world knowledge capabilities approaching those of leading commercial models.

# 2 Related Works

# 2.1 Unified multimodal models

Recent unified multimodal models (UMMs) (Deng et al., 2025a; Huang et al., 2025; Team, 2024a, 2025b; Zhou et al., 2024) aim to jointly support visual understanding and image generation within a shared backbone. Architectures like the Janus family (Chen et al., 2025d; Wu et al., 2025b) and Show-o (Xie et al., 2024) have demonstrated that semantic comprehension and continuous visual synthesis can be effectively harmonized. Despite this, current UMMs remain strictly closed-book systems. Because they rely exclusively on static parametric memory internalized during pre-training,

they frequently hallucinate or suffer from identity drift when prompted with rare, long-tail, or world-dependent entities (Chen et al., 2024, 2026; Huang et al., 2024b). This bottleneck highlights the need to shift from purely parametric generation toward an open-book approach, where the model actively acquires external evidence before synthesis.

# 2.2 Agentic workflows

As reasoning in large language models (Chen et al., 2025b,c; Feng et al., 2025b; Guo et al., 2025) is increasingly evolving into agentic reasoning (Fan et al., 2026; Huang et al., 2026; Shou et al., 2026; Yao et al., 2022; Zeng et al., 2026), the focus is shifting from isolated deliberation to action-oriented planning with tools and environmental interaction. In line with this transition, recent efforts in text-to-image generation (Chen et al., 2025a; Son et al., 2025; Wang et al., 2024) have attempted to incorporate similar agentic pipelines. However, these approaches predominantly adopt a fragile multi-API stitching paradigm (He et al., 2026; Shen et al., 2023; Wu et al., 2023; Yang et al., 2023a,b). They loosely chain together frozen text-only planners, external search tools, and standalone image generators (Feng et al., 2025a; Shen et al., 2023; Wu et al., 2023), failing to realize a true end-to-end UMM agentic framework. This shallow integration inevitably introduces cascading errors and rigidly separates multimodal reasoning from visual synthesis. In contrast, our Unify-Agent pioneers an end-to-end unified agentic framework. By natively consolidating cognitive gap detection, cross-modal evidence grounding, and generation planning within a single cohesive architecture, we eliminate the reliance on brittle API pipelines and achieve genuine reasoning-driven generation.

# 2.3 World knowledge and factual evaluation.

Conventional text-to-image benchmarks predominantly emphasize aesthetic quality and generic prompt alignment (Huang et al., 2023; Lee et al., 2023), largely overlooking the factual correctness of generated content. To address this, recent efforts have pivoted towards knowledge-oriented evaluation. Notably, T2I-FactualBench (Huang et al., 2024b) provides a comprehensive diagnosis of generative models, systematically exposing their struggles with knowledge-intensive concepts across multiple domains. This severe deficiency in explicit world knowledge is further echoed by related benchmarks such as KITTEN (Huang et al., 2024a) and WISE (Niu et al., 2025). Nevertheless, existing evaluations remain fundamentally diagnostic—they quantify the knowledge gap but offer no structural remedy. Motivated by these insights, we not only construct a targeted evaluation for rare identities and long-tail concepts, but also propose an evidence-grounded agentic framework that actively bridges this gap during synthesis.

# 3 Preliminary

# 3.1 Unified Multimodal Model: Bagel

We build our Unify-Agent upon Bagel (Deng et al., 2025b), leveraging its native capabilities to develop a unified and interleaved reasoning framework for world-grounded image synthesis. At its core, Bagel employs a Mixture-of-Transformers (MoT) architecture, integrating a ViT encoder (Tschannen et al., 2025) to process multimodal inputs. This architectural design gracefully unifies visual understanding and continuous image generation within a single foundation model. Specifically, the model disentangles these two core capabilities through dedicated experts:

Multimodal Understanding. The understanding pathway is formulated as an autoregressive next-token prediction task. Handled by a dedicated understanding expert, the model generates context-aware textual responses via a language modeling head. Conditioned on the multimodal input context $C$ , the training objective minimizes the negative log-likelihood:

$$
\mathcal {L} _ {\text {t e x t}} = - \sum_ {t = 1} ^ {T} \log p _ {\theta} \left(x _ {t} \mid x _ {<   t}, C\right), \tag {1}
$$

where $x _ { t }$ is the target text token, $x _ { < t }$ denotes the preceding token sequence, and $C$ encapsulates the provided multimodal context.

Multimodal Generation. Conversely, the generation pathway is designed to synthesize high-fidelity, semantically aligned images. Handled by a generation expert, this process is formulated as a rectified flow (Liu et al., 2022) operating within the latent space of a continuous VAE (Labs et al., 2025). Conditioned on the same multimodal context $C$ , the

model learns a time-conditioned velocity field $u _ { \theta }$ by minimizing the latent flow-matching objective:

$$
\mathcal {L} _ {\text {i m a g e}} = \mathbb {E} _ {t \sim \mathcal {U} (0, 1), z _ {t}} \| u _ {\theta} \left(z _ {t}, t; C\right) - u ^ {\star} \left(z _ {t}, t\right) \| _ {2} ^ {2}, \tag {2}
$$

where $t \sim \mathcal { U } ( 0 , 1 )$ is the continuous timestep, $z _ { t }$ represents the latent state at time $t$ , $\boldsymbol { u } ^ { \star } ( z _ { t } , t )$ is the target vector field, and $u _ { \theta } ( z _ { t } , t ; C )$ is the velocity field predicted by the generation expert.

# 3.2 Motivating Evidence for World-Grounded Synthesis

A central challenge in world-grounded image synthesis is that failures on out-of-distribution concepts, especially rare intellectual properties (IPs), do not primarily stem from insufficient visual fidelity, but from missing world knowledge. When prompted with a long-tail character, scene, or object, a base T2I model often does not know what the concept should look like, which attributes are identity-defining, or how those attributes should be instantiated under compositional instructions. This observation suggests that improving factual generation for rare IPs requires augmenting the model with external knowledge at inference time. Intuitively, textual knowledge injection can supplement semantic definitions, background context, and key attributes, while visual knowledge injection can provide direct appearance anchors that stabilize identity and concept realization.

To validate this intuition, we conduct a preliminary training-free study on 200 examples sampled from three major categories in FactIP Benchmark: scene, character, and object. Using Bagel as the base model, we compare four inference settings: (1) direct prompting with Bagel only, (2) Bagel with text injection, where IP-related background information is appended to the original prompt, (3) Bagel with visual injection, where two ground-truth reference images of the target IP are provided, and (4) Bagel with text $^ +$ visual injection, where both sources of external knowledge are injected simultaneously. As shown in Figure 2, both textual and visual knowledge consistently improve over the prompt-only baseline, confirming that external knowledge is beneficial for factual synthesis. More importantly, visual injection yields substantially larger gains. These results further suggest that enabling a model to proactively search for external knowledge during inference can directly improve its reasoning and generation for rare IP concepts.

At the same time, Figure 2 also reveals the limitation of naive knowledge injection. Although combining text and visual inputs remains beneficial overall, its improvement is slightly weaker than visual injection alone, suggesting

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/3ab26fda335931fc926e49409fe28a93c09c9b5c130e36427b554cad1ab07ad2.jpg]]  
Figure 2 | Absolute performance improvement over promptonly Bagel on rare-IP factual generation under different knowledge injection strategies. We compare text injection, visual injection, and joint text+visual injection across three FactIP categories: Scene, Character, and Object. The purple hatched bars denote the overall average gain across all evaluated samples.

that simply concatenating retrieved text with prompt instructions does not optimally exploit multimodal evidence. Raw text injection may introduce redundant or weakly visual information, which can interfere with instruction following under complex prompts; conversely, raw visual injection, while effective at anchoring identity, may over-constrain the generation process and reduce flexibility in attribute manipulation or compositional reasoning. These findings motivate our next step: rather than directly injecting external knowledge in its raw textual or visual form, we seek a better interface that can transform multimodal evidence into a representation more suitable for generative modeling. This observation directly paves the way for our recaption paradigm, which distills retrieved textual and visual knowledge into a unified, structured, and generation-oriented description for world-grounded synthesis.

# 3.3 Problem Formulation

Standard T2I generation models the conditional distribution $p _ { \theta } ( y \mid x )$ , where an image $y \in \mathcal { V }$ is synthesized from a user prompt $x \in \mathcal { X }$ relying strictly on the parametric memory $\theta$ . However, for world-grounded synthesis involving rare entities, estimating $p _ { \theta } ( y \mid x )$ directly is intractable due to knowledge deficits in $\theta$ . Motivated by the empirical

findings in Section 3.2, we recognize that naively expanding the condition to $p _ { \theta } ( y \mid x , K _ { \mathrm { t e x t } } , K _ { \mathrm { v i s } } )$ , where $\kappa$ represents raw retrieved external knowledge, leads to suboptimal alignment. Raw multimodal evidence introduces formatting noise and conflicting constraints that disrupt the continuous generative process.

Consequently, we formulate world-grounded image synthesis not as a single-step mapping, but as an interleaved generative trajectory over an augmented state space. We introduce four intermediate variables to bridge the gap between prompt understanding and visual synthesis: the cognitive gap assessment $g$ , the textual evidence trace $\tau _ { t }$ , the visual evidence trace $\tau _ { v }$ , and the evidence-grounded recaption $c$ . The joint distribution thus defines the holistic generative process:

$$
p _ {\theta} (y, c, \tau_ {t}, \tau_ {v}, g \mid x) = \underbrace {p _ {\theta} (g \mid x)} _ {\text {G a p D e t e c t i o n}} \cdot \underbrace {p _ {\theta} \left(\tau_ {t} , \tau_ {v} \mid x , g\right)} _ {\text {E v i d e n c e A c q u i s i t i o n}} \cdot \underbrace {p _ {\theta} (c \mid x , g , \tau_ {t} , \tau_ {v})} _ {\text {E v i d e n c e - G r o u n d e d R e c a p t i o n i n g}} \cdot \underbrace {p _ {\theta} (y \mid c , \tau_ {v})} _ {\text {V i s u a l S y n t h e s i s}} \tag {3}
$$

Optimizing Equation 3 requires unifying discrete sequence modeling (for $g , \tau _ { t } , \tau _ { v }$ , and $c$ ) and continuous latent modeling (for $y$ ) within a shared parameter space $\theta$ . Under this unified paradigm, the probabilistic decomposition rigorously defines the four distinct cognitive phases of Unify-Agent:

1. Cognitive Gap Detection $( p _ { \theta } ( g \mid x ) )$ : The model first evaluates the prompt $x$ against its internal parametric memory to identify missing visually-critical attributes, formulating an internal reasoning trace $g$ to decide whether external world knowledge is required.   
2. Evidence Acquisition $( p _ { \theta } ( \tau _ { t } , \tau _ { v } \mid x , g ) )$ : Conditioned on the identified knowledge gap $g$ , the agent proactively interacts with external environments to search and reason over textual evidence $\left( \tau _ { t } \right)$ , and subsequently acquires and reasons over visual evidence $( \tau _ { v } )$ .   
3. Evidence-Grounded Recaptioning $( p _ { \theta } ( c \mid x , g , \tau _ { t } , \tau _ { v } ) )$ : To resolve the modality conflict of raw injection, the model consolidates gathered evidence and the original instruction into a highly structured, generation-oriented recaption $c$ .   
4. Visual Synthesis $( p _ { \theta } ( y \mid c , \tau _ { v } ) )$ : The final image generation is conditioned exclusively on the recaption $c$ and the visual identity anchors $\tau _ { v }$ . By formally enforcing conditional independence from the noisy reasoning history $x , g$ , and $\tau _ { t }$ , this step ensures that the generative prior is strictly guided by refined constraints.

# 4 Data Pipeline

To train a unified multimodal agent for world-grounded image synthesis, we require data that captures not only the final user instruction and output, but also the intermediate process of multimodal reasoning, evidence searching, and evidence aggregation. Meanwhile, to evaluate whether the model can generate images grounded in factual and long-tail world knowledge, we construct a dedicated benchmark. As illustrated in Figure 3, our overall data pipeline therefore comprises two complementary components: (1) training data construction for agent supervised fine-tuning, and (2) a curated FactIP evaluation benchmark. In the following sections, we provide a detailed illustration of the data construction pipeline and strategies.

# 4.1 Training Data Construction

To train Unify-Agent to execute the full Think–Research–Generate pipeline, we construct supervision that covers not only the final generation-oriented recaption, but also the intermediate evidence-acquisition process. Accordingly, each training sample is represented as

$$
\mathcal {D} _ {\mathrm {S F T}} = \left\{\left(x, \tau_ {t}, \tau_ {v}, c\right) \right\}, \tag {4}
$$

where $x$ denotes the original user prompt, $\tau _ { t }$ denotes the textual research trace, $\tau _ { v }$ denotes the visual research trace, and $c$ denotes the final evidence-grounded recaption. Under this formulation, $\tau _ { t }$ and $\tau _ { v }$ supervise how the agent performs external evidence acquisition, while $c$ supervises how the acquired evidence is transformed into a synthesis-ready specification.

Compared with standard instruction tuning for multimodal models, our data differs in two important respects. First, the target output is not a generic assistant response or image caption, but a structured recaption designed to preserve identity-critical visual attributes while remaining faithful to user-specified scene and style constraints. Second, the supervision is process-aware: instead of learning only the final output, the model is trained on explicit multimodal research traces, enabling it to internalize not only what evidence is useful, but also when and how such evidence should be acquired before generation.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/383561a6a653afd2ea21597379350105f61cc434553e319de3acbc4b5c59620e.jpg]]  
Figure 3 | Overview of our data pipeline. Starting from long-tail IP collection, we construct user instructions and Ground Truth images, build multimodal research trajectories with textual and visual evidence, and finally perform evidence-grounded recaption annotation to obtain high-quality training samples. The resulting dataset supports both SFT trajectory learning and the FactIP benchmark, which evaluates generation quality in terms of clarity, content, aesthetics, and relevance.

We construct the training corpus in three stages: (1) task source curation and prompt collection, (2) multimodal research trace construction, and (3) evidence-grounded recaption annotation with generation-based verification.

# 4.1.1 Task Source and Prompt Collection

Task Source. We begin by constructing a large-scale pool of knowledge-intensive IPs that are likely to expose world-knowledge deficiency in image generation models. To this end, we organize a large group of annotators to broadly collect relatively long-tail and less frequently represented IPs from the web, rather than relying on only popular or highly recurrent entities. The collected IPs are categorized into 12 domains, including Celebrity, Animation, Game, Comic, Mythology, Mascot, Animal, Food, Art, Toy, Landmark, and Festival. For each concept, we query BabelNet (Navigli and Ponzetto, 2010) and Wikipedia to retrieve relevant webpage information, and annotators further search for two representative seed images as the ground-truth visual references. We then use GPT-4o (Team, 2024b) to summarize each IP into structured metadata, including its identity-defining attributes and relevant factual descriptions. To ensure data reliability, we perform an additional round of manual verification to remove samples with incorrect metadata, low-quality seed images (e.g., severe watermarks or blurry content), or concepts that are overly common and do not sufficiently challenge factual grounding. After this filtering process, we obtain a final dataset containing 456K examples.

Prompt Collection. Based on the curated IP pool, we construct user instructions designed to evaluate whether a model can faithfully preserve grounded identity while performing creative image generation. Rather than using arbitrary prompts, we generate prompts that retain the original generation intent, such as scene composition, clothing or object specification, atmosphere, and photographic or cinematic style, so that the task requires both factual consistency and compositional flexibility. In particular, for different categories, we use Gemini 3 Pro (Team, 2025a) to create diverse prompts tailored to their domain characteristics. These prompts cover a spectrum of difficulty, ranging from direct rendering of a target identity or concept to more challenging cases where the model must preserve defining attributes while adapting pose, costume, scene, lighting, or aesthetic style. Such a construction improves diversity across tasks and helps avoid overfitting to simple retrieval-like behavior, encouraging the model to handle a broader range of grounded generation scenarios.

# 4.1.2 Multimodal Research Trace Construction

To instantiate the evidence acquisition process in Eq. 3, we construct supervised multimodal research traces that approximate the desired inference-time behavior of the agent. For each prompt $x$ collected from Section 4.1.1, we use a strong teacher agent to externalize the intermediate evidence-seeking process and serialize it into a structured trajectory. Concretely, we employ Claude Opus 4.6 as the trajectory-construction model to perform multi-step research over external tools and to produce high-quality agent traces. This choice is motivated by its strong capability on long-horizon agentic tasks, tool use, and complex research-style workflows, making it well-suited for synthesizing supervision signals that reflect realistic evidence acquisition behavior.

Given a prompt $x$ , trajectory construction is formulated as a sequential evidence acquisition process with two stages: textual research followed by visual research. This ordering is deliberate. Textual evidence first resolves semantic ambiguity and establishes high-level grounding, which in turn enables more precise and context-aware visual search.

Textual research trace. The textual research trace $\tau _ { t }$ captures how the agent acquires semantic grounding from external knowledge sources. Conditioned on the prompt $x$ and the inferred cognitive gap $g$ , the teacher agent first formulates a textual query $q _ { t }$ :

$$
q _ {t} \sim p _ {\theta} \left(q _ {t} \mid x, g\right), \tag {5}
$$

which is issued to an external text searching system to obtain textual evidence

$$
E _ {t} = \operatorname {R e t r i e v e} _ {\text {t e x t}} \left(q _ {t}\right). \tag {6}
$$

The role of $E _ { t }$ is not to directly replace the user prompt, but to provide a compact semantic scaffold for downstream reasoning. In practice, $E _ { t }$ typically contains information such as identity disambiguation, categorical background, role-specific context, and other descriptors that are useful for constraining subsequent visual search. We record both the issued query $q _ { t }$ and the resulting evidence summary $E _ { t }$ as part of the supervised trace, thereby teaching the model not only what knowledge to use, but also how to acquire it.

Visual research trace. After establishing textual grounding, the agent proceeds to visual evidence acquisition. Conditioned on $( x , g , \tau _ { t } )$ , the teacher agent generates a visual query $q _ { v }$ :

$$
q _ {v} \sim p _ {\theta} \left(q _ {v} \mid x, g, \tau_ {t}\right), \tag {7}
$$

where $q _ { v }$ is designed to retrieve images that are both identity-preserving and context-compatible. Compared with naive retrieval based solely on entity names, this query incorporates both prompt-specific constraints and the semantic grounding provided by $\tau _ { t }$ , thereby enabling more precise and scenario-aware visual search.

The initial candidate set is retrieved as

$$
\tilde {E} _ {v} = \operatorname {R e t r i e v e} _ {\text {i m a g e}} (q _ {v}) = \{v _ {1}, \dots , v _ {n} \}. \tag {8}
$$

To identify high-quality visual references, we further employ Gemini 3 Flash (Team, 2025a) as a lightweight visual evaluator to score each candidate along four dimensions: identity consistency, subject salience, image clarity, and watermark cleanliness. Concretely, a preferred image should faithfully match the target identity or concept, present the target as the dominant subject, maintain sufficient visual quality, and avoid heavy watermarks or text-dominant compositions. The overall score for each candidate is computed as

$$
s \left(v _ {i}\right) = \sum_ {k = 1} ^ {4} \lambda_ {k} s _ {k} \left(v _ {i} \mid x, E _ {t}\right). \tag {9}
$$

Based on these scores, the candidate pool is ranked and the top two images are selected as the final visual evidence:

$$
E _ {v} = \arg \operatorname {t o p 2} s \left(v _ {i}\right). \tag {10}
$$

$$
v _ {i} \in \widetilde {E} _ {v}
$$

The resulting images serve as the primary visual anchors for subsequent recaptioning and synthesis. By supervising both the query formulation process and the final selected references, the model learns visual grounding as a structured evidence acquisition procedure, rather than simply consuming pre-selected images.

Trace representation and supervision. The final multimodal research trace is represented as

$$
\left(\tau_ {t}, \tau_ {v}\right) = \left(q _ {t}, E _ {t}, q _ {v}, E _ {v}\right), \tag {11}
$$

where $q _ { t }$ and $q _ { v }$ denote the textual and visual queries, $E _ { t }$ denotes the retrieved textual evidence, and $E _ { v }$ denotes the selected visual evidence. This representation makes the evidence acquisition process explicit and directly aligns with the factorized term $p _ { \theta } ( \tau _ { t } , \tau _ { v } \mid x , g )$ in Eq. 3.

By supervising these intermediate trajectories, we train the model to treat world grounding as a structured, multi-step process of external evidence construction. Rather than collapsing grounding into a single retrieval-augmented prompt, the model learns to actively formulate queries, gather heterogeneous evidence, and organize that evidence into a form suitable for downstream recaptioning and image synthesis.

# 4.1.3 Evidence-Grounded Recaption Annotation

As discussed in Section 3.2, directly injecting externally retrieved evidence into the original user instruction is not an optimal interface for world-grounded image synthesis. Although both textual and visual evidence can improve factual generation, naive multimodal injection remains suboptimal: raw text often introduces redundant or weakly visual information that interferes with instruction following, while raw visual inputs alone may over-constrain the generation process and reduce compositional flexibility. This observation suggests that external evidence should not be consumed in its original form, but instead transformed into a representation that is more compatible with downstream generative modeling.

Motivated by recent advances in prompt rewriting and model-facing specification design for image generation, as exemplified by DALL-E 3 (Betker et al., 2023), PromptEnhancer (Wang et al., 2025), and HunyuanImage 3.0 (Cao et al., 2026), we adopt an evidence-grounded recaption as the core intermediate supervision target. This recaption is not a generic image caption and not a simple paraphrase of the prompt. Instead, it is a structured textual specification intended to control downstream image synthesis. Its role is to consolidate the original instruction with retrieved external evidence into a generation-oriented description that preserves identity-critical attributes while maintaining scene-level and stylistic controllability.

Formally, given the user prompt $x$ and the retrieved multimodal evidence $( E _ { t } , E _ { v } )$ , we construct a grounded recaption

$$
c = \mathcal {C} (x, E _ {t}, E _ {v}), \tag {12}
$$

where $\mathcal { C }$ denotes the recaptioning function. The resulting recaption integrates three complementary sources of information: (1) prompt-level compositional constraints such as scene, pose, clothing, atmosphere, and rendering style; (2) semantic context derived from textual evidence for identity disambiguation and background consistency; and (3) identity-preserving visual cues grounded in the selected reference images. In this way, recaptioning serves as a structured interface that translates heterogeneous external evidence into a unified textual control signal suitable for generation.

To ensure that the annotated recaptions are not only semantically grounded but also operationally useful for image synthesis, we further perform a generation-based validation procedure. Specifically, after obtaining the grounded recaption c, we feed it together with the two selected reference images $E _ { v } = \{ v _ { 1 } , v _ { 2 } \}$ into Nano Banana Pro (Team, 2025a) to synthesize an image

$$
\hat {y} \sim p _ {\phi} (y \mid c, E _ {v}), \tag {13}
$$

where $p _ { \phi }$ denotes the image generator used for data construction. The generated image $\hat { y }$ is then compared against the ground-truth image of the corresponding IP instance using GPT-4o (Team, 2024b) as a multimodal judge, which evaluates whether the synthesized result is sufficiently faithful to the intended identity and visual concept.

Based on the judge outcome, we adopt a reject-sampling (Liu et al., 2024) strategy to improve annotation reliability. If the generated image fails the identity-consistency check, we re-run the recaption-to-generation process and repeat the validation step, for up to five trials in total. If no satisfactory result is obtained after five attempts, we treat the failure as evidence that the underlying trajectory is unreliable—typically due to erroneous retrieval, weak reference quality, or mismatched visual anchors—and discard the entire trajectory from the training set. This process effectively filters out noisy supervision signals that would otherwise teach the model incorrect grounding behavior.

After this recaptioning-and-verification pipeline, we obtain a final set of 143K high-quality trajectory-image pairs for subsequent training. These samples provide supervision not only over the final synthesis target, but also over the

complete evidence-grounded reasoning path that leads to it. As a result, the model is trained to view world-grounded generation as a multi-stage process: acquire external evidence, reorganize it into a synthesis-ready recaption, and generate images that remain faithful to both the retrieved world knowledge and the user’s compositional intent.

# 4.2 FactIP Evaluation Benchmark

# 4.2.1 Benchmark Construction

Our FactIP benchmark is derived from the task pool constructed in Section 4.1.1, but is strictly separated from the training data. Specifically, after collecting long-tail IP candidates and organizing them into user-facing generation tasks, we reserve a disjoint subset for evaluation to ensure that benchmark instances are not used during supervised fine-tuning. Each benchmark sample contains a user instruction, two seed images (used as ground truth), and the necessary metadata associated with the target IP.

We design FactIP to evaluate world-grounded image synthesis under challenging yet realistic settings. In particular, benchmark samples are selected to satisfy four principles: (1) long-tail and knowledge-intensive, such that successful generation requires external or internalized world knowledge beyond common popular concepts; (2) visually grounded, such that each target is associated with a clear visual identity that can be assessed against reference images; (3) difficult for memorization-only models, such that strong performance cannot be achieved by relying solely on pretraining memorization or generic text-to-image priors; and (4) diverse across categories, so that the benchmark covers a broad range of entities and scenarios.

Starting from a larger candidate pool, we first obtain 2,500 samples through manual filtering. We then construct a lightweight test split of 500 samples by further sampling with the same category distribution. During this process, we remove samples with severe information noise, unclear instructions, low-quality reference images, or highly ambiguous identities. This filtering procedure ensures that each retained sample exhibits a clear visual identity and a genuine demand for factual grounding, yielding a high-quality benchmark for evaluating world-grounded generation. Additional benchmark construction details are provided in Appendix D.1.

# 4.2.2 Evaluation Criterion

We evaluate generated images on FactIP using a structured multi-dimensional protocol. In particular, each sample is assessed with the user instruction and the Ground Truth (seed) image as reference, which helps determine both identity consistency and factual faithfulness. Our evaluation considers four dimensions: Clarity, Content, Aesthetics, and Relevance. Among them, Relevance is especially important, as it measures whether the generated result preserves the defining identity cues of the target IP while remaining consistent with the instruction and reference image. For brevity, we defer the full evaluation protocol, scoring details, and weighting scheme to Appendix C.

# 5 Methodology

# 5.1 Unified Fine-Tuning on Multimodal Agent Trajectories

Building upon the unified multimodal formulation of Bagel described in Section 3.1, we further adapt the model to our agentic setting through supervised fine-tuning on interleaved multimodal trajectories from Section 4.1. Each training sample may contain textual reasoning traces, tool-use actions, recaptioning outputs, and image-generation targets, allowing the model to learn world-grounded generation as a unified sequential decision process. To facilitate efficient and unified training, we adopt a sequence packing strategy to concatenate heterogeneous samples into packed sequences, enabling the joint optimization of text-only and image-generation objectives within a single forward pass.

Our fine-tuning objective follows the “think-to-generate” paradigm of Bagel, employing a dual-loss design. Specifically, it combines a language modeling loss $\mathcal { L } _ { \mathrm { t e x t } }$ for multimodal understanding and a latent-space regression loss $\mathcal { L } _ { \mathrm { i m a g e } }$ for image generation:

$$
\mathcal {L} _ {\mathrm {S F T}} = \mathcal {L} _ {\text {t e x t}} + \mathcal {L} _ {\text {i m a g e}} \tag {14}
$$

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/78149f053eba8a6b0aaef5105171191d756af515b45af6a82ccf755ed41961d3.jpg]]  
Figure 4 | Overview of the agentic pipeline of our method. Given an input prompt, our framework first performs prompt understanding and cognitive gap detection to identify missing but visually critical attributes. It then acquires complementary multimodal evidence through textual evidence searching and visual evidence searching. Based on the collected evidence, the model grounds the generation process with two types of constraints: identity-preserving constraints that capture character-specific visual traits, and scene-compositional constraints that specify pose, environment, garment, and overall mood. These grounded constraints are then integrated into an evidence-grounded recaptioning module, which produces a detailed caption for the downstream image generator to synthesize the final image.

Text supervision. For tokens corresponding to reasoning, tool invocation, and recaptioning targets, we apply the autoregressive next-token prediction objective introduced in Section 3.1. In practice, this branch is instantiated as a token-level cross-entropy loss over supervision positions. To better preserve critical structural formats in agent trajectories, we further upweight special tokens associated with reasoning and action boundaries, such as <think>, <tool_call>, and <recaption> tags. This encourages reliable learning of structured reasoning traces and executable output formats.

Image supervision. For samples involving image generation, we retain the latent flow-matching training objective from Section 3.1. Given a target image, we first encode it into a clean latent representation with the frozen VAE, then perturb the latent with Gaussian noise at a sampled timestep, and train the model to predict the corresponding flow velocity field in latent space. This branch enables the model to align generated images with both the input instruction and the retrieved multimodal evidence, while remaining compatible with the unified autoregressive backbone.

Importantly, the two objectives are activated by separate supervision masks within each packed sequence. In addition, we employ a hybrid attention masking strategy to regulate information flow across interleaved reasoning traces, retrieved reference images, recaption tokens, and generation tokens. This design is particularly important for agentic multimodal training, as it preserves the sequential structure of textual reasoning while preventing noisy historical traces from interfering with the final image synthesis stage. Detailed masking rules and visualizations are provided in Appendix B. As a result, text-only trajectory segments contribute only to $\mathcal { L } _ { \mathrm { t e x t } }$ , whereas image-generation segments contribute to both $\mathcal { L } _ { \mathrm { t e x t } }$ and $\mathcal { L } _ { \mathrm { i m a g e } }$ .

# 5.2 Unify-Agent: Inference-Time Multimodal Agentic Pipeline

As illustrated in Figure 4, Unify-Agent formulates world-grounded image synthesis as a sequential inference-time decision process, rather than a single-shot mapping from prompt to image. This design directly instantiates the factorization in Eq. 3: given a user prompt $x$ , the model first determines whether the request contains visually critical knowledge missing from its parametric memory, then actively acquires complementary multimodal evidence, converts the retrieved evidence into generation-relevant constraints, and finally synthesizes the image through an evidence-grounded recaption. By interleaving reasoning, searching, and synthesis within a unified execution policy, the agent is able to compensate for world-knowledge deficiencies while preserving the compositional intent of the original prompt.

Think: Prompt understanding and cognitive gap detection. Unify-Agent begins by interpreting the input prompt $x$ not as a self-sufficient description, but as a partially specified generation request whose visually critical details may be incomplete. To make this explicit, the model first performs structured prompt understanding, decomposing the instruction into semantically meaningful factors such as target identity, scene configuration, stylistic intent, and photographic requirements. Based on this structured representation, the model then estimates whether faithful synthesis depends on knowledge that is absent, ambiguous, or unreliable in its internal parametric memory. This process is captured by the latent gap variable

$$
g \sim p _ {\theta} (g \mid x), \tag {15}
$$

which governs whether external world knowledge should be invoked.

To further characterize the content of this gap, we represent the missing information as a set of knowledge units

$$
\mathcal {M} (x) = \left\{m _ {1}, \dots , m _ {K} \right\}, \tag {16}
$$

where each $m _ { k }$ denotes an identity-critical or visually consequential attribute that is not sufficiently specified by the prompt itself, such as facial structure, hairstyle, signature appearance, object-specific structure, or scene-specific factual details. When $\mathcal { M } ( x ) \neq \emptyset$ , the model enters the subsequent research stage to actively resolve these missing variables; otherwise, it may proceed directly to generation. Crucially, the purpose of this stage is not merely to recognize that an entity is rare or knowledge-intensive, but to determine which missing attributes must be recovered in order to support faithful world-grounded synthesis.

Research: Sequential multimodal evidence acquisition. Conditioned on the inferred gap state $g$ , the agent proceeds to external evidence acquisition, corresponding to the factor

$$
p _ {\theta} \left(\tau_ {t}, \tau_ {v} \mid x, g\right). \tag {17}
$$

Rather than searching for all evidence in a single step, Unify-Agent adopts a sequential multimodal strategy in which textual evidence is acquired before visual evidence. This ordering is intentional: textual search first provides semantic grounding, identity disambiguation, and high-level background knowledge, which then guides a more precise visual search.

Concretely, the agent first formulates a textual query and obtains a textual evidence trace $\tau _ { t }$ , which provides compact semantic context about the target concept. This textual scaffold is then used to refine the subsequent visual search process, enabling the model to formulate a more informed visual query and retrieve a visual evidence trace $\tau _ { v }$ composed of identity-relevant and context-compatible reference images. Compared with naive name-based retrieval, this sequential design allows evidence acquisition to be conditioned jointly on prompt intent and semantic grounding, thereby improving both retrieval precision and downstream grounding quality.

Recaption: Multimodal grounding into executable specifications. A central design choice in Unify-Agent is that retrieved evidence is not passed directly to the image generator in its raw form. As shown in Section 3.2, although naive knowledge injection is helpful, it remains suboptimal: raw text may introduce redundant or weakly visual information, while raw reference images alone may over-constrain the synthesis process. To address this mismatch, the agent first transforms the collected multimodal evidence into a set of grounded constraints that are directly useful for generation.

These constraints take two complementary forms. The first is an identity-preserving constraint, which captures the visual attributes that must remain faithful to the target identity, such as facial structure, appearance cues, or object-defining characteristics. The second is a scene-compositional constraint, which encodes the prompt-specified factors of generation, including pose, environment, garment, mood, composition, and overall presentation. Together, these grounded constraints serve as an abstraction layer that filters noisy raw evidence and reorganizes it into a more generation-compatible representation. The agent then integrates these constraints into an evidence-grounded recaption

$$
c \sim p _ {\theta} (c \mid x, g, \tau_ {t}, \tau_ {v}), \tag {18}
$$

which acts as the final executable specification for synthesis. In this way, recaptioning provides a structured textual interface that unifies the original user intent with retrieved semantic and visual evidence, while preserving both identity fidelity and compositional controllability.

Generate: Evidence-grounded image synthesis. Given the evidence-grounded recaption $c$ and the selected visual anchors $\tau _ { v }$ , the downstream generator finally synthesizes the image according to

$$
y \sim p _ {\theta} (y \mid c, \tau_ {v}). \tag {19}
$$

This formulation reflects our modeling assumption that, once the external evidence has been distilled into a structured recaption and a compact set of visual anchors, the final synthesis process no longer needs to depend directly on the full upstream reasoning history $( x , g , \tau _ { t } )$ . Instead, generation is driven by refined, visually grounded constraints that are specifically tailored for the image generator.

This design brings two advantages. First, it prevents noisy reasoning traces or redundant textual evidence from interfering with the continuous generative process. Second, it preserves the complementary roles of the two conditioning signals: the recaption provides a unified and controllable description of the intended image, while the visual anchors stabilize identity realization and factual appearance. As a result, the generated output remains faithful to both the retrieved world knowledge and the compositional intent of the original prompt.

Overall, Unify-Agent reframes world-grounded image synthesis as an active agentic pipeline of Think, Research, Recaption, and Generate. Rather than treating the user prompt as a complete specification, the model identifies missing knowledge, actively acquires external evidence, reorganizes that evidence into a synthesis-ready representation, and finally produces an image grounded in both world knowledge and user intent. This unified pipeline moves beyond prompt-only generation and enables reliable synthesis of rare, long-tail, and knowledge-intensive concepts.

# 6 Experiments

# 6.1 Experimental Details

Baselines We benchmark our Unify-Agent against three categories of baselines. First, we include leading commercial models such as Seedream-Family(Seedream et al., 2025), Nano Banana-2(Google, 2026), DALLE-3 (Betker et al., 2023), and GPT-Image-1.5(OpenAI, 2025), which serve as upper-bound references for factual text-to-image generation. Second, we consider representative generation-only models, including FLUX.1-dev (Labs, 2025), SD-3.5-large (Esser et al., 2024), Playground-v2.5(Li et al., 2024), Z-Image(Team, 2025c) and Qwen-Image (Wu et al., 2025a), which are widely adopted open-source diffusion alternatives. Third, we evaluate competitive unified MLLMs, such as Janus-Pro-7B (Chen et al., 2025d), Emu3.5(Cui et al., 2025), Echo-4o(Ye et al., 2025) ,Hunyuan-Image-3.0(Cao et al., 2026), and Bagel (Deng et al., 2025a). These represent structurally comparable open-source baselines that natively unify visual understanding and generation. Most of the unified baselines share similar autoregressive or flow-matching training setups, ensuring comparability in architecture. Collectively, this suite provides a comprehensive coverage of both proprietary and open-source models across multiple paradigms.

Evaluation Setup Recently, using multimodal large language models (MLLMs) as automatic judges has become an increasingly common practice for multimodal evaluation, as it provides a scalable and efficient alternative to human annotation while enabling fine-grained assessment of multimodal outputs. Prior work on MLLM-as-a-Judge (Chen et al., 2024) suggests that MLLM-based evaluators can serve as practical proxies for expert judgment in vision-language tasks, particularly in scoring and pairwise comparison. Following this evaluation paradigm, we adopt MLLM-based evaluation on WISE (Niu et al., 2025), T2I-FactualBench (Huang et al., 2024b), and Kitten (Huang et al., 2024a), where ChatGPT-4o (Team, 2024b) is used as the expert evaluator which helps maintain consistency with prior benchmark protocols. For our proposed FactIP Bench, we use Seed2.0 (Seed) as the expert evaluator, considering its stronger multimodal judging capability. Additional details on the evaluation protocol, prompts, and scoring criteria are provided in the Appendix C.

# 6.2 Main Results

FactIP Tab. 1 presents the quantitative results on our FactIP benchmark, which evaluates the capability of models to generate culturally significant concepts and intellectual properties with high factual fidelity. The evaluation is categorized into Character, Object, and Scene, assessing Clarity, Content, Aesthetics, and Relevance. Within the Unified MLLM category, our Unify-Agent achieves the highest Overall score of 73.2, significantly outperforming competitive proprietary and open-weight unified models. Notably, Unify-Agent exhibits exceptional performance in the critical

Table 1 | FactIP benchmark results with category-level weighted averages. Character includes celebrity, animation, game, comic, mythology, and mascot prompts; Object includes animals/plants, food, cultural relic/art, and toy prompts; Scene includes landmark and celebration prompts. denotes the best score and the second-best within each group.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Clarity</td><td colspan="3">Content</td><td colspan="3">Aesthetics</td><td colspan="3">Relevance</td><td rowspan="2">Overall</td></tr><tr><td>Character</td><td>Object</td><td>Scene</td><td>Character</td><td>Object</td><td>Scene</td><td>Character</td><td>Object</td><td>Scene</td><td>Character</td><td>Object</td><td>Scene</td></tr><tr><td colspan="14">Commercial Models</td></tr><tr><td>GPT-Image-1</td><td>94.6</td><td>94.1</td><td>96.2</td><td>83.5</td><td>71.1</td><td>78.4</td><td>89.2</td><td>88.1</td><td>94.1</td><td>61.9</td><td>60.2</td><td>76.4</td><td>69.3</td></tr><tr><td>GPT-Image-1.5</td><td>94.4</td><td>94.2</td><td>95.6</td><td>82.7</td><td>73.2</td><td>77.9</td><td>89.5</td><td>88.1</td><td>93.8</td><td>62.6</td><td>61.5</td><td>77.2</td><td>69.9</td></tr><tr><td>Seeddream-4</td><td>93.9</td><td>83.7</td><td>88.0</td><td>81.0</td><td>92.3</td><td>78.1</td><td>88.2</td><td>81.7</td><td>91.5</td><td>80.5</td><td>90.5</td><td>86.0</td><td>83.0</td></tr><tr><td>Seeddream-4.5</td><td>94.1</td><td>83.1</td><td>87.8</td><td>80.1</td><td>92.2</td><td>76.9</td><td>88.4</td><td>79.7</td><td>90.8</td><td>81.0</td><td>91.3</td><td>83.7</td><td>82.0</td></tr><tr><td>Seeddream-5</td><td>95.0</td><td>86.0</td><td>88.1</td><td>86.7</td><td>94.3</td><td>83.5</td><td>90.3</td><td>87.1</td><td>91.7</td><td>84.3</td><td>91.8</td><td>87.1</td><td>87.3</td></tr><tr><td>Qwen-Image-2.0-Pro</td><td>94.2</td><td>92.7</td><td>90.8</td><td>77.0</td><td>73.6</td><td>76.2</td><td>86.2</td><td>87.5</td><td>90.0</td><td>62.6</td><td>74.0</td><td>79.7</td><td>71.1</td></tr><tr><td>Nano Banana</td><td>94.8</td><td>71.8</td><td>76.7</td><td>44.3</td><td>97.8</td><td>84.0</td><td>76.0</td><td>64.8</td><td>93.7</td><td>74.7</td><td>73.7</td><td>52.4</td><td>56.6</td></tr><tr><td>Nano Banana-Pro</td><td>94.3</td><td>76.9</td><td>86.8</td><td>59.1</td><td>92.6</td><td>69.1</td><td>86.7</td><td>59.9</td><td>91.9</td><td>76.9</td><td>90.3</td><td>72.6</td><td>66.7</td></tr><tr><td>Nano Banana-2</td><td>96.6</td><td>88.2</td><td>90.4</td><td>86.3</td><td>96.0</td><td>86.6</td><td>92.2</td><td>92.2</td><td>95.3</td><td>85.5</td><td>93.9</td><td>89.5</td><td>88.5</td></tr><tr><td colspan="14">Generation Only</td></tr><tr><td>Pixel-Art-XL</td><td>34.6</td><td>16.9</td><td>15.6</td><td>14.7</td><td>5.7</td><td>9.2</td><td>28.2</td><td>14.7</td><td>19.0</td><td>10.8</td><td>3.7</td><td>9.2</td><td>12.1</td></tr><tr><td>FLUX.1-schnell</td><td>85.5</td><td>80.6</td><td>82.1</td><td>35.1</td><td>19.9</td><td>27.6</td><td>68.4</td><td>65.5</td><td>69.7</td><td>13.9</td><td>7.9</td><td>16.9</td><td>24.0</td></tr><tr><td>FLUX.1-dev</td><td>92.5</td><td>89.0</td><td>91.8</td><td>51.9</td><td>40.4</td><td>37.7</td><td>78.4</td><td>75.8</td><td>80.3</td><td>17.0</td><td>8.2</td><td>18.2</td><td>28.9</td></tr><tr><td>FLUX.2-dev</td><td>92.4</td><td>91.7</td><td>90.6</td><td>72.9</td><td>58.6</td><td>66.8</td><td>84.4</td><td>83.8</td><td>88.5</td><td>46.6</td><td>46.3</td><td>61.0</td><td>56.3</td></tr><tr><td>SD-3-medium</td><td>85.1</td><td>81.7</td><td>78.5</td><td>40.9</td><td>22.0</td><td>28.6</td><td>70.6</td><td>64.9</td><td>66.8</td><td>16.2</td><td>6.6</td><td>17.9</td><td>25.7</td></tr><tr><td>SDXL-base-0.9</td><td>82.9</td><td>78.3</td><td>80.2</td><td>40.4</td><td>16.6</td><td>25.3</td><td>74.1</td><td>67.0</td><td>73.2</td><td>17.9</td><td>6.4</td><td>17.1</td><td>26.5</td></tr><tr><td>SD-3.5-large</td><td>85.4</td><td>82.5</td><td>81.1</td><td>44.3</td><td>23.2</td><td>32.1</td><td>75.3</td><td>68.1</td><td>74.5</td><td>18.0</td><td>5.9</td><td>20.2</td><td>27.5</td></tr><tr><td>SD-3.5-medium</td><td>84.3</td><td>73.6</td><td>79.2</td><td>40.9</td><td>17.6</td><td>31.6</td><td>72.5</td><td>54.6</td><td>67.4</td><td>17.7</td><td>7.1</td><td>20.1</td><td>26.5</td></tr><tr><td>Z-Image</td><td>94.3</td><td>92.4</td><td>91.9</td><td>72.5</td><td>62.1</td><td>67.7</td><td>84.9</td><td>83.7</td><td>87.3</td><td>42.8</td><td>44.5</td><td>60.5</td><td>54.2</td></tr><tr><td>Qwen-Image</td><td>94.5</td><td>91.7</td><td>91.7</td><td>72.9</td><td>57.3</td><td>67.9</td><td>84.9</td><td>83.8</td><td>89.4</td><td>45.3</td><td>43.1</td><td>62.6</td><td>55.4</td></tr><tr><td colspan="14">Unified MLLM</td></tr><tr><td>Janus-1.3B</td><td>42.7</td><td>24.9</td><td>29.7</td><td>26.1</td><td>13.3</td><td>19.2</td><td>41.8</td><td>26.5</td><td>37.1</td><td>15.2</td><td>10.3</td><td>21.0</td><td>19.3</td></tr><tr><td>Janus-Pro</td><td>60.0</td><td>40.3</td><td>43.9</td><td>40.1</td><td>25.6</td><td>28.5</td><td>58.2</td><td>46.4</td><td>56.6</td><td>24.3</td><td>21.4</td><td>30.2</td><td>30.3</td></tr><tr><td>Emu3</td><td>82.7</td><td>81.2</td><td>78.6</td><td>48.4</td><td>29.6</td><td>34.8</td><td>73.4</td><td>69.8</td><td>73.7</td><td>19.2</td><td>13.8</td><td>26.1</td><td>30.0</td></tr><tr><td>Echo4o</td><td>89.9</td><td>90.7</td><td>89.7</td><td>65.1</td><td>49.3</td><td>61.0</td><td>80.8</td><td>80.5</td><td>85.0</td><td>36.1</td><td>35.7</td><td>52.0</td><td>47.3</td></tr><tr><td>Hunyuan-Image-3.0</td><td>93.0</td><td>91.1</td><td>90.0</td><td>72.3</td><td>59.7</td><td>70.4</td><td>84.6</td><td>83.0</td><td>87.7</td><td>40.6</td><td>46.8</td><td>63.3</td><td>53.4</td></tr><tr><td>Emu3.5</td><td>94.8</td><td>92.7</td><td>94.3</td><td>64.0</td><td>54.5</td><td>69.3</td><td>85.2</td><td>84.5</td><td>90.6</td><td>49.7</td><td>44.4</td><td>60.8</td><td>57.2</td></tr><tr><td>Bagel</td><td>91.4</td><td>91.8</td><td>90.6</td><td>68.5</td><td>59.1</td><td>65.0</td><td>81.6</td><td>83.3</td><td>87.2</td><td>39.9</td><td>44.0</td><td>50.7</td><td>50.9</td></tr><tr><td>Bagel-CoT</td><td>90.7</td><td>90.8</td><td>90.5</td><td>66.4</td><td>53.7</td><td>62.3</td><td>80.3</td><td>79.3</td><td>85.0</td><td>36.2</td><td>34.7</td><td>47.4</td><td>47.0</td></tr><tr><td>Unify-Agent (Ours)</td><td>92.4</td><td>91.5</td><td>90.7</td><td>75.8</td><td>76.1</td><td>73.6</td><td>83.3</td><td>86.0</td><td>86.4</td><td>67.3</td><td>71.8</td><td>78.2</td><td>73.2</td></tr></table>

Relevance dimension, securing the top scores across Character (67.3), Object (71.8), and Scene (78.2) sub-categories. This demonstrates its superior ability to preserve faithful IP identities and factual visual traits. Furthermore, Unify-Agent vastly surpasses traditional generation-only architectures like FLUX.1-dev (28.9) and SD-3.5-large (27.5), validating the effectiveness of integrating explicit multimodal reasoning and external knowledge for complex factual generation.

WiSE Tab. 2 reports results on the WiSE benchmark, which evaluates text-to-image generation performance across different knowledge dimensions, including cultural, time, space, biology, physics, and chemistry. Our Unify-Agent model attains the best Overall score of 0.77 within the Unified MLLM category, exceeding strong baselines such as BAGEL $+ \mathrm { C o T }$ (0.70). Unify-Agent delivers the strongest performance among unified models across most individual domains, particularly excelling in cultural (0.82), biological (0.72) and chemistry (0.70) knowledge. These results indicate that Unify-Agent effectively integrates complex world knowledge into the visual generation process, narrowing the performance gap with commercial models.

KiTTEN Tab. 3 summarizes the quantitative results on the KiTTEN benchmark, which evaluates fine-grained generation capabilities along two primary dimensions: text alignment and entity alignment. The evaluation spans eight diverse categories, such as Aircraft, Vehicle, and Cuisine. Our Unify-Agent establishes a new state of the art among both generation-only and unified MLLM approaches with an overall score of 4.08, outperforming the strong baseline Imagen-3 (3.50). Unify-Agent achieves the highest overall text alignment (4.22) and entity alignment (3.93) scores, maintaining balanced and superior performance across all sub-categories. This demonstrates the model’s robust capability to preserve intricate visual details and faithfully adhere to specific entity constraints.

Table 2 | Performance comparison on the WiSE benchmark across different knowledge dimensions (WiScore). denotes the best score and the second-best within each group.   

<table><tr><td>Model</td><td>Cultural</td><td>Time</td><td>Space</td><td>Biology</td><td>Physics</td><td>Chemistry</td><td>Overall</td></tr><tr><td colspan="8">Commercial Models</td></tr><tr><td>GPT-Image-1</td><td>0.81</td><td>0.71</td><td>0.89</td><td>0.83</td><td>0.79</td><td>0.74</td><td>0.80</td></tr><tr><td>Nano Banana-Pro</td><td>0.89</td><td>0.80</td><td>0.89</td><td>0.88</td><td>0.86</td><td>0.85</td><td>0.87</td></tr><tr><td>Nano Banana</td><td>0.89</td><td>0.87</td><td>0.95</td><td>0.89</td><td>0.89</td><td>0.79</td><td>0.89</td></tr><tr><td colspan="8">Generation Only</td></tr><tr><td>SD-v1-5</td><td>0.34</td><td>0.35</td><td>0.32</td><td>0.28</td><td>0.29</td><td>0.21</td><td>0.32</td></tr><tr><td>SD-2-1</td><td>0.30</td><td>0.38</td><td>0.35</td><td>0.33</td><td>0.34</td><td>0.21</td><td>0.32</td></tr><tr><td>FLUX.1-schnell</td><td>0.39</td><td>0.44</td><td>0.50</td><td>0.31</td><td>0.44</td><td>0.26</td><td>0.40</td></tr><tr><td>SD-3-medium</td><td>0.42</td><td>0.44</td><td>0.48</td><td>0.39</td><td>0.47</td><td>0.29</td><td>0.42</td></tr><tr><td>SD-XL-base-0.9</td><td>0.43</td><td>0.48</td><td>0.47</td><td>0.44</td><td>0.45</td><td>0.27</td><td>0.43</td></tr><tr><td>SD-3.5-medium</td><td>0.43</td><td>0.50</td><td>0.52</td><td>0.41</td><td>0.53</td><td>0.33</td><td>0.45</td></tr><tr><td>SD-3.5-large</td><td>0.44</td><td>0.50</td><td>0.58</td><td>0.44</td><td>0.52</td><td>0.31</td><td>0.46</td></tr><tr><td>PixArt-Alpha</td><td>0.45</td><td>0.50</td><td>0.48</td><td>0.49</td><td>0.56</td><td>0.34</td><td>0.47</td></tr><tr><td>playground-v2.5</td><td>0.49</td><td>0.58</td><td>0.55</td><td>0.43</td><td>0.48</td><td>0.33</td><td>0.49</td></tr><tr><td>FLUX.1-dev</td><td>0.48</td><td>0.58</td><td>0.62</td><td>0.42</td><td>0.51</td><td>0.35</td><td>0.50</td></tr><tr><td>Qwen-Image</td><td>0.62</td><td>0.63</td><td>0.77</td><td>0.57</td><td>0.75</td><td>0.40</td><td>0.62</td></tr><tr><td colspan="8">Unified MLLM</td></tr><tr><td>Janus-1.3B</td><td>0.16</td><td>0.26</td><td>0.35</td><td>0.28</td><td>0.30</td><td>0.14</td><td>0.23</td></tr><tr><td>Janus-Pro-1B</td><td>0.20</td><td>0.28</td><td>0.45</td><td>0.24</td><td>0.32</td><td>0.16</td><td>0.26</td></tr><tr><td>vila-u-7b-256</td><td>0.26</td><td>0.33</td><td>0.37</td><td>0.35</td><td>0.39</td><td>0.23</td><td>0.31</td></tr><tr><td>Janus-Pro-7B</td><td>0.30</td><td>0.37</td><td>0.49</td><td>0.36</td><td>0.42</td><td>0.26</td><td>0.35</td></tr><tr><td>Emu3</td><td>0.34</td><td>0.45</td><td>0.48</td><td>0.41</td><td>0.45</td><td>0.27</td><td>0.39</td></tr><tr><td>Hunyuan-Image-3.0</td><td>0.58</td><td>0.57</td><td>0.70</td><td>0.56</td><td>0.63</td><td>0.31</td><td>0.57</td></tr><tr><td>BLIP3o-8B</td><td>0.49</td><td>0.51</td><td>0.63</td><td>0.54</td><td>0.63</td><td>0.37</td><td>0.52</td></tr><tr><td>MetaQuery-XL</td><td>0.56</td><td>0.55</td><td>0.62</td><td>0.49</td><td>0.63</td><td>0.41</td><td>0.55</td></tr><tr><td>UniWorld-V1</td><td>0.53</td><td>0.55</td><td>0.73</td><td>0.45</td><td>0.59</td><td>0.41</td><td>0.55</td></tr><tr><td>BAGEL</td><td>0.44</td><td>0.55</td><td>0.68</td><td>0.44</td><td>0.60</td><td>0.39</td><td>0.52</td></tr><tr><td>BAGEL+CoT</td><td>0.76</td><td>0.69</td><td>0.75</td><td>0.65</td><td>0.75</td><td>0.58</td><td>0.70</td></tr><tr><td>Unify-Agent (Ours)</td><td>0.82</td><td>0.75</td><td>0.74</td><td>0.72</td><td>0.73</td><td>0.70</td><td>0.77</td></tr></table>

Table 3 | KiTTEN benchmark results (scores range from 0 to 5). Each category shows Text alignment and Entity alignment scores. denotes the best score and the second-best within each group.   

<table><tr><td rowspan="2">Model</td><td colspan="2">Aircraft</td><td colspan="2">Vehicle</td><td colspan="2">Cuisine</td><td colspan="2">Flower</td><td colspan="2">Insect</td><td colspan="2">Landmark</td><td colspan="2">Plant</td><td colspan="2">Sport</td><td colspan="2">Overall</td><td></td></tr><tr><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>Text</td><td>Entity</td><td>All</td></tr><tr><td colspan="19">Generation Only</td><td></td></tr><tr><td>SD</td><td>3.06</td><td>2.34</td><td>3.62</td><td>3.12</td><td>2.97</td><td>3.09</td><td>4.22</td><td>2.28</td><td>2.84</td><td>1.22</td><td>3.78</td><td>1.97</td><td>3.69</td><td>1.69</td><td>3.47</td><td>3.44</td><td>3.46</td><td>2.39</td><td>2.92</td></tr><tr><td>Flux</td><td>3.31</td><td>2.41</td><td>4.31</td><td>2.97</td><td>4.28</td><td>2.47</td><td>4.12</td><td>1.75</td><td>3.50</td><td>1.12</td><td>3.94</td><td>1.72</td><td>4.44</td><td>1.22</td><td>3.75</td><td>2.69</td><td>3.96</td><td>2.04</td><td>3.00</td></tr><tr><td>Custom-Diff</td><td>3.22</td><td>2.94</td><td>3.50</td><td>3.41</td><td>3.53</td><td>1.78</td><td>3.72</td><td>3.28</td><td>2.66</td><td>1.88</td><td>3.44</td><td>3.34</td><td>3.47</td><td>2.91</td><td>2.75</td><td>2.03</td><td>3.29</td><td>2.70</td><td>3.00</td></tr><tr><td>Imagen</td><td>3.16</td><td>2.81</td><td>3.78</td><td>3.16</td><td>3.75</td><td>3.25</td><td>4.12</td><td>2.72</td><td>2.88</td><td>1.12</td><td>3.53</td><td>2.41</td><td>3.94</td><td>1.59</td><td>3.75</td><td>3.19</td><td>3.61</td><td>2.53</td><td>3.07</td></tr><tr><td>DreamBooth</td><td>3.16</td><td>3.12</td><td>3.66</td><td>3.50</td><td>3.31</td><td>2.66</td><td>3.88</td><td>2.59</td><td>2.94</td><td>1.91</td><td>3.53</td><td>3.53</td><td>3.41</td><td>2.59</td><td>3.09</td><td>2.66</td><td>3.37</td><td>2.82</td><td>3.09</td></tr><tr><td>Instruct-Imagen</td><td>2.53</td><td>3.88</td><td>3.25</td><td>4.06</td><td>2.75</td><td>3.91</td><td>2.41</td><td>4.03</td><td>1.91</td><td>3.22</td><td>2.94</td><td>4.09</td><td>3.00</td><td>3.31</td><td>2.28</td><td>3.25</td><td>2.63</td><td>3.72</td><td>3.17</td></tr><tr><td>Imagen-3</td><td>3.41</td><td>3.19</td><td>4.25</td><td>3.34</td><td>4.25</td><td>4.06</td><td>4.38</td><td>3.03</td><td>4.09</td><td>1.31</td><td>3.88</td><td>2.25</td><td>4.72</td><td>1.69</td><td>4.41</td><td>3.75</td><td>4.17</td><td>2.83</td><td>3.50</td></tr><tr><td colspan="19">Unified MLLM</td><td></td></tr><tr><td>Bagel</td><td>3.51</td><td>2.97</td><td>4.46</td><td>4.08</td><td>3.91</td><td>3.40</td><td>3.44</td><td>2.64</td><td>2.65</td><td>1.58</td><td>3.31</td><td>1.79</td><td>3.01</td><td>1.71</td><td>3.23</td><td>2.98</td><td>3.44</td><td>2.64</td><td>3.04</td></tr><tr><td>Bagel-CoT</td><td>3.67</td><td>2.94</td><td>4.42</td><td>3.98</td><td>4.10</td><td>4.01</td><td>3.56</td><td>3.12</td><td>2.98</td><td>2.12</td><td>3.41</td><td>2.11</td><td>3.31</td><td>2.26</td><td>3.84</td><td>3.63</td><td>3.66</td><td>2.02</td><td>2.84</td></tr><tr><td>Unify-Agent (Ours)</td><td>4.03</td><td>3.59</td><td>4.54</td><td>4.09</td><td>4.36</td><td>4.13</td><td>4.28</td><td>4.16</td><td>4.17</td><td>3.97</td><td>3.99</td><td>3.76</td><td>4.46</td><td>3.81</td><td>3.95</td><td>3.92</td><td>4.22</td><td>3.93</td><td>4.08</td></tr></table>

T2I-FactBench Tab. 4 presents results on T2I-FactBench, a three-tiered benchmark measuring the factual accuracy of T2I models: Single Knowledge Concept Memorization (SKCM), Instantiation (SKCI), and Multiple Concept Composition with Interaction (MKCC). Within the unified MLLM group, Unify-Agent achieves top overall scores in SKCI (77.4) and MKCC (71.5). It also records an SKCM concept score of 69.2, comparing favorably against commercial models like DALLE-3 (55.5). This progression indicates that Unify-Agent captures the intrinsic visual attributes of single concepts (SKCM), adapts them to varied conditions like differing actions or scenes (SKCI), and handles the explicit semantic relationships necessary for multi-concept interactions (MKCC).

Table 4 | Comparison on T2I-FactBench (SKCM, SKCI, and MKCC). denotes the best score and the second-best within each group.   

<table><tr><td rowspan="2">Model</td><td>SKCM</td><td colspan="3">SKCI</td><td colspan="4">MKCC</td></tr><tr><td>Concept</td><td>Concept</td><td>Instantiation</td><td>All</td><td>Concept</td><td>Instantiation</td><td>Composition</td><td>All</td></tr><tr><td colspan="9">Generation Only</td></tr><tr><td>SD v1.5</td><td>40.5</td><td>52.9</td><td>53.9</td><td>53.4</td><td>37.6</td><td>13.4</td><td>15.1</td><td>22.0</td></tr><tr><td>Pixart</td><td>26.4</td><td>46.2</td><td>55.3</td><td>50.8</td><td>35.8</td><td>19.8</td><td>24.3</td><td>26.6</td></tr><tr><td>SD XL</td><td>45.8</td><td>59.9</td><td>65.8</td><td>62.8</td><td>51.7</td><td>28.0</td><td>35.4</td><td>38.4</td></tr><tr><td>Playground</td><td>45.6</td><td>66.1</td><td>62.5</td><td>64.3</td><td>53.8</td><td>35.4</td><td>44.8</td><td>44.7</td></tr><tr><td>Flux.1-dev</td><td>35.6</td><td>54.9</td><td>58.0</td><td>56.5</td><td>56.9</td><td>54.1</td><td>63.8</td><td>58.3</td></tr><tr><td>SD-3.5</td><td>46.2</td><td>64.6</td><td>71.2</td><td>67.9</td><td>68.9</td><td>59.2</td><td>75.5</td><td>67.9</td></tr><tr><td>DALLE-3</td><td>55.5</td><td>72.4</td><td>88.5</td><td>80.5</td><td>71.3</td><td>70.2</td><td>85.6</td><td>75.7</td></tr><tr><td colspan="9">Unified MLLM</td></tr><tr><td>Bagel</td><td>31.8</td><td>52.3</td><td>64.2</td><td>58.2</td><td>68.1</td><td>57.6</td><td>73.2</td><td>66.3</td></tr><tr><td>Bagel-CoT</td><td>34.6</td><td>52.6</td><td>64.5</td><td>58.5</td><td>67.2</td><td>60.3</td><td>77.1</td><td>68.2</td></tr><tr><td>Unify-Agent (Ours)</td><td>69.2</td><td>75.3</td><td>79.6</td><td>77.4</td><td>76.1</td><td>64.8</td><td>73.6</td><td>71.5</td></tr></table>

Table 5 | Ablation study of Unify-Agent on the FactIP benchmark. We report top-level weighted averages over Clarity, Content, Aesthetics, and Relevance. The middle rows show expected performance drops when removing key pipeline components, grounded constraint types, or recaption-stage architectural modules. Numbers in parentheses denote changes relative to the vanilla Bagel baseline; red indicates improvement and green indicates degradation. The best result in each column is marked in bold, and the second-best is underlined.   

<table><tr><td>Variant</td><td>Clarity</td><td>Content</td><td>Aesthetics</td><td>Relevance</td><td>Overall</td></tr><tr><td>Baseline (Vanilla Bagel)</td><td>91.3</td><td>64.2</td><td>84.0</td><td>44.9</td><td>50.9</td></tr><tr><td colspan="6">Pipeline Ablations</td></tr><tr><td>w/o Text-Search</td><td>90.7 (-0.6)</td><td>70.9 (+6.7)</td><td>84.3 (+0.3)</td><td>64.6 (+19.7)</td><td>65.4 (+14.5)</td></tr><tr><td>w/o Image-Search</td><td>92.1 (+0.8)</td><td>73.1 (+8.9)</td><td>85.0 (+1.0)</td><td>50.8 (+5.9)</td><td>56.2 (+5.3)</td></tr><tr><td>w/o Recapition</td><td>83.0 (-8.3)</td><td>69.0 (+4.8)</td><td>74.5 (-9.5)</td><td>60.2 (+15.3)</td><td>62.9 (+12.0)</td></tr><tr><td colspan="6">Constraint Ablations</td></tr><tr><td>Recapition w/o Identity-preserving</td><td>91.5 (+0.2)</td><td>72.6 (+8.4)</td><td>83.4 (-0.6)</td><td>65.9 (+21.0)</td><td>67.7 (+16.8)</td></tr><tr><td>Recapition w/o Scene-compositional</td><td>90.7 (-0.6)</td><td>70.8 (+6.6)</td><td>80.7 (-3.3)</td><td>68.6 (+23.7)</td><td>68.2 (+17.3)</td></tr><tr><td colspan="6">Recapition Architecture Ablations</td></tr><tr><td>Recapition w/o VAE</td><td>90.9 (-0.4)</td><td>74.3 (+10.1)</td><td>84.5 (+0.5)</td><td>70.8 (+25.9)</td><td>71.2 (+20.3)</td></tr><tr><td>Recapition w/o ViT</td><td>88.6 (-2.7)</td><td>68.4 (+4.2)</td><td>81.1 (-2.9)</td><td>58.7 (+13.8)</td><td>61.4 (+10.5)</td></tr><tr><td>Unify-Agent (Full)</td><td>91.2 (-0.1)</td><td>75.2 (+11.0)</td><td>85.2 (+1.2)</td><td>72.4 (+27.5)</td><td>73.2 (+22.3)</td></tr></table>

# 6.3 Analysis of Experimental Results

# 6.3.1 Ablation Study

Tab. 5 presents the ablation results of Unify-Agent on the FactIP benchmark. Compared with the vanilla Bagel baseline, the full model achieves the best performance on all metrics, improving the Overall score from 50.9 to 73.2. The largest gain is observed on Relevance $( 4 4 . 9  7 2 . 4 ) $ ), indicating that the main advantage of Unify-Agent lies in stronger factual grounding and identity fidelity, rather than merely improving generic image quality.

Pipeline-level ablations confirm the contribution of each component. Removing text search reduces the Overall score to 65.4, showing that textual retrieval provides an important semantic scaffold for disambiguation and high-level factual grounding. Removing image search causes a larger degradation, especially on Relevance $7 2 . 4  5 0 . 8 )$ ), and lowers the Overall score to 56.2, which highlights the importance of visual evidence for preserving fine-grained identity and appearance details. Removing recaptioning also leads to a substantial drop (Overall 62.9), verifying that raw retrieved evidence is not an optimal conditioning signal and must be reorganized into a compact, executable specification.

Surrounded by stacks of books and papers in his stone cottage study, William Butler Yeats, writing a manuscript by the light of an oil lamp.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ca4391966dac9bfb2201accd5c9442b6b9b9e643d20169429291b0f34bca3bee.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/dc93203e295e6b6fa887f884a55a1d428c7f63551aeee81670866c11d5e08f43.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/a3c9360d66d56e48fc174f84c54d73e8157952469f687026fa9a3095c87a2739.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/91b70cdd70202911aba9203a2e33d149c9342bbf800ebc192793489dd9f1c288.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/86e4c629943df2c14d428cab4ee98f562f8ee40db72220a5a10a110216d8e839.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/1c72c825da391d3c1bd09468a164a1b91c43ccde41b014a0efe15c50fe2c5638.jpg]]

In a cluttered, equipmentfilled 1930s research laboratory, Sir Henry Dale is peering intently through a brass microscope.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/271a2cbc71e1174cde9b6f1a7bf57142979f22cb77a9bee093ad3b02eb13b24f.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/69e41d2a6205129148e7470f6cb91b5445b9279b309f5beb9769fa827979872b.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/bac4e95632062b1e799431ee8ad45638f20d15bde045e036cda1c3e1a0432258.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/9e427942da4e3c88298353fd65f03cdda3d448bbc37ca7d068a2f33285290939.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/70aa7f4bc3ae3339769b0ebc13e476946ef47572ac5db72ed595d1d1d40d9b4a.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/6ab34cd67c6e0b08f2f468f7c0230edf4937ef07a977d7ed8fef53b6e66ca7d3.jpg]]

An oil painting depicting the immense "Steel Titan" robot from the Near Future chapter of the game LIVE A LIVE. …

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/88db8027861c7e4303a290dc2d8fc4dfd666c21d8dd3489440da8bd915810b63.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/4bc4d656561a9519a29d370e0f8bcf623b333b256006ac09662e5a8034f618fc.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0e56b5287d768c4143202917c18d47ffaaccd981e189f5b4456a9f29139b030b.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b7ad8c002a2a7845a7fb7a59d19c141462293e7f68275e62f20558b354b3d50c.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/73cae295bb6f958a042543fe5760d51b8879e2f3814e3e58e15d556538f9f9da.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/dd7047238a7f31274999e4550bca6a5f930aabbe65e1b674a9f966d1f5d37e04.jpg]]

A detailed product photograph of the ROBOT Spirits MS-06V Zaku Tank "Sand Sheep" action figure.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/08fcbb49bd80d0b458d175c46d60aeece2955bd6c0e3176844fbcc1f3879c694.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/3585d2a74ffaa23d14a4b632b318917118b3dd26ee18f8d2dcf51f2312d18983.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/43f738a080f3ae50c43aad40b7fd021372f4f88d7c4a7ea665a27561f6d00d9f.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/8da94eeaf57ced4db936ba063aa7f5d545cd286600ac586e621a907e648612a7.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/792b78704d7cd85f43fe68ecd2134c9390fbdec98a4ff764d01e8710fcbd616f.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/eb38b9abb56d16c284dc7fb44c85357be7355b719a659d75b1486e0787fe2e19.jpg]]

DUDOO is riding a small yellow tricycle. The basket at the front is filled with small white daisy accessories.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/75117a78b4ec2cc97d307878a9f40fcf65ac4dbc8024d5b4d5df84d6a62fd22b.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ed4d43231d3bb9a96a7840c99b730bd48daa2be4586da008537950414c62d5cb.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/182af60254fc5d433a5cc1d476eb17801d2d1b5b691adacef25049129579962e.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c364ed0fdcfec93213945f16d882ab5099b0875e1c171bd2faec1f9982cb30fb.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/41641721cd121a1000caecfc0cc7d8435d8de910f8092958db838a2b2de9858e.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/1fdc48548294ddf32d3b88b029b6c3425b22578df0c444c60eed9e42b067d52f.jpg]]

Surrounded by towering stacks of books and papers in a home study, Jacques Derrida is seated at a desk, deeply focused on writing by hand.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/42d6fdbb15974bc9713cec45d1a3a23dd14eb3f1ecb7b94bf0ebff3cac6cf1cd.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/efeb7ad5e76c88e0dc73d3cb094fc1388a99d740ab1170eb0f17c273f0ea514c.jpg]]  
Ours

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/99ebe0ba4fcd075d4c867eb07a69c7b3f24140f91950f12d10a11c99e9bb02e2.jpg]]  
Flux-1

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/982b89adae779b3bda9defe1fce2c7193be483932a6467ba5de067872fead327.jpg]]  
Bagel-7b

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/18b207bd84ce1afa414e96284ca596b6f8154e3b42494b5c069bd565d8f88ab1.jpg]]  
Hunyuan

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/bc1dbab65ab26c01176d877eb29b4b4753feecd499fb9160b1b15305f46ef5d7.jpg]]  
Stabel-diffusion   
Figure 5 | Qualitative comparison of multi-image generation results on knowledge-intensive prompts involving historical figures, fictional characters, products, and stylized toys. Our method consistently produces images that better preserve subject identity, fine-grained attributes, and prompt-specific details, while achieving stronger real-world knowledge grounding than competing baselines, including Flux-1, Bagel-7b, Hunyuan, and Stable Diffusion.

Constraint-level ablations show a similarly clear pattern. Without identity-preserving constraints, Relevance drops to 65.9, confirming their role in maintaining subject-specific fidelity. Without scene-compositional constraints, Content and Aesthetics decline to 70.8 and 80.7, respectively, indicating their importance for faithfully instantiating prompt-specified scene attributes. Overall, these results show that the gains of Unify-Agent arise from the joint effect of sequential multimodal evidence acquisition, grounded recaptioning, and structured generation constraints.

# 6.4 Finding: Why Generation Helps Understanding in a Unified Model

The ablations in Tab. 5 provide an additional insight beyond component-level contributions: in a unified understandinggeneration model, generation can in fact improve understanding. This effect is particularly evident in the recaption stage, where the model must transform retrieved reference images into a structured textual specification that is both semantically faithful and generation-ready. In other words, recaption is not a pure language task, but a multimodal understanding

problem that requires the model to identify what appears in the image, which attributes are identity-critical, and how these visual cues should be expressed for downstream synthesis.

This observation helps explain why an end-to-end unified architecture is beneficial. In Bagel-style unified understandinggeneration models, visual inputs are tokenized through a dual $V A E { + } V i T$ design. Within this parallel architecture, the VAE extracts low-level visual latents that preserve appearance details such as color, texture, material, and local structure, while the ViT simultaneously encodes the input images into higher-level semantic tokens that are more suitable for visual understanding and recaptioning. As ViT is effective at modeling global context and long-range dependencies, it functions as a semantic visual encoder that helps the language branch reason about object identity, attribute relations, and scene composition. At the same time, the VAE complements this process by retaining fine-grained perceptual priors that are often crucial for identity-sensitive grounding.

The recaption architecture ablations in Tab. 5 are consistent with this interpretation. Removing ViT causes a substantial drop in performance, especially on Relevance and Overall, indicating that high-level semantic visual tokens are critical for accurately understanding the retrieved reference images. Removing VAE also degrades performance, albeit more moderately, suggesting that low-level appearance cues remain helpful even when higher-level semantic reasoning is available. Together, these results suggest that the benefit of a unified model does not come only from parameter sharing, but from the joint availability of low-level generative latents and high-level semantic visual representations within a single end-to-end framework. This synergy allows the model to produce better recaptions, which in turn leads to better grounded image synthesis.

# 6.4.1 Visualization comparison results

Figure 5 shows that our Unify-Agent consistently outperforms Flux-1, Bagel-7b, Hunyuan, and Stable Diffusion on knowledge-intensive and attribute-sensitive prompts. Across diverse cases including historical figures, fictional entities, fine-grained products, and stylized toys, our method generates images with superior subject fidelity, more accurate attribute binding, and stronger prompt adherence. In particular, Unify-Agent better captures entity-specific identity cues and long-tail concepts, whereas competing methods often suffer from identity drift, missing attributes, or degeneration into generic substitutes. These qualitative results demonstrate that, by combining unified multi-image generation with agentic search for external world knowledge, Unify-Agent achieves more accurate, consistent, and knowledge-grounded visual generation.

# 7 Conclusion

We present Unify-Agent, a unified multimodal agent for world-grounded image synthesis. Rather than treating image generation as a closed-book prompt-to-image mapping, Unify-Agent formulates it as an inference-time process of cognitive gap detection, multimodal evidence acquisition, grounded recaptioning, and final synthesis. By explicitly resolving missing world knowledge through retrieved textual and visual evidence, our framework enables more faithful generation of rare, long-tail, and knowledge-intensive concepts. More broadly, our findings suggest that world-grounded image synthesis is a distinct multimodal reasoning problem that requires tightly coupled understanding, acting, and generation. We hope this work motivates future research on more reliable and open-world generative agents.

# Limitations and Future Work

Our work still has several limitations. Current open-source unified multimodal models remain substantially weaker than the strongest closed-source systems; for example, Bagel still has limited long-context capability and can support only a relatively small number of images within a single context, which constrains more complex agent behaviors. Moreover, although we demonstrate strong results on IP- and concept-centric world-grounded synthesis, our current pipeline is still limited to a relatively shallow one-pass workflow, rather than more general iterative agent behaviors such as interleaved text-image search, reflection, and replanning, which are crucial for harder open-world tasks such as travel planning or academic report generation. At the same time, our results provide encouraging evidence for the feasibility of end-to-end unified agent training and highlight the advantage of unified generation and reasoning multimodal models, where generation and understanding can mutually reinforce each other. In future work, we plan to validate these findings on stronger unified backbones and extend the framework toward more capable multimodal agents that can support longer-horizon planning, repeated search, and adaptive reasoning in more complex real-world settings.

# References

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.   
Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weijie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Linus, Lucaz Liu, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyang Peng, Yuanbo Peng, Xiangwei Shen, Yixuan Shi, Jiale Tao, Yangyu Tao, Qi Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Lucas Wang, Qixun Wang, Weiyan Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fang Yang, Miles Yang, Xiaofeng Yang, Xuan Yang, Zhantao Yang, Jingmiao Yu, Zheng Yuan, Chao Zhang, Jian-Wei Zhang, Peizhen Zhang, Shi-Xue Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zijian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jianchen Zhu, and Zhao Zhong. Hunyuanimage 3.0 technical report, 2026. URL https://arxiv.org/abs/2509.23951.   
Chieh-Yun Chen, Min Shi, Gong Zhang, and Humphrey Shi. T2i-copilot: A training-free multi-agent text-to-image system for enhanced prompt interpretation and interactive generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19396–19405, 2025a.   
Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu, Yaochen Wang, Huichi Zhou, Qihui Zhang, Pan Zhou, Yao Wan, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. arXiv preprint arXiv:2402.04788, 2024.   
Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025b.   
Shuang Chen, Yue Guo, Yimeng Ye, Shijue Huang, Wenbo Hu, Haoxi Li, Manyuan Zhang, Jiayu Chen, Song Guo, and Nanyun Peng. Ares: Multimodal adaptive reasoning via difficulty-aware token-level entropy shaping. arXiv preprint arXiv:2510.08457, 2025c.   
Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025d.   
Zhiyuan Chen, Yuecong Min, Jie Zhang, Bei Yan, Jiahao Wang, Xiaozhen Wang, and Shiguang Shan. A survey of multimodal hallucination evaluation and detection, 2026. URL https://arxiv.org/abs/2507.19024.   
Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners, 2025. URL https://arxiv.org/abs/2510.26583.   
Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining, 2025a. URL https://arxiv.org/abs/2505.14683.   
Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025b.   
Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.   
Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, and Xiangyu Yue. Exploring reasoning reward model for agents, 2026. URL https://arxiv.org/abs/ 2601.22154.

Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, and Xiangyu Yue. Gen-searcher: Reinforcing agentic search for image generation. arXiv preprint arXiv:2603.28767, 2025a.   
Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, Yan Feng, Peng Pei, Xunliang Cai, and Xiangyu Yue. Onethinker: All-in-one reasoning model for image and video, 2025b. URL https://arxiv.org/abs/2512.03043.   
Google. Nano banana 2: Combining pro capabilities with lightning-fast speed. https://blog.google/innova tion-and-ai/technology/ai/nano-banana-2/, 2026.   
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.   
Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, and Weijia Li. Mind-brush: Integrating agentic cognitive search and reasoning into image generation. arXiv preprint arXiv:2602.01756, 2026.   
Hsin-Ping Huang, Xinyi Wang, Yonatan Bitton, Hagai Taitelbaum, Gaurav Singh Tomar, Ming-Wei Chang, Xuhui Jia, Kelvin CK Chan, Hexiang Hu, Yu-Chuan Su, et al. Kitten: A knowledge-intensive evaluation of image generation on visual entities. arXiv preprint arXiv:2410.11824, 2024a.   
Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36: 78723–78747, 2023.   
Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, Junbo Qiao, Yue Guo, Yao Hu, Zhenfei Yin, Philip Torr, Yu Cheng, Wanli Ouyang, and Shaohui Lin. Interleaving reasoning for better text-to-image generation, 2025. URL https: //arxiv.org/abs/2509.06945.   
Wenxuan Huang, Yu Zeng, Qiuchen Wang, Zhen Fang, Shaosheng Cao, Zheng Chu, Qingyu Yin, Shuang Chen, Zhenfei Yin, Lin Chen, et al. Vision-deepresearch: Incentivizing deepresearch capability in multimodal large language models. arXiv preprint arXiv:2601.22060, 2026.   
Ziwei Huang, Wanggui He, Quanyu Long, Yandi Wang, Haoyuan Li, Zhelun Yu, Fangxun Shu, Long Chan, Hao Jiang,

Fei Wu, et al. T2i-factualbench: Benchmarking the factuality of text-to-image models with knowledge-intensive concepts. arXiv preprint arXiv:2412.04300, 2024b.   
Black Forest Labs. Flux.1 [dev]: Text-to-image generation model. https://huggingface.co/black-fores t-labs/FLUX.1-dev, 2025.   
Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.   
Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, et al. Holistic evaluation of text-to-image models. Advances in Neural Information Processing Systems, 36:69981–70011, 2023.   
Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024. URL https://arxiv.org/abs/2402 .17245.   
Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J. Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization, 2024. URL https://arxiv.org/abs/2309.06657.   
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv: 2209.03003, 2022.   
Roberto Navigli and Simone Paolo Ponzetto. BabelNet: Building a very large multilingual semantic network. In Jan Hajič, Sandra Carberry, Stephen Clark, and Joakim Nivre, editors, Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, pages 216–225, Uppsala, Sweden, July 2010. Association for Computational Linguistics. URL https://aclanthology.org/P10-1023/.   
Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.   
OpenAI. Gpt-image-1.5 model documentation. https://developers.openai.com/api/docs/models/ gpt-image-1.5, 2025.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
Bytedance Seed. Seed2. 0 model card: Towards intelligence frontier for real-world complexity. Technical report, Technical report, Bytedance, 2025. URL https://lf3-static. bytednsdoc. com . . . .   
Team Seedream, :, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, Xiaowen Jian, Huafeng Kuang, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yanzuo Lu, Zhengxiong Luo, Tongtong Ou, Guang Shi, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Rui Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Wenxu Wu, Yonghui Wu, Xin Xia, Xuefeng Xiao, Shuang Xu, Xin Yan, Ceyuan Yang, Jianchao Yang, Zhonghua Zhai, Chenlin Zhang, Heng Zhang, Qi Zhang, Xinyu Zhang, Yuwei Zhang, Shijia Zhao, Wenliang Zhao, and Wenjia Zhu. Seedream 4.0: Toward next-generation multimodal image generation, 2025. URL https://arxiv.org/abs/2509.20427.   
Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. 2023. URL https://arxiv.org/abs/2303.17580.   
Quanxin Shou, Fangqi Zhu, Shawn Chen, Puxin Yan, Zhengyang Yan, Yikun Miao, Xiaoyi Pang, Zicong Hong, Ruikai Shi, Hao Huang, et al. Halo: A unified vision-language-action model for embodied multimodal chain-of-thought reasoning. arXiv preprint arXiv:2602.21157, 2026.   
Moo Hyun Son, Jintaek Oh, Sun Bin Mun, Jaechul Roh, and Sehyun Choi. World-to-image: Grounding text-to-image generation with agent-driven world knowledge. arXiv preprint arXiv:2510.04201, 2025.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation, 2024. URL https://arxiv.org/abs/2406.06525.   
Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024a.   
Gemini Team. Gemini: A family of highly capable multimodal models, 2025a. URL https://arxiv.org/abs/ 2312.11805.   
OpenAI Team. Gpt-4o system card, 2024b. URL https://arxiv.org/abs/2410.21276.   
Seed Team. Seed1.5-vl technical report, 2025b. URL https://arxiv.org/abs/2505.07062.   
Z-Image Team. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025c.   
Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.   
Linqing Wang, Ximing Xing, Yiji Cheng, Zhiyuan Zhao, Li Donghao, Hang Tiankai, Li Zhenxi, Jiale Tao, QiXun Wang, Ruihuang Li, Comi Chen, Xin Li, Mingrui Wu, Xinchi Deng, Shuyang Gu, Chunyu Wang, and Qinglin Lu. Promptenhancer: A simple approach to enhance text-to-image models via chain-of-thought prompt rewriting. arXiv preprint arXiv:2509.04545, 2025.   
Zhenyu Wang, Aoxue Li, Zhenguo Li, and Xihui Liu. Genartist: Multimodal llm as an agent for unified image generation and editing. Advances in Neural Information Processing Systems, 37:128374–128395, 2024.   
Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models, 2023. URL https://arxiv.org/abs/2303.04671.   
Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025a. URL https://arxiv.org/abs/2508.02324.   
Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025b.   
Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.   
Hui Yang, Sifu Yue, and Yunzhong He. Auto-gpt for online decision making: Benchmarks and additional opinions, 2023a. URL https://arxiv.org/abs/2306.02224.   
Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action, 2023b. URL https://arxiv.org/abs/2303.11381.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. 2022.   
Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, and Weijia Li. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation, 2025. URL https://arxiv.org/abs/2508.09987.   
Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, et al. Vision-deepresearch benchmark: Rethinking visual and textual search for multimodal large language models. arXiv preprint arXiv:2602.02185, 2026.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

# Appendix

# A Implementation Details

The Supervised Fine-Tuning (SFT) is conducted on our self-constructed interleaved image-and-text dataset with 64 NVIDIA H20 GPUs. The entire fine-tuning process lasted for approximately 10 days, completing a total of 10k gradient steps. For the training objective, our model is optimized using a combined loss formulation that jointly updates the language model, the ViT encoder, and the multimodal connectors, while keeping the continuous generation VAE weights strictly frozen. Specifically, we apply a standard Cross-Entropy (CE) loss for autoregressive text generation and visual token prediction, alongside a Mean Squared Error (MSE) loss on the image-reconstruction latents for the visual generation branch. Both the CE and MSE losses are equally scaled with a weight of 1.0. Furthermore, to improve the model’s instruction-following capability in multimodal contexts, we enable CE loss reweighting, which specifically increases the penalty weight on designated special tokens. To stabilize the optimization over ultra-long interleaved sequences, we apply a constant learning rate scheduler with a peak learning rate of $5 \times 1 0 ^ { - 5 }$ after 500 linear warmup steps, and enforce gradient clipping with a maximum L2 norm of 5.0. A comprehensive list of hyperparameters for the SFT stage is summarized in Table 6.

Table 6 | Hyperparameters for the Supervised Fine-Tuning (SFT) stage.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Model &amp; Architecture Setup</td></tr><tr><td>Base LLM</td><td>Bagel-14B</td></tr><tr><td>Vision Encoder (ViT)</td><td>SigLIP-SO400M-14 (NaViT)</td></tr><tr><td>Visual Generation VAE</td><td>FLUX VAE (Frozen)</td></tr><tr><td>ViT Patch Size</td><td>14</td></tr><tr><td>VAE Latent Patch Size</td><td>2</td></tr><tr><td>Tied Word Embeddings</td><td>False</td></tr><tr><td colspan="2">Data &amp; Sequence Length</td></tr><tr><td>Max Tokens per Sample</td><td>40,240</td></tr><tr><td>Expected Tokens per Batch</td><td>40,240</td></tr><tr><td>Max Packed Tokens (Hard Limit)</td><td>41,520</td></tr><tr><td>Vit Transform</td><td>(378, 980)</td></tr><tr><td>Vae Transform</td><td>(512, 1024)</td></tr><tr><td colspan="2">Optimization &amp; Objective</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Adam (β1, β2, ε)</td><td>(0.9, 0.95, 10-15)</td></tr><tr><td>Learning Rate</td><td>5 × 10-5</td></tr><tr><td>LR Scheduler</td><td>Constant</td></tr><tr><td>Warmup Steps</td><td>500</td></tr><tr><td>Total Training Steps</td><td>10,000</td></tr><tr><td>Max Gradient Norm</td><td>5.0</td></tr><tr><td>Cross-Entropy (CE) Weight</td><td>1.0</td></tr><tr><td>Reconstruction (MSE) Weight</td><td>1.0</td></tr><tr><td>CE Loss Special Token Reweighting</td><td>True</td></tr><tr><td>Special Token CE Weight</td><td>3.0</td></tr><tr><td colspan="2">Hardware &amp; Distributed Setup</td></tr><tr><td>Hardware</td><td>64× NVIDIA H20 GPUs</td></tr><tr><td>Training Duration</td><td>~ 10 Days</td></tr><tr><td>Parallel Strategy</td><td>FSDP (HYBRID_SHARD)</td></tr></table>

# B Attention Masking Strategy

We implement a hybrid attention masking strategy tailored for interleaved agentic data to balance sequential logic with precise visual grounding. For the textual components of the sequence, including agentic dialogs and reasoning steps, a standard causal masking is applied to preserve the temporal and logical flow of the conversation. In contrast, retrieved reference images—represented by a composite of VAE and ViT tokens—employ a full attention mechanism, allowing these tokens to interact globally to facilitate holistic visual feature extraction. Notably, during the subsequent image-generation phase, the latent flow-matching tokens (i.e., VAE noisy tokens) are specifically restricted to attending exclusively to the preceding recaption tokens and the retrieved reference image tokens. By selectively masking out irrelevant textual reasoning traces and historical dialogs at this stage, we prevent large segments of high-noise text from interfering with the synthesis process, thereby ensuring that the generated visual content remains precisely grounded in the condensed recaptioning and the provided visual evidence. See Fig. 6 for details of our attention masking strategy.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2bb0a1d4493f475fc0e673bd9a9c24c4bb10848b52bd6374f9af9492f3dc2d19.jpg]]  
Figure 6 | Hybrid attention masking strategy for agentic image generation. The matrix shows causal attention for reasoning traces and textual dialog (blue) and full attention for reference visual-textual evidence (pink/orange/green). Notably, the generation phase (grey) employs a restricted mask that filters out historical reasoning noise, forcing the latent flow-matching tokens to attend only to relevant reference images and recaptions for high-fidelity synthesis.

# C Evaluation Protocols

To ensure rigorous and standardized evaluation across benchmarks with distinct task formulations and assessment objectives, we adhere to the official evaluation protocols of each benchmark whenever possible. For all existing benchmarks, we preserve their original MLLM-based evaluation settings, including the designated expert evaluators and scoring schemes, to maintain consistency with prior literature and improve the fairness and comparability of our results.

WiSE Evaluation. For the WiSE Benchmark, we employ GPT-4o (Team, 2024b) as the expert evaluator to assess the alignment between generated images and world knowledge. Given the textual prompt and the model-generated result, the evaluator is instructed to score the generation from three complementary aspects: Consistency, Realism, and Aesthetic Quality. Specifically, Consistency evaluates the accuracy and completeness with which the generated image reflects the prompt; Realism assesses the physical plausibility, including adherence to physical laws and accurate material representation; and Aesthetic Quality measures the overall artistic appeal, including composition and color harmony. Each dimension is evaluated on a discrete scale from 0 to 2 (0 for Rejected, 1 for Conditional, and 2 for Exemplary).

To obtain a unified measure of knowledge-image alignment, we define the overall metric, WiScore, as a weighted combination of the three dimensions:

$$
\text {W i S c o r e} = \alpha_ {1} \cdot \text {C o n s i s t e n c y} + \alpha_ {2} \cdot \text {R e a l i s m} + \alpha_ {3} \cdot \text {A e s t h e t i c Q u a l i t y}, \tag {20}
$$

$$
\text {s u b j e c t} \alpha_ {1} + \alpha_ {2} + \alpha_ {3} = 1. \tag {21}
$$

In our experiments, we set $\alpha _ { 1 } = 0 . 7$ , $\alpha _ { 2 } = 0 . 2$ , and $\alpha _ { 3 } = 0 . 1$ . This configuration explicitly prioritizes Consistency, reflecting that accurately representing the intended objects and relationships grounded in world knowledge is the most critical criterion. Meanwhile, Realism and Aesthetic Quality are retained as auxiliary dimensions to ensure overall visual excellence without overshadowing the core knowledge assessment.

KiTTEN Evaluation. For the KiTTEN Benchmark, we utilize GPT-4o (Team, 2024b) as the expert evaluator to systematically assess the fine-grained visual fidelity of generated entities. Given the textual prompt, ground-truth reference images of the target entity, and the model-generated result, the evaluation framework decouples the assessment into two primary dimensions: Text Alignment and Entity Alignment. Specifically, Text Alignment evaluates the degree to which the generated image faithfully adheres to the prompt instructions beyond just the entity; and Entity Alignment assesses how well the generated image captures precise visual details and realistically resembles the target entity based on the provided reference images. Each dimension is independently scored on an integer scale from 1 to 5, where 1 indicates no alignment and 5 denotes complete faithfulness. This decoupled design deliberately avoids fusing the scores into a single metric, allowing for a nuanced analysis of the trade-offs between preserving specific entity knowledge and strictly following complex textual constraints.

T2I-FactBench Evaluation. For the T2I-FactBench, we use a multi-round VQA protocol with GPT-4o across three levels: SKCM, SKCI, and MKCC. The evaluation consists of three progressive stages.

In the first round, Concept Factuality assesses the generated knowledge concept $c _ { i }$ in image $I _ { i }$ against its reference $I _ { i } ^ { R }$ across four dimensions: Shape $( S )$ , Color $( C )$ , Texture $( T )$ , and Feature Details $( F )$ . For each dimension, GPT-4o assigns a binary score (0 or 1). To account for multiple concepts at the MKCC level, the score is defined as:

$$
\text {C o n c e p t F a c t u a l i t y} = \frac {1}{N _ {i}} \sum_ {j = 1} ^ {N _ {i}} \left(\frac {S _ {i j} + C _ {i j} + T _ {i j} + F _ {i j}}{4}\right), \tag {22}
$$

where $N _ { i }$ denotes the number of concepts within $I _ { i }$ , and $S _ { i j } , C _ { i j } , T _ { i j } , F _ { i j }$ represent binary scores for the $j$ -th concept.

The second round, Instantiation Completeness, verifies whether concept $c$ is present and if the instantiation phrase $p$ is successfully realized. A score of 1 is assigned only if both conditions are satisfied.

Finally, for the MKCC level, a third round evaluates Composition Factuality across four dimensions: Seamless Transition $( S _ { i } )$ , Visual Completeness $( V _ { i } )$ , Authenticity $( A _ { i } )$ , and Prompt Following $( P _ { i } )$ . The composition factuality score is defined as:

$$
\text {C o m p o s i t i o n F a c t u a l i t y} = \frac {S _ {i} + V _ {i} + A _ {i} + P _ {i}}{4}, \tag {23}
$$

where $S _ { i } , V _ { i } , A _ { i } , P _ { i } \in \{ 0 , 1 \}$ represent scores for the respective dimensions for the $i$ -th prompt.

FactIP Benchmark Evaluation. For our proposed FactIP Benchmark, we employ Seed2.0 (Seed) as the expert evaluator to conduct a fine-grained and structured assessment of generated images and the evaluation prompt can be found in E.2. Given the textual prompt, two reference images (GT1 and GT2), and the model-generated result (AS), the evaluator is instructed to jointly compare the generated image against both the prompt and the references, and to score it from four complementary aspects: Clarity, Content, Aesthetics, and Relevance. Specifically, Clarity evaluates image sharpness, visual cleanliness, and the richness of perceptual details; Content measures whether the generated image faithfully captures the key semantic elements and compositional requirements specified in the prompt; Aesthetics assesses the overall visual quality, including composition, lighting, color harmony, and stylistic appeal; and Relevance measures whether the generated image preserves the intended IP identity and remains consistent with the defining attributes inferred from the two reference images. Each dimension is scored on an integer scale from 0 to 10 by the expert evaluator. Following the evaluation protocol, the two references (GT1 and GT2) are considered jointly, such that stable identity traits are emphasized while variations already existing across the references are not over-penalized.

To obtain a unified measure of generation quality, we define the overall score as a weighted combination of the four dimensions:

${ \mathrm { O v e r a l l ~ S c o r e } } = \alpha _ { 1 } \cdot { \mathrm { C l a r i t y } } + \alpha _ { 2 } \cdot { \mathrm { C o n t e n t } } + \alpha _ { 3 } \cdot { \mathrm { A e s t h e t i c s } } + \alpha _ { 4 } \cdot { \mathrm { R e l e v a n c e } } ,$ (24)

$\mathrm { s u b j e c t \ t o } \quad \alpha _ { 1 } + \alpha _ { 2 } + \alpha _ { 3 } + \alpha _ { 4 } = 1 .$ (25)

In our experiments, we set $\alpha _ { 1 } = 0 . 0 5$ , $\alpha _ { 2 } = 0 . 1 0$ , $\alpha _ { 3 } = 0 . 1 0$ , and $\alpha _ { 4 } = 0 . 7 5$ . This configuration explicitly prioritizes Relevance, reflecting that consistency with the actual IP identity is the most critical criterion in FactIP Benchmark. Such a design is also aligned with our emphasis on broad world knowledge utilization, since accurately preserving the target IP requires the model to recognize and faithfully express knowledge-intensive identity cues rather than merely producing visually plausible content. Meanwhile, Clarity, Content, and Aesthetics are retained as important auxiliary dimensions to ensure that the final metric also captures perceptual quality and semantic completeness. Finally, the weighted overall score is multiplied by 10, so that the final score range is normalized to [0, 100], where a higher score indicates better IP fidelity and overall generation quality. More Evaluations showcases can be found in G.

# D More Details about FactIP benchmark

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/72395440c872f0583ebb9d2ff89862fd680d729995a7b1453a76d143ba524ce0.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e12c91fce1e64f0ef820b2ee3d5c347ae303d67f3af0e01b9942f4c2ef81f549.jpg]]  
Figure 7 | (a) Hierarchical category distribution of FactIP Bench, consisting of three major groups (Character, Scene, and Object) and 12 fine-grained subcategories. (b) Category-wise comparison of different methods on FactIP Bench, where the radar chart presents the overall scores across all subcategories.

# D.1 Construction

To rigorously evaluate the capability of generative models in synthesizing high-fidelity intellectual property (IP) and culturally significant concepts, we construct the FactIP benchmark. As illustrated in Figure 7(a), the benchmark is meticulously organized into a hierarchical taxonomy. It encompasses three overarching domains—Character, Object, and Scene—which are further decomposed into fine-grained subcategories. This comprehensive structure ensures a diverse and representative evaluation of a model’s factual knowledge grounding. For a detailed statistical breakdown and specific

Table 7 | Detailed category statistics of our benchmark. The full benchmark contains three major categories and 12 fine-grained subcategories, totaling 2,462 prompts.   

<table><tr><td>Category</td><td>Subcategory</td><td>Description</td><td>Num</td></tr><tr><td rowspan="6">CHARACTER</td><td>Animation</td><td>Animated characters, creatures, equipment, and iconic locations from anime and animated media.</td><td>438</td></tr><tr><td>Comic</td><td>Characters and visual elements originating from comic books and manga series.</td><td>363</td></tr><tr><td>Celebrity</td><td>Prominent figures across diverse domains, including scientists, political leaders, business executives, athletes, and entertainment personalities.</td><td>300</td></tr><tr><td>Game</td><td>Video game characters, weapons, equipment, and other in-game visual elements.</td><td>272</td></tr><tr><td>Mascot</td><td>OfficialASCots representing Olympic Games, regional events, and corporate brands.</td><td>77</td></tr><tr><td>Mythology</td><td>Universally recognized mythological narratives and legendary figures, e.g., Kuafu Chasing the Sun.</td><td>50</td></tr><tr><td rowspan="4">OBJECT</td><td>Food</td><td>Cuisines, regional delicacies, desserts, and beverages with cultural significance.</td><td>316</td></tr><tr><td>Cultural Relic / Art</td><td>National treasures, classical calligraphy, paintings, sculptures, and fine art pieces.</td><td>126</td></tr><tr><td>Toy</td><td>Collectible figures, designer toys, and model kits with cultural relevance, e.g., Labubu.</td><td>123</td></tr><tr><td>Animal / Plant</td><td>Individually notable animals and plants with distinct public recognition, e.g., Giant Panda Qizai.</td><td>50</td></tr><tr><td rowspan="2">SCENE</td><td>Landmark</td><td>Renowned scenic spots, architectural landmarks, monuments, and heritage sites.</td><td>297</td></tr><tr><td>Festival / Celebration</td><td>Visual elements and symbols associated with well-known festivals and cultural celebrations.</td><td>50</td></tr></table>

descriptions of each subcategory, please refer to Table 7. In total, the dataset comprises 2,462 meticulously curated prompts, challenging models to generate recognizable entities ranging from global celebrities and intricate art toys to iconic landmarks. To facilitate efficient and accessible model evaluation, we additionally construct a lightweight version, FactIP-Mini, by sampling 500 prompts from the full dataset while strictly preserving the original category distribution.

# D.2 Results

We present a comprehensive performance analysis of various leading generative models on the FactIP benchmark, with all reported results evaluated on the representative FactIP-Mini dataset. Figure 7(b) provides a visual category-wise comparison, highlighting the relative strengths and weaknesses of different architectures across the primary evaluation dimensions. The radar chart succinctly captures the overarching performance trends, revealing that Unified MLLMs and state-of-the-art commercial models generally exhibit superior IP fidelity and factual consistency compared to earlier generation-only paradigms. A detailed, quantitative breakdown of the per-subtask overall scores is provided in Table 8. These fine-grained results emphasize the necessity of advanced multimodal reasoning and extensive factual grounding to accurately render domain-specific visual characteristics. Notably, our proposed Unify-Agent demonstrates highly competitive and balanced performance across all taxonomic branches, underscoring the efficacy of its unified architectural design in handling complex, knowledge-intensive visual generation tasks.

# E Prompt

# E.1 System Prompt

# SYSTEM_PROMPT

You are helping to build a high-quality visual generation dataset.

Your task is to gather information and reference images for creating detailed image descriptions.

Your Goal: Create a detailed recaption for "{image_prompt}" about the IP "{ip_name}" ({country}).

Table 8 | Per-subtask overall scores on the FactIP benchmark. Character includes celebrity, animation, game, comic, mythology, and mascot prompts; Object includes animals/plants, food, cultural relic/art, and toy prompts; Scene includes landmark and celebration prompts. denotes the best score and the second-best within each group.   

<table><tr><td rowspan="2">Model</td><td colspan="6">Character</td><td colspan="4">Object</td><td colspan="2">Scene</td><td rowspan="2">Overall</td></tr><tr><td>Celebrity</td><td>Animation</td><td>Game</td><td>Comic</td><td>Mythology</td><td>Mascot</td><td>Animal</td><td>Food</td><td>Art</td><td>Toy</td><td>Landmark</td><td>Festival</td></tr><tr><td colspan="14">Commercial Models</td></tr><tr><td>GPT-Image-1</td><td>77.0</td><td>63.1</td><td>66.4</td><td>60.3</td><td>95.7</td><td>64.7</td><td>89.0</td><td>88.6</td><td>40.9</td><td>52.2</td><td>76.1</td><td>94.2</td><td>69.3</td></tr><tr><td>GPT-Image-1.5</td><td>77.7</td><td>64.0</td><td>64.1</td><td>61.9</td><td>93.0</td><td>69.4</td><td>88.1</td><td>92.1</td><td>42.8</td><td>48.9</td><td>76.0</td><td>95.4</td><td>69.9</td></tr><tr><td>Seeddream-4.5</td><td>86.0</td><td>75.6</td><td>82.3</td><td>80.8</td><td>88.9</td><td>88.1</td><td>77.0</td><td>92.4</td><td>64.8</td><td>83.0</td><td>83.5</td><td>88.9</td><td>82.0</td></tr><tr><td>Seeddream-4</td><td>86.8</td><td>76.1</td><td>85.3</td><td>81.0</td><td>88.9</td><td>85.2</td><td>80.1</td><td>91.4</td><td>70.2</td><td>83.8</td><td>85.8</td><td>87.5</td><td>83.0</td></tr><tr><td>Seeddream-5</td><td>87.5</td><td>85.0</td><td>89.5</td><td>86.8</td><td>87.8</td><td>90.9</td><td>81.5</td><td>96.0</td><td>77.5</td><td>89.5</td><td>88.8</td><td>82.8</td><td>87.3</td></tr><tr><td>Qwen-Image-2.0-Pro</td><td>68.8</td><td>64.5</td><td>71.9</td><td>64.6</td><td>80.5</td><td>74.3</td><td>75.5</td><td>82.7</td><td>65.0</td><td>79.3</td><td>80.2</td><td>84.0</td><td>71.1</td></tr><tr><td>Nano Banana</td><td>48.8</td><td>53.2</td><td>57.8</td><td>49.0</td><td>70.0</td><td>65.5</td><td>49.4</td><td>64.2</td><td>85.2</td><td>69.1</td><td>59.4</td><td>56.7</td><td>56.6</td></tr><tr><td>Nano Banana-Pro</td><td>75.0</td><td>61.4</td><td>62.9</td><td>54.6</td><td>97.0</td><td>63.6</td><td>87.0</td><td>89.9</td><td>43.8</td><td>46.3</td><td>72.3</td><td>90.3</td><td>66.7</td></tr><tr><td>Nano Banana-2</td><td>92.5</td><td>82.6</td><td>89.0</td><td>83.7</td><td>95.2</td><td>89.1</td><td>88.1</td><td>94.8</td><td>86.9</td><td>94.8</td><td>88.3</td><td>96.4</td><td>88.5</td></tr><tr><td colspan="14">Generation Only</td></tr><tr><td>Pixel-Art-XL</td><td>24.7</td><td>5.6</td><td>12.1</td><td>12.1</td><td>19.8</td><td>5.8</td><td>25.9</td><td>1.4</td><td>2.2</td><td>5.0</td><td>6.8</td><td>26.1</td><td>12.1</td></tr><tr><td>FLUX.1-schnell</td><td>26.0</td><td>16.8</td><td>25.9</td><td>30.1</td><td>50.0</td><td>17.4</td><td>49.6</td><td>11.8</td><td>12.1</td><td>18.9</td><td>19.7</td><td>54.9</td><td>24.0</td></tr><tr><td>SD-3-medium</td><td>33.7</td><td>17.8</td><td>23.8</td><td>31.5</td><td>49.1</td><td>18.7</td><td>49.0</td><td>13.1</td><td>11.3</td><td>16.6</td><td>19.2</td><td>58.9</td><td>25.7</td></tr><tr><td>SDXL-base-0.9</td><td>38.2</td><td>18.7</td><td>27.5</td><td>29.9</td><td>40.8</td><td>15.3</td><td>43.2</td><td>11.5</td><td>11.8</td><td>17.4</td><td>18.7</td><td>60.1</td><td>26.5</td></tr><tr><td>SD-3.5-medium</td><td>33.4</td><td>19.7</td><td>27.2</td><td>32.9</td><td>50.7</td><td>16.2</td><td>47.0</td><td>8.8</td><td>11.1</td><td>16.8</td><td>21.2</td><td>64.3</td><td>26.5</td></tr><tr><td>SD-3.5-large</td><td>35.6</td><td>19.8</td><td>30.4</td><td>31.3</td><td>45.7</td><td>24.6</td><td>42.8</td><td>13.7</td><td>12.3</td><td>17.8</td><td>21.0</td><td>66.7</td><td>27.5</td></tr><tr><td>FLUX.1-dev</td><td>36.1</td><td>21.4</td><td>28.1</td><td>31.9</td><td>57.7</td><td>22.6</td><td>50.9</td><td>18.0</td><td>15.8</td><td>20.9</td><td>23.2</td><td>58.8</td><td>28.9</td></tr><tr><td>Z-Image</td><td>49.6</td><td>52.7</td><td>54.7</td><td>51.3</td><td>67.6</td><td>58.1</td><td>77.3</td><td>75.4</td><td>32.1</td><td>34.9</td><td>62.4</td><td>78.2</td><td>54.2</td></tr><tr><td>FLUX.2-dev</td><td>54.2</td><td>52.9</td><td>55.8</td><td>52.2</td><td>90.4</td><td>62.9</td><td>75.3</td><td>74.0</td><td>33.7</td><td>38.6</td><td>61.9</td><td>81.9</td><td>56.3</td></tr><tr><td>Qwen-Image</td><td>52.5</td><td>56.3</td><td>52.2</td><td>50.8</td><td>79.0</td><td>62.5</td><td>68.8</td><td>69.2</td><td>33.9</td><td>37.9</td><td>65.8</td><td>73.3</td><td>55.4</td></tr><tr><td colspan="14">Unified MLLM</td></tr><tr><td>Janus-1.3B</td><td>16.4</td><td>18.4</td><td>23.5</td><td>26.2</td><td>25.7</td><td>13.3</td><td>38.6</td><td>13.8</td><td>3.8</td><td>9.6</td><td>20.2</td><td>33.7</td><td>19.3</td></tr><tr><td>Janus-Pro</td><td>23.0</td><td>31.6</td><td>33.1</td><td>35.5</td><td>61.1</td><td>27.3</td><td>43.5</td><td>39.1</td><td>9.3</td><td>16.7</td><td>30.3</td><td>45.7</td><td>30.3</td></tr><tr><td>Emu3</td><td>30.6</td><td>28.6</td><td>31.7</td><td>33.8</td><td>36.5</td><td>21.3</td><td>45.6</td><td>26.1</td><td>15.5</td><td>21.3</td><td>30.9</td><td>49.0</td><td>30.0</td></tr><tr><td>Emu3.5</td><td>59.2</td><td>54.1</td><td>58.7</td><td>52.7</td><td>76.0</td><td>57.7</td><td>74.2</td><td>76.2</td><td>28.7</td><td>34.1</td><td>61.3</td><td>87.0</td><td>57.2</td></tr><tr><td>Hunyuan-Image-3.0</td><td>47.4</td><td>53.2</td><td>47.6</td><td>48.6</td><td>82.5</td><td>54.5</td><td>70.5</td><td>78.8</td><td>33.1</td><td>36.0</td><td>64.2</td><td>81.5</td><td>53.4</td></tr><tr><td>Echo4o</td><td>44.9</td><td>48.1</td><td>43.3</td><td>43.5</td><td>66.5</td><td>52.1</td><td>60.5</td><td>62.2</td><td>27.7</td><td>30.3</td><td>55.1</td><td>70.0</td><td>47.3</td></tr><tr><td>Bagel</td><td>46.1</td><td>53.7</td><td>46.5</td><td>48.2</td><td>63.3</td><td>54.3</td><td>66.6</td><td>76.5</td><td>31.6</td><td>33.5</td><td>53.5</td><td>75.2</td><td>50.9</td></tr><tr><td>Bagel-CoT</td><td>44.7</td><td>48.0</td><td>44.6</td><td>46.0</td><td>62.0</td><td>43.7</td><td>58.2</td><td>63.0</td><td>27.0</td><td>30.8</td><td>50.5</td><td>72.8</td><td>47.0</td></tr><tr><td>Unify-Agent (Ours)</td><td>71.9</td><td>71.7</td><td>68.0</td><td>69.0</td><td>76.5</td><td>75.7</td><td>75.5</td><td>80.2</td><td>65.3</td><td>76.3</td><td>70.9</td><td>74.0</td><td>71.7</td></tr></table>

Natural Workflow (think step by step):

1. First, search for background information about this IP/character to understand who they are, their characteristics, style, and context. This knowledge will help you craft more accurate image search queries.   
2. Then, search for reference images of this IP/character. Good reference visuals are essential for the final detailed description.   
3. Finally, I will provide you with the downloaded reference images, and you will generate a detailed <recaption> that references "image_1" and "image_2" specifically.

```txt
Tool Call Format (IMPORTANT - use ONLY this format): <tool_call> {{"name": "tool_name", "arguments": {{"param1": "value1"}}}} </tool_call> Examples: - Text search: <tool_call> {{"name": "text_search", "arguments": {{"q": "search_query", "hl": "zh", "top_k": 5}}} </tool_call> - Image search: <tool_call> {{"name": "search_image", "arguments": {{"q": "image_query", "hl": "zh", "num": 8}}} </tool_call> 
```

# TOOLS_DEFINITION

tools $= [$ { "type": "function", "function": { name": "text_search", "description": "Search the web for text information about a query. Use this to get background information about IPs, people, places, or topics.", "parameters": { "type": "object", "properties": { "q": { "type": "string", "description": "Search query" }, "hl": { "type": "string", "description": "Language code (e.g., 'en', 'zh')"), "default": "en" }, "top_k": { "type": "integer", "description": "Number of results to return", "default": 5 } }, "required": ["q"] } }   
},   
{ "type": "function", "function": { "name": "search_image", "description": "Search for images on the web. Use this to find relevant images about the IP or topic.", "parameters": { "type": "object", "properties": { "q": { "type": "string", "description": "Search query for images" }, "location": { type": "string", "description": "Location for search", default": United States" }, "hl": { "type": "string", "description": "Language code", default": en" }, "num": { type": "integer",

```txt
"description": "Number of images to return", "default": 8 } }， "required": ["q"] } } } 
```

# E.2 Evluation Prompt

# EVALUATION_PROMPT

You are an image evaluation assistant. Compare AS (assistant-generated image) against GT1 and GT2 (ground-truth images) given the Prompt.

Your task is to return exactly 6 fields: 5 integer scores (0-10) and 1 rationale string.

Evaluate these 5 dimensions independently:

1. "clarity": image sharpness, absence of blur/artifacts/noise, and richness of visible details.   
2. "content_quality": faithfulness to the Prompt, subject completeness, and semantic coherence.   
3. "aesthetics": visual appeal, composition, lighting/color harmony, and style consistency.   
4. "text_relevance_ip": IP identity consistency. This means whether AS preserves the same character/object/IP identity as GT1/GT2 based on distinctive traits (e.g. face, hairstyle, costume, colors, species/object-defining features). Do NOT require exact matching of pose, background, camera angle, or composition.

Important instructions:

- Use GT1 and GT2 jointly to infer the stable identity and attributes of the IP.   
- Do not penalize AS for differences that also vary between GT1 and GT2.   
- Score each dimension independently before assigning the overall score.   
- All five score fields must be integers from 0 to 10.   
- "rationale" must be a single short string with at least two concrete evidence points, separated by semicolons.

You MUST respond with ONLY this exact JSON structure, with all 6 keys present and no extra keys:

```jsonl
{"clarity": 7, "content_quality": 8, "aesthetics": 7, "text_relevance_ip": 8, "rationale": "Evidence 1; Evidence 2"} 
```

Do not use markdown fences.

Do not output any text before or after the JSON.

If uncertain, still output the best-effort JSON with all required keys.

# E.3 Judge Prompt

# JUDGE_PROMPT

You are an expert Image Quality Assessor for an AI training pipeline. Your task is to evaluate whether a downloaded image is a high-quality visual reference for a specific Intellectual Property (IP).

You must rate the image on a scale of 0 to 10 based on the following strict rubrics:

# ### 1. IP Consistency (Critical)

- Pass: The image clearly depicts the specific character/object requested.   
- Fail (Score 0): The image shows a completely different character, a landscape, a real person (cosplay) if the IP is anime, or an unrelated object.

# ### 2. Layout & Composition

- Subject-Centric: The IP subject must be the main focus.   
- Face Visibility: For characters, the face must be clearly visible (preferably front-facing or 3/4 view). REJECT if the face is tiny, distant, or back-facing.   
- No Text-Heavy: REJECT images that are primarily movie posters with large text overlays, book covers, or infographics where the subject is obscured by text.   
- No Collages: REJECT split-screens, manga panels, or multiple images stitched together. Single-scene images only.

# ### 3. Visual Quality

- Clarity: The image must be sharp. REJECT if it is severely blurry, pixelated, or has heavy jpeg compression artifacts.

# ### 4. Watermarks & Obstructions

- Reject: Large, obstructive watermarks covering the face or main body (e.g., full-screen stock photo watermarks).   
- Accept: Small, unobtrusive logos in the corner are acceptable but lower the score slightly.

# ### Scoring Guide:

- 0: Wrong IP or completely unusable.   
- 1-5 (Reject): Correct IP but violates a major rule (Blurry, Huge Watermark, Text-Heavy, Collage, Tiny Face).   
- 6-7 (Borderline): Usable but not ideal (Side profile, small corner watermark, medium resolution).   
- 8-10 (Excellent): Perfect reference (High-res, clear front-facing, no text, clean).

# JUDGE_Question

Please evaluate the provided image for the IP: "{ip_name}".

Assess the image based on the system rubrics.

Return your response in JSON format ONLY with the following structure: {{

"score": <int, 0-10>,

```javascript
"reason": "<string, a concise explanation of the score, mentioning any specific flaws like 'text-heavy', 'watermark', or 'blurry'>", "is_text_heavy": bool>, "has_watermark": bool>} }} 
```

# E.4 Prompt-Generation Prompt

# PROMPT-GENERATION_PROMPT

```txt
You are a specialized Image Generation Prompt Expert. Your mission is to generate a single, high-quality image generation prompt based on a specific IP (person/character). You must describe "Who is doing What in Which scene." 
```

## 1. Core Logic & Language Rules (CRITICAL)

Step 1: Determine the IP's Nationality

* Case A: If the IP is Chinese (Mainland, Hong Kong, Taiwan, Macau):

$\star$ Prompt Language: Must be Chinese.   
$\star$ Tag Language: Must be Chinese.   
* Language Code: `zh`

* Case B: If the IP is NOT Chinese (USA, UK, Japan, Korea, Europe, etc.):

$\star$ Prompt Language: Must be English.   
$\star$ Tag Language: Must be English.   
* Language Code: `en`

## 2. Prompt Construction Requirements

1. Content Elements:

* Subject: The IP's name (and brief identity if needed for context).   
* Scene: A specific, realistic location fitting the IP's profession (e.g., Office, Studio, Stadium, Stage, Cafe, Street).   
$\star$ Action: A dynamic verb describing what they are doing (e.g., Interviewing, Singing, coding, running, drinking coffee).

2. Realism Constraint:

$\star$ The scene and action must align with the IP's public persona or profession.   
* Do not hallucinate impossible scenarios unless the IP is a fictional fantasy character.

3. Syntactic Diversity (Avoid Repetition):

$\star$ Do not always use the structure "Name is doing X in Y".   
* Vary your sentence structures:

* *Scene-first*: "In the [Scene], [Name] is [Action]..."   
* *Action-focused*: "[Name] is [Action] while located in [Scene]..."   
* *Descriptive*: "Surrounded by [Context], [Name] is [Action]..."

## 3. Output Format

You must output ONLY the XML tags below. Do not output markdown code blocks (like ```xml), explanations, or conversational filler.

<Image_Prompt>The full descriptive sentence</Image_Prompt>

```xml
<Tag_Name>Category of the action (e.g., Interview, Speech, Daily Life)</Tag_Name>
<Language>zh OR en</Language>
*Note on <Tag_Name>*: If Language is `zh`, the tag must be Chinese. If Language is `en`, the tag must be English (e.g., "Hosting", "Street Snap"). REMEMBER: Output ONLY the three XML tags above, nothing else.
## 4. Examples
Input: Robert Downey Jr. (American Actor)
Output:
<Image_Prompt>Sitting in a relaxed pose on a Hollywood talk show set, Robert Downey Jr. is laughing while telling a story</Image_Prompt>
<Tag_Name>Interview</Tag_Name>
<Language>en</Language>
Input: Gordon Ramsay (British Chef)
Output:
<Image_Prompt>Gordon Ramsay is carefully plating a gourmet dish in a busy high-end restaurant kitchen</Image_Prompt>
<Tag_Name>Cooking</Tag_Name>
<Language>en</Language> 
```

# E.5 Recaption Prompt

# RECAPTION_PROMPT

You are a professional visual language reasoning assistant. Your task is to generate a reasoning process and a final detailed image description based on two reference images (image_1, image_2), the original instruction, and text search results.

# Input Information

1. Reference Images: Explicitly labeled Reference Image 1 (refer to as image_1) and Reference Image 2 (refer to as image_2).   
2. Original Instruction: The user's short request.   
3. Background Info: Text information from previous search steps.

# Output Format

Strictly follow XML format:

<think>

[Deep reasoning here:

1. Analyze visual features of image_1 and image_2.   
2. Combine with background info to plan the fusion.   
3. Explicitly state what comes from image_1 and what comes from image_2.]

</think>

<recaption>

[The final detailed image description, including

"Scene Description" and "Preservation Statement"] </recaption>

Core Rules & Constraints

1. Reference Principle: You MUST strictly use "image_1" and "image_2" to refer to the images. DO NOT use vague terms like "the first image" or "reference picture".   
2. Descriptive Style: The content of <Instruction> must be a description of the final result (Descriptive), NOT an editing command (Imperative).

- BAD: "Please put the man from image_1 on the left..."   
- GOOD: "In this realistic outdoor portrait, the man from image_1 stands on the left..."

3. Preservation Statement: At the end of the description, you MUST explicitly state what specifically is preserved from image_1 and image_2.

- Must include phrases like: "The final image completely preserves [features] from image_1..." and "The final image fully retains [features] from image_2...".   
- Facial Features Preservation (CRITICAL): If image_1 and/or image_2 contain identifiable persons, you MUST preserve their exact facial features as shown in the reference images. Include detailed descriptions such as: face shape, eyebrow shape, eye characteristics, facial expression, skin tone/complexion, hairstyle, and any distinctive facial features. Use the format: "Preserve the exact facial features of [person name/description] as shown in image_1 and image_2: [detailed facial feature description]. Maintain [their/his/her] [appearance/clothing/style] as referenced in both images."

4. Language Consistency (MANDATORY): The content of <recaption> MUST be written entirely in the SAME language as the original instruction. If the original instruction is in Chinese, ALL descriptions in <recaption> must be in Chinese - NO English allowed. If the original instruction is in English, use English throughout. NEVER mix languages within the same description.

Start the task. Output <think> and <recaption>.

# E.6 Summary Prompt

# SUMMARY_PROMPT

Based on the following webpage content, provide a concise summary that is relevant to the query: "{query}"

Webpage Title: {title} Content: {content[:2000]}

Please provide a focused summary (2-3 sentences maximum) that directly addresses the query. Focus on the most relevant information.

You MUST format your response using the following structure: <think> [Your thinking process about what information is most relevant to the query] </think>

<response> [Your concise summary here - 2-3 sentences maximum] </response>

# F Image Generation Showcases

Case 1: Science - Copper Combustion This case details the generation of a scientifically accurate image for the knowledge-based prompt The copper is burning, highlighting the color. Recognizing the specific chemical knowledge required, the model first initiated a text search to verify the characteristic flame test reaction of copper, confirming it produces a distinctive green glow. To capture the nuanced visual dynamics, it then performed a targeted image search for "copper wire burning green flame." Through a rigorous image judgment process, the model filtered out low-quality or irrelevant references, selecting high-quality images that showcased both an intense emerald-green core and organic flame tendrils. During the refined recaptioning stage, the model synthesized these visual elements—integrating the luminous intensity of copper(II) compounds with the fluid motion of turquoise-green flames—while adding professional photographic descriptors, providing precise guidance for the generation, accurately depicting the mesmerizing chemical phenomenon. See Fig. 8 for details.

Case 2: Art Toy - DUDOO Tasty Afternoon Tea Series This case details the image generation process for the DUDOO Tasty Afternoon Tea Series. The model initially performed a round of text search to establish the general background of the series; however, as the first round yielded only high-level information about the product line, a second round was conducted to specifically probe for the visual identity and iconic character traits of "Dudu." This iterative search strategy compensated for the initial information gap, ensuring a comprehensive understanding of the IP’s design elements before sourcing reference images. By integrating these refined character details with visual cues from the retrieved images, the model synthesized an IP-accurate recaption that vividly depicts the "popcorn bucket" theme and matte resin texture. See Fig. 9 for details.

Case 3: Celebrity - William Butler Yeats This case details the generation of an image for a prompt depicting the poet William Butler Yeats in his rustic stone cottage study. The model first initiated a text search to gather precise biographical and physical descriptions, confirming iconic details such as his signature pince-nez glasses and historical background. It then performed a targeted image search to retrieve authentic historical portraits, capturing his distinct facial structure across different ages. By integrating these verified visual characteristics—such as his angular bone structure and tousled hair—into a refined recaption, the model was able to situate the historically accurate figure within the specific atmospheric setting of a manuscript-writing scene. See Fig. 10 for details.

Case 4: Celebrity - Grigory Perelman This case details the image generation process for the legendary mathematician Grigori Perelman. The model initially performed a text search to establish his reclusive persona and historical context in St. Petersburg, providing the thematic foundation for the "sparsely furnished apartment" scene. It then conducted a targeted image search to capture his highly distinctive "bohemian scholarly" appearance, characterized by wild, untamed hair and a thick, flowing beard. Through the image judgement and refined recaptioning stages, the model meticulously analyzed his facial features across different ages—specifically noting the transition from dark to gray-streaked hair—and integrated these traits with atmospheric details like scattered papers and soft natural light. This strategy ensured the final generation achieved profound biographical accuracy while vividly portraying the intense focus of a brilliant mind. See Fig. 11 for details.

Case 5: Celebrity - Bruce Beutler This expert-focused case details the image generation process for Nobel laureate Bruce Beutler in an advanced laboratory setting. The model initially performed a text search to establish his professional identity as a Nobel Prize-winning immunologist and geneticist at UT Southwestern, ensuring the generated environment was contextually authentic. It then sourced reference portraits through image search and applied a rigorous image judgement process to filter for his most representative facial features, such as his receding gray-silver hair and distinctive mustache. By synthesizing these verified biographical details with the technical requirements of the scene in a refined recaption, the model achieved high accuracy in both persona and atmospheric realism. See Fig. 12 for details.

Image Prompt  The copper is burning, highlighting the color.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/871338c90860ef0f89fe547b1a288824f38913361aa30365785e0e6809709ae5.jpg]]

# Text Search

# Model Responses

I'll help you create a detailed image based on the knowledge prompt "The copper is burning, highlighting the color".

Let me start by gathering background information about copper burning and its visual characteristics.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/9bb9de2903695d6a29ab55c627ce99dea763ff3f25be612e9d1cd0e2876b0364.jpg]]

Tool Use <tool_call>{"name": "text_search", "arguments": {"q": "copper wire burning flame color reaction","hl": "en",

"top_k": 2}}</tool_call>

Tool Output

<observation> [1] SOS. Does copper wire give the same flame test color as ... - Reddit

Yes, it would give a green color

[2] The Copper Flame Test - YouTube ...

[3] What happens when you burn copper instead of carbon? - YouTube ...

</observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2f8790462198ec4af97058cf3e20d6d9c294bc7163dcd6db536f37025da2d692.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/04c48a03203a332d2dbd69ed75548a58a4fccff6b01a93060889f666a518b844.jpg]]

# Image Search

# Model Responses

Based on the  text search, I learned that burning copper wire produces a green flame color. Let me search for reference

images that capture this visual characteristic.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/544d30793881af4745fffb5d0afce059564cc3e50e1876442c0399a6c2af8384.jpg]]

Tool Use    <tool_call>{"name": "search_image", "arguments": {"q": "copper wire burning green flame color reaction", "hl": "en",

"num": 5}}</tool_call>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c920ec68686e5a0e71f694a6613fa76d9b99b8c4df2ca704a967244198d4a345.jpg]]

Tool Output <observation> [1] Make Green Fire or Flames

[2] How To Make Green Flames Using Copper Sulfate

[4] Colorful Copper Chemistry around the Campfire | Chemical ...

[4] File: Copper flame test.JPG - Wikimedia Commons

</observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/d052fdf8ca5ed908dd1d9e3c02a0c5ceb1a884193194b71f0d5ba38ecb6fdc0c.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/09d8cd2168ff4183d224ddbf0f07bbfa162ccea0ce6cf6babcd7444c611e9598.jpg]]

# Reference Images

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/9af517a019c49f9e984cac01dcb5950191aa8abfe2f4c3e6576921bdf643c6c8.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/cf04b5041b573d0e2855fb387ab5864e6dbbfc9f60f1c01259f14c7808028b1e.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b09c4b2edce3455e9bba8ae9db9c23b254bfe321d4998aafb0158f7e1e43bc2c.jpg]]

# Image Judgement

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/fd08c3d4c9d14bf88202bb0f5a8dd487e20685a01bd8d49ec397eeb199fd11da.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/611c920a7a52a1e9ab651f62db5e2f0ff3d006ad3f52a9bb2a3a830a25471bc5.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c712044999e43fbbffb60c7dad9ddb03e556a9f29661e8c5e3ce8ce140794a27.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/18ef5d6de65e4854dabeee41060f56d07debd88db2b599629a655be2a29f8d16.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/814e294fdaedc11a6bcdd9dcfd6effc0518c01b1c4053631a7af66b997932622.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0fff6f15c95f565901892da84ab32507d2ae12e7aea98b50f69ae7886b9f1adc.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/93e4b04b1e8ddbc5b63cbda2391e29b2da38ddeb320955bdf9fc745e01600b64.jpg]]

# Refined Recaption

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ffc13e80ff19cdf37d84503c78111bad0746cdf50042c2637e8392a688c72e1b.jpg]]

Thinking <think>Analyzing image_1: This image shows a scientific flame test demonstration of burning copper. The flame displays a stinctive bright green color rising from a dark, shallow dish or saucer containing what appears to be copper(II) nitrate or a similar compound...

Analyzing image_2: This image shows a more casual experimental setup for creating green flames. There's a visible flame with a bright turquoise-green color burning from what appears to be a container with a dark, possibly charred or oxidized surface...

Integration strategy: I will combine the distinctive green flame characteristics from both images - the intense luminous green core from image_1 and the dynamic flame tendrils from image_2 - to create a detailed description of burning copper that emphasizes the unique green coloration and the chemical phenomenon behind it...</think>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/4b1b404b2aeda9f0d9117094cad8c673945d1edd0d68116f671d88430d29515b.jpg]]

Recaption <recaption>A close-up scientific photograph capturing the mesmerizing moment of copper wire combustion, showcasing the istinctive luminous green flame characteristic of copper(II) compounds. The flame rises dramatically from a dark, shallow metallic saucer illed with a crystalline copper(II) nitrate compound, as seen in image_1, where the intense emerald-green core radiates with ethereal intensity, ansitioning through yellow-green gradients to a brilliant blue-orange base at the flame's crown. The copper wire at the center emits a vivid rquoise-green glow similar to the wavy flame tendrils visible in image_2, creating multiple dancing streams of light that reach upward in rganic, flame-like patterns ...</recaption>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/7719079d46f2801f603e09bcf64db26c85dfd613966de286294f44ef1ac5716f.jpg]]

# Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/4a9161637472369300322b402e5811c02deae8d30ad7875071db7bc555057154.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/336a01087b5af1a7695941a8ed8dcf34c66ce5bf962b7bc97b64086774a937be.jpg]]

# Generated Images

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e814f477fc5433ebe1a22a2b972389c1d1fecefeecccfe1c2e7e4bd0e8c7a684.jpg]]  
Figure 8 | Image generated for the prompt: "The copper is burning, highlighting the color"

Image Prompt TNT SPACE出品的嘟嘟（Dudu）美味下午茶系列盲盒⼿办微距摄影，造型为⼀只被设计成爆⽶花桶的嘟嘟，顶部满溢着逼真的爆⽶花，具有极其精致的哑光树脂涂装和丰富的层次细节，表现出⻝物的质感和公仔的分量感，治愈系柔和影棚光，背景是虚化的温馨桌⾯场景

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2d2ec8a629587fc8e1f211834ea6f3d00e516cf5cab9d1e77adf2dd433591dad.jpg]]

# Text Search-turn 1

Model Responses  I'll help you gather information and reference images for this IP. Let me start by searching for background information about "DUDOO Tasty Afternoon Tea Series".

Tool Use <tool_call>{"name": "text_search", "arguments": {"q": "DUDOO 嘟嘟 美味下午茶系列 盲盒 ⼿办 TNT SPACE","hl": "zh", "top_k": 2}}</tool_call>  
$\mathscr P$ Tool Output <observation> [1] Some afternoon just need some sweetness A bit of joy, a ... - Instagram...   
[2]【tnt】DUDOO美味下午茶系列⼿办盲盒新品⾸发潮流玩具 - Taobao...

The DUDOO 嘟嘟美味下午茶 series blind box figurine from TNT SPACE is a newly launched trendy toy... </observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/f0a94eb999bafd84ebae4b0dff601060a1c43dd7ea936571506250f951841a4e.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e2d363017f6233532bf32e4ce7653ce3d970f5e8ed0576566838bd4bd8d976bd.jpg]]

# Text Search-turn 2

Model Responses Based on the text search, I learned that DUDOO 嘟嘟 is a character from TNT SPACE, and the "美味下午茶系列" (Tasty Afternoon Tea Series) is a blind box figure line. Let me search for more specific information about this character first.

Tool Use <tool_call>{"name": "search_image", "arguments": {"q": "嘟嘟 Dudu TNT SPACE ⻆⾊ 形象 特点", "hl": "zh", "top_k": 2}}</tool_call>

Tool Output <observation> No relevant web pages found for the query. </observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e3c138c63f6e01159bae2cedf87e10c5b9f994c8fd0ba29c4ba3355a899336a8.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/68cfc17fcbba9519cc1231bc2a2f3a258424caa91c6032e9bafefc119227ff2e.jpg]]

# Image Search

Model Responses Let me search for reference images of this IP character to understand its visual characteristics better.   
Tool Use <tool_call>{"name": "search_image", "arguments": {"q": "TNT SPACE 嘟嘟 Dudu 美味下午茶系列 盲盒⼿办", "hl": "zh", "num":10}}</tool_call>   
$\mathscr P$ Tool Output <observation> [1] 美味下午茶系列| Dudoo - Dudoo - TNT SPACE - 盲盒系列｜千岛  
[2] 美味下午茶系列| Dudoo - Dudoo - TNT SPACE - 盲盒系列｜千岛  
[3] 美味下午茶系列| Dudoo - Dudoo - TNT SPACE - 盲盒系列｜千岛  
[4] 正版DUDOO美味下午茶系列盲盒⼿办，可爱⼜有趣_潮玩盲盒_淘宝游戏⽹  
[5] 美味下午茶系列| Dudoo - Dudoo - TNT SPACE - 盲盒系列｜千岛

... </observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ae6c5f7c8808bf1f473b6ced4f8970616d72f7de2223358021ce78610c7b5333.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/aaeb64d565d2faf431b0e71377c30c74f9ff9c27a243cf006f2edfc5c921687a.jpg]]

# Image Judgement

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/820330ebc10b8dc28b13bb645a1bc47ef14f18ea12aaff6a16a0fc49c7a8fe8e.jpg]]  
Reference Images   
√

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/08db7442b39033fc1bef11405f955fba68724e6e651b5dbdeeff615b74a9e78d.jpg]]  
√

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/f3e16ea23a875f3b2f057db1d63457a049404e9f98e4f8570c92cdf2adb1f717.jpg]]  
×

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/20576ff860711231d1dbaa5ae8cfb2bca9520563cff9f35d60ac512ec9bb52af.jpg]]  
×

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/06ba63e0dcca6f13132abb066d069c2954fb6550e23844eb731508aaba308894.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/7e63886f4f723094da9eb1dee6172468cdcf6abbf89be4f821855bc64c228021.jpg]]

# Refined Recaption

Thinking <think> 从image_1中，我可以看到嘟嘟（Dudoo）⻆⾊的详细特征：这是⼀个可爱的潮玩⻆⾊，拥有蓬松的奶油⾊/⾦⻩⾊奶油状头发，头顶装饰着粉⾊⼩花和⾦⻩⾊的爆⽶花/⾯条造型。⻆⾊有着圆润可爱的脸型，⼤⼤的蓝⾊眼睛带有星星状⾼光，露出⽛⻮的治愈系笑容，⽩皙的肤⾊。⻆⾊坐在⼀个印有"嘟嘟"字样和彩⾊花朵图案的可爱杯形底座中，底座边缘是蓝⾊的，整体呈现出温馨的下午茶氛围。从image_2中，我可以看到嘟嘟的另⼀个造型：⻆⾊穿着经典的红⽩条纹爆⽶花桶造型外套，胸前印有"BEDD"字样和星星装饰。同样保持了标志性的奶油⾊蓬松头发、圆润脸型、⼤眼睛和可爱笑容。这个造型更加突出了爆⽶花桶的主题元素。整合策略：结合两张参考图的视觉特征，描述⼀个爆⽶花桶造型的嘟嘟微距摄影场景，保留⻆⾊的核⼼⾯部特征（奶油⾊蓬松头发、⼤眼睛、治愈系笑容、圆润脸型），同时强调爆⽶花桶造型的精致细节和哑光树脂涂装的质感。</think>

Recaption <recaption> ⼀张TNT SPACE出品的嘟嘟（Dudoo）美味下午茶系列盲盒⼿办微距摄影作品。画⾯中央是⼀只被精⼼设计成爆⽶花桶造型的嘟嘟公仔，完整保留了如image_1和image_2中所示的⻆⾊⾯部特征：圆润饱满的脸型、蓬松柔软的奶油⾊/⾦⻩⾊奶油状头发、⼀双⽔汪汪的⼤眼睛带有星星形状⾼光、露出整⻬⼩⽛⻮的治愈系微笑、⽩皙细腻的肤⾊。⻆⾊身穿如image_2中展示的经典红⽩条纹爆⽶花桶外套，胸前印有可爱的字⺟标识和星星装饰图案。桶顶满溢着逼真的⾦⻩⾊爆⽶花造型，每⼀颗爆⽶花都经过精细雕刻，呈现出真实⻝物的颗粒感和层次感 ...... </recaption>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/9d136359ea8436236e7fe4711d94acfd55f9149a41a40175c0fab2540b2fe3db.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2eda38f285a81921c765a39141d00ce85e786b12d75b34318534695d1ee49b03.jpg]]  
Generated Images   
Figure 9 | Image generated for the "DUDOO" art toy.

Image Prompt Surrounded by stacks of books and papers in his stone cottage study, William Butler Yeats, wearing pince-nez glasses, pauses thoughtfully while writing a manuscript by the light of an oil lamp.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/73da25b188f3da426f2f5b70764d1b34be6846a5f5fd1f681abb4c209eb00c6a.jpg]]

# Text Search

Model Responses I'll help you create a detailed recaption for this image of William Butler Yeats. Let me start by gathering background information about him. Step 1: Searching for background information about William Butler Yeats

Tool Use <tool_call>{"name": "text_search", "arguments": {"q": "William Butler Yeats Irish poet", "hl": "en", "top_k": 5}}</tool_call>

Tool Output <observation> [1] W. B. Yeats. The partial webpage content provided only includes the navigation interface of the Wikipedia entry for W.B. Yeats, with no relevant details about his biography ...

[2] William Butler Yeats | The Poetry Foundation

[3] William Butler Yeats | Irish Poet, Nobel Laureate & Dramatist ...   
[4] William Butler Yeats – Biographical\nWilliam Butler Yeats was a renowned Irish poet who received the 1923 Nobel Prize in Literature ...   
[5] W. B. Yeats 1865–1939. The provided Tate webpage snippet only includes one core biographical detail about Irish poet William Butler Yeats ...

</observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e4559ffdb41a38af849062954139b15e4d89f2be33476c22f9f505bf9c04e2e9.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/8929ce3e1aef4aaaffd7a6c627c142eb75f43acf3ae67f29b9bae2fa73c88926.jpg]]

# Image Search

Model Responses Now I have good background information about William Butler  Yeats. Let me search for reference images that capture his appearance, particularly focusing on his characteristic look with pince-nez glasses and his writing environment. Step 2: Searching for reference images of William Butler Yeats

Tool Use <tool_call>{"name": "search_image", "arguments": {"q": "William Butler Yeats portrait pince-nez glasses poet photograph", "hl": "zh", "num":5}}</tool_call>

Tool Output <observation> [1] William Butler Yeats Print (c1900). Art Prints, Posters ...

[2] File:Portrait of William Butler Yeats P5656.jpg - Wikimedia ..   
[3] William Butler Yeats free public domain image | Look and Learn ...   
[4] william-butler-yeats-in-pince-nez-886×1024 – Kate Shrewsday ...   
[5] NPG x6397; W.B. Yeats - Portrait - National Portrait Gallery ...

... </observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/1d3ca9a5cd8bd3af2ec2f434112071f8fab7374975ed40b5377d6a24995366da.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c8fa92534fb69e0039043840056387345d1ea7d0cc3125c40c2eeca11fc6b553.jpg]]

# Image Judgement

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2016a1cfa42218182fedc772da99e49e26b96fdaeaa779f363d9bbf20ce45194.jpg]]  
Reference Images   
× ×

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/280e8e197e52d6377b73b9048efdeb6606f94637a82ac600556971f0519104d5.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2d038c7dff312e4650bf250a8f6a810a541aefa8fbd432674aa05f2b231f577f.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/8da6ee8f66420c51b34ca7a2d54b76c63ff0386c4c5671e163c10af2500eb050.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e1d286b694adec4866b5e0ce7829baeba2af1ec73dde55330eedca2deaf9ec8c.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/bfb99b2657b0c004b7baaaf73013cd065590637746a997c5f5c2a387180b2b46.jpg]]

# Refined Recaption

Thinking <think>Analyzing image_1: This is a painted portrait from 1900 showing a younger William Butler Yeats. He has a slender, elongated face with an angular bone structure...

Analyzing image_2: This is a sepia-toned photograph showing Yeats at a slightly older age. The same distinctive facial features are visible - the elongated face shape, prominent angular bone structure, and deep-set eyes...   
Integration strategy: I will combine the visual characteristics from both images to create a detailed scene of Yeats in his stone cottage study. The facial features - the angular face, deep-set contemplative eyes, signature pince-nez glasses, wavy hair - must be preserved. The setting will include period-appropriate details like oil lamp, books, papers, and his characteristic dark formal attire, all while setting the scene in his intimate writing environment.</think>   
Recaption <recaption> In the warm, amber glow of an oil lamp, William Butler Yeats sits at a worn wooden desk in his rustic stone cottage study, momentarily pausing from his manuscript writing with a quill pen suspended mid-air. Preserve the exact facial features of William Butler Yeats as shown in image_1 and image_2: his distinctive elongated, angular face with prominent bone structure, deep-set contemplative eyes beneath slightly arched dark eyebrows, straight refined nose, thin lips with a subtle pinkish hue, and fair complexion. Maintain his signature pince-nez glasses with delicate wire rims perched on his nose as referenced in both images, along with his characteristic tousled, wavy hair swept back from his forehead...</recaption>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0f4b9deb93bbc6eda48c4d252671931f4df57666ee95d9e8913dc50725f253ad.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/4c411e471584a4f4100bacc0273f84f93de4b7d2d4d3f9f00a11089dc469206b.jpg]]  
Generated Images   
Figure 10 | Image generated for William Butler Yeats writing a manuscript.

Image Prompt  Surrounded by stacks of papers and old books in a sparsely furnished apartment room in St. Petersburg, a heavily bearded Grigori Perelman is deeply engrossed in scribbling complex mathematical equations on a notepad.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/bfa8ccfb18c098b1ea7a0886622c06a42ab7332a7bfe7b39a2be41fa5d1c4c8a.jpg]]

# Text Search

Model Responses I'll help you gather information and reference images for creating a detailed image description about Grigoriy Yakovlevich Perel'man. Let me start by searching for background information

Tool Use <tool_call>{"name": "text_search", "arguments": {"q": "Grigory Yegorovich Perlman, mathematician, biography", "hl": "en", "top_k": 2}}</tool_call>

Tool Output <observation> [1] Grigory Perelman - Wikipedia Grigory Yegorovich Perelman is a Russian mathematician, born on June 13, 1966, in Leningrad, former Soviet Union...

[2] The enigmatic mathematical genius Perelman: Say what you want | CAM-Net Digest... </observation> □

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/4be983989e786ab1641362d94e3cb32219b37806cda7269a1d02971bac14f878.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/76a00e6574b3ba30fd5d8009d418ce5abde3fe86e67283d148fcb39796100ed7.jpg]]

# Image Search

Model Responses Now I have a good understanding of Grigoriy Perel'man - he's a Russian mathematician born in 1966 in Leningrad (st. Petersburg), known for his work on the Poincaré conjecture and other Millennium Prize problems. He's described as a mysterious mathematician who has turned down major awards. Let me search for reference images.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/fa48e54959091fa0ae8515b3163508be43018d11a23bff543571294b51fc741d.jpg]]

Tool Use    <tool_call>{"name": "search_image", "arguments": {"q": "Grigory Perelman, mathematician, Russian, Brazos", "hl": "en", m": 10}}</tool_call>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ec74ad00c4c2a69608c3fc82a074136135d7d598a81481a057348c77ec6a07c1.jpg]]

Tool Output <observation> [1] Grigory Perelman - Wikipedia, the free encyclopedia

[2] A Russian genius who loves numbers more than money - Science & Technology Window - RFI - Radio France Internationale   
[3] [The genius in beggar's clothes] The Russian mathematician who refused a million-dollar prize but remained anonymous...   
[4] Russian mathematician Perelman refuses a $1 million prize - Mysterious Earth Science | Nature...   
[5] Grigory Perelman, the mathematician who refused the Fields Medal and the $1 million Clay Prize.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/eba12aeb0a87f1a958cfdf99517084157281b90f1d383138080474ead9d83a90.jpg]]

</observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0f9d8f8c673d93763146628a4c793e4d2b72aad2c15fc427056a4600b148b8fa.jpg]]

# Image Judgement

# Reference Images

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e0d14a00da0fa893de5def5ba5377a119d660da094d3a1b5569039a3067414cf.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/79774d80331de8f860f5e68ed24a470304842cce38294143378420b70d76a2e1.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/55d75aef5614f859411a3850dc890da58189bdaa09f1604ba22f43d51a129bfe.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/df84c7f110be7fbba1d0c5853225786fcbf848674df4784f6bf9d855387f5706.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ea5d6a056c95e5de8e47bc09507c15c0f6a57bc32724001cd896b70e1abb2e88.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e4ec8b8a065832789e840e1de2ef474785742c139227aaf45938ca563ae29697.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/1c2cb27b4ff986dbfc6e519659ac6f339a75b1d8157634b8551911083e9903cf.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/23b1671cd50eff51ae1ce1a2be9ebf41b643b6ef5030e94a7adcb14719f11a9b.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/3df1c4eae52bb193896df1b36b451ae569b5185921b7ce1230ffb00ff61b6b3e.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b2e03ca1b87a08659f7f2ee649267cead47fa8dde7b5164a24c4a538645ac39d.jpg]]

# Refined Recaption

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/7ff2242293943944b414fdfc603552cb0b1d1c38f4edb702d2b1094bbac5438d.jpg]]

Thinking <think> Analyzing image_1: This shows Grigoriy Perel'man with his distinctive appearance - he has long, dark brown wavy r that flows past his shoulders...

Analyzing image_2: This image shows Perel'man with an older appearance - his hair has turned predominantly gray/white and is wild and curly, flowing outward in an untamed manner...

Integration strategy: I need to combine these visual characteristics with the scene description of a sparsely furnished apartment in St.

Petersburg, surrounded by papers and old books, writing mathematical equations. The scene should capture his distinctive appearance - the long flowing hair (graying as in image_2), the full beard (also graying), deep-set eyes, oval face, and scholarly yet bohemian appearance. The setting reflects his reclusive, dedicated mathematical lifestyle $< / \mathrm { t h i n k } >$

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/a9e5739f6a20efb4852612f1b0b326bf867b6ff7e1f46679ea4d1a8e1ebfe63f.jpg]]

Recaption <recaption>In a sparsely furnished apartment room in St. Petersburg, illuminated by soft natural light filtering through a single ndow, sits the legendary mathematician Grigoriy Perel'man, deeply absorbed in his work. Preserve the exact facial features of Grigoriy Perel'man as own in image_1 and image_2: an oval-shaped face with prominent cheekbones, deep-set eyes with a characteristic slight squint, thick dark eyebrows w peppered with gray, and a full, flowing beard that has transitioned from reddish-brown to gray-brown, wild and untamed like the hair that flows st his shoulders in image_2. His complexion is fair with a weathered, sun-touched quality, and his expression carries the intense, contemplative focus brilliant mind at work. Maintain his distinctive bohemian scholarly appearance as referenced in both images...</recaption>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/12e9bb48672cc62dccb2f7354470faebfdf20b4d573f2d044a419bd09db04335.jpg]]

# Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b796b1e9b48cf94f57d37faa04a4831b5f5971853acffd395e5de967f9e49eb3.jpg]]

# Generated Images

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0fa8213970f81e1ea2691bcdcd3d5706b5135d6211ae672404daa4d6c540c525.jpg]]  
Figure 11 | Image generated for Grigory Perelman scribbling mathematical equations.

Image Prompt  Surrounded by advanced laboratory equipment and microscopes, Nobel laureate Bruce Beutler is intently reviewing genetic data on a large monitor.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c2574d3183dceecab19030dc65753d13cef6e74b3306d90c1b5c8dbb7229c023.jpg]]

# Text Search

Model Responses  I'll help you gather information and reference images for creating a detailed image description about Bruce Beutler. Let me start by searching for background information.

Tool Use <tool_call>{"name": "text_search", "arguments": {"q": "Bruce Beutler Nobel laureate immunologist geneticist","hl": "en", "top_k": 5}}</tool_call>

$\circledcirc$ Tool Output <observation> [1] Bruce Beutler, M.D. - Faculty Profile - UT Southwestern Bruce Beutler, MD, discovered an important family of receptors ...

[2] Bruce Beutler. The provided content only features the top navigation and header section ...

[3] Bruce A. Beutler – Facts - NobelPrize.org ...

[4] Bruce A. Beutler. The provided webpage is an official profile page for Bruce A. Beutler ...

[5] Profile of Bruce A. Beutler - PMC ...

</observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ca9622b9fcaca65a4311054a63d465d37ff2bfe46cefc91ad5b297af1e5eef57.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/db34816889d67d94c090ca038b0102eab1fbfd5f4fb88307cdc350f653b73212.jpg]]

# Image Search

Model Responses Now I have a good understanding of Bruce Beutler. He's an American immunologist and geneticist who won the 2011 Nobel Prize in Physiology or Medicine for his work on innate immunity, including the discovery of key components like TNFα and host defense mechanisms. Let me search for reference images of him.

Tool Use <tool_call>{"name": "search_image", "arguments": {"q": "Bruce Beutler Nobel Prize immunologist UT Southwestern", "hl": "en", "num": 5}}</tool_call>   
$\circledcirc$ Tool Output <observation> [1] The Center for the Genetics of Host Defense UTSW-Meet Our ...   
[2] Structural biology provides long-sought solution to innate ...   
[3] Alumnus, Nobel Laureate Bruce Beutler to Speak at UC San ..   
[4] Bruce A. Beutler | American Immunologist & Nobel Laureate ..   
[5] Nobel Prize 2011: Bruce Beutler, M.D. - Southwestern Medical

</observation>

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/efdd62a70afffeb39297d41bf0e70afe018aaf83ffd3f01e4e8114dd3a45a9fd.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2bc4601325226a09e1088748bec61c64c0d7c8a49bf04be734cad100382a5d7b.jpg]]

# Image Judgement

# Reference Images

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/bae858233a970c54c736ba7b8ab0a36d6c1e6beff021c6f19e8f1fc0faf163f4.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/8281c682069da2eb82d1f3092b9c06c3bb34dc342d68998666c739351d006ed6.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/059cfde2bd14be520cf5f334eb3ef97277ba689bd4b9ff56990d25c17b2f0fae.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/289897c7a24c70e39d66f1bd0e0fda7dd20510b84050c7c44418bc876701730e.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/5ad3a19f18a013406661066bb1655233f09745a527051f6fcee9f914d344f0f8.jpg]]

# Refined Recaption

Thinking <think> Analyzing image_1: This is a professional portrait of Bruce Beutler showing him from the shoulders up. He is a middle-aged to older Caucasian man with a distinctive appearance: a balding head with receding gray/silver hair on the sides... Analyzing image_2: This shows Bruce Beutler in a more formal setting, seated in front of wooden bookshelves filled with books. His facial features are consistent with image_1: the same balding pattern with gray hair on the sides, characteristic gray mustache, light eyes with a thoughtful expression, and warm smile. Here he's wearing a dark navy suit with a white dress shirt and a striped tie... Integration strategy: I need to place Bruce Beutler in a laboratory setting surrounded by advanced equipment and microscopes while reviewing genetic data on a monitor. I'll preserve his distinctive facial features (balding head, gray mustache, light eyes, warm expression) as seen in both reference images, while adapting his attire to be appropriate for a laboratory environment (perhaps the dark jacket from image_1 would be suitable).</think>   
Recaption <recaption> In a state-of-the-art genetics laboratory at UT Southwestern, Nobel laureate Bruce Beutler sits before a large high-resolution monitor displaying complex genetic sequencing data and colorful DNA analysis graphs. Preserve the exact facial features of Bruce Beutler as shown in image_1 and image_2: a balding head with receding gray-silver hair along the sides, a distinctive full gray-brown mustache, light blue-gray eyes with an intelligent and focused gaze, fair Caucasian complexion with natural aging lines, and a warm yet concentrated expression. Maintain his professional appearance as referenced in both images, wearing a dark navy zip-up jacket similar to image_1 over a collared shirt ...</recaption>

# Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/105da811e1f953fe87cef06bac26906f2a0887f009ebc4090a2c6b8be1eb5ea8.jpg]]

# Generated Images

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/13f8e4ab6408ac0f92aacaf41f0024228dd7c6dfeda0660449e283b872fdabf5.jpg]]  
Figure 12 | Image generated for Bruce Beutler reviewing data on a monitor.

# G FactIP Benchmark Evaluation Showcases

Case 1 Figure 13 presents a representative Seed 2.0 evaluation example for an image generated from a prompt about Gregg Popovich. By comparing the synthesized result against the reference images (GT1 and GT2), the MLLM-based evaluator produces a fine-grained scoring profile together with an interpretable reasoning trace. In particular, the evaluator verifies both the facial resemblance to Popovich and the semantic consistency of basketball-related attributes, such as the Spurs jerseys and the tactical whiteboard. Based on this multi-aspect assessment, the sample receives a high overall score of 9, illustrating the benchmark’s ability to jointly measure identity fidelity and contextual relevance.

Case 2 As shown in Fig. 14, this example illustrates how Seed 2.0 handles cases where visual quality remains strong but semantic correctness breaks down. For an image intended to depict Max Weber, the evaluator assigns only a moderate overall score of 5. Although the image is rated highly in terms of clarity (9) and aesthetic appeal (8), the reasoning process identifies two decisive issues: the generated face does not faithfully match the reference appearance—most notably due to the beard discrepancy and the addition of glasses absent from GT1/GT2—and the scene includes a modern microphone that introduces a clear historical anachronism. This case highlights that the benchmark does not merely reward surface-level image quality, but can also capture subtle factual and identity-level inconsistencies.

Case 3 Figure 15 provides an example of successful IP-oriented generation under a stylized artistic setting. Given a prompt requesting Habatan rendered in a Baroque oil painting style, the Seed 2.0 evaluator assigns an overall score of 9, reflecting both strong character fidelity and convincing stylistic execution. The accompanying reasoning demonstrates that the evaluator can recognize sophisticated visual properties associated with the Baroque tradition, including dramatic chiaroscuro and dense painterly brushwork, while still preserving the defining attributes of the mascot. Importantly, the analysis also uncovers a subtle but meaningful deviation: the generated image depicts two blue shoes, whereas the ground-truth reference (GT1) specifies one red shoe and one blue shoe. This example underscores the benchmark’s capacity for rigorous fine-grained auditing even in highly artistic generations.

Case 4 The failure mode illustrated in Fig. 16 further reveals the strictness of Seed 2.0 in evaluating identity-specific and text-sensitive attributes. For a prompt requiring Scottie Pippen wearing a classic Chicago Bulls jersey, the evaluator assigns a notably low overall score of 2. The reasoning trace points to several critical deficiencies: the generated subject bears little resemblance to Scottie Pippen when compared with GT1 and GT2, the image quality is compromised by substantial visual artifacts, and the jersey details—including the team branding and the iconic number 33—are rendered incorrectly and are largely illegible. This case demonstrates that the benchmark can effectively penalize failures in character identity, visual integrity, and fine-grained textual faithfulness.

# hlighting the coPrompt

Huddled on the sideline of a basketball court during a timeout, Gregg Popovich is intensely drawing a play on a whiteboard, instructing San Antonio Spurs players surrounding him.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/da1650a6b9d3910bd2c1a2a30585ab7f216c6d35d242ef3e16a5d64b05ad7109.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/7998183f6de623c1cd84a234e3280664e64ad6452b261cd01189a4aa4ff8ebe2.jpg]]  
GT1

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/fcfbe708c930bfae7ed9acda5ba8a4cf810e4cf07b4a4fc1e480c11b5f2fe560.jpg]]  
GT2

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/22d49c371794d553594d927fb325fa10531ee46e1849ce4f1721ea41570ffae5.jpg]]  
Generated Image

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b9ec9d3f81c8c09aeaf5eb884ee00b5b31c95db4eb2cbbfb3b582c34a6a97c52.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ec16563d022b3aca5f017008d3779ce0c1753b90d7198b7ee4ced0fbb464f47f.jpg]]  
MLLM Evaluation

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0540460872ca00823ae93305bc2073c2fc4b7613514e6278d2d67444616dc867.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/35aea27f9cc1e91894110308d7085146dcfbd383e4751823229c91a14b21b0da.jpg]]  
Seed 2.0:

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/79dc6514141135d02ad9281765c88f6cc4bb24a57c46f541343201a8c34ddf1c.jpg]]

# Reasoning

sharp, no blur, right? So 9? Wait wait, let's check. Content quality: the prompt says Popovich on sideline timeout drawing play for Spurs players ...

Overall high quality with minimal flaws. Wait let's make sure scores are integers 0-10. Let's adjust if needed? Wait maybe clarity 9, content quality 9, aesthetics 8, text_relevance_ip 10, overall 9.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/fb1288f600134133054ed7600dd2aca5a66b6bb07a49849fe15cb1bbf621a720.jpg]]

# Rationale

The generated image has high clarity with sharp details of characters oard. Its content fully matches the prompt: Popovich's likeness is

consistent with GT2, it correctly presents the scenario of him drawing a play for surrounding Spurs players during a sideline timeout, all core required elements are included. The composition is natural and conforms to the aesthetic characteristics of real sports event photos. All text and IP related elements are reasonable and relevant, the Spurs jersey content is correct, there is no garbled or irrelevant text. The overall quality is high, with only minor non-critical imperfections.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/9926085153293a0da1793fe642193e1fefb5d78fe2a0bb075e165317cf850e68.jpg]]  
Final Score

Overall: 9

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/51d40f97195e97f7290acfbd6b9d0d3ad58ce360b66e0444ad18c58e6df7795a.jpg]]

clarity: 9 - high clarity with sharp details

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/d96315055bfb9eb14be5e4318c4095353f1e6807993afc0c4f0ac4a045c6425e.jpg]]

content quality: 9 - content fully matches the prompt

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/440a3ba48029efae6eafccb47e9a503a919891bb54bd2fd7af9a32b262976db6.jpg]]

aesthetics: 8 - conforms to the aesthetic characteristics of real sports event photos

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/d33c7db343812518ae9db99ba4afd15573c9ae20896d7d6e5df98579bb592c72.jpg]]  
Figure 13 | MLLM evaluation for Popovich drawing a play.

text relevance: 10 - all text and IP related elements are reasonable and relevant

# hlighting the coPrompt

Surrounded by attentive students in a turn-of-the-century German university lecture hall, Max Weber is delivering a passionate academic speech from behind a wooden podium.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c46089e60a5aa610959102de907989b8084ee4f977a244c39a5782f946665576.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/a067e3b161be751d0a23fa83eeaa4fdebe63e8b91956514d9215d5fdb503d682.jpg]]  
GT1

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/f4260b9c0fce610aa5943f3bdfd404d71dd676888106adfa4868d3d170acb750.jpg]]  
GT2

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/f3326bb794d7769cdc4a2c3440c02992bfb5a216878b3ddf94d532092aa398cf.jpg]]  
Generated Image

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/f3d5654f6129f409907c31caabc564c35d5aea104db80d55042e6a13277ff28b.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e9e7b73e100ca0c489770694edc9ab0135db35287d9c71f53e07a70758f9bced.jpg]]  
MLLM Evaluation

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/5972695f99caed8d1399943ef61b1638094f6d11b54c745d53d2e1480f08bf70.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b4b33ce6105207809ce83a0f4879289bb5db7238738d0f53510bd2d8f433b44c.jpg]]  
Seed 2.0:

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/72bf3e526ddb2e38cfc23a54731e6f44ec9555d66ec0dd8f340795291b7daa16.jpg]]

# Reasoning

Got it, let's evaluate each part step by step. First, clarity: The AS image is clear, no ait, 9? Wait no, let's see. Wait it's sharp, details are good. Maybe 9? Wait no, wait ll score: 5, while the image is visually well-made and fits the general scene, the of the key figure Max Weber and anachronistic detail greatly reduce its adherence to ait let's adjust scores properly. Let's make sure all are integers 0-10. Wait let's orrect, it's very clear. content_quality: 4, as above, correct elements are setting, on, wrong are subject likeness, anachronistic mic. Aesthetics: 8, yes, looks great, composition. Text_relevance_ip: 3? Wait or 4? Let's say 3, because the main subject is re part of the prompt.Overall_score: 5, that makes sense.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/d07014778e309d96fce751cbb8ed844da425937f7c4feecc99413df2f2ca0799.jpg]]

# Rationale

The image has high sharpness and clear details, so clarity scores high. It formance with appropriate lighting, reasonable composition and

historical atmosphere, so aesthetics scores well. But it has obvious defects: the depiction of Max Weber is inconsistent with his real appearance in GT1 and GT2 (with extra glasses, mismatched facial and beard features), and there is an anachronistic modern microphone that did not exist in the turn-of-the-century period. It deviates greatly from the core requirements of the prompt, so content quality and text relevance IP scores are low, leading to a medium overall score.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/06b6fde562614d8bc2af0a69c16c77f23578186023414e6f1eb124f5bc4e6865.jpg]]  
Final Score

Overall: 5

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/6feb18375d4e0ce5fd1038ecdbaff892a7434c92368efbd95a9a0e2f6d594b29.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/2065b4b4e7d1330fe3882cd8b237bad1af494b728e3fe6a45670a29bd2e50c94.jpg]]

clarity: 9 - high sharpness and clear details

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/a96867e156a9c27c34e20f2f6452afb54fe5f1d3bf2d1479f97a6a8cb0075031.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e1391932aecdd2080d3e09b5465d11de75ceca08d6a1dceb4960548b2728aeb0.jpg]]

content quality: 4 -deviates greatly from the core requirements of the prompt

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/21826842bb3f1c0ebf5c2a697ea04175690eb89da32820c086c23e7977d52205.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/36af7d38d9ded321a07eb1e962dbca767a1309719727e57bc51220d97cc559b6.jpg]]

aesthetics: 8 - good visual performance

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/e19791a821dbdec8ef57eb7659f892d204d1f9c05847a2ea60551f52e5d280c6.jpg]]  
Figure 14 | MLLM evaluation for Max Weber delivering a passionate speech.

text relevance: 3 - inconsistent with his real appearance

# Prompt

A dramatic classical Western oil painting in the Baroque style, illustrating the mascot character Habatan. The round, yellow anthropomorphic phoenix is emerging triumphantly from the fiery ashes and rubble of a destroyed landscape, symbolizing recovery from the Great Hanshin Earthquake. Rich textures, chiaroscuro lighting, grand scale, heavy brushwork.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/805070c42b257678926720c28c11ccb413d80da6cf3441a74e6136a939f25137.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/75d947608ecc2f81093ef8f7c9045b26ebead7c799103231df16f439350452ca.jpg]]  
GT1

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/9769913abd505bf937fcabfd9aeb04329a25b383881efccecbfcd1e4d728cb58.jpg]]  
GT2

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/fb30537e85d107088c77452d926dbf111abe8c84997a889fa771a790bf1dcf3d.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/f1ba87bfb55a8b4a0af2c5bca7d4e176ed6845e3d74b1dbd0f826dc8a670c263.jpg]]  
MLLM Evaluation

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/6c071dd2c09025fed13c4dc33336719b8b2f2f5090769ba39e3a5a05b94b5228.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/afb281fc8bef08d25a8fbeb35090a51f2bc0bcbc487a7e613089878ff9209d21.jpg]]  
Seed 2.0:

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/8ce55966ada71a21d628d9f469e22c75109d694bbeb01fdcf01fc2a0a2c32ed0.jpg]]

# Reasoning

Got it, let's go through each key step by step. First, clarity: The AS is a clear

oil painting, no blur, details are sharp, so that's 10? Wait yeah, looks really clear. Content quality: It's well rendered, the character is consistent with GT1, the background matches the prompt. Wait 9? ... Wait now assign scores properly:\nclarity: 10, right, no blurriness at all, super clear. content_quality: 9, only minor shoe color issue, everything else is great.\naesthetics: 9, really good baroque style, dramatic, beautiful lighting, great texture.\ntext_relevance_ip: 9, IP is almost perfect, only shoe color mismatch, prompt requirements all met (baroque, emerging from rubble, fire, etc.)\noverall_score: 9, since it's almost perfect, only tiny flaw.\nWait wait let's make sure. Oh right GT1 has left blue, right red? Wait GT1: left foot blue, right foot red, AS has both blue, that's the only flaw. Yeah.\nNow structure the JSON correctly, all keys present, scores integers 0-10, rationale clear.\nWait let's make the rationale precise:

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/4000c282024cdda96cd84b22077557479f9fb095d26d137a00a7302079546b10.jpg]]

# Rationale

The generated image has perfect clarity with all details clearly visible. Its content quality is h igh, with only a minor flaw of inconsistent shoe color (GT1 has one red and one blue shoe while the generated image has two blue shoes). Aesthetically it fully matches the requested Baroque classical oil painting style, with heavy brushwork, rich textures, chiaroscuro lighting and dramatic grand scale. It has high text relevance and IP consistency: the Habatan ic grand scale.It has high text relevance and IP consistency: the Habatan mascot aligns with core features of GT1/GT2 (round yellow anthropomorphic phoenix, orange wing core features of GT1/GT2 (round yellow anthropomorphic phoenix, orange win patterns, smiling fac e with pink cheeks), and the scene of the character standing triumphantly on fiery destroyed rubb le correctly corresponds to the prompt's symbolic meaning of Great Hanshin Earthquake recover y. The overall performance is excellent with only a tiny imperfection.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ffef12e7cbe14eb67881df1f83c82ef46563aa827b05eb94e0e71df21ef2fa19.jpg]]  
Final Score   
Overall: 9

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/3eb30cfb07815a5e9b44f0a696582641e4e6530af9f3e5eb1fc24bb0ef3c96fb.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/86bcb03394fddefdb75906ea151138538ea080244f998a2a3b1369cdf2b61c74.jpg]]

clarity: 10 - perfect clarity with all details clearly visible

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/3985fb7040ba02367e99396c9d2243942b95a7a156a938316f10900c211c8198.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c31d2312cbd211608e287863ce6164484a67a5e707dba2c6d5064604fa9b7069.jpg]]

content quality: 9 - content quality is high

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/8952343b0e97a4281717658450d8d741ed3009f84e71ae04cb24f20fb3c77728.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/de4bb102190c7e44f11cb6de7ea6a0c0e8f076a526bc39c022f6682e7e5c222e.jpg]]

aesthetics: 9 - fully matches the requested Baroque classical oil painting style

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ddf0090a40b41b8447e61343ccf660a35f4c0cdcea6b5cdc57e520a5a618d9ff.jpg]]  
Figure 15 | MLLM evaluation for the mascot character Habatan in the Baroque style.

text relevance: 9 - high text relevance and IP consistency

# lighting the coPrompt

Wearing the iconic red number 33 Chicago Bulls jersey, Scottie Pippen is aggressively driving towards the basket during a packed NBA playoff game in the 1990s.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/37f4091573d7c719903261756530aac069d7befd1484447004116ca25ec96429.jpg]]  
Ground Truth

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/d431b21bb6227c02f1150893710bf731a7278c00abc361715affc6c3c82938ab.jpg]]  
GT1

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/1c124051cf4815d7c40cc5ed7d587b2e3b772015b82db5235b54d6aeab680309.jpg]]  
GT2

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/818d2cbdf4bdaa0173a862d8f07f7b90c2b17e904c68f8e32bb7404f4f1cd7ff.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/93e9059bc4d0fd52785188d467049b2558117a7dea8c2e0288735ca32ae442ae.jpg]]  
Generated Image

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/db90f6e2184b8e4cd5c49801b0ccb4d95d17439b32ee80d246fd2ba7f7809fce.jpg]]  
MLLM Evaluation

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/a74498a019c531e4795576ba29cb5b46cb13aed98b747b358afea5ca74c554a6.jpg]]

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/d1298c4e17c68ff54dab9d24a99cfd96471781bc69debb6e46cfc37a66471cdc.jpg]]  
Seed 2.0:

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/6019a492a201146b8d87c77d2581cb60795681abebe9efba443f7bbed17ac858.jpg]]

# Reasoning

Got it, let's go through each key one by one. First, clarity: the AS image is

blurry, face is off, details are messed up, so like 3? Wait no, wait. Wait text_relevance_ip: the jersey text is wrong, not Bulls, number is not 33, it's gibberish. Also the person doesn't look like Pippen. Let's break down each: ... Overall score: The image fails to correctly depict the subject, required jersey details, and has poor visual quality, so overall score 2. Wait wait, but let's make all scores integers, right. Wait wait let me adjust to make it right: Wait clarity: 3, correct, because you can tell it's a basketball player near a hoop but details are blurry and artifacted. Content quality: 2, actually, because the person is not Pippen, which is a core content requirement. Oh right, the main subject is wrong. So content quality 2. Aesthetics: 3? Or 2? Let's say 2. Text_relevance_ip: 0, $1 0 0 \%$ wrong text and number, no correct branding. Overall score: 2, that fits.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/c5840969f3244fd25779da78ed95d6c5a2e266eb9371184dfe9f07b012c86c9d.jpg]]

# Rationale

The AS image has heavy artifacts and blurry details, leading to low clarity.

While it generally depicts a basketball player in a red jersey near a basket in a packed arena, the subject does not resemble Scottie Pippen at all, resulting in poor content quality. Distorted facial features and garbled visual elements make it visually unappealing for low aesthetics. The text on the jersey is completely unrecognizable, with no correct 'BULLS' branding or number '33', so text relevance IP scores 0. The image fails to meet most core requirements of the prompt, leading to a low overall score.

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/0f3f006558b176c210b711a3f1c23522e3852a4827e0f6c105287e77623114f7.jpg]]  
Final Score   
Overall: 2

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/93849c9104d0b97cf64eb2f808c749094662e0808f82e4474372fca2868d7664.jpg]]

clarity: 3 - heavy artifacts and blurry details

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/eb732224e9a6c11c07f2a3de3562f89267f11e4b2ce20a7ab28fec835333b062.jpg]]

content quality: 2 - the subject does not resemble Scottie Pippen at all

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/b06ba480ef02be4c08061420f3c63b3d62f0078d85485069397ee0d44bc2e850.jpg]]

aesthetics: 2 - distorted facial features and garbled visual elements

![[Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis/images/ee201f7a3e10cd8503dc86eaf4e645e0d79ac1a266706e291fc0ebcac6ed05a8.jpg]]  
Figure 16 | MLLM evaluation for Scottie Pippen during an NBA playoff game.

text relevance: 0 - the text on the jersey is completely unrecognizable