---
title: "VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation"
source: "https://arxiv.org/html/2603.02681v1"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/04_%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E4%B8%8E%E7%BC%96%E8%BE%91/VisionCreator%2C%20A%20Native%20Visual-Generation%20Agentic%20Model%20with%20Understanding%2C%20Thinking%2C%20Planning%20and%20Creation%2C%202026.no_watermark.zh-CN.dual.pdf"
---
Jinxiang Lai <sup>2∗</sup>, Zexin Lu <sup>1∗</sup>, Jiajun He <sup>1</sup>, Rongwei Quan <sup>1</sup>, Wenzhe Zhao <sup>1</sup>, Qinyu Yang <sup>1</sup>, Qi Chen <sup>1</sup>, Qin Lin <sup>1†</sup>, Chuyue Li <sup>1</sup>, Tao Gao <sup>1</sup>, Yuhao Shan <sup>1</sup>, Shuai Shao <sup>1</sup>,  
Song Guo <sup>2§</sup>, Qinglin Lu <sup>1†</sup>  
<sup>1</sup> Tencent Hunyuan, <sup>2</sup> Hong Kong University of Science and Technology  
<sup>∗</sup> Equal contribution, <sup>§</sup> Corresponding Author, <sup>†</sup> Project lead

###### Abstract

Visual content creation tasks demand a nuanced understanding of design conventions and creative workflows-capabilities challenging for general models, while workflow-based agents lack specialized knowledge for autonomous creative planning. To overcome these challenges, we propose VisionCreator, a native visual-generation agentic model that unifies Understanding, Thinking, Planning, and Creation (UTPC) capabilities within an end-to-end learnable framework. Our work introduces four key contributions: (i) VisGenData-4k and its construction methodology using metacognition-based VisionAgent to generate high-quality creation trajectories with explicit UTPC structures; (ii) The VisionCreator agentic model, optimized through Progressive Specialization Training (PST) and Virtual Reinforcement Learning (VRL) within a high-fidelity simulated environment, enabling stable and efficient acquisition of UTPC capabilities for complex creation tasks; (iii) VisGenBench, a comprehensive benchmark featuring 1.2k test samples across diverse scenarios for standardized evaluation of multi-step visual creation capabilities; (iv) Remarkably, our VisionCreator-8B/32B models demonstrate superior performance over larger closed-source models across multiple evaluation dimensions. Overall, this work provides a foundation for future research in visual-generation agentic systems.

## 1 Introduction

AI-assisted visual content creation has revolutionized workflows from professional design to social media. The field has evolved from single-image generation [^11] [^28] [^13] [^29] [^18] to complex multi-modal synthesis [^33] [^34] [^35] [^37] [^9], demanding systems that can understand creative intent, plan multi-step operations, and autonomously execute intricate workflows. As shown in Fig.2, current approaches to autonomous visual creation can be categorized into three main paradigms, each with distinct limitations: (a) General-purpose Unified Multimodal Models (UMM) [^5] [^16] [^14] [^4] leverage large-scale pre-training to achieve impressive visual understanding, but lack the domain-specific knowledge required for autonomous creative planning and struggle to decompose complex objectives without extensive prompt engineering. (b) Workflow-specific Agent [^33] [^34] [^35] employ predefined pipelines for specific domains like movie generation or story creation, but their rigid architectures cannot adapt to diverse creative tasks or handle unexpected outcomes during execution. (c) Workflow-guided Agent [^37] [^9] orchestrate external tools through carefully designed prompts and coordination logic, leveraging general language models to interpret requests and sequence operations. However, this approach faces several limitations: (i) Reliance on prompt engineering rather than learned domain knowledge, limiting creative understanding; (ii) Explicitly programmed coordination logic that restricts adaptability to diverse tasks; and (iii) Inability to be jointly optimized end-to-end for creative task performance.

![Refer to caption](https://arxiv.org/html/2603.02681v1/figs/radar.png)

Figure 1: Human Evaluation results on VisGenBench-Image and VisGenBench-Video.

To overcome these limitations, we propose VisionCreator, a native visual-generation agentic model that unifies Understanding, Thinking, Planning, and Creation (UTPC) capabilities in an end-to-end learnable framework, as shown in Fig.2 (d). Unlike existing approaches that rely on predefined workflows or external template workflows, our native architecture intrinsically integrates the capabilities of Understanding design conventions and user intent, Thinking through complex creative constraints, Planning multi-step execution trajectories, and Creation of high-quality and diverse visual creation tasks. However, realizing this new paradigm faces several critical challenges:

① Data Bottleneck: Currently, no comprehensive datasets exist for training agents to perform visual content creation through tool invocation. The lack of high-quality trajectories prevents supervised learning of the UTPC capabilities.

② Task Complexity: How to develop models that can handle the full spectrum of visual creation challenges, which encompass (i) Diverse task types, (ii) Varying difficulty levels from basic generation to advanced composition, and (iii) Complex creation tasks requiring 20+ execution steps? Existing approaches face significant limitations: specialized systems excel in narrow domains but fail to generalize across diverse tasks, while general models lack the depth for sophisticated creative reasoning and struggle with long-horizon consistency and adaptive strategy adjustment.

③ Training Difficulty: How to establish an effective and efficient training paradigm for such a native agent? The conventional SFT+RL framework faces significant obstacles: (i) SFT phase struggles to balance general capability preservation with domain-specific specialization, often leading to catastrophic forgetting or insufficient expertise; (ii) Direct online RL training with real tools incurs prohibitive costs and instability due to expensive API invocation and limited concurrency. Furthermore, designing accurate reward signals for multi-step creative trajectories is particularly challenging, as imperfect reward functions are highly vulnerable to reward hacking.

![Refer to caption](https://arxiv.org/html/2603.02681v1/figs/fig1.png)

Figure 2: Framework comparisons. (a) UMM. (b) Workflow-specific Agent. (c) Workflow-guided Agent. (d) Our Native VisionCreator.

To address these challenges, we propose:

(i) VisGenData-4k with UTPC Structure: We design a metacognition-based VisionAgent to generate a comprehensive dataset following the UTPC structure, featuring diverse visual creation tasks across multiple difficulty levels. Through rigorous human quality inspection, we meticulously filter and retain only the highest-quality data samples. The resulting VisGenData-4k provides diverse and high-quality execution trajectories that explicitly capture Understanding of design conventions, Thinking through creative constraints, Planning of multi-step trajectories, and Creation of visual content, offering rich supervision signals for complex creative workflows.

(ii) Progressive Specialization Training (PST): We introduce a novel Progressive Specialization Training methodology that cultivates UTPC capabilities through two-stage optimization. PST effectively addresses the generalization-specialization trade-off by first establishing robust Understanding and Thinking capacities through general foundation learning, followed by targeted domain specialization to enhance Planning and Creation expertise. This progressive strategy not only prevents catastrophic forgetting of general abilities but also efficiently identifies optimal data composition for stagewise specialization, enabling the model to develop comprehensive UTPC capacities while maintaining strong cross-domain reasoning abilities.

(iii) Virtual VisGenEnv Construction: We construct VisGenEnv, a virtual environment for VRL. It features 36 tools with high-fidelity simulation of their behaviors. Multimodal outputs are simulated by returning random samples from a media database, providing correct physical attributes. This design enables effective learning of workflow planning through accurate tool behavior simulation.

(iv) Virtual Reinforcement Learning (VRL) with LtrReward: We develop an innovative Virtual Reinforcement Learning (VRL) paradigm that conducts the entire reinforcement learning using Long Trajectory Reasoning Reward (LtrReward) within the high-fidelity VisGenEnv. This approach bypasses the prohibitive cost of thousands of GPUs by leveraging simulated tool-call behaviors and functional logic, enabling stable and scalable learning of high-quality planning and action trajectories. Moreover, we provide a theoretical analysis that establishes formal guarantees on sim-to-real transfer and real-world performance improvement.

Finally, we introduce VisGenBench, a comprehensive benchmark designed for evaluating visual generation agentic models that operate through multi-step tool invocation to accomplish complex image and video creation tasks. Our benchmark encompasses: (i) Comprehensive Test Suite - featuring 1.2k test samples including 400 image-generation tasks and 800 video-generation tasks; (ii) Diverse Applications - spanning 10 evaluation dimensions across 35+ real-world scenarios; (iii) Standardized Protocol - ensuring reproducible evaluation through structured scoring rubrics.

Overall, our contributions are: (i) The VisionCreator, a novel native visual-generation agentic model that unifies UTPC capabilities in an end-to-end learnable framework; (ii) VisGenData-4k and its construction framework using metacognition-based VisionAgent to generate high-quality creation trajectories with UTPC structures; (iii) A progressive training methodology combining Progressive Specialization Training (PST) and Virtual Reinforcement Learning (VRL) with LtrReward, enabling stable and efficient learning of complex creation trajectories entirely within a virtual environment VisGenEnv; (iv) VisGenBench benchmark with 1.2k test samples across diverse scenarios for standardized evaluation of multi-step visual creation capabilities.

## 2 Related Works

### 2.1 Image Generation

Current image generation models primarily fall into two categories: Autoregressive [^2] [^7] [^10] [^32] [^31] [^22] models and Diffusion [^11] [^28] [^13] [^29] [^18] models. While these models provide powerful single-step image generation capabilities, they primarily focus on the Creation aspect of visual content generation. Our VisionCreator builds upon these fundamental generation technologies but extends them by integrating comprehensive Understanding, Thinking, and Planning capabilities. This allows our agent to not only generate individual images but also reason about complex creative requirements and plan multi-step visual creation workflows that leverage these underlying generation models as tools.

### 2.2 Video Generation

Video generation methods build on image models by adding time-based processing. Approaches like Make-A-Video [^30] and SVD [^1] extend image generation to video, while newer architectures like DiT [^23] and MMDiT [^6] in models such as CogvideoX [^38] show progress in handling longer videos. These video generation tools are important for our agent's creation ability, but they work separately. Our VisionCreator connects these tools through planning and reasoning to handle complete creation tasks.

### 2.3 Visual Generation Agents

Current approaches to autonomous visual creation include three main agent paradigms: (i) Workflow-specific Agents (e.g., MovieAgent [^33], Captain Cinema [^34], MM-StoryAgent [^35]) employ predefined pipelines for specialized domains but lack adaptability to diverse creative tasks. (ii) ComfyUI Workflow Generation methods (e.g., ComfyAgent [^37], ComfyMind [^9], ComfyUI-R1 [^36], ComfyGPT [^12]) specialize in generating ComfyUI-format workflows, which limits their visual creation capability in general API scenarios. (iii) Workflow-guided Agents [^37] [^9] orchestrate external tools through prompt engineering but face limitations in creative understanding depth and end-to-end optimization. These limitations motivate our native visual-generation agent that intrinsically integrates UTPC capabilities in an end-to-end learnable framework.

![Refer to caption](https://arxiv.org/html/2603.02681v1/figs/data_gen.png)

Figure 3: VisGenData-4k construction pipeline.

## 3 VisGenData-4k with UTPC Structure

To tackle the data bottleneck in training visual creation agents, we design VisionAgent, a dataset generation framework based on a Metacognition paradigm. To construct a high-quality VisGenData dataset, VisionAgent employs commercial proprietary models (such as GPT-5, GPT-4o, Veo3, Sora2, etc) for multimodal data generation, and we further filter low-quality trajectories with algorithms and human experts. As shown in Fig. 3, the construction pipeline is as follows: (i) VisionAgent first generates 16k trajectories from 20k queries covering 42 scenarios. (ii) With the rigorous LtrReward and VLM-Grader methods, we remove 10k low-quality trajectories and obtain 6k candidate trajectories. (iii) These subsequently undergo a manual review by human experts, where 2k undesired trajectories are filtered out, resulting in high-quality 4k trajectories.

![Refer to caption](https://arxiv.org/html/2603.02681v1/figs/visionagent.png)

Figure 4: VisionAgent framework for dataset generation.

### 3.1 VisionAgent for Dataset Construction

As shown in Fig.4, our VisionAgent generates high-quality execution trajectories that capture the complete reasoning process for complex visual creation tasks. VisionAgent with metacognition achieves a 72% task success rate, representing a 30% improvement over the baseline method that relies solely on thinking.

Dual-Agent Architecture. Our framework employs a dual-agent architecture that separates task understanding from execution reasoning: (1) TaskAgent: Serves as the task classifier and router. It analyzes user inputs and performs fine-grained task classification across the 21 distinct task types, then selects appropriate predefined workflow templates and tools pool for specific task categories. (2) MetaAgent: Functions as the core reasoning engine with metacognitive capabilities. It receives both the selected workflow and tools pool as inputs, then executes structured reasoning through four standardized reasoning types defined in metacognition.

![Refer to caption](https://arxiv.org/html/2603.02681v1/x1.png)

Figure 5: VisGenData-4k dataset statistics.

Metacognitive Reasoning Process. The metacognition defines four reasoning types: the <think> phase maintains situational awareness and todo-list through continuous state evaluation; the <plan> phase constructs executable task sequences by decomposing objectives and dependencies; the <tool\_call> phase invokes appropriate tools based on plan blueprints and analytical reasoning; and the <answer> phase verifies goal completion, forming a closed-loop execution process. This Metacognitive reasoning process guides the MetaAgent to generate the UTPC trajectory.

Reference Workflow Integration. We incorporate 15 predefined workflow templates as best-practice guides, ensuring planning remains flexible yet stays on track. These workflows provide domain-specific execution patterns for various visual creation tasks, representing 15 distinct application scenarios from storyboards to animated short films.

### 3.2 Dataset Composition and Statistics

Fig.5 shows the statistics of our VisGenData-4k, which exhibits the following key features: (1) Diverse Task Types: Encompassing 21 distinct task types (including storyboards, marketing posters, product marketing videos, animated short films, etc), this diversity is crucial for training agents to handle a broad range of real-world creative demands, significantly enhancing their adaptability and practical applicability. (2) Complex Trajectory Structure: With a mean of 15 steps and 64% of trajectories exceeding 20 steps, this complexity is crucial for training agents to decompose and plan long-horizon tasks, fostering robust problem-solving capabilities in visual creation. (3) Rich Contextual Information: The substantial token length (mean: 29k, 43% over 32k) equips agents with the ability to process and utilize extensive contextual cues, significantly enhancing their capacity for detailed and context-aware generation.

## 4 Agentic Post-Training

### 4.1 Agentic Framework

As shown in Fig. 6, VisionCreator is formulated as a unified agent that integrates Understanding, Thinking, Planning, and Creation (UTPC) capabilities to accomplish complex visual generation tasks. Formally, we model the agent as a policy $\pi_{\theta}$ operating over long-horizon multimodal trajectories: $\tau=(o_{0},a_{0},o_{1},a_{1},\dots,o_{T})$, where $o_{t}$ denotes multimodal observations (textual instructions, intermediate tool feedback, and virtual visual states), and $a_{t}$ denotes agent actions including reasoning tokens, planning steps, and tool invocations. The training process follows a two-stage agentic post-training paradigm: (1) Progressive Specialization Training (PST), which initializes a strong policy prior via supervised learning over expert UTPC trajectories. (2) Virtual Reinforcement Learning (VRL), which further optimizes long-horizon planning and tool-use strategies through large-scale exploration in a simulated environment.

### 4.2 Progressive Specialization Training

The goal of Progressive Specialization Training (PST) is to learn an initial policy $\pi_{\theta_{0}}$ that simultaneously preserves general reasoning competence while acquiring domain-specific visual creation ability, thereby enabling a functional visual content creation agent rather than a narrowly tuned generator. Let the supervised dataset be $\mathcal{D}=\mathcal{D}_{\text{gen}}\cup\mathcal{D}_{\text{vis}}$, where $\mathcal{D}_{\text{gen}}$ contains large-scale general reasoning and tool-use trajectories, and $\mathcal{D}_{\text{vis}}$ contains expert-curated visual creation trajectories (VisGenData-4k). Standard supervised fine-tuning (SFT) minimizes

$$
\mathcal{L}_{\text{SFT}}(\theta)=\mathbb{E}_{(o,a)\sim\mathcal{D}}\left[-\log\pi_{\theta}(a\mid o)\right].
$$

However, naive single-stage SFT exhibits two fundamental failure modes. Training only on $\mathcal{D}_{\text{vis}}$ leads to catastrophic forgetting of general reasoning and planning ability, resulting in nearly zero agent competence; empirically, Tab. 4 shows performance dropping to 0.007, indicating the model is unable to function as a visual creation agent. Conversely, one-stage mixed SFT on $\mathcal{D}_{\text{gen}}\cup\mathcal{D}_{\text{vis}}$ avoids catastrophic forgetting but yields suboptimal specialization, since the dominance of $\mathcal{D}_{\text{gen}}$ suppresses learning of visual-creation behaviors and degrades downstream agent performance. These observations reveal a necessary condition for visual agents:

$$
\textbf{General Competence Preservation}+\textbf{Strong Visual Agent Specialization},
$$

which neither naive SFT strategies can satisfy simultaneously.

PST resolves this conflict through a controlled two-stage curriculum that induces a gradual distribution shift. In Stage 1 (general foundation learning),

$$
\mathcal{D}^{(1)}=\mathcal{D}_{\text{gen}}^{500\text{K}}\cup\lambda\,\mathcal{D}_{\text{vis}},
$$

establishing robust reasoning, planning, and tool-use capabilities while lightly anchoring the policy to the visual generation agent domain. In Stage 2 (targeted specialization),

$$
\mathcal{D}^{(2)}=\mathcal{D}_{\text{gen}}^{200\text{K}}\cup\lambda\,\mathcal{D}_{\text{vis}},
$$

the increased effective influence of $\mathcal{D}_{\text{vis}}$ drives specialization toward visual content creation, while continued exposure to $\mathcal{D}_{\text{gen}}$ prevents catastrophic forgetting. Overall, PST learns a structured initialization

$$
\pi_{\theta_{0}}\approx\arg\min_{\theta}\mathbb{E}_{(o,a)\sim\mathcal{D}^{(1)}\rightarrow\mathcal{D}^{(2)}}\left[-\log\pi_{\theta}(a\mid o)\right].
$$

which constrains downstream reinforcement learning (RL) to a policy region that already satisfies both general competence and visual specialization. Experimental results further validate the necessity of PST. Compared with one-stage SFT, PST achieves substantially stronger performance on visual creation agent tasks, demonstrating that progressive specialization is essential for learning effective UTPC behaviors. Moreover, PST provides a significantly better initialization for RL: the initial reward score before RL training increases from 0.64 (one-stage SFT) to 0.87 (PST), a gain of +0.23. This improved starting point directly translates into optimization efficiency—RL convergence is accelerated by approximately 50%. These findings confirm that PST not only improves final agent capability, but also fundamentally reduces the difficulty of downstream reinforcement learning.

![Refer to caption](https://arxiv.org/html/2603.02681v1/figs/framework.png)

Figure 6: Our Native VisionCreator framework.

### 4.3 Virtual Reinforcement Learning

Building upon the robust foundation established by PST, we refine the model's UTPC capabilities through Virtual Reinforcement Learning (VRL) based on the GRPO algorithm. To enable scalable long-horizon learning without invoking real-world tools, we first construct a high-fidelity virtual environment VisGenEnv that simulates the behavior of visual creation tools. Within this environment, LtrReward components are designed to supervise agent trajectories and guide both planning and execution. To understand that policies learned under these rewards transfer effectively to real-world scenarios, we provide a theoretical analysis of VRL. Building upon these insights, we then introduce a plan-driven reward that integrates planning and execution signals to optimize robust long-horizon visual creation performance.

![Refer to caption](https://arxiv.org/html/2603.02681v1/figs/virtual_env.png)

Figure 7: Comparison of the real environment and our virtual VisGenEnv environment, with an example of using a video generation tool.

#### 4.3.1 Virtual VisGenEnv Environment

To enable scalable long-horizon learning without invoking real-world tools, we first construct a high-fidelity virtual environment called VisGenEnv. This environment serves as a sandbox where the agent can safely explore planning and tool-use strategies, laying the foundation for subsequent reward design and theoretical analysis. VisGenEnv integrates a comprehensive suite of 36 visual creation tools (see Appendix for full list). The core of its design lies in a procedural simulation that accurately replicates the functional logic and behavioral patterns of real tools, including state transitions, parameter validation, and output specifications such as image resolution and video duration. To simulate multimodal outputs, the environment returns media files randomly sampled from a database while ensuring physically correct attributes consistent with tool specifications. This high-fidelity simulation of tool behaviors enables the agent to effectively learn the causal structure of the workflow and master robust planning policies through extensive practice within the virtual setting.

Training agent models by reinforcement learning in the real environment is prohibitively expensive. As illustrated in Fig. 7, supporting a training batch size of 24 with 4 rollouts (i.e., 96 concurrent rollouts in total) quickly becomes computationally intractable. Video tools are particularly costly: each instance requires 8 GPUs and roughly 30 seconds per video, meaning 96 concurrent rollouts would require $8\times 96=768$ GPUs. Deploying multiple real image and video generation tools would require several thousand GPUs, while our virtual environment VisGenEnv enables long-horizon exploration with only a few GPUs—thus saving thousands of GPU resources.

![Refer to caption](https://arxiv.org/html/2603.02681v1/x2.png)

Figure 8: LtrReward Components.

#### 4.3.2 LtrReward Components

With the virtual environment in place, as shown in Fig. 8, we now define LtrReward components $R_{\text{vrt}}$ (i.e., virtual reward applicable to VisGenEnv) as reward signals that guide the agent's learning, which consist of Plan Reward $R_{\text{plan}}$ and Fine-grained Reward $R_{\text{fine}}$.

Plan Reward $R_{\text{plan}}$ evaluates the overall quality of the task plan using a proposed vPlanJudger, an expert-informed LLM evaluator that leverages a curated repository of expert reference plans to provide in-context guidance. By performing cross-referenced reasoning between the candidate plan and expert-authored strategies, the vPlanJudger computes a multidimensional alignment score focusing on five key facets: (1) Requirement Fulfillment, a binary check on whether the output's modality and quantity align with the user request; (2) Logical Coherence, verifying the causal validity of sub-task sequencing; (3) Pragmatic Executability, ensuring each step is grounded within the available toolset or LLM capabilities to avoid hallucinatory actions; (4) Decomposition Atomicity, which evaluates whether the plan is partitioned into actionable atomic tasks; and (5) Expert-Guided Optimality, which rewards task-specific best practices such as identity consistency for multi-shot content, beat-aligned audio-visual synchronization, and the strategic minimization of complexity.

The Fine-grained Reward $R_{\text{fine}}$ integrates both rule-based and effect-based signals to ensure structurally valid execution and successful task realization. Specifically: (1) Rule-based components include Format Compliance $R_{\text{format}}$, which validates UTPC structural correctness via parsing of tags, ordering, content, and JSON validity; Tool Invocation $R_{\text{tool}}$, which scores execution success with graded penalties for intermediate or final failures; and Visual Consistency $R_{\text{cons}}$, which rewards appropriate use of reference-based generation when consistency is required. (2) Effect-based components include Result Achievement $R_{\text{result}}$, which verifies output constraints such as image count and video duration within tolerance bounds, and Trajectory Coherence $R_{\text{traj}}$, which evaluates alignment between planning intent and executed actions through an LLM-evaluator. Together, these rewards provide trajectory-level supervision that encourages correct agentic structure, reliable tool usage, and coherent visual creation outcomes.

#### 4.3.3 Theoretical Foundations of Virtual Reinforcement Learning

Based on the constructed virtual environment and the LtrReward components, we provide a theoretical analysis to explain the effectiveness of VRL when transferred to real-world execution. The theoretical legitimacy of VRL rests on its ability to maintain policy efficacy despite the intrinsic discrepancies between virtual simulation and real-world execution. Specifically, VRL operates under a Rollout Gap, where the agent lacks real visual feedback to rectify its trajectory, and an Objective Inconsistency, caused by substituting the vision reward $R_{\text{vision}}$ (which measures perceptual quality across multiple visual dimensions) with a structural proxy $R_{\text{result}}$. To evaluate how these discrepancies affect policy transfer, we model the sim-to-real transition as a function of four synergistic variables: (i) Tool Capability ($C_{\text{tool}}$), quantifying the reliability of the generative engine; (ii) PST Prior ($\pi_{\text{pst}}$), anchoring the agent's initial reasoning within a distribution derived from real expert data; (iii) Plan Sufficiency ($\Phi_{\text{plan}}$), measuring the causal link between logical correctness and visual quality; and (iv) Result Reward ($R_{\text{result}}$), ensuring the structural completion of tasks.

The following theorems establish the mathematical foundation of VRL: Theorem 4.1 provides an error bound guarantee, proving that the sim-to-real gap remains controllable under the joint constraint of these variables; Theorem 4.2 characterizes the real-world performance gain as a competition between Causal Improvement and Transfer Loss, showing that VRL yields non-negative improvement whenever the causal reward gain dominates the bounded sim-to-real error.

###### Theorem 4.1 (Virtual-to-Real Error Bound).

Let $J_{\text{real}}(\pi)$ and $J_{\text{vrt}}(\pi)$ be the expected returns of policy $\pi$ in real and virtual environments. And $\delta,\alpha,\beta$ are environment-specific scaling factors. The transfer error $\mathcal{E}(\pi)=|J_{\text{real}}(\pi)-J_{\text{vrt}}(\pi)|$ is bounded by:

$$
\mathcal{E}(\pi)\leq\underbrace{\delta(1-C_{\text{tool}})}_{\text{Dynamics Gap}}+\underbrace{\alpha\cdot D_{\text{KL}}(\pi_{\text{vrt}}|\pi_{\text{pst}})}_{\text{Action Bias Bound}}+\underbrace{\beta(1-\Phi_{\text{plan}}\cdot R_{\text{result}})}_{\text{Goal Alignment Error}}
$$

Theorem 4.1 quantifies how the sim-to-real divergence is suppressed: (i) Dynamics Gap is minimized by $C_{\text{tool}}$, ensuring virtual procedural logic mirrors real API behavior; (ii) Action Bias Bound is constrained by the PST prior, which prevents policy drift in the absence of real visual feedback by maintaining consistency with expert decision-making; (iii) Goal Alignment Error is mitigated by the coupling of $\Phi_{\text{plan}}$ and $R_{\text{result}}$, ensuring the virtual completion objective serves as a reliable proxy for real-world success.

###### Theorem 4.2 (Real-World Improvement of VRL).

Under the error bound $\mathcal{E}$, the real-world performance gain depends on the dominance of Causal Improvement over Transfer Loss:

$$
J_{\text{real}}(\pi_{\text{VRL}})-J_{\text{real}}(\pi_{\text{pst}})\geq\underbrace{\Gamma\cdot\mathbb{E}_{\pi}[\Delta R_{\text{vrt}}]}_{\text{Causal Improvement}}-\underbrace{\mathcal{E}(\pi)}_{\text{Transfer Loss}}
$$

where $\Gamma=C_{\text{tool}}\cdot\Phi_{\text{plan}}\cdot\kappa(\pi_{\text{pst}})$ is the effectiveness coefficient, and $\kappa(\pi_{\text{pst}})$ denotes the anchoring strength of the PST prior in constraining policy exploration. Virtual reward $R_{\text{vrt}}$ consisting of $R_{\text{plan}}$ and $R_{\text{fine}}$, and $\mathbb{E}_{\pi}[\Delta R_{\text{vrt}}]$ denotes the expected increment of virtual reward, representing the agent's logic optimization in planning and execution.

The practical transferability of VRL is validated by the convergence behavior in our experiments, where the agent achieves an average virtual reward exceeding 95%. This saturation of total virtual reward $R_{\text{vrt}}$ indicates that the Causal Improvement term is maximized, providing a substantial logical buffer to offset transfer discrepancies. By substituting these empirical results into Theorem 4.1, we observe that the Action Bias Bound is strictly suppressed by the PST prior, while the Goal Alignment Error is mitigated by the coupling of $\Phi_{\text{plan}}$ and $R_{\text{result}}$, remaining stable as the agent masters structural completion. Consequently, the Transfer Loss $\mathcal{E}(\pi)$ is primarily governed by the Dynamics Gap $\delta(1-C_{\text{tool}})$. This reveals a critical insight: VRL efficacy is fundamentally a function of generative tool quality. As $C_{\text{tool}}$ increases—meaning the underlying visual creation tools become more reliable and follow procedural logic more closely—the transfer loss diminishes, allowing the massive logical gains from virtual training to translate effectively into superior real-world visual quality. Therefore, we derive the following corollary:

###### Corollary 4.3 (Fidelity-Anchored Transfer).

Provided the virtual reward $R_{\text{vrt}}$ reaches a near-optimal level, the real-world gain of VRL is monotonically non-decreasing with respect to $C_{\text{tool}}$.

#### 4.3.4 Plan-Driven Reward Design

Theorems 4.1 and 4.2 indicate that real-world improvement critically depends on planning quality. Motivated by this insight, we adopt a plan-driven reward that enforces causal dependency between planning and execution:

$$
\displaystyle R_{\text{vrt}}=R_{\text{plan}}\times R_{\text{fine}}[R_{\text{tool}}+R_{\text{format}}+R_{\text{result}}+R_{\text{traj}}+R_{\text{cons}}].
$$

Here, $R_{\text{plan}}$ measures plan correctness, while $R_{\text{fine}}$ captures execution-level structural validity. The multiplicative coupling ensures that execution alone cannot achieve high reward without a valid plan, and maximal reward is obtained only when a correct plan is faithfully executed. This mechanism directly aligns with Theorem 4.2, promoting robust long-horizon planning and tool-use strategies within virtual training.

## 5 Experiment

### 5.1 VisGenBench

Existing video generation benchmark VBench-2.0 [^40] has made significant contributions to evaluating the quality of individual-generated videos. But it lacks the capability to evaluate multi-step visual creation trajectories that involve complex tool invocation and long-horizon planning. While ComfyBench [^37] attempts to assess multi-step trajectories, it is specifically designed for ComfyUI [^3] and evaluates agent performance based solely on ComfyUI execution success, making it unsuitable for general API-based tool invocation scenarios. To address this critical gap, we introduce VisGenBench, a comprehensive benchmark designed for evaluating visual generation agentic models that operate through multi-step tool invocation to accomplish complex image and video creation tasks.

Table 1: Test dataset composition of VisGenBench, with 400 image tasks and 800 video tasks.

<table><tbody><tr><td rowspan="2">Type</td><td>Content</td><td>Content</td><td>Object</td><td>Scene</td><td>Style</td><td rowspan="2">Variety</td><td>Visual</td><td>Video</td><td>Video</td></tr><tr><td>Creative</td><td>Match</td><td>Consistency</td><td>Consistency</td><td>Consistency</td><td>Amount</td><td>Duration</td><td>Storyboard</td></tr><tr><td>Image Tasks</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>100</td><td>–</td><td>–</td></tr><tr><td>Video Tasks</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>100</td><td>200</td><td>200</td></tr><tr><td>Total</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>200</td><td>200</td><td>200</td></tr></tbody></table>

#### 5.1.1 Test Dataset Composition

As shown in Tab. 1, the VisGenBench consists of a total of 1.2k test samples, including 400 image-generation tasks and 800 video-generation tasks. Each task is designed to reflect multi-step creation trajectories, requiring to generation of many images and videos. The benchmark spans 10 evaluation dimensions and covers 35+ real-world application scenarios, encompassing domains such as advertising, storytelling, entertainment, animation, etc.

#### 5.1.2 Evaluation Framework

The VisGenBench evaluation framework integrates both objective and subjective assessments to measure an agent’s ability to perform multi-step visual generation tasks.

Objective Evaluation Objective evaluation focuses on quantifiable and automatically measurable aspects of the generated content. Specifically, it consists of two components: (1) Success Rate: Measures whether the model successfully returns valid images/videos when requested by user. A generation containing the correct modality is counted as Success. (2) Basic Visual Attributes: Quantitative evaluation of the generated results, including visual quantity, video storyboard count, and video duration. These attributes are automatically assessed using standardized tools.

Subjective Evaluation Subjective aspects such as visual consistency, diversity, storytelling quality, and audio perception cannot be fully captured through traditional metrics. We therefore introduce a VLM-Grader with pre-defined fine-grained scoring rubrics, implemented using the Gemini2.5-Pro model. For each subjective evaluation dimension, we define a tailored meta evaluation list—a structured rubric containing detailed scoring items (e.g., character consistency, style coherence, narrative flow, audio synchronization, etc). Gemini2.5-Pro provides a meta-evaluation score for each meta-item, and the aggregated score forms the overall result for that dimension. To align automated scoring with human judgment, we calibrate Gemini2.5-Pro's meta-evaluation intensity on VisGenBench. This ensures that both mean scores and relative rankings evaluated by Gemini2.5-Pro remain consistent with expert human assessments, achieving a human-aligned evaluation process.

Table 2: Comparisons on VisGenBench by VLM Evaluation. S-Rate: Success Rate, O-Score: Overall Score. The best and second-best results are highlighted.

| Method | Creative | Match | Object | Scene | Style | Variety | Amount | Duration | Storyboard | S-Rate | O-Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-5 | 0.683 | 0.641 | 0.593 | 0.579 | 0.638 | 0.232 | 0.620 | 0.263 | 0.660 | 0.863 | 0.577 |
| Gemini2.5-Pro | 0.777 | 0.802 | 0.625 | 0.602 | 0.573 | 0.345 | 0.540 | 0.376 | 0.700 | 0.933 | 0.627 |
| Qwen3-VL-8B-Tk | 0.104 | 0.078 | 0.100 | 0.065 | 0.109 | 0.014 | 0.160 | 0.034 | 0.040 | 0.142 | 0.085 |
| VisionCreator-8B | 0.651 | 0.661 | 0.645 | 0.638 | 0.595 | 0.211 | 0.480 | 0.429 | 0.580 | 0.925 | 0.581 |

Table 3: Comparisons on VisGenBench by Human Evaluation. All models use the new version system prompt, which differs from Tab.2. Overall Score = (Success Rate of Image $\times$ Human Evaluation of Image $+$ Success Rate of Video $\times$ Human Evaluation of Video) $/$ 2. The performance comparisons of all detailed human evaluation dimensions are shown in Fig. 1.

<table><tbody><tr><td rowspan="2">Model</td><td colspan="2">Success Rate</td><td colspan="2">Human Evaluation</td><td rowspan="2">Overall Score</td></tr><tr><td>Image</td><td>Video</td><td>Image</td><td>Video</td></tr><tr><td>GPT-5</td><td>95.95%</td><td>93.00%</td><td>3.52</td><td>3.25</td><td>3.19</td></tr><tr><td>Gemini2.5-Pro</td><td>91.00%</td><td>84.00%</td><td>3.53</td><td>3.35</td><td>3.01</td></tr><tr><td>Qwen3-VL-32B-Thinking</td><td>97.00%</td><td>93.00%</td><td>3.47</td><td>3.23</td><td>3.18</td></tr><tr><td>Qwen3-VL-32B-RL</td><td>91.00%</td><td>87.00%</td><td>3.51</td><td>3.40</td><td>3.07</td></tr><tr><td>Qwen3-VL-32B-SFT</td><td>96.00%</td><td>94.00%</td><td>3.53</td><td>3.37</td><td>3.27</td></tr><tr><td>VisionCreator-32B</td><td>99.00%</td><td>96.00%</td><td>3.53</td><td>3.49</td><td>3.42</td></tr></tbody></table>

### 5.2 Results on VisGenBench by VLM Evaluation

As shown in Tab. 2, our VisionCreator-8B demonstrates remarkable performance that is highly competitive with much larger commercial models (GPT-5 and Gemini2.5-Pro), while significantly outperforming its base model Qwen3-VL-8B-Thinking. The key findings highlight several advantages of our approach: (1) Superior Success Rate and Reliability: VisionCreator-8B achieves an impressive success rate of 0.925, surpassing GPT-5 (0.863) and approaching Gemini2.5-Pro (0.933). This demonstrates the effectiveness of our UTPC framework in ensuring task completion reliability, a crucial requirement for practical visual creation applications. (2) Exceptional Consistency Performance: VisionCreator-8B achieves the highest scores in object consistency (0.645) and scene consistency (0.638) among all compared models, including the much larger Gemini2.5-Pro and GPT-5. This validates our model's strong capability in maintaining visual coherence throughout multi-step creation processes, a core benefit of the native agentic architecture. (3) The results validate our core hypothesis: a specialized native visual creation agent, even with significantly fewer parameters, can achieve performance competitive with general-purpose commercial giants through targeted architectural design and training methodology. VisionCreator's particular strengths in success rate and consistency metrics underscore the practical advantages of our UTPC framework for real-world visual content creation applications.

Table 4: Ablation study with VisionCreator-8B on VisGenBench-104 comparing different training strategies. VisGenBench-104 is a sampled subset of VisGenBench. Model configurations: RL1: PST + Result+Format reward); RL2: PST + Plan×(Result+Format) reward; RL3: Qwen3-VL + Plan×(Result+Format) reward; RL4: PST + Plan×Fine reward; v1: 3×VisGenData-4k; v2: 3×VisGenData-4k + General-1M; v3: 20×VisGenData-4k + General-1M; v4: PST + 3×VisGenData-4k + General-1%; v5: PST + 3×VisGenData-4k + General-5%; v6: PST + 3×VisGenData-4k + General-10%; v7: PST + 3×VisGenData-4k + General-20%.

| Method | Creative | Match | Object | Scene | Style | Variety | Amount | Duration | Storyboard | S-Rate | O-Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL1 | 0.534 | 0.817 | 0.694 | 0.547 | 0.579 | 0.249 | 1.000 | 0.397 | 0.625 | 0.904 | 0.634 |
| RL2 | 0.579 | 0.808 | 0.677 | 0.479 | 0.558 | 0.265 | 0.800 | 0.478 | 0.875 | 0.942 | 0.644 |
| RL3 | 0.671 | 0.674 | 0.621 | 0.622 | 0.555 | 0.217 | 0.800 | 0.513 | 0.750 | 0.885 | 0.631 |
| RL4 | 0.573 | 0.794 | 0.672 | 0.696 | 0.569 | 0.150 | 1.000 | 0.534 | 0.625 | 0.925 | 0.654 |
| v1 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 | 0.007 |
| v2 | 0.230 | 0.334 | 0.382 | 0.339 | 0.396 | 0.163 | 0.600 | 0.134 | 0.500 | 0.490 | 0.357 |
| v3 | 0.262 | 0.422 | 0.300 | 0.260 | 0.473 | 0.068 | 0.600 | 0.100 | 0.250 | 0.481 | 0.322 |
| v4 | 0.283 | 0.468 | 0.399 | 0.295 | 0.318 | 0.000 | 0.600 | 0.183 | 0.000 | 0.442 | 0.299 |
| v5 | 0.266 | 0.361 | 0.366 | 0.246 | 0.201 | 0.084 | 0.200 | 0.029 | 0.125 | 0.490 | 0.237 |
| v6 | 0.239 | 0.310 | 0.326 | 0.194 | 0.273 | 0.149 | 0.600 | 0.098 | 0.125 | 0.413 | 0.273 |
| v7 | 0.420 | 0.701 | 0.430 | 0.447 | 0.430 | 0.028 | 0.600 | 0.344 | 0.375 | 0.625 | 0.440 |

### 5.3 Results on VisGenBench by Human Evaluation

In addition to automated VLM-based evaluation (Tab. 2), we conduct a thorough human evaluation to assess the perceptual quality of multi-step visual creation tasks, including images and videos (Tab. 3), which shows that: (1) Overall Findings: VisionCreator-32B achieves the highest Overall Score of 3.42, surpassing both GPT-5 (3.19) and Gemini2.5-Pro (3.01). This indicates that our UTPC framework not only ensures task success in an automated setting but also delivers outputs that are qualitatively preferred by human evaluators. (2) Image vs. Video Performance: VisionCreator-32B excels across both modalities, with 99% image success and 96% video success, accompanied by strong human evaluation scores (3.53 for images, 3.49 for videos). This balanced performance highlights the model's capability to maintain coherent multi-step planning and execution for both static and dynamic content. (3) Implications: The human evaluation corroborates trends observed in VLM-based metrics, validating that the model's planning-driven reward design and VRL training not only improve automated success metrics but also enhance perceptual quality, consistency, and user satisfaction in real-world multi-step visual creation tasks.

### 5.4 Ablation Studies

We conduct ablation studies on sampled VisGenBench-104, where key findings from Tab. 4 include: (1) Effectiveness of PST. Our PST with v7 (PST + 3×VisGenData-4k + General-20%) achieves significant improvement over SFT with v2 (3×VisGenData-4k + General-1M) (0.440 vs. 0.357). Performance improves with increasing general data ratio (v4→v5→v6→v7), confirming balanced specialization prevents overfitting while maintaining generalization. (2) Data Configuration Strategies. Simply increasing specialized data scale does not guarantee improvement. v3 (20×VisGenData-4k + General-1M) underperforms v2 (3×VisGenData-4k + General-1M) (0.322 vs. 0.357), indicating excessive data repetition causes overfitting. Our PST strategy achieves better performance through optimized data ratios. (3) Virtual Reinforcement Learning. All VRL models substantially outperform SFT variants. RL4 (PST + Plan×Fine reward) improves Overall Score by 49% over the best PST model v7 (0.654 vs. 0.440), demonstrating VRL's effectiveness. (4) Reward Function Designs. Building upon RL1, RL2 (PST + Plan×(Result+Format) reward) which incorporates additional plan reward, demonstrates improved performance with a higher Success Rate (0.942 vs. 0.904) and Overall Score (0.644 vs. 0.634). RL4 achieves the best Overall Score (0.654) and demonstrates strong comprehensive performance across multiple dimensions, proving fine-grained rewards enhance model capability. (5) Importance of Pre-training Foundation RL2 (PST + Plan×(Result+Format) reward) outperforms RL3 (Qwen3-VL + Plan×(Result+Format) reward) (0.644 vs. 0.631) despite identical rewards, with RL2 achieving a notably higher Success Rate of 0.942 compared to 0.885 for RL3, validating PST provides a stronger foundation for RL training.

![Refer to caption](https://arxiv.org/html/2603.02681v1/x3.png)

Figure 9: Visualization comparisons of consistency.

## 6 Conclusion

We present VisionCreator, a native visual-generation agent that unifies Understanding, Thinking, Planning, and Creation (UTPC) in an end-to-end framework. Our contributions include: (1) VisGenData-4k with UTPC structures via metacognition-based VisionAgent; (2) Progressive Specialization Training and Virtual Reinforcement Learning for stable capability acquisition; (3) VisGenBench for multi-step visual creation evaluation. Experiments show VisionCreator outperforms larger closed-source models, validating our approach. This work establishes a foundation for visual-generation agentic systems and autonomous creative content generation.

## Detailed Theoretical Derivations of VRL Theorems

This appendix provides detailed mathematical derivations and proofs for the two VRL theorems presented in the main text. The derivation process is divided into three stages: formal modeling and definitions, derivation of the error upper bound (Theorems 4.1), and analysis of performance improvement (Theorems 4.2).

### Stage 1: Formal Modeling and Definitions

We first formalize the agent's policy, environment, and rewards to establish the foundation for subsequent derivations.

#### 1.1 Formalization of Environment and Policy

Definition A.1 (MDP Tuple): The real-world task is modeled as a Markov Decision Process (MDP), denoted as $\mathcal{M}_{\text{real}}=(\mathcal{S},\mathcal{A},P_{\text{real}},R_{\text{vision}},\rho_{0},\gamma)$.

- $\mathcal{S}$: State space, containing multimodal observations $o_{t}$ (textual instructions, tool feedback, virtual visual states).
- $\mathcal{A}$: Action space, containing reasoning tokens, planning steps, and tool invocations.
- $P_{\text{real}}(s^{\prime}|s,a)$: Dynamic transition probability of the real environment.
- $R_{\text{vision}}(s,a,s^{\prime})$: Real reward function, measuring the perceptual quality of generated content (e.g., aesthetics, alignment).
- $\rho_{0}$: Initial state distribution.
- $\gamma\in(0,1)$: Discount factor.

Definition A.2 (Virtual Environment): The virtual environment is $\mathcal{M}_{\text{vrt}}=(\mathcal{S},\mathcal{A},P_{\text{vrt}},R_{\text{vrt}},\rho_{0},\gamma)$. Its core differences are:

- $P_{\text{vrt}}(s^{\prime}|s,a)$: Tool dynamics simulated by VisGenEnv, with fidelity quantified by the tool capability $C_{\text{tool}}\in[0,1]$.
- $R_{\text{vrt}}$: Virtual reward function, composed of $R_{\text{plan}}$ and $R_{\text{fine}}$ according to the plan-driven reward design. It is a structural proxy reward that substitutes for the computationally infeasible $R_{\text{vision}}$ in the virtual environment.

Definition A.3 (Policy and Return): Let $\pi$ be a policy (mapping from states to actions). The expected discounted return of policy $\pi$ in environment $\mathcal{M}$ is defined as:

$$
J(\pi;\mathcal{M})=\mathbb{E}_{\tau\sim(\pi,\mathcal{M})}\left[\sum_{t=0}^{\infty}\gamma^{t}R(s_{t},a_{t},s_{t+1})\right].
$$

Here, the trajectory $\tau=(s_{0},a_{0},s_{1},a_{1},\dots)$ is generated by $s_{0}\sim\rho_{0}$, $a_{t}\sim\pi(\cdot|s_{t})$, and $s_{t+1}\sim P(\cdot|s_{t},a_{t})$. For brevity, we denote $J_{\text{real}}(\pi)=J(\pi;\mathcal{M}_{\text{real}})$ and $J_{\text{vrt}}(\pi)=J(\pi;\mathcal{M}_{\text{vrt}})$.

#### 1.2 Key Variables and Core Assumptions

Definition A.4 (Key Variables):

- Tool capability $C_{\text{tool}}$: Measures how well the virtual environment dynamics $P_{\text{vrt}}$ approximate the real dynamics $P_{\text{real}}$. $C_{\text{tool}}=1$ indicates perfect simulation.
- PST prior $\pi_{\text{pst}}$: The initialization policy obtained through Progressive Specialization Training (PST). Its behavioral distribution on real expert data is denoted as $d_{\text{pst}}(s,a)$.
- Plan sufficiency $\Phi_{\text{plan}}\in[0,1]$: Measures the strength of the causal link between a "logically correct" plan and the final "high-quality visual output".
- Result reward $R_{\text{result}}\in[0,1]$: A subcomponent of $R_{\text{fine}}$ that evaluates whether the task is structurally completed (e.g., number of images, video duration).

Assumption A.1 (Dynamic Difference Upper Bound): There exists a constant $\delta>0$ related to environment complexity such that for all state-action pairs $(s,a)$,

$$
|P_{\text{real}}(\cdot|s,a)-P_{\text{vrt}}(\cdot|s,a)|_{1}\leq\delta(1-C_{\text{tool}}).
$$

This assumption stems from the high-fidelity simulation design of VisGenEnv: higher tool capability $C_{\text{tool}}$ leads to smaller differences between virtual and real transitions.

Assumption A.2 (Reward Proxy Error): The relationship between the real visual reward $R_{\text{vision}}$ and the proxy reward is modulated by plan sufficiency $\Phi_{\text{plan}}$ and result reward $R_{\text{result}}$. There exists a constant $\beta>0$ such that for meaningful trajectories (i.e., when planning logic is correct), the reward difference satisfies:

$$
|R_{\text{vision}}(s,a,s^{\prime})-\Phi_{\text{plan}}\cdot R_{\text{result}}(s,a,s^{\prime})|\leq\beta(1-\Phi_{\text{plan}}\cdot R_{\text{result}}).
$$

This assumption reflects the design philosophy of LtrReward: when planning is sufficient ($\Phi_{\text{plan}}\approx 1$) and the task is perfectly completed structurally ($R_{\text{result}}\approx 1$), the real visual quality also tends to be high.

Assumption A.3 (KL Constraint on Policy Deviation): The policy $\pi_{\text{vrt}}$ trained in the virtual environment differs from the PST prior $\pi_{\text{pst}}$ in the state-action distribution. This difference can be measured by the KL divergence $D_{\text{KL}}(\pi_{\text{vrt}}|\pi_{\text{pst}})$, and its impact on the return difference is linearly bounded. That is, there exists a constant $\alpha>0$ such that the related performance difference is constrained by it.

Definition A.5 (Sim-to-Real Error): For a given policy $\pi$, its sim-to-real error is defined as:

$$
\mathcal{E}(\pi)=|J_{\text{real}}(\pi)-J_{\text{vrt}}(\pi)|.
$$

### Stage 2: Derivation of Theorems (Virtual-to-Real Error Upper Bound)

Theorems 4.1 (Virtual-to-Real Error Upper Bound) Restated: Under Assumptions A.1, A.2, A.3, for any policy $\pi$ (trained in the virtual environment, denoted as $\pi_{\text{vrt}}$), its sim-to-real error $\mathcal{E}(\pi)$ satisfies:

$$
\mathcal{E}(\pi)\leq\underbrace{\delta(1-C_{\text{tool}})}_{\text{Dynamics Gap}}+\underbrace{\alpha\cdot D_{\text{KL}}(\pi_{\text{vrt}}|\pi_{\text{pst}})}_{\text{Action Bias Bound}}+\underbrace{\beta(1-\Phi_{\text{plan}}\cdot R_{\text{result}})}_{\text{Goal Alignment Error}}.
$$

Proof:

We decompose the total error into three separately bounded components via the triangle inequality and constrain each using the above assumptions.

Step 2.1: Decompose Total Error Consider an intermediate environment $\mathcal{M}_{\text{hybrid}}=(\mathcal{S},\mathcal{A},P_{\text{vrt}},$ $R_{\text{vision}},\rho_{0},\gamma)$, which uses the virtual environment dynamics $P_{\text{vrt}}$ but retains the real reward $R_{\text{vision}}$. Denote $J_{\text{hybrid}}(\pi)=J(\pi;\mathcal{M}_{\text{hybrid}})$. Then:

$$
\displaystyle\mathcal{E}(\pi)
$$
 
$$
\displaystyle=|J_{\text{real}}(\pi)-J_{\text{vrt}}(\pi)|
$$
 
$$
\displaystyle\leq|J_{\text{real}}(\pi)-J_{\text{hybrid}}(\pi)|\quad
$$
 
$$
\displaystyle+|J_{\text{hybrid}}(\pi)-J_{\text{vrt}}^{\text{ideal}}(\pi)|
$$
 
$$
\displaystyle+|J_{\text{vrt}}^{\text{ideal}}(\pi)-J_{\text{vrt}}(\pi)|.
$$

Here $J_{\text{vrt}}^{\text{ideal}}(\pi)$ represents the ideal return under dynamics $P_{\text{vrt}}$ and reward $R_{\text{vrt}}$ with the policy perfectly constrained by the PST prior (no deviation). We next upper bound each term.

Step 2.2: Bounding Term I (Dynamics Gap) Term I measures the return difference due to the difference between dynamic models $P_{\text{real}}$ and $P_{\text{vrt}}$. According to Assumption A.1 and the Performance Difference Lemma, for any policy $\pi$,

$$
|J_{\text{real}}(\pi)-J_{\text{hybrid}}(\pi)|\leq\frac{\gamma\cdot\delta(1-C_{\text{tool}})}{(1-\gamma)^{2}}\cdot\max_{s,a}|R_{\text{vision}}(s,a)|.
$$

Let $R_{\text{max}}=\max_{s,a}|R_{\text{vision}}(s,a)|$ and define $\delta^{\prime}=\frac{\gamma R_{\text{max}}}{(1-\gamma)^{2}}\delta$, we obtain:

$$
\text{Term I}\leq\delta^{\prime}(1-C_{\text{tool}}).
$$

In the theorem statement, constant factors are absorbed into $\delta$, so we have Term I $\leq\delta(1-C_{\text{tool}})$.

Step 2.3: Bounding Term II (Reward Gap) Term II measures the difference between using the real reward $R_{\text{vision}}$ and using the proxy reward $\Phi_{\text{plan}}\cdot R_{\text{result}}$ (as the core part of $R_{\text{vrt}}$) under the same dynamics. According to Assumption A.2, for each step in the trajectory, the reward difference is bounded. Applying the Performance Difference Lemma (reward difference part) again yields:

$$
|J_{\text{hybrid}}(\pi)-J_{\text{vrt}}^{\text{ideal}}(\pi)|\leq\frac{\beta(1-\Phi_{\text{plan}}\cdot R_{\text{result}})}{1-\gamma}.
$$

Define $\beta^{\prime}=\beta/(1-\gamma)$, then Term II $\leq\beta^{\prime}(1-\Phi_{\text{plan}}\cdot R_{\text{result}})$. In the theorem statement, $\beta^{\prime}$ is written as $\beta$.

Step 2.4: Bounding Term III (Policy Bias) Term III measures the return loss due to the deviation of the virtually trained policy $\pi_{\text{vrt}}$ from the ideal PST prior $\pi_{\text{pst}}$. According to Assumption A.3, there exists a constant $\alpha>0$ such that:

$$
|J_{\text{vrt}}^{\text{ideal}}(\pi)-J_{\text{vrt}}(\pi)|\leq\alpha\cdot D_{\text{KL}}(\pi_{\text{vrt}}|\pi_{\text{pst}}).
$$

This assumption stems from the "anchoring" effect of the PST prior on the policy exploration space, preventing catastrophic policy drift in the absence of real visual feedback.

Step 2.5: Combining Error Upper Bounds Summing the upper bounds of Term I, II, and III, we obtain:

$$
\mathcal{E}(\pi)\leq\delta^{\prime}(1-C_{\text{tool}})+\beta^{\prime}(1-\Phi_{\text{plan}}\cdot R_{\text{result}})+\alpha\cdot D_{\text{KL}}(\pi_{\text{vrt}}|\pi_{\text{pst}}).
$$

Relabeling constants $\delta^{\prime}\to\delta$, $\beta^{\prime}\to\beta$ yields the form in Theorems 4.1. $\blacksquare$

### Stage 3: Derivation of Theorems (Real-World Performance Improvement Lower Bound)

Theorems 4.2 (Real-World Improvement of VRL) Restated: Let $\pi_{\text{pst}}$ be the initial policy after PST training, and $\pi_{\text{VRL}}$ be the policy optimized through Virtual Reinforcement Learning (VRL). Define the virtual optimization gain as $\Delta_{\text{vrt}}=J_{\text{vrt}}(\pi_{\text{VRL}})-J_{\text{vrt}}(\pi_{\text{pst}})=\mathbb{E}_{\pi}[\Delta(R_{\text{plan}}+R_{\text{fine}})]$. Then, under the error bound of Theorems 4.1, the real-world performance improvement satisfies:

$$
J_{\text{real}}(\pi_{\text{VRL}})-J_{\text{real}}(\pi_{\text{pst}})\geq\underbrace{\Gamma\cdot\Delta_{\text{vrt}}}_{\text{Causal Improvement}}-\underbrace{\mathcal{E}(\pi_{\text{VRL}})}_{\text{Transfer Loss}},
$$

where $\Gamma=C_{\text{tool}}\cdot\Phi_{\text{plan}}\cdot\kappa(\pi_{\text{pst}})$ is the effectiveness coefficient, and $\kappa(\pi_{\text{pst}})\in(0,1]$ denotes the Anchoring Strength of the PST prior in constraining policy exploration.

Proof:

Step 3.1: Establish Inequality Based on Error Bound From Theorems 4.1, for any policy $\pi$, we have $J_{\text{real}}(\pi)\geq J_{\text{vrt}}(\pi)-\mathcal{E}(\pi)$. Applying this inequality to $\pi_{\text{VRL}}$ and $\pi_{\text{pst}}$ respectively:

$$
\displaystyle J_{\text{real}}(\pi_{\text{VRL}})
$$
 
$$
\displaystyle\geq J_{\text{vrt}}(\pi_{\text{VRL}})-\mathcal{E}(\pi_{\text{VRL}}),
$$
$$
\displaystyle J_{\text{real}}(\pi_{\text{pst}})
$$
 
$$
\displaystyle\geq J_{\text{vrt}}(\pi_{\text{pst}})-\mathcal{E}(\pi_{\text{pst}}).
$$

Subtracting the second inequality from the first yields:

$$
J_{\text{real}}(\pi_{\text{VRL}})-J_{\text{real}}(\pi_{\text{pst}})\geq\left[J_{\text{vrt}}(\pi_{\text{VRL}})-J_{\text{vrt}}(\pi_{\text{pst}})\right]-\left[\mathcal{E}(\pi_{\text{VRL}})-\mathcal{E}(\pi_{\text{pst}})\right].
$$

Since $\pi_{\text{pst}}$ itself is trained on real expert data, its sim-to-real error $\mathcal{E}(\pi_{\text{pst}})$ is expected to be small (aligned during PST). Therefore, the lower bound of performance improvement is mainly affected by the error $\mathcal{E}(\pi_{\text{VRL}})$ of $\pi_{\text{VRL}}$. Conservatively setting the transfer loss term as $\mathcal{E}(\pi_{\text{VRL}})$ gives:

$$
J_{\text{real}}(\pi_{\text{VRL}})-J_{\text{real}}(\pi_{\text{pst}})\geq\Delta_{\text{vrt}}-\mathcal{E}(\pi_{\text{VRL}}).\quad\text{(1)}
$$

Step 3.2: Relating Virtual Gain to Real Gain (Causal Improvement) The $\Delta_{\text{vrt}}$ in inequality (1) is the gain in virtual reward. We need to relate it to real performance improvement. This relies on a core idea: optimizing "planning and execution logic" in the virtual environment, as long as the simulation is sufficiently credible, causally leads to improved real-world visual quality. Define the effectiveness coefficient $\Gamma$, which quantifies the expected increment in real reward per unit increment in virtual reward. We model it as the product of three key factors:

- $C_{\text{tool}}$: Tool capability determines the probability of logical execution being reproduced in reality.
- $\Phi_{\text{plan}}$: Plan sufficiency determines the strength of association between correct logic and high-quality output.
- $\kappa(\pi_{\text{pst}})$: Anchoring strength of the PST prior, indicating the degree to which the policy remains in a "reasonable" distribution region during VRL optimization, with $\kappa\in(0,1]$. Strong anchoring ($\kappa\approx 1$) ensures the optimization direction remains effective in the real world.

Therefore, we assume a monotonic relationship:

$$
J_{\text{real}}(\pi_{\text{VRL}})-J_{\text{real}}(\pi_{\text{pst}})\geq\Gamma\cdot\Delta_{\text{vrt}}-\mathcal{E}(\pi_{\text{VRL}}),\quad\text{where }\Gamma=C_{\text{tool}}\cdot\Phi_{\text{plan}}\cdot\kappa(\pi_{\text{pst}}).\quad\text{(2)}
$$

When $\Gamma>0$, the logical improvement brought by virtual optimization can be partially translated into real-world improvement.

Step 3.3: Derive the Final Lower Bound Substituting $\Delta_{\text{vrt}}=\mathbb{E}_{\pi}[\Delta R_{\text{vrt}}]$ into inequality (2) yields the lower bound stated in Theorems 4.2:

$$
J_{\text{real}}(\pi_{\text{VRL}})-J_{\text{real}}(\pi_{\text{pst}})\geq\Gamma\cdot\mathbb{E}_{\pi}[\Delta R_{\text{vrt}}]-\mathcal{E}(\pi_{\text{VRL}}).
$$

Step 3.4: Condition for Non-Negative Improvement From the inequality in Theorems 4.2, the sufficient condition for non-negative improvement in real-world performance (i.e., $J_{\text{real}}(\pi_{\text{VRL}})\geq J_{\text{real}}(\pi_{\text{pst}})$) is directly obtained as:

$$
\Gamma\cdot\mathbb{E}_{\pi}[\Delta R_{\text{vrt}}]\geq\mathcal{E}(\pi_{\text{VRL}}).
$$

This means that the Causal Improvement brought by virtual training must be sufficient to cover the Transfer Loss arising from simulation imperfections. This does not require $C_{\text{tool}}=1$ or $\Phi_{\text{plan}}=1$; as long as their product combined with the anchoring strength is large enough to make $\Gamma$ sufficiently large, and VRL can effectively increase $\Delta R_{\text{vrt}}$ (as shown in experiments where virtual reward exceeds 95%), positive transfer is guaranteed. $\blacksquare$

### Summary

Through formal modeling, this derivation decomposes the challenge of sim-to-real transfer into differences at the dynamic, reward, and policy levels, and quantifies their upper bounds using key variables such as tool capability, plan sufficiency, and PST prior. Theorems 4.1 shows that systematic error can be controlled by improving tool fidelity, strengthening PST anchoring, and optimizing plan-result alignment. Theorems 4.2 further proves that as long as virtual training can effectively enhance the agent's logical capabilities (Causal Improvement) and this improvement outweighs the bounded systematic error (Transfer Loss), performance improvement in the real world is guaranteed. This provides a solid theoretical foundation for the application of virtual reinforcement learning in high-dimensional, long-horizon tasks such as visual creation.

Table 5: Human Evaluation of Detailed Dimensions on VisGenBench-Image (Score = Success Rate $\times$ Human Evaluation Score)

<table><tbody><tr><td rowspan="2">Model</td><td>Semantic</td><td>Style</td><td>Emotion</td><td>Subject</td><td>Design</td><td>Visual</td><td>Text</td><td rowspan="2">Creativity</td><td rowspan="2">Overall</td></tr><tr><td>Matching</td><td>Matching</td><td>Matching</td><td>Consistency</td><td>Integrity</td><td>Integrity</td><td>Quality</td></tr><tr><td>GPT-5</td><td>3.4883</td><td>3.6214</td><td>2.9656</td><td>3.6024</td><td>3.4408</td><td>3.4218</td><td>2.7565</td><td>3.4408</td><td>3.3458</td></tr><tr><td>Gemini2.5-Pro</td><td>3.3943</td><td>3.5399</td><td>2.8119</td><td>3.4034</td><td>3.2214</td><td>3.2669</td><td>2.7300</td><td>3.3215</td><td>3.2123</td></tr><tr><td>Qwen3-VL-32B-Tk</td><td>3.4435</td><td>3.7248</td><td>2.9876</td><td>3.5890</td><td>3.4047</td><td>3.4823</td><td>2.8130</td><td>3.4726</td><td>3.3659</td></tr><tr><td>Qwen3-VL-32B-SFT</td><td>3.3504</td><td>3.8016</td><td>2.8896</td><td>3.7632</td><td>3.4368</td><td>3.5040</td><td>2.8224</td><td>3.5232</td><td>3.3888</td></tr><tr><td>VisionCreator-32B</td><td>3.6432</td><td>3.8412</td><td>3.1581</td><td>3.7620</td><td>3.4452</td><td>3.6531</td><td>2.8809</td><td>3.5739</td><td>3.4947</td></tr></tbody></table>

Table 6: Human Evaluation of Detailed Dimensions on VisGenBench-Video (Score = Success Rate $\times$ Human Evaluation Score)

Model Script Story- Content Subject Video Visual board Consistency Consistency Effect Motion GPT-5 3.1062 2.9202 3.1434 3.1713 3.0597 3.0039 Gemini2.5-Pro 2.9484 2.6796 3.0156 2.856 2.8728 2.6628 Qwen3-VL-32B-Thinking 3.069 2.9016 3.1713 3.162 2.9574 2.9388 Qwen3-VL-32B-SFT 3.6002 2.867 3.5814 3.3652 3.243 2.9328 VisionCreator-32B 3.5616 3.1872 3.5808 3.4176 3.4752 3.2256

Model Audio-Visual Music Dubbing Subtitle Transition Editing Overall GPT-5 3.0411 3.2643 2.8644 2.9788 2.8812 2.8392 2.814 Gemini2.5-Pro 2.8056 2.7888 2.8728 2.8812 2.8392 2.562 2.814 Qwen3-VL-32B-Thinking 2.9481 2.9202 3.1341 3.069 2.9295 2.8737 3.0039 Qwen3-VL-32B-SFT 3.1772 3.0644 3.2148 3.0174 3.0268 2.9328 3.1678 VisionCreator-32B 3.3792 3.2928 3.3888 3.3216 3.2352 3.1104 3.3504

Table 7: General-purpose Datasets.

<table><tbody><tr><td>Category</td><td>Name</td><td>Quantity</td></tr><tr><td rowspan="3">NLP</td><td>DeepSeek-R1-Distill-110k <sup><a href="#fn:19">19</a></sup></td><td>110k</td></tr><tr><td>LONGCOT-Refine-500K <sup><a href="#fn:26">26</a></sup></td><td>500k</td></tr><tr><td>alpaca-gpt4-data <sup><a href="#fn:24">24</a></sup> <sup><a href="#fn:25">25</a></sup></td><td>100k</td></tr><tr><td>Multimodal</td><td>M3IT <sup><a href="#fn:15">15</a></sup></td><td>1592k</td></tr><tr><td rowspan="6">Tool Calling</td><td>function-calling-chatml <sup><a href="#fn:8">8</a></sup></td><td>112k</td></tr><tr><td>xlam-function-calling-60k <sup><a href="#fn:39">39</a></sup></td><td>60k</td></tr><tr><td>ms-agent <sup><a href="#fn:21">21</a></sup></td><td>600k</td></tr><tr><td>ToolACE <sup><a href="#fn:20">20</a></sup></td><td>11k</td></tr><tr><td>ToolBench <sup><a href="#fn:27">27</a></sup></td><td>123k</td></tr><tr><td>AFM <sup><a href="#fn:17">17</a></sup></td><td>76k</td></tr></tbody></table>

Table 8: Task Distribution in VisGenData-4k Dataset

<table><tbody><tr><td colspan="2">Video Generation Tasks</td><td colspan="2">Image Generation Tasks</td></tr><tr><td>No.</td><td>Task Type</td><td>No.</td><td>Task Type</td></tr><tr><td>1.</td><td>Product marketing videos</td><td>1.</td><td>Product images</td></tr><tr><td>2.</td><td>Public service advertisements</td><td>2.</td><td>Detail pages</td></tr><tr><td>3.</td><td>Corporate promotion videos</td><td>3.</td><td>Key Visual (KV)</td></tr><tr><td>4.</td><td>Brand story videos</td><td>4.</td><td>Landing pages / H5 graphics</td></tr><tr><td>5.</td><td>Event promotion videos</td><td>5.</td><td>Complete brand visual identity</td></tr><tr><td>6.</td><td>Instructional videos</td><td>6.</td><td>Banner graphics</td></tr><tr><td>7.</td><td>Popular science documentaries</td><td>7.</td><td>Official account cover images</td></tr><tr><td>8.</td><td>Music videos (MV)</td><td>8.</td><td>Xiaohongshu covers</td></tr><tr><td>9.</td><td>Concert recordings</td><td>9.</td><td>Marketing posters</td></tr><tr><td>10.</td><td>Variety shows</td><td>10.</td><td>Avatar design</td></tr><tr><td>11.</td><td>Story videos</td><td>11.</td><td>Static emoji generation</td></tr><tr><td>12.</td><td>Video podcasts</td><td>12.</td><td>ICON design</td></tr><tr><td>13.</td><td>Picture books</td><td>13.</td><td>LOGO design</td></tr><tr><td>14.</td><td>Dynamic comics</td><td>14.</td><td>Mini-game UI design</td></tr><tr><td>15.</td><td>Animated short films</td><td>15.</td><td>Character design</td></tr><tr><td>16.</td><td>Animated movies</td><td>16.</td><td>Character action design</td></tr><tr><td>17.</td><td>Game adaptation films</td><td>17.</td><td>Scene design</td></tr><tr><td>18.</td><td>Game videos</td><td>18.</td><td>Storyboards</td></tr><tr><td>19.</td><td>Movies</td><td>19.</td><td>Picture Book</td></tr><tr><td>20.</td><td>Short dramas</td><td>20.</td><td>Stylization</td></tr><tr><td>21.</td><td>Story explanations</td><td>21.</td><td>Realistic Photography</td></tr></tbody></table>

Table 9: VisGenEnv integrates 36 visual creation tools.

<table><tbody><tr><td>Tool Category</td><td>Tool Function</td><td>Tool Name</td></tr><tr><td rowspan="6">Text-to-Text</td><td>Storyboard Text Polishing (Claude)</td><td>tool_prompt_refine</td></tr><tr><td>Storyboard Generation (Claude)</td><td>tool_video_shot_gen</td></tr><tr><td>Script Tool (Claude)</td><td>tool_video_script_gen</td></tr><tr><td>Storyboard Polishing (Claude)</td><td>tool_storyboard_polish</td></tr><tr><td>Script & Storyboard Polishing</td><td>tool_script_storyboard_merge</td></tr><tr><td>Text-to-Video (Veo3)</td><td>tool_text2video_veo</td></tr><tr><td rowspan="5">Text-to-Image</td><td>Text-to-Image (nano-banana)</td><td>tool_text2image_gemini</td></tr><tr><td>Text-to-Image (hunyuan)</td><td>tool_text2image_hunyuan</td></tr><tr><td>Text-to-Image (ByteDance)</td><td>tool_text2image_seed</td></tr><tr><td>Text-to-Image (GPT)</td><td>tool_text2image_gpt</td></tr><tr><td>Text-to-Image (Qwen)</td><td>tool_text2image_qwen</td></tr><tr><td rowspan="3">Image-to-Image</td><td>Image-to-Image (nano-banana)</td><td>tool_image_edit_gemini</td></tr><tr><td>Image-to-Image (Qwen)</td><td>tool_image_edit_qwen</td></tr><tr><td>Image-to-Image (GPT)</td><td>tool_image_edit_gpt</td></tr><tr><td rowspan="2">Image-to-Video</td><td>Image-to-Video (Keling)</td><td>tool_image2video_keling</td></tr><tr><td>Image-to-Video (Veo3)</td><td>tool_image2video_veo3</td></tr><tr><td rowspan="12">Audio Generation</td><td>Music Generation (Suno)</td><td>tool_music_suno</td></tr><tr><td>Video Sound Effect Generation</td><td>tool_sound_fx_gen</td></tr><tr><td>TTS Generation</td><td>tool_tts_generation</td></tr><tr><td>Video Composition (MoviePy)</td><td>tool_video_composite</td></tr><tr><td>Video Clip - MoviePy Post-processing</td><td>tool_video_postprocess</td></tr><tr><td>Video Generation Automation Pipeline</td><td>tool_video_auto_pipeline</td></tr><tr><td>Beat Detection Tool</td><td>tool_beat_detect</td></tr><tr><td>Video Editing (Trim)</td><td>tool_video_trim_edit</td></tr><tr><td>Video Speed Change</td><td>tool_video_speed_adjust</td></tr><tr><td>TTS + Composition Tool</td><td>tool_tts_composite</td></tr><tr><td>Audio Editing</td><td>tool_audio_edit_cut</td></tr><tr><td>Add Subtitles</td><td>tool_subtitle_add_text</td></tr><tr><td>Multimodal</td><td>Video Understanding (Gemini2.5-Pro)</td><td>tool_video_analysis</td></tr><tr><td>Understanding</td><td>Audio Understanding (Gemini2.5-Pro)</td><td>tool_audio_analysis</td></tr><tr><td></td><td>Image Understanding (Gemini2.5-Pro)</td><td>tool_image_analysis</td></tr><tr><td rowspan="5">Other</td><td>Tavily Search - Content Extraction</td><td>tool_search_content</td></tr><tr><td>Inspiration Search</td><td>tool_search_inspire</td></tr><tr><td>Summary Tool</td><td>tool_content_summary</td></tr><tr><td>To-Do List</td><td>tool_task_manager</td></tr><tr><td>HTML Generation Tool</td><td>tool_html_builder</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2603.02681v1/x4.png)

Figure 10: Visualizations for scene consistency.

![Refer to caption](https://arxiv.org/html/2603.02681v1/x5.png)

Figure 11: Visualizations for object consistency.

![Refer to caption](https://arxiv.org/html/2603.02681v1/x6.png)

Figure 12: Demo page.

[^1]: Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. *arXiv preprint arXiv:2311.15127*, 2023.

[^2]: Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In *International conference on machine learning*, pp. 1691–1703. PMLR, 2020.

[^3]: comfyanonymous. Comfyui. [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), 2023. GitHub repository.

[^4]: Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. *arXiv preprint arXiv:2510.26583*, 2025.

[^5]: Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. *arXiv preprint arXiv:2505.14683*, 2025.

[^6]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first International Conference on Machine Learning*, 2024.

[^7]: Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. *arXiv preprint arXiv:2410.13863*, 2024.

[^8]: Glaive AI. Function-calling-chatml: Function calling dataset in chatml format. [https://modelscope.cn/datasets/AI-ModelScope/function-calling-chatml](https://modelscope.cn/datasets/AI-ModelScope/function-calling-chatml), 2023.

[^9]: Litao Guo, Xinli Xu, Luozhou Wang, Jiantao Lin, Jinsong Zhou, Zixin Zhang, Bolan Su, and Ying-Cong Chen. Comfymind: Toward general-purpose generation via tree-based planning and reactive feedback. *arXiv preprint arXiv:2505.17908*, 2025.

[^10]: Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. *arXiv preprint arXiv:2412.04431*, 2024.

[^11]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.

[^12]: Oucheng Huang, Yuhang Ma, Zeng Zhao, Mingrui Wu, Jiayi Ji, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun, and Rongrong Ji. Comfygpt: A self-optimizing multi-agent system for comprehensive comfyui workflow generation. *arXiv preprint arXiv:2503.17671*, 2025.

[^13]: Black Forest Labs. Flux. [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux), 2024.

[^14]: Jinxiang Lai, Jie Zhang, Jun Liu, Jian Li, Xiaocheng Lu, and Song Guo. Spider: Any-to-many multimodal llm. *arXiv preprint arXiv:2411.09439*, 2024.

[^15]: Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, Lingpeng Kong, and Qi Liu. M <sup>3</sup> it: A large-scale dataset towards multi-modal multilingual instruction tuning. In *arXiv preprint arXiv:2306.04387*, June 2023. URL [https://arxiv.org/abs/2306.04387](https://arxiv.org/abs/2306.04387).

[^16]: Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Zichun Liao, Yusuke Kato, Kazuki Kozuka, and Aditya Grover. Omniflow: Any-to-any generation with multi-modal rectified flows. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 13178–13188, 2025a.

[^17]: Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tiannan Wang, Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang, and Wangchunshu Zhou. Chain-of-agents: End-to-end agent foundation models via multi-agent distillation and agentic rl. *arXiv preprint arXiv:2508.13167*, August 2025b. URL [https://arxiv.org/abs/2508.13167](https://arxiv.org/abs/2508.13167).

[^18]: Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. *arXiv preprint arXiv:2412.00131*, 2024.

[^19]: Cong Liu. Chinese-deepseek-r1-distill-data-110k-sft. [https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT), 2025.

[^20]: Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, and Enhong Chen. Toolace: Winning the points of llm function calling. In *International Conference on Learning Representations (ICLR)*, 2025. URL [https://arxiv.org/abs/2409.00920](https://arxiv.org/abs/2409.00920).

[^21]: ModelScope Team. Ms-agent: Modelscope agent sft dataset. [https://modelscope.cn/datasets/iic/ms\_agent](https://modelscope.cn/datasets/iic/ms_agent), 2024.

[^22]: Yatian Pang, Peng Jin, Shuo Yang, Bin Lin, Bin Zhu, Zhenyu Tang, Liuhan Chen, Francis EH Tay, Ser-Nam Lim, Harry Yang, et al. Next patch prediction for autoregressive visual generation. *arXiv preprint arXiv:2412.15321*, 2024.

[^23]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 4195–4205, 2023.

[^24]: Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Alpaca-gpt4-data-en: English instruction following dataset. [https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en), 2023a.

[^25]: Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Alpaca-gpt4-data-zh: Chinese instruction following dataset. [https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh), 2023b.

[^26]: PowerInfer Team. Longcot-refine-500k: Long chain-of-thought reasoning dataset. [https://modelscope.cn/datasets/PowerInfer/LONGCOT-Refine-500K](https://modelscope.cn/datasets/PowerInfer/LONGCOT-Refine-500K), 2025.

[^27]: Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis. In *International Conference on Learning Representations (ICLR)*, 2024. URL [https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789).

[^28]: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.

[^29]: Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022.

[^30]: Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. *arXiv preprint arXiv:2209.14792*, 2022.

[^31]: Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. *arXiv preprint arXiv:2406.06525*, 2024.

[^32]: Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. *arXiv preprint arXiv:2404.02905*, 2024.

[^33]: Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. Automated movie generation via multi-agent cot planning. *arXiv preprint arXiv:2503.07314*, 2025.

[^34]: Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. *arXiv preprint arXiv:2507.18634*, 2025.

[^35]: Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, and Mengyue Wu. Mm-storyagent: Immersive narrated storybook video generation with a multi-agent paradigm across text, image and audio. *arXiv preprint arXiv:2503.05242*, 2025a.

[^36]: Zhenran Xu, Yiyu Wang, Xue Yang, Longyue Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, and Min Zhang. Comfyui-r1: Exploring reasoning models for workflow generation. *arXiv preprint arXiv:2506.09790*, 2025b.

[^37]: Xiangyuan Xue, Zeyu Lu, Di Huang, Zidong Wang, Wanli Ouyang, and Lei Bai. Comfybench: Benchmarking llm-based agents in comfyui for autonomously designing collaborative ai systems. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 24614–24624, 2025.

[^38]: Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. *arXiv preprint arXiv:2408.06072*, 2024.

[^39]: Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Hoang, Shirley Kokane, Weiran Yao, Juntao Tan, Akshara Prabhakar, Haolin Chen, Zhiwei Liu, Yihao Feng, Tulika Awalgaonkar, Rithesh Murthy, Eric Hu, Zeyuan Chen, Ran Xu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, and Caiming Xiong. xlam: A family of large action models to empower ai agent systems. *arXiv preprint arXiv:2409.03215*, September 2024. URL [https://arxiv.org/abs/2409.03215](https://arxiv.org/abs/2409.03215).

[^40]: Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. *arXiv preprint arXiv:2503.21755*, 2025.