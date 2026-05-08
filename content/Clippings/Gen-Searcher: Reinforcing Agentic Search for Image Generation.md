---
title: "Gen-Searcher: Reinforcing Agentic Search for Image Generation"
source: "https://arxiv.org/abs/2603.28767"
published: "2026"
created: "2026-05-07"
description: null
tags: ["clippings"]
arxiv: "2603.28767"
url: "https://arxiv.org/abs/2603.28767"
---

# Gen-Searcher: Reinforcing Agentic Search for Image Generation

Kaituo Feng1 Manyuan Zhang1† Shawn Chen2 Yunlong Lin1 Kaixuan Fan1

Yilei Jiang1 Hongyu Li Dian Zheng1 Chenyang Wang3 Xiangyu Yue1‡

1MMLab, CUHK 2UCLA 3UC Berkeley

Home: https://gen-searcher.vercel.app/

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/60d71ff1f8e17ebe9a1b32d9219ecf319f4aceb172be18287db681c560c13c36.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/d6bee97a976432ce5f775a2ce3725a1a9e211b22b8aa5b5e1857fbf74ddcef34.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/8193ec948dd8518e9ffcdc8c7422446129803a3fa8f6e87a8e4971f266fa77ea.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/c882b234fe5243c8589efa06fa89969d548f29e6b95d8a2677f0319e6d95f004.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/88c427f142787bce86c0568ef5fe1fb8592e6c8cc1818acbe4169774223b02d9.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/7d79d544500d3936aa255e412ce0debee8259199509b73b2fa254b737ade4ca4.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/dd063ef409ef8448e2658b650eb6c1f8f588a6499ba7227a448b221fd231961d.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/efe94c12424525664baa11a8bcd8a6d4b43d3f2f63c9149e1580c17be41d6cd7.jpg]]

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/d7cf691b17779d51d563f21925a41849b8608b031cc748b8e2fa8e91e9989ea5.jpg]]  
Figure 1: Generated images using our proposed Gen-Searcher.

# ABSTRACT

Recent image generation models have shown strong capabilities in generating high-fidelity and photorealistic images. However, they are fundamentally constrained by frozen internal knowledge, thus often failing on real-world scenarios that are knowledge-intensive or require up-to-date information. In this paper, we present Gen-Searcher, as the first attempt to train a search-augmented image generation agent, which performs multi-hop reasoning and search to collect the textual knowledge and reference images needed for grounded generation. To achieve this, we construct a tailored data pipeline and curate two high-quality datasets, Gen-Searcher-SFT-10k and Gen-Searcher-RL-6k, containing diverse search-intensive prompts and corresponding ground-truth synthesis images. We further introduce KnowGen, a comprehensive benchmark that explicitly requires search-grounded external knowledge for image generation and evaluates models from multiple dimensions. Based on these resources, we train Gen-Searcher with SFT followed by agentic reinforcement learning with dual reward feedback, which combines text-based and image-based rewards to provide more stable

and informative learning signals for GRPO training. Experiments show that Gen-Searcher brings substantial gains, improving Qwen-Image by around 16 points on KnowGen and 15 points on WISE. We hope this work can serve as an open foundation for search agents in image generation, and we fully open-source our data, models, and code.

# 1 Introduction

Recent text-to-image generation models have achieved remarkable progress in producing high-fidelity and photorealistic images [1, 2, 3]. Despite these advances, most of them remain fundamentally limited by frozen internal knowledge acquired during pretraining. As a result, these models often struggle with real-world prompts that are knowledgeintensive or require up-to-date information. For instance, generating images involving specific landmarks, public figures, newly released products, or other evolving real-world entities often requires external knowledge that cannot be reliably inferred from the model’s parametric memory alone. In many cases, the necessary information is not directly available from a single source, and instead requires multi-hop search over the web, where the model should iteratively search, browse, and analyze evidence from multiple sources before generation. Currently, only a few advanced proprietary models like Nano Banana Pro [2], support search before generation, yet they remain limited to text search without retrieving visual references.

To mitigate this limitation, prior work has explored RAG-based approaches that retrieve relevant knowledge from external databases to support generation [4, 5, 6]. However, these methods are limited by the coverage and freshness of static databases, which cannot fully capture the vast and evolving knowledge of the real world. In addition, the similarity-based single-round shallow retrieval makes them inadequate for complex real-world queries that require deep search. Besides, a few prompt-based workflows [7, 8] are recently proposed to enhance image generation models by directly searching for information on the web. However, these works rely on manually designed prompting strategies to guide search and generation without training. As a result, the search behavior is often brittle and suboptimal, lacking the ability to adaptively plan search steps, refine queries, or reason over retrieved evidence.

Inspired by the recent success of agentic reinforcement learning (agentic RL) [9, 10] in deep research tasks, we ask a natural question: can we train a search agent for image generation that actively performs multi-hop web search and reasoning to gather knowledge in the web for grounded image generation?

In this paper, we introduce Gen-Searcher, as the first attempt to train a multimodal deep search agent for image generation. To achieve this goal, we build a dedicated data pipeline, in order to overcome the lack of suitable training data for this task. Specifically, we first construct text prompts that require deep web search for image generation through two strategies. Our primary approach uses carefully designed prompt engineering to instruct Gemini 3 Pro [11] to generate multi-hop search-intensive prompts across around 20 diverse categories, including celebrities, anime, physics, chemistry, posters, art, etc. The second approach converts existing deep research datasets into image-generation-oriented prompts, primarily covering general news scenarios. After constructing the prompts, we iteratively employ Gemini 3 Pro together with search tools to produce agentic trajectories, where the system performs search, browsing, and reasoning to gather sufficient information before producing a final search-grounded prompt along with relevant reference images. The resulting prompts are then fed into Nano Banana Pro to synthesize the corresponding images as ground truth. To ensure data quality, we further employ Seed1.8 [12] to score and filter the generated samples. Based on this pipeline, we construct two high-quality training datasets, Gen-Searcher-SFT-10k and Gen-Searcher-RL-6k, as well as KnowGen, a comprehensive benchmark for evaluating search-grounded image generation on real-world, knowledge-intensive prompts. We also introduce K-Score for evaluation on KnowGen benchmark.

Based on the constructed training datasets, we train Gen-Searcher in two stages: SFT equips the model with basic tool-use abilities, enabling multi-step search, browsing, and reasoning for image generation; while agentic RL based on GRPO [13, 14] further optimizes its tool-calling trajectories, encouraging the model to produce higher-quality search-grounded prompts for generation. Notably, we find that this task poses unique challenges: due to the limited capability and high generation variance of open-source image generators such as Qwen-Image [1], sometimes even correct searched information may still fail to produce high-quality images, making pure image-based rewards noisy and unstable. To address this issue, we introduce an additional text-based reward that evaluates whether the collected search-grounded prompt contains sufficient information to generate the target image. We then combine both text-based and image-based rewards to train the agent with GRPO, providing more stable and informative feedback for optimizing the search and reasoning process.

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/f42cd52a422b67aed2bd717e118bf8384482f647a16e68cb98b3489d719d384f.jpg]]  
Figure 2: Our proposed Gen-Searcher enables search-grounded generation in real-world knowledge-intensive scenarios.

We conduct extensive experiments to evaluate the effectiveness of Gen-Searcher. Our method consistently improves image generation performance across different backbones, bringing around 16-point gains for Qwen-Image on KnowGen. We further observe strong transferability: a Gen-Searcher trained with Qwen-Image can be directly applied to Seedream 4.5 [15] and Nano Banana Pro [2] without additional training, yielding about 16-point and 3-point improvements, respectively. Beyond our proposed KnowGen benchmark, we also evaluate on WISE [16], where Gen-Searcher improves Qwen-Image from 0.62 to 0.77, demonstrating its strong generalization in knowledge-intensive image generation tasks.

Our contributions can be summarized as follows:

• We propose Gen-Searcher, the first attempt to explore training a multimodal deep search agent for image generation. We fully open-source our project and hope that Gen-Searcher can serve as an open foundation for future research.   
• To support training, we build a dedicated data pipeline to construct search-intensive image generation data, resulting in two training datasets, Gen-Searcher-SFT-10k and Gen-Searcher-RL-6k. In addition, we introduce KnowGen, a new and challenging benchmark designed to evaluate search-grounded image generation in knowledge-intensive real-world scenarios.   
• Extensive experiments validate the effectiveness of our proposed Gen-Searcher. For example, our method improves Qwen-Image by around 16 points on KnowGen and around 15 points on WISE.

# 2 Related Works

# 2.1 Image Generation Models

Recent years have witnessed rapid progress in image generation models, enabling the synthesis of high-fidelity and photorealistic images from natural language prompts [2, 17, 3, 18]. Early GAN-based methods demonstrated the potential of neural image synthesis, while diffusion models later became the dominant paradigm [19]. This progress has driven the development of a series of powerful models, including Stable Diffusion [20], Imagen [21], and more recent models like FLUX [22], Qwen-Image [1], LongCat-Image [23], Z-Image [3], and Nano Banana Pro [2] have further advanced image quality, multilingual text rendering, instruction following, and generation efficiency. Nevertheless, most of existing models still largely depend on frozen parametric knowledge acquired during pretraining, which limits their ability to handle prompts requiring rich world knowledge or up-to-date external information. Although a few advanced proprietary models such as Nano Banana Pro [2], incorporate search before generation, they remain limited to text-based search without visual search, often leading to inaccurate visual features in the generated images.

# 2.2 Agentic Reinforcement Learning

Agentic reinforcement learning (RL) has recently emerged as an effective paradigm for training large language model (LLM) agents to perform multi-step reasoning and tool interaction [24, 25, 26]. Building on RL with verifiable rewards [27, 28, 29, 30, 31], recent studies explore training agents that can interact with external tools and environments through long-horizon trajectories. For example, ARPO [24] introduces an agentic RL algorithm designed for multi-turn tool-use agents, incorporating an entropy-aware rollout strategy to encourage exploration. GiGPO [32] proposes a hierarchical group-based RL method that provides finer-grained step-level credit assignment for multi-turn agents. AdaTooler-V [25] proposes an adaptive tool-usage framework for image and video tool use, which dynamically adjusts reward scales to encourage invoking visual tools only when they provide measurable benefits. Vision-DeepResearch [33] enables multimodal agents to perform long-horizon visual and textual search over real-world search engines. However, training search agents for knowledge-intensive image generation with agentic RL remains unexplored.

# 3 Method

# 3.1 Dataset Construction

High-quality training data is essential for developing a search agent capable of performing multi-hop deep search and reasoning for image generation. However, such data does not naturally exist, since it requires aligned pairs of search-intensive prompts, agentic search trajectories, and grounded images. To address this challenge, we design a dedicated data pipeline that automatically constructs training data for search-grounded image generation. The overall pipeline consists of four stages: text prompt construction, agentic trajectory generation, grounded image synthesis, and data filtering and curation. An illustration of our data pipeline can be found in Figure 3

Text Prompt Construction. We first construct text prompts that require deep web search before image generation. To ensure diversity and realistic search difficulty, we adopt two complementary strategies. Our primary approach uses carefully designed prompt engineering to instruct Gemini 3 Pro [11] to generate multi-hop search-intensive prompts across a broad range of categories, including Anime, Architecture, Art, Astronomy, Biology, Celebrities, Chemistry, Culture, Engineering, Film, Game, Geography, History, Industry, Medicine, Physics, Politics, Posters, Religion, and Sports. These prompts are explicitly designed such that the required information cannot be obtained by single-turn search, and instead requires multi-step evidence aggregation and analysis across the web.

As a complementary strategy, we convert samples from existing deep research question answering datasets [34, 9] into image-generation-oriented prompts. In particular, we use Gemini 3 Pro to transform information-seeking questions into prompts that require generating a grounded visual depiction of the queried entity or event. This strategy primarily contributes prompts related to General News, further expanding the coverage of diverse knowledge scenarios.

Agentic Trajectory Generation. Given the constructed text prompts, we generate agentic search trajectories to perform deep search and gather sufficient evidence for producing the final search-grounded prompt together with selected reference images for accurate visual features. Meanwhile, these trajectories also serve as valuable supervision data for subsequent supervised fine-tuning. Specifically, we employ Gemini 3 Pro together with a set of search tools in a multi-turn manner. The tool set includes search for retrieving textual information from the web, image_search for searching relevant images via textual queries, and browse for reading and analyzing the detailed contents of retrieved webpages. During this process, the agent continuously analyzes textual and visual feedback from the environment, identifies useful evidence and reference images, and plans the next action accordingly. Through this multi-turn reasoning and search procedure, the agent progressively aggregates information from multiple sources before finally producing a grounded prompt and a set of relevant reference images for image synthesis.

Ground-Truth Image Synthesis. After obtaining the final grounded prompts and visual references, we synthesize the corresponding images using the proprietary image generation model Nano Banana Pro [2]. The generated images serve as synthesis ground truth for training the search agent. This process results in approximately 30K raw samples consisting of prompts, search trajectories, grounded prompts, reference images, and ground-truth images.

Data Filtering and Benchmark Construction. To ensure data quality, we further employ another strong proprietary model Seed1.8 [12] to score the generated samples from multiple perspectives, including whether the prompt genuinely

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/c9d86cfe29784e8eeffdf87691984ab0ce57c745301702725ec12a7b08d9b6d5.jpg]]  
Figure 3: An illustration of our data curation pipeline.

requires search, the correctness of the generated content, faithfulness to the prompt, visual aesthetics, text rendering clarity, and safety considerations. These model-based scores are combined with rule-based filtering, such as removing prompts with excessively long token lengths or inconsistent search results. After filtering, we obtain approximately 17K high-quality samples.

From this curated dataset, we select 630 human-verified samples to construct a held-out benchmark named KnowGen, which will be introduced later. The remaining 16K samples are used for training and are split into two datasets: Gen-Searcher-SFT-10k for supervised fine-tuning and Gen-Searcher-RL-6k for agentic reinforcement learning. We strictly ensure that no overlap exists between the training data and the evaluation benchmark.

# 3.2 KnowGen Benchmark

For evaluation, we introduce KnowGen, a comprehensive benchmark designed to evaluate search-grounded image generation under knowledge-intensive real-world scenarios. Unlike conventional text-to-image benchmarks that mainly emphasize prompt following or visual quality, KnowGen explicitly focuses on knowledge-intensive and searchdependent generation scenarios, where solving the prompt often requires retrieving and aggregating evidence from the web. Each sample in KnowGen is built to require non-trivial external knowledge, and many samples further demand multi-hop search over multiple sources. To ensure reliability, all evaluation samples are manually verified.

Category Composition. To provide broad coverage over different types of search-grounded generation tasks, we divide 630 samples in KnowGen into two high-level subsets: Science & Knowledge and Pop Culture & News. The Science & Knowledge subset includes these categories: astronomy, biology, chemistry, physics, engineering, medicine, industry, architecture, history, geography, religion, politics, culture, art, and sports. These tasks typically require factual world knowledge, entity disambiguation, or domain-specific information, and often involve fine-grained grounded details that must be visually or textually realized correctly. The Pop Culture & News subset covers prompts related to anime, games, films, celebrities, posters, and General News. Compared with the first subset, these tasks more frequently involve rapidly changing real-world information, popular culture entities, and prompt-required text or appearance details that must be rendered accurately. This two-part design allows KnowGen to evaluate both relatively stable knowledge-intensive scenarios and dynamic, high-update real-world scenarios within a unified benchmark. Figure 4 illustrates the categories and examples of our proposed KnowGen benchmark.

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/bc996e45f8d974072af78f05ab89c8207e9a86df914e7b1f13f64e1e6c5a8b4e.jpg]]  
(a) Data categories and distribution of our KnowGen.

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/8ae4b9be84a0e10335481709e36539a99dcbad1fe48e71af138d497d6e173463.jpg]]  
(b) Examples from our KnowGen.

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/ece69b6630e7d8af160577a652c97567f5544b8260217dfc495200796b247787.jpg]]  
Figure 4: Overview of the KnowGen benchmark.

Evaluation Metric. To evaluate generation quality on KnowGen, we introduce K-Score, a metric designed to assess search-grounded image generation from multiple perspectives. We adopt GPT-4.1 [35] as the judge to evaluate model outputs, following WISE benchmark [16]. For each sample, the evaluator takes as input the original text prompt, the ground-truth reference image, and the model-generated image, and scores the generated result from four dimensions: faithfulness, visual_correctness, text_accuracy, and aesthetics. Faithfulness measures whether the generated image follows the prompt at the scene-structure level, including the required subjects, relations, setting, and requested format. Visual correctness evaluates whether the key grounded visual attributes are correct with respect to the target concept and consistent with the reference image, such as subject appearance, object features, or other externally verifiable visual cues. Text accuracy measures whether any prompt-required readable text in the image is present, legible, and correct; when the prompt does not require readable text, this dimension is treated as not applicable and not counted into the average score. Aesthetics measures the overall visual quality and artistic appeal of the generated image, including composition, color harmony, lighting, etc. This dimension evaluates whether the image appears visually polished and aesthetically pleasing. The evaluation prompt can be found in Appendix A.

Following our evaluation design, each dimension is scored using a three-level discrete scale of $\{ 0 , 0 . 5 , 1 \}$ . Specifically, a score of 1 indicates the generated image fully satisfies the requirement of that dimension, 0.5 indicates that dimension is largely correct or satisfied but contains minor issues or partial mismatches, and 0 indicates the generation fails to meet the key requirement of that dimension. The final K-Score is computed as a weighted combination of these four dimensions:

K-Score = 0.1 · Faithfulness + 0.4 · Visual Correctness + 0.4 · Text Accuracy + 0.1 · Aesthetics.

This weighting emphasizes the two most critical aspects of search-grounded image generation, namely correctly rendering grounded visual attributes and accurately reproducing required textual content, while still accounting for overall prompt adherence and image aesthetic. We report K-Score both by the two high-level subsets and as an overall average on KnowGen.

# 3.3 Training Scheme

In this section, we train Gen-Searcher to act as a multimodal deep search agent that can iteratively gather external knowledge and visual evidence from the web for image generation. Our training scheme follows a two-stage pipeline, consisting of SFT and agentic RL.

# Prompt

A stunning, artistic travel editorial poster...: the mysterious Moai statues of Easter Island, the frozen taiga of Oymyakon, and the mirror-like salt flats of Salar de Uyuni....their average temperature ranges in April, and their probability of rain in April...Both the visual features and the text must be completely correct.

# 1. Think and Tool Call

<think>The user wants a travel editorial poster featuring three specific locations...Plan:\n1. Search for the climate data...2. Search for visual references for each location...3. Combine this information into a grounded prompt...</think> <tool_call>{"name": "search", "arguments": {"queries": ["Easter Island average temperature range April rain probability", "Oymyakon average temperature April rain probability", "Salar de Uyuni average temperature range April rain probability"] }}</tool_call>

# Tool Response

<tool_response>--- search result for [Easter Island average temperature range April rain probability]... expect warm maximum daytime temperatures of around $2 5 \mathrm { ^ \circ C } . . .$ . search result for [Oymyakon average temperature April rain probability]... there's a $4 \%$ chance of rain.... search result for [Salar de Uyuni average temperature range April rain probability] ... </tool_response>

# 2. Think and Tool Call

<think>I have gathered some climate data, but I need to be precise about the "average temperature range" and "probability of rain" for April. For Easter Island:\n- One source says: "average high of $2 5 . 4 ^ { \circ } \mathsf C$ (77.7°F) ... Another source says: nighttime temperatures around $2 2 ^ { \circ } \mathbb { C } .$ .. </think> <tool_call>\n{"name": "search", "arguments": {"queries": ["Easter Island April average temperature high low range", "Easter Island April rain probability percentage", ...... "Salar de Uyuni April average temperature high low range”......] }}</tool_call>

# Tool Response

<tool_response>\n--- search result for [Easter Island April average temperature high low range]... during April are $2 5 . 4 ^ { \circ } \mathsf C$ $( 7 7 . 7 ^ { \circ } \mathsf { F } )$ and ${ } 1 8 . 9 ^ { \circ } \mathsf { C }$ (66°F) respectively... search result for [Salar de Uyuni average temperature range April rain probability]... 6mm rainfall over 3 rainy days... search result for... Oymyakon High Temp $. 5 ^ { \circ } C 2 3 ^ { \circ } F$ Low Temp - $\cdot 1 9 ^ { \circ } \mathbb { C } .$ ... </tool_response>

# 3. Think and Tool Call

<think>I have gathered the following data: Easter Island... One source says "25.4°C / 18.9°C". Another says "25°C / 22°C". I will use a range like $\because 2 5 ^ { \circ } C /$ $1 9 ^ { \circ } \mathsf { C } ^ { \mathfrak { n } }$ ... to be safe... says "10 days of rain" (approx $3 3 \%$ )... Oymyakon:.... I'll use "-5°C / -19°C”.... Salar de Uyuni:... Now I need visual references... for the three locations to ensure the "stunning, artistic" aspect is grounded. 1. Easter Island Moai statues (blue sky, clear view). 2. Oymyakon frozen taiga (snowy, frozen river/landscape. 3. ...</think> <tool_call>\n{"name": "image_search", "arguments": {"query": "Easter Island Moai statues scenic view"}}</tool_call>"

# Tool Response

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/617a04f135cd6c3763e2366f1e0e4b34a43c57699240c13319f32b855290bc20.jpg]]  
IMG_001

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/8a85716653617fed1b60410c5514e3afa0ab535cbbe08fa9b3ed5d5d9f001c1f.jpg]]  
IMG_002

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/840a4c733270b8b925a71bf8de3ad776d12df12bc75873af284cf1a2473fa6e4.jpg]]  
IMG_003

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/e1d32ecc52babcc855f499bca87fee40272a1db3ca41173af9277b8e1d69303b.jpg]]  
IMG_004

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/22e244c40f347d3974ffa81b1c1d8a0aea775780d909f20acbf5e3e91e10d231.jpg]]  
IMG_005

# 4. Think and Tool Call

<think>I have good images for Easter Island (IMG_002 is a nice panoramic view, IMG_004 is a close up). I'll pick IMG_002 for the iconic view. Now I need images for ...<think> <tool_call> {"name": "image_search", "arguments": {"query": "Oymyakon frozen taiga landscape” }} </tool_call>

# Tool Response

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/686cf04e8914cb54734d77712b9abc056d47ae24fee1c5defa04778fad7bf21f.jpg]]  
IMG_006

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/3537e7d75d8423ba73f65a6ed21528e8e4fc90ab2fbb5ac4addf1418bce81a48.jpg]]  
IMG_007

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/b28a9d506fe350f651fb279d7727acdbaadff8921861709ab128b6196ab88b70.jpg]]  
IMG_008   
IMG_009

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/8bb947d2c617a40a636b0ebb499008183b0610c4632075eb8f5cf81804f61b58.jpg]]  
IMG_010

# 5. Think and Tool Call

<think>I have good images for Oymyakon (IMG_006, IMG_009). IMG_006 is a nice winter landscape. Now I need images for Salar de Uyuni.</think> <tool_call>\n{"name": "image_search", "arguments": {"query": "Salar de Uyuni mirror effect salt flats"}} </tool_call>

# Tool Response

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/391abf293ef0538e30895f813d64cfc8824faefda2eb5c618ccc1dc2b52ebac2.jpg]]  
IMG_011

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/6255fa44e4706bfb105750f0293de861917a90500bc47ad20e790886e150f2bc.jpg]]  
IMG_012

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/0a0091e8e44cd7e9f18c714adc9ff7491393c593df29d50da90ac31e259590f7.jpg]]  
IMG_013

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/34687102481b741112a0b830651251292b5c3820cc85cf5194f73fe8ff513bfb.jpg]]  
IMG_014

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/213c8773b8325130d36a7ed43d14e714dbffa865e9c2c9373c11586d14aec319.jpg]]  
IMG_015

# 6. Final Answer

<think>I have all the necessary information and visual references... </think> <answer>{ “gen_prompt”:“A stunning...On the left, first reference image to depict the sunny.... with ‘25°C / ${ \boldsymbol { \bf 1 9 } } ^ { \circ } { \bf C } ^ { \prime }$ and $30 \%$ Rain... "reference_images": [{"img_id":"IMG_002..."img_id":"IMG_006”..."img_id":"IMG_011”...</answer>

# Image

# Generation

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/764ae4f55f755ca8cbd288c399bdf0d305ec85fe16029fbe13aaf4ae2fa4e454.jpg]]  
Figure 5: An inference example of Gen-Searcher.

Search Tools. Gen-Searcher is equipped with three search tools. The first is search, which performs web text search and returns the top- $k$ relevant webpage URLs for each query with their short snippets. This tool is mainly used to verify factual information such as entity names, event details, dates, locations, and concise descriptions. The second is image_search, which retrieves the top- $k$ relevant images given a textual query, together with image URLs and brief descriptions, allowing the agent to ground identities, objects, landmarks, outfits, and other fine-grained appearance details. The third is browse, which takes a webpage URL as input and returns a summary of the page content; in our implementation, this summary is produced by Qwen3-VL-30B-A3B-Instruct. This tool is used when shallow search results are insufficient and the agent needs to extract specific evidence from a webpage. At each step, the agent observes the current prompt and accumulated search feedback, and then decides whether to continue searching, retrieve visual references, browse a page for more details, or terminate with a final grounded prompt and selected reference images.

Two-Stage Training. We initialize Gen-Searcher from Qwen3-VL-8B-Instruct. In the first stage, we perform supervised fine-tuning on Gen-Searcher-SFT-10k, which teaches the model to perform multi-turn tool use, including issuing search queries, interpreting textual and visual feedback, selecting useful reference images, and composing a final search-grounded prompt. In the second stage, we further optimize the model on Gen-Searcher-RL-6k with reinforcement learning, enabling it to learn more effective search strategies and produce improved tool-calling

trajectories. It is worth noting that during training, the image generator remains fixed; we only optimize Qwen3-VL-8B-Instruct to produce search-grounded prompts along with corresponding reference images. Figure 5 illustrates a representative inference trajectory of Gen-Searcher.

Dual Reward Feedback Design. A natural choice for RL in our setting is to directly use an image-based reward (e.g., K-Score) to evaluate the final generated image. However, relying on image reward alone leads to substantial noise and instability. This is because the final image quality depends not only on the correctness of the retrieved evidence, but also on the capability and stochasticity of the downstream image generator. In particular, for open-source generators such as Qwen-Image [1], even when the agent has collected correct information, complex prompts may still fail to produce high-quality images, and even similar grounded prompts can result in noticeably different generations. As a result, pure image-based reward introduces large variance and makes policy optimization unstable.

To address this issue, we introduce an additional text-based reward, denoted as $R _ { \mathrm { t e x t } }$ , which evaluates whether the final output text contains sufficient, correct, and generation-relevant information for synthesizing the target image. We also use GPT-4.1 as the judge to score this reward on a five-level scale, with values in $\{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ . Compared with image reward, text reward provides more direct supervision on the quality of information gathering, evidence aggregation. However, using only text reward is also insufficient, since text that appears to contain sufficient information does not necessarily support high-quality image generation. Optimizing only text reward would therefore ignore the actual end-task generation outcome and may encourage outputs that are textually informative but not practically effective for generation. The corresponding prompt can be found in Appendix B.

Therefore, we combine both signals and adopt a dual-feedback reward design, where the text-based reward supervises the quality of the gathered information and the image-based reward reflects the final generation performance. The final reward is computed as

$$
R = (1 - \alpha) R _ {\text {i m a g e}} + \alpha R _ {\text {t e x t}}, \tag {1}
$$

where $\alpha$ is a balancing hyperparameter. Here we simply set $\alpha = 0 . 5$ and use K-Score as $R _ { \mathrm { i m a g e } }$

Optimization. After computing the final reward, we optimize the policy using GRPO [13]. For each sampled output $o _ { i }$ under query $q$ , the advantage is computed by normalizing its reward with the mean and standard deviation of rewards within the sampled group:

$$
A _ {i} = \frac {R _ {i} - \operatorname {m e a n} \left(\left\{R _ {j} \right\}\right)}{\operatorname {s t d} \left(\left\{R _ {j} \right\}\right)}. \tag {2}
$$

The final policy update follows the standard GRPO objective:

$$
\begin{array}{l} \mathcal {J} _ {\mathrm {G R P O}} = \mathbb {E} _ {q, \left\{o _ {i} \right\}} \left[ \right. \frac {1}{G} \sum_ {i = 1} ^ {G} \left(\min  \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {\mathrm {o l d}}} (o _ {i} | q)} A _ {i}, \operatorname {c l i p} \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {\mathrm {o l d}}} (o _ {i} | q)}, 1 - \epsilon , 1 + \epsilon\right) A _ {i}\right)\right. \\ \left. \left. - \beta_ {\mathrm {K L}} D _ {\mathrm {K L}} \left(\pi_ {\theta} \| \pi_ {\mathrm {r e f}}\right)\right) \right], \\ \end{array}
$$

where the variables and hyper-parameters are defined following the original GRPO algorithm [13].

# 4 Experiments

# 4.1 Setup

Training Details. We train Gen-Searcher-8B using 8 NVIDIA H800 GPUs, with Qwen3-VL-8B-Instruct [36] as the base model. We first perform supervised fine-tuning on Gen-Searcher-SFT-10k, and then further conduct agentic RL training on Gen-Searcher-RL-6k. For both SFT and RL, we use AdamW as the optimizer. The learning rate is set to $1 \times 1 0 ^ { - 5 }$ for SFT and $1 \times 1 0 ^ { - 6 }$ for RL, and the batch size is set to 8 in both stages. During RL training, we additionally deploy Qwen-Image-Edit-2509 on 16 H800 GPUs to support rollout image generation, since we find that the 2509 version provides better text rendering quality than the 2511 version. We also deploy Qwen3-VL-30B-Instruct-A3B [36] on 8 H800 GPUs as the summary model for the browse tool. For efficiency, we set the group size to 6, limit the maximum number of interaction turns to 10, allow at most 5 returned images per turn, and set the maximum context length to 36K. The model response length per turn is limited to 4K. Following prior practice, we mask out overlong rollouts and rollouts with repetitive responses during training. The training process consumes around one day.

Table 1: Performance of different models on our KnowGen benchmark. Visual cor. and Text acc. denote Visual correctness and Text accuracy, respectively. The overall K-Score is averaged over the Science & Knowledge and Pop Culture & News subsets.   

<table><tr><td rowspan="2">Models</td><td colspan="4">Science &amp; Knowledge</td><td colspan="4">Pop Culture &amp; News</td><td>Overall</td></tr><tr><td>Visual cor.</td><td>Text acc.</td><td>Faithfulness</td><td>Aesthetics</td><td>Visual cor.</td><td>Text acc.</td><td>Faithfulness</td><td>Aesthetics</td><td>K-Score</td></tr><tr><td>GPT-Image-1 [37]</td><td>20.92</td><td>27.89</td><td>72.79</td><td>63.95</td><td>19.43</td><td>31.98</td><td>84.64</td><td>61.60</td><td>34.19</td></tr><tr><td>GPT-Image-1.5 [38]</td><td>29.25</td><td>40.14</td><td>81.29</td><td>77.21</td><td>29.43</td><td>46.22</td><td>89.64</td><td>71.17</td><td>44.97</td></tr><tr><td>Nano Banana [39]</td><td>18.03</td><td>19.39</td><td>72.79</td><td>65.82</td><td>14.24</td><td>26.04</td><td>84.39</td><td>70.91</td><td>30.24</td></tr><tr><td>Nano Banana Pro [2]</td><td>39.46</td><td>49.32</td><td>86.22</td><td>70.92</td><td>30.51</td><td>53.37</td><td>91.07</td><td>68.75</td><td>50.38</td></tr><tr><td>Seedream 4.0 [40]</td><td>13.10</td><td>24.66</td><td>62.93</td><td>64.97</td><td>11.90</td><td>22.19</td><td>78.72</td><td>70.24</td><td>28.21</td></tr><tr><td>Seedream 4.5 [15]</td><td>14.46</td><td>26.19</td><td>64.46</td><td>65.65</td><td>12.50</td><td>31.77</td><td>81.25</td><td>69.05</td><td>31.01</td></tr><tr><td>SD-3.5-Medium [41]</td><td>5.61</td><td>2.21</td><td>30.44</td><td>48.47</td><td>3.12</td><td>0.58</td><td>58.18</td><td>54.76</td><td>11.90</td></tr><tr><td>SD-3.5-Large [20]</td><td>5.44</td><td>2.04</td><td>31.29</td><td>46.77</td><td>5.21</td><td>2.01</td><td>55.36</td><td>58.33</td><td>12.53</td></tr><tr><td>Lumina-Image 2.0 [42]</td><td>1.19</td><td>0.34</td><td>30.95</td><td>36.05</td><td>2.68</td><td>0.58</td><td>54.76</td><td>47.62</td><td>9.43</td></tr><tr><td>FLUX.1-dev [22]</td><td>2.89</td><td>0.34</td><td>28.91</td><td>50.17</td><td>2.38</td><td>1.16</td><td>54.46</td><td>53.72</td><td>10.71</td></tr><tr><td>FLUX.1-Krea [22]</td><td>3.91</td><td>1.53</td><td>33.16</td><td>48.13</td><td>4.32</td><td>2.02</td><td>62.05</td><td>53.87</td><td>12.22</td></tr><tr><td>FLUX.2-klein-4B [43]</td><td>4.59</td><td>1.53</td><td>37.07</td><td>45.58</td><td>3.42</td><td>0.86</td><td>62.05</td><td>55.51</td><td>12.09</td></tr><tr><td>FLUX.2-klein-9B [43]</td><td>6.12</td><td>0.34</td><td>42.69</td><td>50.85</td><td>5.06</td><td>1.72</td><td>69.05</td><td>59.08</td><td>13.73</td></tr><tr><td>BAGEL [44]</td><td>4.93</td><td>1.70</td><td>43.37</td><td>51.87</td><td>8.33</td><td>2.59</td><td>64.14</td><td>53.57</td><td>13.85</td></tr><tr><td>HunyuanImage-3.0 [45]</td><td>4.76</td><td>1.19</td><td>40.14</td><td>56.46</td><td>6.10</td><td>2.51</td><td>63.99</td><td>64.14</td><td>14.15</td></tr><tr><td>Qwen-Image [1]</td><td>6.80</td><td>0.34</td><td>47.45</td><td>56.80</td><td>7.59</td><td>1.40</td><td>68.90</td><td>61.90</td><td>14.98</td></tr><tr><td>Z-Image-Turbo [3]</td><td>3.91</td><td>1.02</td><td>28.40</td><td>50.85</td><td>4.32</td><td>3.45</td><td>50.15</td><td>55.21</td><td>11.77</td></tr><tr><td>Z-Image [3]</td><td>6.80</td><td>2.72</td><td>41.16</td><td>43.54</td><td>7.89</td><td>2.00</td><td>70.24</td><td>57.29</td><td>14.49</td></tr><tr><td>Gen-Searcher-8B + Qwen-Image</td><td>26.87</td><td>17.18</td><td>65.14</td><td>55.44</td><td>25.30</td><td>23.55</td><td>76.64</td><td>61.46</td><td>31.52</td></tr><tr><td>Gen-Searcher-8B + Seedream 4.5</td><td>36.35</td><td>43.52</td><td>75.77</td><td>61.26</td><td>39.04</td><td>45.86</td><td>85.74</td><td>63.96</td><td>47.29</td></tr><tr><td>Gen-Searcher-8B + Nano Banana Pro</td><td>45.07</td><td>49.32</td><td>86.56</td><td>64.80</td><td>43.01</td><td>52.30</td><td>90.92</td><td>64.88</td><td>53.30</td></tr></table>

Benchmarks and Evaluation. We evaluate Gen-Searcher on two benchmarks. The first is KnowGen, our proposed benchmark for real-world search-grounded image generation, which focuses on real-world, knowledge-intensive prompts that often require external search and multi-step evidence aggregation. The second is WISE [16], a relatively simpler benchmark for knowledge-based image generation. During inference, we set the decoding parameters to temperature $= 0 . 6$ and top- $p = 0 . 9$ , and use a maximum context length of 64K. At test time, we first feed the original text prompt into Gen-Searcher, which produces a search-grounded prompt together with selected reference images, and then pass them to the downstream image generator for final image synthesis. If Gen-Searcher fails to produce a final search-grounded prompt due to issues such as overlong context or tool-call failure, we fall back to using the original prompt for generation. For model families that separate text-only and editing models, such as Qwen-Image and Qwen-Image-Edit, we use the text-only model for generation from pure text input and the editing model for generation conditioned on both text and reference images.

# 4.2 Main Results on KnowGen

Challenging Nature of the KnowGen Benchmark. Table 1 presents the main results on KnowGen. Overall, we can find that KnowGen is a highly challenging benchmark for current image generation models, especially for open-source ones. Even strong open-source baselines such as Qwen-Image, HunyuanImage-3.0, FLUX, and Z-Image achieve only

Table 2: Performance of different models on the WISE benchmark.   

<table><tr><td>Models</td><td>Cultural</td><td>Time</td><td>Space</td><td>Biology</td><td>Physics</td><td>Chemistry</td><td>Overall</td></tr><tr><td>FLUX.1-dev [22]</td><td>0.48</td><td>0.58</td><td>0.62</td><td>0.42</td><td>0.51</td><td>0.35</td><td>0.50</td></tr><tr><td>FLUX.1-schnell [22]</td><td>0.39</td><td>0.44</td><td>0.50</td><td>0.31</td><td>0.44</td><td>0.26</td><td>0.40</td></tr><tr><td>SD-3-Medium [46]</td><td>0.42</td><td>0.44</td><td>0.48</td><td>0.39</td><td>0.47</td><td>0.29</td><td>0.42</td></tr><tr><td>SD-3.5-Medium [41]</td><td>0.43</td><td>0.50</td><td>0.52</td><td>0.41</td><td>0.53</td><td>0.33</td><td>0.45</td></tr><tr><td>SD-3.5-Large [20]</td><td>0.44</td><td>0.50</td><td>0.58</td><td>0.44</td><td>0.52</td><td>0.31</td><td>0.46</td></tr><tr><td>Emu3 [47]</td><td>0.34</td><td>0.45</td><td>0.48</td><td>0.41</td><td>0.45</td><td>0.27</td><td>0.39</td></tr><tr><td>Qwen-Image [1]</td><td>0.62</td><td>0.63</td><td>0.77</td><td>0.57</td><td>0.75</td><td>0.40</td><td>0.62</td></tr><tr><td>HunyuanImage-3.0 [45]</td><td>0.58</td><td>0.57</td><td>0.70</td><td>0.56</td><td>0.63</td><td>0.31</td><td>0.57</td></tr><tr><td>LongCat-Image [23]</td><td>0.66</td><td>0.61</td><td>0.72</td><td>0.66</td><td>0.72</td><td>0.49</td><td>0.65</td></tr><tr><td>Gen-Searcher-8B + Qwen-Image</td><td>0.80</td><td>0.71</td><td>0.82</td><td>0.76</td><td>0.74</td><td>0.75</td><td>0.77</td></tr></table>

around 9 to 15 K-Score, showing that knowledge-intensive and search-grounded image generation remains far beyond the capability of standard text-to-image systems. In contrast, proprietary models perform substantially better, with Nano Banana Pro achieving the strongest baseline result of 50.38 and GPT-Image-1.5 reaching 44.97. This large gap indicates that KnowGen poses significant challenges in both grounded knowledge retrieval and faithful visual realization, and also highlights the clear difference between open-source and proprietary systems in handling such tasks.

Effectiveness of Gen-Searcher. Our method consistently brings significant performance gains across different image generation backbones on KnowGen. When combined with Qwen-Image, Gen-Searcher-8B improves the overall K-Score from 14.98 to 31.52, yielding a gain of 16.54 points. This large improvement shows that Gen-Searcher can substantially compensate for the lack of built-in search capability in open-source image generators by actively gathering grounded textual evidence and visual references from the web. More importantly, Gen-Searcher is not merely learning a generator-specific prompting heuristic; instead, it learns a transferable search-and-grounding policy that generalizes across different downstream image generators. Notably, although Gen-Searcher is trained with Qwen-Image as the rollout generator during RL, it transfers well to other generators at test time. In particular, it improves Seedream 4.5 from 31.01 to 47.29, a gain of 16.28 points, and further boosts Nano Banana Pro from 50.38 to 53.30, achieving the best overall result in the table. These results demonstrate not only the effectiveness of our search agent, but also its strong transferability and robustness across image generators with very different native capabilities. Collectively, they suggest that our proposed Gen-Searcher is a general and powerful model for enhancing image generation in real-world knowledge-intensive scenarios.

Analysis of Different Dimensions. Analysis of the four evaluation dimensions shows that the gains from Gen-Searcher mainly come from improvements in visual correctness and text accuracy, which are also the two most important components in KnowGen. This indicates that our search framework enables the image generator to better produce accurate visual attributes and textual content that require real-world knowledge. In some cases, we observe slight decreases in aesthetics, which may stem from the fact that the generator needs to integrate information from multiple retrieved reference images and therefore cannot always produce the most ideal or visually pleasing composition. We also find an interesting pattern on Nano Banana Pro: its improvement mainly comes from visual correctness, while text accuracy remains almost unchanged. A possible explanation is that Nano Banana Pro already supports text-based search internally, which helps preserve text-related performance, but it does not retrieve visual reference images, leaving substantial room for improvement in grounding fine-grained visual attributes.

# 4.3 Performance on WISE

Table 2 reports the performance of different models on the WISE benchmark. Compared with KnowGen, WISE is a relatively simpler benchmark, but it still requires a certain amount of world knowledge for correct image generation.

Table 3: Ablation Study.   

<table><tr><td>Methods</td><td>KnowGen</td></tr><tr><td>Qwen-Image</td><td>14.98</td></tr><tr><td>Qwen-Image + workflow</td><td>22.91</td></tr><tr><td>Qwen-Image + Gen-Searcher-SFT</td><td>28.15</td></tr><tr><td>Qwen-Image + Gen-Searcher w.o. text reward</td><td>29.59</td></tr><tr><td>Qwen-Image + Gen-Searcher w.o. image reward</td><td>29.36</td></tr><tr><td>Qwen-Image + Gen-Searcher</td><td>31.52</td></tr></table>

Our Gen-Searcher- $\mathrm { 8 B + }$ Qwen-Image achieves the best overall performance of 0.77, significantly improving over the original Qwen-Image baseline at 0.62 by 0.15. It also surpasses all other compared open-source models, including LongCat-Image, HunyuanImage-3.0, and FLUX.1-dev. Looking at individual categories, our method brings clear gains on Cultural, Time, Space, Biology, and especially Chemistry, where the score improves from 0.40 to 0.75. These results further demonstrate that Gen-Searcher generalizes beyond KnowGen and can effectively enhance image generation on knowledge-based image generation benchmarks.

# 4.4 Ablation Study

In this section, we study several variants to verify the effectiveness of different components in Gen-Searcher and to better understand the role played by each design choice in the overall framework. Specifically, we compare: (1) the vanilla Qwen-Image baseline without any search augmentation, which directly generates images from the original prompt; (2) Qwen-Image $^ +$ workflow, which uses Qwen3-VL-8B-Instruct as the search agent in a manually designed prompt-based search workflow without any additional training; (3) Qwen-Image $^ +$ Gen-Searcher-SFT, which applies only supervised fine-tuning to train Gen-Searcher without reinforcement learning; (4) Qwen-Image $^ +$ Gen-Searcher w.o. text reward, which removes the text-based reward and uses only image-based reward during RL training; (5) Qwen-Image $^ +$ Gen-Searcher w.o. image reward, which removes the image-based reward and uses only text-based reward during RL training; and (6) the full Gen-Searcher model, which includes both SFT initialization and the proposed dual reward feedback design during agentic RL training.

As shown in Table 3, all components contribute positively to the final performance. Compared with the plain Qwen-Image baseline, the prompt-based workflow improves the KnowGen score from 14.98 to 22.91, showing that introducing external search alone provides benefits for knowledge-intensive image generation. Replacing the prompt-based workflow with Gen-Searcher-SFT further improves the score to 28.15, demonstrating the advantage of learning tool-use behavior directly from trajectory data instead of relying on manually designed prompting rules. This suggests that supervised learning on curated search trajectories enables the model to better organize search actions, integrate retrieved evidence, and produce more effective grounded prompts for generation. Agentic reinforcement learning brings additional gains beyond SFT, and the full Gen-Searcher reaches the best performance of 31.52. This shows that while SFT provides a strong initialization for basic tool use, RL is still crucial for further optimizing long-horizon search behavior and improving the overall quality of the collected evidence and final outputs. Moreover, removing either the text reward or the image reward leads to clear degradation, with scores dropping to 29.59 and 29.36, respectively. This confirms that the two reward signals play complementary roles. The text reward provides more direct supervision on whether the agent has gathered sufficient and correct information at the textual level, while the image reward aligns the policy with the final generation outcome and encourages the collected evidence to be practically useful for image synthesis. Overall, the ablation results validate the effectiveness of our overall framework, including learned search behavior, agentic RL optimization, and the proposed dual-reward design.

# 4.5 Qualitative Visualization Analysis

Figure 6 shows representative qualitative examples on the KnowGen benchmark. Overall, we find that Gen-Searcher consistently enhances both the quality and correctness of generated images across different downstream generators

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/5b47093137ba88d2a45d506b0ce771d960820e668a4ea54e1fe2386752c4339a.jpg]]  
Figure 6: Examples of generated images by different methods on our KnowGen benchmark.

in knowledge-intensive, real-world scenarios. First, we observe that Nano Banana Pro still falls short in generating accurate fine-grained visual attributes in real-world, knowledge-intensive scenarios, because it cannot perform image search for precise visual references. As a result, the generated identity, object appearance, or architectural details may deviate from the target even when some textual information is correct. In contrast, Gen-Searcher improves Nano Banana Pro by searching relevant reference images and grounding the generation with more accurate visual evidence. An interesting finding is that for Qwen-Image, even when the search agent has already collected correct information, the final generation sometimes can still be inaccurate due to limitations of the image generator itself (e.g., multi-subject consistency issue, poor text rendering issue). The fourth row in Figure 6 provides one such example, where the searched content is correct but the generated image still fails to faithfully realize the required multi-character details. In summary, these examples show that Gen-Searcher can substantially improve generation by providing grounded textual and visual evidence for both strong proprietary model Nano Banana Pro and open-source model Qwen-Image, while some failure cases indicate that the capability of the downstream image generator also remains a challenge.

# 4.6 Parameter Analysis

We further analyze the balancing coefficient $\alpha$ between the text reward and the image reward in our dual-feedback design. Figure 3 demonstrates the performance of our Gen-Searcher using different $\alpha$ for RL training. We observe that setting $\alpha = 0$ or $\alpha = 1 . 0$ leads to clear performance degradation, indicating that both reward signals are necessary for effective training. This is consistent with our motivation: relying only on image reward introduces high variance due to the stochasticity and limited capability of the downstream generator, while relying only on text reward ignores whether the gathered information can actually support high-quality image synthesis. In contrast, we find that performance remains consistently strong when $\alpha$ is set in the range of 0.3 to 0.6, showing that our method is relatively insensitive to this hyperparameter over a relatively broad range.

![[Gen-Searcher: Reinforcing Agentic Search for Image Generation/images/a8e020a03a95ec505449792c690cd1a0283e917bd1e8e4fe78daef8bdd4db13a.jpg]]  
Figure 7: Parameter Analysis on $\alpha$

# 5 Conclusion

In this paper, we present Gen-Searcher, the first attempt to train a multimodal deep search agent for knowledge-intensive image generation with agentic RL. To enable this setting, we build a dedicated data pipeline, construct two training datasets Gen-Searcher-SFT-10k, Gen-Searcher-RL-6k, and introduce the KnowGen benchmark together with K-Score for evaluating real-world knowledge-intensive image generation. Based on these resources, we train Gen-Searcher through a two-stage scheme of supervised fine-tuning and agentic reinforcement learning with dual reward feedback. Extensive experiments show that Gen-Searcher brings substantial gains across different image generation backbones on both KnowGen and WISE, while also exhibiting strong transferability across image generators. We hope this work can serve as an open foundation for future research on search agents for real-world image generation.

# References

[1] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.   
[2] Google DeepMind. Gemini image pro: High-quality image generation. https://deepmind.google/models/ gemini-image/pro/, 2025.   
[3] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.   
[4] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. arXiv preprint arXiv:2209.14491, 2022.   
[5] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas Müller, and Björn Ommer. Retrieval-augmented diffusion models. Advances in Neural Information Processing Systems, 35:15309–15324, 2022.   
[6] Zhiyou Xiao, Qinhan Yu, Binghui Li, Geng Chen, Chong Chen, and Wentao Zhang. M2io-r1: An efficient rl-enhanced reasoning framework for multimodal retrieval augmented multimodal generation. arXiv preprint arXiv:2508.06328, 2025.   
[7] Chuanhao Li, Jianwen Sun, Yukang Feng, Mingliang Zhai, Yifan Chang, and Kaipeng Zhang. Ia-t2i: Internet-augmented text-to-image generation. arXiv preprint arXiv:2505.15779, 2025.   
[8] Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, and Weijia Li. Mind-brush: Integrating agentic cognitive search and reasoning into image generation. arXiv preprint arXiv:2602.01756, 2026.   
[9] Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, et al. Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025.   
[10] Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, and Xiangyu Yue. Exploring reasoning reward model for agents. arXiv preprint arXiv:2601.22154, 2026.   
[11] Google DeepMind. Gemini 3 pro. https://deepmind.google/models/gemini/pro/, 2025.   
[12] Bytedance Seed. Seed1.8 model card: Towards generalized real-world agency. https://seed.bytedance.com/en/ seed1_8, 2025.

[13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[14] Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025.   
[15] Bytedance Seed. Seedream 4.5. https://seed.bytedance.com/en/seedream4_5, 2025.   
[16] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.   
[17] Hongyu Li, Manyuan Zhang, Dian Zheng, Ziyu Guo, Yimeng Jia, Kaituo Feng, Hao Yu, Yexin Liu, Yan Feng, Peng Pei, et al. Editthinker: Unlocking iterative reasoning for any image editor. arXiv preprint arXiv:2512.05965, 2025.   
[18] Dian Zheng, Manyuan Zhang, Hongyu Li, Kai Zou, Hongbo Liu, Ziyu Guo, Kaituo Feng, Yexin Liu, Ying Luo, Yan Feng, et al. Architecture decoupling is not all you need for unified multimodal model. arXiv preprint arXiv:2511.22663, 2025.   
[19] Hang Chen, Qian Xiang, Jiaxin Hu, Meilin Ye, Chao Yu, Hao Cheng, and Lei Zhang. Comprehensive exploration of diffusion models in image generation: a survey. Artificial Intelligence Review, 58(4):99, 2025.   
[20] Stability AI. Stable diffusion 3.5 large. https://huggingface.co/stabilityai/stable-diffusion-3. 5-large, 2024.   
[21] Google DeepMind. Imagen. https://deepmind.google/models/imagen/, 2025.   
[22] black-forest labs. Flux 1. https://github.com/black-forest-labs/flux, 2024.   
[23] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.   
[24] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, et al. Agentic reinforced policy optimization. arXiv preprint arXiv:2507.19849, 2025.   
[25] Chaoyang Wang, Kaituo Feng, Dongyang Chen, Zhongyu Wang, Zhixun Li, Sicheng Gao, Meng Meng, Xu Zhou, Manyuan Zhang, Yuzhang Shang, et al. Adatooler-v: Adaptive tool-use for images and videos. arXiv preprint arXiv:2512.16918, 2025.   
[26] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965, 2025.   
[27] Xiaoying Zhang, Yipeng Zhang, Hao Sun, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106, 2025.   
[28] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.   
[29] Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025.   
[30] Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. Sophiavl-r1: Reinforcing mllms reasoning with thinking reward. arXiv preprint arXiv:2505.17018, 2025.   
[31] Shuang Chen, Yue Guo, Yimeng Ye, Shijue Huang, Wenbo Hu, Haoxi Li, Manyuan Zhang, Jiayu Chen, Song Guo, and Nanyun Peng. Ares: Multimodal adaptive reasoning via difficulty-aware token-level entropy shaping. arXiv preprint arXiv:2510.08457, 2025.   
[32] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.   
[33] Wenxuan Huang, Yu Zeng, Qiuchen Wang, Zhen Fang, Shaosheng Cao, Zheng Chu, Qingyu Yin, Shuang Chen, Zhenfei Yin, Lin Chen, et al. Vision-deepresearch: Incentivizing deepresearch capability in multimodal large language models. arXiv preprint arXiv:2601.22060, 2026.   
[34] Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, et al. Simpledeepsearcher: Deep information seeking via web-powered reasoning trajectory synthesis. arXiv preprint arXiv:2505.16834, 2025.   
[35] OpenAI. Introducing gpt-4.1 in the api. https://openai.com/index/gpt-4-1/, 2025.   
[36] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.   
[37] OpenAI. Gpt-image-1: Models and capabilities for image generation. https://platform.openai.com/docs/ models/gpt-image-1, 2024.   
[38] OpenAI. Gpt-image-1.5: Enhanced visual reasoning and creative generation. https://platform.openai.com/docs/ models/gpt-image-1.5, 2025.

[39] Google DeepMind. Gemini image: High-quality image generation. https://deepmind.google/models/ gemini-image/flash/, 2025.   
[40] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.   
[41] Stability AI. Stable diffusion 3.5 medium. https://huggingface.co/stabilityai/stable-diffusion-3. 5-medium, 2024.   
[42] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Xinyue Li, Dongyang Liu, Xiangyang Zhu, et al. Luminaimage 2.0: A unified and efficient image generative framework. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20031–20042, 2025.   
[43] black-forest labs. Flux 2. https://github.com/black-forest-labs/flux2, 2025.   
[44] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.   
[45] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.   
[46] Stability AI. Stable diffusion 3 medium. https://huggingface.co/stabilityai/ stable-diffusion-3-medium, 2024.   
[47] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

# A KnowGen Benchmark Evaluation Prompt

# K-Score Evaluation Prompt

You are a strict and professional expert evaluator for AI-generated image grounded with world knowledge (MODEL EVALUATION).

You will receive:

1) A task prompt (what the image must show).   
2) Image 1: the generated image (model output to be evaluated).   
3) Image 2: the ground-truth reference image (a strong reference implementation).

All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

Critical clarification (VERY IMPORTANT):

- This is NOT a pixel-level similarity task.

- Image 2 (GT) is a REFERENCE for intended identity, key grounded details, and stable visual attributes.   
Image 1 may use a different camera angle/layout as long as it still satisfies the prompt.   
- Focus on whether prompt-required, externally-checkable (search-grounded) details are correctly AND verifiably realized in Image 1.   
- Do NOT assume correctness if a key detail is not clearly visible/readable. If unverifiable, score lower.

Output format (MUST follow exactly):

Output ONLY one valid JSON object with EXACTLY these keys:

"rationale": string,

"faithfulness": number,

"visual_correctness": number,

"text_accuracy": number,

"aesthetics": number

"text_accuracy_na": boolean,

SCORING SCALE (VERY IMPORTANT):

- 1 (Exemplary) is rare and requires perfect success for that dimension.   
- 0.5 (Conditional) means mostly correct but not perfect.   
- 0 (Rejected) means failed on important requirements.

- "rationale" must be 5-10 short sentences, evidence-based, referring only to what is visible.   
- "text_accuracy_na" should be true if the prompt does not require any readable text, otherwise it should be false.

Implicit required step (ENFORCED via rationale):

- In the rationale, you MUST explicitly list the extracted prompt hard constraints (2-5, or more if needed) BEFORE scoring.

If you cannot identify the constraints, you must still list what you believe are the hard constraints.

Evaluation procedure (follow silently, but the rationale MUST reflect it):

1) Extract the prompt’s TOP hard constraints (2-5, or more if needed): required subjects/identities, setting/props, relations/counts, required style, and any externally-checkable requirements (readable text/landmark/logo/badge/version/year/etc.).   
2) Score Image 1 against the constraints. Use Image 2 only as a reference for stable identity/visual attributes and grounded evidence.   
3) If a key requirement is not verifiable (too small/blurred/occluded/warped), do NOT assume it is correct; score lower. 4) Assessment of the primary subjects' visual identity correctness and consistency is mandatory in every case.

Boundary between visual_correctness vs text_accuracy:   
- Visual-only grounded cues (subject visual features, logo SHAPE, badge EMBLEM geometry, landmark facade/massing, outfit/weapon silhouette, object geometry)   
belong to visual_correctness.   
- Any grounded cue that must be READ as text (spelling, year numbers, titles, institution names, badge text) belongs to text_accuracy.

STRICT 3-LEVEL RUBRICS

(Each dimension uses ONLY {0, 0.5, 1})

1) faithfulness (overall prompt adherence: presence & structure only; not GT-identity correctness):   
- This score does NOT require matching GT’s exact identity or fine-grained visual features; it focuses on whether Image 1 includes the prompt-requested elements and scene structure (who/what is present, what is happening, where it happens, and the required style/format).

(Exemplary) Score = 1 ONLY IF:

- Image 1 clearly includes everything the prompt asks for in terms of visible content and structure:   
all required subjects/entities are present, the required setting and key props appear,   
required actions/relations/counts are shown, and the required style/format is followed.   
Any required in-scene evidence elements requested by the prompt (e.g., a plaque/sign, a map, a report paper, a badge) are present as elements.

(Conditional) Score $= ~ 0 , 5$ ONLY IF:

Image 1 includes almost all prompt-requested content and structure, with only minor omissions or minor staging differences   
that do not change what the scene is supposed to depict (e.g., small placement differences, slight simplification of a secondary prop).

(Rejected) Score = 0 IF:

One or more prompt-requested essential elements are not shown at all, or the scene structure clearly does not match the prompt’s request

(e.g., missing a required subject/entity, missing the required setting, missing the required key prop/evidence element,

missing the requested action/relationship/count, or not following the requested style/format).

2) visual_correctness (GT visual-feature agreement is the core; extremely strict): (Exemplary) Score = 1 ONLY IF:   
The prompt-required primary subjects/objects in Image 1 match the GT reference (Image 2) in visual characteristics with NO substantive changes.   
This means: the same face/hairstyle silhouette, the same armor/clothing design and key colors/patterns, the same distinctive props/object geometry, the same emblem/logo/landmark facade/massing cues when applicable, etc.   
- Any meaningful difference in these stable visual features disqualifies a score of 1.

(Conditional) Score $= ~ 0 , 5$ ONLY IF:

- Image 1 can still be considered the same overall visual instance as the GT, and the differences are limited to relatively minor variations, allowing some changes to the visual features (face, hairstyle, armor design, key colors/patterns, key prop shapes), while the overall identity and major visual features remain recognizable and broadly consistent.   
- IMPORTANT: "same role archetype" (generic knight/princess/warrior) alone does NOT qualify for 0.5.

(Rejected) Score = 0 IF:   
- Any substantive mismatch vs GT in stable visual features of the required subjects/objects (different face/hair/armor design/color scheme/emblem/prop geometry/landmark cues), even if the overall scene still looks plausible or stylistically similar.

3) text_accuracy (required readable text; ALL relevant text must be correct AND very clearly readable; NO partial credit for wrong text):

Rule:

If the prompt does NOT require any readable text: you MUST output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON. In your rationale state that the prompt did not require readable text.

- If the prompt DOES require readable text: output "text_accuracy_na": false and score "text_accuracy" (0, 0.5, or 1) per the criteria below.

(Exemplary) Score = 1 ONLY IF:

- ALL required text AND any prompt-involved text elements are:

(a) present,   
(b) very clearly readable (crisp, unambiguous),   
(c) correct and consistent with the prompt’s requirements.

(Conditional) Score = 0.5 ONLY IF:

- Much of the required/prompt-involved text is readable and generally correct, and although parts may contain inaccuracies, omissions, or deviations, the overall meaning remains clear and is not seriously inconsistent with the prompt requirements.

(Rejected) Score = 0 IF:

- Any required/prompt-involved text is missing, unclear, not very readable, gibberish, placeholder, OR incorrect.   
- Even if perfectly readable, if content is not correct, text_accuracy MUST be 0.

4) aesthetics:

(Exemplary) Score = 1 ONLY IF:   
- Masterpiece-level composition and polish, AND Image 1 is NOT worse than GT in overall aesthetic quality.   
(Conditional) Score = 0.5 ONLY IF:   
- Very beautiful and polished, but slightly worse than GT (ONLY slightly) OR slightly less refined than top-tier. (Rejected) Score $\qquad = \ 0$ IF:   
Anything clearly worse than GT in a noticeable way, OR merely average/OK-looking, OR cluttered/awkward framing, OR visible artifacts/noise that harm the overall appeal.

Rationale requirements (MANDATORY):

- Start with: "Constraints:" and list the extracted constraints (2-5, or more if needed).   
- State whether the prompt required readable text; if not required, output "text_accuracy_na": true and "text_accuracy":   
0.5 in the JSON and say so in the rationale.   
- Mention 2-5 key comparisons (or more if needed) to GT focused on stable identity/visual traits (NOT demanding identical layout).   
- Keep within 10 sentences.

Output JSON only. No markdown. No extra text.

# B Text Reward Prompt

# Text Reward Prompt

You are an expert evaluator for a text-based image generation pipeline.

You will receive:

1) Task prompt: the original user requirement (what image we want to generate).   
2) Ground-truth reference image: the target image we want the pipeline to produce.   
3) Model's answer: the model's output in <answer>, containing:

- gen_prompt: a natural-language prompt for an image generator (composition, style, subjects, etc.).

reference_images: a list of chosen reference images (each with img_id, title, note, etc.) that the model selected from search to guide generation.

Your task (TEXT $^ +$ VISUAL):

From both TEXT and VISUAL perspectives, judge how well this answer (gen_prompt + reference image choices) would support generating an image that matches the ground-truth.

- You are NOT evaluating an actual generated image here. You are evaluating whether the model's textual output (search choices + generation prompt) is well-aligned with the task and the GT: i.e., if we had a perfect image generator, would this answer be sufficient to produce the GT?

- Consider: Does the gen_prompt capture the key requirements from the task and the GT? Are the chosen reference images (by their titles/notes and what they typically show) appropriate for producing the GT? Are there critical missing or wrong elements?

Output format (MUST follow exactly):

Output ONLY one valid JSON object with EXACTLY these keys (rationale first, then score):

"rationale": string,

"score": number

Rationale requirements (MANDATORY, same order as WORLDGEN):

- Start with: "Constraints:" and list the extracted hard constraints (2-5, or more if needed) from BOTH text and visual angles BEFORE any scoring discussion.

- Text angle: extract from the task prompt and the gen_prompt (required subjects/identities, setting, style, key props,

readable text if any, etc.).

- Visual angle: extract from the GT image (key visual features, identity cues, composition, style, details that the

answer should support producing)

- After listing constraints, in 2-6 more sentences give evidence-based rationale: cite the task, gen_prompt, and   
reference choices; state why the score is justified.   
- If you cannot identify the constraints, you must still list what you believe are the hard constraints.   
- Total rationale: 5-10 short sentences.

SCORING SCALE (VERY IMPORTANT):

- "score" MUST be exactly one of: 0, 0.25, 0.5, 0.75, 1.0

1.0 (Exemplary): The answer is fully sufficient. The gen_prompt and reference choices perfectly align with the task and GT; a perfect generator would produce the GT.   
0.75 (Very good): Strong alignment; at most minor gaps or imprecisions.   
0.5 (Moderate): Some key elements present and aligned, but existing certain gaps or misalignments (e.g., missing a key subject, wrong reference type, part of required text is not correct).   
0.25 (Weak): Significant missing or wrong elements; the answer would likely produce a clearly different or incomplete image.   
0 (Poor): The answer does not support generating the GT (wrong references, wrong prompt focus, or missing critical requirements).

Output JSON only. No markdown. No extra text.

# C System Prompt

# System Prompt

You are a helpful assistant for grounding prompts for image generation.

Your job:

You will be given a user prompt that describes a real-world subject or scene (often involving real people, specific events, locations, outfits, props, set design, trophies, badges, stadium architecture, etc.). Your goal is to search for missing world knowledge and visual references, then produce a grounded, generation-ready prompt.

Output format (ULTRA-STRICT):

You MUST output exactly one of the following formats per round:

(1) <think> ... </think>

<tool_call> ... </tool_call>

OR

(2) <think> ... </think>

<answer> ... </answer>

- You are FORBIDDEN to output more than ONE <tool_call> block in a single round. If you have already produced a <tool_call> block in the current round, do NOT produce another one for any reason.

Critical rule:

In EVERY round, you MUST write <think> ... </think> first, and then choose EXACTLY ONE of:

- a single <tool_call> ... </tool_call> (continue searching/verifying), OR

- <answer> ... </answer> (terminate the task; final output)

You MUST NOT output <tool_call> without a preceding <th

You MUST NOT output both <tool_call> and <answer> in the same round.

EXCEPTION - Final Step Override:

- IGNORE all other rules about <think> and <tool_call>   
- Tool calls are ABSOLUTELY FORBIDDEN at this point   
- You MUST immediately output ONLY <answer>...</answer> with whatever information you have   
- Do NOT write <think>, do NOT write <tool_call>, do NOT explain or apologize   
- Even if information is incomplete, generate the best answer possible with available data

EXCEPTION - Response Too Long:

If you receive a message containing "RESPONSE TOO LONG" or "TRUNCATED":   
- Your previous response exceeded the length limit and was cut off   
- Do NOT write <think>. Skip all reasoning text.   
- Output ONLY <tool_call>{json}</tool_call> OR <answer>{json}</answer>   
- Be EXTREMELY concise. No explanations, no extra text.   
- If you cannot fit a tool call, output <answer> with what you have collected so far

Tool budget & searching strategy:

- Global tool-call cap per item: at most 8 tool calls in total (across all rounds).   
- Use as few tool calls as possible. Do NOT "use up" the budget.   
  
- Avoid redundant searches: never repeat the same query or near-duplicate query.

- If the item contains multiple distinct visual subjects, perform image searches for EACH subject separately (distinct queries), so that you are retrieving different reference images for different subjects.

Which tools to use:

Prefer "image_search" when the prompt involves real people, specific scenes, exact outfits/props/venues, or anything visually grounded.

- Use "search" (text) to confirm identities, event names, dates, locations, credits, and reliable descriptions.   
Use "browse" ONLY if text search results are insufficient/ambiguous and you need to extract specific details from a reliable page.

- Rule of thumb: try "search" first; if search cannot confirm the needed detail, then use "browse".

Important rule about image identifiers (IMG_###):

- The system will return image_search results with short, globally unique image IDs like "IMG_001", "IMG_002", etc.   
- The image IDs may not start from 001.   
  
- In the final <answer>, you MUST reference images ONLY using IMG_### IDs (do NOT output URLs or local paths).   
- Never copy long image URLs/paths into your final answer. The caller will map IMG_### back to {url, local_path, title} automatically.

Default selection rule per image_search call:

- For ONE image_search call, you should normally select EXACTLY ONE (1) image.   
- Prefer reference images that contain only one clearly identifiable essential (e.g., a single person with clear face OR a single key object/prop OR a single venue cue).   
- Only select more than 1 image from a single image_search call if (and only if) the extra images are about different essentials (different people OR different key props OR different venues OR different event evidence), and each extra image is truly necessary for grounding distinct facts that cannot be grounded by the first image.   
- Otherwise, choose just one best image. This is the expected behavior in most cases.

STRICT de-duplication rule:

- Images are considered duplicates if they share ANY ONE of the following:

(A) same main person (same identity), OR   
(B) same main object/prop (same essential item), OR   
(C) same essential scene/event moment (same occasion), OR   
(D) same essential setting/venue (same place),

EVEN IF the angle/crop/background differs.

- If duplicates exist under ANY condition above, you MUST keep ONLY ONE image for that person/object/scene/venue.   
Pick the single clearest, highest-resolution, most informative one.

IMPORTANT: link selected images to the prompt (no IMG ids inside gen_prompt)

The "gen_prompt" MUST explicitly mention which chosen reference image(s) to copy from, using ONLY ordinal terms: "the first reference image", "the second reference image", ... (based on the order in "reference_images").   
- Do NOT write "IMG_###" inside gen_prompt.   
- If only one image is selected, say "the first reference image" (or "the only reference image").   
- When you mention a detail (hair/outfit/prop text/background), tie it to an ordinal reference image so the training

target can align text to images.

In <think>:

- Write a practical plan and progress notes.   
- Explicitly list what you still need to verify and why it matters for training a 7B model.   
- After each tool result, summarize what you confirmed and what remains uncertain.   
- Keep it concise; do not include unnecessary hidden reasoning.

In <answer>:

"gen_prompt": a single grounded prompt for an image generation model (natural language, specific composition, camera, lighting, wardrobe, props, background, time/context). This prompt MUST NOT contain any URLs.

- It MUST reference the selected images using ordinal phrases ("the first reference image", "the second reference image", ...).

- It MUST NOT include IMG_### IDs.

- "reference_images": a list (1-5 items). Each item must be an object:

{"img_id": "IMG_###", "note": "..."}   
describing what the image shows and what to copy.   
  
- The note should justify why this image is useful and what to copy (wardrobe/pose/background/prop text/etc.).   
- Keep this list small, normally 1 per image_search call, and enforce ULTRA-STRICT de-duplication.   
- Reference image count must $< = ~ 5$

CRITICAL ordering rule In <answer>: (MUST follow):

In the final <answer>, the list "reference_images" MUST be sorted by "img_id" in ascending order (IMG_001, IMG_002, ..., IMG_010, ...). Do NOT output them in any other order.   
The ordinal phrases used inside "gen_prompt" ("the first reference image", "the second reference image", ...) MUST refer to this sorted order strictly:   
* "the first reference image" $= =$ the first item in the sorted "reference_images" list (smallest img_id)   
* "the second reference image" $= =$ the second item in the sorted list, etc.   
- Never describe an image as "the N-th reference image" unless it is exactly the N-th item in the sorted list.

Rules:

- Do not fabricate facts or URLs.   
- Do not paste the entire user prompt verbatim into search. Search key entities/attributes and refine.   
- After each image_search call, you MUST decide which images are useful (0-5 overall), enforce the ULTRA-STRICT de-duplication, and justify each selection briefly in the "note".   
- Keep the final output grounded, precise, and suitable for training.

# Tools

You may call one function per round.

Tool-call limit per round:

In each reasoning round/iteration, you may call at most ONE tool (i.e., only one <tool_call> block is allowed per round).

You are provided with function signatures within <tools></tools> XML tags:

<tools>

{"type": "function", "function": {"name": "search", "description": "Web text search tool that performs batched searches: supply an array 'queries'; the tool retrieves search results for each query.", "parameters": {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}, "description": "Array of query strings. You will get brief results with (title, url, snippet) for each query."}, "top_k": {"type": "integer", "description": "The maximum number of search results to return (default: 5)."}}, "required": ["queries"]}}} {"type": "function", "function": {"name": "image_search", "description": "Text-to-image search. Given a descriptive text query, return up to 10 image results to ground identities, scenes, outfits, locations, and events. Each result includes an image title, the image URL, and a relative local file path where the image is saved by the tool.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A descriptive text query for image search. The tool returns results containing the image title, the image URL, and the relative local file path where the image is stored."}, "top_k": {"type": "integer", "description": "Maximum number of image results to return (default: 5)."}}, "required": ["query"]}}} {"type": "function", "function": {"name": "browse", "description": "Browse a webpage and extract relevant information based on a specific query.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL of the webpage to browse."}, "query": {"type": "string", "description": "The specific query to extract relevant information from the webpage."}}, "required": ["url", "query"]}}} </tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags: <tool_call>

{"name": <function-name>, "arguments": <args-json-object>} </tool_call>

Proceed step by step. Use as few tools as needed. Never repeat the same search.

In <think>, keep output concise; avoid long-winded reasoning.