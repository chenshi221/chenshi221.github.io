---
title: "VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents"
source: "https://arxiv.org/abs/2507.04590"
author: "Rui Meng et al."
published: "2025"
created: "2026-05-07"
description: null
tags: ["clippings"]
arxiv: "2507.04590"
url: "https://arxiv.org/abs/2507.04590"
---

# VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents

Rui Meng1∗† Ziyan Jiang2∗ Ye Liu1 Mingyi $\mathbf { S } \mathbf { u } ^ { 3 }$ Xinyi Yang1 Yuepeng Fu4 Can Qin1 Zeyuan Chen1 Ran ${ \bf \chi } _ { \bf u } ^ { 1 }$ Caiming Xiong1 Yingbo Zhou1 Wenhu Chen3 Semih Yavuz1

1Salesforce Research 2UC Santa Barbara 3University of Waterloo 4Tsinghua University

https://tiger-ai-lab.github.io/VLM2Vec/

# Abstract

Multimodal embedding models have been crucial in enabling various downstream tasks such as semantic similarity, information retrieval, and clustering over different modalities. However, existing multimodal embeddings like VLM2Vec, E5-V, GME are predominantly focused on natural images, with limited support for other visual forms such as videos and visual documents. This restricts their applicability in real-world scenarios, including AI agents, multi-modal search and recommendation, and retrieval-augmented generation (RAG). To close this gap, we propose VLM2Vec-V2, a unified framework for learning embeddings across diverse visual forms. First, we introduce MMEB-V2, a comprehensive benchmark that extends MMEB with five new task types: visual document retrieval, video retrieval, temporal grounding, video classification and video question answering – spanning text, image, video, and visual document inputs. Next, we train VLM2Vec-V2, a general-purpose embedding model that supports text, image, video, and visual document inputs. Extensive experiments show that VLM2Vec-V2 achieves strong performance not only on the newly introduced video and document retrieval tasks, but also improves over prior baselines on the original image benchmarks. Through extensive evaluation, our study offers insights into the generalizability of various multimodal embedding models and highlights effective strategies for unified embedding learning, laying the groundwork for more scalable and adaptable representation learning in both research and real-world settings.

# 1 Introduction

Embedding models play a crucial role in connecting data across various modalities. By encoding heterogeneous multimodal data into a shared dense representation space, they enable many down-stream applications like classification, clustering, retrieval, etc. In recent years, we have witnessed significant advances in embedding models, largely driven by the progress of large foundation models. For instance, recent breakthroughs in text embedding (Su et al., 2023; Wang et al., 2024a; Meng et al., 2024; BehnamGhader et al., 2024) have been achieved by integrating pretrained large language models with multi-task instruction embedding tuning. Similarly, Jiang et al. (2024); Zhang et al. (2024b); Chen et al. (2025) demonstrated strong performance across multiple text-image tasks by instruction-tuning vision language models (VLMs) into effective embedding models.

Existing multimodal embedding models are trained on datasets like MMEB (Jiang et al., 2024) and M-BEIR (Wei et al., 2023), which are focused predominantly on natural images or photographs, sourced from MSCOCO (Lin et al., 2014), Flickr (Plummer et al., 2015) and

ImageNet (Deng et al., 2009) datasets. These datasets fail to cover broader forms of visual information like documents, pdf, websites, videos, slides, etc. The lack of coverage causes the existing embedding models to fall behind on many realistic tasks like article searching, website searching, Youtube video search, etc.

To address these limitations, we introduce MMEB-V2, an advanced multimodal embedding dataset designed to train and evaluate embedding models across three key visual modalities: images, videos, and visual documents. Expanding on the original MMEB (Jiang et al., 2024) framework, MMEB-V2 broadens the evaluation scope to encompass five new tasks, including four video-based tasks—Video Retrieval, Moment Retrieval, Video Classification, and Video Question Answering — as well as one task centered on visual documents: Visual Document Retrieval. This comprehensive suite of tasks allows for robust assessment of multimodal embedding models across static, temporal, and structured visual data settings.

Built on top of MMEB-V2, we propose VLM2Vec-V2, a strong multimodal embedding model fine-tuned from state-of-the-art vision-language models (Wang et al., 2024b). VLM2Vec-V2 is trained using a mixture of instruction-following tasks spanning multiple task categories, enabling it to produce unified representations for a wide variety of visual modalities.

Through VLM2Vec-V2 and MMEB-V2, we aim to investigate the following research questions: How well can a multimodal embedding generalize across diverse visual modalities? What are the key ingredients for training robust and versatile multimodal embedding models? What are the key challenges in representing temporal information in videos and structured information in visual documents?

The contributions of this work are threefold.

• We propose MMEB-V2, a comprehensive dataset for systematically evaluating embedding models on diverse tasks involving videos, images, and visual documents.   
• We develop VLM2Vec-V2, a unified multimodal embedding model that supports diverse input formats, and follows task instructions to produce general-purpose embeddings to support various downstream tasks.   
• Our experiments show that VLM2Vec-V2 outperforms prior baselines across 78 datasets. Through detailed ablations, we identify effective training strategies for learning embedding models across modalities.

# 2 MMEB-V2: Expanding Embedding Benchmarks Beyond Image-Text

We introduce MMEB-V2, a comprehensive dataset designed to evaluate model performance on multimodal embedding tasks involving various combinations of text, image, video, and visual document modalities. In addition to the five task categories that involve natural images and text, MMEB-V2 includes four video understanding tasks and one visual document understanding task. Figure 1 presents an overview of MMEB-V2.

Each task presents the model with a query and a corresponding set of candidate responses, where the goal is to select the correct target. The query consists of a combination of an instruction, a textual component, and a video or document. For video-based tasks, videos are represented as sequences of frames sampled at uniform intervals from the raw footage to ensure consistent temporal coverage. Instructions serve to specify the model’s objective (e.g., “Recognize the category of the video contents.”), and query texts can be questions, descriptions, or commands specific to the video (e.g., “How many red socks are above the fireplace at the end of this video?”, “Select the clips of videos that contain a dolphin.”). Each task is associated with a specific target type, which varies depending on the nature of the task. For example, in video classification, the model must identify the activity or object class label (e.g., “Yoga” or “Car”).

To make the dataset practical and accessible for future research, we selectively downsample certain datasets to ensure that the full benchmark can be run within a reasonable amount of time. Our datasets span diverse domains, including sports, object recognition, daily-life activities, and movie or TV show clips. These samples are drawn from varied sources such

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/89776f0d0be6682fdf8b6bc9b1c5ba575ea82314612e745a72491e032ce6e986.jpg]]

# MMEB-V2

Massive Multimodal Embedding Benchmark

# Multi-task

9 Meta-tasks/78 Tasks

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/0953bb2745ab37dccb221100c97b6704e09b1adf922b98f04d5eecfeeebc43f1.jpg]]

# Multimodal

Text/Image/Video/Document

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/11c41c514a27686fb5253a5159ef4ac676628f9d05a83a5ad71fbdc426e2e49f.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/e37ce85ef64591f69a9111a7a060ddc0b1d9d66f732c568baf2e8329f14fdeff.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/d17c700d1adb8bfecacfc66a427f02c493242e7156337c342380af3fb38eabf4.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/1fa0bb43ad7d2a3495c0e765a256faaac37e60a6de0ff52200dc9e39789bc4f6.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/2b8e5fcd1776474ec24eeb6bf0e1a1b51973942b654b1be26cc8ea4dadbac9a4.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/53c1c98be56698f440e87a6a163aaf648d2adc780ea78bf82510a4ebfe93a0cf.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/f781eff544fe88cc9c7c7227a6b4bd0205e312d9cf408e4a760ad25443186173.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/c3ee7c774e2f858ca5bec57fc73bafb997ccfcd7c8be9e089e3c20337592b2c1.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/046a2d95eb79087f740ac259df612cd1db1392d64c5ac5c4a68079c2480f7156.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/040652e90acba7225c1a38a029909f704b1ef6c9f9aae79f9df80b8edda2435a.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/1f24c23b239cabb9b10701f14686ac81c7249090c7dd99df7bf256822a0bfcd5.jpg]]  
Figure 1: An overview of MMEB-V2, which includes 9 meta-tasks and 78 tasks in total. In addition to the original MMEB benchmark, MMEB-V2 introduces five new meta-tasks focused on video and visual documents. Tasks from MMEB are indicated with blue borders, while newly introduced tasks in MMEB-V2 are marked with red borders.

as YouTube, professional productions, and crowd-sourced content, ensuring both diversity and real-world relevance. Summary statistics for each task are presented in Table 1. Details about constructing each dataset are provided in Appendix A.2.

• Video Retrieval (V-RET) The query consists of an instruction, a descriptive text related to the video content, and a sequence of video frames. The model must retrieve the correct corresponding video from a pool of thousands of video candidates.   
• Moment Retrieval (M-RET) The query consists of an instruction, a textual description, and optionally a full video, with the goal of retrieving the temporal segments that best matches the description. The model must select the ground-truth clip from approximately 10 candidate segments within the full video.   
• Video Classification (V-CLS) Given an instruction and a sequence of video frames, the model is tasked with predicting the correct class label—related to the scene or action depicted—from a set of possible classes.   
• Video QA (V-QA) The input consists of an instruction, a textual question, and a video. The model must select the correct answer from several options, including one correct choice and many distractors.   
• Visual Document Retrieval (VisDoc) This task category evaluates the model’s ability to retrieve structured visual documents—such as multi-page PDFs and slide decks—in response to natural language queries. We include five datasets in this benchmark. ViDoRe V1 & V2 (Faysse et al., 2024; Mace et al. ´ , 2025) and VisRAG (Yu et al., 2024) are composed of multiple document QA datasets and cover a broad range of document types and use cases (e.g., charts and slides), though they were not originally designed for retrieval. To complement them, we include ViDoSeek (Wang et al., 2025) and MMLongBench-Doc (Ma et al., 2024), which provide fine-grained, page-level annotations suitable for retrieval evaluation. Besides, we reformat the two datasets to support both document-level and page-level evaluation. The final VisDoc score is computed as the average of NDCG@5 scores across 24 tasks.

# 3 Unified Embedding Model for Video, Image, and Visual Document

Unifying embedding learning across diverse modalities and tasks is inherently challenging due to their distinct structural and semantic characteristics. Our goal is to align data from

Table 1: The statistics of MMEB-V2, which includes 42 tasks across five meta-task categories in addition to the original MMEB, are summarized below. Here, we list only the additional datasets introduced beyond those in MMEB. We consider four modalities (MOD): T (Text), I (Image), V (Video), and D (Visual Document).   

<table><tr><td>Task</td><td>Query MOD</td><td>Target MOD</td><td>Domain</td><td>#Query</td><td>#Candidates</td></tr><tr><td colspan="6">Video Retrieval (5 Tasks)</td></tr><tr><td>DiDeMo</td><td>T</td><td>V</td><td>Open</td><td>1,004</td><td>1,004</td></tr><tr><td>MSR-VTT</td><td>T</td><td>V</td><td>Open</td><td>1,000</td><td>1,000</td></tr><tr><td>MSVD</td><td>T</td><td>V</td><td>Open</td><td>670</td><td>670</td></tr><tr><td>VATEX</td><td>T</td><td>V</td><td>Open</td><td>4,468</td><td>4,468</td></tr><tr><td>YouCook2</td><td>T</td><td>V</td><td>Cooking</td><td>3,179</td><td>3,179</td></tr><tr><td colspan="6">Moment Retrieval (3 Tasks)</td></tr><tr><td>QVHighlights</td><td>T + V</td><td>V</td><td>Vlog/News</td><td>1,083</td><td>10</td></tr><tr><td>Charades-STA</td><td>T + V</td><td>V</td><td>Activity</td><td>727</td><td>10</td></tr><tr><td>MomentSeeker</td><td>I + V</td><td>V</td><td>Open</td><td>1,800</td><td>10</td></tr><tr><td colspan="6">Video Classification (5 Tasks)</td></tr><tr><td>Kinetics-700</td><td>V</td><td>T</td><td>Open</td><td>1,000</td><td>700</td></tr><tr><td>SSv2</td><td>V</td><td>T</td><td>Human-Object Interaction</td><td>1,000</td><td>174</td></tr><tr><td>HMDB51</td><td>V</td><td>T</td><td>Open</td><td>1,000</td><td>51</td></tr><tr><td>UCF101</td><td>V</td><td>T</td><td>Open</td><td>1,000</td><td>101</td></tr><tr><td>Breakfast</td><td>V</td><td>T</td><td>Cooking</td><td>433</td><td>10</td></tr><tr><td colspan="6">Video QA (5 Tasks)</td></tr><tr><td>MVBench</td><td>V + T</td><td>T</td><td>Spatial/Temporal</td><td>4,000</td><td>3 ~ 5</td></tr><tr><td>Video-MME</td><td>V + T</td><td>T</td><td>Real-world</td><td>900</td><td>4</td></tr><tr><td>NExT-QA</td><td>V + T</td><td>T</td><td>Daily activity</td><td>8,564</td><td>5</td></tr><tr><td>EgoSchema</td><td>V + T</td><td>T</td><td>Egocentric</td><td>500</td><td>5</td></tr><tr><td>ActivityNetQA</td><td>V + T</td><td>T</td><td>Activity</td><td>1000</td><td>2</td></tr><tr><td colspan="6">Visual Document Retrieval (24 Tasks)</td></tr><tr><td>ViDoRe (10)</td><td>T</td><td>D</td><td>Documents</td><td>280 - 1,646</td><td>70 - 999</td></tr><tr><td>ViDoRe-V2 (4)</td><td>T</td><td>D</td><td>Documents</td><td>52 - 640</td><td>452 - 1,538</td></tr><tr><td>VisRAG (6)</td><td>T</td><td>D</td><td>Documents</td><td>63 - 816</td><td>500 - 9,590</td></tr><tr><td>ViDoSeek (2)</td><td>T</td><td>D</td><td>Documents</td><td>1,142</td><td>5,349</td></tr><tr><td>MMLongBench-Doc (2)</td><td>T</td><td>D</td><td>Documents</td><td>838</td><td>6,492</td></tr></table>

different modalities in a shared embedding space, while guiding the model’s behavior through natural language instructions that define each task.

This section outlines our approach, including multimodal input formatting (Section 3.1), unified encoding with a shared backbone (Section 3.2), instruction-guided contrastive training (Section 3.3), and strategic sampling to balance data sources (Section 3.4).

# 3.1 Unified Representation of Multimodal Data

Our objective is to learn a unified embedding space that supports diverse visual modalities and tasks. This requires a model backbone capable of flexibly encoding interleaved sequences of text, images, and videos, while also handling long-form inputs such as fulllength videos and multi-page visual documents. Vision-language models (Liu et al., 2024a) have shown strong performance across benchmarks and have proven effective as foundations for multimodal embedding models (Jiang et al., 2024; Lin et al., 2024).

Based on these criteria, we adopt Qwen2-VL (Wang et al., 2024b) as the backbone of VLM2Vec-V2. Qwen2-VL is particularly well-suited for our needs, offering (1) Naive Dynamic Resolution for efficiently processing inputs with variable resolutions, (2) Multimodal Rotary Position Embedding (M-RoPE) to capture spatial and temporal structure, and (3) a unified architecture that integrates 2D and 3D convolutions for consistent image and video understanding. These capabilities enable scalable and generalizable encoding across heterogeneous multimodal data.

# 3.2 Contrastive Learning

We adopt contrastive training to adapt a vision-language model into an embedding model.

Given a pretrained VLM, we feed query and target into it to obtain the query and target embeddings $( \mathbf { h } _ { q _ { \mathrm { i n s t } } } , \mathbf { h } _ { t ^ { + } } )$ by taking the last layer vector representation of the last token. To train the embedding model, we adopt the standard InfoNCE loss (Oord et al., 2018) $\mathcal { L }$ over the in-batch negatives and hard negatives:

$$
\min  \mathcal {L} = - \log \frac {\phi \left(\mathbf {h} _ {q _ {\text {i n s t}}}, \mathbf {h} _ {t ^ {+}}\right)}{\phi \left(\mathbf {h} _ {q _ {\text {i n s t}}}, \mathbf {h} _ {t ^ {+}}\right) + \sum_ {t ^ {-} \in \mathbb {N}} \phi \left(\mathbf {h} _ {q _ {\text {i n s t}}}, \mathbf {h} _ {t ^ {-}}\right)}, \tag {1}
$$

where N denotes the set of all negatives, and $\phi ( \mathbf { h } _ { q } , \mathbf { h } _ { t } )$ is a function that computes the matching score between query $q$ and target t. In this paper, we adopt the temperature-scaled cosine similarity function as $\begin{array} { r } { \phi ( \mathbf { h } _ { q } , \mathbf { h } _ { t } ) = \exp ( \frac { 1 } { \tau } \cos ( \mathbf { \bar { h } } _ { q } , \mathbf { h } _ { t } ) ) } \end{array}$ ),where $\tau$ is a temperature.

# 3.3 Multi-modal Data Formatting

To train a unified embedding model across diverse tasks and modalities, we adopt a standardized format for all query-target pairs. Each training example is denoted as a pair $( q , t ^ { + } )$ , where $q$ is the query and $t ^ { + }$ is the positive target. Both components can be a single image, multiple images, text, or interleaved sequences of text and images.

In addition to the raw input pair, we introduce an instruction inst that specifies the taskspecific relationship between the query and target. This guidance helps the model better contextualize and generalize across heterogeneous tasks. We apply the instruction to the original query $q$ to generate an instruction-conditioned version $q _ { \mathrm { i n s t } }$ :

$$
q _ {\text {i n s t}} = [ \text {V I S U A L T O K E N} ] \text {I n s t r u c t :} \{\text {t a s k i n s t r u c t i o n} \} \backslash \mathrm {n} \quad \text {Q u e r y :} \{q \}, \tag {2}
$$

where {task instruction} is a one-sentence description of the embedding task, e.g. “’Find a video that contains the following visual content:’.

[VISUAL TOKEN] is a modality-specific token prepended to indicate whether the visual input is an image or a video. For example, Qwen2-VL uses $< | i m a g e _ { - } p a d | >$ $<$ for image inputs and $< | v i d e o _ { - } p a d | >$ for video inputs.

Similarly, we optionally apply a simple instruction to the target input $t ^ { + }$ to guide representation learning, e.g. “Understand the content of the provided video:”:

$$
t ^ {+} = \left[ \text {V I S U A L \_ T O K E N} \right] \{\text {t a r g e t \_ i n s t r u c t i o n} \}. \tag {3}
$$

This unified formatting allows the model to handle multimodal inputs consistently while leveraging instruction signals to improve cross-task and cross-modality generalization.

# 3.4 Data Sampling Strategies

To support effective multi-task training over heterogeneous data sources, we design a flexible and scalable data sampling pipeline with two key components.

First, we perform on-the-fly batch mixing guided by a pre-defined sampling weight table. This table specifies the relative probabilities of sampling from each dataset, enabling controlled exposure to different task types. By dynamically drawing samples during training, we ensure balanced coverage and prevent overfitting to any single modality or domain.

Second, we introduce an interleaved sub-batching strategy to enhance the hardness and stability of contrastive learning. Specifically, each full batch (e.g., size 1024) is divided into smaller sub-batches (e.g., 8 sub-batches of size 128), where each sub-batch is sampled independently. Compared to per-sample independent sampling, grouping similar samples into sub-batches increases intra-sub-batch homogeneity, which raises the difficulty of contrastive discrimination. At the same time, interleaving multiple such sub-batches preserves cross-task diversity within the full batch, avoiding the instability commonly observed in completely homogeneous batches that originate from a single source. This strategy strikes a balance between sample diversity and structural consistency, fostering more stable and robust optimization dynamics.

# 4 Experiments

# 4.1 Experiment Setting

# 4.1.1 Training Data

To train VLM2Vec-V2 effectively across diverse multimodal tasks, we curate a training dataset comprising three main sources: video-language instruction data, visual document retrieval, and image-based vision tasks.

First, we utilize training data from LLaVA-Hound (Zhang et al., 2024a), which includes synthetic video-caption pairs and video QA examples generated by ChatGPT. Specifically, we use 300k video-caption pairs and 240k video QA pairs. For the caption data, we adopt two formats: using the caption as the query and video as the target in a video retrieval setup, or using the video as the query to retrieve the most relevant textual description from candidate captions.

Second, for visual document retrieval tasks, we incorporate datasets from ViDoRe (Faysse et al., 2024) and VisRAG (Yu et al., 2024), including colpali train set (118k), VisRAG synthetic (239k), and VisRAG in-domain (123k), which provide training examples for imagebased document understanding and retrieval.

Finally, we include image-text datasets from MMEB-train (Jiang et al., 2024) to support generalization across a wide range of visual understanding tasks, including question answering, classification, retrieval, and visual grounding. These datasets help improve the robustness of the learned embeddings across multiple tasks.

# 4.1.2 Training Setting

We train VLM2Vec-V2 using Qwen2-VL 2B (Wang et al., 2024b) as backbone, a batch size of 1,024 for 2K/5K steps and an interleaved sub-batch size of 64, with the loss temperature set to 0.02. To support scalable training, we use GradCache (Gao et al., 2021) to enable large global batch sizes and run all experiments on 8 H100 GPUs. For parameter-efficient training, we apply LoRA tuning with a rank of 16 and scaling factor $\alpha \stackrel { \bullet } { = } 3 2$ using the PEFT framework (Mangrulkar et al., 2022). We use 8 uniformly sampled frames to represent each video during both training and evaluation.

We use $\mathrm { H i t @ 1 }$ as the primary evaluation metric for all video and image tasks, measuring the proportion of queries where the correct target is ranked at the top. For visual document tasks, we report NDCG@5 to remain consistent with prior work in this domain.

# 4.1.3 Baselines

We compare against several VLM embedding models, including GME (Zhang et al., 2024b), VLM2Vec (Jiang et al., 2024), and LamRA (Liu et al., 2024b), most of which are primarily trained on image-text pairs. While these models are not explicitly designed for video tasks, many can be adapted to handle video inputs by encoding multiple frames as sequential images. For video evaluation, GME and LamRA use a single middle frame, while the remaining models use 8 uniformly sampled frames.

In addition, to provide a fair comparison across modalities, we evaluate VLM2Vec-V2 against representative models specialized for each modality. Specifically, we include Col-Pali (Faysse et al., 2024) (v1.3), a model tailored for document retrieval using a late interaction matching mechanism.

# 4.2 Main Results

Table 2 presents a comprehensive comparison between VLM2Vec-V2 and a diverse set of baseline models across 78 datasets covering image, video, and visual document tasks. Full results are detailed in Appendix A.4. VLM2Vec-V2 achieves the highest overall average score (58.0), outperforming multiple strong baselines, including GME, LamRA and VLM2Vec,

Table 2: Performance comparison between baseline models and our VLM2Vec-V2 across image, video, and visual document tasks. CLS: classification, QA: question answering, RET: retrieval, GD: grounding, MRET: moment retrieval, VDR: ViDoRe, VR: VisRAG, OOD: out-of-domain.   

<table><tr><td rowspan="2">Model</td><td colspan="5">Image</td><td colspan="5">Video</td><td colspan="5">VisDoc</td><td rowspan="2">All</td></tr><tr><td>CLS</td><td>QA</td><td>RET</td><td>GD</td><td>Overall</td><td>CLS</td><td>QA</td><td>RET</td><td>MRET</td><td>Overall</td><td>VDRv1</td><td>VDRv2</td><td>VR</td><td>OOD</td><td>Overall</td></tr><tr><td># of Datasets →</td><td>10</td><td>10</td><td>12</td><td>4</td><td>36</td><td>5</td><td>5</td><td>5</td><td>3</td><td>18</td><td>10</td><td>4</td><td>6</td><td>4</td><td>24</td><td>78</td></tr><tr><td colspan="17">Baseline Models</td></tr><tr><td>ColPali v1.3 (3B)</td><td>40.3</td><td>11.5</td><td>48.1</td><td>40.3</td><td>34.9</td><td>26.7</td><td>37.8</td><td>21.6</td><td>25.5</td><td>28.2</td><td>83.6</td><td>52.0</td><td>81.1</td><td>43.1</td><td>71.0</td><td>44.4</td></tr><tr><td>GME (2B)</td><td>54.4</td><td>29.9</td><td>66.9</td><td>55.5</td><td>51.9</td><td>34.9</td><td>42.0</td><td>25.6</td><td>32.4</td><td>33.9</td><td>86.1</td><td>54.0</td><td>82.5</td><td>43.1</td><td>72.7</td><td>54.1</td></tr><tr><td>GME (7B)</td><td>57.7</td><td>34.7</td><td>71.2</td><td>59.3</td><td>56.0</td><td>37.4</td><td>50.4</td><td>28.4</td><td>38.2</td><td>38.6</td><td>89.4</td><td>55.6</td><td>85.0</td><td>44.4</td><td>75.2</td><td>57.8</td></tr><tr><td>LamRA-Qwen2 (7B)</td><td>59.2</td><td>26.5</td><td>70.0</td><td>62.7</td><td>54.1</td><td>39.3</td><td>42.6</td><td>24.3</td><td>34.6</td><td>35.2</td><td>22.0</td><td>11.5</td><td>37.4</td><td>21.0</td><td>23.9</td><td>40.4</td></tr><tr><td>LamRA-Qwen2.5 (7B)</td><td>51.7</td><td>34.1</td><td>66.9</td><td>56.7</td><td>52.4</td><td>32.9</td><td>42.6</td><td>23.2</td><td>37.6</td><td>33.7</td><td>56.3</td><td>33.3</td><td>58.2</td><td>40.1</td><td>50.2</td><td>47.4</td></tr><tr><td>VLM2Vec-Qwen2VL (2B)</td><td>58.7</td><td>49.3</td><td>65.0</td><td>72.9</td><td>59.7</td><td>33.4</td><td>30.5</td><td>20.6</td><td>33.0</td><td>29.0</td><td>49.8</td><td>13.5</td><td>51.8</td><td>33.5</td><td>41.6</td><td>47.0</td></tr><tr><td>VLM2Vec-Qwen2VL (7B)</td><td>62.7</td><td>56.9</td><td>69.4</td><td>82.2</td><td>65.5</td><td>39.1</td><td>30.0</td><td>29.0</td><td>40.6</td><td>34.0</td><td>56.9</td><td>9.4</td><td>59.1</td><td>38.1</td><td>46.4</td><td>52.3</td></tr><tr><td colspan="17">Ours</td></tr><tr><td>VLM2Vec-V2 (2B)</td><td>62.9</td><td>56.3</td><td>69.5</td><td>77.3</td><td>64.9</td><td>39.3</td><td>34.3</td><td>28.8</td><td>38.5</td><td>34.9</td><td>75.5</td><td>44.9</td><td>79.4</td><td>39.4</td><td>65.4</td><td>58.0</td></tr></table>

which were built on the same Qwen2-VL backbone. This highlights the effectiveness of our unified training approach in delivering strong and balanced performance across different modalities and tasks. On image tasks, VLM2Vec-V2 shows strong results, outperforming most baselines by a large margin and achieving performance comparable to VLM2Vec-7B despite being only 2B in size. For video tasks, it achieves competitive performance despite being trained on a relatively small amount of video data. In visual document retrieval, VLM2Vec-V2 outperforms all VLM2Vec variants, although still trailing behind ColPali, which is specifically optimized for VisDoc tasks.

# 4.3 Ablation Analysis

# 4.3.1 Generalization Across Modalities

To evaluate the impact of different modality sources on model performance, we consider three types of training data: image-based data (Image), visual document data (VisDoc), and video data (Video). In addition to training models on each individual modality, we construct datasets that combine two modalities (Image+VisDoc, Image+Video) as well as all three modalities (Image+VisDoc+Video). This setup allows us to systematically analyze the contribution of each modality and the effect of multi-modal combinations on model performance. By comparing models trained with single-modality and multi-modality data, we aim to assess how multi-modal training influences generalization and task effectiveness. Vidore-V2 is not used in the ablation studies.

As shown in Table 3, among single-modality models, training on image data yields the highest average performance. For two-modality combinations, Image+Video slightly outperforms Image+VisDoc, with noticeable gains on image benchmarks in particular. Notably, incorporating all three modalities leads to the best performance on visual document tasks and the highest overall score, highlighting the benefit of comprehensive training data.

Table 3: Performance comparison of models trained on different combinations of modality data. Rows indicate evaluation performance per modality (Image, Video, VisDoc), while columns represent the modality or modality combinations used during training.   

<table><tr><td>Modality</td><td>Image</td><td>VisDoc</td><td>Video</td><td>Image+Video</td><td>Image+VisDoc</td><td>Image+Video+VisDoc</td></tr><tr><td>Image</td><td>62.5</td><td>27.9</td><td>33.9</td><td>63.3</td><td>62.4</td><td>62.7</td></tr><tr><td>VisDoc</td><td>41.5</td><td>42.6</td><td>47.9</td><td>51.9</td><td>47.4</td><td>52.2</td></tr><tr><td>Video</td><td>31.5</td><td>29.1</td><td>19.9</td><td>29.7</td><td>33.3</td><td>32.4</td></tr><tr><td>AVG</td><td>45.2</td><td>33.2</td><td>33.9</td><td>48.3</td><td>47.7</td><td>49.1</td></tr></table>

# 4.3.2 Ablation Study on Data Sampling Strategies

As part of our ablation study, we investigate the impact of interleaved sub-batching on model performance across the three modalities. When the interleaved sub-batch size (IB) is set to 0, all samples in the batch are randomly drawn from different sources without grouping. A value of 64 indicates that a batch of size 1,024 is divided into sub-batches of size 64, resulting in data from 16 distinct sources per batch. At the other extreme, an IB of 1024 means the entire batch comes from a single source, effectively disabling interleaving. This setup allows us to analyze how different levels of source mixing influence training dynamics and cross-modal generalization.

As shown in Table 4, increasing the sub-batch size consistently improves performance for both VisDoc and Video. Conversely, the best performance on the Image modality is achieved with a sub-batch size of 64, exhibiting an inverted U-shaped trend—monotonically increasing from 0 to 64, followed by a monotonic decline from 64 to 1024.

Table 4: Performance comparison across different sub-batch size for different modalities.   

<table><tr><td>Modality</td><td>IB0</td><td>IB32</td><td>IB64</td><td>IB128</td><td>IB1024</td></tr><tr><td>Image</td><td>61.2</td><td>62.3</td><td>63.2</td><td>62.0</td><td>60.7</td></tr><tr><td>VisDoc</td><td>48.6</td><td>51.0</td><td>52.1</td><td>53.9</td><td>54.3</td></tr><tr><td>Video</td><td>34.6</td><td>33.2</td><td>33.5</td><td>34.5</td><td>35.4</td></tr></table>

# 4.3.3 Ablation Study on Model Settings

We investigate the impact of different LoRA ranks (8, 16, 32) on model performance across modalities to understand how the capacity of parameter-efficient tuning affects generalization. As shown in the left part of Figure 2, a LoRA rank of 16 yields the best overall performance across image, video, and visual document tasks. This suggests that a moderate number of tunable parameters is beneficial for handling diverse modalities, while further increasing the rank to 32 does not lead to additional gains.

We also examine performance across training steps to understand how each modality benefits from continued training. As shown in the right part of Figure 2, all three modalities exhibit improved performance with increased training steps. Notably, there is no clear sign of saturation by 5K steps, particularly for VisDoc and Video, suggesting that further gains may be achievable with extended training. We leave a more in-depth exploration of long-horizon training and convergence behavior to future work.

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/581f90aad184740a6ba23816597ee0a44cb4a80e9cdd95dcc6a554bc42a41c6f.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/6a77e14a58c9276fd0284f642e6abb84975d1dc639f3f0c2e3f1cdedea5afa7d.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/2dea31fa57c18dfece1783fdf69b5ecba0b9d96917d87333efa769b0a5f00b1e.jpg]]

![[VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents/images/3f4f9e2e379ab3e61f2302f05832fb04e7c3441ea690bf40eadd8ff2835ac7d5.jpg]]  
Figure 2: The left figure shows performance across LoRA ranks for different modalities, while the right figure illustrates performance trends across training steps.

# 5 Related Works

# 5.1 Multimodal Embedding Benchmarks

Numerous benchmarks have been proposed to evaluate multimodal models, with most early efforts focusing on static image-text pairs. Datasets such as MSCOCO (Lin et al., 2014),

Flickr30K (Plummer et al., 2015), and Conceptual Captions (Sharma et al., 2018) enabled progress in tasks like image captioning and retrieval. Linear probing is also a common evaluation setting that trains a linear layer for image classification (Radford et al., 2021) to investigate the generalization of representation vectors. More recent benchmarks like M-BEIR (Wei et al., 2023) and MMEB (Jiang et al., 2024) introduced multi-task evaluations for multimodal embedding models, covering tasks such as retrieval and QA. However, these benchmarks remain limited to static images and short contexts.

Video-based benchmarks such as MSR-VTT (Xu et al., 2016), QVHighlights (Lei et al., 2021), and ActivityNet Captions (Krishna et al., 2017) target retrieval and captioning tasks, yet lack unified evaluation frameworks for embeddings. Our MMEB-V2 addresses these gaps by integrating instruction-following tasks across videos and structured documents, providing a comprehensive embedding benchmark for diverse visual modalities.

# 5.2 Video Representation Learning

Video representation learning has evolved significantly, progressing from early convolutional approaches to sophisticated transformer-based architectures. Traditional visionlanguage models like CLIP (Radford et al., 2021) and BLIP (Li et al., 2022), while effective for image-text tasks, often struggle to capture the temporal dynamics inherent in video data. To address this, recent models have been developed to better handle the complexities of video understanding. VideoCLIP (Xu et al., 2021), VideoCoCa (Yan et al., 2022) integrates contrastive learning with captioning objectives to enhance video-text representation alignment. InternVideo2 (Wang et al., 2024c) employs a progressive training approach that unifies masked video modeling, cross-modal contrastive learning, and next-token prediction, resulting in superior performance on over 60 video and audio tasks.

Recent models like LLaVE (Lan et al., 2025) and LamRA (Liu et al., 2024b), though trained exclusively on image-text data, have demonstrated the ability to generalize to text-video retrieval tasks in a zero-shot manner. These advancements highlight the ongoing efforts to develop models capable of effectively understanding and representing the complex temporal and semantic information in video data.

# 5.3 Visual Document Representation Learning

Visual document representation learning has become increasingly vital for tasks such as document retrieval, understanding, and retrieval-augmented generation (RAG). Traditional text-based models often struggle to capture the rich visual and structural information present in documents, necessitating approaches that integrate both visual and textual modalities.

One notable advancement is ColPali (Faysse et al., 2024), which leverages vision-language models to enhance document retrieval efficiency by effectively capturing both textual and visual features. In the realm of retrieval-augmented generation, VisRAG (Yu et al., 2024) establishes a vision-based RAG pipeline that directly embeds documents as images using vision-language models, thereby preserving the original document information and outperforming traditional text-based RAG systems. Similarly, ViDoRAG (Wang et al., 2025) introduces a multi-agent framework tailored for complex reasoning across visual documents, employing a dynamic iterative reasoning process to enhance retrieval and generation tasks. Furthermore, benchmarks like MMLongBench-Doc (Ma et al., 2024) have been developed to assess long-context document understanding with visualizations, providing a comprehensive evaluation framework for multimodal models.

# 5.4 Unified Modality Retrieval

Unified modality retrieval methods aim to build models capable of retrieving information across multiple data types—such as text, images, audio, and video—within a single framework. Approaches like GME (Zhang et al., 2024b) and Uni-Retrieval (Jia et al., 2025) leverage multimodal large language models and prompt-tuning to accommodate diverse queries and modalities, achieving strong performance on universal benchmarks. Meanwhile, methods such as UniversalRAG (Yeo et al., 2025) and UniRAG (Sharifymoghaddam et al.,

2025) improve retrieval-augmented generation by dynamically routing queries to the most suitable modality and granularity, enhancing both flexibility and accuracy. However, none of these models are designed to unify image, video, and visual document retrieval within a single framework, as our VLM2Vec-V2 does.

# 6 Conclusion

We introduced MMEB-V2, a comprehensive benchmark for evaluating multimodal embedding models across text, image, video, and visual document modalities. Alongside it, we proposed VLM2Vec-V2, a strong baseline trained via contrastive learning across a diverse range of tasks and modality combinations. Our extensive experiments demonstrate the effectiveness of VLM2Vec-V2 and the diagnostic value of MMEB-V2.

# References

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pp. 5803–5812, 2017.   
Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. Llm2vec: Large language models are secretly powerful text encoders. arXiv preprint arXiv:2404.05961, 2024.   
Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset, 2022. URL https://arxiv.org/abs/1907.06987.   
Boqi Chen, Anuj Khare, Gaurav Kumar, Arjun Akula, and Pradyumna Narayana. Seeing beyond: Enhancing visual question answering with multi-modal retrieval. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pp. 410– 421, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025.coling-industry.35/.   
David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011.   
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.   
Manuel Faysse, Hugues Sibille, Tony Wu, Gautier Viaud, Celine Hudelot, and Pierre ´ Colombo. Colpali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449, 2024.   
Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.   
Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pp. 5267–5275, 2017.   
Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. Scaling deep contrastive learning batch size under memory limited setup. arXiv preprint arXiv:2101.06983, 2021.   
Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne ´ Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The ”something something” video database for learning and evaluating visual common sense, 2017. URL https://arxiv.org/abs/1706.04261.

Yanhao Jia, Xinyi Wu, Hao Li, Qinglin Zhang, Yuxiao Hu, Shuai Zhao, and Wenqi Fan. Uni-retrieval: A multi-style retrieval framework for stem’s education. arXiv preprint arXiv:2502.05863, 2025.   
Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160, 2024.   
Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Densecaptioning events in videos. In Proceedings of the IEEE international conference on computer vision, pp. 706–715, 2017.   
H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. Hmdb: A large video database for human motion recognition. In 2011 International Conference on Computer Vision, pp. 2556–2563, 2011. doi: 10.1109/ICCV.2011.6126543.   
Hilde Kuehne, Ali Arslan, and Thomas Serre. The language of actions: Recovering the syntax and semantics of goal-directed human activities. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pp. 780–787, 2014. doi: 10.1109/CVPR.2014.105.   
Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812, 2025.   
Jie Lei, Tamara L. Berg, and Mohit Bansal. Qvhighlights: Detecting moments and highlights in videos via natural language queries. ArXiv, abs/2107.09609, 2021. URL https://api. semanticscholar.org/CorpusID:236133968.   
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.   
Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22195–22206, 2024.   
Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571, 2024.   
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a.   
Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts. arXiv preprint arXiv:1907.13487, 2019.   
Yikun Liu, Pingan Chen, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. Lamra: Large multimodal model as your advanced retrieval assistant. arXiv preprint arXiv:2412.01720, 2024b.   
Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860, 2021.   
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024.

Quentin Mace, Ant ´ onio Loison, and Manuel Faysse. Vidore benchmark v2: Raising the bar ´ for visual retrieval. arXiv preprint arXiv:2505.17166, 2025.   
Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023.   
Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. https: //github.com/huggingface/peft, 2022.   
Rui Meng, Ye Liu, Shafiq Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. Sfrembedding-2: Advanced text embedding with multi-stage training, 2024. URL https: //huggingface.co/Salesforce/SFR-Embedding-2 R.   
Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2630–2640, 2019.   
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.   
Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pp. 2641–2649, 2015.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Sahel Sharifymoghaddam, Shivani Upadhyay, Wenhu Chen, and Jimmy Lin. Unirag: Universal retrieval augmentation for large vision language models. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 2026–2039, 2025.   
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018.   
Gunnar A Sigurdsson, Gul Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav ¨ Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pp. 510–526. Springer, 2016.   
Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild, 2012. URL https://arxiv.org/abs/1212.0402.   
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instructionfinetuned text embeddings. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 1102–1121, 2023.   
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2024a.   
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024b.

Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. arXiv preprint arXiv:2502.18017, 2025.   
Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4581–4591, 2019.   
Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024c.   
Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. arXiv preprint arXiv:2311.17136, 2023.   
Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021.   
Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021.   
Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.   
Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Videococa: Video-text modeling with zero-shot transfer from contrastive captioners. arXiv preprint arXiv:2212.04979, 2022.   
Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, and Sung Ju Hwang. Universalrag: Retrieval-augmented generation over multiple corpora with diverse modalities and granularities. arXiv preprint arXiv:2504.20734, 2025.   
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594, 2024.   
Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In Proceedings of the European conference on computer vision (ECCV), pp. 471–487, 2018.   
Huaying Yuan, Jian Ni, Yueze Wang, Junjie Zhou, Zhengyang Liang, Zheng Liu, Zhao Cao, Zhicheng Dou, and Ji-Rong Wen. Momentseeker: A comprehensive benchmark and a strong baseline for moment retrieval within long videos. arXiv preprint arXiv:2502.12558, 2025.   
Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024a.   
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855, 2024b.   
Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

# A Appendix

# A.1 Author Contributions

The VLM2Vec-V2 project was a collaborative effort. Overall project leadership and research guidance were provided by Semih Yavuz, Wenhu Chen, Yingbo Zhou, Caiming Xiong, Ran Xu, and Zeyuan Chen. The specific contributions of the core authors are as follows:

• Project and Research Leadership: Rui and Ziyan co-drove the project’s technical direction, spearheaded the overall model development and led the creation of the MMEB-v2 benchmark. Semih managed the project’s progress, while he and Wenhu provided key research guidance throughout the process.

• Codebase and Infrastructure: Rui led the development of the codebase, implementing the v2 refactoring for training and evaluation, and redesigning the data processing infrastructure. Ziyan refactored the evaluation pipeline to integrate the diverse set of tasks. Xinyi contributed to the evaluation for visual document tasks.

• Benchmark and Data Curation: The creation of the MMEB-v2 benchmark was a significant team effort.

– Video Tasks: Contributions to the video benchmarks were made by Ziyan (Video Retrieval), Mingyi (Video Classification), Yuepeng (Moment Retrieval), and Rui (Video QA).   
– Visual Documents: Xinyi curated the majority of the visual document datasets. Ye contributed the ViDoSeek and MMLongBench datasets, and Rui curated ViDoRe-V2. Ziyan developed the corresponding data parsers and evaluation logic.

• Modeling and Experiments: Ye conducted the training experiments for most models. Can ran the overall evaluations and reported the scores. Xinyi conducted the evaluation for visual document tasks. The collection of baseline results was a joint effort by Ziyan, Rui, Mingyi, Xinyi, and Yuepeng.

• Maintenance: Our team is committed to the long-term and active maintenance of the leaderboard and code package. All co-authors contribute to maintaining the code package, and Mingyi is responsible for maintaining the leaderboard.

# A.2 Details of Baseline Models

VLM2Vec (Jiang et al., 2024) converts vision-language models (VLMs) into the embedding models capable of handling diverse tasks. It reformulates all tasks as instruction-following ranking problems. Using contrastive learning and task-specific instructions, VLM2VEC learns to produce fixed-dimensional embeddings aligned across modalities.

ColPali (Faysse et al., 2024) leverages a vision-language model trained to generate highquality multi-vector embeddings from document page images. Combined with a late interaction matching mechanism, it achieves strong performance on visual document retrieval tasks.

LamRA (Liu et al., 2024b) explores the use of large multimodal models (LMMs) for retrieval, unifying diverse retrieval tasks under a single framework without task-specific fine-tuning. It achieves this by employing two-stage training—language-only pretraining followed by multimodal instruction tuning—to enhance retrieval effectiveness.

GME (Zhang et al., 2024b) is a unified multimodal embedding model finetuned from Qwen2- VL. It supports retrieval across single-modal, cross-modal, and fused-modal settings. GME is trained via contrastive learning using a diverse set of multimodal pairs including text, images, and image-text combinations.

# A.3 Details of Benchmark Construction

# A.3.1 Video Retrieval

MSR-VTT (Xu et al., 2016) is a dataset composed of 10K open-domain videos, each video clip ranging from 10 to 32 seconds in length and accompanied by a total of 200K captions. Following JSFusion (Yu et al., 2018), we sampled 1K clip-text pairs to incorporate into our benchmark. The query side contains both the instruction and the video caption, while the candidates consist of all 1K videos.

DiDeMo (Anne Hendricks et al., 2017) consists of 10K videos collected from Flickr, each trimmed to a maximum of 30 seconds. Each video includes approximately 3 to 5 annotated pairs of descriptions and their corresponding distinct moments. Following previous work (Liu et al., 2019; Luo et al., 2021), we concatenate these descriptions and perform “paragraph-to-video” retrieval on this benchmark. The official test split, which contains 1,004 paragraph-video pairs, is used.

MSVD (Chen & Dolan, 2011) contains 80K English descriptions for 1,970 YouTube videos, each ranging from 1 to 62 seconds in length. Each video is annotated with approximately 40 sentences. We use the official test split, which includes 670 videos, and select one sentence per video to construct 670 test cases.

YouCook2 (Zhou et al., 2018) consists of 14K video clips sourced from 2K instructional cooking videos on YouTube. Each video contains multiple actions performed by the chef, accompanied by corresponding textual descriptions and temporal annotations. Each video clip is extracted and annotated with a single sentence. We follow the common practice (Miech et al., 2019) of using the validation split and removing videos that also appear in HowTo100M. Different papers may report slightly varying numbers of test cases, typically ranging from 3.1K to 3.3K. Our benchmark includes 3,179 clip-text pairs from YouCook2.

VATEX (Wang et al., 2019) contains 41,250 video clips sourced from Kinetics-600 dataset and 825K sentence-level descriptions. The public test set originally contained 6K videos. However, since many of them have been removed or set to private and are no longer accessible online, we use only a subset of 4,468 available videos. For each video, we select one description to include in our benchmark.

# A.3.2 Moment Retrieval

QVHighlights (Lei et al., 2021) is a dataset comprising 10K videos collected from YouTube, covering a diverse range of topics. Each video is annotated with high-quality labels for both query-based video moment retrieval and highlight detection. In our embedding benchmark, we adopt the standard practice of ranking candidate clips and evaluating performance using Recall $\bar { \textcircled { a } } 1$ . In contrast, the QVHighlights paper and some other Vision-Language Models like InternVideo2 (Wang et al., 2024c) evaluate models using Recall $@ 0 . 5$ and Recall@0.7 with Intersection over Union (IoU) as a threshold, a metric that is not well-suited for embedding-based approaches.

Charades-STA (Gao et al., 2017), derived from the Charades (Sigurdsson et al., 2016) dataset, includes sentence-level temporal annotations for approximately 10K videos. Unlike its predecessor, Charades, Charades-STA replaces annotated action types with temporal sentences that describe actions. To minimize ambiguity in candidate clips, we created a filtered subset of the Charades-STA test set by applying a condition that selects videos where the relevant segment occupies less than one-third of the total video length.

MomentSeeker (Yuan et al., 2025) is a dataset designed to benchmark multimodal retrievers on long video moment retrieval tasks. Containing 1.8K queries, MomentSeeker consists of 4 subtasks with various query-side modalities. Additionally, MomentSeeker spans a diverse range of topics, including egocentric videos, cartoons, sports, and movies. For each query, we uniformly sampled nine negative clips and included all the ground truth clips as positive examples.

# A.3.3 Video Classification

Kinetics-700-2020 (Carreira et al., 2022) is made up of approximately 648K Youtube video clips, covering a wide range of human actions, around 700 labels in total, such as cooking, driving, and drawing. Each video clip lasts 3 seconds on average. We sampled 1K video answer pairs from the validation set into our benchmark. The candidate texts are the list of all the labels. The raw video data are retrieved from CVD Foundation Github.

Something Something v2 (Goyal et al., 2017) is the updated version of the Something Something v1 dataset. It consists of 220K crowd-source videos focusing on the physical interactions between humans and objects, with an average length of 4.03 seconds and a total of 174 action classes. We randomly sampled 1000 videos-text pairs from the validation split into our benchmark. The candidate texts are the list of all action classes. The raw video data are retrieved from Qualcomm.

HMDB51 (Kuehne et al., 2011) is composed of 6K video clips, including both movies and web videos, with 51 action labels, such as catch, drink, and kick. We sampled 1K frames-text pairs from the test splits into our benchmark. The candidate texts are all labels. The raw video data are retrieved from the official website.

Breakfast (Kuehne et al., 2014) contains around 1.9K crowdsource video clips in the wild, more than 70 hours of total length, which are about preparing for 10 different types of breakfast, such as cereal, milk, pancakes, and fried eggs. There are 6 different camera viewpoints, and we only selected the clips filmed with camera 01, ignoring those filmed by other cameras. We used all the video clips of camera 01, around 433 samples in total. The candidate texts are the 10 types of breakfast. The raw video data are retrieved from the official website.

UCF101 (Soomro et al., 2012) is an open domain video data set consisting of approximately 13K videos with 101 action categories, such as applying makeup, sports and playing instruments. We sampled 1K clip-text pairs from test splits into our benchmarks, and the candidate texts are all the action categories. The raw video data are retrieved from the official website.

# A.3.4 Video QA

While embedding models are not primarily designed for open-ended visual question answering, QA tasks offer a valuable way to assess whether a model can effectively understand visual inputs for different downstream purposes. They also enable fair comparison between embedding-based and generation-based approaches. To this end, we select multi-choice QA benchmarks that span a wide range of task types and are relatively less dependent on knowledge or reasoning abilities. We retain the original dataset configurations to ensure compatibility with prior work and facilitate direct comparison with existing models.

MVBench (Li et al., 2024) is a comprehensive benchmark designed to evaluate multimodal large language models on video understanding, with a particular focus on temporal understanding. It defines 20 video tasks covering a wide spectrum of temporal abilities – from perception to cognition – by transforming static tasks into dynamic, multiple-choice QA formats.

Video-MME (Fu et al., 2024) is a full-spectrum benchmark for evaluating Multi-modal Large Language Models (MLLMs) on video understanding. It consists of 2,700 manually annotated QA pairs based on 900 videos (totaling 254 hours). It ensures broad scenario coverage and captures diverse temporal dynamics by including a variety of video types and durations.

NExT-QA (Xiao et al., 2021) is a video question answering benchmark focused on causal and temporal reasoning in untrimmed daily activity videos. It supports both multiple-choice (what we use) and open-ended QA formats. In our study, we utilize only the multiple-choice portion to evaluate models’ ability to reason about complex action dynamics and object interactions.

EgoSchema (Mangalam et al., 2023) is a diagnostic benchmark for long-form video understanding, constructed from Ego4D and comprising over 5,000 multiple-choice QA pairs spanning more than 250 hours of egocentric video. Each question is grounded in a 3- minute clip and targets long-range temporal reasoning. In our study, we use a subset of 500 questions for which answer annotations are publicly available.

# A.3.5 Visual Document Retrieval

ViDoRe (Faysse et al., 2024; Mace et al. ´ , 2025) is a benchmark designed to evaluate document retrieval systems. The first version (v1) includes 10 subtasks. The dataset originates from two sources: (1) for academic tasks, it repurposes widely used visual question-answering (VQA) benchmarks, treating each question as a query and the corresponding page as the gold document; (2) for practical tasks, publicly accessible PDF documents are collected, and queries relevant to document pages are generated using Claude-3 Sonnet.

To address the saturation of the original ViDoRe benchmark, ViDoRe-v2 introduces more realistic and challenging retrieval tasks, including four new diverse and multilingual datasets.

VisRAG (Yu et al., 2024) serves as the test set for the VisRAG pipeline, which assesses multimodal retrievers in document retrieval. This benchmark consists of six subtasks adapted from VQA datasets, with a filtering process applied to exclude context-dependent queries unsuitable for retrieval.

ViDoSeek (Wang et al., 2025) is a large-scale document collection question-answering dataset originally designed to evaluate retrieval-augmented generation (RAG) performance requiring complex reasoning. We adapt it for retrieval by using questions as queries and reference pages as gold images, with each query linked to relevant images from a collection of approximately 5,000 images. The dataset covers diverse content types, including text, charts, tables, and structured layouts.

MMLongBench-Doc (Ma et al., 2024) is a long-context, multimodal VQA benchmark containing 1,082 expert-annotated questions. Unlike previous VQA datasets, it is built on 135 lengthy PDF documents, averaging 47.5 pages each. To ensure comprehensive evaluation, questions require evidence from multiple sources (text, images, charts, tables, and layout structures) and locations (e.g., specific page numbers). We repurpose this dataset for retrieval, treating questions as queries and evidence pages as gold images, with each query linked to relevant images from a collection of approximately 6,000 images.

# A.4 Detailed Scores

Table 5: Performance comparison of various models on the full MMEB-v2 benchmark, covering 78 tasks across image, video, and visual document modalities. Numbers in parentheses represent the task count for each category.   

<table><tr><td></td><td>ColPali v1.3</td><td>GME-2B</td><td>GME-7B</td><td>LamRA-Qwen2</td><td>LamRA-Qwen2.5</td><td>VLM2Vec-2B</td><td>VLM2Vec-7B</td><td>VLM2Vec-V2.0</td></tr><tr><td>Avg - All (78 tasks)</td><td>44.4</td><td>54.1</td><td>57.8</td><td>40.4</td><td>47.4</td><td>47.0</td><td>52.3</td><td>58.0</td></tr><tr><td>Avg - Image (36 tasks, Hit@1)</td><td>34.9</td><td>51.9</td><td>56.0</td><td>54.1</td><td>52.4</td><td>59.7</td><td>65.5</td><td>64.9</td></tr><tr><td>Avg - Video (18 tasks, Hit@1)</td><td>28.2</td><td>33.6</td><td>38.4</td><td>35.0</td><td>33.6</td><td>28.6</td><td>33.7</td><td>34.6</td></tr><tr><td>Avg - Visdoc (24 tasks, NDCG@5)</td><td>71.0</td><td>72.7</td><td>75.2</td><td>23.9</td><td>50.2</td><td>41.6</td><td>46.4</td><td>65.4</td></tr><tr><td>I-CLS (10)</td><td>40.3</td><td>54.4</td><td>57.7</td><td>59.2</td><td>51.7</td><td>58.7</td><td>62.7</td><td>62.9</td></tr><tr><td>I-QA (10)</td><td>11.5</td><td>29.9</td><td>34.7</td><td>26.5</td><td>34.1</td><td>49.3</td><td>56.9</td><td>56.3</td></tr><tr><td>I-RET (12)</td><td>48.1</td><td>66.9</td><td>71.2</td><td>70.0</td><td>66.9</td><td>65.0</td><td>69.4</td><td>69.5</td></tr><tr><td>I-VG (4)</td><td>40.3</td><td>55.5</td><td>59.3</td><td>62.7</td><td>56.7</td><td>72.9</td><td>82.2</td><td>77.3</td></tr><tr><td>V-CLS (5)</td><td>26.7</td><td>34.9</td><td>37.4</td><td>39.3</td><td>32.9</td><td>33.4</td><td>39.1</td><td>39.3</td></tr><tr><td>V-QA (5)</td><td>37.8</td><td>42.0</td><td>50.4</td><td>42.6</td><td>42.6</td><td>30.5</td><td>30.0</td><td>34.3</td></tr><tr><td>V-RET (5)</td><td>21.6</td><td>25.6</td><td>28.4</td><td>24.3</td><td>23.2</td><td>20.6</td><td>29.0</td><td>28.8</td></tr><tr><td>V-MR (3)</td><td>25.5</td><td>31.1</td><td>37.0</td><td>32.8</td><td>37.2</td><td>30.7</td><td>38.9</td><td>36.8</td></tr><tr><td>VD-Vidore-V1 (10)</td><td>83.6</td><td>86.1</td><td>89.4</td><td>22.0</td><td>56.3</td><td>49.8</td><td>56.9</td><td>75.5</td></tr><tr><td>VD-Vidore-V2 (4)</td><td>52.0</td><td>54.0</td><td>55.6</td><td>11.5</td><td>33.3</td><td>13.5</td><td>9.4</td><td>44.9</td></tr><tr><td>VD-VisRAG (6)</td><td>81.1</td><td>82.5</td><td>85.0</td><td>37.4</td><td>58.2</td><td>51.8</td><td>59.1</td><td>79.4</td></tr><tr><td>VD-OOD (4)</td><td>43.1</td><td>43.1</td><td>44.4</td><td>21.0</td><td>40.1</td><td>33.5</td><td>38.1</td><td>39.4</td></tr><tr><td>ImageNet-1K</td><td>42.4</td><td>58.3</td><td>64.6</td><td>72.3</td><td>58.9</td><td>77.5</td><td>80.1</td><td>80.8</td></tr><tr><td>N24News</td><td>25.5</td><td>50.1</td><td>50.5</td><td>51.3</td><td>29.8</td><td>73.7</td><td>79.7</td><td>72.9</td></tr><tr><td>HatefulMemes</td><td>50.6</td><td>52.5</td><td>53.6</td><td>49.0</td><td>51.3</td><td>58.3</td><td>69.7</td><td>56.3</td></tr><tr><td>VOC2007</td><td>69.8</td><td>75.9</td><td>80.3</td><td>80.1</td><td>78.7</td><td>74.3</td><td>80.7</td><td>85.0</td></tr><tr><td>SUN397</td><td>56.1</td><td>67.3</td><td>69.5</td><td>68.5</td><td>66.5</td><td>73.8</td><td>77.4</td><td>71.0</td></tr><tr><td>Place365</td><td>27.5</td><td>35.8</td><td>39.1</td><td>40.6</td><td>37.4</td><td>35.3</td><td>37.4</td><td>35.9</td></tr><tr><td>ImageNet-A</td><td>14.9</td><td>28.8</td><td>41.2</td><td>47.0</td><td>36.3</td><td>50.9</td><td>58.1</td><td>47.4</td></tr><tr><td>ImageNet-R</td><td>64.6</td><td>78.6</td><td>83.9</td><td>88.5</td><td>77.0</td><td>84.7</td><td>73.9</td><td>89.3</td></tr><tr><td>ObjectNet</td><td>45.6</td><td>70.6</td><td>69.0</td><td>66.4</td><td>59.4</td><td>37.1</td><td>40.1</td><td>65.2</td></tr><tr><td>Country211</td><td>6.0</td><td>26.5</td><td>24.8</td><td>28.3</td><td>21.7</td><td>21.5</td><td>29.8</td><td>25.2</td></tr><tr><td>OK-VQA</td><td>9.4</td><td>29.9</td><td>33.2</td><td>37.8</td><td>39.9</td><td>48.5</td><td>56.8</td><td>51.5</td></tr><tr><td>A-OKVQA</td><td>6.6</td><td>18.6</td><td>21.0</td><td>27.0</td><td>34.1</td><td>39.5</td><td>47.3</td><td>43.6</td></tr><tr><td>DocVQA</td><td>11.3</td><td>29.8</td><td>41.4</td><td>22.3</td><td>37.1</td><td>82.5</td><td>89.7</td><td>90.1</td></tr><tr><td>InfographicsVQA</td><td>5.0</td><td>11.6</td><td>20.3</td><td>16.5</td><td>23.7</td><td>47.7</td><td>60.0</td><td>58.8</td></tr><tr><td>ChartQA</td><td>5.7</td><td>13.4</td><td>17.8</td><td>11.7</td><td>15.0</td><td>42.3</td><td>56.9</td><td>47.4</td></tr><tr><td>Visual7W</td><td>6.1</td><td>16.2</td><td>22.2</td><td>19.6</td><td>24.6</td><td>51.2</td><td>52.7</td><td>52.9</td></tr><tr><td>ScienceQA</td><td>16.3</td><td>27.3</td><td>28.0</td><td>26.3</td><td>31.3</td><td>30.7</td><td>38.5</td><td>38.2</td></tr><tr><td>VizWiz</td><td>27.6</td><td>37.0</td><td>39.0</td><td>32.0</td><td>32.0</td><td>38.6</td><td>39.9</td><td>43.3</td></tr><tr><td>GQA</td><td>8.3</td><td>75.1</td><td>76.9</td><td>38.5</td><td>57.4</td><td>48.3</td><td>55.1</td><td>64.9</td></tr><tr><td>TextVQA</td><td>18.8</td><td>39.7</td><td>46.8</td><td>33.0</td><td>46.1</td><td>63.3</td><td>71.6</td><td>72.2</td></tr><tr><td>VisDial</td><td>41.2</td><td>48.1</td><td>60.8</td><td>61.3</td><td>62.5</td><td>74.3</td><td>81.9</td><td>82.7</td></tr><tr><td>CIRR</td><td>8.2</td><td>44.2</td><td>54.9</td><td>51.7</td><td>44.7</td><td>46.8</td><td>51.1</td><td>57.5</td></tr><tr><td>VisualNews.t2i</td><td>50.1</td><td>74.7</td><td>79.7</td><td>70.4</td><td>70.1</td><td>73.1</td><td>80.5</td><td>74.5</td></tr><tr><td>VisualNews.i2t</td><td>47.6</td><td>78.3</td><td>83.6</td><td>83.9</td><td>74.2</td><td>73.7</td><td>81.2</td><td>78.2</td></tr><tr><td>MSCOCO.t2i</td><td>59.2</td><td>68.1</td><td>71.2</td><td>72.2</td><td>65.7</td><td>73.4</td><td>77.2</td><td>75.3</td></tr><tr><td>MSCOCO.i2t</td><td>49.9</td><td>63.1</td><td>57.7</td><td>73.7</td><td>71.1</td><td>68.5</td><td>73.9</td><td>71.4</td></tr><tr><td>NIGHTS</td><td>65.5</td><td>67.0</td><td>67.6</td><td>65.6</td><td>64.4</td><td>66.3</td><td>67.6</td><td>68.6</td></tr><tr><td>WebQA</td><td>53.8</td><td>88.8</td><td>91.4</td><td>81.0</td><td>85.7</td><td>85.9</td><td>88.3</td><td>90.6</td></tr><tr><td>FashionIQ</td><td>5.9</td><td>32.9</td><td>37.8</td><td>42.0</td><td>33.4</td><td>14.0</td><td>17.1</td><td>19.5</td></tr><tr><td>Wiki-SS-NQ</td><td>80.5</td><td>73.9</td><td>78.2</td><td>69.7</td><td>67.0</td><td>54.2</td><td>62.3</td><td>66.9</td></tr><tr><td>OVEN</td><td>50.0</td><td>72.3</td><td>75.1</td><td>82.0</td><td>84.8</td><td>68.3</td><td>66.5</td><td>64.3</td></tr><tr><td>EDIS</td><td>64.7</td><td>91.8</td><td>96.0</td><td>85.9</td><td>78.7</td><td>81.2</td><td>85.7</td><td>84.1</td></tr><tr><td>MSCOCO</td><td>36.7</td><td>28.6</td><td>31.4</td><td>44.8</td><td>36.0</td><td>66.5</td><td>75.7</td><td>67.1</td></tr><tr><td>RefCOCO</td><td>64.5</td><td>55.9</td><td>60.9</td><td>62.8</td><td>57.1</td><td>80.9</td><td>87.6</td><td>87.1</td></tr><tr><td>RefCOCO-Matching</td><td>3.9</td><td>73.3</td><td>78.4</td><td>75.7</td><td>82.6</td><td>75.7</td><td>84.6</td><td>85.8</td></tr><tr><td>Visual7W-Pointing</td><td>56.1</td><td>64.1</td><td>66.5</td><td>67.3</td><td>51.2</td><td>68.3</td><td>81.0</td><td>69.2</td></tr><tr><td>K700</td><td>23.4</td><td>35.2</td><td>39.7</td><td>42.3</td><td>32.1</td><td>31.4</td><td>35.5</td><td>38.0</td></tr><tr><td>SmithSmithV2</td><td>25.1</td><td>29.9</td><td>30.6</td><td>36.3</td><td>25.3</td><td>30.9</td><td>32.1</td><td>42.8</td></tr><tr><td>HMDBS1</td><td>24.8</td><td>43.4</td><td>47.9</td><td>40.5</td><td>33.8</td><td>33.8</td><td>42.2</td><td>40.9</td></tr><tr><td>UCF101</td><td>49.4</td><td>52.4</td><td>54.7</td><td>60.4</td><td>53.0</td><td>57.5</td><td>61.8</td><td>60.0</td></tr><tr><td>Breakfast</td><td>10.9</td><td>13.6</td><td>14.3</td><td>16.9</td><td>20.1</td><td>13.4</td><td>23.8</td><td>14.8</td></tr><tr><td>MVBench</td><td>33.7</td><td>37.5</td><td>46.6</td><td>37.2</td><td>37.6</td><td>30.5</td><td>28.5</td><td>33.7</td></tr><tr><td>Video-MME</td><td>30.6</td><td>34.3</td><td>39.2</td><td>34.1</td><td>35.1</td><td>26.9</td><td>27.8</td><td>30.7</td></tr><tr><td>NExTQA</td><td>35.2</td><td>39.5</td><td>53.6</td><td>43.7</td><td>44.9</td><td>20.3</td><td>20.3</td><td>20.9</td></tr><tr><td>EgoSchema</td><td>38.4</td><td>40.8</td><td>46.8</td><td>44.8</td><td>47.0</td><td>25.4</td><td>21.8</td><td>34.0</td></tr><tr><td>ActivityNetQA</td><td>51.3</td><td>58.0</td><td>65.6</td><td>53.2</td><td>48.5</td><td>49.6</td><td>51.4</td><td>52.3</td></tr><tr><td>DiDeMo</td><td>22.8</td><td>22.0</td><td>26.4</td><td>24.8</td><td>22.8</td><td>19.4</td><td>29.3</td><td>30.4</td></tr><tr><td>MSRV-VTT</td><td>17.6</td><td>27.3</td><td>31.8</td><td>22.1</td><td>25.0</td><td>25.2</td><td>34.5</td><td>28.3</td></tr><tr><td>MSVD</td><td>45.4</td><td>47.6</td><td>49.7</td><td>46.1</td><td>41.9</td><td>38.2</td><td>46.7</td><td>48.1</td></tr><tr><td>VATEX</td><td>16.7</td><td>23.0</td><td>24.9</td><td>19.1</td><td>18.7</td><td>16.2</td><td>25.5</td><td>26.5</td></tr><tr><td>YouCook2</td><td>5.3</td><td>7.9</td><td>9.1</td><td>9.2</td><td>7.5</td><td>4.1</td><td>9.0</td><td>10.6</td></tr><tr><td>QVHighlight</td><td>19.9</td><td>43.6</td><td>59.5</td><td>53.8</td><td>60.9</td><td>44.2</td><td>57.7</td><td>49.4</td></tr><tr><td>Charades-STA</td><td>29.0</td><td>14.9</td><td>14.0</td><td>10.9</td><td>18.8</td><td>13.6</td><td>19.8</td><td>20.2</td></tr><tr><td>MomentSeeker</td><td>27.6</td><td>34.8</td><td>37.4</td><td>33.8</td><td>31.8</td><td>34.4</td><td>39.3</td><td>40.8</td></tr><tr><td>ViDoRe-arxivqa</td><td>81.7</td><td>82.8</td><td>86.9</td><td>10.8</td><td>53.0</td><td>48.9</td><td>60.2</td><td>80.6</td></tr><tr><td>ViDoRe.docvqa</td><td>56.6</td><td>53.1</td><td>57.5</td><td>19.1</td><td>25.4</td><td>27.0</td><td>34.7</td><td>44.9</td></tr><tr><td>ViDoRe.infovqa</td><td>84.9</td><td>90.2</td><td>91.6</td><td>46.3</td><td>72.3</td><td>67.2</td><td>70.4</td><td>83.7</td></tr><tr><td>ViDoRe.tabfquad</td><td>86.9</td><td>93.3</td><td>94.6</td><td>42.8</td><td>66.1</td><td>62.6</td><td>78.2</td><td>89.2</td></tr><tr><td>ViDoRe.tabtdqa</td><td>70.9</td><td>69.9</td><td>74.1</td><td>11.4</td><td>25.9</td><td>19.8</td><td>27.6</td><td>43.8</td></tr><tr><td>ViDoRe-shiftproject</td><td>75.1</td><td>89.5</td><td>96.8</td><td>12.0</td><td>27.3</td><td>41.8</td><td>38.6</td><td>60.8</td></tr><tr><td>ViDoRe.artificialintelligence</td><td>95.7</td><td>97.5</td><td>99.6</td><td>10.3</td><td>72.0</td><td>55.0</td><td>67.7</td><td>88.5</td></tr><tr><td>ViDoRe-energy</td><td>94.7</td><td>91.9</td><td>95.3</td><td>24.8</td><td>65.2</td><td>59.1</td><td>60.4</td><td>86.5</td></tr><tr><td>ViDoRe-government工作报告</td><td>93.6</td><td>94.6</td><td>98.8</td><td>16.4</td><td>72.2</td><td>57.1</td><td>61.8</td><td>85.0</td></tr><tr><td>ViDoRe.health-industry</td><td>95.9</td><td>98.7</td><td>99.3</td><td>25.9</td><td>83.8</td><td>59.6</td><td>69.9</td><td>92.2</td></tr><tr><td>ViDoRe.esgreports_human_labeled_v2</td><td>51.3</td><td>60.0</td><td>63.4</td><td>7.6</td><td>33.0</td><td>12.6</td><td>6.8</td><td>45.6</td></tr><tr><td>ViDoRe.biomedical_lectures_v2,multilingual</td><td>54.7</td><td>54.0</td><td>49.5</td><td>13.3</td><td>35.9</td><td>7.4</td><td>5.1</td><td>44.3</td></tr><tr><td>ViDoRe_economicsreports_v2,multilingual</td><td>49.0</td><td>50.2</td><td>54.2</td><td>19.1</td><td>31.9</td><td>13.9</td><td>13.9</td><td>43.0</td></tr><tr><td>ViDoRe_eugenysReports_v2,multilingual</td><td>52.9</td><td>50.7</td><td>55.4</td><td>5.9</td><td>32.5</td><td>20.1</td><td>11.9</td><td>46.6</td></tr><tr><td>VisRAG_ArxivQA</td><td>80.9</td><td>82.0</td><td>87.4</td><td>2.0</td><td>37.7</td><td>41.8</td><td>52.6</td><td>76.9</td></tr><tr><td>VisRAG_ChartQA</td><td>78.2</td><td>79.9</td><td>81.9</td><td>41.3</td><td>65.9</td><td>57.9</td><td>70.2</td><td>84.4</td></tr><tr><td>VisRAG_MP-DocVQA</td><td>86.8</td><td>84.4</td><td>89.2</td><td>33.4</td><td>54.5</td><td>43.2</td><td>52.8</td><td>71.8</td></tr><tr><td>VisRAG_SlideVQA</td><td>95.0</td><td>93.4</td><td>94.5</td><td>56.5</td><td>76.5</td><td>74.0</td><td>72.8</td><td>91.5</td></tr><tr><td>VisRAG_InfoVQA</td><td>85.7</td><td>91.4</td><td>93.5</td><td>56.3</td><td>73.3</td><td>70.7</td><td>72.0</td><td>85.7</td></tr><tr><td>VisRAG_PlotQA</td><td>60.3</td><td>64.1</td><td>63.4</td><td>34.6</td><td>41.2</td><td>23.4</td><td>34.4</td><td>66.1</td></tr><tr><td>ViDoSeek-page</td><td>22.2</td><td>21.6</td><td>23.2</td><td>11.3</td><td>23.1</td><td>17.7</td><td>22.3</td><td>21.9</td></tr><tr><td>ViDoSeek-doc</td><td>83.7</td><td>83.6</td><td>83.9</td><td>37.1</td><td>80.3</td><td>74.3</td><td>77.8</td><td>80.2</td></tr><tr><td>MMLongBench-page</td><td>14.2</td><td>15.8</td><td>16.2</td><td>8.0</td><td>13.5</td><td>9.6</td><td>11.8</td><td>11.9</td></tr><tr><td>MMLongBench-doc</td><td>52.5</td><td>51.4</td><td>54.3</td><td>27.6</td><td>43.5</td><td>32.6</td><td>40.5</td><td>43.7</td></tr></table>