---
title: "ObjEmbed: Towards Universal Multimodal Object Embeddings"
source: "https://arxiv.org/html/2602.01753v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Shenghao Fu    Yukun Su    Fengyun Rao    Jing LYU    Xiaohua Xie <sup>∗</sup>    Wei-Shi Zheng <sup>∗</sup>

###### Abstract

Aligning objects with corresponding textual descriptions is a fundamental challenge and a realistic requirement in vision-language understanding. While recent multimodal embedding models excel at global image-text alignment, they often struggle with fine-grained alignment between image regions and specific phrases. In this work, we present ObjEmbed, a novel MLLM embedding model that decomposes the input image into multiple regional embeddings, each corresponding to an individual object, along with global embeddings. It supports a wide range of visual understanding tasks like visual grounding, local image retrieval, and global image retrieval. ObjEmbed enjoys three key properties: (1) Object-Oriented Representation: It captures both semantic and spatial aspects of objects by generating two complementary embeddings for each region: an object embedding for semantic matching and an IoU embedding that predicts localization quality. The final object matching score combines semantic similarity with the predicted IoU, enabling more accurate retrieval. (2) Versatility: It seamlessly handles both region-level and image-level tasks. (3) Efficient Encoding: All objects in an image, along with the full image, are encoded in a single forward pass for high efficiency. Superior performance on 18 diverse benchmarks demonstrates its strong semantic discrimination. Code is available at [https://github.com/WeChatCV/ObjEmbed](https://github.com/WeChatCV/ObjEmbed).

Machine Learning

## 1 Introduction

Multimodal embedding models have emerged as a cornerstone in bridging heterogeneous data modalities, such as vision, language, and audio, into a unified semantic space, enabling rich cross-modal understanding, retrieval, and reasoning. Recent advances in large-scale image-text contrastive learning [^49] [^64] [^60] have led to significant progress in multimodal representation learning, especially in aligning images and corresponding captions. Powered by large multimodal models [^2] [^1], embedding models [^66] [^67] [^23] [^28] can generate task-specific embeddings following user instructions, generate cross-modal embeddings with arbitrary modality combinations, and even perform high-level summary and deep reasoning through chain-of-thought.

![Refer to caption](https://arxiv.org/html/2602.01753v2/pics/compare_results.png)

Figure 1: ObjEmbed achieves balanced and superior performance across a wide span of benchmarks.

However, encoding objects within images and aligning them with text queries are still challenging for recent embedding models. Such capabilities are often critical in real-world applications such as autonomous driving (e.g., distant traffic signs), robotics (e.g., small parts manipulation), and digital content safety moderation. Reliable retrieval and representation of objects demand precise localization and strong semantic discrimination, yet remain under-addressed in current frameworks. While FG-CLIP [^60] [^59] improves regional alignment by combining regional and global contrastive objectives, its object embeddings lack explicit modeling of bounding box quality, limiting their reliability in localization-sensitive tasks. Another line of research, open-vocabulary object detection [^43] [^16] [^15], directly aligning text embeddings with regions of interest, can accurately localize objects and perform open-vocabulary recognition. However, due to limited training data, their generalization ability is largely constrained.

To encode objects with high semantic discrimination and precise localization awareness, we introduce ObjEmbed, an MLLM-based object embedding model that encodes all objects within an image as embeddings. Given an input image, regions of interest (RoIs) are first extracted using an off-the-shelf proposal generator and then encoded as a sequence of tokens. Each object is represented by two special tokens: (1) an object token to capture fine-grained semantic content; and (2) an IoU token to predict the quality of the corresponding bounding box by regressing its IoU score with the ground truth. The object and IoU tokens, along with global image tokens, are processed in parallel by a large language model (LLM) to ensure efficiency. The final-layer hidden states of these tokens serve as the object embeddings, IoU embeddings, and image embeddings, respectively. Similarly, text queries are independently encoded into text embeddings through the same LLM backbone, enabling seamless cross-modal alignment. With this novel architecture, ObjEmbed produces object-centric representations that jointly encode semantic meaning and localization confidence, enabling accurate recognition, precise localization, and robust cross-modal retrieval.

Equipped with this object-centric design, ObjEmbed supports a wide range of downstream applications in a unified framework: (1) Object detection and referring expression comprehension: Class names or natural language expressions are encoded as text embeddings and matched against all object embeddings in the image. The final object matching score combines both semantic similarity (between object and text embeddings) and predicted localization quality (from the IoU embedding), computed as their product, effectively balancing semantic relevance and spatial accuracy. (2) Local image retrieval: When the query describes only a specific region or object, we compute the image-level relevance as the maximum matching score across all detected objects. This strategy enables fine-grained, part-aware retrieval even when the target occupies a small portion of the image. (3) Global image retrieval: Since ObjEmbed also retains global image embeddings, they can be directly used for standard image-text retrieval tasks, ensuring compatibility with conventional benchmarks and applications.

After training on 1.3M samples, ObjEmbed demonstrates strong performance across a wide range of vision tasks, summarized in Figure 1. On object detection, it achieves 53.0% mAP on the COCO dataset, a highly competitive result compared to specialist models. For referring expression comprehension, ObjEmbed attains an average accuracy of 89.5 on RefCOCO/+/g, reflecting its robust ability to align language with visual objects. Most notably, ObjEmbed excels in local image retrieval, which requires fine-grained cross-image comparisons. In this setting, ObjEmbed outperforms existing global image embedding models by around 20 points on four standard benchmarks. It also achieves competitive performance on global image retrieval tasks, despite using a small-scale training set. We hope that ObjEmbed can serve as a strong and general-purpose baseline for future research in object representation learning.

## 2 Related Work

![Refer to caption](https://arxiv.org/html/2602.01753v2/x1.png)

Figure 2: The architecture of ObjEmbed. ObjEmbed is a single-tower model built upon a large multimodal language model, enhanced with an object projector and five special tokens ( ⟨ \\langle object ⟩ \\rangle, iou global local\_text, and global\_text ) whose hidden states from the last layer are used as embeddings. ObjEmbed encodes all object embeddings, IoU embeddings, and the global image embeddings in a single forward pass. The visual embeddings and the text embeddings share the same LLM encoder. For detected objects, object embeddings are initialized from RoI features. And the final matching score is computed as the product of the predicted IoU score (predicted from IoU embeddings) and the classification score (predicted from object embeddings and local text embeddings).

### 2.1 Multimodal Embedding Models

Contrastive language-image pre-training [^49], which aims to align matched image-text embeddings while pushing away others, is an effective and scalable way to learn transferable image representations. Other improvements, including sigmoid loss [^64], well-curated data [^61] [^9], hard negative samples [^56], and multi-task learning [^52], further improving its effectiveness. The development of large multimodal models further equips embedding models with the ability to follow instructions, cross-modality combinations, and deep understanding and reasoning [^67] [^23] [^33] [^28]. However, representing object features and assessing their localization quality are still challenging for global image embedding models. FG-CLIP [^60] [^59] tries to mitigate the problem by introducing regional contrastive learning. It cannot assess localization quality, thus it can only tackle the classification problem. CLARE [^19] is the first method tackling the object retrieval task by training a traditional detector with contrastive language-instance alignment. However, due to limited data and model capacity, the generalization ability is constrained.

### 2.2 Open-Vocabulary Object Detection

Open-vocabulary object detection aims to detect arbitrary objects described by text queries. Thus, detectors should encode objects into embeddings and align them with text embeddings. To align with the text space, some methods [^15] [^17] [^57] distill the features of CLIP [^49], some methods [^13] [^14] [^12] integrate the CLIP model as a module or backbone, while others [^43] [^16] use deep fusion layers for cross-modal alignment. Although open-vocabulary object detectors excel at object localization, detectors aligning with CLIP cannot truly inherit open-vocabulary capacity, while deep-fusion methods cannot produce query-agnostic object embeddings and the training data is hard to scale up. These limitations motivate us to develop a novel MLLM-based object embedding model with localization awareness and robust transferability.

## 3 Method

### 3.1 Model Architecture

In this work, we aim to build a novel MLLM-based object embedding model, featuring three key properties: (1) Object-Oriented Representation: it captures both semantic and spatial aspects of objects and assigns higher matching scores to objects with tighter and more accurate boxes; (2) Versatility: the model has the ability to tackle both object-level and image-level tasks; (3) Efficient Encoding: the model encodes all objects in an image in a single forward pass. To achieve the goal, we finetune a standard large multimodal model, Qwen3-VL-instruct [^1], and introduce five special tokens whose hidden states from the last layer are used as embeddings: (1) $\langle$ object $\rangle$: the object embedding to represent semantic details; (2) $\langle$ iou $\rangle$: the IoU embedding to assess the box quality of each object; (3) $\langle$ global $\rangle$: the global image embedding representing the full image; (4) $\langle$ local\_text $\rangle$: the text embedding for matching objects; and (5) $\langle$ global\_text $\rangle$: the text embedding for matching images. The overall architecture is shown in Figure 2 and the different usages of each token are detailed as follows:

Representing objects as a sequence of embeddings. Following WeDetect-Ref [^13], we first use a universal proposal generator, WeDetect-Uni [^13], to generate top-N proposals (100 proposals in this work) for each image. Each RoI feature is extracted via RoIAlign and then compressed into a single token via an object projector. These object features will replace the $\langle$ object $\rangle$ tokens and are organized sequentially before sending to the large language model. Each object will be separated by prompts “Object i: $\langle$ object $\rangle$ ” to ensure distinctiveness. By representing objects as a sequence of embeddings, the model can encode all objects simultaneously with high efficiency.

Assessing box quality via IoU embeddings. To equip the model with location awareness, we explicitly model object location by predicting an IoU score for each detected object. We empirically find that using a single token to jointly learn location and classification leads to optimization conflicts. To mitigate this, we introduce a dedicated special token $\langle$ iou $\rangle$ to represent the quality of each bounding box. This token immediately follows the corresponding object token, forming the structured sequence: “Object i: $\langle$ object $\rangle$ $\langle$ iou $\rangle$.” The final IoU score is computed by applying a linear head to the IoU embedding and then multiplying it by the classification score to produce the final object matching score.

Integrating global image embeddings with object embeddings. We also introduce a global image token $\langle$ global $\rangle$ to encode full image content. As demonstrated in the next subsection, we will use both a long text caption and a short text caption as labels to learn global image embeddings. We use two same but separate global tokens for different kinds of captions. The global image embeddings and object embeddings will be encoded in a single forward pass and the final template is:

$\langle|$ vision\_start $|\rangle$ IMAGE $\langle|$ vision\_end $|\rangle$ Task Instruction The coarse global image is $\langle$ global $\rangle$. Object 0: $\langle$ object $\rangle$ $\langle$ iou $\rangle$. $\cdots$ Object N: $\langle$ object $\rangle$ $\langle$ iou $\rangle$. The detailed global image is $\langle$ global $\rangle$.

where IMAGE is the full image. Task instructions are used to separate different tasks, like object detection and REC, and are detailed in Appendix A.

Representing text queries as embeddings. Finally, we introduce a local text token $\langle$ local\_text $\rangle$ and a global text token $\langle$ global\_text $\rangle$ to represent text queries. The local text token is used to match object embeddings while the global text token is used to match global image embeddings. The encoding templates are “Find an object that matches the given caption. CAPTION $\langle$ local\_text $\rangle$ ” and “Find an image that matches the given caption. CAPTION $\langle$ global\_text $\rangle$ ”. We use the same model to encode text and visual embeddings.

Efficiency Analysis. With the template described above, all objects in an image, as well as the full image, are encoded in a single forward pass without the need for time-consuming autoregressive token prediction. Each object consumes only 8 tokens. When the full image is encoded into 1000 tokens, the total sequence length remains under 2000, requiring minimal GPU memory and enabling efficient acceleration with FlashAttention-2 [^10].

### 3.2 Training Objective

For each image, we annotate a long image caption $C_{long}$, a short image caption $C_{short}$, and a set of objects $\{O_{i}\}_{i=1}^{M}$, where each object $O_{i}$ is associated with an object description $C_{obj}^{i}$ (class names or REC-like descriptions) and a bounding box $B^{i}$. The overall training objective comprises three components: region-level contrastive learning, image-level contrastive learning, and IoU regression, detailed as follows:

Region-level contrastive learning. Unlike traditional contrastive language-image pre-training, where image-caption pairs are one-to-one matched, an object description may correspond to multiple instances, while some proposals remain unmatched due to missing annotations or low-quality bounding boxes. To handle this partial and many-to-one matching, we employ sigmoid focal loss [^36] for supervision, a choice commonly adopted in object detection. Specifically, for each object description $C_{\text{obj}}^{i}$, we treat each region proposal $p_{j}$ as a positive sample if $\text{IoU}(p_{j},B^{i})>0.5$, and negative otherwise. We compute the similarity between the proposal object embedding $e_{p}^{j}$ and the local text embedding $e_{t}^{i}$ of $C_{\text{obj}}^{i}$ as:

$$
s_{ij}=\frac{e_{p}^{j}\cdot e_{t}^{i}}{\|e_{p}^{j}\|\|e_{t}^{i}\|},
$$

Then, the similarities are optimized via sigmoid focal loss:

$$
\mathcal{L}_{\text{region}}=\sum_{i=1}^{M}\sum_{j=1}^{N}\mathcal{L}_{\text{focal}}\left(s_{ij},y_{ij}\right),
$$

where $y_{ij}=1$ if $\text{IoU}(p_{j},B^{i})>0.5$, and 0 otherwise. $M$ is the number of annotations and $N$ is the number of proposals.

Image-level contrastive learning. In line with region-level contrastive learning, we also use sigmoid focal loss [^64] for image-level supervision $\mathcal{L}_{\text{image}}$. Specifically, the first global image embedding is only supervised by short captions while the second global image embedding is only supervised by long captions. And we will collect captions from other GPUs to enlarge the negative samples.

IoU regression. The IoU loss $\mathcal{L}_{\text{iou}}$ is also formulated as sigmoid focal loss with ground truth IoUs $u_{j}^{*}$ between proposals and corresponding boxes as labels:

$$
\mathcal{L}_{\text{iou}}=-\sum_{j=1}^{N^{+}}\mathcal{L}_{\text{focal}}\left(\hat{u}_{j},u_{j}^{*}\right),
$$

where $\hat{u}_{j}$ is predicted IoU scores. Since not all objects in an image are annotated, $\mathcal{L}_{\text{iou}}$ is only applied to $N^{+}$ positive proposals.

The total training objective is a weighted combination:

$$
\mathcal{L}_{\text{total}}=\lambda_{1}\mathcal{L}_{\text{region}}+\lambda_{2}\mathcal{L}_{\text{image}}+\lambda_{3}\mathcal{L}_{\text{iou}},
$$

where $\lambda_{1}$, $\lambda_{2}$, and $\lambda_{3}$ are hyperparameters.

Table 1: The overview of ObjEmbed training dataset.

<table><tbody><tr><td>Type</td><td>#Sample</td><td>Dataset</td></tr><tr><td rowspan="2">DET</td><td rowspan="2">380k</td><td>COCO (111k) <sup><a href="#fn:37">37</a></sup>, LVIS (94k) <sup><a href="#fn:18">18</a></sup></td></tr><tr><td>V3Det (175k) <sup><a href="#fn:54">54</a></sup></td></tr><tr><td rowspan="5">REC</td><td rowspan="5">909k</td><td>RefCOCO/+/g (28k) <sup><a href="#fn:24">24</a></sup> <sup><a href="#fn:63">63</a></sup> <sup><a href="#fn:44">44</a></sup></td></tr><tr><td>FG-OVD (175k) <sup><a href="#fn:3">3</a></sup>, HumanRef (43k) <sup><a href="#fn:22">22</a></sup></td></tr><tr><td>grefcoco (15k) <sup><a href="#fn:38">38</a></sup>, Ref-L4 (9k) <sup><a href="#fn:5">5</a></sup></td></tr><tr><td>DAM (77k) <sup><a href="#fn:35">35</a></sup>, FineCops-Ref (29k) <sup><a href="#fn:40">40</a></sup></td></tr><tr><td>REIRCOCO (25k) <sup><a href="#fn:19">19</a></sup>, self-collected data (508k)</td></tr><tr><td>SUM</td><td>1289k</td><td></td></tr></tbody></table>

Table 2: Object detection results. Specialist detectors are typically designed for target tasks and trained on the target datasets.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="4">COCO</td><td>COCO-O</td><td>ODinW13</td><td colspan="4">FG-OVD</td><td colspan="3">D3</td></tr><tr><td>AP</td><td>AP <sub>s</sub></td><td>AP <sub>m</sub></td><td>AP <sub>l</sub></td><td>AP</td><td>AP</td><td>hard</td><td>medium</td><td>easy</td><td>trivial</td><td>full</td><td>Pres</td><td>Abs</td></tr><tr><td>Specialist Detectors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DINO-R50 <sup><a href="#fn:65">65</a></sup></td><td>51.2</td><td>35.0</td><td>54.3</td><td>65.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GUIDED <sup><a href="#fn:30">30</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>57.5</td><td>69.5</td><td>73.3</td><td>72.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Weak-to-Strong <sup><a href="#fn:48">48</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>30.8</td><td>31.0</td><td>30.4</td></tr><tr><td>Open-vocabulary Detectors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLIP-T <sup><a href="#fn:32">32</a></sup></td><td>46.1</td><td>-</td><td>-</td><td>-</td><td>29.0</td><td>46.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>19.1</td><td>18.3</td><td>21.5</td></tr><tr><td>OWLv2 (L/14) <sup><a href="#fn:47">47</a></sup></td><td>37.9</td><td>24.9</td><td>41.2</td><td>53.7</td><td>42.7</td><td>50.1</td><td>25.4</td><td>41.2</td><td>42.8</td><td>63.2</td><td>22.8</td><td>22.1</td><td>24.7</td></tr><tr><td>Grounding-DINO-T <sup><a href="#fn:43">43</a></sup></td><td>47.9</td><td>33.4</td><td>51.2</td><td>62.2</td><td>37.6</td><td>51.4</td><td>17.0</td><td>28.4</td><td>31.0</td><td>62.5</td><td>20.7</td><td>20.1</td><td>22.5</td></tr><tr><td>LLMDet-T <sup><a href="#fn:16">16</a></sup></td><td>54.9</td><td>40.1</td><td>58.3</td><td>68.6</td><td>36.1</td><td>52.1</td><td>15.0</td><td>26.2</td><td>23.8</td><td>55.4</td><td>17.2</td><td>17.1</td><td>17.6</td></tr><tr><td>WeDetect-B <sup><a href="#fn:13">13</a></sup></td><td>52.1</td><td>34.8</td><td>57.1</td><td>69.2</td><td>44.1</td><td>53.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VLM-FO1-3B <sup><a href="#fn:42">42</a></sup></td><td>44.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>44.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LMM-Det-7B <sup><a href="#fn:31">31</a></sup></td><td>47.5</td><td>34.7</td><td>51.8</td><td>60.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>WeDetect-Ref-2B <sup><a href="#fn:13">13</a></sup></td><td>49.9</td><td>34.0</td><td>58.0</td><td>68.9</td><td>55.8</td><td>48.2</td><td>28.7</td><td>42.5</td><td>48.1</td><td>65.6</td><td>41.8</td><td>43.9</td><td>35.4</td></tr><tr><td>WeDetect-Ref-4B <sup><a href="#fn:13">13</a></sup></td><td>50.0</td><td>34.7</td><td>57.6</td><td>69.2</td><td>56.0</td><td>47.3</td><td>31.1</td><td>45.7</td><td>50.1</td><td>68.4</td><td>42.0</td><td>44.0</td><td>35.8</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-2B <sup><a href="#fn:1">1</a></sup></td><td>16.9</td><td>6.2</td><td>20.7</td><td>36.4</td><td>29.4</td><td>43.4</td><td>59.3</td><td>58.9</td><td>53.8</td><td>55.9</td><td>16.0</td><td>17.0</td><td>11.2</td></tr><tr><td>Qwen3-VL-4B <sup><a href="#fn:1">1</a></sup></td><td>29.1</td><td>12.9</td><td>33.9</td><td>54.2</td><td>44.4</td><td>48.2</td><td>62.0</td><td>62.6</td><td>64.0</td><td>45.7</td><td>24.3</td><td>25.8</td><td>19.7</td></tr><tr><td>ObjEmbed-2B</td><td>52.9</td><td>35.8</td><td>59.7</td><td>72.5</td><td>59.1</td><td>49.8</td><td>65.7</td><td>74.3</td><td>77.3</td><td>76.2</td><td>39.4</td><td>41.1</td><td>34.2</td></tr><tr><td>ObjEmbed-4B</td><td>53.0</td><td>35.6</td><td>59.6</td><td>72.2</td><td>58.0</td><td>50.8</td><td>66.1</td><td>74.1</td><td>77.2</td><td>76.0</td><td>40.2</td><td>42.2</td><td>34.4</td></tr></tbody></table>

### 3.3 Training Data Construction

To reduce annotation costs, we aggregate existing open-source object detection and referring expression comprehension datasets that provide region-level annotations. Due to limited data coverage, we further curate 300k images from SA-1B [^25] and 200k images self-crawled from licensed websites. These images are first annotated with class-agnostic bounding boxes using WeDetect-Uni [^13], followed by generating unique instance-level descriptions for each object using Qwen3-VL-235B [^1]. We find that the instance-level descriptions should be as unique as possible to minimize false-negative conflicts and can not include subjective content. Finally, each image is assigned both a short and a long caption, also generated by Qwen3-VL-235B. The prompts used for annotation are provided in Appendix B.

The overall training data are summarized in Table 1, comprising 1.3M images and 8.1M bounding boxes.

## 4 Experiment

### 4.1 Implementation Details

ObjEmbed is finetuned from Qwen3-VL-Instruct [^1] by first initializing the object projector following WeDetect-Ref [^13] and then training under the objective described in Equation 4. The model is trained using 16 GPUs, with a batch size of 2 images per GPU, and an initial learning rate of $2e^{-5}$. All parameters are updated during training except those in the frozen vision encoder. Training proceeds for two epochs. The loss weights $\lambda_{1}$, $\lambda_{2}$, and $\lambda_{3}$ are set to 1.0, 1.0 and 0.25. Input images are resized such that the number of visual tokens ranges from 900 to 1200, corresponding to 900\*32\*32 to 1200\*32\*32 pixels, ensuring adaptive computation based on image content.

### 4.2 Comparisons on Fine-Grained Region-Level Tasks

Table 3: Evaluation results on referring expression comprehension datasets. The evaluation metric is the Top-1 accuracy.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td><td>Avg.</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td><td></td></tr><tr><td>Grounding-DINO-L <sup><a href="#fn:43">43</a></sup></td><td>90.6</td><td>93.2</td><td>88.2</td><td>82.8</td><td>89.0</td><td>75.9</td><td>86.1</td><td>87.0</td><td>86.6</td></tr><tr><td>CLARE-L <sup><a href="#fn:19">19</a></sup></td><td>91.4</td><td>92.0</td><td>90.0</td><td>80.1</td><td>83.3</td><td>74.4</td><td>86.3</td><td>86.7</td><td>85.5</td></tr><tr><td>Qwen2.5-VL 3B <sup><a href="#fn:2">2</a></sup></td><td>89.1</td><td>91.7</td><td>84.0</td><td>82.4</td><td>88.0</td><td>74.1</td><td>85.2</td><td>85.7</td><td>85.0</td></tr><tr><td>Qwen2.5-VL 7B <sup><a href="#fn:2">2</a></sup></td><td>90.0</td><td>92.5</td><td>85.4</td><td>84.2</td><td>89.1</td><td>76.9</td><td>87.2</td><td>87.2</td><td>86.6</td></tr><tr><td>InternVL2.5-8B <sup><a href="#fn:8">8</a></sup></td><td>90.3</td><td>94.5</td><td>85.9</td><td>85.2</td><td>91.5</td><td>78.8</td><td>86.7</td><td>87.6</td><td>87.6</td></tr><tr><td>InternVL3.5-38B <sup><a href="#fn:55">55</a></sup></td><td>90.3</td><td>91.8</td><td>89.0</td><td>87.5</td><td>90.0</td><td>84.7</td><td>89.7</td><td>89.9</td><td>89.1</td></tr><tr><td>Octopus 7B <sup><a href="#fn:68">68</a></sup></td><td>89.0</td><td>92.6</td><td>83.4</td><td>83.6</td><td>89.4</td><td>76.0</td><td>84.3</td><td>86.3</td><td>85.6</td></tr><tr><td>VLM-R1 3B <sup><a href="#fn:50">50</a></sup></td><td>90.1</td><td>92.3</td><td>85.2</td><td>84.2</td><td>89.4</td><td>76.8</td><td>85.6</td><td>86.8</td><td>86.3</td></tr><tr><td>Rex-Omni 3B <sup><a href="#fn:20">20</a></sup></td><td>86.6</td><td>89.5</td><td>82.8</td><td>79.6</td><td>84.8</td><td>71.4</td><td>85.3</td><td>86.2</td><td>83.3</td></tr><tr><td>ChatRex 7B <sup><a href="#fn:21">21</a></sup></td><td>91.0</td><td>94.1</td><td>87.0</td><td>89.8</td><td>91.9</td><td>79.3</td><td>89.8</td><td>90.0</td><td>89.1</td></tr><tr><td>VLM-FO1 3B <sup><a href="#fn:41">41</a></sup></td><td>91.1</td><td>93.7</td><td>87.6</td><td>86.4</td><td>91.9</td><td>80.6</td><td>88.9</td><td>88.3</td><td>88.6</td></tr><tr><td>RexSeek 7B <sup><a href="#fn:22">22</a></sup></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>84.0</td><td>84.4</td><td>-</td></tr><tr><td>Qwen3-VL 2B <sup><a href="#fn:1">1</a></sup></td><td>88.2</td><td>91.0</td><td>83.1</td><td>78.6</td><td>85.2</td><td>70.4</td><td>84.7</td><td>85.0</td><td>83.3</td></tr><tr><td>Qwen3-VL 4B <sup><a href="#fn:1">1</a></sup></td><td>90.7</td><td>92.2</td><td>86.7</td><td>82.9</td><td>89.4</td><td>75.6</td><td>87.3</td><td>87.7</td><td>86.6</td></tr><tr><td>ObjEmbed-2B</td><td>91.7</td><td>93.5</td><td>88.3</td><td>83.4</td><td>90.1</td><td>76.6</td><td>88.6</td><td>88.6</td><td>87.6</td></tr><tr><td>ObjEmbed-4B</td><td>92.5</td><td>94.3</td><td>90.2</td><td>86.1</td><td>91.4</td><td>80.6</td><td>89.9</td><td>90.7</td><td>89.5</td></tr></tbody></table>

Table 4: Evaluation results on local image retrieval datasets. The evaluation metric for SORCE-1K and REIRCOCO is Recall@1 while the evaluation metric for ILIAS is mAP@50, a metric evaluating the top 50 predictions.

<table><tbody><tr><td rowspan="2">Method</td><td>SORCE-1K</td><td>REIRCOCO</td><td colspan="2">ILIAS</td><td>Avg.</td></tr><tr><td>T2I</td><td>T2I</td><td>T2I</td><td>I2I</td><td></td></tr><tr><td>CLIP ViT-L/14</td><td>32.6</td><td>16.0</td><td>44.2</td><td>39.9</td><td>33.2</td></tr><tr><td>SigLIP2 ViT-So/16</td><td>34.2</td><td>22.9</td><td>45.2</td><td>55.4</td><td>39.4</td></tr><tr><td>MetaCLIP2 ViT-H/14</td><td>37.5</td><td>19.2</td><td>49.6</td><td>42.6</td><td>37.2</td></tr><tr><td>FG-CLIP2 ViT-So/16</td><td>46.9</td><td>27.3</td><td>57.9</td><td>64.1</td><td>49.1</td></tr><tr><td>GME-2B</td><td>28.3</td><td>19.8</td><td>35.7</td><td>44.9</td><td>32.2</td></tr><tr><td>GME-7B</td><td>30.6</td><td>24.6</td><td>33.5</td><td>40.4</td><td>32.3</td></tr><tr><td>VLM2Vec-2B</td><td>26.3</td><td>20.1</td><td>29.8</td><td>39.7</td><td>29.0</td></tr><tr><td>VLM2Vec-V2-2B</td><td>24.8</td><td>24.5</td><td>32.0</td><td>36.4</td><td>29.4</td></tr><tr><td>QQMM-embed-v2-7B</td><td>47.7</td><td>32.7</td><td>44.1</td><td>48.0</td><td>43.1</td></tr><tr><td>RzenEmbed-7B</td><td>36.0</td><td>30.4</td><td>39.2</td><td>42.0</td><td>36.9</td></tr><tr><td>UME-R1-2B</td><td>36.1</td><td>23.3</td><td>31.6</td><td>38.9</td><td>32.5</td></tr><tr><td>UME-R1-7B</td><td>41.3</td><td>28.2</td><td>37.6</td><td>41.7</td><td>37.2</td></tr><tr><td>Qwen3-VL-Embedding-2B</td><td>45.4</td><td>28.0</td><td>47.3</td><td>57.9</td><td>44.7</td></tr><tr><td>Qwen3-VL-Embedding-8B</td><td>49.1</td><td>32.6</td><td>51.2</td><td>59.2</td><td>48.0</td></tr><tr><td>FG-CLIP2 ViT-So/16 (RoIAlign)</td><td>28.4</td><td>16.0</td><td>53.9</td><td>72.5</td><td>42.7</td></tr><tr><td>ObjEmbed-2B</td><td>67.3</td><td>37.5</td><td>76.5</td><td>84.0</td><td>66.3</td></tr><tr><td>ObjEmbed-4B</td><td>71.7</td><td>39.3</td><td>77.6</td><td>85.3</td><td>68.5</td></tr></tbody></table>

Table 5: Evaluation results on global image retrieval datasets. The evaluation metrics are Recall@1.

<table><tbody><tr><td rowspan="3">Method</td><td colspan="4">Long Image Captions</td><td colspan="4">Short Image Captions</td><td colspan="4">Multilingual Image Captions</td><td>Avg.</td></tr><tr><td colspan="2">ShareGPT4V</td><td colspan="2">DCI</td><td colspan="2">COCO</td><td colspan="2">Flickr30K</td><td colspan="2">COCO-CN</td><td colspan="2">Flickr30K-CN</td><td></td></tr><tr><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td></td></tr><tr><td>CLIP ViT-L/14 <sup><a href="#fn:49">49</a></sup></td><td>86.5</td><td>83.6</td><td>37.2</td><td>36.4</td><td>58.0</td><td>37.1</td><td>87.4</td><td>67.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>EVA-CLIP ViT-L/14 <sup><a href="#fn:51">51</a></sup></td><td>91.5</td><td>89.4</td><td>47.2</td><td>47.8</td><td>64.2</td><td>47.9</td><td>89.2</td><td>77.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SigLIP2 ViT-So/16 <sup><a href="#fn:52">52</a></sup></td><td>78.6</td><td>79.5</td><td>46.0</td><td>47.1</td><td>71.0</td><td>55.8</td><td>94.1</td><td>82.5</td><td>72.0</td><td>50.7</td><td>78.4</td><td>51.7</td><td>67.3</td></tr><tr><td>MetaCLIP2 ViT-H/14 <sup><a href="#fn:9">9</a></sup></td><td>93.9</td><td>89.2</td><td>53.0</td><td>50.2</td><td>66.8</td><td>47.7</td><td>91.9</td><td>77.0</td><td>80.1</td><td>63.1</td><td>89.3</td><td>72.2</td><td>72.9</td></tr><tr><td>FG-CLIP2 ViT-So/16 <sup><a href="#fn:59">59</a></sup></td><td>97.5</td><td>96.7</td><td>70.6</td><td>72.1</td><td>74.6</td><td>56.7</td><td>95.9</td><td>85.0</td><td>83.2</td><td>68.1</td><td>91.5</td><td>77.2</td><td>80.8</td></tr><tr><td>GME-2B <sup><a href="#fn:67">67</a></sup></td><td>92.8</td><td>91.7</td><td>52.9</td><td>58.8</td><td>67.3</td><td>51.7</td><td>88.0</td><td>75.1</td><td>79.7</td><td>66.7</td><td>85.5</td><td>71.1</td><td>73.4</td></tr><tr><td>GME-7B <sup><a href="#fn:67">67</a></sup></td><td>89.3</td><td>92.2</td><td>59.8</td><td>64.6</td><td>67.8</td><td>55.6</td><td>91.1</td><td>81.1</td><td>81.7</td><td>71.8</td><td>91.8</td><td>79.1</td><td>77.2</td></tr><tr><td>VLM2Vec-2B <sup><a href="#fn:23">23</a></sup></td><td>77.7</td><td>74.3</td><td>33.2</td><td>44.2</td><td>53.5</td><td>39.2</td><td>83.0</td><td>68.5</td><td>67.9</td><td>52.6</td><td>76.1</td><td>56.9</td><td>60.6</td></tr><tr><td>VLM2Vec-V2-2B <sup><a href="#fn:46">46</a></sup></td><td>92.1</td><td>92.4</td><td>53.1</td><td>65.3</td><td>62.6</td><td>50.3</td><td>89.2</td><td>80.3</td><td>74.4</td><td>60.3</td><td>84.0</td><td>67.1</td><td>72.6</td></tr><tr><td>UME-R1-2B <sup><a href="#fn:28">28</a></sup></td><td>92.9</td><td>93.9</td><td>59.8</td><td>67.9</td><td>71.4</td><td>54.4</td><td>91.6</td><td>79.1</td><td>83.6</td><td>70.1</td><td>87.9</td><td>75.0</td><td>77.3</td></tr><tr><td>Qwen3-VL-Embedding-2B <sup><a href="#fn:33">33</a></sup></td><td>97.8</td><td>96.7</td><td>77.9</td><td>79.7</td><td>69.6</td><td>55.2</td><td>92.9</td><td>81.9</td><td>83.0</td><td>72.2</td><td>91.9</td><td>78.6</td><td>81.5</td></tr><tr><td>ObjEmbed-2B</td><td>97.1</td><td>97.3</td><td>74.5</td><td>74.6</td><td>75.2</td><td>51.0</td><td>94.2</td><td>80.0</td><td>86.0</td><td>66.7</td><td>94.0</td><td>76.1</td><td>80.6</td></tr><tr><td>ObjEmbed-4B</td><td>97.5</td><td>97.7</td><td>77.2</td><td>76.9</td><td>75.7</td><td>52.2</td><td>94.2</td><td>80.4</td><td>87.4</td><td>68.8</td><td>94.7</td><td>77.3</td><td>81.7</td></tr></tbody></table>

Results on object detection benchmarks. Object detection aims to simultaneously localize and classify all target objects within an image. The standard evaluation metric is mAP, computed over IoU thresholds ranging from 0.50 to 0.95, which places strong emphasis on localization accuracy. We evaluate on five benchmarks: COCO [^37] for general object detection, COCO-O [^45] for out-of-distribution detection, ODinW13 [^29] for detection in the wild, FG-OVD [^3] for fine-grained attribute-aware recognition, and D3 [^58] for language-based object detection. As shown in Table 2, traditional detectors excel at precise localization and handling multi-object scenes, but their semantic understanding is limited to fixed class vocabularies with short category names. Consequently, their performance on FG-OVD and D3 is low. In contrast, multimodal large language models (MLLMs) demonstrate superior language comprehension and can interpret complex textual descriptions, yet suffer from poor localization accuracy due to coarse spatial reasoning, resulting in low performance on COCO and ODinW13. Equipped with the dual-token design, ObjEmbed achieves a favorable balance between high semantic discriminability and precise location assessment. It understands rich language inputs, prioritizes well-localized predictions with higher scores, and is robust to domain variances, leading to consistently strong performance across all five benchmarks.

Results on referring expression comprehension benchmarks. Referring expression comprehension aims to localize the unique object in an image that is described by a natural language expression. This task demands fine-grained multimodal reasoning, including deep linguistic understanding, interpretation of complex phrases, and contextual reasoning over spatial and semantic relationships among objects. The standard evaluation metric is accuracy at an IoU threshold of 0.5. We evaluate on three standard benchmarks: RefCOCO [^24], RefCOCO+ [^63], and RefCOCOg [^44]. As shown in Table 3, ObjEmbed achieves an average accuracy of 89.5, surpassing both specialized MLLMs designed for referring tasks and significantly larger general-purpose multimodal models. This strong performance demonstrates the high semantic discriminability of our object embeddings.

Results on local image retrieval benchmarks. Local image retrieval aims to retrieve a target image from a gallery based on a query that describes only a small region or specific object within it. The query can be a textual description (text-to-image, T2I) or an image exemplar (image-to-image, I2I), both requiring fine-grained cross-modal alignment between local regions and external queries. For text-based retrieval (T2I), we evaluate on three benchmarks: SORCE-1K [^39], REIRCOCO [^19], and ILIAS [^26]. On REIRCOCO and ILIAS, we adopt a simplified evaluation protocol (detailed in Appendix C). The evaluation metric is Recall@1 for SORCE-1K and REIRCOCO, and mAP@50 for ILIAS. These metrics assess whether the correct target images are ranked highly in the retrieval results, without considering bounding box localization accuracy. In our framework, we compute the overall image similarity by taking the maximum matching score among all objects. As shown in Table 4, we compare ObjEmbed with other global embedding models that represent the entire image as a single, holistic feature vector. However, such global representations suffer from semantic ambiguity and fail to capture fine-grained details, leading to poor performance in local retrieval tasks. Even FG-CLIP2 [^59], which incorporates regional contrastive learning, underperforms significantly in T2I retrieval when using object embeddings extracted by RoIAlign with the same object proposals and scoring strategy as ours. We hypothesize that this is due to misalignment between region features and complex queries. In contrast, our model represents each object with fine-grained details and gets an average score of 68.5, surpassing other models by around 20 points. For the image-based retrieval (I2I) task, we use global image embeddings to represent image exemplars and to retrieve target images. Surprisingly, although our model does not optimize for image-based retrieval tasks, the model aligns global text embeddings and global image embeddings in a unified semantic space. Therefore, it can transfer from text-based retrieval tasks to image-based retrieval tasks successfully.

### 4.3 Comparisons on Image-Level Tasks

In Table 5, we evaluate ObjEmbed on traditional image-text retrieval benchmarks, including long text retrieval (ShareGPT4V [^6] and DCI [^53]), short text retrieval (COCO [^7] and Flickr30K [^62]), and multilingual text retrieval (COCO-CN [^34] and Flikr30K-CN [^27]). Despite using a relatively small-scale training set, ObjEmbed achieves an overall score of 81.7 points, which is highly competitive with both traditional CLIP-style models and recent LMM-based embedding approaches. Moreover, it demonstrates robust generalization across varying text lengths and multiple languages.

### 4.4 Ablation Study

Table 6: Ablation studies on different object token designs.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="4">COCO</td><td>RefCOCO</td></tr><tr><td>AP</td><td>AP <sub>s</sub></td><td>AP <sub>m</sub></td><td>AP <sub>l</sub></td><td>Avg.</td></tr><tr><td>single token & label=1</td><td>37.1</td><td>28.8</td><td>47.9</td><td>52.8</td><td>86.8</td></tr><tr><td>single token & label=IoU</td><td>42.3</td><td>29.5</td><td>50.9</td><td>60.9</td><td>87.1</td></tr><tr><td>two tokens (iou+cls)</td><td>45.1</td><td>27.1</td><td>51.1</td><td>67.1</td><td>86.8</td></tr><tr><td>two tokens (cls+iou)</td><td>45.5</td><td>27.3</td><td>51.4</td><td>66.6</td><td>86.6</td></tr></tbody></table>

Table 7: Ablation studies on instructions.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="4">COCO</td><td>RefCOCO</td></tr><tr><td>AP</td><td>AP <sub>s</sub></td><td>AP <sub>m</sub></td><td>AP <sub>l</sub></td><td>Avg.</td></tr><tr><td>None</td><td>42.1</td><td>28.0</td><td>52.5</td><td>63.1</td><td>86.9</td></tr><tr><td>object instruction</td><td>45.5</td><td>27.3</td><td>51.4</td><td>66.6</td><td>86.6</td></tr><tr><td>object & task instructions</td><td>47.1</td><td>32.1</td><td>55.6</td><td>67.1</td><td>86.9</td></tr></tbody></table>

Table 8: Ablation studies on different training objectives. LIR <sup>∗</sup> denotes the average scores of local image retrieval tasks except for the I2I task. GIR denotes the average scores of global image retrieval tasks.

| Method | COCO | RefCOCO | LIR <sup>∗</sup> | GIR |
| --- | --- | --- | --- | --- |
| object-level | 53.0 | 88.1 | 60.2 | \- |
| image-level | \- | \- | \- | 81.2 |
| object-level & image-level | 52.8 | 87.4 | 62.9 | 81.6 |

Table 9: Ablation studies on image-level supervision design. ‘Share’ denotes whether global text embeddings and local text embeddings share the same special token. ‘#token’ denotes the number of global image tokens and ‘Type’ can be a single long caption (long), a single short caption (short), a single randomly selected caption (mix), or two captions used simultaneously (both). ‘LIR’ denotes the average scores of local image retrieval tasks. ‘GIR’ denotes the average scores of global image retrieval tasks.

| Exp. | Share | #token | Type | COCO | RefCOCO | LIR | GIR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ✓ | 1 | mix | 52.4 | 86.0 | 67.3 | 80.1 |
| 2 |  | 1 | mix | 52.7 | 86.8 | 67.6 | 80.1 |
| 3 |  | 1 | long | 52.6 | 86.1 | 67.2 | 72.7 |
| 4 |  | 1 | short | 52.6 | 86.3 | 67.4 | 80.2 |
| 5 |  | 1 | both | 52.8 | 86.6 | 67.5 | 80.6 |
| 6 |  | 2 | both | 52.8 | 87.4 | 68.6 | 81.6 |

In this subsection, we explore the effects of different designs with the 4B model, partial data, and a one-epoch training schedule.

Ablation studies on object token design. As objects are sensitive to localization quality, an object embedding model should have the ability to assess the quality of boxes. To achieve the goal, we change classification labels in sigmoid focal loss to box IoUs, which increases 5.2% mAP on COCO, as shown in Table 6. However, the classification and box localization may have conflicting training objectives. We find that decoupling an object into two tokens, one for classification and the other for IoU regression, further improves 3.2% mAP on COCO. And placing the classification token in front of the IoU token is slightly better. Since REC places less emphasis on precise localization, its performance is relatively robust to the choice of IoU design.

Ablation studies on instructions. As shown in Table 7, we study the effects of instructions. As we encode multiple objects simultaneously, using the object instruction (“Object i: $\langle$ object $\rangle$ $\langle$ iou $\rangle$.”) to separate different objects can increase the distinctiveness of each object and increase 3.4% mAP on COCO. Further, different tasks focus on different aspects of objects. Object detection focuses on the shared properties within a class while referring expression comprehension requires encoding instance-specific features. Incorporating task instructions to guide the model encoding different features for different tasks is beneficial. Task prompts used in this work are shown in Appendix A.

Ablation studies on different training objectives. In this work, we build a universal object embedding model along with the global image representation ability. In Table 8, we find that incorporating an image-level training objective with an object-level training objective can increase the local image retrieval performance by 2.7 points while maintaining the performance on object detection and REC. Further, comparing with single image-level training, training with both objectives can also boost the global image retrieval performance by 0.4 points, demonstrating the mutual benefits between the two training objectives.

Ablation studies on image-level supervision design. In this work, we use both local and global text embeddings and long and short captions as global text embeddings to learn global image embeddings. We study their effects in Table 9. As local text embeddings match with object embeddings while global text embeddings match with global image embeddings, we find that not sharing the text tokens can mitigate task discrepancies and get higher performance on COCO (+0.3% mAP) and RefCOCO (+0.8), comparing Exp1 and Exp2. Further, using only short captions, long captions, or a mixture of them can not handle complex retrieval requirements. Using both captions for supervision achieves the highest 80.6 global retrieval results (Exp2-Exp5). Finally, using two global image tokens (Exp6), one for short captions and one for long captions, gets the highest results. And we find that the performance on object detection and REC is quite robust to the image-level designs.

### 4.5 Discussion

Table 10: Effect of the quality of proposals. ‘mix’ denotes a setting where a portion of the generated object proposals are randomly replaced with ground truth bounding boxes. We report fixed AP [^11] on LVIS. ‘AR’ is computed as the mean recall across IoU thresholds ranging from 0.50 to 0.95.

<table><tbody><tr><td rowspan="2">mix</td><td colspan="5">COCO</td><td colspan="5">LVIS v1 val</td></tr><tr><td>AR</td><td>AP</td><td>AP <sub>s</sub></td><td>AP <sub>m</sub></td><td>AP <sub>l</sub></td><td>AR</td><td>AP</td><td>AP <sub>s</sub></td><td>AP <sub>m</sub></td><td>AP <sub>l</sub></td></tr><tr><td></td><td>66.7</td><td>53.0</td><td>35.6</td><td>59.6</td><td>72.2</td><td>50.8</td><td>49.0</td><td>53.7</td><td>50.2</td><td>45.5</td></tr><tr><td>✓</td><td>100.0</td><td>65.2</td><td>55.3</td><td>70.1</td><td>77.5</td><td>100.0</td><td>66.6</td><td>66.7</td><td>67.1</td><td>65.9</td></tr></tbody></table>

How does proposal quality affect performance? In ObjEmbed, we employ a state-of-the-art proposal generator, WeDetect-Uni [^13], to produce 100 object proposals per image. Therefore, the recall rate is essential as the model can not encode missing objects. As shown in Table 10, WeDetect-Uni achieves an Average Recall (AR) of 66.7 on COCO and 50.8 on LVISv1 val [^18]. To assess the upper bound, we conduct an oracle experiment in which ground truth bounding boxes are randomly mixed into the generated proposals (denoted as ‘mix’ in the table). With access to ground truth regions, ObjEmbed achieves a significant gain of 12.2% AP on COCO and 17.6% AP on LVIS, demonstrating that (1) the model can accurately align object proposals with corresponding text descriptions, and (2) it effectively ranks high-quality proposals above low-quality ones during matching. These results indicate that the representation learning in ObjEmbed is orthogonal to proposal quality but its overall performance is still limited by the recall rate of proposals. Therefore, improving proposal quality through fine-tuning the proposal network on target datasets or leveraging human-annotated bounding boxes can further boost performance.

Table 11: Ablation studies on supporting box regression.

| box regression | COCO | RefCOCO | LIR | GIR |
| --- | --- | --- | --- | --- |
| ✓ | 52.5 | 86.2 | 68.1 | 81.5 |
|  | 52.8 | 87.4 | 68.6 | 81.6 |

Can we directly use ObjEmbed to regress high-quality bounding boxes, similar to object detectors? Given the importance of proposal quality, a natural question arises: can the embedding model itself be leveraged to refine proposals by predicting bounding box offsets, thereby improving localization accuracy? To investigate this, we conduct experiments where the model uses the IoU embeddings to predict both IoU scores and bounding box offsets simultaneously. The offset regression head is trained with a combination of L1 loss and IoU loss, following the standard practice in DETR [^4]. Similar to IoU regression loss, the bounding box regression loss is applied only to positive proposals with an IoU greater than 0.5. As a result, the regression head can refine existing bounding boxes but is unable to generate missing ones. As shown in Table 11, we find that incorporating box regression degrades overall performance, possibly due to learning conflicts.

## 5 Conclusion

In this work, we present ObjEmbed, a novel MLLM-based object embedding model that features object-oriented representation, versatility, and efficient encoding. In our framework, each object is represented by two complementary embeddings: an object embedding for semantic matching and an IoU embedding for assessing localization quality. This decoupled design reduces learning complexity while maintaining encoding efficiency. ObjEmbed can be seamlessly applied to a wide range of downstream tasks, including object detection, referring expression comprehension, local image retrieval, and global image retrieval. The consistently high and balanced performance across 18 diverse benchmarks demonstrates the effectiveness and generalization capability of our approach.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

## Appendix A Details of Task Instructions

ObjEmbed is a versatile embedding model applicable to a wide range of downstream tasks. However, different tasks emphasize distinct aspects of object representation. For instance, object detection relies on shared semantic properties within object categories, whereas referring expression comprehension requires fine-grained, instance-specific features to distinguish between visually similar objects.

To address these divergent requirements and mitigate potential task conflicts, we introduce task-specific instructions to guide the model in generating context-aware embeddings tailored to each task. The instructions used during training are listed as follows:

Object Detection

- Detect all objects in the image by identifying the common visual features of their respective classes.
- Localize each object by matching it to the archetypal visual form of its category.
- Detect all objects in the image by recognizing the shared visual attributes of their respective categories.
- Identify every object in the scene based on the core visual characteristics that define its class.
- Locate all objects by using the fundamental visual properties common to their object class.
- Perform object detection by referencing the shared visual patterns that characterize each class.
- Find every object in the picture by matching it to the defining visual traits of its category.
- Identify all objects present by their class-defining visual features, which are common across all instances.
- Detect every object based on the visual essence shared by all members of its class.
- Localize each object in the image according to the general visual blueprint of its category.
- Identify all objects by applying the common visual criteria that define their respective classes.

Referring expression comprehension

- Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects.
- Identify the single instance mentioned in the text by considering its distinct visual features, its location within the scene, and its context relative to nearby items.
- Ground the referring expression by pinpointing the object that matches the description’s details regarding its appearance, placement, and interaction with other elements.
- Find the objects that correspond to the given description, paying close attention to its specific details, its position, and how it relates to its neighbors.
- Disambiguate and find the correct object by carefully examining the provided description of its instance-specific properties, its coordinates in the image, and its spatial arrangement with other objects.
- Resolve the reference by identifying the object that uniquely matches the specified details, including its appearance, its place in the scene, and its connections to adjacent objects.
- Pinpoint the described instance by evaluating its specific visual traits, its spatial context, and its relational properties with other objects in the image.
- To locate the referred object, you must analyze three things from the description: 1) its unique visual details (e.g., color, texture), 2) its precise location, and 3) its relationship to the objects around it.
- Find the specific object the text is referring to by synthesizing information about its individual characteristics, its location, and its interactions within the scene.

Celebrity

- Identify the famous person depicted in this image.
- Recognize and name the public figure featured in this picture.
- Please provide the name of the well-known individual in this image.
- Identify all recognizable celebrities in this image.
- State the name of the celebrity shown.

## Appendix B Details of Dataset Annotation

To train ObjEmbed, each image in the training dataset is annotated with a long caption, a short caption, and several regions of interest, where each region is associated with a corresponding object description. To minimize false-negative conflicts during contrastive learning, captions are designed to be as diverse and distinctive as possible. Furthermore, they should exclude subjective or interpretive content to ensure objectivity and consistency. To achieve high-quality annotations, we carefully design structured prompts and employ a state-of-the-art multimodal large language model, Qwen-VL-235B [^1], as the automated annotator. The annotation prompts are as follows:

<svg height="461.27" id="A2.p2.pic1" overflow="visible" version="1.1" viewBox="0 0 600 461.27" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,461.27) matrix(1 0 0 -1 0 0)"><g fill="#333333" fill-opacity="1.0" style="--ltx-fill-color:#333333;"><path d="M 0 5.91 L 0 455.37 C 0 458.63 2.64 461.27 5.91 461.27 L 594.09 461.27 C 597.36 461.27 600 458.63 600 455.37 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0" style="--ltx-fill-color:#F2F2F2;"><path d="M 1.97 5.91 L 1.97 437.16 L 598.03 437.16 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 445.76)"><foreignObject color="#FFFFFF" height="12.3" overflow="visible" style="--ltx-fg-color:#FFFFFF;--ltx-fo-width:40.23em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;" transform="matrix(1 0 0 -1 0 9.61)" width="556.69"><span id="A2.p2.pic1.5.5.5.1.1" style="width:40.23em;"><span id="A2.p2.pic1.5.5.5.1.1.1">The prompt for annotating region-level captions</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.47)"><foreignObject color="#000000" height="411.57" overflow="visible" style="--ltx-fg-color:#000000;--ltx-fo-width:40.23em;--ltx-fo-height:29.55em;--ltx-fo-depth:0.19em;" transform="matrix(1 0 0 -1 0 408.88)" width="556.69"><span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4" style="width:40.23em;"><span id="A2.p2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2">The full image is <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\langle"><semantics><mo stretchy="false">⟨</mo> <annotation encoding="application/x-tex">\langle</annotation></semantics></math> FULL_IMAGE <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\rangle"><semantics><mo stretchy="false">⟩</mo> <annotation encoding="application/x-tex">\rangle</annotation></semantics></math></span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4">The cropped object is at [x1, y1, x2, y2], <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\langle"><semantics><mo stretchy="false">⟨</mo> <annotation encoding="application/x-tex">\langle</annotation></semantics></math> CROP_IMAGE <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\rangle"><semantics><mo stretchy="false">⟩</mo> <annotation encoding="application/x-tex">\rangle</annotation></semantics></math></span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.5">Analyze the given image and generate multiple detailed descriptions for the given object (’instance’) found in the image. Each description must be accurate and unique, focusing solely on the information available in the image and annotations. Do not include any information that is not explicitly present in the image or annotation data. The descriptions should ensure that the object can be uniquely identified from a large set of similar images.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.6">**Description Generation**:</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.7">1. Generate concise, clear descriptions.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.8">2. Focus mainly on the object itself using:</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.9">- The object’s inherent properties.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.10">- The special details that can be used for separating other instances of the same category.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.11">- Ensure each description allows the object to be uniquely identifiable within the image.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.12">- Ensure diversity without referencing prior descriptions.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.13">- Avoid direct mention of coordinate values.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.14">3. For instances that are heavily occluded, blurry, or too small to be recognized due to a tiny bounding box, directly return “Instance quality is poor.”</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.15">**Description Style**:</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.16">1. Use short sentences.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.17">2. Minimize commas, avoid long or complex sentences.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.18">3. Each description must reflect the interesting, accurate, and clear representation of the object, emphasizing the object as the focal point.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.19">4. Each description should be more natural and aligned with human language conventions.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.20">5. Each description must use the described object as the subject of the sentence.</span> <span id="A2.p2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.4.21">Output the descriptions in JSON format.</span></span></foreignObject></g></g></svg>

<svg height="411.46" id="A2.p3.pic1" overflow="visible" version="1.1" viewBox="0 0 600 411.46" width="600"><g fill="#000000" stroke="#000000" stroke-width="0.4pt" style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,411.46) matrix(1 0 0 -1 0 0)"><g fill="#333333" fill-opacity="1.0" style="--ltx-fill-color:#333333;"><path d="M 0 5.91 L 0 405.56 C 0 408.82 2.64 411.46 5.91 411.46 L 594.09 411.46 C 597.36 411.46 600 408.82 600 405.56 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" style="stroke:none"></path></g><g fill="#F2F2F2" fill-opacity="1.0" style="--ltx-fill-color:#F2F2F2;"><path d="M 1.97 5.91 L 1.97 387.35 L 598.03 387.35 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" style="stroke:none"></path></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 395.95)"><foreignObject color="#FFFFFF" height="12.3" overflow="visible" style="--ltx-fg-color:#FFFFFF;--ltx-fo-width:40.23em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;" transform="matrix(1 0 0 -1 0 9.61)" width="556.69"><span id="A2.p3.pic1.3.3.3.1.1" style="width:40.23em;"><span id="A2.p3.pic1.3.3.3.1.1.1">The prompt for annotating image-level captions</span> </span></foreignObject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 16.47)"><foreignObject color="#000000" height="361.76" overflow="visible" style="--ltx-fg-color:#000000;--ltx-fo-width:40.23em;--ltx-fo-height:25.95em;--ltx-fo-depth:0.19em;" transform="matrix(1 0 0 -1 0 359.07)" width="556.69"><span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2" style="width:40.23em;"><span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2">The full image is <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\langle"><semantics><mo stretchy="false">⟨</mo> <annotation encoding="application/x-tex">\langle</annotation></semantics></math> FULL_IMAGE <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\rangle"><semantics><mo stretchy="false">⟩</mo> <annotation encoding="application/x-tex">\rangle</annotation></semantics></math></span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3">You are a precise, factual image cataloger. Your task is to generate a literal description of the image for a visual database. You should generate a **short caption** and a **long caption** for each image.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.4">For short captions, follow these rules strictly:</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.5">1. **Identify Core Elements:** Describe the primary entities, objects, and the surrounding environment.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.6">2. **Be Concise:** The entire description must be a single, clear sentence or phrase under 30 words.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.7">3. **Be Natural:** Each description should be more natural and aligned with human language conventions.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.8">For long captions, follow these rules strictly:</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.9">1. **Include Key Details:** Mention essential visual attributes like color, count, spatial relationships (e.g., “on the left”, “in the background”), and relationships between objects.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.10">2. **Be Objective:** Describe only what you can see. Strictly avoid any subjective language, atmosphere (e.g., “peaceful”, “sad”), or interpretation of intent, actions, or the purpose of objects.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.11">3. **Be Concise:** The entire description must be under 100 words but more than 50 words.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.12">4. **Be Natural:** Each description should be more natural and aligned with human language conventions. Sentences need to be smooth and coherent.</span> <span id="A2.p3.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.13">Describe the image and output the descriptions in JSON format.</span></span></foreignObject></g></g></svg>

## Appendix C Details of Local Image Retrieval Benchmarks

Local image retrieval is a challenging task in which the textual or visual query corresponds to only a small region or specific object within an image, rather than the entire scene. In this work, we evaluate on three established benchmarks:

SORCE-1K [^39] comprises 1,023 carefully curated images with complex backgrounds and textual queries that describe less prominent small objects with minimum surrounding context. The target objects typically occupy less than 10% of the image area, posing significant challenges for global image embedding models that focus on holistic scene understanding. Each query is associated with exactly one positive image, and performance is evaluated using Recall@1.

REIRCOCO [^19] consists of 4,994 images from the COCO dataset, each annotated with a referring expression describing a specific object. The original evaluation protocol requires both image retrieval and object localization. However, since global embedding models lack localization capabilities, we adapt the protocol to a standard text-to-image retrieval setting, where the goal is to retrieve the correct image containing the described object. We report performance using Recall@1.

ILIAS [^26] supports both text-based and image-based local retrieval, with queries formulated as a natural language description or a large image exemplar. The original dataset includes 5 million distractors, making full-scale evaluation computationally infeasible. Instead, we construct a manageable gallery using all 4,715 positive images. Different from the benchmarks mentioned above, ILIAS contains only 1,232 queries, each potentially matching multiple positive images. We evaluate using mAP@50, which computes mean average precision over the top 50 retrieved results.

## Appendix D Limitation

As a pioneering MLLM-based object embedding model, ObjEmbed can be further improved in the following directions:

- Scaling up training data: Due to resource constraints, we currently train on only 1.3M samples, significantly fewer than those used in CLIP-series models. Scaling up the pretraining data through broader data collection could enhance model performance.
- Hard negative mining: Hard negatives are essential for learning discriminative embeddings. However, effective mining must be balanced with the mitigation of the false negative problem. Integrating robust hard negative sampling strategies while accounting for annotation incompleteness can further boost performance.

## Appendix E Visualization

Visualizations of referring expression comprehension results. In addition to strong performance on standard referring benchmarks (e.g., RefCOCO, RefCOCO+, RefCOCOg), in Figure 3, we demonstrate that ObjEmbed exhibits not only strong OCR capabilities (a, b, d), but also commonsense reasoning (c) and image-image matching abilities (e, f), highlighting its generalizability and versatility.

Visualizations of local image retrieval results. In Figure 4, we present qualitative results for three queries from the SORCE-1K dataset, showing the top-3 retrieved images. Our ObjEmbed not only ranks the correct target images as the top result but also accurately localizes the queried objects within the images. In contrast, even the state-of-the-art global image embedding model, Qwen-VL-Embedding-8B [^33], fails to retrieve the correct images in these challenging cases, highlighting the limitations of holistic representations in capturing fine-grained, localized visual content.

![Refer to caption](https://arxiv.org/html/2602.01753v2/x2.png)

Figure 3: Visualizations of referring expression comprehension results with text queries and image queries.

![Refer to caption](https://arxiv.org/html/2602.01753v2/x3.png)

Figure 4: Visualizations of retrieval results on SORCE-1K. Our ObjEmbed successfully ranks the target image as the top result and accurately localizes the target objects (highlighted with red bounding boxes). In contrast, global image embedding models, like Qwen3-VL-Embedding 8B, tend to overlook small objects.

Visualizations of self-annotated data. Figure 5 shows representative examples from our self-annotated dataset. Thanks to carefully designed prompts and the use of frontier MLLMs, the generated captions are both accurate and highly distinctive.

![Refer to caption](https://arxiv.org/html/2602.01753v2/x4.png)

Figure 5: Visualizations of self-annotated data. Each image is annotated with high-quality image-level and object-level captions. Images come from SA-1B 25.

[^1]: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: Appendix B, §1, §3.1, §3.3, Table 2, Table 2, §4.1, Table 3, Table 3.

[^2]: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. Cited by: §1, Table 3, Table 3.

[^3]: The devil is in the fine-grained details: evaluating open-vocabulary object detectors for fine-grained understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22520–22529. Cited by: Table 1, §4.2.

[^4]: End-to-end object detection with transformers. In European conference on computer vision, pp. 213–229. Cited by: §4.5.

[^5]: Revisiting referring expression comprehension evaluation in the era of large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 513–524. Cited by: Table 1.

[^6]: Sharegpt4v: improving large multi-modal models with better captions. In European Conference on Computer Vision, pp. 370–387. Cited by: §4.3.

[^7]: Microsoft coco captions: data collection and evaluation server. arXiv preprint arXiv:1504.00325. Cited by: §4.3.

[^8]: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271. Cited by: Table 3.

[^9]: Meta clip 2: a worldwide scaling recipe. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: §2.1, Table 5.

[^10]: FlashAttention-2: faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), Cited by: §3.1.

[^11]: Evaluating large-vocabulary object detectors: the devil is in the details. arXiv preprint arXiv:2102.01066. Cited by: Table 10, Table 10.

[^12]: Lami-detr: open-vocabulary detection with language model instruction. In European Conference on Computer Vision, pp. 312–328. Cited by: §2.2.

[^13]: WeDetect: fast open-vocabulary object detection as retrieval. arXiv preprint arXiv:2512.12309. Cited by: §2.2, §3.1, §3.3, Table 2, Table 2, Table 2, §4.1, §4.5.

[^14]: Frozen-detr: enhancing detr with image understanding from frozen foundation models. In Advances in Neural Information Processing Systems, Vol. 37, pp. 105949–105971. Cited by: §2.2.

[^15]: A hierarchical semantic distillation framework for open-vocabulary object detection. IEEE Transactions on Multimedia. Cited by: §1, §2.2.

[^16]: Llmdet: learning strong open-vocabulary object detectors under the supervision of large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14987–14997. Cited by: §1, §2.2, Table 2.

[^17]: Open-vocabulary object detection via vision and language knowledge distillation. International Conference on Learning Representations. Cited by: §2.2.

[^18]: Lvis: a dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5356–5364. Cited by: Table 1, §4.5.

[^19]: Referring expression instance retrieval and a strong end-to-end baseline. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 4464–4473. Cited by: Appendix C, §2.1, Table 1, §4.2, Table 3.

[^20]: Detect anything via next point prediction. arXiv preprint arXiv:2510.12798. Cited by: Table 3.

[^21]: Chatrex: taming multimodal llm for joint perception and understanding. arXiv preprint arXiv:2411.18363. Cited by: Table 3.

[^22]: Referring to any person. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Cited by: Table 1, Table 3.

[^23]: Vlm2vec: training vision-language models for massive multimodal embedding tasks. In International Conference on Learning Representations, Cited by: §1, §2.1, Table 5.

[^24]: Referitgame: referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 787–798. Cited by: Table 1, §4.2.

[^25]: Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026. Cited by: Figure 5, Figure 5, §3.3.

[^26]: Ilias: instance-level image retrieval at scale. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14777–14787. Cited by: Appendix C, §4.2.

[^27]: Fluency-guided cross-lingual image captioning. In Proceedings of the 25th ACM international conference on Multimedia, pp. 1549–1557. Cited by: §4.3.

[^28]: UME-r1: exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405. Cited by: §1, §2.1, Table 5.

[^29]: Elevater: a benchmark and toolkit for evaluating language-augmented visual models. In Advances in Neural Information Processing Systems, Vol. 35, pp. 9287–9301. Cited by: §4.2.

[^30]: GUIDED: granular understanding via identification, detection, and discrimination for fine-grained open-vocabulary object detection. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: Table 2.

[^31]: Lmm-det: make large multimodal models excel in object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 308–318. Cited by: Table 2.

[^32]: Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10965–10975. Cited by: Table 2.

[^33]: Qwen3-vl-embedding and qwen3-vl-reranker: a unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720. Cited by: Appendix E, §2.1, Table 5.

[^34]: COCO-cn for cross-lingual image tagging, captioning, and retrieval. IEEE Transactions on Multimedia 21 (9), pp. 2347–2360. Cited by: §4.3.

[^35]: Describe anything: detailed localized image and video captioning. In Proceedings of the IEEE/CVF international conference on computer vision, Cited by: Table 1.

[^36]: Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pp. 2980–2988. Cited by: §3.2.

[^37]: Microsoft coco: common objects in context. In European conference on computer vision, pp. 740–755. Cited by: Table 1, §4.2.

[^38]: Gres: generalized referring expression segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 23592–23601. Cited by: Table 1.

[^39]: SORCE: small object retrieval in complex environments. arXiv preprint arXiv:2505.24441. Cited by: Appendix C, §4.2.

[^40]: Finecops-ref: a new dataset and task for fine-grained compositional referring expression comprehension. arXiv preprint arXiv:2409.14750. Cited by: Table 1.

[^41]: VLM-fo1: bridging the gap between high-level reasoning and fine-grained perception in vlms. arXiv preprint arXiv:2509.25916. Cited by: Table 3.

[^42]: VLM-fo1: bridging the gap between high-level reasoning and fine-grained perception in vlms. arXiv preprint arXiv:2509.25916. Cited by: Table 2.

[^43]: Grounding dino: marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pp. 38–55. Cited by: §1, §2.2, Table 2, Table 3.

[^44]: Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 11–20. Cited by: Table 1, §4.2.

[^45]: Coco-o: a benchmark for object detectors under natural distribution shifts. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6339–6350. Cited by: §4.2.

[^46]: Vlm2vec-v2: advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590. Cited by: Table 5.

[^47]: Scaling open-vocabulary object detection. In Advances in Neural Information Processing Systems, Vol. 36, pp. 72983–73007. Cited by: Table 2.

[^48]: Weak-to-strong compositional learning from generative models for language-based object detection. In European Conference on Computer Vision, pp. 1–19. Cited by: Table 2.

[^49]: Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. Cited by: §1, §2.1, §2.2, Table 5.

[^50]: Vlm-r1: a stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615. Cited by: Table 3.

[^51]: Eva-clip: improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389. Cited by: Table 5.

[^52]: Siglip 2: multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786. Cited by: §2.1, Table 5.

[^53]: A picture is worth more than 77 text tokens: evaluating clip-style models on dense captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26700–26709. Cited by: §4.3.

[^54]: V3det: vast vocabulary visual detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19844–19854. Cited by: Table 1.

[^55]: Internvl3. 5: advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265. Cited by: Table 3.

[^56]: Hq-clip: leveraging large vision-language models to create high-quality image-text datasets and clip models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22447–22456. Cited by: §2.1.

[^57]: Aligning bag of regions for open-vocabulary object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15254–15264. Cited by: §2.2.

[^58]: Described object detection: liberating object detection with flexible expressions. In Advances in Neural Information Processing Systems, Vol. 36, pp. 79095–79107. Cited by: §4.2.

[^59]: FG-clip 2: a bilingual fine-grained vision-language alignment model. arXiv preprint arXiv:2510.10921. Cited by: §1, §2.1, §4.2, Table 5.

[^60]: FG-clip: fine-grained visual and textual alignment. In International conference on machine learning, Cited by: §1, §1, §2.1.

[^61]: Demystifying clip data. In International Conference on Learning Representations, Cited by: §2.1.

[^62]: From image descriptions to visual denotations: new similarity metrics for semantic inference over event descriptions. Transactions of the association for computational linguistics 2, pp. 67–78. Cited by: §4.3.

[^63]: Modeling context in referring expressions. In European conference on computer vision, pp. 69–85. Cited by: Table 1, §4.2.

[^64]: Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986. Cited by: §1, §2.1, §3.2.

[^65]: Dino: detr with improved denoising anchor boxes for end-to-end object detection. In International Conference on Learning Representations, Cited by: Table 2.

[^66]: Magiclens: self-supervised image retrieval with open-ended instructions. In International conference on machine learning, Cited by: §1.

[^67]: GME: improving universal multimodal retrieval by multimodal llms. In Proceedings of the Computer Vision and Pattern Recognition Conference, Cited by: §1, §2.1, Table 5, Table 5.

[^68]: Octopus: a multi-modal llm with parallel recognition and sequential understanding. In Advances in Neural Information Processing Systems, Cited by: Table 3.