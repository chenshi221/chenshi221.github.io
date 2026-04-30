---
title: "Affective Image Editing: Shaping Emotional Factors via Text Descriptions"
source: "https://arxiv.org/html/2505.18699v1"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
∎

<sup>1</sup> <sup>2</sup> <sup>3</sup> <sup>4</sup> <sup>5</sup> <sup>6</sup> <sup>7</sup> <sup>8</sup> <sup>9</sup> <sup>10</sup> <sup>11</sup> <sup>12</sup>

Peixuan Zhang <sup>†</sup>    Shuchen Weng <sup>†</sup>    Chengxuan Zhu    Binghao Tang    Zijian Jia     
Si Li    Boxin Shi

(Received: date / Accepted: date)

###### Abstract

In daily life, images as common affective stimuli have widespread applications. Despite significant progress in text-driven image editing, there is limited work focusing on understanding users’ emotional requests. In this paper, we introduce AIEdiT for Affective Image Editing using Text descriptions, which evokes specific emotions by adaptively shaping multiple emotional factors across the entire images. To represent universal emotional priors, we build the continuous emotional spectrum and extract nuanced emotional requests. To manipulate emotional factors, we design the emotional mapper to translate visually-abstract emotional requests to visually-concrete semantic representations. To ensure that editing results evoke specific emotions, we introduce an MLLM to supervise the model training. During inference, we strategically distort visual elements and subsequently shape corresponding emotional factors to edit images according to users’ instructions. Additionally, we introduce a large-scale dataset that includes the emotion-aligned text and image pair set for training and evaluation. Extensive experiments demonstrate that AIEdiT achieves superior performance, effectively reflecting users’ emotional requests.

###### Keywords:

Image editingAffective computingImage generationCross-modal embedding

## 1 Introduction

Affective images have significant potential to convey personal feelings and thoughts [^60]. As the common affective stimuli, affective images enable individuals from diverse backgrounds to understand emotional intentions between each other [^76]. Consequently, they are widely used in films, advertisements, and social networks, enhancing the appeal and attractiveness for content creators [^8] [^53]. These widespread applications motivate researchers to explore approaches to further reduce the skill barriers and costs of creating affective images, empowering a broader audience to create and modify emotional visual content.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x1.png)

Figure 1: Illustration of the proposed AIEdiT method: Given text descriptions that express users’ emotional requests, AIEdiT creates affective images that evoke specific emotions through fine-grained text descriptions. In comparison, general text-driven editing methods ( e.g., MagicBrush 70 and InstructPix2Pix 3 ) fail to translate users’ emotional requests. Meanwhile, relevant affective image editing methods either focus on limited emotional factors (, color tune in AIFormer 56 ) or use coarse-grained control instructions (, emotion categories in EmoEdit 63 ).

Although existing general text-driven image editing methods [^57] [^72] have made notable improvements in controllability to synthesize visually-pleasing results, they still struggle to understand users’ emotional requests [^12] due to the inherent subjectivity and ambiguity of emotions [^18] [^37] [^36]. While recent affective image editing methods can effectively synthesize emotionally faithful results by integrating emotional priors, they primarily focus on editing limited emotional factors (*e.g.*, color tune [^56]) or use coarse-grained control instructions (*e.g.*, emotion categories [^39] [^62] [^27] [^63]). This restricts the options available for users to express nuanced individual emotions. Therefore, there remains a significant challenge in designing a general affective image editing method that allows for comprehensive manipulation of emotional factors through a fine-grained interactive approach.

In this paper, we introduce AIEdiT, a method for Affective Image Editing using Text descriptions. AIEdiT could create affective images that evoke specific emotions by adaptively shaping multiple emotional factors (*e.g.*, color tune, object categories, and facial expression) across the entire image according to users’ instructions. By leveraging the interactivity and flexibility of text descriptions, users are able to freely describe their nuanced emotions, allowing AIEdiT to effectively translate emotional requests into images, as shown in Fig. 1: In the first row, AIEdiT adjusts the color tone by making the original images brighter; In the second row, AIEdiT modifies object categories by transforming the fire into a tranquil stream, creating an atmosphere of contentment; In the last row, AIEdiT controls the facial expression, changing the cat from angry to sorrowful.

We build AIEdiT based on the pre-trained cross-modality generative model [^45], including a text encoder [^41] to capture complex semantics and diffusion models [^16] [^50] to create photo-realistic images. To represent universal emotional priors encapsulated in text descriptions, we additionally adopt BERT [^11] to extract emotional requests and build the emotional spectrum by learning a continuous emotional representation in a contrastive learning manner. An emotional mapper is further designed to translate visually-abstract emotional requests to visually-concrete semantic representations, which consists of a stack of Transformer blocks equipped with self-attention and cross-attention mechanisms. To ensure the created images evoke specific emotions, we take the Multimodal Large Language Model (MLLM) [^6] as the supervisor, providing robust guidance for emotional understanding and consistency. During inference, we strategically add noise to distort visual semantic representations and subsequently shape the corresponding emotional factors according to user-provided text descriptions, finally producing emotionally faithful results.

To provide training data annotated with nuanced emotions, we extend existing EmoSet [^61] into the EmoTIPS dataset, a high-quality Emotion-aligned Text and Image Pair Set, ensuring rich and diverse emotional annotation. We adopt three validation criteria and conduct four human evaluation experiments to ensure the emotional consistency between texts and images. Four quantitative metrics are further designed to comprehensively evaluate the quality of image editing.

In summary, our contributions are listed as follows:

- We propose AIEdiT for affective image editing, which uses emotional text descriptions to shape emotional factors and create emotionally faithful results.
- We build the emotional spectrum to represent universal emotional priors and design the emotional mapper for translating emotional requests into visual elements.
- We take an MLLM as the supervisor of aligning created images with multiple emotional factors. Additionally, we introduce the EmoTIPS dataset for model training.

## 2 Related work

### 2.1 Multi-modal Sentiment Analysis

Sentiment analysis is becoming increasingly significant in both natural language processing (NLP) and computer vision (CV) fields [^71]. In NLP, emotion analysis of text is typically performed at the word level [^17] and aspect level [^73], leveraging extracted textual features to accurately categorize sentiments. In CV, early approaches to image emotion analysis focus on separately analyzing color and textures [^74] [^75], object categories [^66] [^44], and facial expression [^9] [^23] [^54] [^59] properties of user-provided images. Based on these foundational studies, recent research [^42] [^55] is able to integrate multiple emotional factors and therefore achieve a more comprehensive understanding of image emotions. The advent of large language models (LLMs) demonstrates significant success in text understanding and response generation [^52] [^65]. Researchers [^2] are encouraged to leverage the extensive prior knowledge embedded in LLMs to enhance text emotion understanding. To further improve the interaction between images and text, multimodal adapters [^28] and multi-layer perceptions [^6] are applied to give LLMs multimodal processing capabilities. This advancement shows great potential for image emotion analysis and the joint sentiment analysis of text and images [^26] [^24] [^30].

### 2.2 Text-driven Image Editing

Since text is a flexible and user-friendly medium that accurately expresses users’ intentions, it has been widely used to guide image editing goals. The first attempts to use text as a condition are focused on generating text-related images for specific object categories [^58] [^69]. With the introduction of large-scale image-text datasets [^48] and high-quality generative models [^16] [^50], recent cross-modality generation models have the ability to generate a variety of scenarios [^45] [^47]. This inspires researchers to leverage pre-trained generative priors, and further explore approaches to edit user-provided images by fine-tuning parameters [^22] [^46], proposing additional adapters [^57] [^72], and utilizing zero-shot, training-free strategies [^31] [^38]. As a result, text-driven image editing methods cover major generative research areas, *e.g.*, low-level image processing [^4] [^68] and high-level image translation [^14] [^38]. To enable models to better understand users’ emotional requests, researchers collect large-scale emotional datasets with detailed text descriptions [^1] [^33] and propose AIFormer [^56] to modify specific properties of affective images. Although recent studies explore approaches using discrete emotion categories [^62] [^27] [^63] [^39] to manipulate more comprehensive emotional factors, the coarse-grained nature of emotion categories limits users’ ability to express their nuanced emotional requests, where text descriptions continue to offer distinct advantages.

## 3 EmoTIPS Dataset

Although numerous affective image samples annotated with emotion categories are available [^67] [^29], they oversimplify emotional priors, making them inadequate to represent nuanced emotions. Additionally, existing datasets with emotional text descriptions [^1] [^33] lack sufficient samples (over 100K), posing challenges for training a general model based on the limited emotional knowledge included. As a result, we introduce the EmoTIPS dataset <sup>1</sup>, an Emotion-aligned Text and Image Pair Set, to enable image editing models to effectively understand users’ emotional requests and achieve comprehensive manipulation of emotional factors. We collect affective images from the EmoSet [^61] and generate corresponding emotional text descriptions for each sample using the MLLM [^6]. Specifically, the pipeline for dataset construction includes three phases: annotation, validation, and evaluation, as detailed below.

Annotation. Due to the inherent subjectivity and ambiguity of emotions, we employ a Chain-of-Thought prompting strategy [^20] to guide the MLLM in accurately understanding and describing emotions step by step: (i) prompting the MLLM to identify the activities of the main objects in the image; (ii) incorporating the detected content as additional prompts to have the MLLM estimate emotional cues within the image; and (iii) combining the previously estimated image content and emotional cues to prompt the generation of full text descriptions. Notably, while constructing the dataset, we intentionally exclude descriptions of object appearances that can directly express emotions (*e.g.*, colors, styles, and facial expressions), thereby focusing the text descriptions exclusively on emotional cues. As a result, the annotated text descriptions include only the main objects and their corresponding emotions. We present an annotation sample and prompts used in the Chain-of-Thought prompting strategy in Fig. 2.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x2.png)

Figure 2: Visualization of the annotation process which a Chain-of-Thought prompting strategy.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x3.png)

Figure 3: Visualization of the three validation criteria: keyword detection, emotion classification, and text-image retrieval.

Validation. Due to the nature of the LLMs that generate diverse responses, we adopt three validation criteria to ensure the generated texts accurately describe the emotional cues within the images: (i) Keyword Detection. We utilize the VAD dictionary [^34] to classify emotional words into binary categories of positive and negative. Samples that do not include words matching the sentiment polarity of the images are discarded. (ii) Emotion Classification. We utilize the emotional LLMs [^30] and image classifier [^13] to estimate the emotion categories on the Mikel’s wheel [^32]. Samples where the categories do not align are discarded. (iii) Text-Image Retrieval. We randomly select $127$ additional images from the dataset and use a pre-trained text-image retrieval model [^25]. Samples are discarded if the paired image does not rank within the top 10 retrieval results. For clarify, we visualize the annotation pipeline and aforementioned three validation criteria in Fig. 3.

Evaluation. We conduct four human evaluation experiments to assess the quality of our annotated dataset, focusing on whether (i) the images are emotionally expressive; (ii) the text descriptions are emotionally expressive; (iii) the images and text descriptions share similar emotional expression; and (iv) the text descriptions accurately describe the content of the images. For each experiment, 25 volunteers are asked to evaluate 100 random samples given the choices of “Failed”, “Borderline”, “Acceptable” and “Perfect”. As shown in Table 1, in each experiment, over 90% volunteers rate the annotation quality as “Acceptable” or higher, demonstrating the robustness of our dataset.

Table 1: Percentage (%) of user ratings in the four experiments of human evaluation for the EmoTIPS dataset. Throughout the paper, best scores are highlighted in bold.

Rating Exp-I Exp-II Exp-III Exp-IV Failed $0.00$ $0.00$ $1.72$ $0.56$ Borderline $1.76$ $3.04$ $7.24$ $5.12$ Acceptable $9.64$ $21.68$ $32.56$ $26.60$ Perfect $\mathbf{88.60}$ $\mathbf{75.28}$ $\mathbf{58.48}$ $\mathbf{67.72}$

Summary. The introduced EmoTIPS dataset includes 1M affective images sourced from EmoSet [^61], with each image paired with a corresponding emotional text description. Additionally, we have reserved 3K samples specifically for model evaluation. Each sample consists of the original image, a target image that has a similar embedding [^49], an emotional text description of the target image, and an emotional category distribution of the target image. During evaluation, the model is only provided with the original image and an emotional text description as the editing instruction. The emotional category distribution of the target image serves as the ground truth for calculating quantitative scores. Additionally, we show randomly selected samples from the dataset in Fig. 4 to present the diversity.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x4.png)

Figure 4: Randomly selected samples from the EmoTIPS dataset.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x5.png)

Figure 5: Overview of the proposed AIEdiT model. (a) To represent universal emotional priors encapsulated in text descriptions, we build the continuous emotional spectrum and extract nuanced emotional requests (Sec. 4.1 ); (b) To manipulate emotional factors, We design the emotional mapper to translate visually-abstract emotional requests into visually-concrete semantic representations (Sec. 4.2 ); (c) To ensure specific emotions are effectively evoked, we introduce the MLLM to supervise emotional understanding (Sec. 4.3 ); During inference, we strategically distort the visual semantic representations and shape the corresponding emotional factors unser the guidance of text descriptions (Sec. 4.4 ). The emotional mapper is designed as a multi-modal Transformer, facilitating global interaction among input conditions, composed of a stack of Multi-head Self-Attention (MSA) blocks, Multi-head Cross-Attention (MCA) blocks, and Feed-Forward Networks (FFN). During the training, we keep the latent diffusion model frozen to preserve its generative ability.

## 4 Methodology

In this section, we first build the continuous emotional spectrum to represent universal emotional priors encapsulated in text descriptions (Sec. 4.1). Then, we design an emotional mapper that translates visually-abstract emotions into visually-concrete semantic representations (Sec. 4.2). Next, we introduce a multi-modal large language model to supervise emotional understanding, which ensures the created images accurately evoke specific emotions (Sec. 4.3). Based on this mapping relationship between abstract emotional requests and concrete emotional factors, we present a strategy for editing affective images by manipulating emotional factors across the entire image according to users’ instructions (Sec. 4.4), along with the training details (Sec. 4.5).

### 4.1 Building Emotional Spectrum

Previous works [^27] [^63] manipulate the emotional factors of affective images using discrete emotion categories. However, such coarse-grained control instructions limit users’ ability to express nuanced emotional requests, resulting in reduced controllability when shaping emotional factors (*e.g.*, the emotion of awe can be conveyed through the brilliance of fireworks or the grandeur of mountains). This motivates us to represent emotional priors encapsulated in text descriptions and build a continuous emotional spectrum, enabling flexible and accurate correspondences between users’ requests and emotional factors.

As presented in Fig. 5 (a), we first adopt BERT [^11] as the text encoder to extract emotional requests $r\in\mathbb{R}^{C^{\mathrm{t}}\times N^{\mathrm{l}}}$, where $C^{\mathrm{t}}$ is the number of embedding channels and $N^{\mathrm{l}}$ is the length of text descriptions. Since affective images evoking similar emotions often share common emotional expressions, we first estimate the emotional distribution for each image. Specifically, we use a ResNet [^13] pre-trained on EmoSet [^61] to estimate the distribution $d\in\mathbb{R}^{N^{\mathrm{c}}}$ over the $N^{\mathrm{c}}$ emotion categories defined by Mikel’s wheel [^32]. Next, we create samples $s=(r,d)$ by pairing each emotional request $r$ with the corresponding emotional distribution $d$. We further define relationships between these samples based on positions defined by the Mikel’s wheel [^32], where emotions located in the same region are considered positive pairs, while emotions located opposite each other (*e.g.*, awe and disgust) are treated as negative pairs. Therefore, we sample $N^{\mathrm{p}}$ such pairs using the batch-hard mining [^7], structured as tuples $[s^{\mathrm{anc}}_{i},s^{\mathrm{pos}}_{i},s^{\mathrm{neg}}_{i}]$, where $i\in\{1,\dots,N^{\mathrm{p}}\}$ is the index, $s^{\mathrm{anc}}$, $s^{\mathrm{pos}}$, and $s^{\mathrm{neg}}$ represent anchor, positive, and negative pairs, respectively. After that, we build the emotional spectrum by learning a continuous emotional representation:

$$
\!\!\mathcal{L}_{\mathrm{cl}}\!=\!\sum_{i=1}^{N^{\mathrm{p}}}\mathrm{max}\big{%
(}0,\mathrm{dis}(s^{\mathrm{anc}}_{i},s^{\mathrm{pos}}_{i})-\mathrm{dis}(s^{%
\mathrm{anc}}_{i},s^{\mathrm{neg}}_{i})+\alpha\big{)},\!\!
$$

where $\mathrm{dis}(s_{i},s_{j})=\|r_{i}-r_{j}\|_{2}/\|d_{i}-d_{j}\|_{2}$ is a distance function that measures the sentiment similarity between pairs and $\alpha=0.2$ is a hyper-parameter that controls the emotional margin. When forming training tuples, we randomly select samples from the dataset. The loss reduces the emotional distance between samples with similar emotions while increasing the distance between those that evoke opposite emotions, allowing the emotional spectrum to reflect emotions across all emotional dimensions. After building the emotional spectrum, the BERT encoder can extract emotional requests encapsulated in text descriptions and remain frozen.

### 4.2 Designing Emotional Mapper

With the emotional spectrum constructed, emotional requests can be extracted from text descriptions. However, there remains a challenge in accurately translating these visually-abstract emotional requests into visually-concrete semantic representations. Therefore, we design the emotional mapper to address this issue.

As illustrated in Fig. 5 (b), we employ a pre-trained CLIP model [^41] to extract text semantics $\hat{r}\in\mathbb{R}^{C^{\mathrm{s}}\times N^{\mathrm{l}}}$, where $C^{\mathrm{s}}$ is the number of embedding channels. Given that CLIP aligns visual and textual representations, we intend to further translate text semantics into emotion-aligned visual elements. To achieve this, we introduce a linear layer to adaptively extract the key semantic features $f^{\mathrm{k}}\in\mathbb{R}^{C^{\mathrm{s}}}$. These features, along with the emotional requests $r$ and text semantics $\hat{r}$, are fed into a multi-modal Transformer. This Transformer consists of a stack of Multi-head Self-Attention blocks (MSA) to capture overall emotional semantics, Multi-head Cross-Attention blocks (MCA) to integrate emotional semantics with related visual elements, and Feed-Forward Networks (FFN) to deeply understand the translation of text semantics. Throughout the translation process, the key semantic features $f^{\mathrm{k}}$ are used to scale the emotional requests after each sub-module as:

$$
\hat{f}^{\mathrm{r}}=\big{(}1+\mathbf{W}_{1}f^{\mathrm{k}}\big{)}\odot\frac{f^%
{\mathrm{r}}-\mu}{\sigma}+\mathbf{W}_{2}f^{\mathrm{k}},
$$

where $f^{\mathrm{r}}$ and $\hat{f}^{\mathrm{r}}$ represent the intermediate feature maps of emotional requests before and after scaling, respectively. $\mu$ and $\sigma$ denote the mean and standard deviation of $f^{\mathrm{r}}$, and $\odot$ means element-wise multiplication, while $\mathbf{W}_{\{1,2\}}$ are learnable matrices. Finally, the emotional mapper outputs the visually-concrete semantic representation $r^{\prime}\in\mathbb{R}^{{C^{\mathrm{s}}\times N^{\mathrm{l}}}}$.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x6.png)

Figure 6: Visualization of MLLM prompts used for emotional supervision.

### 4.3 Supervising Emotional Understanding

Emotional factors (*e.g.*, color tune, object categories, and facial expression) [^43] [^44] play a crucial role in how affective images evoke emotions. In the absence of target emotional images, the MLLM [^6] is introduced to supervise the emotional mapper to effectively create visual elements with corresponding emotional factors.

As shown in Fig. 5 (c), the MLLM is tasked with responding to our questions based on the image content and its comprehensive emotional understanding. We specifically design prompts for the MLLM to effectively analyze representative emotional factors and provide comprehensive responses. After the MLLM responds to each pre-defined prompt, we extract the CLIP embeddings of the responses and apply a sentiment alignment loss to minimize the mean square error with the semantic representation generated by the emotional mapper:

$$
\mathcal{L}_{\mathrm{sa}}=\sum_{i=1}^{N^{\mathrm{r}}}\|\phi(x^{\mathrm{t}})-%
\varphi(x^{\mathrm{r}})\|_{2},
$$

where $\phi$ and $\varphi$ refer to the emotional mapper and text encoder in CLIP [^41], respectively. $x^{\mathrm{t}}$ and $x^{\mathrm{r}}$ represent the user-provided text descriptions and responses from the MLLM separately, and $N^{\mathrm{r}}$ indicates the number of responses. Additionally, to maintain the nature of the diffusion model, we preserve the noise prediction loss:

$$
\mathcal{L}_{\mathrm{dm}}=\mathbb{E}_{t,z,\epsilon\sim\mathcal{N}(0,1)}\big{[}%
\|\epsilon-\epsilon_{\theta}\big{(}z_{t},t,\phi(x^{\mathrm{t}})\big{)}\|_{2}%
\big{]},
$$

where $\epsilon_{\theta}$ is the diffusion model and $t$ is the timestep in the denoising process. The total training loss is a combination:

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{sa}}+\beta\mathcal{L}_{%
\mathrm{dm}},
$$

where $\beta=10$. In practice, we only fine-tune the designed emotional mapper, and keep the backbone latent diffusion model and the autoencoder frozen, to preserve the generative ability of pre-trained models.

For better reproducibility of our results, We present the prompts used for supervision in Fig. 6. Notably, directly replacing the emotional mapper with the MLLM is infeasible. This is because MLLMs require visual input to extract relevant cues, whereas during editing, the target image is unavailable, and cues have to be derived solely from the user-provided text descriptions. Instead, we train the emotional mapper via self-reconstruction, thereby avoiding the need for target images.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x7.png)

Figure 7: Visualization of the correspondences between manipulated emotional factors and the number of noise-adding steps t = 25, 37 49 𝑡 t=25,37,49 italic\_t = 25, 37, 49.

### 4.4 Editing Affective Images

With the emotional mapper, the model effectively interprets emotional requests and translates them into visual elements aligned with the intended emotions. To further leverage the emotional knowledge learned by the emotional mapper and the generative prior preserved in diffusion models, we propose the editing strategy to modify affective images during the inference process.

Technically, for the user-provided affective image $I_{\mathrm{in}}$, we adopt a pre-trained image encoder $\mathcal{E}$ to map it into the latent space as $z=\mathcal{E}(I_{\mathrm{in}})$. Then, we strategically distort the visual semantic representations of the latent code by adding appropriate noise. Since the forward process of diffusion models [^16] [^50] can be expressed as a linear combination of the original latent code and a noise variable $\epsilon\sim\mathcal{N}(0,1)$, we can directly calculate the noised latent code distorted by adding noise $t$ times as $z_{t}=\sqrt{\alpha_{t}}z+\sqrt{1-\alpha_{t}}\epsilon$, where $\alpha_{t}$ is a hyperparameter. Next, we iteratively refine the distorted latent code and shape corresponding emotional factors under the guidance of text descriptions as $z_{t-1}=1/\sqrt{a_{t}}\cdot\big{(}z_{t}-\epsilon_{\theta}(z_{t},t,\phi({x^{%
\mathrm{t}}}))\cdot(1-\alpha_{t})/{\sqrt{1-\bar{\alpha}_{t}}}\big{)}+\sigma_{t}\epsilon$, where $\epsilon_{\theta}$ is our denoising network equipped with the emotional mapper, $\bar{\alpha}_{t}=\smash{\prod_{s=1}^{t}\alpha_{s}}$ is the cumulative noise amount, and $\sigma_{t}$ is a hyperparameter. After $t$ iterations, we obtain the restored latent code $z_{0}$. Finally, a pre-trained image decoder $\mathcal{D}$ maps the edited latent to the image domain as $I_{\mathrm{out}}=\mathcal{D}(z_{0})$.

As illustrated in Fig. 7, we observe that as the amount of added noise increases, the edited visual features change from color tone to object category, and eventually to overall semantics. This progression corresponds to emotional factors ranging from low to high levels [^76], indicating that the emotional mapper translates emotional requests following the hierarchical structure of emotional factors.

### 4.5 Training Details

We initialize the backbone with SD1.5 parameters, keeping them frozen to maintain generative priors. Training is performed on two NVIDIA GeForce 3090 GPUs using the Adam optimizer [^19] at a learning rate of $5\times 10^{-5}$. Initially, we spend 36 hours constructing the emotional spectrum, followed by an additional 96 hours optimizing the emotional mapper with emotional spectrum frozen.

## 5 Experiment

### 5.1 Quantitative Evaluation Metrics

We utilize three quantitative metrics to evaluate the performance of relevant methods. For each metric, we detail what it measures and how it is calculated below:

- The Fréchet Inception Distance (FID) [^15] is a metric used to evaluate whether edited images share a similar distribution with real images. Specifically, we utilize the pre-trained Inception-v3 network [^51] to extract features from both the editing and real images, and calculate the mean and covariance of the extracted features to create corresponding multivariate Gaussians. Finally, we measure the Fréchet distance between these Gaussians to assess the distribution similarity between editing and real images to reflect visual quality.
- The Sematic Clarity (Sem-C) is a metric used to evaluate whether the edited images present clearly recognizable visual contents to evoke corresponding emotions. Following EmoGen [^62], we categorize the edited images using the pre-trained object classifier from ImageNet [^10] and the scene classifier from PLACES365 [^77]. This score is then calculated as the average of the highest probability assigned by either classifier across all images.
- The Kullback-Leibler Divergence (KLD) [^21] is a metric used to evaluate the matching accuracy between editing results and emotional requests. Specifically, we use the fine-tuned ResNet-50 [^13] to categorize the editing results and target images into emotional categories based on Mikel’s wheel [^32], forming corresponding emotional distributions. After that, we calculate the Kullback-Leibler divergence between them to assess the emotional distance. Instead of directly calculating the accuracy, KLD provides a more nuanced evaluation of whether the target emotions are effectively evoked.

Additionally, we present user preference (Pref.) for the edited images of each method through a user study, demonstrating whether the created images align with the user-provided text descriptions.

### 5.2 Comparison with State-of-the-art Methods

To demonstrate the advantages of our AIEdiT in shaping multiple emotional factors based on text descriptions, we compare it with state-of-the-art general text-driven image editing methods (*i.e.*, ControlNet [^72], InstructPix2Pix [^3], and MGIE [^12]) and affective image editing methods (*i.e.*, AIFormer [^56] and EmoEdit [^63]).

Qualitative comparisons. In Fig. LABEL:fig:comparison, we present visual quality comparisons with the aforementioned methods. Intuitively, general text-driven image editing methods struggle to understand the emotional requests. Specifically, ControlNet [^72] overly modifies the object appearance (*e.g.*, Fig. LABEL:fig:comparison first row, the unnatural blue sky glow); InstructPix2Pix [^3] often presents fewer modifications (*e.g.*, Fig. LABEL:fig:comparison second row, the calm lake surface); MGIE [^12] struggle to enhance the visual appeal, (*e.g.*, Fig. LABEL:fig:comparison third row, the skewed eyes of cat); On the other hand, affective image editing methods are limited by their controllability. AIFormer [^56] can solely evoke specific emotions by modifying low-level features, it fails when the image content is ill-suited (*e.g.*, Fig. LABEL:fig:comparison fourth row, attempting to convey contentment within a desolate environment; Although EmoEdit [^63] edits images based on emotional categories, unable to evoke fine-grained emotions encapsulated in text descriptions (*e.g.*, Fig. LABEL:fig:comparison fifth row, the beach garbage partially altered, but still disorganized). In contrast, our AIEdiT accurately shapes corresponding emotional factors to reflect users’ emotional requests.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x8.png)

Figure 8: Ablation study results with different variants of the AIEdiT. After removing the proposed modules, the created images cannot accurately reflect users’ emotional requests.

Quantitative comparisons. We present quantitative comparison results in Table 2 compared with general text-driven image editing methods [^72] [^3] [^12] and affective image editing methods [^56] [^63]. This demonstrates that our method outperforms across all three quantitative metrics, demonstrating AIEdiT’s ability to create photo-realistic images (FID) with clearly recognizable visual content (Sem-C) that accurately reflect users’ emotional requests (KLD).

User study. To further assess the subjective preferences, We conduct a user study to evaluate whether our method is preferred by human observers. In the experiment, participants are shown an original image, an emotional text description, and the corresponding results generated by the related general text-driven image editing methods and affective image editing methods. They are asked to select the result that best aligns with the emotional text descriptions. We conduct the user study on Amazon Mechanical Turk (AMT) using 100 samples randomly selected from the EmoTIPS dataset, with results polled from 25 volunteers. As shown in Table 2, our method achieves the highest user preference, demonstrating its superior performance to produce affective images that closely align with user-provided emotional requests.

### 5.3 Ablation Study

We discard various modules and create four variants to study the impact of our proposed modules and designed losses. The evaluation scores and visual results of the ablation study are shown in Table 2 and Fig. 8, respectively.

W/o ESB (Emotional Spectrum Building). We discard the emotional spectrum built for the BERT. Without this emotional prior, this variant struggles to understand users’ emotional requests (*e.g.*, Fig. 8 first row, fireworks are less relevant to a sense of solitude).

W/o MCA (Multi-head Cross-Attention). We remove all multi-head cross-attention blocks in the emotional mapper. Lacking the integration with related visual elements, the image content changes inaccurately (*e.g.*, Fig. 8 second row, wood inexplicably appears in front of the bench).

W/o KSE (Key Semantic Extraction). We disable the linear layer for key semantic extraction and provide scaling instructions. As a result, this ablation has difficulty fully translating emotional requests (*e.g.*, Fig. 8 third row, the beach becomes dirty but not sufficiently disgusting).

W/o SAL (Sentiment Alignment Loss). We discard the sentiment alignment loss, preventing the MLLM from supervision. Consequently, the model struggles to shape corresponding emotional factors (*e.g.*, Fig. 8 fourth row, transforming the cemetery into a brick wall is meaningless).

Table 2: Quantitative experiment results of comparison and ablation. $\uparrow$ ($\downarrow$) means higher (lower) is better.

Method FID $\downarrow$ Sem-C $\uparrow$ KLD $\downarrow$ Pref. (%) $\uparrow$ Comparison with state-of-the-art methods ControlNet $32.58$ $0.656$ $2.7694$ $9.12$ InstructPix2Pix $32.77$ $0.649$ $2.8952$ $6.36$ MGIE $29.82$ $0.655$ $2.8302$ $15.08$ AIFormer $38.56$ $0.526$ $2.6940$ $7.32$ EmoEdit $37.10$ $0.620$ $2.6219$ $22.44$ Ours (AIEdiT) $\mathbf{27.93}$ $\mathbf{0.685}$ $\mathbf{2.4373}$ $\mathbf{39.68}$ Ablation study W/o ESB $29.67$ $0.672$ $2.6160$ $N/A$ W/o MCA $28.98$ $0.666$ $2.4894$ $N/A$ W/o KSE $28.14$ $0.657$ $2.5139$ $N/A$ W/o SAL $31.03$ $0.651$ $2.5597$ $N/A$

Table 3: Quantitative experiment results for different hyper-parameters $t=25,37,49$.

Parameter FID $\downarrow$ Sem-C $\uparrow$ KLD $\downarrow$ Pref (%) $\uparrow$ $t$ = 25 $\mathbf{20.40}$ $0.625$ $2.4805$ $27.96$ $t$ = 37 $27.93$ $\mathbf{0.685}$ $2.4373$ $\mathbf{45.32}$ $t$ = 49 $32.37$ $0.666$ $\mathbf{2.4301}$ $26.72$

### 5.4 Hyper-parameter Analysis

We evaluate the impact of the hyper-parameter $t$, which determines the amount of added noise in editing strategies (Sec. 4.4). As previously illustrated qualitatively in Fig. 7, increasing $t$ leads the image manipulation to gradually evolve from low-level color tone towards high-level overall semantics to achieve the desired emotional expression. To further validate this relationship quantitatively, we present additional experiment results in Table 3, which confirms this trend.

Specifically, as the noise-adding step $t$ increases, edited images become greater levels of distortion and reduced visual fidelity compared to the originals. This decline in visual quality is reflected in worse FID scores, indicating a larger feature gap. Concurrently, higher noise levels prove more effective in shaping the corresponding emotional factors, leading to improved KLD scores that indicate a more accurate match to the user’s emotional requests. We observe that adding an optimal level of noise can produce clear visual content while maximizing emotional expression, resulting in the highest Sem-C and user study scores. Therefore, we select $t=37$ for all of our experiments to balance the visual quality and emotional fidelity.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x9.png)

Figure 9: Additional application scenarios of AIEdiT.

### 5.5 Application

We present additional application scenarios of AIEdiT and show the results in Fig. 9. Technical details of each scenario are clarified as follows:

- Local manipulation. By providing an additional mask to indicate the regions to be edited, AIEdiT is able to preserve the remaining regions unchanged. Specifically, during the inference process, the mask is applied to prevent specific regions from being distorted. This could be formulated as $z_{t}=z_{t}^{\prime}\odot\mathbf{M}+z^{\mathrm{in}}_{t}\odot(1-\mathbf{M})$, where $z_{t}^{\prime}$ and $z^{\mathrm{in}}_{t}$ are the latent codes refined from $z_{t+1}$ and distorted from the original images, respectively. $\mathbf{M}$ refers to the binary mask provided by the user, with values of one indicating the regions to be edited and zeros denote the unchanged regions.
- Concept replacement. Given affective images and corresponding text descriptions, AIEdiT is capable of replacing the original emotional concepts. Specifically, null-text inversion [^35] is applied to invert the original image with a corresponding text prompt into the domain of our AIEdiT. Following the users’ instructions, the cross-attention maps is then modified to establish correspondences between the emotional concepts and emotional factors, using the Prompt-to-Prompt approach [^14]. This allows the emotion to be changed while keeping the instance similar.
- Semantic adjustment. Given affective images and target text descriptions, AIEdiT can adjust the overall image semantics to reflect different emotional requests. Specifically, the original semantics of the target text descriptions are first extracted using CLIP [^41], which are then optimized to better align with the image semantics. Next, we fine-tune the backbone parameters while keeping the emotional mapper and text semantics fixed. After interpolating between the optimized and original semantics, we feed the results into our fine-tuned AIEdiT to generate the adjusted images.

Table 4: Task differences between our proposed AIEdiT and relevant affective image creation methods.

Method Input modality Editing object Creation mode Category Text Specificity Variety Generation Editing EmoEditor ✓ ✓ ✓ EmoEdit ✓ ✓ ✓ C2A2 ✓ ✓ ✓ AIFormer ✓ ✓ ✓ EmoGen ✓ ✓ ✓ Ours (AIEdiT) ✓ ✓ ✓ ✓

## 6 Discussion

In this section, we illustrate in detail the task differences between AIEdiT and other relevant affective image creation methods (Sec. 6.1). Then, we demonstrate the model’s potential for generating affective images from random Gaussian noise, evaluating its performance qualitatively and quantitatively against relevant image generation methods (Sec. 6.2). Finally, we present and analyze failure cases to identify areas for potential improvement (Sec. 6.3).

### 6.1 Task Differences

We summarize the differences between our proposed AIEdiT and recent affective image creation methods in Table 4. These methods are distinguished based on three aspects: (i) Input modality: whether the method is guided by emotion categories or text descriptions; (ii) Editing object: whether the method manipulates specific properties or various factors. (iii) Creation mode: whether the method functions as an image generation or editing model; ✓indicates the model supports the corresponding functionality. We further highlight their differences as follows:

- Although EmoEditor [^27] and EmoEdit [^63] are closely related methods to edit affective images, their reliance on coarse-grained emotion categories restricts the expression of nuanced emotions and makes it challenging to preserve instance appearance during editing. We only conduct the comparison with EmoEdit [^63] in Sec. 5.2 since EmoEditor [^27] is not publicly available. We further qualitatively compare the controllability differences in Fig. 10 by directly pasting their reported results.
- C2A2 [^39] and AIFormer [^56] focus on specific image properties (*i.e.*, facial expression and color tune), which limits their application in general affective image editing for achieving more complex emotional editing goals. We only compare AIFormer [^56] in Sec. 5.2 since C2A2 [^39] is not publicly available.
- EmoGEN [^62] is an affective image generation model that produces images from scratch based on coarse-grained emotion categories, rather than editing specific input images provided by the user. We further discuss relevant generation method in Sec. 6.2.

In contrast, leveraging the flexibility of text descriptions, AIEdiT can specify the instance to be edited while maintaining its appearance similar and allows users to express their subjective feelings that may be difficult to categorize within a specific emotion category (*e.g.*, Fig. 10 second row, the fire, smoke, and chaos create a sense of emergency and danger). Consequently, to the best of our knowledge, AIEdiT is the first method to provide comprehensive manipulation of emotional factors through fine-grained text descriptions, allowing for more nuanced emotional expression.

![Refer to caption](https://arxiv.org/html/2505.18699v1/x10.png)

Figure 10: Illustration of our AIEdiT’s advantages. Leveraging text descriptions to reshape emotional factors offers more nuanced emotional expression than emotion category-based methods.

Table 5: Quantitative experiment results comparing with relevant image generation methods.

Methods FID $\downarrow$ Sem-C $\uparrow$ KLD $\downarrow$ Pref (%) $\uparrow$ SD $34.41$ $0.671$ $1.2953$ $11.56$ SDXL $36.86$ $0.681$ $1.2118$ $17.56$ RPG $44.17$ $0.678$ $1.2171$ $15.20$ PixArt $45.96$ $0.677$ $1.2583$ $12.40$ EmoGen $57.31$ $0.698$ $2.1436$ $9.16$ Ours (AIEdiT) $\mathbf{33.80}$ $\mathbf{0.705}$ $\mathbf{1.2004}$ $\mathbf{34.12}$

![Refer to caption](https://arxiv.org/html/2505.18699v1/x11.png)

Figure 11: Visual quality comparisons with relevant image generation methods.

### 6.2 Performance on Image Generation Task

AIEdiT can also function as an affective image generator by editing random Gaussian noise. To demonstrate its effectiveness in generating affective visual elements, we provide additional comparisons with relevant image generation methods, including general image generation methods (*e.g.*, SD [^45], SDXL [^40], RPG [^64], PixArt [^5]) and affective image generation method (EmoGen [^62]).

Qualitative comparisons. We present the qualitative comparison results in Fig. 11. General text-driven image generation methods [^5] [^40] [^45] [^64] struggle to translate visually-abstract emotional requests into visually-concrete semantic representations (*e.g.*, Fig. 11 third row, conveying a sense of warmth and happiness). On the other hand, although affective image generation method [^62] can produce images to evoke specific emotions, it relies on coarse-grained emotion categories, still facing challenges in generating images with specific instances and nuanced emotional expressions (*e.g.*, Fig. 11 fourth row, generating a brown dog with a discomforted expression). Instead, our AIEdiT allows users to specify the instance and accurately shape corresponding emotional factors.

Quantitative comparisons. We present quantitative metrics to evaluate the performance of relevant image generation methods on the EmoTIPS dataset. As reported in Table 5, AIEdiT achieves the highest scores across all metrics, demonstrating its ability to create photo-realistic results (FID) with clearly recognizable visual content (Sem-C) that accurately reflect users’ emotional requests (KLD).

![Refer to caption](https://arxiv.org/html/2505.18699v1/x12.png)

Figure 12: Visualization of failure cases.

### 6.3 Failure Case

Although AIEdiT is tailored for affective image editing, it still struggles to interpret complex text descriptions that include mixed emotions with multiple elements. To illustrate this, we present two failure cases in Fig. 12. Specifically, AIEdiT has difficulty editing flowers to convey both hope and melancholy (Fig. 12, left). Similarly, it struggles to effectively render both the boat and lake to express joy and sorrow (Fig. 12, right).

## 7 Conclusion

In this paper, we introduce AIEdiT, a method for Affective Image Editing using Text descriptions. AIEdiT introduces two key components: the emotional spectrum that represents universal emotional priors, and the emotional mapper that translates users’ emotional requests into visual semantic representations. Leveraging an MLLM as a supervisor along with an appropriate sampling strategy, AIEdiT can manipulate emotional factors across the entire image according to users’ instructions. Extensive experiments demonstrate the effectiveness of our approach. Additionally, we contribute the EmoTIPS dataset, which provides Emotion-aligned Text and Image Pair Sets, further benefiting the community.

Limitation. Our approach remains limited by the current capabilities of MLLMs in understanding subjective and ambiguous emotions. Although we have adopted three validation criteria to filter out unqualified samples, as shown in Table 1, there are still fewer than 1.72% failed ratings and 1.76%-7.28% borderline ratings. While our proposed AIEdiT demonstrates robustness in understanding nuanced emotional requests, these unqualified samples may still add noise, potentially impacting the accuracy of emotional understanding. We believe this limitation could be further alleviated with the continued advancement of MLLMs.

## Acknowledgement

This work was supported in part by National Natural Science Foundation of China (Grant No. 62136001, 62088102, and U23B2052), National Science and Technology Major Project (Grant No. 2021ZD0109803), Program for Youth Innovative Research Team of BUPT (Grant No. 2023YQTD02).

[^1]: Achlioptas P, Ovsjanikov M, Haydarov K, Elhoseiny M, Guibas LJ (2021) Artemis: Affective language for visual art. In: Proc. of Computer Vision and Pattern Recognition

[^2]: Anand S, Devulapally NK, Bhattacharjee SD, Yuan J (2023) Multi-label emotion analysis in conversation via multimodal knowledge distillation

[^3]: Brooks T, Holynski A, Efros AA (2023) InstructPix2Pix: Learning to follow image editing instructions. In: Proc. of Computer Vision and Pattern Recognition

[^4]: Chang Z, Weng S, Zhang P, Li Y, Li S, Shi B, et al (2023) L-CAD: Language-based colorization with any-level descriptions using diffusion priors. Proc of Advances in Neural Information Processing Systems

[^5]: Chen J, Yu J, Ge C, Yao L, Xie E, Wu Y, Wang Z, Kwok J, Luo P, Lu H, et al (2023) PixArt- $\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:231000426

[^6]: Chen L, Li J, Dong X, Zhang P, He C, Wang J, Zhao F, Lin D (2024) ShareGPT4V: Improving large multi-modal models with better captions. In: Proc. of European Conference on Computer Vision

[^7]: Cheng D, Gong Y, Zhou S, Wang J, Zheng N (2016) Person re-identification by multi-channel parts-based cnn with improved triplet loss function. In: CVPR

[^8]: Chowdhury RM, Olsen GD, Pracejus JW (2008) Affective responses to images in print advertising: Affect integration in a simultaneous presentation context. Journal of Advertising

[^9]: Cohn JF, Kanade T (2007) Use of automated facial image analysis for measurement of emotion expression. Handbook of emotion elicitation and assessment

[^10]: Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L (2009) ImageNet: A large-scale hierarchical image database. In: Proc. of Computer Vision and Pattern Recognition

[^11]: Devlin J, Chang MW, Lee K, Toutanova K (2019) BERT: Pre-training of deep bidirectional transformers for language understanding. In: North American Chapter of the Association for Computational Linguistics

[^12]: Fu TJ, Hu W, Du X, Wang WY, Yang Y, Gan Z (2024) Guiding instruction-based image editing via multimodal large language models

[^13]: He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image recognition. In: Proc. of Computer Vision and Pattern Recognition

[^14]: Hertz A, Mokady R, Tenenbaum J, Aberman K, Pritch Y, Cohen-Or D (2023) Prompt-to-prompt image editing with cross attention control

[^15]: Heusel M, Ramsauer H, Unterthiner T, Nessler B, Hochreiter S (2017) GANs trained by a two time-scale update rule converge to a local nash equilibrium. Proc of Advances in Neural Information Processing Systems

[^16]: Ho J, Jain A, Abbeel P (2020) Denoising diffusion probabilistic models. Proc of Advances in Neural Information Processing Systems

[^17]: Ito T, Tsubouchi K, Sakaji H, Yamashita T, Izumi K (2020) Word-level contextual sentiment analysis with interpretability. In: Proc. of the AAAI Conference on Artificial Intelligence

[^18]: Kim H, Somerville LH, Johnstone T, Alexander AL, Whalen PJ (2003) Inverse amygdala and medial prefrontal cortex responses to surprised faces. Neuroreport

[^19]: Kingma DP, Ba J (2014) Adam: A method for stochastic optimization. arXiv preprint arXiv:14126980

[^20]: Kojima T, Gu SS, Reid M, Matsuo Y, Iwasawa Y (2022) Large language models are zero-shot reasoners. Proc of Advances in Neural Information Processing Systems

[^21]: Kullback S, Leibler RA (1951) On information and sufficiency. The Annals of Mathematical Statistics

[^22]: Kumari N, Zhang B, Zhang R, Shechtman E, Zhu JY (2023) Multi-concept customization of text-to-image diffusion. In: Proc. of Computer Vision and Pattern Recognition

[^23]: Kundu T, Saravanan C (2017) Advancements and recent trends in emotion recognition using facial image analysis and machine learning models. In: International Conference on Electrical, Electronics, Communication, Computer, and Optimization Techniques

[^24]: Li C, Wang J, Zhang Y, Zhu K, Hou W, Lian J, Luo F, Yang Q, Xie X (2023a) Large language models understand and can be enhanced by emotional stimuli. arXiv preprint arXiv:230711760

[^25]: Li J, Li D, Savarese S, Hoi S (2023b) BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In: Proc. of International Conference on Machine Learning

[^26]: Li Z, Chen G, Shao R, Jiang D, Nie L (2024) Enhancing the emotional generation capability of large language models via emotional chain-of-thought. arXiv preprint arXiv:240106836

[^27]: Lin Q, Zhang J, Ong YS, Zhang M (2024) Make me happier: Evoking emotions through image diffusion models. arXiv preprint arXiv:240308255

[^28]: Liu H, Li C, Wu Q, Lee YJ (2024a) Visual instruction tuning. Proc of Advances in Neural Information Processing Systems

[^29]: Liu S, Zhang X, Yang J (2022) SER30K: A large-scale dataset for sticker emotion recognition

[^30]: Liu Z, Yang K, Zhang T, Xie Q, Yu Z, Ananiadou S (2024b) EmoLLMs: A series of emotional large language models and annotation tools for comprehensive affective analysis. arXiv preprint arXiv:240108508

[^31]: Meng C, He Y, Song Y, Song J, Wu J, Zhu JY, Ermon S (2022) SDEdit: Guided image synthesis and editing with stochastic differential equations

[^32]: Mikels JA, Fredrickson BL, Larkin GR, Lindberg CM, Maglio SJ, Reuter-Lorenz PA (2005) Emotional category data on images from the international affective picture system. Behavior Research Methods

[^33]: Mohamed Y, Khan FF, Haydarov K, Elhoseiny M (2022) It is okay to not be okay: Overcoming emotional bias in affective image captioning by contrastive data collection. In: Proc. of Computer Vision and Pattern Recognition

[^34]: Mohammad S (2018) Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 english words

[^35]: Mokady R, Hertz A, Aberman K, Pritch Y, Cohen-Or D (2023) Null-text inversion for editing real images using guided diffusion models. In: Proc. of Computer Vision and Pattern Recognition

[^36]: Neta M, Norris CJ, Whalen PJ (2009) Corrugator muscle responses are associated with individual differences in positivity-negativity bias. Emotion

[^37]: Neta M, Berkebile MM, Freeman JB (2021) The dynamic process of ambiguous emotion perception. Cognition and Emotion

[^38]: Parmar G, Kumar Singh K, Zhang R, Li Y, Lu J, Zhu JY (2023) Zero-shot image-to-image translation. In: Proc. of ACM SIGGRAPH

[^39]: Paskaleva R, Holubakha M, Ilic A, Motamed S, Van Gool L, Paudel D (2024) A unified and interpretable emotion representation and expression generation. In: Proc. of Computer Vision and Pattern Recognition

[^40]: Podell D, English Z, Lacey K, Blattmann A, Dockhorn T, Müller J, Penna J, Rombach R (2023) SDXL: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:230701952

[^41]: Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, et al (2021) Learning transferable visual models from natural language supervision. In: Proc. of International Conference on Machine Learning

[^42]: Rao T, Xu M, Liu H, Wang J, Burnett I (2016) Multi-scale blocks based image emotion classification using multiple instance learning. In: Proc. of International Conference on Image Processing

[^43]: Rao T, Li X, Zhang H, Xu M (2019) Multi-level region-based convolutional neural network for image emotion classification. Neurocomputing

[^44]: Rao T, Li X, Xu M (2020) Learning multi-level deep representations for image emotion classification. Neural processing letters

[^45]: Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B (2022) High-resolution image synthesis with latent diffusion models. In: Proc. of Computer Vision and Pattern Recognition

[^46]: Ruiz N, Li Y, Jampani V, Pritch Y, Rubinstein M, Aberman K (2023) DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proc. of Computer Vision and Pattern Recognition

[^47]: Saharia C, Chan W, Saxena S, Li L, Whang J, Denton EL, Ghasemipour K, Gontijo Lopes R, Karagol Ayan B, Salimans T, et al (2022) Photorealistic text-to-image diffusion models with deep language understanding. Proc of Advances in Neural Information Processing Systems

[^48]: Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, et al (2022) LAION-5B: An open large-scale dataset for training next generation image-text models. Proc of Advances in Neural Information Processing Systems

[^49]: Simonyan K, Zisserman A (2014) Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:14091556

[^50]: Song J, Meng C, Ermon S (2020) Denoising diffusion implicit models

[^51]: Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z (2016) Rethinking the inception architecture for computer vision. In: Proc. of International Conference on Computer Vision

[^52]: Touvron H, Martin L, Stone K, Albert P, Almahairi A, Babaei Y, Bashlykov N, Batra S, Bhargava P, Bhosale S, et al (2023) Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:230709288

[^53]: Uhrig MK, Trautmann N, Baumgärtner U, Treede RD, Henrich F, Hiller W, Marschall S (2016) Emotion elicitation: A comparison of pictures and films. Frontiers in Psychology

[^54]: Wang L, Jia G, Jiang N, Wu H, Yang J (2022) Ease: Robust facial expression recognition via emotion ambiguity-sensitive cooperative networks

[^55]: Wang X, Jia J, Yin J, Cai L (2013) Interpretable aesthetic features for affective image classification. In: Proc. of International Conference on Image Processing

[^56]: Weng S, Zhang P, Chang Z, Wang X, Li S, Shi B (2023) Affective image filter: Reflecting emotions from text to images. In: Proc. of International Conference on Computer Vision

[^57]: Xie S, Zhang Z, Lin Z, Hinz T, Zhang K (2023) SmartBrush: Text and shape guided object inpainting with diffusion model. In: Proc. of Computer Vision and Pattern Recognition

[^58]: Xu T, Zhang P, Huang Q, Zhang H, Gan Z, Huang X, He X (2018) AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks. In: Proc. of Computer Vision and Pattern Recognition

[^59]: Yang D, Chen Z, Wang Y, Wang S, Li M, Liu S, Zhao X, Huang S, Dong Z, Zhai P, Zhang L (2023a) Context de-confounded emotion recognition. In: Proc. of Computer Vision and Pattern Recognition

[^60]: Yang J, Li J, Wang X, Ding Y, Gao X (2021) Stimuli-aware visual emotion analysis. IEEE Transactions on Image Processing 30:7432–7445

[^61]: Yang J, Huang Q, Ding T, Lischinski D, Cohen-Or D, Huang H (2023b) EmoSet: A large-scale visual emotion dataset with rich attributes. In: Proc. of International Conference on Computer Vision

[^62]: Yang J, Feng J, Huang H (2024a) EmoGen: Emotional image content generation with text-to-image diffusion models. In: Proc. of Computer Vision and Pattern Recognition

[^63]: Yang J, Feng J, Luo W, Lischinski D, Cohen-Or D, Huang H (2025) Emoedit: Evoking emotions through image manipulation. In: Proc. of Computer Vision and Pattern Recognition

[^64]: Yang L, Yu Z, Meng C, Xu M, Ermon S, Cui B (2024b) Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In: Proc. of International Conference on Machine Learning

[^65]: Yang Z, Li L, Lin K, Wang J, Lin CC, Liu Z, Wang L (2023c) The dawn of LMMs: Preliminary explorations with GPT-4V (ision). arXiv preprint arXiv:230917421

[^66]: Yao X, She D, Zhang H, Yang J, Cheng MM, Wang L (2020) Adaptive deep metric learning for affective image retrieval and classification. IEEE Transactions on Multimedia

[^67]: You Q, Luo J, Jin H, Yang J (2016) Building a large scale dataset for image emotion recognition: The fine print and the benchmark. In: Proc. of the AAAI Conference on Artificial Intelligence

[^68]: Zabari N, Azulay A, Gorkor A, Halperin T, Fried O (2023) Diffusing colors: Image colorization with text guided diffusion. In: Proc. of ACM SIGGRAPH Asia

[^69]: Zhang H, Xu T, Li H, Zhang S, Wang X, Huang X, Metaxas DN (2017) StackGAN: Text to photo-realistic image synthesis with stacked generative adversarial networks. In: Proc. of International Conference on Computer Vision

[^70]: Zhang K, Mo L, Chen W, Sun H, Su Y (2024) MagicBrush: A manually annotated dataset for instruction-guided image editing. Proc of Advances in Neural Information Processing Systems

[^71]: Zhang L, Wang S, Liu B (2018) Deep learning for sentiment analysis: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery

[^72]: Zhang L, Rao A, Agrawala M (2023) Adding conditional control to text-to-image diffusion models. In: Proc. of International Conference on Computer Vision

[^73]: Zhang W, Li X, Deng Y, Bing L, Lam W (2022) A survey on aspect-based sentiment analysis: Tasks, methods, and challenges. IEEE Transactions on Knowledge and Data Engineering

[^74]: Zhao S, Gao Y, Jiang X, Yao H, Chua TS, Sun X (2014a) Exploring principles-of-art features for image emotion recognition

[^75]: Zhao S, Yao H, Yang Y, Zhang Y (2014b) Affective image retrieval via multi-graph learning

[^76]: Zhao S, Yao X, Yang J, Jia G, Ding G, Chua TS, Schuller BW, Keutzer K (2021) Affective image content analysis: Two decades review and new perspectives

[^77]: Zhou B, Lapedriza A, Khosla A, Oliva A, Torralba A (2017) Places: A 10 million image database for scene recognition