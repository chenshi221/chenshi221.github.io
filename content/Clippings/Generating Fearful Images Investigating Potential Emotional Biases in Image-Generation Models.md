---
title: "Generating Fearful Images: Investigating Potential Emotional Biases in Image-Generation Models"
source: "https://arxiv.org/html/2411.05985v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
---
Maneet Mehta Independent ResearcherUSA and Cody Buntain University of MarylandCollege Park, MDUSA

###### Abstract.

This paper examines potential biases and inconsistencies in the emotions evoked by images produced by generative artificial intelligence (AI) models and their potential bias toward negative emotions. We assess this bias by comparing the emotions evoked by an AI-produced image to the emotions evoked by prompts used to create those images. As a first step, the study evaluates three approaches for identifying emotions in images—traditional supervised learning, zero-shot learning with vision-language models, and cross-modal auto-captioning—trained on and compared with a large dataset of image-emotion annotations that categorizes images across eight emotional types. Results show fine-tuned models, particularly Google’s Vision Transformer (ViT), significantly outperform zero-shot and caption-based methods in recognizing emotions in images. For a cross-modality comparison, we then analyze the differences between emotions in text prompts—via existing text-based emotion-recognition models—and the emotions evoked in the resulting images. Findings indicate that AI-generated images frequently produce images with negative emotions, particularly fear, regardless of the original prompt. This emotional skew in generative models could amplify negative affective content in digital spaces, perpetuating its prevalence and impact. The study advocates for a multidisciplinary approach to better align AI emotion recognition with psychological insights and address potential biases in generative AI outputs across digital media.

images, emotion, generative ai, bias

## 1\. Introduction

The information environment increasingly relies on visual media, with visual-first platforms like YouTube, Instagram, Pinterest, and TikTok comprising four of the top five most-used social media platforms in the US [^17]. In studying these online spaces, the academic consensus has increasingly solidified around two findings: 1) the presence of visual media increases online engagement [^2] [^14], and 2) highly emotional content—especially negative—also increases this engagement [^34] [^28]. At the same time, generative AI models’ ever-growing hunger for training data has lead to increased use of social media and creator-content as training data for modern large- and visual-language models. Together, these factors have the potential to produce an “unvirtuous cycle” between generative AI models and the information environment from which large volumes of training data are collected. In this cycle, content creators produce more emotionally evocative visual media to gain more engagement, generative AI systems then over-represent negative emotions in the media they produce, which then feeds back into the information environment, receives more engagement, gains a larger share of visual media we see, and gets ingested in the next round of training generative-AI systems.

This paper begins an investigation of this unvirtuous cycle by developing scalable emotion-recognition methods for images and comparing these emotions to those present in the underlying prompts. To support the emotion-recognition task, we evaluate three approaches: traditional supervised methods using pre-trained computer-vision models, zero-short learning with state-of-the-art vision-language models, and cross-modal auto-captioning approaches. Leveraging the EmoSet dataset [^39], which contains manually labeled image-emotion pairs for 118,102 images, we demonstrate that fine-tuning computer vision models—particularly Google’s Vision Transformer (ViT) [^11] —substantially outperforms zero-shot and auto-captioning approaches, with textual auto-captioning approaches performing particularly poorly in comparison. Using this fine-tuned ViT model coupled with a state-of-the-art model for emotion recognition in text [^6] [^5], we compare the emotions present in a text-based generative-AI prompt to the emotions evoked by the produced image. Ideally, the distributions of emotion present in generated images should mirror the distribution in the underlying prompt used to generate this image. We suspect, however, that this unvirtuous cycle between generative AI and the information environment instead produces an anti-social outcome: that generative-AI systems are biased toward producing images that evoke negative emotions regardless of the underlying prompt.

Results show that, for the task of emotion recognition in images, fine-tuning off-the-shelf computer vision models, particularly Google’s ViT ($F_{1}=0.7343$), substantially outperforms zero-shot learning using vision-language models (best $F_{1}=0.3150$) and automated captioning approaches (best $F_{1}=0.2154$)—zero-shot and auto-captioning are much closer to each other performance-wise. Then, to assess the potential biases between the emotions evoked by AI-generated images and the text prompts that are used to generate them, we apply these models to image-prompt pairs from [^37]. We find, for at least one popular generative AI text-to-image model (i.e., StableDiffusion), the emotions evoked by an image do appear more negative than their source prompts, where the StableDiffusion images from the dataset substantially over-represent fear, whereas the prompts primarily show excitement. To extend this analysis beyond one generative model, we also sample 200 prompts from the DiffusionDB dataset and use GPT-4o to generate new images using state-of-the-art generative processes; our results are consistent for this subset as well, where GPT-4o images also over-represent fear relative to the source prompts. These results yield evidence of the concerning hypothesis above, namely that a AI-based image-generation models may nudge their users and audiences toward negative emotions. This finding is particularly concerning given the state of our modern information environment, where we know negative emotions are contagious [^20]. Hence, we advocate for a multidisciplinary approach to better align AI emotion recognition with psychological insights and address potential biases in generative AI outputs across digital media.

## 2\. Related Work

This investigation builds on work from two main communities: scholars of fairness and bias in AI systems, and computational social science scholars of the information environment. From the first community—which includes many aspects of AI evaluation and auditing—major concerns have arisen over the potential for unintended harms caused by AI systems, such as the unvirtuous cycle we outline. [^33] frames such concerns as an “accountability gap” between system development and deployment, where such systems are generally not evaluated until after they have been deployed and have potentially “already negatively impacted users.” Many instances of related work have focused on gender and ethnic biases in generative systems, such as findings outlined in [^8], where “generated images included a disproportionately high proportion of white male medical students” compared to the actual population. Likewise, the study of DALL-E in [^4] demonstrates pervasive gender and racial biases across depictions of occupations. These issues are not limited to image generation either, as [^7] illustrates language-specific biases as well. Hence, strong academic consensus exists concerning generative AI’s potential “to reproduce, exacerbate, and reinforce extant human social biases” [^4].

To date, however, few studies have examined how these systems are biased in the kind of *emotional* content they produce. Worryingly, [^20] shows how negative emotions spread in online spaces, such that increasing the supply and exposure to negative emotional content can impact the psychological well-being of those exposed *and* downstream audiences. This effect and others concerning emotion in the modern information environment are complex, covering a wide variety of fields. Journalism has reckoned with the role of emotion in its field, as [^19] describes an erosion of traditional objectivity-focused journalism in the face of crisis reporting, and [^36] further advances this shift as part of a reaction to the new media environment, where audiences are generally more emotionally engaged. Similarly, political science has long engaged with the role of emotion in politics, with multiple competing theories driving continued research [^23]. Understanding the intersection of emotion and political science in the modern, online information environment has increasingly garnered attention as well, with new findings on the role of emotion in information quality and mobilization. [^2], for example, finds online audiences are particularly mobilized to engage in protest when exposed to images of their friends and images that evoke enthusiasm and fear.

Further, studies of information quality in online spaces increasingly point to the role of emotion in mediating consumption of high-quality content. We have strong evidence that emotion drives online engagement with political content [^28], and political elites benefit from highly hostile and emotionally engaging content [^34]. [^24] further demonstrates how emotionality increases susceptibility to uptake of low-quality content.

Given these biases toward and effects of negative emotions, our hypothesis that generative AI systems trained using the vast volumes of online, digital-trace data would, as [^4] suggests, “reproduce, exacerbate, and reinforce” this anti-social bias. Understanding—and by extension controlling—this bias is critical to advancing solutions to the AI accountability gap [^33] describes, while also necessary for enhancing the quality and resiliency of our information space.

## 3\. Methods

### 3.1. Datasets for Images, Emotion, and AI Prompts

For model finetuning and initial evaluation, we leverage the EmoSet dataset. Each of the 118,102 images in the multi-class dataset is annotated as one of 8 discreet emotions: Amusement, Awe, Contentment, Excitement, Anger, Disgust, Fear, or Sadness.

To evaluate our central hypothesis that generative AI systems are biased towards negative emotions, we leverage the DiffusionDB dataset: a text-to-image prompt dataset containing 14 million images generated by Stable Diffusion using prompts and hyperparameters specified by real users. We evaluate the salience of emotions within both the prompts and images contained in the 2m-first-5k subset of this dataset.

### 3.2. Emotion Identification in Images

We seek to explore and evaluate different approaches for identifying emotions in AI-generated images, aiming to find the most effective strategies for capturing complex emotional cues. To this end, we leveraged three distinct mechanisms, each targeting a different aspect of emotion recognition. First, we employed zero-shot image classification models (BLIP [^21], CLIP [^32], and ALBEF) to establish a baseline without task-specific fine-tuning. These models were chosen to assess the potential of large-scale pre-trained models to generalize across a wide variety of emotional contexts. Second, we fine-tuned image classification models such as Google ViT, Microsoft SWIN Transformer, and ConvNeXT [^11] [^22] [^38], focusing on training these models on specialized emotion datasets to enhance their ability to recognize subtle and context-specific emotions. Finally, we implemented an auto-captioning pipeline using BLIP-2 and GPT-4 Vision to generate detailed captions for each image, which were then analyzed by text-based emotion classifiers to evaluate whether adding contextual information could improve emotion recognition accuracy. This presents insights into the curious empirical question of how emotion insight/characteristics are transferred across modalities

To rigorously assess the effectiveness of each approach, we measured macro-average precision, macro-average recall, and macro-average F1 scores. These metrics allowed us to evaluate not only how accurately each model predicted emotions but also how well they generalized across different emotional categories. This three-pronged methodology provided a comprehensive analysis, allowing us to compare the strengths and weaknesses of each approach and gain insights into which strategies were most effective for capturing the diverse range of emotions present in AI-generated images. By combining zero-shot models, fine-tuned classifiers, and a hybrid captioning approach, we aimed to explore the interplay between visual features and textual context in emotion recognition tasks.

#### 3.2.1. Zero-Shot Learning with Vision-Language Models

As an initial step in our study, we employed zero-shot image classification models—BLIP, CLIP, and ALBEF—from the LaViS (Language and Vision) toolkit to establish benchmark results for emotion recognition in AI-generated images. These models were selected for their ability to perform image-text alignment tasks without requiring task-specific fine-tuning, allowing us to rapidly assess the baseline capabilities of general-purpose vision-language models in capturing emotional content.

CLIP (Contrastive Language-Image Pretraining), developed by OpenAI, is trained on a vast dataset of image-text pairs to learn a shared embedding space for images and text. By measuring the similarity between an image and textual descriptions, CLIP can perform zero-shot classification by selecting the label whose text best matches the image. This makes it a powerful tool for tasks requiring semantic understanding without additional training.

ALBEF (Align Before Fuse) enhances image-text alignment through contrastive learning before fusing the modalities for downstream tasks. By aligning visual and textual representations prior to fusion, ALBEF captures nuanced cross-modal relationships, making it effective for tasks like image-text retrieval and, in our case, preliminary emotion recognition.

BLIP (Bootstrapping Language-Image Pre-training) focuses on bootstrapping the learning of vision-language representations using large-scale, noisy web data. It employs a bootstrapping approach that iteratively refines the model’s predictions, improving its understanding of the relationship between images and text. BLIP is versatile for tasks such as image captioning and zero-shot image classification, bridging the gap between vision and language without extensive fine-tuning.

By utilizing these models, we aimed to quickly evaluate the feasibility of extracting emotional content from images using models that leverage large-scale pre-training on diverse datasets. Although these zero-shot models performed less effectively than the fine-tuned image classifiers, their use was instrumental in establishing initial benchmark results. They highlighted the limitations of applying general-purpose models to specialized tasks like emotion recognition, emphasizing the need for models trained on domain-specific data. This initial benchmarking informed our subsequent decision to fine-tune more specialized models.

#### 3.2.2. Fine-Tuning Vision Models for Classification

In this study, we selected Google’s Vision Transformer, ConvNeXT, and Microsoft’s Swin Transformer for fine-tuning on the task of extracting emotions from AI-generated images. These models were chosen due to their diverse architectural designs, strong performance on benchmark datasets, and potential to capture the nuanced features associated with emotional content. By selecting these three models, we aimed to explore a spectrum of architectural paradigms—pure transformers (ViT), modernized CNNs (ConvNeXT), and hybrid architectures (Swin)—to assess their effectiveness in extracting emotions from images.

The Vision Transformer (ViT) applies the transformer architecture to computer vision by treating images as sequences of patches. This allows ViT to leverage self-attention mechanisms to capture global context and long-range dependencies within an image, which is crucial for recognizing emotions that emerge from the overall composition and interrelationships of different regions.

ConvNeXT was selected as a modernized convolutional neural network (CNN) that incorporates design principles from transformers while retaining the efficiency of traditional CNNs. Its convolutional layers excel at capturing local features and fine-grained details, essential for identifying subtle emotional cues conveyed through textures and small visual elements.

The Swin Transformer introduces a hierarchical architecture with shifted windows, enabling it to process images at multiple scales. This multi-scale representation is particularly useful for capturing both coarse and fine details associated with different emotional expressions. Its efficient computation balances modeling power with practical resource usage, making it suitable for complex tasks like emotion recognition.

This diversity in design allows us to evaluate how different approaches handle the emotion extraction task, leveraging their unique strengths. Additionally, all three models have demonstrated state-of-the-art performance on image classification benchmarks, indicating their strong feature extraction capabilities essential for our study. The availability of pre-trained weights for these models facilitates effective fine-tuning on our specialized dataset, which is crucial given the complexity of emotion recognition. Furthermore, the combination of global context modeling (ViT), local feature extraction (ConvNeXT), and multi-scale representation (Swin Transformer) provides a comprehensive toolkit for capturing the multifaceted and subtle nature of emotional content in images.

After fine-tuning, we gather macro-precision, macro-recall, and macro-f1 scores for each base model architecture and select the one with the highest macro f1.

#### 3.2.3. Auto-Captioning Models

To test cross-modality emotion capture, we implement an auto-captioning pipeline to generate textual descriptions of images, which were then analyzed using textual emotion classification models (results evaluated on the DeMuX/MeMO [^6] [^5] model with Semeval [^27], Paletz [^29], and GoEmotions [^10] models, explained in the following section). We selected BLIP-2 and GPT-4 Vision as our captioning models for this task. The choice of these models was driven by their advanced capabilities in image captioning and their potential to enhance emotion recognition through detailed and context-rich descriptions.

By incorporating BLIP-2 and GPT-4 Vision into our auto-captioning pipeline, we aimed to leverage their respective strengths in generating high-quality image captions to enhance emotion recognition accuracy. BLIP-2’s proven effectiveness in zero-shot tasks and its upgraded capabilities made it a suitable choice for autonomous caption generation. In contrast, GPT-4 Vision’s advanced vision-language integration and the use of a specific prompt allowed us to explore the impact of detailed and exhaustive captions on emotion classification outcomes. This strategic selection enabled us to investigate how the quality and depth of image captions influence the performance of textual emotion classification models, contributing valuable insights to the fields of affective computing and multimodal analysis.

Our decision to include BLIP-2 was influenced by the strong performance of its predecessor, BLIP, in our initial zero-shot model tests for emotion recognition. BLIP demonstrated superior ability among zero-shot models to capture relevant features associated with emotional content in images. Building on this success, we opted to use BLIP-2, the upgraded version, to leverage its improved performance and enhanced capabilities in image captioning.

BLIP-2 excels in generating descriptive and contextually rich captions without the need for supplementary prompts, allowing for an unbiased and autonomous interpretation of the visual content. Its bootstrapping approach iteratively refines predictions, improving the model’s understanding of complex visual cues, including those related to emotions. By utilizing BLIP-2 in our pipeline, we aimed to generate accurate captions that effectively represent the emotional nuances present in the images, providing a solid foundation for subsequent textual emotion classification.

We also selected GPT-4 Vision for its state-of-the-art vision capabilities and proficiency in generating comprehensive and detailed image captions. GPT-4 Vision integrates advanced language understanding with visual processing, enabling it to produce exhaustive descriptions that capture subtle and complex aspects of the image content.

Our rationale for including GPT-4 Vision was to assess whether the exhaustiveness and detail in the captions would significantly impact the accuracy of emotion recognition when analyzed by textual emotion classification models. Given that GPT-4 Vision is at the forefront of current technology, we anticipated that its ability to generate more detailed and nuanced captions would enhance the detection of emotional content by providing richer context and capturing subtle visual cues associated with different emotions.

To focus the captioning process on emotional aspects, we provided GPT-4 Vision with the prompt: ”Generate a descriptive caption for this image that can help a model identify the emotion present.” This directive was intended to guide the model toward highlighting elements relevant to emotion recognition, potentially improving the performance of the subsequent classification models.

##### Down-Sampling to Minimize GPT Costs

Restricted by API pricing, we opted to only caption 200 images rather than the full validation subset that BLIP2 captioned, allowing us to cost-effectively generate benchmarks.

#### 3.2.4. Validating Automatic Emotion Recognition in Images

To assess the validity of our highest-performing emotion-recognition model for visual media, we follow a method outlined in [^9]. We sample $n=200$ image-label pairs from our automatically labeled images and assess whether we agree (A), agree with doubt (D), or disagree (N) with the emotion label. From these measures, we compute the acceptance rate as $(A+D)/(A+D+N)$. For comparison, we also measure this acceptance rate against the original EmoSet dataset. For EmoSet, we see an acceptance rate = $0.96$, and for Google’s ViT, we see $0.86$, lower but still quite high.

### 3.3. Emotion Identification in Text

Two points in this research require emotion extraction from text. Firstly, our Auto-captioning pipelines utilize visual image captioning tools to caption a series of images. Those captions are then processed through text-emotion extraction models. Secondly, to extract emotion distributions from the prompts in the DiffusionDB dataset, which we later compare with emotion distributions from the corresponding images.

We leverage three models from the DeMuX-MEmo models to classify text into discrete emotions due to its state-of-the-art ability to handle complex, multi-label emotion recognition tasks. These models were specifically designed to leverage the relationships between emotions, making it an ideal choice for classifying the emotional content present in text. We specifically test models pretrained on the Semeval-2018, Paletz, and GoEmotions benchmarks.

One of the core reasons for selecting DeMuX-MEmo is its superior performance in multi-label settings, where more than one emotion may be present in a single piece of text. Unlike traditional models that treat each label independently, DeMuX-MEmo integrates the correlations between emotions, understanding that certain emotions often co-occur. For instance, emotions like fear and sadness are frequently linked, and recognizing such connections is critical for accurate emotion classification. This capability is particularly valuable in our work, where texts often convey complex emotional states, rather than single, isolated emotions.

DeMuX-MEmo also includes a regularization mechanism that leverages both global and local emotion correlations. By understanding these relationships, the model reduces the risk of making contradictory predictions. For example, if a text is classified as expressing both ”joy” and ”sadness,” the model will use its learned correlations to resolve these conflicting predictions more intelligently than models without this capability. This level of sophistication is essential for our work, where subtle emotional nuances are common.

### 3.4. Comparing Emotion Across Modalities

Assessing whether substantial bias exists in emotions evoked by AI-generated images compared to those that are evoked by the prompts that create them is a key motivation for this work. Directly measuring this divergence is problematic, however, given the different emotions present in the EmoSet dataset versus those in our text-based methods (e.g., the GoEmotion-based pretrained DeMuX-MEmo model includes “joy” and “surprise”, which are not present in the EmoSet labels, and conversely, EmoSet includes “awe”, which is not present in the GoEmotion pretrained dataset). Consequently, we measure two related values: First, we assess Spearman’s $\rho$ rank correlation between emotions in images and the emotions in text, following the validation process outlined in [^29]. In this assessment, we expect positive and significant correlations among related emotions.

As our second evaluation, we directly compare the prevalence of positive and negative emotions present in each modality provided by the [^37] dataset. If our central hypothesis about potential biases toward negative emotions among generative AI systems, the DiffusionDB dataset should over-represent negative emotions in the images compared to text.

## 4\. Results

### 4.1. Emotion Classification in Images

Table 1 shows results for the emotion-recognition task across our three learning modalities—via multimodal zero-shot models, fine-tuned computer vision models, and automatic image captioning. The zero-shot models in Table 1(a)—BLIP-base, CLIP-ViT-B-16, and ALBEF-base—served as our initial benchmark for emotion recognition without fine-tuning. Across all three models, the overall macro-averaged F1-scores were notably low, indicating limited effectiveness in generalizing emotional cues from the images. Notably, some models demonstrated a tendency to heavily over-predict particular emotions, as evidenced by high class recall but poor class precision for those emotions (emotion recognition performance is provided in the supplemental material). For example, CLIP-ViT-B-16 achieved a recall of 0.722 for “Amusement” but had near-zero recall for “Contentment” and “Disgust”. The model identified a disproportionately high number of images as “Amusement” while struggling to recognize other emotions accurately.

The fine-tuned image classifiers in Table 1(b)—Google’s ViT, Microsoft’s SWIN Transformer, and ConvNeXT—performed significantly better in recognizing emotions than the zero-shot models. These results suggest task-specific training greatly enhanced their ability to capture the diverse range of emotions present in images. Among these models, Google ViT emerged as the strongest performer, with an overall macro-averaged F1-score higher than all other methods.

Lastly, Table 1(c) shows results of the novel use of auto-captioning methods to translate images into text-based captions, where we can leverage the many text-based emotion-recognition models. To assess performance in emotion extraction across the EmoSet dataset, we primarily relied on BLIP-2 (Table 1(c)), as we could run this model on-premises across the full dataset. To assess performance across a variety of text-based emotion-recognition models, we make use of three pre-trained configurations in the Demux-MEmo [^6] package, each of which is trained on a separate text-based emotion dataset: GoEmotions [^10], [^29], and SemEval [^27]. Regardless of the selected pre-trained model configuration, however, we see that auto-captioning is less performant than the zero-shot multimodal models.

#### 4.1.1. Comparing BLIP2 versus GPT-4o

To investigate potential performance implications from our selection of BLIP2 for automatic captioning, we also generated a sample of image captions using the more expensive and performance GPT-4o interface. Results using GPT-4o are shown in Table 2, where we see that the pre-trained version of Demux-MEmo on the GoEmotions data outperforms the equivalent model using BLIP-2, but the results are not sufficiently higher compared to the zero-shot model to change our approach. This difference in performance may be attributable to more detailed captions compared to BLIP-2. This outcome suggests that the increased descriptiveness of GPT-4o’s captions, which were expected to provide more context and detail for emotion classification, did not lead to a substantial improvement in performance. Looking within specific emotions for instance, the F1-scores for key emotions like “Excitement” and “Fear” were similar across both pipelines. The comparable performance between the BLIP-2 and GPT-4o pipelines indicate that the descriptiveness of captions may not be the primary factor influencing emotion recognition accuracy. Instead, the challenge appears to stem from the inherent difficulty in accurately transferring emotional cues from visual to textual modality.

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| BLIP-base | 0.4039 | 0.3275 | 0.3150 |
| CLIP-ViT-B16 | 0.3809 | 0.2417 | 0.1695 |
| ALBEF-base | 0.3459 | 0.2371 | 0.2033 |

(a) Zero-Shot

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Google ViT | 0.7380 | 0.7313 | 0.7343 |
| MS SWIN | 0.7065 | 0.6932 | 0.6973 |
| ConvNeXT | 0.6865 | 0.6685 | 0.6740 |

(b) Fine-Tuned

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| GoEmotions | 0.2552 | 0.2197 | 0.2154 |
| Paletz | 0.1425 | 0.1808 | 0.1225 |
| SemEval | 0.1525 | 0.2081 | 0.1406 |

(c) Auto-Captioning

Table 1. Image-based Emotion-Recognition Macro-Average Performance Across Learning Strategies (bolded results are highest). Results show fine-tuned computer vision models outperform zero-shot and caption-based methods by a substantial margin when predicting emotion classes in EmoSet [^39]. Auto-captioning methods use BLIP2 to generate captions and Demux-MEmo [^6] for text-based emotion extraction.

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| GoEmotions | 0.2932 | 0.2610 | 0.2236 |
| Paletz | 0.1171 | 0.1531 | 0.0799 |
| SemEval | 0.1224 | 0.1513 | 0.0896 |

Table 2. Sampled Auto-Captioning via GPT-4o. Results are comparable to BLIP-2-based auto-captioning for the GoEmotions pre-trained Demux-MEmo model but are appear substantially below the BLIP-2 for the other pre-trained model configurations.

### 4.2. Comparing Emotions in Prompts and Generated Images

#### 4.2.1. Cross-Correlations Between Modalities

Figure 1 shows the Spearman correlations among pairs of emotions extracted from the prompt (using DeMuX) and from the image (using the Google ViT). This figure suggests negative emotions (anger, disgust, fear, and sadness) align relatively well across these modalities compared to positive emotions. For the positive emotions, joy in prompts corresponds to both amusement and contentment in images, and excited correlates with awe, but surprise and amusement in prompts is less correlated with the positive emotions in images. For context, the scale of correlations seen here— $[-0.2,+0.2]$ —is close to that in [^29], where most correlations within the positive and negative sets of emotions are in the 0.3 range.

![Refer to caption](https://arxiv.org/html/2411.05985v2/x1.png)

Figure 1. Cross-Modality Spearman ρ 𝜌 \\rho italic\_ρ Correlations Between DeMuX and EmoSet. Results show general alignment for the negative emotions (anger, disgust, fear, and sadness). Alignment is less clear in the positive emotions, where joy in prompts corresponds to both amusement and contentment in images, and excited correlates with awe, but surprise and amusement in prompts is less correlated with the positive emotions in images.

For comparison, we also take the highest-performing model by macro-F1 [^1] from the Papers with Code leaderboard on text classification in emotions,<sup>2</sup> a RoBERTa derivative. This model uses Amazon’s SageMaker and its Hugging Face Deep Learning container. Results for the negative emotions are quite similar, though the positive emotions do not exhibit as high correlations as with DeMuX (potentially an artifact of a difference in DeMuX’s multi-label approach to discrete emotion classification versus the sagemaker-RoBERTa-base-emotion multi-class construction).

![Refer to caption](https://arxiv.org/html/2411.05985v2/x2.png)

Figure 2. Cross-Modality Spearman ρ 𝜌 \\rho italic\_ρ Correlations Between Amazon’s SageMaker and EmoSet. Negative emotions appear similarly aligned compared to DeMuX, but positive emotions show less alignment.

#### 4.2.2. Comparing Most Prevalent Emotions

Table 3 compares the rankings of most prevalent emotions in images (via Google ViT) to those present in the prompts. Comparing the emotions evoked by the images to either the DeMuX- or RoBERTa-based models, fear is far more prevalent in the images than in the prompts. Looking specifically at the comparison between Google ViT’s image labels and DeMuX’s prompt labels (where cross correlations are stronger than with the RoBERTa model), fear goes from the most prevalent in the images to the second-to-least prevalent in the prompt. This result yields evidence that, for at least the images from DiffusionDB, AI-generated visuals tend to produce more negative and fearful content than the prompt may otherwise suggest.

Table 3. Cross-Modality Emotion Rankings Between Images and Prompts. Values in the parentheses show average proportions. At least for the DiffusionDB-produced images, those images appear to over-represent fear relative to the emotions evoked by the text-based prompts.

<table><thead><tr><th></th><th><em>Image</em></th><th colspan="2"><em>Text</em></th></tr><tr><th></th><th>Google ViT</th><th>DeMuX</th><th>RoBERTa</th></tr></thead><tbody><tr><th>1</th><td>Fear (0.33)</td><td>Excitement (0.43)</td><td>Joy (0.54)</td></tr><tr><th>2</th><td>Amusement (0.20)</td><td>Joy (0.18)</td><td>Anger (0.24)</td></tr><tr><th>3</th><td>Awe (0.12)</td><td>Surprise (0.11)</td><td>Fear (0.10)</td></tr><tr><th>4</th><td>Anger (0.09)</td><td>Amusement (0.09)</td><td>Sadness (0.06)</td></tr><tr><th>5</th><td>Sadness (0.08)</td><td>Anger (0.06)</td><td>Surprise (0.04)</td></tr><tr><th>6</th><td>Excitement (0.07)</td><td>Disgust (0.05)</td><td>Love (0.03)</td></tr><tr><th>7</th><td>Contentment (0.06)</td><td>Fear (0.05)</td><td>- -</td></tr><tr><th>8</th><td>Disgust (0.04)</td><td>Sadness (0.04)</td><td>- -</td></tr></tbody></table>

As a post-hoc analysis using a more recent generative AI model, we sample an additional 200 prompts from the DiffusionDB dataset and use OpenAI’s GPT-4o system to generate new images from these prompts. Applying our Google ViT emotion recognition model to these newly generated images yields the emotion distribution in Table 4. Again, fear dominates the images, and excitement dominates the prompts. These results are largely consistent with those from DiffusionDB-generated images.

Table 4. Cross-Modality Emotion Rankings Between Images and Prompts Using ChatGPT-4o. GPT-4o-generated images also over-represent fear relative to the emotions evoked by the text-based prompts.

|  | *Image* | *Text* |
| --- | --- | --- |
|  | Google ViT (GPT-4o Images) | DeMuX |
| 1 | Fear (0.38) | Excitement (0.43) |
| 2 | Amusement (0.24) | Joy (0.17) |
| 3 | Awe (0.11) | Surprise (0.11) |
| 4 | Sadness (0.08) | Amusement (0.09) |
| 5 | Anger (0.07) | Anger (0.07) |
| 6 | Excitement (0.06) | Fear (0.06) |
| 7 | Contentment (0.05) | Disgust (0.05) |
| 8 | Disgust (0.02) | Sadness (0.04) |

In both cases, these results are likely not attributable solely to modeling errors, as our manual validation of our fine-tuned ViT model still has a high acceptance rate ($0.86$). A resulting interval around the fear emotion in both StableDiffusion and GPT-4o would not alter its top ranking in prevalence.

## 5\. Discussion and Limitations

### 5.1. Unvirtuous Cycles and Emotional Bias

While work from affective computing, psychology, communications, AI, and journalism have all studied various aspects of emotions, emotional inducement, and emotionality in imagery, the modern information ecosystem presents challenges at the intersection of these spaces. In isolation, findings from HCI show images tend to increase engagement with content in online spaces, findings from psychology show emotional elicitation varies by modality [^35], and still other findings show journalistic sources’ preference for violent visuals are both influence by and can impact real-world loss of life [^26]. Likewise, studies of generative AI systems show clear non-conscious racial and gender biases [^12] [^18]. Given the feedback loop between humans and the AI systems governing our tools and online spaces, one should anticipate large, generative AI systems that are trained on extensive collections of digital trace data will reflect similar biases.

Our results find evidence for such emotional biases in the automated production and generation of visual media. These findings persist in images generated by both StableDiffusion and by DALL-E, suggesting a result endemic to generative models rather than an isolated incident; naturally, a larger sample of models is needed to examine this possibility further, but these two methods are broadly available to the general public. This availability, while valuable for democratizing content creation, increases the potential harms negative emotional bias may wrought. As [^20] demonstrates, negative emotion is contagious in online spaces; this contagious is especially problematic in light of increasing trends of anxiety disorder in the US, especially among youths [^16], with such harms especially impacting young women [^13]. These unintended consequences are not limited to emotional well-being and mental health either. [^2] also demonstrates the implications of negative imagery for political mobilization and instability, where exposure to negative—especially fear-inducing—visuals increased likelihood to protest.

### 5.2. Breaking Unvirtuous Cycles

We hope a main outcome of this work is to stimulate additional work in addressing emotional bias in generative AI systems and raising emotional bias to an important dimension of studies in bias. Crucially, however, a first step in breaking this unvirutous cycle is raising *awareness* of such emotional bias, as this awareness can empower users to acknowledge and account for its existence. We need not rely solely on individual awareness though, as we can also adapt interfaces from the fairness literature [^30] into new tools that empower users and designers to create more robust, intentional, and pro-social generative-AI systems.

### 5.3. Limitations in Emotion Labeling: Mapping Across Diverse Emotion Sets

EmoSet employs the Mikel model with eight categories, i.e., amusement, awe, contentment, excitement, anger, disgust, fear, sadness, where the former four are positive emotions and the latter four are negative ones. The Paletz model, however, contains 22 distinct emotions; GoEmotions contains 27 distinct emotions; and SemEval contains 11 emotions. Additionally, there are inconsistencies across different datasets and model configurations in terms of which emotions are even contained. GoEmotions, Paletz, and SemEval all lack awe and contentment, and SemEval furthermore lacks amusement and excitement. These inconsistencies pose a minimal concern in our prompt-image emotion comparison analysis, as $\rho$ correlations are evaluated regardless of emotion synonymity and are left to further analysis by the reader. However, in evaluating auto-captioning pipelines, this poses a notable challenge, as discreet emotions in Paletz, GoEmotions, and SemEval need to be mapped to corresponding emotions in EmoSet even if they aren’t directly present in order to evaluate F1 scores.

To resolve these inconsistencies, we discard extraneous emotion configurations from the models that don’t map to EmoSet Labels. For EmoSet labels that do not have an exact match to a DeMuX-MeMO configuration, we map a corresponding emotion from the configuration. Table 5 identifies mappings made across models and datasets for pipeline evaluation.

The cause behind low F1 scores for the auto-captioning pipeline could lie within the fundamental multi-class vs. multi-label modality issue, the mapping inconsistency issue, or a tertiary, unidentified issue.

Table 5. Emotion mappings across models and datasets. Distributions for emotions in the lower half were not analyzed. \*Embarrassment, \*\*Dissapointment

| EmoSet | GoEmotions | Paletz | SemEval 2018 |
| --- | --- | --- | --- |
| amusement | amusement | amusement | joy |
| awe | surprise | wonder | surprise |
| contentment | joy | happiness | trust |
| excitement | excitement | excitement | optimism |
| anger | anger | anger | anger |
| disgust | disgust | disgust | disgust |
| fear | fear | fear | fear |
| sadness | sadness | sadness | sadness |
|  | love | love | anticipation |
|  | optimism | surprise | love |
|  | embar\* | hate | pessimism |
|  | admiration | contempt |  |
|  | pride | embr\* |  |
|  | gratitude | admiration |  |
|  | relief | sexual attraction |  |
|  | confusion | cuteness/kama muta |  |
|  | annoyance | pride |  |
|  | approval | nostalgia |  |
|  | caring | empathic pain |  |
|  | curiosity | gratitude |  |
|  | desire | envy |  |
|  | disapt\*\* | relief |  |
|  | disapproval | confusion |  |
|  | grief |  |  |
|  | nervousness |  |  |
|  | realization |  |  |
|  | remorse |  |  |

### 5.4. Limitations in Emotion Labeling: Multi-Class Versus Multi-Label

This investigation has revealed a key limitation in the current datasets available, particularly in the mismatch between the computational task’s structure and the underlying psychological theory. Namely, EmoSet’s annotation process constrains every image to evoke a single emotion. In this process, images first automatically receive an emotion label based on keywords associated with Mikels’ eight-category model [^25], retrieved during a web-scraping process. Human annotators then assess these labels for validity (rather than independently generating these labels manually). To finalize an image’s emotion label, at least 7 out of 10 annotators must agree. Although [^39] recognizes the inherent ambiguity of emotion recognition, this process still assigns only a single emotion to each image, creating a multi-label classification problem. Additionally, because annotators merely confirm emotions, they may assign an image to a certain emotion even if that emotion does not dominate the image’s overall emotional impact. For example, if an emotion evokes mostly contentment and some joy, but is automatically assigned the emotion of joy, annotators will confirm that the emotion evokes joy even while disagreeing that joy is the image’s primary emotion.

While the EmoSet annotation process ensures consistency and scalability while paralleling existing emotion-recognition tasks from natural language process [^3], it oversimplifies the complexities of emotional evocation. Specifically, as outlined in [^29], emotions are states that can be experienced and evoked simultaneously, and emotion annotation should reflect this multi-label structure inherent to the task. This difference in multi-class versus multi-label structure also has important implications for the modeling of emotion and its impact on social media engagement.

We can also point to instances in the EmoSet dataset where such deviations are present, such as in Figure 3(a), which shows a statue of the Buddha. The associated EmoSet label for this image is one of “awe”, though depictions of the Buddha are often associated with themes of joy, contentment, awe, and other positive emotions. Similarly, Figure 3(b) shows a yawning cat, labeled “anger” in EmoSet, but this image may also evoke contentment, amusement, or cuteness/kama muta, the last of which a particularly important dimension in online spaces [^15]. This issue is not new, as images have long been understood to evoke similar yet distinct emotions between various viewers, both in the computer vision space [^31] and in psychology [^35] —where different modalities have shown varying efficacy in emotional inducement.

![Refer to caption](https://arxiv.org/html/2411.05985v2/extracted/6068292/figures/awe-07047.png)

(a) Image Awe\_07047, containing a smiling Buddha figure. Annotated as ”Awe”.

This mismatch is not specific to EmoSet—as the foundation in [^25] explicitly seeks to identify *the most salient emotion* in a given image—but structuring the emotion recognition task as a multi-class one is problematic for two key reasons:

1. Images are known to evoke multiple similar and distinct emotions, and two annotators may disagree on which emotion is the most salient. Forcing one emotion then impacts the validity of the label.
2. Emotions are not easily distinguishable, as we see significant overlap between Amusement, Awe, Contentment, and Joy. Similarly, other emotion classifications beyond those used by EmoSet contain both positive and negative emotions with much overlap. Training models to identify singular emotions within images without considering overlap between alternate emotions raises similar validity concerns.

### 5.5. Limitations in Emotion Labeling: Lacking a “Neutral” Class

Image classification models fine-tuned on EmoSet (i.e this paper’s ViT, ConvNeXT, and SWIN models) treat multi-class image emotion classification as a zero-sum task and confidently project that a given image contains a significant amount of at least one emotion. In reality, many images may be neutral or may strongly evoke multiple separate emotions. Further complicating the matter is that EmoSet has no ”neutral” classification for training purposes and similarly doesn’t quantify how evocative a given emotion is. This creates skewed comparisons with the text emotion classification models, which identify multiple distinct emotions without treating the task as zero-sum. While the text models can identify relatively low amounts of multiple emotions within a single prompt, the image models are forced to identify high levels of only a single emotion, even if there are other emotions within the same image and/or the identified emotion is relatively minimal within the image because the image is primarily neutral.

### 5.6. Future Work

#### 5.6.1. Auto-Captioning

A key area for future work involves examining the impact of caption length and descriptiveness on the accuracy of emotion recognition in images. Longer, more descriptive captions—such as those generated by ChatGPT— are richer in detail and contextual information. For example, the smiling Buddha image in Figure 3(a) is captioned as:

BLIP:

a buddha statue with a colorful ribbon around it

ChatGPT:

A serene and weathered statue of the Buddha in a meditative pose, draped with a colorful fabric sash, exuding a sense of peace, tranquility, and timeless wisdom. The aged stone and soft expression of the figure evoke an aura of calm reflection and spiritual depth.

Thus, we hypothesize that ChatGPT as an auto-captioning model should be more effective. However, the results did not align with this expectation. Table 2 and Table 1(c) demonstrate that ChatGPT captions led to only a marginal improvement in the GoEmotions model, and the other models performed worse with ChatGPT-generated captions compared to BLIP. Considering a smaller sample size of evaluated GPT captions, we essentially see a negligible effect from the descriptiveness of the captions.

We propose three potential explanations for these results: (1) ChatGPT may hallucinate emotions not present in the image, leading to inaccurate captions; (2) the task itself may mismatch with the training paradigms of the emotion recognition models, which are not explicitly designed to identify emotions from cross-modal data; and (3) inherent challenges, such as the multi-class versus multi-label nature of emotion recognition and cross-modality limitations, may contribute to the observed discrepancies.

Based on these results and hypothesized rationales, we call for further investigation and research into the broader effectiveness of auto-captioning techniques for emotion identification and whether the descriptiveness of captions meaningfully impacts model accuracy.

#### 5.6.2. Novel Datasets

To address the limitations identified in sections 5.2 - 5.4, we call for the construction of a new, comprehensive image-emotion dataset. Each image should be annotated with a numerical rating for a set of distinct emotions; i.e any given image should have a rating 0-10 for amusement, awe, contentment, etc. Treating emotions in images in this multi-label manner and furthermore quantifying the emotions should allow models to more comprehensively identify emotional salience and classify images with unclear salience.

## 6\. Conclusions

Though this paper assesses and selects multiple methods for recognizing emotions in images and text, this work is in service of the larger question concerning the potential for an unvirtuous cycle emerging in the feedback loop between content created for/or online spaces and the content produced by generative AI systems. Our analysis finds evidence that at least two generative AI models, one of which is at or near current state of the art, are likely to over-represent negative emotion in the images that they generate compared to the prompts from which these images were generated. This bias towards negative emotion has substantial implications for the health of the online information environment and the emotional and psychological well-being of the users of these generative AI systems. While these results may only be specific to Stable Diffusion and ChatGPT-4o/DALL-E, the popularity and ease of general access to these models may exacerbate this emotional biases. Through this work, we hope to shed light on this potential unvirtuous cycle and anti-social feedback loop so that future work can establish new measures to correct for it and educate and empower users of these technologies.

[^1]: 

[^2]: Andreu Casas and Nora Webb Williams. 2019. Images that Matter: Online Protests and the Mobilizing Role of Pictures. *Political Research Quarterly* 72, 2 (2019), 360–375. [https://doi.org/10.1177/1065912918786805](https://doi.org/10.1177/1065912918786805)

[^3]: Chatterjee, Narahari, Ankush and, Joshi, Kedhar Nath and, Agrawal, Meghana and, and Puneet. \[n. d.\]. SemEval-2019 Task 3: EmoContext Contextual Emotion Detection in Text. In *Proceedings of the 13th International Workshop on Semantic Evaluation*. Association for Computational Linguistics, Minneapolis, Minnesota, USA, 39–48. [https://doi.org/10.18653/v1/s19-2005](https://doi.org/10.18653/v1/s19-2005)

[^4]: Marc Cheong, Ehsan Abedin, Marinus Ferreira, Ritsaart Reimann, Shalom Chalson, Pamela Robinson, Joanne Byrne, Leah Ruppanner, Mark Alfano, and Colin Klein. 2024. Investigating Gender and Racial Biases in DALL-E Mini Images. *ACM J. Responsib. Comput.* 1, 2 (2024). [https://doi.org/10.1145/3649883](https://doi.org/10.1145/3649883)

[^5]: Georgios Chochlakis, Gireesh Mahajan, Sabyasachee Baruah, Keith Burghardt, Kristina Lerman, and Shrikanth Narayanan. 2023a. Leveraging Label Correlations in a Multi-Label Setting: a Case Study in Emotion. In *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. 1–5. [https://doi.org/10.1109/ICASSP49357.2023.10096864](https://doi.org/10.1109/ICASSP49357.2023.10096864)

[^6]: Georgios Chochlakis, Gireesh Mahajan, Sabyasachee Baruah, Keith Burghardt, Kristina Lerman, and Shrikanth Narayanan. 2023b. Using Emotion Embeddings to Transfer Knowledge between Emotions, Languages, and Annotation Formats. *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* 00 (2023), 1–5. [https://doi.org/10.1109/icassp49357.2023.10095597](https://doi.org/10.1109/icassp49357.2023.10095597)

[^7]: Monojit Choudhury. 2023. Generative AI has a language problem. *Nature Human Behaviour* 7, 11 (2023), 1802–1803. [https://doi.org/10.1038/s41562-023-01716-4](https://doi.org/10.1038/s41562-023-01716-4)

[^8]: Geoffrey Currie, Josie Currie, Sam Anderson, and Johnathan Hewis. 2024. Gender bias in generative artificial intelligence text-to-image depiction of medical students. *Health Education Journal* 83, 7 (2024), 732–746. [https://doi.org/10.1177/00178969241274621](https://doi.org/10.1177/00178969241274621)

[^9]: Luna De Bruyne, Toni GLA van der Meer, Orphée De Clercq, and Véronique Hoste. 2024. Using State-of-the-art Emotion Detection Models in a Crisis Communication Context. *Computational Communication Research* 6, 1 (2024), 1.

[^10]: Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A Dataset of Fine-Grained Emotions. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 4040–4054. [https://doi.org/10.18653/v1/2020.acl-main.372](https://doi.org/10.18653/v1/2020.acl-main.372)

[^11]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR* (2021).

[^12]: Emilio Ferrara. 2023. Fairness and Bias in Artificial Intelligence: A Brief Survey of Sources, Impacts, and Mitigation Strategies. *Sci* 6, 1 (2023), 3. [https://doi.org/10.3390/sci6010003](https://doi.org/10.3390/sci6010003) arXiv:2304.07683

[^13]: A.W. Geiger and Leslie Davis. 2019. *A growing number of American teenagers – particularly girls – are facing depression*. Technical Report. Pew Research.

[^14]: Stephanie Geise, Diana Panke, and Axel Heck. 2021. Still Images—Moving People? How Media Images of Protest Issues and Movements Influence Participatory Intentions. *The International Journal of Press/Politics* 26, 1 (2021), 92–118. [https://doi.org/10.1177/1940161220968534](https://doi.org/10.1177/1940161220968534)

[^15]: Ewa M. Golonka, Kelly M. Jones, Patrick Sheehan, Nick B. Pandža, Susannah B. F. Paletz, C. Anton Rytting, and Michael A. Johns. 2023. The construct of cuteness: A validity study for measuring content and evoked emotions on social media. *Frontiers in Psychology* 14 (2023), 1068373. [https://doi.org/10.3389/fpsyg.2023.1068373](https://doi.org/10.3389/fpsyg.2023.1068373)

[^16]: Renee D Goodwin, Andrea H Weinberger, June H Kim, Melody Wu, and Sandro Galea. 2020. Trends in anxiety among adults in the United States, 2008-2018: Rapid increases among young adults. *Journal of psychiatric research* 130 (2020), 441–446. [https://doi.org/10.1016/j.jpsychires.2020.08.014](https://doi.org/10.1016/j.jpsychires.2020.08.014)

[^17]: Jeffrey Gottfried. 2024. *Americans’ Social Media Use*. Technical Report. Pew Research Center. [https://www.pewresearch.org/internet/wp-content/uploads/sites/9/2024/01/PI\_2024.01.31\_Social-Media-use\_report.pdf](https://www.pewresearch.org/internet/wp-content/uploads/sites/9/2024/01/PI_2024.01.31_Social-Media-use_report.pdf)

[^18]: Janna Hastings. 2024. Preventing harm from non-conscious bias in medical generative AI. *The Lancet Digital Health* 6, 1 (2024), e2–e3. [https://doi.org/10.1016/s2589-7500(23)00246-7](https://doi.org/10.1016/s2589-7500\(23\)00246-7)

[^19]: Johana Kotisova. 2019. The elephant in the newsroom: Current research on journalism and emotion. *Sociology Compass* 13, 5 (2019). [https://doi.org/10.1111/soc4.12677](https://doi.org/10.1111/soc4.12677)

[^20]: Adam D. I. Kramer, Jamie E. Guillory, and Jeffrey T. Hancock. 2014. Experimental evidence of massive-scale emotional contagion through social networks. *Proceedings of the National Academy of Sciences* 111, 24 (2014), 8788–8790. [https://doi.org/10.1073/pnas.1320040111](https://doi.org/10.1073/pnas.1320040111) arXiv:https://www.pnas.org/doi/pdf/10.1073/pnas.1320040111

[^21]: Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. arXiv:2201.12086 \[cs.CV\] [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)

[^22]: Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. 2021. Swin Transformer V2: Scaling Up Capacity and Resolution. *CoRR* abs/2111.09883 (2021). arXiv:2111.09883 [https://arxiv.org/abs/2111.09883](https://arxiv.org/abs/2111.09883)

[^23]: George E. Marcus. 2023. Evaluating the status of theories of emotion in political science and psychology. *Frontiers in Political Science* 4 (2023), 1080884. [https://doi.org/10.3389/fpos.2022.1080884](https://doi.org/10.3389/fpos.2022.1080884)

[^24]: Cameron Martel, Gordon Pennycook, and David G. Rand. 2020. Reliance on emotion promotes belief in fake news. *Cognitive Research: Principles and Implications* 5, 1 (2020), 47. [https://doi.org/10.1186/s41235-020-00252-3](https://doi.org/10.1186/s41235-020-00252-3)

[^25]: Joseph A. Mikels, Barbara L. Fredrickson, Gregory R. Larkin, Casey M. Lindberg, Sam J. Maglio, and Patricia A. Reuter-Lorenz. 2005. Emotional category data on images from the international affective picture system. *Behavior Research Methods* 37, 4 (2005), 626–630. [https://doi.org/10.3758/bf03192732](https://doi.org/10.3758/bf03192732)

[^26]: Ross A. Miller and Karen Albert. 2015. If It Leads, It Bleeds (and If It Bleeds, It Leads): Media Coverage and Fatalities in Militarized Interstate Disputes. *Political Communication* 32, 1 (2015), 61–82. [https://doi.org/10.1080/10584609.2014.880976](https://doi.org/10.1080/10584609.2014.880976)

[^27]: Saif Mohammad, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana Kiritchenko. 2018. SemEval-2018 Task 1: Affect in Tweets. In *Proceedings of the 12th International Workshop on Semantic Evaluation*, Marianna Apidianaki, Saif M. Mohammad, Jonathan May, Ekaterina Shutova, Steven Bethard, and Marine Carpuat (Eds.). Association for Computational Linguistics, New Orleans, Louisiana, 1–17. [https://doi.org/10.18653/v1/S18-1001](https://doi.org/10.18653/v1/S18-1001)

[^28]: Susannah B. F. Paletz, Michael A. Johns, Egle E. Murauskaite, Ewa M. Golonka, Nick B. Pandža, C. Anton Rytting, Cody Buntain, and Devin Ellis. 2023. Emotional content and sharing on Facebook: A theory cage match. *Science Advances* 9, 39 (2023), eade9231. [https://doi.org/10.1126/sciadv.ade9231](https://doi.org/10.1126/sciadv.ade9231)

[^29]: Susannah B. F. Paletz, Ewa M. Golonka, Nick B. Pandža, Grace Stanton, David Ryan, Nikki Adams, C. Anton Rytting, Egle E. Murauskaite, Cody Buntain, Michael A. Johns, and Petra Bradley. 2024. Social media emotions annotation guide (SMEmo): Development and initial validity. *Behavior Research Methods* 56, 5 (2024), 4435–4485. [https://doi.org/10.3758/s13428-023-02195-1](https://doi.org/10.3758/s13428-023-02195-1)

[^30]: Gourab K. Patro, Lorenzo Porcaro, Laura Mitchell, Qiuyue Zhang, Meike Zehlike, and Nikhil Garg. 2022. Fair ranking: a critical review, challenges, and future directions. In *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency* *(FAccT ’22)*. Association for Computing Machinery, New York, NY, USA, 1929–1942. [https://doi.org/10.1145/3531146.3533238](https://doi.org/10.1145/3531146.3533238)

[^31]: Kuan-Chuan Peng, Tsuhan Chen, Amir Sadovnik, and Andrew C. Gallagher. 2015. A Mixed Bag of Emotions: Model, Predict, and Transfer Emotion Distributions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[^32]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In *Proceedings of the 38th International Conference on Machine Learning* *(Proceedings of Machine Learning Research, Vol. 139)*, Marina Meila and Tong Zhang (Eds.). PMLR, 8748–8763. [https://proceedings.mlr.press/v139/radford21a.html](https://proceedings.mlr.press/v139/radford21a.html)

[^33]: Inioluwa Deborah Raji, Andrew Smart, Rebecca N. White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. \[n. d.\]. Closing the AI accountability gap: defining an end-to-end framework for internal algorithmic auditing. In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency* *(FAT\* ’20)*. Association for Computing Machinery, New York, NY, USA, 33–44. [https://doi.org/10.1145/3351095.3372873](https://doi.org/10.1145/3351095.3372873)

[^34]: Steve Rathje, Jay J. Van Bavel, and Sander van der Linden. 2021. Out-group animosity drives engagement on social media. *Proceedings of the National Academy of Sciences* 118, 26 (2021), e2024292118. [https://doi.org/10.1073/pnas.2024292118](https://doi.org/10.1073/pnas.2024292118)

[^35]: Ewa Siedlecka and Thomas F. Denson. 2019. Experimental Methods for Inducing Basic Emotions: A Qualitative Review. *Emotion Review* 11, 1 (2019), 87–97. [https://doi.org/10.1177/1754073917749016](https://doi.org/10.1177/1754073917749016)

[^36]: Karin Wahl-Jorgensen. 2020. An Emotional Turn in Journalism Studies? *Digital Journalism* 8, 2 (2020), 175–194. [https://doi.org/10.1080/21670811.2019.1697626](https://doi.org/10.1080/21670811.2019.1697626)

[^37]: Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. 2022. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. *arXiv preprint arXiv:2210.14896* (2022).

[^38]: Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. *CoRR* abs/2301.00808 (2023). [https://doi.org/10.48550/arXiv.2301.00808](https://doi.org/10.48550/arXiv.2301.00808) arXiv:2301.00808

[^39]: Jingyuan Yang, Qirui Huang, Tingting Ding, Dani Lischinski, Daniel Cohen-Or, and Hui Huang. 2023. EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes. *Proceedings of the IEEE/CVF International Conference on Computer Vision* (2023). [https://doi.org/10.48550/arxiv.2307.07961](https://doi.org/10.48550/arxiv.2307.07961) arXiv:2307.07961