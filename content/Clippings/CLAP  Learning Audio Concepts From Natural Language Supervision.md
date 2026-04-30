---
title: "CLAP : Learning Audio Concepts From Natural Language Supervision"
source: "https://ar5iv.labs.arxiv.org/html/2206.04769"
author:
published:
created: 2026-04-30
description: "Mainstream Audio Analytics models are trained to learn under the paradigm of one class label to many recordings focusing on one task. Learning under such restricted supervision limits the flexibility of models because …"
tags:
  - "clippings"
---
###### Abstract

Mainstream Audio Analytics models are trained to learn under the paradigm of one class label to many recordings focusing on one task. Learning under such restricted supervision limits the flexibility of models because they require labeled audio for training and can only predict the predefined categories. Instead, we propose to learn audio concepts from natural language supervision. We call our approach Contrastive Language-Audio Pretraining (CLAP), which learns to connect language and audio by using two encoders and a contrastive learning to bring audio and text descriptions into a joint multimodal space. We trained CLAP with 128k audio and text pairs and evaluated it on 16 downstream tasks across 8 domains, such as Sound Event Classification, Music tasks, and Speech-related tasks. Although CLAP was trained with significantly less pairs than similar computer vision models, it establishes SoTA for Zero-Shot performance. Additionally, we evaluated CLAP in a supervised learning setup and achieve SoTA in 5 tasks. Hence, CLAP’s Zero-Shot capability removes the need of training with class labels, enables flexible class prediction at inference time, and generalizes to multiple downstream tasks.

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2206.04769/assets/figures/clap_diagram_v2.png)

Fig. 1: CLAP jointly trains an audio and a text encoder to learn the (dis)similarity of audio and text pairs in a batch using contrastive learning. At testing time, the pretrained encoders are used to extract audio embeddings from the testing audio and text embeddings from the class labels. Zero-Shot linear classification is achieved by computing cosine similarity between the embeddings.

The human auditory system can hear sounds and extract the kind of decisions or meanings we need to interact with our surroundings [^1]. For example, if we are in a soccer game and suddenly hear the crowd cheering joyfully, we can assume the local team scored! Computer models aim to understand audio cues by automatically process audio signals and extract meaning [^1]. Mainstream Machine Learning models break the human hearing into tasks, such as the classification of sound events and acoustic scenes. Such models are trained by associating audio recordings to class labels of predefined categories for a specific task and can only predict specific categories [^2]. Learning under such restricted supervision limits the flexibility to predict unseen classes.

Self-Supervised Learning (SSL) pretrains models with unlabeled audio, avoiding the limited supervision of learning from class labels. However, SSL excludes semantic knowledge from natural language. The pretrained model is then adapted to a downstream task in a supervised setup learning under the class label paradigm [^3] [^4] [^5]. Mainstream and SSL models have static output layers that can only predict the predefined categories. On the other hand, models that enable Zero-Shot predictions can take an input audio and yield a prediction score for any class typed by the user. Zero-shot requires no training stage so there are no predefined categories. To enable such flexibility and generalization, models need to learn the relationships between the acoustic semantics and language semantics.

A middle path between both approaches is to learn audio concepts from natural language supervision, which is under-explored. Computer Vision has successfully developed models that learn image representations with natural language supervision, achieving high performance across different downstream tasks in Zero-Shot predictions and adapted (e.g. finetuned) to target datasets in a supervised setup. Examples are Open AI’s CLIP [^6], Florence [^7], and [^8]. In the audio domain, Wav2clip [^9] and Audioclip [^10] distill from CLIP and are trained with audio and class labels from AudioSet [^11] instead of audio and natural language. Although the performance was promising, it is yet to be explored how natural language can benefit flexibility and generalization to new classes and tasks.

We call our approach Contrastive Language-Audio Pretraining (CLAP), which learns to connect natural language and audio by using two encoders and contrastive learning to bring audio and text descriptions into a joint multimodal space. Our main contributions are first, to introduce our CLAP model trained with 128k audio and text pairs. Second, our model enables Zero-Shot predictions, thus it removes the need of training and forcing a predefined set of categories and enables flexible class prediction at inference time. Third, CLAP generalizes to 16 downstream tasks across 8 domains by establishing Zero-Shot SoTA performance. We also include insights about the model in Section 4.

## 2 Method

CLAP is illustrated in Fig 1. The input is audio and text pairs passed to an audio encoder and a text encoder. Both representations are connected in joint multimodal space with linear projections. The space is learned with the (dis)similarity of audio and text pairs in a batch using contrastive learning. The pretrained encoders with their projection layers can be used to compute audio and text embeddings and enable Zero-Shot Classification. Our method is inspired by the CLIP model [^6].

### 2.1 Contrastive Language-Audio Pretraining

Let the processed audio be $X_{a}$ s.t. $X_{a}\in\mathbb{R}^{F\times T}$ where $F$ are the number of spectral components (e.g. Mel bins) and $T$ are the number of time bins. Let the text be represented by $X_{t}$. Each audio-text pair in a batch of $N$ is represented as $\{X_{a},X_{t}\}_{i}$ where $i\in[0,N]$. For convenience, we dropped the $i$ notation, and henceforth $\{X_{a},X_{t}\}$ will denote a batch of N.

From the pairs, the audio and text are passed through an audio encoder and a text encoder respectively. Let $f_{a}(.)$ represent the audio encoder and $f_{t}(.)$ represent the text encoder. For a batch of N:

$$
\hat{X}_{a}=f_{a}(X_{a});\hat{X}_{t}=f_{t}(X_{t})
$$

where $\hat{X}_{a}\in\mathbb{R}^{N\times V}$ are the audio representations of dimensionality $V$, and $\hat{X}_{t}\in\mathbb{R}^{N\times U}$ are the text representations of dimensionality $U$.

We brought audio and text representations, $\hat{X}_{a}$ and $\hat{X}_{t}$, into a joint multimodal space of dimension $d$ by using a learnable linear projection:

$$
E_{a}=L_{a}(X_{a});E_{t}=L_{t}(X_{t})
$$

where $E_{a}\in\mathbb{R}^{N\times d}$, $E_{t}\in\mathbb{R}^{N\times d}$, $L_{a}$ and $L_{t}$ are the linear projections for audio and text respectively.

Now that the audio and text embeddings ($E_{a}$, $E_{t}$) are comparable, we can measure similarity:

$$
C=\tau*(E_{t}\cdot E_{a}^{\top})
$$

where $\tau$ is a temperature parameter to scale the range of logits. The similarity matrix $C\in\mathbb{R}^{N\times N}$ has $N$ correct pairs in the diagonal and $N^{2}-N$ incorrect pairs in the off-diagonal.

$$
\mathcal{L}=0.5*(\ell_{text}(C)+\ell_{audio}(C))
$$

where $\ell_{k}=\frac{1}{N}\sum_{i=0}^{N}\log diag(softmax(C))$ along text and audio axis respectively. We used this symmetric cross-entropy loss ($\mathcal{L}$) over the similarity matrix to jointly train the audio encoder and the text encoder along with their linear projections.

### 2.2 Zero-Shot Linear Classification

For Zero-Shot classification, we used CLAP’s ability to determine the similarity between audio and text. Let’s consider a target dataset with $C$ class labels and $N$ test audios. First, we compute audio embeddings and text embeddings for $N$ audios and $C$ classes using the pretrained encoders and their projection layers. Second, because both the embeddings are in a common space, we compute the cosine similarity between each testing audio and all the class labels. Each audio will have as many logits as class labels. Third, logits are turned into a probability distribution by applying softmax for binary or multiclass and sigmoid for multilabel classification.

## 3 Experiments

Sound Event Classification Music Model ESC50 FSD50K US8K DCASE17 Task 4 AudioSet Music Speech Music Genres Mri. Stroke Mri. Tonic Random 0.02 $<$ 0.005 0.1 0.05 $<$ 0.0018 0.5 0.1 0.1 0.1667 Benchmark (ZS) 0.6940 [^10] 0.0302 [^9] 0.6531 [^10] - - - - - - CLAP(ZS) 0.826 0.3024 0.7324 0.3 0.058 1.0 0.252 0.3447 0.1965 Benchmark (Best) 0.9715 [^10] 0.641 [^12] 0.90-7 [^10] 0.646 [^13] 0.471 [^14] 0.992 [^12] 0.883 [^12] 0.975 [^12] 0.942 [^12] CLAP (Best) 0.9670 0.5859 0.8796 0.5938 - 1.0 0.9130 0.9794 0.9534

Instrument Classification Acoustic Scene Classification Emotion Recognition Keyword Spotting Vocal Sound Classification Speaker Counting Model Beijing Opera TUT2017 CRE MA-D RAV DESS Speech Comm. Vocal Sound Libri Count Random 0.25 0.06 0.1667 0.125 0.083 0.1667 0.090 CLAP (ZS) 0.4746 0.2963 0.1784 0.1599 0.1063 0.4945 0.1788 Benchmark (Best) 0.975 [^12] 0.843 [^15] 0.752 [^12] 0.8182 [^16] 0.987 [^17] 0.905 [^18] 0.785 [^12] CLAP (Best) 0.9026 0.7463 0.6834 0.6436 0.9683 0.9795 0.7783

Table 1: CLAP (ZS) Zero-Shot outperforms the literature. CLAP (Best) is the best performance among our supervised setups. Higher is better for all numbers, DCASE17 employs F1, FSD50K and AudioSet employs mAP, everything else uses accuracy.

### 3.1 Datasets

Training. We used 128,010 audio and text pairs from 4 datasets to construct the training dataset for CLAP. We extracted 36,796 pairs from FSD50k [^19], 29,646 pairs from ClothoV2 [^20], 44,292 from AudioCaps [^21], 17,276 pairs from MACS [^22]. The dataset details are in appendix Section A and Table 4.  
Downstream Tasks. We used 16 datasets from 8 different domains as downstream tasks. Five tasks are Sound Event Classification. Five tasks are music related, classification of music vs speech, music genres, strokes and tonics. One task is Acoustic Scene Classification. Four are speech-based, Emotion Recognition, Keyword Spotting, and Vocal Sound Classification (e.g. cough, sneeze, laughter). One is counting speakers in a recording (0 to 10 speakers). The datasets are in Table 1, and details are in appendix Section B and Table 5.

### 3.2 Experimental setup

Pre-processing. We used log Mel spectrogram representations of audio with a sampling rate of 44.1 KHz, hop size of 320 secs, window size 1024 secs, and 64 Mel bins in the range of 50-8000 Hz. During training, each audio clip is randomly truncated to a continuous segment of 5 secs, or padded if shorter. The captions were not altered. The batches with audio and text pairs are randomly sampled at training.  
Encoders. We chose CNN14 [^23] model as the audio encoder to provide a fair comparison to previous SoTA models. The model has 80.8 million parameters, an embedding size of 2048, and was pretrained with 2M audio clips from AudioSet. The text encoder chosen is BERT [^24]. We use HuggingFace [^25] implementation of BERT base uncased. The model has 110 million parameters. We limited the max text sequence length to 100 chars for computational efficiency. The \[CLS\] token from the final layer of BERT is used as the text embedding with a size of 768. Both, the audio and text embeddings are projected into a multimodal space with two learnable projection matrices resulting in an output dimension of 1024. The temperature parameter $\tau$ is learnable and initialised to 0.007. To prevent training instability, the logits scaled by $\tau$ are clipped to a maximum value of 100.  
Training. We trained by unfreezing both encoders for 40 epochs. We use Adam Optimiser [^26] with an initial learning rate $10^{-3}$ and reduce the learning rate on plateau by $10^{-1}$ with a patience of 10. The models are implemented with PyTorch’s Distributed Data-Parallel and use 16GB V100 GPUs with scaling from 8 to 24 GPUs.

### 3.3 Evaluation setups for CLAP

Zero-shot Evaluation studies the generalisation of CLAP to unseen classes and audios. The setup is explained in Section 2.2. Instead of using the class label, we constructed a natural language prompting, ‘This is a sound of \[class label\]’. The prompt was kept the same for all the domains except three. For Emotion Recognition we used ‘this person is feeling \[class label\]’, for Keyword Spotting we only use the keyword, and for Speaker Counting we used ‘\[number between 0 - 10\] persons speaking’.  
Supervised Feature Extraction Evaluation studies the quality of audio representation learned by CLAP. Given a downstream task, we used CLAP as a feature extractor followed by training a classifier of 1 or 3 fully-connected layers (Freeze\_L1 and Freeze\_L3), similar to [^23]. We used a learning rate of $10^{-3}$ with Adam Optimizer for 30 epochs. We did not perform grid search for tuning hyperparameters due to computation constraints.  
Supervised Finetune Evaluation benchmarks CLAP against the best performance for each task in the literature. Given a downstream task, we unfroze and finetuned the audio encoder together with an attached 1 or 3 fully-connected layers. We used a learning rate of $10^{-4}$ with Adam Optimizer for 30 epochs. We did not perform grid search for tuning hyperparameters due to computation constraints.

## 4 Results and Discussion

For baseline comparisons we considered the best model performances in the literature for Zero-Shot Learning ‘Benchmark (ZS)’ and for Supervised Learning ‘Benchmark (Best)’ and reported them in Table 1. We also discuss the effect of freezing the encoders and the effect of prompts for Zero-Shot.

### 4.1 Zero-Shot (ZS) results

CLAP (ZS) achieved SoTA on established Sound Event Classification (SEC) datasets like FSD50K, US8K and ESC50. For ESC50, CLAP achieved 82.6% accuracy (acc) beating human performance of 81% and AudioCLIP (69%) by an absolute 12%. In US8K, CLAP achieved 73% acc outperforming AudioCLIP (65%) by an absolute 8%. For the multi-label dataset FSD50K, CLAP beat Wav2CLIP (3%) by an absolute 27% mAP. On task GTZAN’s Music vs Speech Classification, CLAP even beat supervised models achieving 100% acc. These results point to the possibility of having reliable audio models with no training involved.

CLAP (ZS) performed better than random on all downstream tasks and achieved good to slightly better than random on some music and speech-related tasks. CLAP achieved 47% acc in Instrument Classification, an absolute 22% higher than random. In the Vocal Sound dataset achieved 50% acc, an absolute 33% improvement over random. In Emotion Recognition (ER) and Keyword Spotting (KWS) CLAP outperformed random by up to an absolute 4% acc.

### 4.2 Supervised results

CLAP (Best) is the best performance among supervised setups and achieved SoTA on 5 datasets. CLAP achieved in GTZAN Music vs Speech Classification 100% acc, in GTZAN Music Genre Classification 91.3% acc, in Mri. Stroke Classification 97.94% acc, in Mri. Tonic Classification with 95.34% acc, and in Vocal Sounds Classification 97.95% acc. In other tasks CLAP underperformed SoTA by at most 7%. The lowest performing task was ER’s RAVDESS with 64% acc vs a SoTA of 81%.

CLAP performs better in domains like SEC than in others like ER, which was more evident in (ZS) than in (Best). We hypothesis that SEC tasks perform better because CLAP’s training data consist of audio captioning datasets, which mainly include the description of sound events, acoustic scenes, actions, and objects. On the other hand, the training data is scarce on human speech and the captions do not describe aspects of its content or context. Therefore, CLAP underperforms on human speech tasks like KWS and ER. We posit that as we increase training data and increase human speech-based captioning, CLAP’s performance on speech datasets will increase.

### 4.3 Effect of freezing CLAP encoders

We studied how freezing the audio and/or text encoder during training affected performance of the downstream tasks. We computed the average of the CLAP (ZS) performance across the downstream tasks. The results are shown in 2. The best Avg. CLAP (ZS) score is obtained by unfreezing both encoders and the worst score by freezing both encoders. This is expected because unfreezing both encoders allows them to learn the multimodal information from the pairs. Surprisingly, unfreezing the text encoder performed better than unfreezing the audio encoder. Our intuition was that unfreezing the audio encoder would enable learning beyond the SEC coming from the pretrained AudioSet information. However, unfreezing the text encoder was better for CLAP, and a similar insight was found for CLIP models in Computer Vision [^27]. This valuable finding suggests that, under the CLAP learning paradigm, it is possible to use an audio encoder of choice and turn it into a Zero-Shot classifier.

| Audio encoder (frozen) | Text encoder (frozen) | Avg. ZS score | ESC50 (acc) |
| --- | --- | --- | --- |
| ✓ | ✓ | 0.2809 | 0.5555 |
| ✗ | ✓ | 0.2818 | 0.6415 |
| ✓ | ✗ | 0.3109 | 0.7631 |
| ✗ | ✗ | 0.3265 | 0.826 |

Table 2: Effect of freezing text and/or audio encoders on CLAP (ZS) performance across all tasks and ESC50.

### 4.4 Changing prompts in Zero-Shot evaluation

The training data of CLAP consists of natural language captions containing one or more sentences. However, the vast majority of datasets have class labels defined by a few words– dog barking’ and ‘sneezing’. Using single words instead of language description affects how Zero-Shot learning transfers. To overcome this distribution difference, we used standard template prompts ‘This is a sound of \[class label\]’. Experimentally, we found that using appropriate prompts improved Zero-Shot performance of CLAP. For example, Table 3, shows how changing prompts for ESC50 leads to a 5% acc increase in performance.

| Prompt | ESC50 (acc) |
| --- | --- |
| ‘i can hear \[class label\]’ | 0.786 |
| ‘this is an audio of \[class label\]’ | 0.8005 |
| ‘\[class label\]’ | 0.812 |
| ‘this is \[class label\]’ | 0.8135 |
| ‘this is a sound of \[class label\]’ | 0.826 |

Table 3: Effect of different prompts on ESC50 (ZS).

## 5 Conclusion

We introduced CLAPfor learning audio concepts from natural language supervision. CLAP does not require gold standard class labels for training, enables flexible class prediction, and generalizes to multiple downstream tasks. The training data consists of 128k audio-text pairs which is at least 0.001% smaller than what similar Computer Vision models used. Nonetheless, CLAP establishes SoTA in Zero-Shot performance and SoTA in supervised performance for 5 tasks. Hence, CLAP shows potential for building an audio foundation model that can learn by natural language supervision and can generalize to a wide range of tasks while achieving SoTA performance.

## References

## Appendix A Training datasets

FSD50k [^19] is a sound event classification dataset with audio clips from freesound.org. The duration of the clips ranges from 0.3 to 30 seconds. We used the 36k clips from training and validation. We constructed the caption for each clip by concatenating the two sentences the associated title and description in the metadata. We ignored the class label.  
ClothoV2 [^20] is an audio captioning dataset consisting of 7k audio clips. The duration of the clips range from 15 to 30 seconds. Each clip has 5 captions annotated by different participants. Thus, we created 5 pairs for each clip extending the number of audio-text pairs by 5 times.  
AudioCaps [^21] is an audio captioning dataset consisting of 46k audio clips from AudioSet. The duration of the clips is 10 seconds. Each clip has a caption annotated via crowd-sourcing.  
MACS [^22] is an audio captioning dataset consisting of 4k audio clips. The duration of the clips is 10 seconds. Each clip is captioned by multiple participants. Similar to ClohtoV2, we paired the same audio with a each of their associated captions to create a larger set of pairs consisting of 17k. At the time of downloading the datasets, not all clips were available from the web links.

| Dataset | Pairs | Unique audios | Unique captions |
| --- | --- | --- | --- |
| FSD50k | 36,796 | 36,796 | 36,796 |
| ClothoV2 | 29,646 | 5,929 | 29,646 |
| AudioCaps | 44,292 | 44,292 | 44,292 |
| MACS | 17,276 | 3,930 | 17,276 |
|  | 128,010 | 90,947 | 128,010 |

Table 4: Training dataset statistics.

<table><tbody><tr><td>Domain</td><td>Dataset</td><td>Files</td><td>Dur. (secs)</td><td>Classes</td><td>Metric</td><td>Setup</td></tr><tr><td rowspan="5">Sound Event Classification (SEC)</td><td>ESC50</td><td>2k</td><td>5</td><td>50</td><td>ACC</td><td>5 folds</td></tr><tr><td>FSD50K</td><td>51k</td><td>0.3 - 30</td><td>200</td><td>mAP</td><td>train/val/test</td></tr><tr><td>UrbanSound8K</td><td>8k</td><td><math><semantics><mo>≤</mo> <annotation>\leq</annotation></semantics></math> 4</td><td>10</td><td>ACC</td><td>10 folds</td></tr><tr><td>DCASE2017 Task4</td><td>52k</td><td>10</td><td>17</td><td>ACC</td><td>train/val/test</td></tr><tr><td>AudioSet</td><td><math><semantics><mo>∼</mo> <csymbol>similar-to</csymbol> <annotation>\sim</annotation></semantics></math> 2M</td><td>10</td><td>527</td><td>mAP</td><td>train/val/test</td></tr><tr><td rowspan="4">Music</td><td>GTZAN Music Speech</td><td>120</td><td>30</td><td>2</td><td>ACC</td><td>10 folds</td></tr><tr><td>GTZAN Music Genre</td><td>1k</td><td>30</td><td>10</td><td>ACC</td><td>10 folds</td></tr><tr><td>Mridangam Stroke</td><td>7k</td><td>0.81</td><td>10</td><td>ACC</td><td>5 folds</td></tr><tr><td>Mridangam Tonic</td><td>7k</td><td>0.81</td><td>6</td><td>ACC</td><td>5 folds</td></tr><tr><td>Instrument Classification</td><td>Beijing Opera Percussions</td><td>236</td><td>4.77</td><td>4</td><td>ACC</td><td>5 folds</td></tr><tr><td>Acoustic Scene Classification</td><td>TUT 2017</td><td>6.3k</td><td>10</td><td>15</td><td>ACC</td><td>train/val/test</td></tr><tr><td rowspan="2">Emotion Recognition</td><td>CREMA-D</td><td>7k</td><td>5</td><td>6</td><td>ACC</td><td>5 folds</td></tr><tr><td>RAVDESS</td><td>2.5k</td><td><math><semantics><mo>≤</mo> <annotation>\leq</annotation></semantics></math> 5</td><td>8</td><td>ACC</td><td>5 folds</td></tr><tr><td>Keyword</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spotting</td><td>Speech Commands</td><td>100k</td><td>1</td><td>12</td><td>ACC</td><td>train/val/test</td></tr><tr><td>Vocal Sound Classification</td><td>Vocal Sound</td><td>21k</td><td>5</td><td>6</td><td>ACC</td><td>train/val/test</td></tr><tr><td>Speaker Counting</td><td>LibriCount 10</td><td>5k</td><td>5</td><td>11</td><td>ACC</td><td>5 folds</td></tr></tbody></table>

Table 5: Details from the 16 datasets used as Downstream Tasks.

## Appendix B Downstream datasets

ESC50 is an environmental classification dataset comprising of 50 events. The dataset consists of 2k files of 5 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.  
FSD50K is a sound event classification dataset comprising of 200 events. The dataset consists of 51k files ranging from 0.3 to 30 seconds each. The evaluation setup is train/val/test and the evaluation metric is mAP.  
UrbanSound8K is urban sound classification dataset comprising of 10 sounds. The dataset consists of 8k files of 4 seconds each. The evaluation setup is 10 fold cross validation and the evaluation metric is accuracy.  
DCASE2017 Task4 is a sound event classification dataset comprising of 17 sounds recorded in domestic environment. The dataset consists of 30k files of 10 seconds each. The evaluation setup is train/val/test and the evaluation metric is accuracy.  
AudioSet is a sound event classification dataset comprising of 527 sounds from YouTube videos. The dataset consists of 2M files of 10 seconds each. The evaluation setup is train/val/test and the evaluation metric is accuracy.  
TUT 2017 is an acoustic scene classification dataset comprising of 15 acoustic scenes in both outdoor and indoor environment. The dataset consists of 52k files of 10 seconds each. The evaluation setup is train/val/test and the evaluation metric is accuracy.  
GTZAN Music Speech is a binary classification dataset where the aim is to distinguish between human speech and music. The dataset consists of 120 files of 30 seconds each. The evaluation setup is 10 fold cross validation and the evaluation metric is accuracy.  
GTZAN Genres is music genre classification dataset comprising of 10 genres. The dataset consists of 1k files of 30 seconds each. The evaluation setup is 10 fold cross validation and the evaluation metric is accuracy.  
Mridangam Stroke is music stroke classification dataset comprising of 10 strokes from Mridangam (pitched percussion instrument). The dataset consists of 1k files of 0.81 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.  
Mridangam Tonic is music tonic classification dataset comprising of 6 tonics from Mridangam (pitched percussion instrument). The dataset consists of 1k files of 0.81 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.  
Beijing Opera Percussions is an instrument classification dataset comprising of 4 percussion instruments from Beijing Opera. The dataset consists of 236 files of 4.77 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.  
CREMA-D is an emotion recognition dataset comprising of 6 emotions. The dataset consists of 7k files of 5 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.  
RAVDESS is an emotion recognition dataset comprising of 8 emotions. The dataset consists of 2.5k files of 5 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.  
Speech Commands V2 is an keyword spotting dataset comprising of 13 commands. The dataset consists of 100k files of 1 seconds each. The evaluation setup is train/val/test and the evaluation metric is accuracy.  
Vocal Sound is a human vocal sound classification dataset comprising of 6 vocalizations. The dataset consists of 21k files of 5 seconds each. The evaluation setup is train/val/test and the evaluation metric is accuracy.  
LibriCount is a speaker count estimation dataset comprising of simulated cocktail party environment audios consisting of 0 to 10 speakers. The dataset consists of 5k files of 5 seconds each. The evaluation setup is 5 fold cross validation and the evaluation metric is accuracy.

<table><tbody><tr><td></td><td colspan="4">Sound Event Classification</td><td colspan="4">Music</td></tr><tr><td>Model</td><td>ESC50</td><td>FSD50K</td><td>US8K</td><td>DCASE17 Task 4</td><td>Music Speech</td><td>Music Genres</td><td>Mri. Stroke</td><td>Mri. Tonic</td></tr><tr><td>YAMNet</td><td>0.8375</td><td>-</td><td>-</td><td>-</td><td>0.969</td><td>0.847</td><td>-</td><td>-</td></tr><tr><td>Open L3</td><td>0.7505</td><td>0.4470</td><td>0.7823</td><td>-</td><td>0.969</td><td>0.879</td><td>0.9666</td><td>0.9369</td></tr><tr><td>Wav2CLIP</td><td>0.7589</td><td>0.3617</td><td>-</td><td>-</td><td>0.946</td><td>0.748</td><td>0.9471</td><td>0.8289</td></tr><tr><td>PaNN</td><td>0.9085</td><td>-</td><td>-</td><td>-</td><td>0.992</td><td>0.860</td><td>0.9390</td><td>0.8244</td></tr><tr><td>Wav2Vec2</td><td>0.5610</td><td>0.1164</td><td>-</td><td>-</td><td>0.946</td><td>0.780</td><td>0.9432</td><td>0.8283</td></tr><tr><td>CLAP (S)</td><td>0.9310</td><td>0.5905</td><td>0.8389</td><td>0.5330</td><td>1.0</td><td>0.7930</td><td>0.7754</td><td>0.6391</td></tr><tr><td>CLAP (F)</td><td>0.9670</td><td>0.5859</td><td>0.8796</td><td>0.5938</td><td>1.0</td><td>0.9130</td><td>0.9794</td><td>0.9534</td></tr></tbody></table>

<table><tbody><tr><td></td><td>Instrument Classification</td><td>Acoustic Scene Classification</td><td colspan="2">Emotion Recognition</td><td>Keyword Spotting</td><td>Human Vocalization</td><td>Speaker Estimation</td></tr><tr><td>Model</td><td>Beijing Opera</td><td>TUT 2017</td><td>CRE MA-D</td><td>RAV DESS</td><td>Speech Comm.</td><td>Vocal Sounds</td><td>Libri Count</td></tr><tr><td>YAMNet</td><td>0.941</td><td>-</td><td>0.453</td><td>0.479</td><td>0.4104</td><td>-</td><td>0.6526</td></tr><tr><td>OpenL3</td><td>0.975</td><td>-</td><td>0.550</td><td>0.604</td><td>0.7634</td><td>-</td><td>0.6414</td></tr><tr><td>Wav2CLIP</td><td>0.936</td><td>-</td><td>0.512</td><td>0.684</td><td>0.3466</td><td>-</td><td>0.5276</td></tr><tr><td>PaNN</td><td>0.911</td><td>-</td><td>0.555</td><td>0.429</td><td>0.6182</td><td>-</td><td>0.6516</td></tr><tr><td>Wav2Vec2</td><td>0.907</td><td>-</td><td>0.6562</td><td>-</td><td>0.8785</td><td>-</td><td>0.6921</td></tr><tr><td>CLAP (S)</td><td>0.7754</td><td>0.7099</td><td>0.2830</td><td>0.4515</td><td>0.3708</td><td>0.8411</td><td>0.5715</td></tr><tr><td>CLAP (F)</td><td>0.9026</td><td>0.7463</td><td>0.6834</td><td>0.6436</td><td>0.9683</td><td>0.9795</td><td>0.7783</td></tr></tbody></table>

Table 6: CLAP’s Supervised Feature Extraction (S) and Supervised Finetune (F) performance against SoTA models.

| Dataset | Audio captions |
| --- | --- |
| ClothoV2 | A bow playing a stringed instrument in a one note tone repeatedly before violins join to create the melody |
| ClothoV2 | An insect buzzing in the foreground as birds chirp in the background |
| ClothoV2 | A camp fire crackles as the flames burn branches and leaves |
| AudioCaps | Several sirens are wailing and a horn is honked twice |
| AudioCaps | Church bells chime loudly and repeatedly |
| AudioCaps | Beating drum getting faster than children voices and clapping then adult male voice |
| FSD50K | Water dripping in a cave or underground temple |
| FSD50K | Canada geese flying down and landing near a lakeshore. Recorded with an Olympus LS-14. |
| FSD50K | Leaves falling in a forest near a pond. Recorded in October 2017 in a German forest using a Zoom H2n. |
| MACS | Two people having a conversation nearby while a lot of adults and a child talk far away |
| MACS | Birds are making a lot of noises and a distant child yells |
| MACS | Dog barks followed by adults talking and children voices |

Table 7: Randomly sampled raw captions from each dataset.

## Appendix C Discussion

### C.1 Batch size and CLAP performance

Appropriate batch size is a key contributor in performance of any contrastive learning methods that uses positive-negative pairs. In computer vision literature, larger batch size has shown to improve performance [^6] [^7] [^27].

We use Tesla V100 16 GB GPUs for this analysis. The number of GPUs used vary from 4 to 24 for batch size from 32 to 768. We measure the zero-shot performance by computing average performance across N downstream tasks listed in table 5. Our findings also indicate that the larger batch size does lead to improved CLAP performance. However, we saw decreased performance with batch size of 768. This might be an anamoly in an increasing zero-shot performance trend. We leave the larger batch size investigation to future work.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2206.04769/assets/figures/batch_size.png)

Fig. 2: Effect of batch size on zero-shot performance

### C.2 Training with AudioSet

AudioSet is a multilabled sound event dataset with 2M audio clips, but it does not come with a text descriptions per clip, thus making it complicated to extract audio and text pairs for training. We tried constructing the text description with the title and the class label(s). However, adding about 1.7M pairs to the existing 128k pairs resulted in a performance dropped in the overall zero-shot numbers. For example, ESC50 performance dropped from 82.6% to 67.15% acc and US8K dropped from 73.24% to 70.93% acc. Only Speech Commands V2 (SCV2) performance improved from 10% to 15% acc. Perhaps due to the large amount of audio containing speech in AudioSet. The quality of audio-text pairs is key in training CLAP. In AudioSet, often the YouTube titles and descriptions do not describe the acoustic content of the video segment under consideration but instead describe the video as a whole. More intelligent ways of generating descriptions can benefit CLAP training with AudioSet and other many similar datasets. In general, finding helpful training data for CLAP based on public datasets is difficult, thus relying on large-scale noisy pairs is the only scalable approach.

[^1]: Richard F Lyon, “Machine hearing: An emerging field \[exploratory dsp\],” IEEE signal processing magazine, vol. 27, pp. 131–139, 2010.

[^2]: Annamaria Mesaros, Aleksandr Diment, Benjamin Elizalde, Toni Heittola, Emmanuel Vincent, Bhiksha Raj, and Tuomas Virtanen, “Sound event detection in the dcase 2017 challenge,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 6, pp. 992–1006, 2019.

[^3]: Yu Zhang, Daniel S Park, Wei Han, James Qin, Anmol Gulati, Joel Shor, Aren Jansen, Yuanzhong Xu, Yanping Huang, Shibo Wang, et al., “Bigssl: Exploring the frontier of large-scale semi-supervised learning for automatic speech recognition,” arXiv preprint arXiv:2109.13226, 2021.

[^4]: Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al., “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” arXiv preprint arXiv:2110.13900, 2021.

[^5]: Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in Neural Information Processing Systems, vol. 33, pp. 12449–12460, 2020.

[^6]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.

[^7]: Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al., “Florence: A new foundation model for computer vision,” arXiv preprint arXiv:2111.11432, 2021.

[^8]: Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International Conference on Machine Learning. PMLR, 2021, pp. 4904–4916.

[^9]: Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, and Juan Pablo Bello, “Wav2clip: Learning robust audio representations from clip,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 4563–4567.

[^10]: Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel, “Audioclip: Extending clip to image, text and audio,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 976–980.

[^11]: Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 776–780.

[^12]: Joseph Turian, Jordie Shier, Humair Raj Khan, Bhiksha Raj, Björn W Schuller, Christian J Steinmetz, Colin Malloy, George Tzanetakis, Gissel Velarde, Kirk McNally, et al., “Hear 2021: Holistic evaluation of audio representations,” arXiv preprint arXiv:2203.03022, 2022.

[^13]: Qiuqiang Kong, Yong Xu, Wenwu Wang, and Mark D Plumbley, “Sound event detection of weakly labelled data with cnn-transformer and automatic threshold optimization,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2450–2460, 2020.

[^14]: Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, “Efficient training of audio transformers with patchout,” arXiv preprint arXiv:2110.05069, 2021.

[^15]: Yoonchang Han, Jeongsoo Park, and Kyogu Lee, “Convolutional neural networks with binaural representations and background subtraction for acoustic scene classification,” the Detection and Classification of Acoustic Scenes and Events (DCASE), pp. 1–5, 2017.

[^16]: Cristina Luna-Jiménez, Ricardo Kleinlein, David Griol, Zoraida Callejas, Juan M. Montero, and Fernando Fernández-Martínez, “A proposal for multimodal emotion recognition using aural transformers and action units on ravdess dataset,” Applied Sciences, vol. 12, no. 1, 2022.

[^17]: Byeonggeun Kim, Simyung Chang, Jinkyu Lee, and Dooyong Sung, “Broadcasted residual learning for efficient keyword spotting,” arXiv preprint arXiv:2106.04140, 2021.

[^18]: Yuan Gong, Jin Yu, and James Glass, “Vocalsound: A dataset for improving human vocal sounds recognition,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 151–155.

[^19]: Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra, “Fsd50k: An open dataset of human-labeled sound events,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 829–852, 2022.

[^20]: Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen, “Clotho: an audio captioning dataset,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 736–740.

[^21]: Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim, “AudioCaps: Generating Captions for Audios in The Wild,” in NAACL-HLT, 2019.

[^22]: Irene Martín-Morató and Annamaria Mesaros, “What is the ground truth? reliability of multi-annotator data for audio tagging,” in 2021 29th European Signal Processing Conference (EUSIPCO). IEEE, 2021, pp. 76–80.

[^23]: Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE/ACM Trans. Audio, Speech and Lang. Proc., vol. 28, pp. 2880–2894, jan 2020.

[^24]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota, June 2019, pp. 4171–4186, Association for Computational Linguistics.

[^25]: Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al., “Huggingface’s transformers: State-of-the-art natural language processing,” arXiv preprint arXiv:1910.03771, 2019.

[^26]: Diederik P. Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” in ICLR (Poster), 2015.

[^27]: Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer, “Lit: Zero-shot transfer with locked-image text tuning,” arXiv preprint arXiv:2111.07991, 2021.