---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
source: "https://arxiv.org/html/1810.04805v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/01_%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%9E%B6%E6%9E%84/BERT%2C%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding%2C%20Jacob%20Devlin%20et%20al.%2C%202018.no_watermark.zh-CN.dual.pdf"
---
Jacob Devlin  Ming-Wei Chang  Kenton Lee  Kristina Toutanova  
Google AI Language  
{jacobdevlin,mingweichang,kentonl,kristout}@google.com

###### Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models [^36] [^38], BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## 1 Introduction

Language model pre-training has been shown to be effective for improving many natural language processing tasks [^15] [^36] [^38] [^21]. These include sentence-level tasks such as natural language inference [^6] [^51] and paraphrasing [^17], which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level [^44] [^39].

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo [^36], uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) [^38], introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters. The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer [^46]. Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT alleviates the previously mentioned unidirectionality constraint by using a “masked language model” (MLM) pre-training objective, inspired by the Cloze task [^43]. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also use a “next sentence prediction” task that jointly pre-trains text-pair representations. The contributions of our paper are as follows:

- We demonstrate the importance of bidirectional pre-training for language representations. Unlike [^38], which uses unidirectional language models for pre-training, BERT uses masked language models to enable pre-trained deep bidirectional representations. This is also in contrast to [^36], which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs.
- We show that pre-trained representations reduce the need for many heavily-engineered task-specific architectures. BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.
- BERT advances the state of the art for eleven NLP tasks. The code and pre-trained models are available at [https://github.com/google-research/bert](https://github.com/google-research/bert).

## 2 Related Work

There is a long history of pre-training general language representations, and we briefly review the most widely-used approaches in this section.

### 2.1 Unsupervised Feature-based Approaches

Learning widely applicable representations of words has been an active area of research for decades, including non-neural [^7] [^3] [^5] and neural [^31] [^34] methods. Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch [^45]. To pre-train word embedding vectors, left-to-right language modeling objectives have been used [^32], as well as objectives to discriminate correct from incorrect words in left and right context [^31].

These approaches have been generalized to coarser granularities, such as sentence embeddings [^25] [^28] or paragraph embeddings [^26]. To train sentence representations, prior work has used objectives to rank candidate next sentences [^23] [^28], left-to-right generation of next sentence words given a representation of the previous sentence [^25], or denoising auto-encoder derived objectives [^20].

![Refer to caption](https://arxiv.org/html/1810.04805v2/x1.png)

Figure 1: Overall pre-training and fine-tuning procedures for BERT. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. During fine-tuning, all parameters are fine-tuned. \[CLS\] is a special symbol added in front of every input example, and \[SEP\] is a special separator token (e.g. separating questions/answers).

ELMo and its predecessor [^35] [^36] generalize traditional word embedding research along a different dimension. They extract *context-sensitive* features from a left-to-right and a right-to-left language model. The contextual representation of each token is the concatenation of the left-to-right and right-to-left representations. When integrating contextual word embeddings with existing task-specific architectures, ELMo advances the state of the art for several major NLP benchmarks [^36] including question answering [^39], sentiment analysis [^41], and named entity recognition [^44]. [^30] proposed learning contextual representations through a task to predict a single word from both left and right context using LSTMs. Similar to ELMo, their model is feature-based and not deeply bidirectional. [^18] shows that the cloze task can be used to improve the robustness of text generation models.

### 2.2 Unsupervised Fine-tuning Approaches

As with the feature-based approaches, the first works in this direction only pre-trained word embedding parameters from unlabeled text [^13].

More recently, sentence or document encoders which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task [^15] [^21] [^38]. The advantage of these approaches is that few parameters need to be learned from scratch. At least partly due to this advantage, OpenAI GPT [^38] achieved previously state-of-the-art results on many sentence-level tasks from the GLUE benchmark [^48]. Left-to-right language modeling and auto-encoder objectives have been used for pre-training such models [^21] [^38] [^15].

### 2.3 Transfer Learning from Supervised Data

There has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference [^14] and machine translation [^29]. Computer vision research has also demonstrated the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained with ImageNet [^16] [^53].

## 3 BERT

We introduce BERT and its detailed implementation in this section. There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section.

A distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.

#### Model Architecture

BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in [^46] and released in the tensor2tensor library.<sup>1</sup> Because the use of Transformers has become common and our implementation is almost identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to [^46] as well as excellent guides such as “The Annotated Transformer.” <sup>2</sup>

In this work, we denote the number of layers (i.e., Transformer blocks) as $L$, the hidden size as $H$, and the number of self-attention heads as $A$.<sup>3</sup> We primarily report results on two model sizes: BERT ${}_{\textsc{BASE}}$ (L=12, H=768, A=12, Total Parameters=110M) and BERT ${}_{\textsc{LARGE}}$ (L=24, H=1024, A=16, Total Parameters=340M).

BERT ${}_{\textsc{BASE}}$ was chosen to have the same model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.<sup>4</sup>

#### Input/Output Representations

To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., $\langle$ Question, Answer $\rangle$) in one token sequence. Throughout this work, a “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

We use WordPiece embeddings [^52] with a 30,000 token vocabulary. The first token of every sequence is always a special classification token (\[CLS\]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token (\[SEP\]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. As shown in Figure 1, we denote input embedding as $E$, the final hidden vector of the special \[CLS\] token as $C\in\mathbb{R}^{H}$, and the final hidden vector for the $i^{\rm th}$ input token as $T_{i}\in\mathbb{R}^{H}$.

For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings. A visualization of this construction can be seen in Figure 2.

Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.

### 3.1 Pre-training BERT

Unlike [^36] and [^38], we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step is presented in the left part of Figure 1.

#### Task #1: Masked LM

Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.

In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature [^43]. In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders [^47], we only predict the masked words rather than reconstructing the entire input.

Although this allows us to obtain a bidirectional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the \[MASK\] token does not appear during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual \[MASK\] token. The training data generator chooses 15% of the token positions at random for prediction. If the $i$ -th token is chosen, we replace the $i$ -th token with (1) the \[MASK\] token 80% of the time (2) a random token 10% of the time (3) the unchanged $i$ -th token 10% of the time. Then, $T_{i}$ will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.

#### Task #2: Next Sentence Prediction (NSP)

Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, $C$ is used for next sentence prediction (NSP).<sup>5</sup> Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI. <sup>6</sup> The NSP task is closely related to representation-learning objectives used in [^23] and [^28]. However, in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters.

Pre-training data The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the BooksCorpus (800M words) [^56] and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark [^9] in order to extract long contiguous sequences.

### 3.2 Fine-tuning BERT

Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks—whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs. For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention, such as [^33] [^40]. BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes *bidirectional* cross attention between two sentences.

For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end. At the input, sentence A and sentence B from pre-training are analogous to (1) sentence pairs in paraphrasing, (2) hypothesis-premise pairs in entailment, (3) question-passage pairs in question answering, and (4) a degenerate text- $\varnothing$ pair in text classification or sequence tagging. At the output, the token representations are fed into an output layer for token-level tasks, such as sequence tagging or question answering, and the \[CLS\] representation is fed into an output layer for classification, such as entailment or sentiment analysis.

Compared to pre-training, fine-tuning is relatively inexpensive. All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model.<sup>7</sup> We describe the task-specific details in the corresponding subsections of Section 4. More details can be found in Appendix A.5.

| System | MNLI-(m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 392k | 363k | 108k | 67k | 8.5k | 5.7k | 3.5k | 2.5k | \- |
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT ${}_{\textsc{BASE}}$ | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| BERT ${}_{\textsc{LARGE}}$ | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | 82.1 |

Table 1: GLUE Test results, scored by the evaluation server ([https://gluebenchmark.com/leaderboard](https://gluebenchmark.com/leaderboard)). The number below each task denotes the number of training examples. The “Average” column is slightly different than the official GLUE score, since we exclude the problematic WNLI set.<sup>9</sup> BERT and OpenAI GPT are single-model, single task. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks. We exclude entries that use BERT as one of their components.

<sup>9</sup>

## 4 Experiments

In this section, we present BERT fine-tuning results on 11 NLP tasks.

### 4.1 GLUE

The General Language Understanding Evaluation (GLUE) benchmark [^48] is a collection of diverse natural language understanding tasks. Detailed descriptions of GLUE datasets are included in Appendix B.1.

To fine-tune on GLUE, we represent the input sequence (for single sentence or sentence pairs) as described in Section 3, and use the final hidden vector $C\in\mathbb{R}^{H}$ corresponding to the first input token (\[CLS\]) as the aggregate representation. The only new parameters introduced during fine-tuning are classification layer weights $W\in\mathbb{R}^{K\times H}$, where $K$ is the number of labels. We compute a standard classification loss with $C$ and $W$, i.e., $\log({\rm softmax}(CW^{T}))$.

We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set. Additionally, for BERT ${}_{\textsc{LARGE}}$ we found that fine-tuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set. With random restarts, we use the same pre-trained checkpoint but perform different fine-tuning data shuffling and classifier layer initialization.<sup>10</sup>

Results are presented in Table 9. Both BERT ${}_{\textsc{BASE}}$ and BERT ${}_{\textsc{LARGE}}$ outperform all systems on all tasks by a substantial margin, obtaining 4.5% and 7.0% respective average accuracy improvement over the prior state of the art. Note that BERT ${}_{\textsc{BASE}}$ and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking. For the largest and most widely reported GLUE task, MNLI, BERT obtains a 4.6% absolute accuracy improvement. On the official GLUE leaderboard <sup>11</sup>, BERT ${}_{\textsc{LARGE}}$ obtains a score of 80.5, compared to OpenAI GPT, which obtains 72.8 as of the date of writing.

We find that BERT ${}_{\textsc{LARGE}}$ significantly outperforms BERT ${}_{\textsc{BASE}}$ across all tasks, especially those with very little training data. The effect of model size is explored more thoroughly in Section 5.2.

### 4.2 SQuAD v1.1

The Stanford Question Answering Dataset (SQuAD v1.1) is a collection of 100k crowdsourced question/answer pairs [^39]. Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage.

As shown in Figure 1, in the question answering task, we represent the input question and passage as a single packed sequence, with the question using the A embedding and the passage using the B embedding. We only introduce a start vector $S\in\mathbb{R}^{H}$ and an end vector $E\in\mathbb{R}^{H}$ during fine-tuning. The probability of word $i$ being the start of the answer span is computed as a dot product between $T_{i}$ and $S$ followed by a softmax over all of the words in the paragraph: $P_{i}=\frac{e^{S{\cdot}T_{i}}}{\sum_{j}e^{S{\cdot}T_{j}}}$. The analogous formula is used for the end of the answer span. The score of a candidate span from position $i$ to position $j$ is defined as $S{\cdot}T_{i}+E{\cdot}T_{j}$, and the maximum scoring span where $j\geq i$ is used as a prediction. The training objective is the sum of the log-likelihoods of the correct start and end positions. We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32.

<table><tbody><tr><th>System</th><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><th></th><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><th colspan="5">Top Leaderboard Systems (Dec 10th, 2018)</th></tr><tr><th>Human</th><td>-</td><td>-</td><td>82.3</td><td>91.2</td></tr><tr><th>#1 Ensemble - nlnet</th><td>-</td><td>-</td><td>86.0</td><td>91.7</td></tr><tr><th>#2 Ensemble - QANet</th><td>-</td><td>-</td><td>84.5</td><td>90.5</td></tr><tr><th colspan="5">Published</th></tr><tr><th>BiDAF+ELMo (Single)</th><td>-</td><td>85.6</td><td>-</td><td>85.8</td></tr><tr><th>R.M. Reader (Ensemble)</th><td>81.2</td><td>87.9</td><td>82.3</td><td>88.5</td></tr><tr><th colspan="5">Ours</th></tr><tr><th>BERT <math><semantics><msub><mtext>BASE</mtext></msub> <annotation>{}_{\textsc{BASE}}</annotation></semantics></math> (Single)</th><td>80.8</td><td>88.5</td><td>-</td><td>-</td></tr><tr><th>BERT <math><semantics><msub><mtext>LARGE</mtext></msub> <annotation>{}_{\textsc{LARGE}}</annotation></semantics></math> (Single)</th><td>84.1</td><td>90.9</td><td>-</td><td>-</td></tr><tr><th>BERT <math><semantics><msub><mtext>LARGE</mtext></msub> <annotation>{}_{\textsc{LARGE}}</annotation></semantics></math> (Ensemble)</th><td>85.8</td><td>91.8</td><td>-</td><td>-</td></tr><tr><th>BERT <math><semantics><msub><mtext>LARGE</mtext></msub> <annotation>{}_{\textsc{LARGE}}</annotation></semantics></math> (Sgl.+TriviaQA)</th><td>84.2</td><td>91.1</td><td>85.1</td><td>91.8</td></tr><tr><th>BERT <math><semantics><msub><mtext>LARGE</mtext></msub> <annotation>{}_{\textsc{LARGE}}</annotation></semantics></math> (Ens.+TriviaQA)</th><td>86.2</td><td>92.2</td><td>87.4</td><td>93.2</td></tr></tbody></table>

Table 2: SQuAD 1.1 results. The BERT ensemble is 7x systems which use different pre-training checkpoints and fine-tuning seeds.

<table><tbody><tr><th>System</th><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><th></th><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><th colspan="5">Top Leaderboard Systems (Dec 10th, 2018)</th></tr><tr><th>Human</th><td>86.3</td><td>89.0</td><td>86.9</td><td>89.5</td></tr><tr><th>#1 Single - MIR-MRC (F-Net)</th><td>-</td><td>-</td><td>74.8</td><td>78.0</td></tr><tr><th>#2 Single - nlnet</th><td>-</td><td>-</td><td>74.2</td><td>77.1</td></tr><tr><th colspan="5">Published</th></tr><tr><th>unet (Ensemble)</th><td>-</td><td>-</td><td>71.4</td><td>74.9</td></tr><tr><th>SLQA+ (Single)</th><td>-</td><td></td><td>71.4</td><td>74.4</td></tr><tr><th colspan="5">Ours</th></tr><tr><th>BERT <math><semantics><msub><mtext>LARGE</mtext></msub> <annotation>{}_{\textsc{LARGE}}</annotation></semantics></math> (Single)</th><td>78.7</td><td>81.9</td><td>80.0</td><td>83.1</td></tr></tbody></table>

Table 3: SQuAD 2.0 results. We exclude entries that use BERT as one of their components.

Table 2 shows top leaderboard entries as well as results from top published systems [^40] [^11] [^36] [^22]. The top results from the SQuAD leaderboard do not have up-to-date public system descriptions available,<sup>12</sup> and are allowed to use any public data when training their systems. We therefore use modest data augmentation in our system by first fine-tuning on TriviaQA [^24] befor fine-tuning on SQuAD.

Our best performing system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. In fact, our single BERT model outperforms the top ensemble system in terms of F1 score. Without TriviaQA fine-tuning data, we only lose 0.1-0.4 F1, still outperforming all existing systems by a wide margin.<sup>13</sup>

### 4.3 SQuAD v2.0

The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.

We use a simple approach to extend the SQuAD v1.1 BERT model for this task. We treat questions that do not have an answer as having an answer span with start and end at the \[CLS\] token. The probability space for the start and end answer span positions is extended to include the position of the \[CLS\] token. For prediction, we compare the score of the no-answer span: $s_{\tt null}=S{\cdot}C+E{\cdot}C$ to the score of the best non-null span $\hat{s_{i,j}}$ = ${\tt max}_{j\geq i}S{\cdot}T_{i}+E{\cdot}T_{j}$. We predict a non-null answer when $\hat{s_{i,j}}>s_{\tt null}+\tau$, where the threshold $\tau$ is selected on the dev set to maximize F1. We did not use TriviaQA data for this model. We fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.

The results compared to prior leaderboard entries and top published work [^42] [^49] are shown in Table 3, excluding systems that use BERT as one of their components. We observe a +5.1 F1 improvement over the previous best system.

### 4.4 SWAG

The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference [^55]. Given a sentence, the task is to choose the most plausible continuation among four choices.

When fine-tuning on the SWAG dataset, we construct four input sequences, each containing the concatenation of the given sentence (sentence A) and a possible continuation (sentence B). The only task-specific parameters introduced is a vector whose dot product with the \[CLS\] token representation $C$ denotes a score for each choice which is normalized with a softmax layer.

We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16. Results are presented in Table 4. BERT ${}_{\textsc{LARGE}}$ outperforms the authors’ baseline ESIM+ELMo system by +27.1% and OpenAI GPT by 8.3%.

| System | Dev | Test |
| --- | --- | --- |
| ESIM+GloVe | 51.9 | 52.7 |
| ESIM+ELMo | 59.1 | 59.2 |
| OpenAI GPT | \- | 78.0 |
| BERT ${}_{\textsc{BASE}}$ | 81.6 | \- |
| BERT ${}_{\textsc{LARGE}}$ | 86.6 | 86.3 |
| Human (expert) <sup>†</sup> | \- | 85.0 |
| Human (5 annotations) <sup>†</sup> | \- | 88.0 |

Table 4: SWAG Dev and Test accuracies. <sup>†</sup> Human performance is measured with 100 samples, as reported in the SWAG paper.

## 5 Ablation Studies

In this section, we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance. Additional ablation studies can be found in Appendix C.

### 5.1 Effect of Pre-training Tasks

We demonstrate the importance of the deep bidirectionality of BERT by evaluating two pre-training objectives using exactly the same pre-training data, fine-tuning scheme, and hyperparameters as BERT ${}_{\textsc{BASE}}$:

No NSP: A bidirectional model which is trained using the “masked LM” (MLM) but without the “next sentence prediction” (NSP) task.  
LTR & No NSP: A left-context-only model which is trained using a standard Left-to-Right (LTR) LM, rather than an MLM. The left-only constraint was also applied at fine-tuning, because removing it introduced a pre-train/fine-tune mismatch that degraded downstream performance. Additionally, this model was pre-trained without the NSP task. This is directly comparable to OpenAI GPT, but using our larger training dataset, our input representation, and our fine-tuning scheme.

<table><thead><tr><th></th><th colspan="5">Dev Set</th></tr><tr><th>Tasks</th><th>MNLI-m</th><th>QNLI</th><th>MRPC</th><th>SST-2</th><th>SQuAD</th></tr><tr><th></th><th>(Acc)</th><th>(Acc)</th><th>(Acc)</th><th>(Acc)</th><th>(F1)</th></tr></thead><tbody><tr><th>BERT <math><semantics><msub><mtext>BASE</mtext></msub> <annotation>{}_{\textsc{BASE}}</annotation></semantics></math></th><td>84.4</td><td>88.4</td><td>86.7</td><td>92.7</td><td>88.5</td></tr><tr><th>No NSP</th><td>83.9</td><td>84.9</td><td>86.5</td><td>92.6</td><td>87.9</td></tr><tr><th>LTR & No NSP</th><td>82.1</td><td>84.3</td><td>77.5</td><td>92.1</td><td>77.8</td></tr><tr><th>+ BiLSTM</th><td>82.1</td><td>84.1</td><td>75.7</td><td>91.6</td><td>84.9</td></tr></tbody></table>

Table 5: Ablation over the pre-training tasks using the BERT ${}_{\textsc{BASE}}$ architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a randomly initialized BiLSTM on top of the “LTR + No NSP” model during fine-tuning.

We first examine the impact brought by the NSP task. In Table 5, we show that removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1. Next, we evaluate the impact of training bidirectional representations by comparing “No NSP” to “LTR & No NSP”. The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD.

For SQuAD it is intuitively clear that a LTR model will perform poorly at token predictions, since the token-level hidden states have no right-side context. In order to make a good faith attempt at strengthening the LTR system, we added a randomly initialized BiLSTM on top. This does significantly improve results on SQuAD, but the results are still far worse than those of the pre-trained bidirectional models. The BiLSTM hurts performance on the GLUE tasks.

We recognize that it would also be possible to train separate LTR and RTL models and represent each token as the concatenation of the two models, as ELMo does. However: (a) this is twice as expensive as a single bidirectional model; (b) this is non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question; (c) this it is strictly less powerful than a deep bidirectional model, since it can use both left and right context at every layer.

### 5.2 Effect of Model Size

In this section, we explore the effect of model size on fine-tuning task accuracy. We trained a number of BERT models with a differing number of layers, hidden units, and attention heads, while otherwise using the same hyperparameters and training procedure as described previously.

Results on selected GLUE tasks are shown in Table 6. In this table, we report the average Dev Set accuracy from 5 random restarts of fine-tuning. We can see that larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples, and is substantially different from the pre-training tasks. It is also perhaps surprising that we are able to achieve such significant improvements on top of models which are already quite large relative to the existing literature. For example, the largest Transformer explored in [^46] is (L=6, H=1024, A=16) with 100M parameters for the encoder, and the largest Transformer we have found in the literature is (L=64, H=512, A=2) with 235M parameters [^2]. By contrast, BERT ${}_{\textsc{BASE}}$ contains 110M parameters and BERT ${}_{\textsc{LARGE}}$ contains 340M parameters.

<table><thead><tr><th colspan="3">Hyperparams</th><th></th><th colspan="3">Dev Set Accuracy</th></tr><tr><th>#L</th><th>#H</th><th>#A</th><th>LM (ppl)</th><th>MNLI-m</th><th>MRPC</th><th>SST-2</th></tr></thead><tbody><tr><th>3</th><th>768</th><th>12</th><td>5.84</td><td>77.9</td><td>79.8</td><td>88.4</td></tr><tr><th>6</th><th>768</th><th>3</th><td>5.24</td><td>80.6</td><td>82.2</td><td>90.7</td></tr><tr><th>6</th><th>768</th><th>12</th><td>4.68</td><td>81.9</td><td>84.8</td><td>91.3</td></tr><tr><th>12</th><th>768</th><th>12</th><td>3.99</td><td>84.4</td><td>86.7</td><td>92.9</td></tr><tr><th>12</th><th>1024</th><th>16</th><td>3.54</td><td>85.7</td><td>86.9</td><td>93.3</td></tr><tr><th>24</th><th>1024</th><th>16</th><td>3.23</td><td>86.6</td><td>87.8</td><td>93.7</td></tr></tbody></table>

Table 6: Ablation over BERT model size. #L = the number of layers; #H = hidden size; #A = number of attention heads. “LM (ppl)” is the masked LM perplexity of held-out training data.

It has long been known that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling, which is demonstrated by the LM perplexity of held-out training data shown in Table 6. However, we believe that this is the first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained. [^37] presented mixed results on the downstream task impact of increasing the pre-trained bi-LM size from two to four layers and [^30] mentioned in passing that increasing hidden dimension size from 200 to 600 helped, but increasing further to 1,000 did not bring further improvements. Both of these prior works used a feature-based approach — we hypothesize that when the model is fine-tuned directly on the downstream tasks and uses only a very small number of randomly initialized additional parameters, the task-specific models can benefit from the larger, more expressive pre-trained representations even when downstream task data is very small.

### 5.3 Feature-based Approach with BERT

All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added to the pre-trained model, and all parameters are jointly fine-tuned on a downstream task. However, the feature-based approach, where fixed features are extracted from the pre-trained model, has certain advantages. First, not all tasks can be easily represented by a Transformer encoder architecture, and therefore require a task-specific model architecture to be added. Second, there are major computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top of this representation.

In this section, we compare the two approaches by applying BERT to the CoNLL-2003 Named Entity Recognition (NER) task [^44]. In the input to BERT, we use a case-preserving WordPiece model, and we include the maximal document context provided by the data. Following standard practice, we formulate this as a tagging task but do not use a CRF layer in the output. We use the representation of the first sub-token as the input to the token-level classifier over the NER label set.

To ablate the fine-tuning approach, we apply the feature-based approach by extracting the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiLSTM before the classification layer.

Results are presented in Table 7. BERT ${}_{\textsc{LARGE}}$ performs competitively with state-of-the-art methods. The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer, which is only 0.3 F1 behind fine-tuning the entire model. This demonstrates that BERT is effective for both fine-tuning and feature-based approaches.

| System | Dev F1 | Test F1 |
| --- | --- | --- |
| ELMo [^36] | 95.7 | 92.2 |
| CVT [^12] | \- | 92.6 |
| CSE [^1] | \- | 93.1 |
| Fine-tuning approach |  |  |
| BERT ${}_{\textsc{LARGE}}$ | 96.6 | 92.8 |
| BERT ${}_{\textsc{BASE}}$ | 96.4 | 92.4 |
| Feature-based approach (BERT ${}_{\textsc{BASE}}$) |  |  |
| Embeddings | 91.0 | \- |
| Second-to-Last Hidden | 95.6 | \- |
| Last Hidden | 94.9 | \- |
| Weighted Sum Last Four Hidden | 95.9 | \- |
| Concat Last Four Hidden | 96.1 | \- |
| Weighted Sum All 12 Layers | 95.5 | \- |

Table 7: CoNLL-2003 Named Entity Recognition results. Hyperparameters were selected using the Dev set. The reported Dev and Test scores are averaged over 5 random restarts using those hyperparameters.

## 6 Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures. Our major contribution is further generalizing these findings to deep *bidirectional* architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.

## References

Appendix for “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”

We organize the appendix into three sections:

- Additional implementation details for BERT are presented in Appendix A;
- Additional details for our experiments are presented in Appendix B; and
- Additional ablation studies are presented in Appendix C.
	We present additional ablation studies for BERT including:
	- Effect of Number of Training Steps; and
	- Ablation for Different Masking Procedures.

## Appendix A Additional Details for BERT

![Refer to caption](https://arxiv.org/html/1810.04805v2/x3.png)

Figure 3: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-to-left LSTMs to generate features for downstream tasks. Among the three, only BERT representations are jointly conditioned on both left and right context in all layers. In addition to the architecture differences, BERT and OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.

### A.1 Illustration of the Pre-training Tasks

We provide examples of the pre-training tasks in the following.

#### Masked LM and the Masking Procedure

Assuming the unlabeled sentence is my dog is hairy, and during the random masking procedure we chose the 4-th token (which corresponding to hairy), our masking procedure can be further illustrated by

- 80% of the time: Replace the word with the \[MASK\] token, e.g., my dog is hairy $\rightarrow$ my dog is \[MASK\]
- 10% of the time: Replace the word with a random word, e.g., my dog is hairy $\rightarrow$ my dog is apple
- 10% of the time: Keep the word unchanged, e.g., my dog is hairy $\rightarrow$ my dog is hairy. The purpose of this is to bias the representation towards the actual observed word.

The advantage of this procedure is that the Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Additionally, because random replacement only occurs for 1.5% of all tokens (i.e., 10% of 15%), this does not seem to harm the model’s language understanding capability. In Section C.2, we evaluate the impact this procedure.

Compared to standard langauge model training, the masked LM only make predictions on 15% of tokens in each batch, which suggests that more pre-training steps may be required for the model to converge. In Section C.1 we demonstrate that MLM does converge marginally slower than a left-to-right model (which predicts every token), but the empirical improvements of the MLM model far outweigh the increased training cost.

#### Next Sentence Prediction

The next sentence prediction task can be illustrated in the following examples.

$$
\displaystyle=\text{\tt{[CLS] the man went to [MASK] store [SEP]}}
$$
 
$$
\displaystyle=\text{\tt{IsNext}}
$$
 
$$
\displaystyle=\text{\tt{[CLS] the man [MASK] to the store [SEP]}}
$$
 
$$
\displaystyle=\text{\tt{NotNext}}
$$

### A.2 Pre-training Procedure

To generate each training input sequence, we sample two spans of text from the corpus, which we refer to as “sentences” even though they are typically much longer than single sentences (but can be shorter also). The first sentence receives the A embedding and the second receives the B embedding. 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence, which is done for the “next sentence prediction” task. They are sampled such that the combined length is $\leq$ 512 tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%, and no special consideration given to partial word pieces.

We train with batch size of 256 sequences (256 sequences \* 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. We use Adam with learning rate of 1e-4, ${\beta}_{1}=0.9$, ${\beta}_{2}=0.999$, L2 weight decay of $0.01$, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a gelu activation [^19] rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood.

Training of BERT ${}_{\textsc{BASE}}$ was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total).<sup>14</sup> Training of BERT ${}_{\textsc{LARGE}}$ was performed on 16 Cloud TPUs (64 TPU chips total). Each pre-training took 4 days to complete.

Longer sequences are disproportionately expensive because attention is quadratic to the sequence length. To speed up pretraing in our experiments, we pre-train the model with sequence length of 128 for 90% of the steps. Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings.

### A.3 Fine-tuning Procedure

For fine-tuning, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The dropout probability was always kept at 0.1. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4

We also observed that large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to simply run an exhaustive search over the above parameters and choose the model that performs best on the development set.

### A.4 Comparison of BERT, ELMo,and OpenAI GPT

Here we studies the differences in recent popular representation learning models including ELMo, OpenAI GPT and BERT. The comparisons between the model architectures are shown visually in Figure 3. Note that in addition to the architecture differences, BERT and OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.

The most comparable existing pre-training method to BERT is OpenAI GPT, which trains a left-to-right Transformer LM on a large text corpus. In fact, many of the design decisions in BERT were intentionally made to make it as close to GPT as possible so that the two methods could be minimally compared. The core argument of this work is that the bi-directionality and the two pre-training tasks presented in Section 3.1 account for the majority of the empirical improvements, but we do note that there are several other differences between how BERT and GPT were trained:

- GPT is trained on the BooksCorpus (800M words); BERT is trained on the BooksCorpus (800M words) and Wikipedia (2,500M words).
- GPT uses a sentence separator (\[SEP\]) and classifier token (\[CLS\]) which are only introduced at fine-tuning time; BERT learns \[SEP\], \[CLS\] and sentence A/B embeddings during pre-training.
- GPT was trained for 1M steps with a batch size of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words.
- GPT used the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate which performs the best on the development set.

To isolate the effect of these differences, we perform ablation experiments in Section 5.1 which demonstrate that the majority of the improvements are in fact coming from the two pre-training tasks and the bidirectionality they enable.

### A.5 Illustrations of Fine-tuning on Different Tasks

The illustration of fine-tuning BERT on different tasks can be seen in Figure 4. Our task-specific models are formed by incorporating BERT with one additional output layer, so a minimal number of parameters need to be learned from scratch. Among the tasks, (a) and (b) are sequence-level tasks while (c) and (d) are token-level tasks. In the figure, $E$ represents the input embedding, $T_{i}$ represents the contextual representation of token $i$, \[CLS\] is the special symbol for classification output, and \[SEP\] is the special symbol to separate non-consecutive token sequences.

![Refer to caption](https://arxiv.org/html/1810.04805v2/x4.png)

Figure 4: Illustrations of Fine-tuning BERT on Different Tasks.

## Appendix B Detailed Experimental Setup

### B.1 Detailed Descriptions for the GLUE Benchmark Experiments.

Our GLUE results in Table9 are obtained from [https://gluebenchmark.com/leaderboard](https://gluebenchmark.com/leaderboard) and [https://blog.openai.com/language-unsupervised](https://blog.openai.com/language-unsupervised). The GLUE benchmark includes the following datasets, the descriptions of which were originally summarized in [^48]:

#### MNLI

Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task [^51]. Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one.

#### QQP

Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent [^10].

#### QNLI

Question Natural Language Inference is a version of the Stanford Question Answering Dataset [^39] which has been converted to a binary classification task [^48]. The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.

#### SST-2

The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment [^41].

#### CoLA

The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not [^50].

#### STS-B

The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources [^8]. They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning.

#### MRPC

Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent [^17].

#### RTE

Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data [^4].<sup>15</sup>

#### WNLI

Winograd NLI is a small natural language inference dataset [^27]. The GLUE webpage notes that there are issues with the construction of this dataset, <sup>16</sup> and every trained system that’s been submitted to GLUE has performed worse than the 65.1 baseline accuracy of predicting the majority class. We therefore exclude this set to be fair to OpenAI GPT. For our GLUE submission, we always predicted the majority class.

## Appendix C Additional Ablation Studies

### C.1 Effect of Number of Training Steps

Figure 5 presents MNLI Dev accuracy after fine-tuning from a checkpoint that has been pre-trained for $k$ steps. This allows us to answer the following questions:

<svg id="A3.F5.pic1" height="408.27" overflow="visible" version="1.1" viewBox="0 0 570.11 408.27" width="570.11"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,408.27) matrix(1 0 0 -1 0 0) translate(42.01,0) translate(0,41.47) matrix(1.0 0.0 0.0 1.0 -42.01 -41.47)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g transform="matrix(1 0 0 1 0 0) translate(42.01,0) translate(0,41.47)"><g style="--ltx-stroke-color:#BFBFBF;--ltx-fill-color:#BFBFBF;--ltx-fg-color:#BFBFBF;" stroke-width="0.4pt" fill="#BFBFBF" stroke="#BFBFBF" color="#BFBFBF"><path style="fill:none" d="M 101.55 0 L 101.55 357.73 M 203.09 0 L 203.09 357.73 M 304.64 0 L 304.64 357.73 M 406.19 0 L 406.19 357.73 M 507.73 0 L 507.73 357.73"></path></g><g style="--ltx-stroke-color:#BFBFBF;--ltx-fill-color:#BFBFBF;--ltx-fg-color:#BFBFBF;" stroke-width="0.4pt" fill="#BFBFBF" stroke="#BFBFBF" color="#BFBFBF"><path style="fill:none" d="M 0 0 L 507.73 0 M 0 35.77 L 507.73 35.77 M 0 71.55 L 507.73 71.55 M 0 107.32 L 507.73 107.32 M 0 143.09 L 507.73 143.09 M 0 178.87 L 507.73 178.87 M 0 214.64 L 507.73 214.64 M 0 250.41 L 507.73 250.41 M 0 286.19 L 507.73 286.19 M 0 321.96 L 507.73 321.96 M 0 357.73 L 507.73 357.73"></path></g><g style="--ltx-stroke-color:#808080;--ltx-fill-color:#808080;--ltx-fg-color:#808080;" stroke-width="0.2pt" fill="#808080" stroke="#808080" color="#808080"><path style="fill:none" d="M 101.55 0 L 101.55 5.91 M 203.09 0 L 203.09 5.91 M 304.64 0 L 304.64 5.91 M 406.19 0 L 406.19 5.91 M 507.73 0 L 507.73 5.91 M 101.55 357.73 L 101.55 351.83 M 203.09 357.73 L 203.09 351.83 M 304.64 357.73 L 304.64 351.83 M 406.19 357.73 L 406.19 351.83 M 507.73 357.73 L 507.73 351.83"></path></g><g style="--ltx-stroke-color:#808080;--ltx-fill-color:#808080;--ltx-fg-color:#808080;" stroke-width="0.2pt" fill="#808080" stroke="#808080" color="#808080"><path style="fill:none" d="M 0 0 L 5.91 0 M 0 35.77 L 5.91 35.77 M 0 71.55 L 5.91 71.55 M 0 107.32 L 5.91 107.32 M 0 143.09 L 5.91 143.09 M 0 178.87 L 5.91 178.87 M 0 214.64 L 5.91 214.64 M 0 250.41 L 5.91 250.41 M 0 286.19 L 5.91 286.19 M 0 321.96 L 5.91 321.96 M 0 357.73 L 5.91 357.73 M 507.73 0 L 501.83 0 M 507.73 35.77 L 501.83 35.77 M 507.73 71.55 L 501.83 71.55 M 507.73 107.32 L 501.83 107.32 M 507.73 143.09 L 501.83 143.09 M 507.73 178.87 L 501.83 178.87 M 507.73 214.64 L 501.83 214.64 M 507.73 250.41 L 501.83 250.41 M 507.73 286.19 L 501.83 286.19 M 507.73 321.96 L 501.83 321.96 M 507.73 357.73 L 501.83 357.73"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" stroke="#000000" fill="#000000" stroke-width="0.4pt"><path style="fill:none" d="M 0 0 L 0 357.73 L 507.73 357.73 L 507.73 0 L 0 0 Z"></path><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 91.17 -13.81)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="20.76" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="200"><semantics><mn>200</mn> <annotation encoding="application/x-tex">200</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 192.72 -13.81)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="20.76" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="400"><semantics><mn>400</mn> <annotation encoding="application/x-tex">400</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 294.26 -13.81)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="20.76" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="600"><semantics><mn>600</mn> <annotation encoding="application/x-tex">600</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 395.81 -13.81)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="20.76" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="800"><semantics><mn>800</mn> <annotation encoding="application/x-tex">800</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 491.97 -13.81)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:2.28em;--ltx-fo-height:0.64em;--ltx-fo-depth:0.19em;" width="31.52" height="11.61" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="1{,}000"><semantics><mrow><mn>1</mn><mo>,</mo><mn>000</mn></mrow> <annotation encoding="application/x-tex">1{,}000</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 -4.46)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="75"><semantics><mn>75</mn> <annotation encoding="application/x-tex">75</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 31.31)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="76"><semantics><mn>76</mn> <annotation encoding="application/x-tex">76</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 67.09)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="77"><semantics><mn>77</mn> <annotation encoding="application/x-tex">77</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 102.86)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="78"><semantics><mn>78</mn> <annotation encoding="application/x-tex">78</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 138.63)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="79"><semantics><mn>79</mn> <annotation encoding="application/x-tex">79</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 174.41)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="80"><semantics><mn>80</mn> <annotation encoding="application/x-tex">80</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 210.18)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="81"><semantics><mn>81</mn> <annotation encoding="application/x-tex">81</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 245.96)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="82"><semantics><mn>82</mn> <annotation encoding="application/x-tex">82</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 281.73)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="83"><semantics><mn>83</mn> <annotation encoding="application/x-tex">83</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 317.5)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="84"><semantics><mn>84</mn> <annotation encoding="application/x-tex">84</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 -18.73 353.28)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.64em;--ltx-fo-depth:0em;" width="13.84" height="8.92" transform="matrix(1 0 0 -1 0 8.92)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="85"><semantics><mn>85</mn> <annotation encoding="application/x-tex">85</annotation></semantics></math></foreignObject></g> <clipPath id="pgfcp1"><path d="M 0 0 L 507.73 0 L 507.73 357.73 L 0 357.73 Z"></path></clipPath><g clip-path="url(#pgfcp1)"><g style="--ltx-stroke-color:#4285F4;--ltx-fill-color:#4285F4;--ltx-fg-color:#4285F4;" stroke="#4285F4" fill="#4285F4" color="#4285F4"><path style="fill:none" d="M 15.23 128.78 L 25.39 164.56 L 50.77 196.75 L 101.55 257.57 L 203.09 293.34 L 304.64 321.96 L 406.19 332.69 L 507.73 336.27"></path></g><g></g><g style="--ltx-stroke-color:#DB4437;--ltx-fill-color:#DB4437;--ltx-fg-color:#DB4437;" stroke="#DB4437" fill="#DB4437" color="#DB4437"><path style="fill:none" d="M 15.23 157.4 L 25.39 171.71 L 50.77 189.6 L 101.55 214.64 L 203.09 239.68 L 304.64 246.84 L 406.19 253.99 L 507.73 257.57"></path></g><g></g></g><g style="--ltx-stroke-color:#4285F4;--ltx-fill-color:#4285F4;--ltx-fg-color:#4285F4;" stroke="#4285F4" fill="#4285F4" color="#4285F4"><path style="fill:none" d="M 15.23 132.94 L 18.83 126.71 L 11.64 126.71 Z"></path><path style="fill:none" d="M 25.39 168.71 L 28.98 162.48 L 21.79 162.48 Z"></path><path style="fill:none" d="M 50.77 200.9 L 54.37 194.68 L 47.18 194.68 Z"></path><path style="fill:none" d="M 101.55 261.72 L 105.14 255.49 L 97.95 255.49 Z"></path><path style="fill:none" d="M 203.09 297.49 L 206.69 291.27 L 199.5 291.27 Z"></path><path style="fill:none" d="M 304.64 326.11 L 308.23 319.88 L 301.04 319.88 Z"></path><path style="fill:none" d="M 406.19 336.84 L 409.78 330.62 L 402.59 330.62 Z"></path><path style="fill:none" d="M 507.73 340.42 L 511.33 334.19 L 504.14 334.19 Z"></path></g><g style="--ltx-stroke-color:#DB4437;--ltx-fill-color:#DB4437;--ltx-fg-color:#DB4437;" stroke="#DB4437" fill="#DB4437" color="#DB4437"><path style="fill:none" d="M 12.3 154.47 L 18.17 160.34 M 12.3 160.34 L 18.17 154.47"></path><path style="fill:none" d="M 22.45 168.78 L 28.32 174.65 M 22.45 174.65 L 28.32 168.78"></path><path style="fill:none" d="M 47.84 186.66 L 53.71 192.53 M 47.84 192.53 L 53.71 186.66"></path><path style="fill:none" d="M 98.61 211.71 L 104.48 217.58 M 98.61 217.58 L 104.48 211.71"></path><path style="fill:none" d="M 200.16 236.75 L 206.03 242.62 M 200.16 242.62 L 206.03 236.75"></path><path style="fill:none" d="M 301.7 243.9 L 307.58 249.77 M 301.7 249.77 L 307.58 243.9"></path><path style="fill:none" d="M 403.25 251.06 L 409.12 256.93 M 403.25 256.93 L 409.12 251.06"></path><path style="fill:none" d="M 504.8 254.63 L 510.67 260.5 M 504.8 260.5 L 510.67 254.63"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 158.3 -33.4)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:13.81em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="191.14" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible"><span id="A3.F5.pic1.19.19.19.19.19.1.1">Pre-training Steps (Thousands)</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(0.0 1.0 -1.0 0.0 -27.94 115.81)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:9.11em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.19em;" width="126.11" height="12.15" transform="matrix(1 0 0 -1 0 9.46)" overflow="visible"><span id="A3.F5.pic1.20.20.20.20.20.1.1">MNLI Dev Accuracy</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#000000"><path d="M 305.84 0.28 h 201.61 v 41.51 h -201.61 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 309.99 12.04)"><g transform="matrix(1 0 0 -1 0 26.98)"><g transform="matrix(1 0 0 1 0 8.99)"><g style="--ltx-stroke-color:#4285F4;--ltx-fill-color:#4285F4;--ltx-fg-color:#4285F4;" transform="matrix(1 0 0 -1 0 0) translate(0.28,0)" fill="#4285F4" stroke="#4285F4" color="#4285F4"><path style="fill:none" d="M 0 0 L 11.81 0 L 23.62 0"></path><path style="fill:none" d="M 11.81 4.15 L 15.41 -2.08 L 8.22 -2.08 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 -1 24.45 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -4.15)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:11.35em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="157.06" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible"><span id="A3.F5.pic1.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.17.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1">BERT <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{}_{\textsc{BASE}}"><semantics><msub><mtext>BASE</mtext></msub> <annotation encoding="application/x-tex">{}_{\textsc{BASE}}</annotation></semantics></math> (Masked LM)</span></foreignObject></g></g> <g transform="matrix(1 0 0 1 0 26.98)"><g style="--ltx-stroke-color:#DB4437;--ltx-fill-color:#DB4437;--ltx-fg-color:#DB4437;" transform="matrix(1 0 0 -1 0 0) translate(0.28,0)" fill="#DB4437" stroke="#DB4437" color="#DB4437"><path style="fill:none" d="M 0 0 L 11.81 0 L 23.62 0"></path><path style="fill:none" d="M 8.88 -2.94 L 14.75 2.94 M 8.88 2.94 L 14.75 -2.94"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1 0 0 -1 24.45 0) translate(-0.28,0) matrix(1.0 0.0 0.0 1.0 3.04 -4.15)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:11.91em;--ltx-fo-height:0.75em;--ltx-fo-depth:0.25em;" width="164.86" height="13.84" transform="matrix(1 0 0 -1 0 10.38)" overflow="visible"><span id="A3.F5.pic1.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.18.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.1.1.1">BERT <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="{}_{\textsc{BASE}}"><semantics><msub><mtext>BASE</mtext></msub> <annotation encoding="application/x-tex">{}_{\textsc{BASE}}</annotation></semantics></math> (Left-to-Right)</span></foreignObject></g></g></g></g></g></g></g></svg>

Figure 5: Ablation over number of training steps. This shows the MNLI accuracy after fine-tuning, starting from model parameters that have been pre-trained for $k$ steps. The x-axis is the value of $k$.

1. Question: Does BERT really need such a large amount of pre-training (128,000 words/batch \* 1,000,000 steps) to achieve high fine-tuning accuracy?  
	Answer: Yes, BERT ${}_{\textsc{BASE}}$ achieves almost 1.0% additional accuracy on MNLI when trained on 1M steps compared to 500k steps.
2. Question: Does MLM pre-training converge slower than LTR pre-training, since only 15% of words are predicted in each batch rather than every word?  
	Answer: The MLM model does converge slightly slower than the LTR model. However, in terms of absolute accuracy the MLM model begins to outperform the LTR model almost immediately.

### C.2 Ablation for Different Masking Procedures

In Section 3.1, we mention that BERT uses a mixed strategy for masking the target tokens when pre-training with the masked language model (MLM) objective. The following is an ablation study to evaluate the effect of different masking strategies.

Note that the purpose of the masking strategies is to reduce the mismatch between pre-training and fine-tuning, as the \[MASK\] symbol never appears during the fine-tuning stage. We report the Dev results for both MNLI and NER. For NER, we report both fine-tuning and feature-based approaches, as we expect the mismatch will be amplified for the feature-based approach as the model will not have the chance to adjust the representations.

<table><thead><tr><th colspan="3">Masking Rates</th><th colspan="3">Dev Set Results</th></tr></thead><tbody><tr><th>Mask</th><th>Same</th><th>Rnd</th><td>MNLI</td><td colspan="2">NER</td></tr><tr><th></th><th></th><th></th><td>Fine-tune</td><td>Fine-tune</td><td>Feature-based</td></tr><tr><th>80%</th><th>10%</th><th>10%</th><td>84.2</td><td>95.4</td><td>94.9</td></tr><tr><th>100%</th><th>0%</th><th>0%</th><td>84.3</td><td>94.9</td><td>94.0</td></tr><tr><th>80%</th><th>0%</th><th>20%</th><td>84.1</td><td>95.2</td><td>94.6</td></tr><tr><th>80%</th><th>20%</th><th>0%</th><td>84.4</td><td>95.2</td><td>94.7</td></tr><tr><th>0%</th><th>20%</th><th>80%</th><td>83.7</td><td>94.8</td><td>94.6</td></tr><tr><th>0%</th><th>0%</th><th>100%</th><td>83.6</td><td>94.9</td><td>94.6</td></tr></tbody></table>

Table 8: Ablation over different masking strategies.

The results are presented in Table 8. In the table, Mask means that we replace the target token with the \[MASK\] symbol for MLM; Same means that we keep the target token as is; Rnd means that we replace the target token with another random token.

The numbers in the left part of the table represent the probabilities of the specific strategies used during MLM pre-training (BERT uses 80%, 10%, 10%). The right part of the paper represents the Dev set results. For the feature-based approach, we concatenate the last 4 layers of BERT as the features, which was shown to be the best approach in Section 5.3.

From the table it can be seen that fine-tuning is surprisingly robust to different masking strategies. However, as expected, using only the Mask strategy was problematic when applying the feature-based approach to NER. Interestingly, using only the Rnd strategy performs much worse than our strategy as well.

[^1]: Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string embeddings for sequence labeling. In *Proceedings of the 27th International Conference on Computational Linguistics*, pages 1638–1649.

[^2]: Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. 2018. Character-level language modeling with deeper self-attention. *arXiv preprint arXiv:1808.04444*.

[^3]: Rie Kubota Ando and Tong Zhang. 2005. A framework for learning predictive structures from multiple tasks and unlabeled data. *Journal of Machine Learning Research*, 6(Nov):1817–1853.

[^4]: Luisa Bentivogli, Bernardo Magnini, Ido Dagan, Hoa Trang Dang, and Danilo Giampiccolo. 2009. The fifth PASCAL recognizing textual entailment challenge. In *TAC*. NIST.

[^5]: John Blitzer, Ryan McDonald, and Fernando Pereira. 2006. Domain adaptation with structural correspondence learning. In *Proceedings of the 2006 conference on empirical methods in natural language processing*, pages 120–128. Association for Computational Linguistics.

[^6]: Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In *EMNLP*. Association for Computational Linguistics.

[^7]: Peter F Brown, Peter V Desouza, Robert L Mercer, Vincent J Della Pietra, and Jenifer C Lai. 1992. Class-based n-gram models of natural language. *Computational linguistics*, 18(4):467–479.

[^8]: Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. 2017. [Semeval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation](https://doi.org/10.18653/v1/S17-2001). In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, pages 1–14, Vancouver, Canada. Association for Computational Linguistics.

[^9]: Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2013. One billion word benchmark for measuring progress in statistical language modeling. *arXiv preprint arXiv:1312.3005*.

[^10]: Z. Chen, H. Zhang, X. Zhang, and L. Zhao. 2018. [Quora question pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs).

[^11]: Christopher Clark and Matt Gardner. 2018. Simple and effective multi-paragraph reading comprehension. In *ACL*.

[^12]: Kevin Clark, Minh-Thang Luong, Christopher D Manning, and Quoc Le. 2018. Semi-supervised sequence modeling with cross-view training. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1914–1925.

[^13]: Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In *Proceedings of the 25th international conference on Machine learning*, pages 160–167. ACM.

[^14]: Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. 2017. [Supervised learning of universal sentence representations from natural language inference data](https://www.aclweb.org/anthology/D17-1070). In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.

[^15]: Andrew M Dai and Quoc V Le. 2015. Semi-supervised sequence learning. In *Advances in neural information processing systems*, pages 3079–3087.

[^16]: J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In *CVPR09*.

[^17]: William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*.

[^18]: William Fedus, Ian Goodfellow, and Andrew M Dai. 2018. Maskgan: Better text generation via filling in the\_. *arXiv preprint arXiv:1801.07736*.

[^19]: Dan Hendrycks and Kevin Gimpel. 2016. [Bridging nonlinearities and stochastic regularizers with gaussian error linear units](http://arxiv.org/abs/1606.08415). *CoRR*, abs/1606.08415.

[^20]: Felix Hill, Kyunghyun Cho, and Anna Korhonen. 2016. Learning distributed representations of sentences from unlabelled data. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Association for Computational Linguistics.

[^21]: Jeremy Howard and Sebastian Ruder. 2018. [Universal language model fine-tuning for text classification](http://arxiv.org/abs/1801.06146). In *ACL*. Association for Computational Linguistics.

[^22]: Minghao Hu, Yuxing Peng, Zhen Huang, Xipeng Qiu, Furu Wei, and Ming Zhou. 2018. Reinforced mnemonic reader for machine reading comprehension. In *IJCAI*.

[^23]: Yacine Jernite, Samuel R. Bowman, and David Sontag. 2017. [Discourse-based objectives for fast unsupervised sentence representation learning](http://arxiv.org/abs/1705.00557). *CoRR*, abs/1705.00557.

[^24]: Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In *ACL*.

[^25]: Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Skip-thought vectors. In *Advances in neural information processing systems*, pages 3294–3302.

[^26]: Quoc Le and Tomas Mikolov. 2014. Distributed representations of sentences and documents. In *International Conference on Machine Learning*, pages 1188–1196.

[^27]: Hector J Levesque, Ernest Davis, and Leora Morgenstern. 2011. The winograd schema challenge. In *Aaai spring symposium: Logical formalizations of commonsense reasoning*, volume 46, page 47.

[^28]: Lajanugen Logeswaran and Honglak Lee. 2018. [An efficient framework for learning sentence representations](https://openreview.net/forum?id=rJvJXZb0W). In *International Conference on Learning Representations*.

[^29]: Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Contextualized word vectors. In *NIPS*.

[^30]: Oren Melamud, Jacob Goldberger, and Ido Dagan. 2016. context2vec: Learning generic context embedding with bidirectional LSTM. In *CoNLL*.

[^31]: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems 26*, pages 3111–3119. Curran Associates, Inc.

[^32]: Andriy Mnih and Geoffrey E Hinton. 2009. [A scalable hierarchical distributed language model](http://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf). In D. Koller, D. Schuurmans, Y. Bengio, and L. Bottou, editors, *Advances in Neural Information Processing Systems 21*, pages 1081–1088. Curran Associates, Inc.

[^33]: Ankur P Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. 2016. A decomposable attention model for natural language inference. In *EMNLP*.

[^34]: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [Glove: Global vectors for word representation](http://www.aclweb.org/anthology/D14-1162). In *Empirical Methods in Natural Language Processing (EMNLP)*, pages 1532–1543.

[^35]: Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In *ACL*.

[^36]: Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018a. Deep contextualized word representations. In *NAACL*.

[^37]: Matthew Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. 2018b. Dissecting contextual word embeddings: Architecture and representation. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1499–1509.

[^38]: Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI.

[^39]: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2383–2392.

[^40]: Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. 2017. Bidirectional attention flow for machine comprehension. In *ICLR*.

[^41]: Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the 2013 conference on empirical methods in natural language processing*, pages 1631–1642.

[^42]: Fu Sun, Linyang Li, Xipeng Qiu, and Yang Liu. 2018. U-net: Machine reading comprehension with unanswerable questions. *arXiv preprint arXiv:1810.06638*.

[^43]: Wilson L Taylor. 1953. “Cloze procedure”: A new tool for measuring readability. *Journalism Bulletin*, 30(4):415–433.

[^44]: Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Language-independent named entity recognition. In *CoNLL*.

[^45]: Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: A simple and general method for semi-supervised learning. In *Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics*, ACL ’10, pages 384–394.

[^46]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 6000–6010.

[^47]: Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th international conference on Machine learning*, pages 1096–1103. ACM.

[^48]: Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018a. Glue: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, pages 353–355.

[^49]: Wei Wang, Ming Yan, and Chen Wu. 2018b. Multi-granularity hierarchical attention fusion networks for reading comprehension and question answering. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics.

[^50]: Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. 2018. Neural network acceptability judgments. *arXiv preprint arXiv:1805.12471*.

[^51]: Adina Williams, Nikita Nangia, and Samuel R Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In *NAACL*.

[^52]: Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*.

[^53]: Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks? In *Advances in neural information processing systems*, pages 3320–3328.

[^54]: Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V Le. 2018. QANet: Combining local convolution with global self-attention for reading comprehension. In *ICLR*.

[^55]: Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

[^56]: Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In *Proceedings of the IEEE international conference on computer vision*, pages 19–27.