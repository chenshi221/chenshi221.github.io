---
title: "OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning"
source: "https://arxiv.org/html/2501.00321v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/02_%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B/OCRBench%20v2%2C%20An%20Improved%20Benchmark%20for%20Evaluating%20Large%20Multimodal%20Models%20on%20Visual%20Text%20Localization%20and%20Reasoning%2C%20Ling%20Fu%20et%20al.%2C%202024_dual_%E6%99%BA%E8%B0%B14Flash.pdf"
---
Ling Fu <sup>1</sup>  Zhebin Kuang <sup>1</sup>  Jiajun Song <sup>1</sup>  Mingxin Huang <sup>2</sup>  Biao Yang <sup>1</sup>    
Yuzhe Li <sup>1</sup>  Linghao Zhu <sup>1</sup>  Qidi Luo <sup>1</sup>  Xinyu Wang <sup>1</sup>  Hao Lu <sup>1</sup>  Zhang Li <sup>1</sup>  
Guozhi Tang <sup>4</sup>  Bin Shan <sup>4</sup>  Chunhui Lin <sup>4</sup>  Qi Liu <sup>4</sup>  Binghong Wu <sup>4</sup>  
Hao Feng <sup>4</sup>  Hao Liu <sup>4</sup>  Can Huang <sup>4</sup>  Jingqun Tang <sup>4</sup>  Wei Chen <sup>1</sup>  
Lianwen Jin <sup>2</sup>  Yuliang Liu <sup>1</sup>  Xiang Bai <sup>1</sup>  
  
<sup>1</sup> Huazhong University of Science and Technology   <sup>2</sup> South China University of Technology  
<sup>3</sup> University of Adelaide   <sup>4</sup> ByteDance

###### Abstract

Scoring the Optical Character Recognition (OCR) capabilities of Large Multimodal Models (LMMs) has witnessed growing interest. Existing benchmarks have highlighted the impressive performance of LMMs in text recognition; however, their abilities in certain challenging tasks, such as text localization, handwritten content extraction, and logical reasoning, remain underexplored. To bridge this gap, we introduce OCRBench v2, a large-scale bilingual text-centric benchmark with currently the most comprehensive set of tasks ($4\times$ more tasks than the previous multi-scene benchmark OCRBench), the widest coverage of scenarios ($31$ diverse scenarios), and thorough evaluation metrics, with $10,000$ human-verified question-answering pairs and a high proportion of difficult samples. Moreover, we construct a private test set with $1,500$ manually annotated images. The consistent evaluation trends observed across both public and private test sets validate the OCRBench v2’s reliability. After carefully benchmarking state-of-the-art LMMs, we find that most LMMs score below $50$ ($100$ in total) and suffer from five-type limitations, including less frequently encountered text recognition, fine-grained perception, layout perception, complex element parsing, and logical reasoning. The project website is at: [https://99franklin.github.io/ocrbench\_v2](https://99franklin.github.io/ocrbench_v2/).

## 1 Introduction

The emergence of Large Language Models (LLMs) [^1] [^2] [^3] has greatly improved the understanding and generation of structured text. However, in reality, much of the textual content is unstructured; it appears within images, videos, and other non-textual media in varied positions, orientations, and shapes. The need for processing such unstructured content leads to the study of Large Multimodal Models (LMMs) [^4] [^5] [^6] that extend the text-only LLMs to additional modalities. By pretraining on multimodal data, LMMs acquire the zero-shot ability to interpret across diverse media, such as recognizing and understanding complex visual scene text [^7]. Such capability represents a significant advancement over standard Optical Character Recognition (OCR), because LMMs not only spot text but also interpret its semantic relevance to a scene.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x1.png)

Figure 1: Large multimodal models struggle with text-intensive tasks accurately. They are prone to errors in tasks like text localization, handwritten content extraction, and mathematical reasoning, revealing limitations in tackling complex textual information within images.

Compared with classic OCR that typically relies on task-specific models to spot text, the increasing capability of LMMs to process multimodal inputs has opened new potential to redefine the area of OCR. OCR has therefore become an important aspect of recent LMM evaluations. Some text-focused tasks have been included in standard benchmarks to assess the proficiency of LMMs in recognizing and interpreting textual content [^8] [^9]. Typically, text-based Visual Question Answering (VQA) datasets [^10] [^11] [^12] are repurposed to evaluate OCR by framing generic VQA into questions that require accurate reading of embedded text. However, many of these datasets are initially created for classic OCR models, which are of limited diversity, depth, and suitability for evaluating LMMs. A common drawback is that, many questions lack sufficient complexity to assess the reasoning abilities of LMMs on scene text, and some can even be answered without visual input [^13] [^12].

More recently, several customized benchmarks [^14] [^15] [^16] [^17] [^18] have explored the OCR capabilities of LMMs. For example, OCRBench [^14] consolidates $5$ core text-oriented tasks to evaluate LMM performance across traditional OCR functions. Other datasets, such as ComTQA [^19] and ChartX [^20], focus on structured text interpretation like table and chart understanding. While such effort represents a leap over standard OCR benchmarks, they remain limited in both data diversity and quantity (see Tab. 1), often leading to rapid performance saturation. For example, recent LMMs such as Qwen2.5-VL [^21] have achieved 96.4% accuracy on the DocVQA dataset [^22], nearly matching human performance at 98.1%, and 88.8% on OCRBench [^14]. This raises an important question for the community: Do models perform well enough on text-oriented visual understanding tasks in the LMM era, or do existing benchmarks fail to capture the broader challenges in diverse environments?

To answer the question above, we conducted preliminary tests with several state-of-the-art LMMs, including Qwen2.5-VL-7B [^21], InternVL3-14B [^23], and GPT-4o [^24]. These tests assessed performance on text-oriented tasks, such as text localization, handwritten content extraction, and document-based logical reasoning. As illustrated in Fig. 1, each model can fail on one of the text-intensive tasks. These failures reveal a gap in detailed visual perception across different models, which constrains their effectiveness in tasks requiring accurate text localization, recognition, and contextual understanding within images. Recent benchmarks, such as OmniDocBench [^25], CC-OCR [^26], and MMLONGBENCH-DOC [^27], have broadened evaluation to cover more comprehensive scenarios, including fine-grained document parsing and multi-page document understanding. Their analyses reveal the limited capabilities of LMMs for practical OCR applications and highlight the growing need for benchmarks that allow for more robust and varied evaluation of LMMs.

To bridge this gap, we propose OCRBench v2, a comprehensive benchmark designed to assess LMMs across diverse text-oriented visual understanding tasks. As shown in Fig. 3, OCRBench v2 assesses eight core text-reading abilities, including text recognition, text referring, text spotting, relation extraction, element parsing, mathematical calculation, visual text understanding, and knowledge reasoning, organized into a total of $23$ concrete tasks. This benchmark provides $10,000$ high-quality, human-validated instruction-response pairs and also six types of evaluation metrics, which offers a rigorous framework for evaluating LMM performance in complex, practical OCR scenarios. For better evaluation quality, we further collect and label $1,500$ additional text-images from scratch, reserved as the private test set. This private data serves as an independently curated test set to validate model generalization. In summary, the contributions of this work are three-fold:

- OCRBench v2: an improved benchmark designed to assess eight core OCR competencies and covers $23$ tasks across $31$ diverse scenarios, which provides a thorough evaluation framework encapsulating fundamental and advanced text-centric challenges.
- We systematically evaluate state-of-the-art LMMs, ranging from commercial APIs to open-source models, which establishes broad baselines for OCR performance and enables a comparative understanding of model capabilities across varied text-oriented visual understanding tasks.
- We provide a detailed analysis to identify factors affecting the OCR capabilities of LMMs. The analysis examines performance across various dimensions such as model generalization to diverse text types, model robustness, and the ability to tackle complex visual-textual relations.

| Benchmark | #Scenario | #Task | #Image | #Instruction |
| --- | --- | --- | --- | --- |
| OCRbench [^14] | $\sim 14$ | 5 | 0.9k | 1k |
| Seed-bench-2-plus [^15] | $\sim 8$ | 1 | 0.6k | 2.3k |
| CONTEXTUAL [^16] | $\sim 11$ | 1 | 0.5k | 0.5k |
| Fox [^17] | 2 | 9 | 0.7k | 2.2k |
| MMTab-eval [^28] | 1 | 9 | 23k | 49k |
| ComTQA [^19] | 1 | 4 | 1.6k | 9k |
| ChartX [^20] | 1 | 7 | 6k | 6k |
| MMC [^29] | 1 | 9 | 1.7k | 2.9k |
| OmniDocBench [^25] | 9 | 5 | 1k | 1k |
| MMLONGBENCH-DOC [^27] | 7 | 2 | 6.4k | 1.1k |
| OCRBench v2 (Ours) | 31 | 23 | 9.5k | 10k |

Table 1: Comparison between the proposed benchmark and existing text-centric datasets.

## 2 Related Work

#### OCR-Enhanced LMMs.

Inspired by LLMs, visual encoders are integrated into them to create LMMs capable of processing both images and text. Early LMMs exhibit strong zero-shot OCR capabilities, motivating the exploration of text-centric LMMs. For instance, some work [^30] [^31] use text-centric instruction-tuning to enhance OCR-related abilities. But they are restricted to low-res inputs, limiting the ability to recognize dense and small text. To address this, several studies [^32] [^33] [^34] shift attention to increasing the input resolution. As the resolution of inputs increases, so does computational cost. To tackle this issue, TextMonkey [^7] introduces a Token Resampler to compress redundant visual feature tokens, mPLUG-DocOwl2 [^35] presents a DocCompressor module for compressing high-res images, and DocKylin [^36] adopts adaptive pixel slimming and dynamic token slimming modules to reduce redundant regions. To enhance layout perception, DocLayLLM [^37] integrates layout information into LMMs inputs, LayTokenLLM [^38] shares position IDs between text and layout tokens, DocMark [^39] utilizes adaptive generation of markup languages to build structured document representations, while Marten [^40] introduces an additional mask generator during pre-training. Despite strong results on existing benchmarks, challenges remain unsolved in certain key areas such as text localization, entity extraction, and logical reasoning.

#### Benchmarks for Text-Centric LMMs.

Previous efforts have focused on creating scenario-specific benchmarks to assess LMMs. For example, DocVQA [^22], ChartQA [^41], Infographics VQA [^42], and TextVQA [^10] evaluate models on document understanding, chart reasoning, infographic interpretation, and scene text comprehension, respectively. To broaden evaluation scope, OCRBench [^14] introduces a holistic evaluation framework covering five text-oriented tasks, while CONTEXTUAL [^16] and SEED-Bench-2-Plus [^15] introduce context-sensitive and diverse real-world images. Other benchmarks target specific challenges such as dense text understanding [^43], complex structure parsing [^26], and fine-grained document analysis [^25]. To provide a more thorough assessment, some benchmarks design multiple tasks within a specific scenario. TableVQA-Bench [^18], MMTab [^28], and ComTQA [^19] explore table-based tasks, while ChartY [^44], ChartX [^20], and MMC [^29] focus on chart information extraction and reasoning. Recently, DUDE [^45], MM-NIAH [^46], MP-DocVQA [^47], MMLONGBENCH-DOC [^27], and LongDocURL [^48] explore the long document understanding capability of LMMs. In this work, we establish OCRBench v2, a systematic benchmark to reveal the limitations of LMMs in diverse single-image, text-related scenarios.

## 3 Why Do We Need OCRBench v2?

#### Limitations of Existing Benchmarks.

Recent evaluations of LMMs’ OCR capabilities have made significant progress, yet most existing benchmarks exhibit limitations. Datasets like DocVQA, ChartQA, and TextVQA are often narrow in scope, focusing predominantly on text recognition within specific domains such as forms, tables, or documents. While useful for isolated capabilities, they fall short in task diversity, instruction complexity, and structured output formats that better reflect the multimodal nature of LMMs. In particular, many of these benchmarks were originally tailored for traditional OCR systems that prior to the emergence of LMMs. Furthermore, as illustrated in Fig. 2, complex task-specific processes are needed for LMMs when extended to more text-oriented tasks, which limits the evaluation of their broader capabilities.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x2.png)

Figure 2: As evaluation for LMMs expands to diverse text-oriented tasks, existing datasets often require task-specific handling, making unified and scalable evaluation difficult.

#### The Necessity of Unified Multi-task Evaluation.

With the emergence of LMMs, current models show generalization ability to handle multiple tasks. To assess these multi-task models, unified benchmarks like LongDocURL [^48], OmniDocBench [^25], CCOCR [^26], OCRBench [^14], CONTEXTUAL [^16], SEED-Bench-2-Plus [^15], have been proposed and successfully demonstrated the value of evaluating text-oriented models across diverse tasks. These benchmarks show the importance of unified evaluation frameworks in guiding model development. However, as model capabilities expand, existing benchmarks with limited task coverage result in fragmented and sometimes misleading insights. To address this, a unified benchmark is essential to: 1) Understand generalization: Can a model perform consistently across varied text-centric tasks? 2) Diagnose failure models: Does a model that excels in recognition also succeed in reasoning, localization, and parsing? 3) Guide model development: Unified evaluation provides clearer signals for architecture and training improvements.

As shown in Fig. 3, OCRBench v2 tackles this by combining 23 tasks under 8 core capabilities within one framework. This holistic design enables systematic comparison of models and highlights trade-offs (e.g., performance on reasoning vs. recognition) that isolated benchmarks cannot reveal.

#### How OCRBench v2 Addresses the Gaps.

OCRBench v2 is a comprehensive, and high-difficulty benchmark specifically built to evaluate LMMs in realistic OCR settings, with key advantages: 1) Breadth of coverage: With 31 scenarios, we ensure diverse contextual challenges; 2) Task variety: The benchmark spans 8 OCR-related capabilities, many of which are poorly handled by current LMMs; 3) Instruction complexity: Human-authored prompts and structured outputs (e.g., Markdown, JSON, LaTeX) raise the bar beyond simple answer extraction; 4) Private evaluation test set: To prevent overfitting and training contamination, we additionally provide a private test set.

Ultimately, OCRBench v2 fills a critical gap by offering a unified and challenging benchmark that reflects the practical needs of OCR in the LMM era. It not only measures what current models can do, but more importantly, reveals what they still cannot.

#### Design Rationale: Focusing on Single-Image Text Tasks.

While designing OCRBench v2, we focus on challenges in single-image, text-related scenarios, and do not extend our study to multi-image tasks. This design choice is grounded in two considerations: 1) Single-image understanding is the foundation for more complex multimodal tasks. Many existing models still perform unsatisfactorily in various single-image scenarios, which motivates our work; 2) Given long-context inputs, multi-page tasks have more emphasis on long-sequence modeling, requiring specific benchmarks to assess this capability individually. For example, MMLONGBENCH-DOC focuses on evaluating the ability of LMMs to locate and understand content across pages in long documents.

#### Private Dataset for Reliable Evaluation.

To further enhance the assessment quality, we also construct a private test set. This data comprises $1,500$ manually collected text-rich images with human-annotated labels, covering 23 tasks aligned with the distribution of the public data. The images are from diverse sources, including printed books, e-books, scanned documents, and web content. During data collection and annotation, we meticulously curated samples to align with practical text-oriented applications. Given that benchmarks may be contaminated in massive internet-scraped pre-training data of LMMs, this data will not be released. Instead, we maintain a regularly updated leaderboard to reflect the performance of advanced LMMs. Moreover, consistent performance trends and model rankings observed on both the public and private test sets (see Section 5.2) indicate the benchmark’s well-founded design and its effectiveness in identifying model capabilities.

## 4 Benchmark Construction

![Refer to caption](https://arxiv.org/html/2501.00321v2/x3.png)

Figure 3: Sample visualizations for each task. OCRBench v2 comprises 23 sub-tasks grouped under 8 core OCR capabilities. Tasks marked with contain both English and Chinese instructions, while other tasks are either English-only or Chinese-only (Zoomed in for better clarity).

In this section, we describe the task description, annotation curation, statistics, and evaluation criteria. Due to space limitations, more details can be found in the Appendix.

### 4.1 Task Description

To provide a comprehensive evaluation framework for text-reading tasks, we categorize OCR capabilities into eight core areas, each encompassing specific sub-tasks that address various aspects of text comprehension and interpretation. Fig. 3 exhibits samples for each task, with visual inputs and corresponding instructions. Detailed descriptions of these core capabilities are as follows.

Text Recognition. This fundamental capability focuses on perceiving textual content. The related tasks include (fine-grained) text recognition and full-page OCR.

Text Referring. Determining the location of texts accurately is necessary for real-world OCR applications. This ability is evaluated with text grounding and VQA with position tasks.

Text Spotting. Text spotting is a widely studied OCR task that requires models to output both the location and content of text. We consider it a distinct capability due to this unique output format.

Relation Extraction. Given that texts are often densely arranged in images, the ability to extract and map visual components is essential. This capability is assessed through key information extraction, key information mapping, and handwritten content extraction.

Element Parsing. LMMs face the need of parsing complex elements for downstream applications. This ability is evaluated via table parsing, chart parsing, document parsing, and formula recognition.

Mathematical Calculation. Math calculation is essential for LMMs to address numerical reasoning tasks. Hence, text counting is introduced to assess the textual perception ability. Besides, we enhance the math QA data by rendering textual questions into images, accompanied by geometric figures.

Visual Text Understanding. To tackle sophisticated tasks involving human interaction, LMMs need to comprehend the semantic information of texts, a capability we term visual text understanding. This ability is evaluated by document classification and diagram QA. Additionally, we include basic VQA instructions where answers are located directly within the image, which refers to cognition VQA.

Knowledge Reasoning. Some tasks require complex inference and world knowledge, including science QA, APP agent interactions, ASCII art classification, text translation, and reasoning VQA (where answers are not directly visible in images).

### 4.2 Annotation Curation

Dataset Collection. To ensure data diversity, we manually harvest and screen $81$ text-rich academic datasets. To ensure diverse scenario coverage, we also supplement them with additional private data. In all, our dataset comprises $31$ typical scenarios (see Appendix for the full list).

Instruction Formatting. To convert existing annotations into the LMM-compatible instruction format, we design specific prompts for each task. For complex tasks such as document parsing that require structured output, we include a format example to minimize the impact of instruction-following ability and focus the evaluation on OCR capabilities. Additionally, considering the distinct training strategies for localization tasks in LMMs, we standardize the coordinates by normalizing them with image sizes and by scaling to the range of $[0,1000]$. Such a standardization is explicitly specified in the related prompts. For the completeness of the tasks and scenarios, we carefully annotate the additional images to ensure they align with the designed task requirements.

Manual Verification. To ensure data quality, we manually review all instructions for public data and correct approximately $1\%$ annotation errors.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x4.png)

Figure 4: OCR-related statistics and prompt quality assessment of OCRBench v2.

### 4.3 Statistics of OCRBench v2

Here we present the OCR-related statistics and the measurement of prompt quality. As shown in Fig. 4 (a) and (b), we count the distribution of line-level OCR results of $7,400$ English and $2,600$ Chinese images. And Fig. 4 (c) exhibits the average number of line-level OCR results per category. These statistics demonstrate that the text information is sufficiently rich in OCRBench v2. In addition, Fig. 4 (d) compares the Average Entropy, Type-Token Ratio, and Average Variability Index of the questions between OCRBench v2 and OCRBench. OCRBench v2 presents higher values across all three metrics, indicating more diverse, less redundant, and structurally varied questions. This suggests it provides a more comprehensive and challenging benchmark for LMMs.

<table><tbody><tr><th>Method</th><td>Recognition</td><td>Referring</td><td>Spotting</td><td>Extraction</td><td>Parsing</td><td>Calculation</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="10">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>41.3</td><td>18.8</td><td>0</td><td>49.5</td><td>21.2</td><td>17.3</td><td>55.2</td><td>48.9</td><td>31.5</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>46.0</td><td>20.8</td><td>0.1</td><td>58.3</td><td>25.3</td><td>23.3</td><td>64.4</td><td>53.0</td><td>36.4</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>35.2</td><td>0</td><td>0</td><td>16.6</td><td>16.3</td><td>14.4</td><td>59.8</td><td>42.3</td><td>23.1</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>39.1</td><td>0.7</td><td>0</td><td>19.0</td><td>12.2</td><td>19.0</td><td>61.1</td><td>40.2</td><td>23.9</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>52.4</td><td>21.3</td><td>0.1</td><td>45.5</td><td>7.6</td><td>28.5</td><td>65.3</td><td>55.0</td><td>34.5</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>45.3</td><td>21.5</td><td>0</td><td>53.6</td><td>19.2</td><td>19.5</td><td>63.5</td><td>55.5</td><td>34.7</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>48.9</td><td>21.6</td><td>0</td><td>66.3</td><td>35.5</td><td>29.8</td><td>66.9</td><td>53.7</td><td>40.3</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:21">21</a></sup></th><td>68.8</td><td>25.7</td><td>1.2</td><td>80.2</td><td>30.4</td><td>38.2</td><td>73.2</td><td>56.2</td><td>46.7</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>67.3</td><td>36.9</td><td>11.2</td><td>89.0</td><td>38.4</td><td>38.4</td><td>79.2</td><td>60.5</td><td>52.6</td></tr><tr><th>Deepseek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>62.7</td><td>28.0</td><td>0.1</td><td>77.5</td><td>32.7</td><td>14.3</td><td>77.1</td><td>53.9</td><td>43.3</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>66.9</td><td>29.5</td><td>0.5</td><td>70.8</td><td>33.4</td><td>31.9</td><td>69.9</td><td>57.9</td><td>45.1</td></tr><tr><th>GLM-4V-9B <sup><a href="#fn:57">57</a></sup></th><td>61.8</td><td>22.6</td><td>0</td><td>71.7</td><td>31.6</td><td>22.6</td><td>72.1</td><td>58.4</td><td>42.6</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>73.2</td><td>24.6</td><td>0.7</td><td>62.4</td><td>44.8</td><td>40.6</td><td>72.7</td><td>62.6</td><td>47.7</td></tr><tr><th colspan="10">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>61.2</td><td>26.7</td><td>0</td><td>77.5</td><td>36.3</td><td>43.4</td><td>71.1</td><td>55.5</td><td>46.5</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>57.9</td><td>23.3</td><td>0.6</td><td>70.8</td><td>31.5</td><td>38.8</td><td>65.9</td><td>55.1</td><td>43.0</td></tr><tr><th>Gemini-Pro <sup><a href="#fn:60">60</a></sup></th><td>61.2</td><td>39.5</td><td>13.5</td><td>79.3</td><td>39.2</td><td>47.7</td><td>75.5</td><td>59.3</td><td>51.9</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>62.2</td><td>28.4</td><td>1.3</td><td>56.6</td><td>37.8</td><td>40.8</td><td>73.5</td><td>60.9</td><td>45.2</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>67.8</td><td>31.3</td><td>7.2</td><td>73.6</td><td>37.2</td><td>27.8</td><td>69.8</td><td>58.6</td><td>46.7</td></tr></tbody></table>

Table 2: Evaluation of existing LMMs on English tasks of OCRBench v2’s public data. “Recognition”, “Referring”, “Spotting”, “Extraction”, “Parsing”, “Calculation”, “Understanding”, and “Reasoning” refer to text recognition, text referring, text spotting, relation extraction, element parsing, mathematical calculation, visual text understanding, and knowledge reasoning, respectively. Higher values indicate better performance. Best performance is in boldface, and the second best is underlined. The notations apply to all subsequent figures.

### 4.4 Evaluation Criteria

We adopt six types of evaluation metrics tailored to specific task categories. In the following, we present an overview of the evaluation metrics and their applicability to specific tasks.

Parsing Type. To evaluate the element parsing ability of LMMs, we assess their performance in transforming input images into structured formats, including HTML, Markdown, and JSON. TEDS [^63] is employed to measure the structural similarity between outputs and the desired format.

Localization Type. For text referring, the IoU score is applied to quantify the distance between the predicted regions and the ground truth.

Extraction Type. To evaluate relation extraction, we employ the F1 score to assess key information extraction and mapping. Since this evaluation requires structural extraction of information from the output of LMMs, the format is provided in the given prompt.

Long Reading Type. To assess performance on long text reading tasks, BLEU [^64], METEOR [^65], F1 score, and edit distance are used to assess the similarity between predicted text and ground truth.

Counting Type. In text counting, LMMs are required to count the number of text instances. Thus, we use the L1 distance to measure the absolute difference between predicted and ground truth counts. The final score is then normalized to the range of $[0,1]$ based on the ground truth.

Basic VQA Type. For questions where the original data provides options, we use exact string matching to compute accuracy. In other cases, we follow the approach of OCRBench to check whether the ground truth is contained in the prediction for short answers (fewer than $5$ words) and employ ANLS to measure prediction quality for longer answers ($5$ words or more).

## 5 Results and Findings

<table><tbody><tr><th>Method</th><td>LLM Size</td><td>Recognition</td><td>Extraction</td><td>Parsing</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="8">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>8B</td><td>5.7</td><td>2.9</td><td>12.2</td><td>7.5</td><td>17.2</td><td>9.1</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>8B</td><td>14.8</td><td>15.7</td><td>13.7</td><td>16.0</td><td>28.7</td><td>17.8</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>8B</td><td>4.6</td><td>11.2</td><td>8.4</td><td>21.5</td><td>20.0</td><td>13.1</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>8B</td><td>23.5</td><td>14.8</td><td>8.4</td><td>19.9</td><td>12.2</td><td>15.8</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>8B</td><td>7.1</td><td>15.0</td><td>9.2</td><td>9.0</td><td>23.7</td><td>12.8</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>8B</td><td>5.3</td><td>14.9</td><td>12.6</td><td>8.5</td><td>8.1</td><td>9.9</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>12B</td><td>13.4</td><td>10.9</td><td>21.0</td><td>7.0</td><td>20.7</td><td>14.6</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:21">21</a></sup></th><td>8B</td><td>75.3</td><td>61.4</td><td>41.8</td><td>59.3</td><td>40.4</td><td>55.6</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>14B</td><td>66.2</td><td>64.8</td><td>33.5</td><td>63.4</td><td>50.6</td><td>55.7</td></tr><tr><th>Deepseek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>16B</td><td>60.9</td><td>50.6</td><td>28.3</td><td>53.0</td><td>20.5</td><td>42.7</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>7B</td><td>53.0</td><td>49.4</td><td>27.1</td><td>43.5</td><td>32.7</td><td>41.1</td></tr><tr><th>GLM-4V-9B <sup><a href="#fn:57">57</a></sup></th><td>9B</td><td>24.4</td><td>60.6</td><td>20.4</td><td>52.8</td><td>25.2</td><td>36.6</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>7B</td><td>72.2</td><td>50.8</td><td>37.7</td><td>47.9</td><td>37.4</td><td>49.2</td></tr><tr><th colspan="8">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>-</td><td>21.6</td><td>53.0</td><td>29.8</td><td>38.5</td><td>18.2</td><td>32.2</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>-</td><td>13.1</td><td>38.9</td><td>27.2</td><td>28.8</td><td>16.9</td><td>25.0</td></tr><tr><th>Gemini-Pro <sup><a href="#fn:60">60</a></sup></th><td>-</td><td>52.5</td><td>47.3</td><td>30.9</td><td>51.5</td><td>33.4</td><td>43.1</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>-</td><td>21.0</td><td>56.2</td><td>35.2</td><td>55.0</td><td>30.5</td><td>39.6</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>-</td><td>56.7</td><td>41.1</td><td>37.6</td><td>38.3</td><td>39.2</td><td>42.6</td></tr></tbody></table>

Table 3: Evaluation of existing LMMs on Chinese tasks of OCRBench v2’ public data. “LLM Size” indicates the number of parameters of the language model employed in each method.

Here we first benchmark state-of-the-art LMMs on OCRBench v2, presenting the quantitative analysis, then summarize key findings of current limitations for LMMs. All results are presented as percentages.

### 5.1 Baselines

The tested LMMs in the section includes LLaVA-Next-8B [^49], LLaVA-OV-7B [^50], Monkey [^51], TextMonkey [^7], Molmo-7B [^52], Cambrian-1-8B [^53], Pixtral-12B [^54], Qwen2.5-VL-7B [^21], InternVL3-14B [^23], Deepseek-VL2-Tiny [^55], MiniCPM-o-2.6 [^56], GLM-4v-9B [^57], Ovis2-8B [^58], GPT4o [^24], GPT4o-mini [^59], Gemini-1.5-Pro [^60], Claude3.5-sonnet [^61], and Step-1V [^62]. Due to space limitations, more LMM evaluation results can be found in the Sec. A.8.

### 5.2 Main Results

Evaluation results on public data are shown in Tab. 2 and Tab. 3. While LMMs perform well on some basic capabilities such as text recognition and visual text understanding, most LMMs achieve low scores in other capabilities, such as text spotting and element parsing, mostly below $50$. In particular, some LMMs show significant limitations in text spotting capabilities, failing to precisely locate and recognize the texts. Additionally, LMMs demonstrate inadequate abilities in element parsing and mathematical calculation, which are crucial for complicated tasks like document analysis and mathematical reasoning. Besides, after comparing the performance of LMMs on visual text understanding and knowledge reasoning capabilities, we find that they perform poorly in knowledge reasoning. This suggests the deficiency of LMMs in logical reasoning.

Evaluation results on private data are shown in Tab. 4 and Tab. 5. We observe similar evaluation trends to those in the public test set experiments. Overall, LMMs exhibit unsatisfactory performance in text referring, text spotting, element parsing, mathematical calculation, and knowledge reasoning capabilities. In addition, closed-source LMMs outperform their open-source counterparts, demonstrating stronger generalization capabilities. The consistent results across both public and private test sets confirm the soundness of OCRBench v2’s task design, data collection process, and evaluation metrics, and demonstrate its effectiveness in revealing the capability limitations of current LMMs.

<table><tbody><tr><th>Method</th><td>Recognition</td><td>Referring</td><td>Spotting</td><td>Extraction</td><td>Parsing</td><td>Calculation</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="10">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>41.4</td><td>17.0</td><td>0</td><td>49.0</td><td>12.9</td><td>16.1</td><td>60.9</td><td>30.5</td><td>28.5</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>45.4</td><td>18.5</td><td>0</td><td>60.0</td><td>15.5</td><td>32.0</td><td>59.0</td><td>39.3</td><td>33.7</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>31.5</td><td>0.1</td><td>0</td><td>34.4</td><td>26.3</td><td>17.7</td><td>61.4</td><td>22.4</td><td>24.2</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>39.8</td><td>1.6</td><td>0</td><td>27.6</td><td>24.8</td><td>10.2</td><td>62.3</td><td>21.2</td><td>23.4</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>40.8</td><td>19.5</td><td>0</td><td>51.7</td><td>10.0</td><td>33.9</td><td>67.0</td><td>48.0</td><td>33.9</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>44.0</td><td>19.0</td><td>0</td><td>52.3</td><td>19.0</td><td>20.7</td><td>64.0</td><td>39.3</td><td>32.3</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>45.1</td><td>21.8</td><td>0</td><td>71.6</td><td>21.7</td><td>30.4</td><td>77.3</td><td>39.5</td><td>38.4</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:66">66</a></sup></th><td>51.5</td><td>24.5</td><td>3.1</td><td>64.8</td><td>13.1</td><td>53.3</td><td>78.6</td><td>45.5</td><td>41.8</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>55.8</td><td>24.5</td><td>2.1</td><td>89.3</td><td>21.0</td><td>59.5</td><td>72.0</td><td>50.0</td><td>46.8</td></tr><tr><th>Deepseek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>56.6</td><td>23.7</td><td>0</td><td>86.4</td><td>18.9</td><td>30.6</td><td>72.2</td><td>39.5</td><td>41.0</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>54.1</td><td>24.7</td><td>0.3</td><td>74.4</td><td>17.6</td><td>39.2</td><td>75.7</td><td>47.0</td><td>41.6</td></tr><tr><th>GLM-4v-9B <sup><a href="#fn:57">57</a></sup></th><td>52.7</td><td>20.6</td><td>0</td><td>79.4</td><td>15.9</td><td>21.5</td><td>74.7</td><td>32.0</td><td>37.1</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>54.2</td><td>20.9</td><td>0</td><td>83.6</td><td>24.2</td><td>54.7</td><td>74.1</td><td>57.3</td><td>46.1</td></tr><tr><th colspan="10">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>58.6</td><td>23.4</td><td>0</td><td>87.4</td><td>23.1</td><td>51.6</td><td>74.4</td><td>62.3</td><td>47.6</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>55.3</td><td>21.8</td><td>0</td><td>85.4</td><td>20.6</td><td>45.2</td><td>75.5</td><td>49.0</td><td>44.1</td></tr><tr><th>Gemini1.5-Pro <sup><a href="#fn:60">60</a></sup></th><td>59.1</td><td>41.2</td><td>6.6</td><td>89.5</td><td>22.4</td><td>54.7</td><td>78.8</td><td>60.3</td><td>51.6</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>52.9</td><td>24.9</td><td>2.5</td><td>86.9</td><td>23.8</td><td>61.4</td><td>74.4</td><td>53.0</td><td>47.5</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>56.7</td><td>27.4</td><td>2.6</td><td>86.3</td><td>33.3</td><td>42.6</td><td>76.6</td><td>48.7</td><td>46.8</td></tr></tbody></table>

Table 4: Evaluation of existing LMMs on English tasks of OCRBench v2’s private data.

<table><tbody><tr><th>Method</th><td>LLM Size</td><td>Recognition</td><td>Extraction</td><td>Parsing</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="8">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>8B</td><td>2.8</td><td>0.9</td><td>14.9</td><td>20.0</td><td>7.4</td><td>9.2</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>8B</td><td>5.4</td><td>13.6</td><td>20.3</td><td>34.0</td><td>13.6</td><td>17.4</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>8B</td><td>1.5</td><td>28.4</td><td>29.1</td><td>40.0</td><td>8.3</td><td>21.5</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>8B</td><td>10.5</td><td>15.2</td><td>30.2</td><td>44.0</td><td>7.6</td><td>21.5</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>8B</td><td>3.4</td><td>29.8</td><td>6.6</td><td>24.0</td><td>11.1</td><td>15.0</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>8B</td><td>2.4</td><td>19.8</td><td>26.7</td><td>36.0</td><td>7.6</td><td>18.5</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>12B</td><td>6.2</td><td>22.3</td><td>11.4</td><td>26.0</td><td>14.0</td><td>16.0</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:66">66</a></sup></th><td>8B</td><td>24.4</td><td>78.9</td><td>33.1</td><td>82.0</td><td>29.0</td><td>49.5</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>14B</td><td>62.1</td><td>59.5</td><td>33.2</td><td>80.0</td><td>29.2</td><td>52.8</td></tr><tr><th>DeepSeek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>16B</td><td>51.6</td><td>56.3</td><td>27.8</td><td>79.6</td><td>25.3</td><td>48.1</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>7B</td><td>54.0</td><td>62.4</td><td>24.1</td><td>68.0</td><td>29.8</td><td>47.7</td></tr><tr><th>GLM-4v-9B <sup><a href="#fn:57">57</a></sup></th><td>9B</td><td>60.6</td><td>65.2</td><td>32.4</td><td>82.0</td><td>18.2</td><td>51.7</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>7B</td><td>61.0</td><td>67.7</td><td>43.6</td><td>82.0</td><td>25.6</td><td>56.0</td></tr><tr><th colspan="8">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>-</td><td>41.7</td><td>52.1</td><td>29.0</td><td>76.0</td><td>29.4</td><td>45.7</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>-</td><td>20.0</td><td>53.6</td><td>27.9</td><td>66.0</td><td>19.6</td><td>37.4</td></tr><tr><th>Gemini1.5-Pro <sup><a href="#fn:60">60</a></sup></th><td>-</td><td>71.4</td><td>63.8</td><td>30.5</td><td>82.0</td><td>29.9</td><td>55.5</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>-</td><td>34.2</td><td>62.5</td><td>35.2</td><td>78.0</td><td>32.2</td><td>48.4</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>-</td><td>65.2</td><td>64.9</td><td>33.1</td><td>78.0</td><td>25.5</td><td>53.4</td></tr></tbody></table>

Table 5: Evaluation of existing LMMs on Chinese tasks of OCRBench v2’s private data.

### 5.3 Main Findings

We provide in-depth analyses for LMMs’ common limitations, including rare text recognition, fine-grained spatial perception, layout perception, complex element analysis, and logical reasoning.

Finding 1. LMMs still face challenges with less frequently encountered texts, such as dot matrix texts and mathematical formulas. This performance gap highlights the continuing challenges LMMs face in real-world text recognition. For instance, occluded text, CAPTCHA, and dot-matrix text are considered low-frequency text, whereas other types belong to high-frequency text. Notably, recognition accuracy varies significantly across these categories. For example, InternVL3-14B achieves 79.1% accuracy on high-frequency texts but drops to 46.7% on low-frequency ones.

Finding 2. Current LMMs still exhibit limited performance in tasks requiring precise spatial understanding, such as text referring and text spotting. For instance, while InternVL3-14B achieves a response accuracy of 78.3% on the VQA with position task, its IoU score for answer region localization is only 12.9%. This suggests that although LMMs can roughly identify where the answer is located, they struggle to output the exact region.

Finding 3. While LMMs achieve good performance on basic text recognition, they struggle with complex layouts such as overlapping or rotated texts. For example, GPT-4o fails to detect the characters in overlapping handwritten text and misrecognizes numbers in 90° rotated images, revealing LMMs’ limitations in handling texts with complex layouts. Rotating images in the DocVQA dataset led to a significant performance drop of $55.7\%$ for InternVL3-14B (from $90.9\%$ to $35.2\%$).

Finding 4. LMMs still struggle to parse text into structured formats in downstream applications such as document digitalization. For instance, InternVL3-14B achieves 94.4% accuracy in unpaired entities matching, but its performance drops to 84.9% in key information extraction, where the model is required to identify the corresponding value given an entity. The performance further degrades in element parsing tasks that demand structured outputs.

Finding 5. Despite recent advances, LMMs still face challenges in complex mathematical and textual reasoning tasks. To assess their capabilities, we evaluated InternVL3-14B on the private test set covering reasoning VQA, ScienceQA, and APP agent tasks. Questions were categorized into five types: common sense reasoning, visual-text understanding, pattern recognition, calculation, and expert knowledge. Human ratings showed the model achieved accuracies of $72.9\%$, $83.0\%$, $69.2\%$, $56.5\%$, and $71.8\%$, respectively, indicating notable variation.

## 6 Conclusion

In this work, we introduce OCRBench v2, a comprehensive benchmark designed to evaluate the OCR capabilities of LMMs. Covering $23$ tasks across $31$ diverse scenarios, our benchmark systematically assesses eight core capabilities that are essential for text-oriented visual understanding tasks. It includes $10,000$ high-quality QA pairs and six rigorous evaluation metrics. In addition, we curate a private test set of $1,500$ manually labeled images to ensure robust generalization evaluation. Leveraging this benchmark, we conduct extensive experiments on representative LMMs. Through in-depth analysis of experimental results, we identify critical limitations of current models and uncover key factors that affect their OCR performance. We hope OCRBench v2 could aid future research on enhancing LMMs’ text understanding ability.

## References

## Appendix A Technical Appendices and Supplementary Material

This supplementary material contains the following content:

- Sec. A.1: Comparison experiments between LMMs and some text-centric expert models.
- Sec. A.2: Data collection.
- Sec. A.3: Task definitions.
- Sec. A.4: Additional statistics of OCRBench v2.
- Sec. A.5: Evaluation metrics.
- Sec. A.6: Experimental setting for the evaluation process.
- Sec. A.7: Compute resources for the evaluation process.
- Sec. A.8: Evaluation results for LMMs on OCRBench v2.
- Sec. A.9: Potential factors affecting OCR capabilities
- Sec. A.10: Visualization samples for task examples.
- Sec. A.11: Visualization samples for failure cases.
- Sec. A.12: Discussion of broader impacts.
- Sec. A.13: Discussion of limitations.

### A.1 Comparison with LMMs and Text-centric Expert Models

Comparison with text recognizers. We compare LMMs with several representative scene text recognizers, including CRNN [^67], ABINet [^68], ASTER [^69], MASTER [^70], and SVTR [^71], on the text recognition task. The weights of these models are loaded from mmocr <sup>1</sup>. The results are shown in Tab. 6, where we selected $5$ representative LMMs, including Qwen2.5VL-7B [^66], InternVL3-14B [^23], GPT4o [^1], Gemini1.5-Pro [^60], and Step-1V [^62]. The results demonstrate that LMMs exhibit remarkable text recognition capabilities, validating our motivation to evaluate LMMs on more challenging OCR-related tasks.

| Method | Accuracy |
| --- | --- |
| CRNN [^67] | 38.1 |
| ABINet [^68] | 62.4 |
| ASTER [^69] | 50.0 |
| MASTER [^70] | 54.1 |
| SVTR [^71] | 57.8 |
| Qwen2.5VL-7B [^66] | 73.0 |
| InternVL3-14B [^23] | 71.1 |
| GPT4o [^1] | 74.1 |
| Gemini1.5-Pro [^60] | 64.1 |
| Step-1V [^62] | 75.4 |

Table 6: Comparison between LMMs and text recognizers.

Comparison with text spotters. We also compare LMMs with ABCNet series [^72] [^73] and TESTR [^74] on the text spotting task. The ABCNet series utilize the official weights <sup>2</sup>, and TESTR is also initialized with its publicly released checkpoint <sup>3</sup>. These models were fine-tuned with TotalText [^75]. The results are shown in Tab. 7. Although LMMs demonstrate promising capabilities in text recognition, there remains notable potential for improvement in the text spotting task.

| Method | F1 score |
| --- | --- |
| ABCNet [^72] | 32.2 |
| ABCNetV2 [^73] | 44.2 |
| TESTR [^74] | 51.8 |
| Qwen2.5VL-7B [^66] | 1.2 |
| InternVL3-14B [^23] | 11.2 |
| Gemini1.5-Pro [^60] | 13.5 |
| GPT4o [^1] | 0 |
| Step-1V [^62] | 7.2 |

Table 7: Comparison between LMMs and text spotters.

Comparison with GOT. We notice a recent work, GOT [^76], that can parse the textual elements within images. We conduct comparison experiments between GOT and some representative LMMs, and the results are shown in Tab. 8. We observe that LMMs show advantages in general text recognition, while GOT demonstrates better performance in the document parsing task.

| Method | Rec | FG-Rec | Full-Rec | Doc-Parse |
| --- | --- | --- | --- | --- |
| GOT [^76] | 64.1 | 52.9 | 73.3 | 53.9 |
| Qwen2.5VL-7B [^66] | 73.0 | 36.4 | 84.2 | 39.1 |
| InternVL3-14B [^23] | 71.1 | 36.4 | 83.0 | 36.9 |
| GPT4o [^1] | 74.1 | 13.8 | 54.1 | 35.9 |
| Gemini1.5-Pro [^60] | 64.1 | 22.9 | 83.9 | 40.5 |
| Step-1V [^62] | 76.8 | 24.8 | 74.8 | 36.0 |

Table 8: Comparison between LMMs and GOT [^76].

### A.2 Data Collection

Text Recognition. The data for text recognition task are sampled from ICDAR2013 [^77], SVT [^78], IIIT5K [^79], ICDAR2015 [^80], SCUT-CTW1500 [^81], COCO-Text [^82], CUTE80 [^83], TotalText, SVTP [^84], WordArt [^85], NonSemanticText [^14], IAM [^86], ORAND-CAR-2014 [^87], HOST [^88], and WOST [^88]. Meanwhile, CAPTCHA (Completely Automated Public Turing Test to Tell Humans Apart) images are sourced from a CAPTCHA dataset <sup>4</sup> and a number CAPTCHA dataset <sup>5</sup>. Additionally, dot matrix images in the text recognition task are manually collected from the web page.

Fine-grained Text Recognition. In the fine-grained text recognition task, images are sampled from the test sets of Fox [^17], Totaltext, COCO-Text, CTW1500 [^89], and ICDAR2015. We use the original annotations for Fox, while the other datasets are manually re-annotated.

Full-page OCR. The data sources for full-page OCR task include Fox, HierText [^90], CTW [^91], RCTW-17 [^92], ReCTS [^93], LSVT2019 [^94], M6Doc [^95], and CDLA <sup>6</sup>.

Text Grounding. The images for the text grounding task are sampled from testset of Totaltext, COCO-Text, CTW1500, and ICDAR2015. QA pairs and bounding boxes annotations are based on their official OCR annotations.

VQA with Position. The images used for VQA with position task are sampled from the test sets of TextVQA [^10] and RICO [^96], with QA pairs and bounding box annotations derived from their original datasets.

Text Spotting. The data sources for the text spotting task include Totaltext, COCO-Text, CTW1500, and ICDAR2015.

Key Information Extraction. The data sources for key information extraction task include FUNSD [^97], SROIE [^98], POIE [^99], M6Doc, XFUND [^100], ICDAR2023-SVRD [^101], and a private dataset of photographed receipts.

Key Information Mapping. The data sources for the key information mapping task include FUNSD and POIE.

Handwritten Content Extraction. This task’s data is our private data, which contains real exam paper data with student information removed and manually annotated QA pairs.

Table Parsing. The images for table parsing task are selected from MMTab [^28], WTW [^102], TabRecSet [^103] and flush table recognition competition <sup>7</sup>.

Chart Parsing. The data sources for the chart parsing task come from OneChart [^44] and MMC [^29].

Document Parsing. The data sources for document parsing task come from DoTA [^104], DocVQA [^105], M6Doc, and CDLA.

Formula Recognition. The data sources for the formula Recognition task includes HME100K [^106], IM2LATEX-100K [^107], M2E [^108], MathWriting [^109], MLHME-38K <sup>8</sup>, CASIA-CSDB [^110], and some private data.

Math QA. The data sources for the math QA task includes MathMatics [^111], MathVerse [^112], MathVision [^107], and MathVista [^113].

Text Counting. The data for the text counting task are collected from IIIT5K, SVT, ICDAR2013, HierText, and TotalText.

Cognition VQA. The data sources for the cognition VQA task include EST-VQA [^12], OCRVQA [^114], ST-VQA [^11], TEXTVQA, DIR300 [^115], ChartQA [^41], DVQA [^116], PlotQA [^117], InfoVQA [^118], WTW, PubTabNet [^119], WTQ [^120], CORD [^121], LLaVAR [^30], WebSRC [^122], DocVQA, M6Doc, XFUND, Publaynet [^123], RVL-CDIP [^124], ScreenQA [^125], SlideVQA [^126], a movie poster collection dataset <sup>9</sup>, a website screenshot collection dataset <sup>10</sup>, and a private receipt photograph dataset.

Diagram QA. The data sources for the diagram QA task include AI2D [^127] and TextBookQA [^128].

Document Classification. The images for the document classification task are collected from RVL-CDIP.

Reasoning VQA. The reasoning VQA task shares some common data sources with the cognition VQA task. Additionally, portions of the reasoning VQA dataset are drawn from MMSI [^129] and CMMMU [^130].

Science QA. The images and annotations of the science QA task are collected from ScienceQA [^131] and MMMU-Pro [^132]

APP Agent. The data source of the APP agent task is RICO.

ASCII Art Classification. The data sources for the ASCII art classification task is ASCIIEval [^133].

Text Translation. The datasets collected for text translation task includes memes <sup>11</sup>, MSRA-TD500 [^134], MTWI2018 [^135], M6Doc, ICDAR2023-SVRD, EST-VQA, RCTW17 [^136], DAST1500 [^137], XFUND, ArT2019 [^138], ChartQA, CDLA, ICDAR2015, SlideVQA, Fintabnet [^139], ScienceQA, InfoVQA, COMICS-Dialogue <sup>12</sup>, and ExpressExpense SRD <sup>13</sup>.

### A.3 Task Definitions

In this section, we introduce the definition of each task, and the visualizations for each task can be found in Sec. A.10.

Text Recognition. Text recognition refers to the fundamental OCR ability on text image patches, which asks LMMs to read the text content. To comprehensively evaluate LMMs’ text recognition ability across diverse scenarios, our collection incorporates various text types, including regular text, irregular text, artistic text, handwriting text, digit string text, non-semantic text, occluded text, doc matrix text, and CAPTCHA text.

Fine-grained Text Recognition. This task requires LLMs to read and comprehend textual content within the given region. It evaluates LLMs’ fine-grained perception capabilities in understanding text in natural scenes and documents.

Full-page OCR. Full-page OCR [^17] task requires LMMs to extract and recognize all text content from the given images. Converting text into digital format facilitates subsequent processing and analysis of text images.

Text grounding. In this task, users would provide a text string and require LMMs to locate its specific location, evaluating LMMs’ fine-grained perception capabilities.

VQA with Position. For VQA with position task, LMMs need to not only respond to the question but also provide the exact position coordinates that directly correspond to the answer. We ask LMMs to output both information in JSON format for convenient evaluation, and the coordinates are required to be normalized with image sizes and scaled to the range of $[0,1000]$.

Text Spotting. Text spotting task needs LMMs to output the localization and content of all appeared text simultaneously. Due to the interference of background elements and the large number of text instances, this task demands high fine-grained perception capabilities from the model. Besides, the coordinates are required to be normalized with image sizes and scaled to the range of $[0,1000]$.

| Scene | Number | Scene | Number | Scene | Number |
| --- | --- | --- | --- | --- | --- |
| Schematic diagram | 1238 | Scientific paper | 799 | Word | 728 |
| Table(filled) | 705 | Chart | 620 | Receipts | 609 |
| Questions | 581 | Mathematical formula | 475 | Product labels | 434 |
| Phone screenshot | 431 | Indoor scenes | 395 | Industry research reports | 343 |
| Poster | 264 | Street scene | 224 | ASCII Art | 199 |
| Shop sign | 189 | Financial reports | 153 | Chemical formula | 149 |
| Textbook | 148 | Magazine | 146 | Email | 111 |
| Web screenshot | 99 | Details page | 95 | Verification code | 87 |
| Resumes | 67 | Illustration | 61 | Newspaper | 52 |
| Road signs | 43 | Menus | 31 | Notify | 30 |
| Questionnaire | 29 |  |  |  |  |

Table 9: The number of images included in each scene category in public data.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x5.png)

Figure 5: Overview of the eight testable text-reading capabilities and associated tasks in OCRBench v2. Each color represents a distinct capability type.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x6.png)

Figure 6: The quantity distribution of English tasks of public data.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x7.png)

Figure 7: The quantity distribution of Chinese tasks of public data.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x8.png)

Figure 8: The OCR lines distribution of English tasks of public data.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x9.png)

Figure 9: The OCR lines distribution of Chinese tasks of public data.

Key Information Extraction. The key information extraction task is to extract the necessary information from densely arranged text. In this task, we provide some desired entities as keys and demand LMMs to output the corresponding values to form the output JSON string.

Key Information Mapping. In this task, we provide a set of entity keys and their corresponding values in the prompt. The LMMs are then asked to match and pair these keys with their respective values into groups.

Handwritten Content Extraction. To investigate the information extraction capabilities of LMMs in educational scenarios, we collect some Chinese examination papers, containing both printed question text and handwritten student responses. There are four types of questions in these examination papers, including single-choice, multiple-choice, true or false, and brief response questions. The prompts require LMMs to extract the handwritten content for specific questions.

Table Parsing. Table parsing task requires LMMs to parse the given table into structured text, including Markdown and HTML format.

Chart Parsing. Apart from tables, charts can also be converted to structured information. In this task, LLMs are required to transform visual charts into JSON format.

Document Parsing. In the document parsing task, both text and the complex elements, including charts, tables, and formulas, are required to be parsed.

Formula Recognition. This task asks LMMs to recognize the given formula in the LaTeX format. The collection includes mathematical and chemical formulas.

Math QA. Math QA task evaluates the LMMs’ mathematical calculation ability. In particular, we render the mathematical problem description and related figures into images and ask LMMs to answer the questions within the images.

Text Counting. Text counting task is built to evaluate the quantity property perceiving ability of LMMs, including the character frequency in words and the word counting in the given image.

Cognition VQA. In OCRBench v2, we split text-centric VQA instructions into cognition VQA and Reasoning VQA based on whether the answers can be directly found in the images. Cognition VQA task refers to the instructions where answers are explicitly present in the given image. This task evaluates the fundamental text-centric question-answering ability based on visual content.

Diagram QA. In the diagram QA task, LMMs need to respond to the question about the given diagrams, reflecting LMMs’ ability to understand the relationship between the visual elements.

Document Classification. Document classification task asks LMMs to classify the category of the given document image. The included categories are letters, forms, emails, handwritten documents, advertisements, scientific reports, scientific publications, specifications, file folders, news articles, budgets, invoices, presentations, questionnaires, resumes, and memos.

Reasoning VQA. In reasoning VQA tasks, the answers often do not directly appear in the image. This forces LMMs to perform logical reasoning to respond to questions based on visual information.

Science QA. In the Science QA task, LMMs are required to respond to the scientific problem. We use PaddleOCR <sup>14</sup> to extract text from the collected images and filter out those with fewer than four OCR results. Additionally, when extra subject-related knowledge is provided by the source, we incorporate it by rendering it into the images.

APP Agent. For the APP agent task, LMMs need to understand the relationship between textual content, icons, and world knowledge to respond to the question from the user, simulating the real-world application scene.

ASCII Art Classification. We incorporate a recent image classification task that uses images composed purely of ASCII characters [^133]. This task is included in OCRBench v2 to evaluate LMMs’ ability to assess LMMs’ pattern recognition and visual abstraction abilities.

Text Translation. In the text translation task, LMMs need to execute translation between Chinese and English texts, evaluating LMMs’ semantic understanding abilities.

### A.4 Additional Statistics of OCRBench v2

Scene Coverage. Our dataset can be divided into $31$ classic scenes according to the scene of the image. The specific scenes and the corresponding number of pictures are shown in Tab. 9.

Statistics of each task. Fig. 5 shows an overview of each task in OCRBench v2.The distribution of $23$ tasks in OCRBench v2 is displayed in Fig. 6 and Fig. 7. Additionally, we calculate and present the average number of OCR text lines per task in Fig. 8 and Fig. 9. As illustrated in these figures, the task distribution is well-balanced, with each task containing adequate textual information for analysis.

### A.5 Evaluation Metrics

Parsing Type. We use Tree-Edit-Distance-based Similarity (TEDS) [^63] to evaluate parsing tasks, which require LMMs to transform the images to structured formats. Tree Edit Distance (TED) refers to the minimum number of edits to transform one tree into another. TEDS is based on TED to calculate the similarity of two trees. Assuming $T_{1}$ and $T_{2}$ are two different trees, $TED(T_{1},T_{2})$ refers to their TED, and the TEDS is defined as:

$$
TEDS(T_{1},T_{2})=1-\frac{TED(T_{1},T_{2})}{\max(|T_{1}|,|T_{2}|)},
$$

where $|T_{1}|$ and $|T_{2}|$ is the number of nodes of trees, $TED(T_{1},T_{2})$ can be calculated by dynamic programming algorithm. If $T_{1}$ and $T_{2}$ are identical, then their TEDS equals $1$. As the structural difference between two trees increases, their TED value becomes larger, resulting in the TEDS approaching $0$.

Localization Type. In the text referring and spotting tasks, LMMs are required to provide regression bounding boxes of target objects. IoU score is adopted to measure the distance between the predicted regions and the ground truth.

$$
IoU(B_{1},B_{2})=\frac{Intersect(B_{1},B_{2})}{Union(B_{1},B_{2})},
$$

where $Intersect(B_{1},B_{2})$ refers to the overlap area of bounding box $B_{1}$ and $B_{2}$, while $Union(B_{1},B_{2})$ refers to their union area.

Extraction Type. The F1 score is used to evaluate LMMs’ relation extraction capability. Given the predicted and ground truth Key-Value pairs, the F1 score is formulated as follows:

$$
\displaystyle Precision=\frac{N_{3}}{N_{2}},
$$
$$
\displaystyle Recall=\frac{N_{3}}{N_{1}},
$$
$$
\displaystyle Fmean=\frac{2*Precision*Recall}{Precision+Recall},
$$

where $N_{1}$, $N_{2}$, and $N_{3}$ denote the number of ground-truth Key-Value pairs, predicted Key-Value pairs, and correctly matched Key-Value pairs, respectively.

Long Reading Type. To evaluate LMMs’ ability to recognize text across entire paragraphs or pages, BLEU [^64], METEOR [^65], F1 score, and normalized edit distance are employed. And the final score is the average value of these metrics.

BLEU evaluates prediction quality by comparing n-gram match rates between the prediction and ground truth sequences. For each n-gram type, precision is calculated as the ratio of matching n-grams to total predicted n-grams. The final BLEU score is the geometric mean of these precision values multiplied by a penalty $BP$, which is defined as:

$$
\displaystyle BLEU=BP*exp(\sum^{N}_{n=1}{w_{n}\log{p_{n}}}),
$$
$$
\displaystyle BP=\begin{cases}1&L_{p}\geq L_{g}\\
e^{(1-\frac{L_{p}}{L_{g}})}&L_{p}<L_{g}\end{cases},
$$

where $p_{n}$ represents the precision of n-grams, $L_{p}$ represents the length of prediction sequence, $L_{g}$ represents the length of ground truth sequence, $w_{n}$ is weight factor, usually evenly distributed ($w_{n}=\frac{1}{N}$). Typically, $N$ is set to 4.

METEOR employs a semantic-aware matching strategy with four levels. 1) Exact Match: words in the prediction that are identical to the ground truth. 2) Stem match: matching words that have the same word stem. 3) Synonym Match: matching words based on synonymous relationships. 4) Paraphrase Match: Matching similar phrases at the phrase level. These matches are combined to calculate precision and recall, from which a weighted harmonic mean F1 score is derived as:

$$
\displaystyle P_{meteor}=\frac{N_{match}}{N_{pred}},
$$
$$
\displaystyle R_{meteor}=\frac{N_{match}}{N_{gt}},
$$
$$
\displaystyle F_{meteor}=\frac{10*P_{meteor}*R_{meteor}}{P_{meteor}+9*R_{meteor}},
$$

where $N_{match}$, $N_{pred}$, and $N_{gt}$ represent the number of matched items, words in prediction, and words in ground truth, respectively. The final METEOR score is obtained by multiplying the $F_{meteor}$ by the penalty adjustment factor. The calculation is formulated as follows:

$$
\displaystyle METEOR=F_{meteor}*(1-BP_{meteor}),
$$
$$
\displaystyle BP_{meteor}=0.5*\frac{N_{chunk}}{N_{match}},
$$

where $N_{chunk}$ refers to the number of contiguous matching phrases. More chunks indicate greater word order differences, resulting in a heavier penalty.

The calculation method of the F1 score in long reading metrics follows the same approach as discussed in extraction metrics, as shown in Equations 3, 4, 5.

Normalized Edit Distance (NED) measures string similarity by computing the minimum number of operations needed to transform one string into another. And then NED is normalized by the length of the longer string. The calculation is formulated as follows:

$$
\displaystyle NED(S_{1},S_{2})=\frac{ED(S_{1},S_{2})}{\text{max}(len(S_{1}),len(S_{2}))}
$$

where $ED(S_{1},S_{2})$ represents the edit distance between the prediction string $S_{1}$ and the ground truth $S_{2}$. The $NED$ value of 0 indicates identical strings, while 1 indicates completely different strings.

Counting Type. In OCRBench v2, character frequency counting and word counting tasks are included. For character frequency, we use exact match evaluation since the answers are typically single-digit integers. For word counting, we evaluate using the L1 distance between predicted and ground truth counts, normalized to $[0,1]$ based on the ground truth. This can be formulated as follows:

$$
\displaystyle score=\begin{cases}0&C_{pred}\leq 0\\
1-\frac{|C_{pred}-C_{gt}|}{C_{gt}})&0<C_{pred}<2*C_{gt}\\
0&C_{pred}\geq 2*C_{gt}\\
\end{cases},
$$

where $C_{pred}$ and $C_{gt}$ denote the predicted count and ground truth count, respectively.

Basic VQA Type. The remaining tasks in OCRBench v2 are basic VQA types, and we employ different evaluation metrics based on question types. For multiple-choice questions, we use exact matching between predictions and answer options. In other cases, we check whether the ground truth is contained in the prediction for answers shorter than $5$ words, and use ANLS for longer answers.

### A.6 Experimental setting

The detailed public data construction are shown in Sec. A.2 and Sec. A.5. Private data consists of unlabeled images collected manually from websites and real life. At the same time, we annotated and checked the private test set to ensure the quality. The environment configuration of each open-source model experiment strictly complies with the official version and uses the official pre-trained model and inference code. The model parameters of the open-source model and the API parameters of the closed-source model use the official default parameters for fair. Specifically, we use the official API versions: GPT-4o (gpt-4o-2024-08-06), GPT-4o-mini (gpt-4o-mini-2024-07-18), and Gemini 1.5 Pro (gemini1.5-pro-002).

### A.7 Compute resources

Evaluations of open-source models were conducted on 8×NVIDIA GeForce RTX 4090 (24GB) and a NVIDIA H800 Tensor Core GPU (80GB). The closed-source experiments obtained the results by calling the official API.

### A.8 Results and Discussions

Tab. 10, Tab. 11, Tab. 12, and Tab. 13 exhibit the results of $39$ open-source models and $5$ closed-source models on the public and private test sets of OCRBench v2

Evaluation results on public data are shown in Tab. 10 and Tab. 11. Most LMMs performed well in tasks such as Understanding, Recognition, Extraction, which shows that current models have basic OCR capabilities. However, they performed poorly in tasks such as Referring, Spotting, Parsing, and Calculation. The scores of all models are basically below 50 points, which shows that the models still lack the ability in text localization, logical reasoning, and understanding complex elements.

Evaluation results on private data are shown in Tab. 12 and Tab. 13. The performance trends of the models on private and public datasets are consistent. In addition, most models perform worse on private datasets than on public datasets, which shows that private data may be more challenging for LMMs due to the lack of training, and also reflects the importance of private data construction.

<table><tbody><tr><th>Method</th><td>Recognition</td><td>Referring</td><td>Spotting</td><td>Extraction</td><td>Parsing</td><td>Calculation</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="10">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>41.3</td><td>18.8</td><td>0</td><td>49.5</td><td>21.2</td><td>17.3</td><td>55.2</td><td>48.9</td><td>31.5</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>46.0</td><td>20.8</td><td>0.1</td><td>58.3</td><td>25.3</td><td>23.3</td><td>64.4</td><td>53.0</td><td>36.4</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>35.2</td><td>0</td><td>0</td><td>16.6</td><td>16.3</td><td>14.4</td><td>59.8</td><td>42.3</td><td>23.1</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>39.1</td><td>0.7</td><td>0</td><td>19.0</td><td>12.2</td><td>19.0</td><td>61.1</td><td>40.2</td><td>23.9</td></tr><tr><th>XComposer2-4KHD <sup><a href="#fn:140">140</a></sup></th><td>45.1</td><td>21.8</td><td>0.1</td><td>15.9</td><td>11.7</td><td>15.7</td><td>66.8</td><td>45.9</td><td>27.9</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>52.4</td><td>21.3</td><td>0.1</td><td>45.5</td><td>7.6</td><td>28.5</td><td>65.3</td><td>55.0</td><td>34.5</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>45.3</td><td>21.5</td><td>0</td><td>53.6</td><td>19.2</td><td>19.5</td><td>63.5</td><td>55.5</td><td>34.7</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>48.9</td><td>21.6</td><td>0</td><td>66.3</td><td>35.5</td><td>29.8</td><td>66.9</td><td>53.7</td><td>40.3</td></tr><tr><th>EMU2-chat <sup><a href="#fn:141">141</a></sup></th><td>42.1</td><td>0.2</td><td>0</td><td>12.5</td><td>8.1</td><td>11.2</td><td>42.7</td><td>33.4</td><td>18.8</td></tr><tr><th>mPLUG-Owl3 <sup><a href="#fn:142">142</a></sup></th><td>41.6</td><td>14.0</td><td>0.6</td><td>24.4</td><td>10.9</td><td>11.1</td><td>52.2</td><td>46.0</td><td>25.1</td></tr><tr><th>CogVLM-chat <sup><a href="#fn:143">143</a></sup></th><td>50.9</td><td>0</td><td>0</td><td>0.2</td><td>8.4</td><td>15.0</td><td>58.1</td><td>41.7</td><td>21.8</td></tr><tr><th>Qwen-VL <sup><a href="#fn:4">4</a></sup></th><td>34.6</td><td>7.5</td><td>0</td><td>18.2</td><td>20.0</td><td>8.1</td><td>57.2</td><td>41.1</td><td>23.3</td></tr><tr><th>Qwen-VL-chat <sup><a href="#fn:4">4</a></sup></th><td>34.5</td><td>4.1</td><td>0</td><td>25.9</td><td>14.0</td><td>13.8</td><td>55.7</td><td>39.5</td><td>23.4</td></tr><tr><th>Qwen2-Vl-7B <sup><a href="#fn:66">66</a></sup></th><td>72.1</td><td>47.9</td><td>17.5</td><td>82.5</td><td>25.5</td><td>25.4</td><td>78.4</td><td>61.5</td><td>51.4</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:21">21</a></sup></th><td>68.8</td><td>25.7</td><td>1.2</td><td>80.2</td><td>30.4</td><td>38.2</td><td>73.2</td><td>56.2</td><td>46.7</td></tr><tr><th>InternVL2-8B <sup><a href="#fn:144">144</a></sup></th><td>49.9</td><td>23.1</td><td>0.5</td><td>65.2</td><td>24.8</td><td>26.7</td><td>73.5</td><td>52.9</td><td>39.6</td></tr><tr><th>InternVL2-26B <sup><a href="#fn:144">144</a></sup></th><td>63.4</td><td>26.1</td><td>0</td><td>76.8</td><td>37.8</td><td>32.3</td><td>79.4</td><td>58.9</td><td>46.8</td></tr><tr><th>InternVL2.5-8B <sup><a href="#fn:23">23</a></sup></th><td>59.0</td><td>25.0</td><td>1.4</td><td>77.5</td><td>35.1</td><td>29.4</td><td>75.3</td><td>57.2</td><td>45.0</td></tr><tr><th>InternVL2.5-26B <sup><a href="#fn:23">23</a></sup></th><td>65.6</td><td>26.1</td><td>1.6</td><td>86.9</td><td>36.2</td><td>37.4</td><td>78.3</td><td>62.9</td><td>49.4</td></tr><tr><th>InternVL3-8B <sup><a href="#fn:23">23</a></sup></th><td>68.6</td><td>30.4</td><td>8.8</td><td>85.3</td><td>34.0</td><td>27.1</td><td>77.5</td><td>60.3</td><td>49.0</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>67.3</td><td>36.9</td><td>11.2</td><td>89.0</td><td>38.4</td><td>38.4</td><td>79.2</td><td>60.5</td><td>52.6</td></tr><tr><th>Deepseek-VL-7B <sup><a href="#fn:145">145</a></sup></th><td>37.1</td><td>15.4</td><td>0</td><td>23.5</td><td>14.6</td><td>20.8</td><td>53.3</td><td>52.9</td><td>27.2</td></tr><tr><th>Deepseek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>62.7</td><td>28.0</td><td>0.1</td><td>77.5</td><td>32.7</td><td>14.3</td><td>77.1</td><td>53.9</td><td>43.3</td></tr><tr><th>MiniCPM-V-2.6 <sup><a href="#fn:56">56</a></sup></th><td>66.8</td><td>6.0</td><td>0.8</td><td>62.0</td><td>28.8</td><td>32.4</td><td>73.7</td><td>52.1</td><td>40.3</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>66.9</td><td>29.5</td><td>0.5</td><td>70.8</td><td>33.4</td><td>31.9</td><td>69.9</td><td>57.9</td><td>45.1</td></tr><tr><th>GLM-4V-9B <sup><a href="#fn:57">57</a></sup></th><td>61.8</td><td>22.6</td><td>0</td><td>71.7</td><td>31.6</td><td>22.6</td><td>72.1</td><td>58.4</td><td>42.6</td></tr><tr><th>VILA1.5-8B <sup><a href="#fn:146">146</a></sup></th><td>35.3</td><td>15.5</td><td>0</td><td>21.1</td><td>12.7</td><td>17.3</td><td>46.3</td><td>40.3</td><td>23.6</td></tr><tr><th>LLaVAR <sup><a href="#fn:30">30</a></sup></th><td>37.3</td><td>0</td><td>0</td><td>1.0</td><td>9.9</td><td>12.3</td><td>34.6</td><td>27.0</td><td>15.3</td></tr><tr><th>UReader <sup><a href="#fn:33">33</a></sup></th><td>22.4</td><td>0.1</td><td>0</td><td>0</td><td>9.2</td><td>7.9</td><td>41.0</td><td>29.1</td><td>13.7</td></tr><tr><th>DocOwl2 <sup><a href="#fn:147">147</a></sup></th><td>24.0</td><td>9.7</td><td>0</td><td>13.4</td><td>13.5</td><td>8.8</td><td>53.7</td><td>32.0</td><td>19.4</td></tr><tr><th>Yi-VL-6B <sup><a href="#fn:148">148</a></sup></th><td>28.9</td><td>2.9</td><td>0</td><td>9.7</td><td>12.9</td><td>15.8</td><td>36.1</td><td>32.0</td><td>17.3</td></tr><tr><th>Janus-1.3B <sup><a href="#fn:149">149</a></sup></th><td>46.1</td><td>0</td><td>0</td><td>0.2</td><td>14.5</td><td>13.5</td><td>36.0</td><td>39.1</td><td>18.7</td></tr><tr><th>Eagle-X5-7B <sup><a href="#fn:150">150</a></sup></th><td>34.7</td><td>17.8</td><td>0</td><td>21.7</td><td>20.6</td><td>21.5</td><td>61.0</td><td>42.6</td><td>27.5</td></tr><tr><th>Idefics3-8B <sup><a href="#fn:151">151</a></sup></th><td>23.8</td><td>13.2</td><td>0</td><td>63.2</td><td>23.8</td><td>23.0</td><td>65.8</td><td>44.9</td><td>32.2</td></tr><tr><th>Phi-4-MultiModal <sup><a href="#fn:152">152</a></sup></th><td>63.7</td><td>16.4</td><td>0</td><td>40.4</td><td>19.1</td><td>18.3</td><td>69.8</td><td>53.9</td><td>35.2</td></tr><tr><th>SAIL-VL-1.6-8B <sup><a href="#fn:153">153</a></sup></th><td>67.7</td><td>28.6</td><td>2.8</td><td>70.5</td><td>25.9</td><td>29.5</td><td>73.9</td><td>59.7</td><td>44.8</td></tr><tr><th>Kimi-VL-A3B-16B <sup><a href="#fn:154">154</a></sup></th><td>56.5</td><td>13.8</td><td>0</td><td>59.2</td><td>33.8</td><td>32.9</td><td>75.5</td><td>56.7</td><td>41.1</td></tr><tr><th>Ovis1.6-3B <sup><a href="#fn:58">58</a></sup></th><td>59.2</td><td>14.3</td><td>0</td><td>65.0</td><td>32.1</td><td>29.0</td><td>69.8</td><td>56.8</td><td>40.8</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>73.2</td><td>24.6</td><td>0.7</td><td>62.4</td><td>44.8</td><td>40.6</td><td>72.7</td><td>62.6</td><td>47.7</td></tr><tr><th colspan="10">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>61.2</td><td>26.7</td><td>0</td><td>77.5</td><td>36.3</td><td>43.4</td><td>71.1</td><td>55.5</td><td>46.5</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>57.9</td><td>23.3</td><td>0.6</td><td>70.8</td><td>31.5</td><td>38.8</td><td>65.9</td><td>55.1</td><td>43.0</td></tr><tr><th>Gemini-Pro <sup><a href="#fn:60">60</a></sup></th><td>61.2</td><td>39.5</td><td>13.5</td><td>79.3</td><td>39.2</td><td>47.7</td><td>75.5</td><td>59.3</td><td>51.9</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>62.2</td><td>28.4</td><td>1.3</td><td>56.6</td><td>37.8</td><td>40.8</td><td>73.5</td><td>60.9</td><td>45.2</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>67.8</td><td>31.3</td><td>7.2</td><td>73.6</td><td>37.2</td><td>27.8</td><td>69.8</td><td>58.6</td><td>46.7</td></tr></tbody></table>

Table 10: Evaluation of existing LMMs on English tasks of OCRBench v2’s public data. “Recognition”, “Referring”, “Spotting”, “Extraction”, “Parsing”, “Calculation”, “Understanding”, and “Reasoning” refer to text recognition, text referring, text spotting, relation extraction, element parsing, mathematical calculation, visual text understanding, and knowledge reasoning, respectively. Higher values indicate better performance. Best performance is in boldface, and the second best is underlined. The notations apply to all subsequent figures.

<table><tbody><tr><th>Method</th><td>LLM Size</td><td>Recognition</td><td>Extraction</td><td>Parsing</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="8">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>8B</td><td>5.7</td><td>2.9</td><td>12.2</td><td>7.5</td><td>17.2</td><td>9.1</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>8B</td><td>14.8</td><td>15.7</td><td>13.7</td><td>16.0</td><td>28.7</td><td>17.8</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>8B</td><td>4.6</td><td>11.2</td><td>8.4</td><td>21.5</td><td>20.0</td><td>13.1</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>8B</td><td>23.5</td><td>14.8</td><td>8.4</td><td>19.9</td><td>12.2</td><td>15.8</td></tr><tr><th>XComposer2-4KHD <sup><a href="#fn:140">140</a></sup></th><td>7B</td><td>16.7</td><td>18.8</td><td>12.1</td><td>27.5</td><td>2.3</td><td>15.5</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>8B</td><td>7.1</td><td>15.0</td><td>9.2</td><td>9.0</td><td>23.7</td><td>12.8</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>8B</td><td>5.3</td><td>14.9</td><td>12.6</td><td>8.5</td><td>8.1</td><td>9.9</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>12B</td><td>13.4</td><td>10.9</td><td>21.0</td><td>7.0</td><td>20.7</td><td>14.6</td></tr><tr><th>EMU2-chat <sup><a href="#fn:141">141</a></sup></th><td>37B</td><td>2.3</td><td>0.5</td><td>8.5</td><td>1.0</td><td>7.3</td><td>3.9</td></tr><tr><th>mPLUG-Owl3 <sup><a href="#fn:142">142</a></sup></th><td>8B</td><td>6.6</td><td>17.9</td><td>9.7</td><td>6.0</td><td>26.1</td><td>13.3</td></tr><tr><th>CogVLM-chat <sup><a href="#fn:143">143</a></sup></th><td>7B</td><td>5.5</td><td>10.0</td><td>9.8</td><td>1.5</td><td>2.5</td><td>5.9</td></tr><tr><th>Qwen-VL <sup><a href="#fn:4">4</a></sup></th><td>8B</td><td>7.2</td><td>5.3</td><td>10.7</td><td>11.5</td><td>11.2</td><td>9.2</td></tr><tr><th>Qwen-VL-chat <sup><a href="#fn:4">4</a></sup></th><td>8B</td><td>9.5</td><td>8.2</td><td>9.3</td><td>11.0</td><td>21.1</td><td>11.8</td></tr><tr><th>Qwen2-Vl-7B <sup><a href="#fn:66">66</a></sup></th><td>7B</td><td>51.3</td><td>51.4</td><td>21.6</td><td>52.5</td><td>37.5</td><td>42.9</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:21">21</a></sup></th><td>7B</td><td>75.3</td><td>61.4</td><td>41.8</td><td>59.3</td><td>40.4</td><td>55.6</td></tr><tr><th>InternVL2-8B <sup><a href="#fn:144">144</a></sup></th><td>8B</td><td>20.6</td><td>45.2</td><td>23.2</td><td>54.4</td><td>38.1</td><td>36.3</td></tr><tr><th>InternVL2-26B <sup><a href="#fn:144">144</a></sup></th><td>26B</td><td>21.9</td><td>46.0</td><td>34.8</td><td>50.9</td><td>34.8</td><td>37.7</td></tr><tr><th>InternVL2.5-8B <sup><a href="#fn:23">23</a></sup></th><td>8B</td><td>52.8</td><td>52.8</td><td>28.6</td><td>56.4</td><td>40.5</td><td>46.2</td></tr><tr><th>InternVL2.5-26B <sup><a href="#fn:23">23</a></sup></th><td>26B</td><td>32.4</td><td>56.1</td><td>32.6</td><td>56.3</td><td>43.6</td><td>44.2</td></tr><tr><th>InternVL3-8B <sup><a href="#fn:23">23</a></sup></th><td>8B</td><td>68.9</td><td>62.0</td><td>31.6</td><td>57.9</td><td>47.3</td><td>53.5</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>14B</td><td>66.2</td><td>64.8</td><td>33.5</td><td>63.4</td><td>50.6</td><td>55.7</td></tr><tr><th>Deepseek-VL-7B <sup><a href="#fn:145">145</a></sup></th><td>7B</td><td>8.0</td><td>13.3</td><td>15.7</td><td>5.5</td><td>18.5</td><td>12.2</td></tr><tr><th>Deepseek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>16B</td><td>60.9</td><td>50.6</td><td>28.3</td><td>53.0</td><td>20.5</td><td>42.7</td></tr><tr><th>MiniCPM-V-2.6 <sup><a href="#fn:56">56</a></sup></th><td>8B</td><td>51.0</td><td>29.9</td><td>21.2</td><td>34.0</td><td>33.6</td><td>33.9</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>7B</td><td>53.0</td><td>49.4</td><td>27.1</td><td>43.5</td><td>32.7</td><td>41.1</td></tr><tr><th>GLM-4V-9B <sup><a href="#fn:57">57</a></sup></th><td>9B</td><td>24.4</td><td>60.6</td><td>20.4</td><td>52.8</td><td>25.2</td><td>36.6</td></tr><tr><th>VILA1.5-8B <sup><a href="#fn:146">146</a></sup></th><td>8B</td><td>5.4</td><td>8.8</td><td>8.5</td><td>3.0</td><td>15.5</td><td>8.2</td></tr><tr><th>LLaVAR <sup><a href="#fn:30">30</a></sup></th><td>13B</td><td>2.3</td><td>1.7</td><td>8.9</td><td>0</td><td>2.5</td><td>3.1</td></tr><tr><th>UReader <sup><a href="#fn:33">33</a></sup></th><td>7B</td><td>6.8</td><td>2.7</td><td>8.4</td><td>2.5</td><td>7.2</td><td>5.5</td></tr><tr><th>DocOwl2 <sup><a href="#fn:147">147</a></sup></th><td>7B</td><td>4.2</td><td>10.3</td><td>8.6</td><td>4.0</td><td>9.6</td><td>7.3</td></tr><tr><th>Yi-VL-6B <sup><a href="#fn:148">148</a></sup></th><td>6B</td><td>4.8</td><td>4.4</td><td>8.5</td><td>4.0</td><td>25.0</td><td>9.4</td></tr><tr><th>Janus-1.3B <sup><a href="#fn:149">149</a></sup></th><td>1.3B</td><td>7.6</td><td>8.7</td><td>11.4</td><td>4.5</td><td>10.7</td><td>8.6</td></tr><tr><th>Eagle-X5-7B <sup><a href="#fn:150">150</a></sup></th><td>8B</td><td>7.5</td><td>12.0</td><td>11.6</td><td>5.0</td><td>19.2</td><td>11.1</td></tr><tr><th>Idefics3-8B <sup><a href="#fn:151">151</a></sup></th><td>8B</td><td>7.0</td><td>15.5</td><td>15.9</td><td>9.0</td><td>18.1</td><td>13.1</td></tr><tr><th>Phi-4-MultiModal <sup><a href="#fn:152">152</a></sup></th><td>5.6B</td><td>51.5</td><td>32.3</td><td>12.1</td><td>34.4</td><td>23.0</td><td>30.7</td></tr><tr><th>SAIL-VL-1.6-8B <sup><a href="#fn:153">153</a></sup></th><td>8B</td><td>31.2</td><td>40.0</td><td>23.9</td><td>42.3</td><td>35.0</td><td>34.5</td></tr><tr><th>Kimi-VL-A3B-16B <sup><a href="#fn:154">154</a></sup></th><td>16B</td><td>57.2</td><td>54.7</td><td>31.5</td><td>52.5</td><td>31.4</td><td>45.5</td></tr><tr><th>Ovis1.6-3B <sup><a href="#fn:58">58</a></sup></th><td>3B</td><td>11.5</td><td>23.7</td><td>22.8</td><td>28.8</td><td>18.9</td><td>21.1</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>7B</td><td>72.2</td><td>50.8</td><td>37.7</td><td>47.9</td><td>37.4</td><td>49.2</td></tr><tr><th colspan="8">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>-</td><td>21.6</td><td>53.0</td><td>29.8</td><td>38.5</td><td>18.2</td><td>32.2</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>-</td><td>13.1</td><td>38.9</td><td>27.2</td><td>28.8</td><td>16.9</td><td>25.0</td></tr><tr><th>Gemini-Pro <sup><a href="#fn:60">60</a></sup></th><td>-</td><td>52.5</td><td>47.3</td><td>30.9</td><td>51.5</td><td>33.4</td><td>43.1</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>-</td><td>21.0</td><td>56.2</td><td>35.2</td><td>55.0</td><td>30.5</td><td>39.6</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>-</td><td>56.7</td><td>41.1</td><td>37.6</td><td>38.3</td><td>39.2</td><td>42.6</td></tr></tbody></table>

Table 11: Evaluation of existing LMMs on Chinese tasks of OCRBench v2’ public data. “LLM Size” indicates the number of parameters of the language model employed in each method.

<table><tbody><tr><th>Method</th><td>Recognition</td><td>Referring</td><td>Spotting</td><td>Extraction</td><td>Parsing</td><td>Calculation</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="10">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>41.4</td><td>17.0</td><td>0</td><td>49.0</td><td>12.9</td><td>16.1</td><td>60.9</td><td>30.5</td><td>28.5</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>45.4</td><td>18.5</td><td>0</td><td>60.0</td><td>15.5</td><td>32.0</td><td>59.0</td><td>39.3</td><td>33.7</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>31.5</td><td>0.1</td><td>0</td><td>34.4</td><td>26.3</td><td>17.7</td><td>61.4</td><td>22.4</td><td>24.2</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>39.8</td><td>1.6</td><td>0</td><td>27.6</td><td>24.8</td><td>10.2</td><td>62.3</td><td>21.2</td><td>23.4</td></tr><tr><th>XComposer2-4KHD <sup><a href="#fn:140">140</a></sup></th><td>39.5</td><td>12.0</td><td>0</td><td>69.7</td><td>26.0</td><td>20.2</td><td>68.2</td><td>35.8</td><td>33.9</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>40.8</td><td>19.5</td><td>0</td><td>51.7</td><td>10.0</td><td>33.9</td><td>67.0</td><td>48.0</td><td>33.9</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>44.0</td><td>19.0</td><td>0</td><td>52.3</td><td>19.0</td><td>20.7</td><td>64.0</td><td>39.3</td><td>32.3</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>45.1</td><td>21.8</td><td>0</td><td>71.6</td><td>21.7</td><td>30.4</td><td>77.3</td><td>39.5</td><td>38.4</td></tr><tr><th>EMU2-chat <sup><a href="#fn:141">141</a></sup></th><td>34.3</td><td>0</td><td>0</td><td>20.4</td><td>21.3</td><td>20.3</td><td>47.1</td><td>18.3</td><td>20.2</td></tr><tr><th>mPLUG-Owl3 <sup><a href="#fn:142">142</a></sup></th><td>34.9</td><td>17.0</td><td>0</td><td>12.0</td><td>14.9</td><td>24.1</td><td>50.7</td><td>25.5</td><td>22.4</td></tr><tr><th>CogVLM-chat <sup><a href="#fn:143">143</a></sup></th><td>40.8</td><td>0</td><td>0</td><td>1.6</td><td>18.6</td><td>10.9</td><td>60.2</td><td>26.8</td><td>19.9</td></tr><tr><th>Qwen-VL <sup><a href="#fn:4">4</a></sup></th><td>35.9</td><td>4.2</td><td>0</td><td>38.7</td><td>28.5</td><td>13.8</td><td>60.1</td><td>16.9</td><td>24.8</td></tr><tr><th>Qwen-VL-chat <sup><a href="#fn:4">4</a></sup></th><td>34.1</td><td>12.6</td><td>0.1</td><td>42.6</td><td>19.5</td><td>18.4</td><td>58.3</td><td>20.3</td><td>25.7</td></tr><tr><th>Qwen2-Vl-7B <sup><a href="#fn:66">66</a></sup></th><td>47.0</td><td>42.0</td><td>1.5</td><td>90.2</td><td>13.7</td><td>36.4</td><td>71.1</td><td>36.6</td><td>42.3</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:66">66</a></sup></th><td>51.5</td><td>24.5</td><td>3.1</td><td>64.8</td><td>13.1</td><td>53.3</td><td>78.6</td><td>45.5</td><td>41.8</td></tr><tr><th>InternVL2-8B <sup><a href="#fn:144">144</a></sup></th><td>43.0</td><td>21.6</td><td>0</td><td>70.2</td><td>19.2</td><td>35.6</td><td>65.9</td><td>33.6</td><td>36.1</td></tr><tr><th>InternVL2-26B <sup><a href="#fn:144">144</a></sup></th><td>56.0</td><td>21.2</td><td>0</td><td>80.5</td><td>23.9</td><td>40.3</td><td>72.1</td><td>40.7</td><td>41.8</td></tr><tr><th>InternVL2.5-8B <sup><a href="#fn:23">23</a></sup></th><td>48.9</td><td>21.2</td><td>0</td><td>82.1</td><td>20.3</td><td>41.2</td><td>67.8</td><td>42.3</td><td>40.5</td></tr><tr><th>InternVL2.5-26B <sup><a href="#fn:23">23</a></sup></th><td>53.5</td><td>21.4</td><td>0</td><td>84.0</td><td>21.4</td><td>51.5</td><td>67.5</td><td>41.5</td><td>42.6</td></tr><tr><th>InternVL3-8B <sup><a href="#fn:23">23</a></sup></th><td>49.7</td><td>22.3</td><td>0.2</td><td>86.8</td><td>22.4</td><td>57.0</td><td>70.7</td><td>53.0</td><td>45.3</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>55.8</td><td>24.5</td><td>2.1</td><td>89.3</td><td>21.0</td><td>59.5</td><td>72.0</td><td>50.0</td><td>46.8</td></tr><tr><th>Deepseek-VL-7B <sup><a href="#fn:145">145</a></sup></th><td>33.5</td><td>13.7</td><td>0</td><td>19.1</td><td>11.7</td><td>24.8</td><td>60.5</td><td>32.5</td><td>24.5</td></tr><tr><th>Deepseek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>56.6</td><td>23.7</td><td>0</td><td>86.4</td><td>18.9</td><td>30.6</td><td>72.2</td><td>39.5</td><td>41.0</td></tr><tr><th>MiniCPM-V-2.6 <sup><a href="#fn:56">56</a></sup></th><td>52.2</td><td>18.6</td><td>0.3</td><td>45.8</td><td>19.6</td><td>20.9</td><td>68.9</td><td>37.3</td><td>33.0</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>54.1</td><td>24.7</td><td>0.3</td><td>74.4</td><td>17.6</td><td>39.2</td><td>75.7</td><td>47.0</td><td>41.6</td></tr><tr><th>GLM-4v-9B <sup><a href="#fn:57">57</a></sup></th><td>52.7</td><td>20.6</td><td>0</td><td>79.4</td><td>15.9</td><td>21.5</td><td>74.7</td><td>32.0</td><td>37.1</td></tr><tr><th>VILA1.5-8B <sup><a href="#fn:146">146</a></sup></th><td>36.0</td><td>14.5</td><td>0</td><td>26.0</td><td>17.4</td><td>20.3</td><td>44.7</td><td>27.0</td><td>23.2</td></tr><tr><th>LLaVAR <sup><a href="#fn:30">30</a></sup></th><td>13.8</td><td>0</td><td>0</td><td>8.3</td><td>15.2</td><td>4.4</td><td>42.4</td><td>15.0</td><td>12.4</td></tr><tr><th>UReader <sup><a href="#fn:33">33</a></sup></th><td>20.9</td><td>0</td><td>0</td><td>0</td><td>20.7</td><td>11.3</td><td>39.0</td><td>20.8</td><td>14.1</td></tr><tr><th>DocOwl2 <sup><a href="#fn:147">147</a></sup></th><td>25.4</td><td>7.5</td><td>0</td><td>47.1</td><td>26.2</td><td>8.3</td><td>52.8</td><td>19.5</td><td>23.4</td></tr><tr><th>Yi-VL-6B <sup><a href="#fn:148">148</a></sup></th><td>31.1</td><td>4.0</td><td>0</td><td>23.4</td><td>22.5</td><td>18.1</td><td>43.0</td><td>15.5</td><td>19.7</td></tr><tr><th>Janus-1.3B <sup><a href="#fn:149">149</a></sup></th><td>32.6</td><td>0</td><td>0</td><td>0.3</td><td>13.0</td><td>18.4</td><td>32.1</td><td>17.9</td><td>14.3</td></tr><tr><th>Eagle-X5-7B <sup><a href="#fn:150">150</a></sup></th><td>34.6</td><td>18.5</td><td>0</td><td>9.7</td><td>18.5</td><td>24.0</td><td>63.1</td><td>37.0</td><td>25.7</td></tr><tr><th>Idefics3-8B <sup><a href="#fn:151">151</a></sup></th><td>37.4</td><td>13.0</td><td>0</td><td>28.9</td><td>19.4</td><td>21.1</td><td>65.4</td><td>21.8</td><td>26.0</td></tr><tr><th>Phi-4-MultiModal <sup><a href="#fn:152">152</a></sup></th><td>58.4</td><td>19.0</td><td>0</td><td>53.5</td><td>38.7</td><td>28.7</td><td>66.8</td><td>39.8</td><td>38.1</td></tr><tr><th>SAIL-VL-1.6-8B <sup><a href="#fn:153">153</a></sup></th><td>56.7</td><td>24.1</td><td>2.2</td><td>79.3</td><td>22.8</td><td>45.4</td><td>69.2</td><td>45.3</td><td>43.1</td></tr><tr><th>Kimi-VL-A3B-16B <sup><a href="#fn:154">154</a></sup></th><td>49.1</td><td>13.5</td><td>0</td><td>28.8</td><td>21.9</td><td>37.6</td><td>69.4</td><td>36.2</td><td>32.1</td></tr><tr><th>Ovis1.6-3B <sup><a href="#fn:58">58</a></sup></th><td>48.5</td><td>19.5</td><td>0</td><td>69.2</td><td>20.7</td><td>22.1</td><td>74.6</td><td>49.5</td><td>38.0</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>54.2</td><td>20.9</td><td>0</td><td>83.6</td><td>24.2</td><td>54.7</td><td>74.1</td><td>57.3</td><td>46.1</td></tr><tr><th colspan="10">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>58.6</td><td>23.4</td><td>0</td><td>87.4</td><td>23.1</td><td>51.6</td><td>74.4</td><td>62.3</td><td>47.6</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>55.3</td><td>21.8</td><td>0</td><td>85.4</td><td>20.6</td><td>45.2</td><td>75.5</td><td>49.0</td><td>44.1</td></tr><tr><th>Gemini1.5-Pro <sup><a href="#fn:60">60</a></sup></th><td>59.1</td><td>41.2</td><td>6.6</td><td>89.5</td><td>22.4</td><td>54.7</td><td>78.8</td><td>60.3</td><td>51.6</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>52.9</td><td>24.9</td><td>2.5</td><td>86.9</td><td>23.8</td><td>61.4</td><td>74.4</td><td>53.0</td><td>47.5</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>56.7</td><td>27.4</td><td>2.6</td><td>86.3</td><td>33.3</td><td>42.6</td><td>76.6</td><td>48.7</td><td>46.8</td></tr></tbody></table>

Table 12: Evaluation of existing LMMs on English tasks of OCRBench v2’s private data.

<table><tbody><tr><th>Method</th><td>LLM Size</td><td>Recognition</td><td>Extraction</td><td>Parsing</td><td>Understanding</td><td>Reasoning</td><td>Average</td></tr><tr><th colspan="8">Open-source LMMs</th></tr><tr><th>LLaVA-Next-8B <sup><a href="#fn:49">49</a></sup></th><td>8B</td><td>2.8</td><td>0.9</td><td>14.9</td><td>20.0</td><td>7.4</td><td>9.2</td></tr><tr><th>LLaVA-OV-7B <sup><a href="#fn:50">50</a></sup></th><td>8B</td><td>5.4</td><td>13.6</td><td>20.3</td><td>34.0</td><td>13.6</td><td>17.4</td></tr><tr><th>Monkey <sup><a href="#fn:51">51</a></sup></th><td>8B</td><td>1.5</td><td>28.4</td><td>29.1</td><td>40.0</td><td>8.3</td><td>21.5</td></tr><tr><th>TextMonkey <sup><a href="#fn:7">7</a></sup></th><td>8B</td><td>10.5</td><td>15.2</td><td>30.2</td><td>44.0</td><td>7.6</td><td>21.5</td></tr><tr><th>XComposer2-4KHD <sup><a href="#fn:140">140</a></sup></th><td>7B</td><td>12.9</td><td>38.6</td><td>37.5</td><td>60.0</td><td>13.1</td><td>32.4</td></tr><tr><th>Molmo-7B <sup><a href="#fn:52">52</a></sup></th><td>8B</td><td>3.4</td><td>29.8</td><td>6.6</td><td>24.0</td><td>11.1</td><td>15.0</td></tr><tr><th>Cambrian-1-8B <sup><a href="#fn:53">53</a></sup></th><td>8B</td><td>2.4</td><td>19.8</td><td>26.7</td><td>36.0</td><td>7.6</td><td>18.5</td></tr><tr><th>Pixtral-12B <sup><a href="#fn:54">54</a></sup></th><td>12B</td><td>6.2</td><td>22.3</td><td>11.4</td><td>26.0</td><td>14.0</td><td>16.0</td></tr><tr><th>EMU2-chat <sup><a href="#fn:141">141</a></sup></th><td>37B</td><td>1.2</td><td>3.0</td><td>29.3</td><td>4.0</td><td>3.6</td><td>8.2</td></tr><tr><th>mPLUG-Owl3 <sup><a href="#fn:142">142</a></sup></th><td>8B</td><td>1.6</td><td>27.4</td><td>27.3</td><td>16.0</td><td>10.0</td><td>16.5</td></tr><tr><th>CogVLM-chat <sup><a href="#fn:143">143</a></sup></th><td>7B</td><td>2.4</td><td>16.2</td><td>22.5</td><td>20.0</td><td>3.1</td><td>12.8</td></tr><tr><th>Qwen-VL <sup><a href="#fn:4">4</a></sup></th><td>8B</td><td>4.3</td><td>0</td><td>30.6</td><td>38.0</td><td>5.1</td><td>15.6</td></tr><tr><th>Qwen-VL-chat <sup><a href="#fn:4">4</a></sup></th><td>8B</td><td>9.1</td><td>3.6</td><td>18.9</td><td>44.0</td><td>7.1</td><td>16.5</td></tr><tr><th>Qwen2-Vl-7B <sup><a href="#fn:66">66</a></sup></th><td>7B</td><td>23.7</td><td>63.5</td><td>27.9</td><td>80.0</td><td>28.5</td><td>44.7</td></tr><tr><th>Qwen2.5-VL-7B <sup><a href="#fn:66">66</a></sup></th><td>8B</td><td>24.4</td><td>78.9</td><td>33.1</td><td>82.0</td><td>29.0</td><td>49.5</td></tr><tr><th>InternVL2-8B <sup><a href="#fn:144">144</a></sup></th><td>8B</td><td>35.2</td><td>42.8</td><td>26.1</td><td>78.0</td><td>24.4</td><td>41.3</td></tr><tr><th>InternVL2-26B <sup><a href="#fn:144">144</a></sup></th><td>26B</td><td>20.4</td><td>50.7</td><td>29.0</td><td>76.0</td><td>14.5</td><td>38.1</td></tr><tr><th>InternVL2.5-8B <sup><a href="#fn:23">23</a></sup></th><td>8B</td><td>42.8</td><td>47.9</td><td>27.3</td><td>80.0</td><td>23.5</td><td>44.3</td></tr><tr><th>InternVL2.5-26B <sup><a href="#fn:23">23</a></sup></th><td>26B</td><td>40.2</td><td>42.7</td><td>25.6</td><td>74.0</td><td>27.0</td><td>41.9</td></tr><tr><th>InternVL3-8B <sup><a href="#fn:23">23</a></sup></th><td>8B</td><td>57.7</td><td>55.8</td><td>29.9</td><td>72.0</td><td>29.4</td><td>49.0</td></tr><tr><th>InternVL3-14B <sup><a href="#fn:23">23</a></sup></th><td>14B</td><td>62.1</td><td>59.5</td><td>33.2</td><td>80.0</td><td>29.2</td><td>52.8</td></tr><tr><th>Deepseek-VL-7B <sup><a href="#fn:145">145</a></sup></th><td>7B</td><td>3.2</td><td>14.7</td><td>10.7</td><td>30.0</td><td>9.8</td><td>13.7</td></tr><tr><th>DeepSeek-VL2-Small <sup><a href="#fn:55">55</a></sup></th><td>16B</td><td>51.6</td><td>56.3</td><td>27.8</td><td>79.6</td><td>25.3</td><td>48.1</td></tr><tr><th>MiniCPM-V-2.6 <sup><a href="#fn:56">56</a></sup></th><td>8B</td><td>53.1</td><td>53.2</td><td>32.8</td><td>76.0</td><td>23.4</td><td>47.7</td></tr><tr><th>MiniCPM-o-2.6 <sup><a href="#fn:56">56</a></sup></th><td>7B</td><td>54.0</td><td>62.4</td><td>24.1</td><td>68.0</td><td>29.8</td><td>47.7</td></tr><tr><th>GLM-4v-9B <sup><a href="#fn:57">57</a></sup></th><td>9B</td><td>60.6</td><td>65.2</td><td>32.4</td><td>82.0</td><td>18.2</td><td>51.7</td></tr><tr><th>VILA1.5-8B <sup><a href="#fn:146">146</a></sup></th><td>8B</td><td>1.4</td><td>9.1</td><td>22.2</td><td>16.0</td><td>6.4</td><td>11.0</td></tr><tr><th>LLaVAR <sup><a href="#fn:30">30</a></sup></th><td>13B</td><td>2.2</td><td>2.0</td><td>27.1</td><td>10.0</td><td>1.9</td><td>8.6</td></tr><tr><th>UReader <sup><a href="#fn:33">33</a></sup></th><td>7B</td><td>0.3</td><td>2.0</td><td>28.1</td><td>12.0</td><td>2.4</td><td>9.0</td></tr><tr><th>DocOwl2 <sup><a href="#fn:147">147</a></sup></th><td>7B</td><td>1.0</td><td>17.8</td><td>29.4</td><td>20.0</td><td>3.9</td><td>14.4</td></tr><tr><th>Yi-VL-6B <sup><a href="#fn:148">148</a></sup></th><td>6B</td><td>1.6</td><td>6.4</td><td>28.8</td><td>10.0</td><td>5.3</td><td>10.4</td></tr><tr><th>Janus-1.3B <sup><a href="#fn:149">149</a></sup></th><td>1.3B</td><td>4.1</td><td>2.2</td><td>10.4</td><td>14.0</td><td>6.7</td><td>7.5</td></tr><tr><th>Eagle-X5-7B <sup><a href="#fn:150">150</a></sup></th><td>8B</td><td>1.9</td><td>16.1</td><td>13.6</td><td>22.0</td><td>8.1</td><td>12.3</td></tr><tr><th>Idefics3-8B <sup><a href="#fn:151">151</a></sup></th><td>8B</td><td>2.9</td><td>29.0</td><td>12.3</td><td>26.0</td><td>7.9</td><td>15.6</td></tr><tr><th>Phi-4-MultiModal <sup><a href="#fn:152">152</a></sup></th><td>5.6B</td><td>30.5</td><td>40.5</td><td>42.7</td><td>56.0</td><td>16.9</td><td>37.3</td></tr><tr><th>SAIL-VL-1.6-8B <sup><a href="#fn:153">153</a></sup></th><td>8B</td><td>35.8</td><td>41.5</td><td>35.7</td><td>76.0</td><td>23.9</td><td>42.6</td></tr><tr><th>Kimi-VL-A3B-16B <sup><a href="#fn:154">154</a></sup></th><td>16B</td><td>54.0</td><td>71.1</td><td>32.5</td><td>84.0</td><td>28.7</td><td>54.1</td></tr><tr><th>Ovis1.6-3B <sup><a href="#fn:58">58</a></sup></th><td>3B</td><td>22.5</td><td>33.3</td><td>31.5</td><td>54.0</td><td>17.0</td><td>31.7</td></tr><tr><th>Ovis2-8B <sup><a href="#fn:58">58</a></sup></th><td>7B</td><td>61.0</td><td>67.7</td><td>43.6</td><td>82.0</td><td>25.6</td><td>56.0</td></tr><tr><th colspan="8">Closed-source LMMs</th></tr><tr><th>GPT-4o <sup><a href="#fn:1">1</a></sup></th><td>-</td><td>41.7</td><td>52.1</td><td>29.0</td><td>76.0</td><td>29.4</td><td>45.7</td></tr><tr><th>GPT-4o-mini <sup><a href="#fn:59">59</a></sup></th><td>-</td><td>20.0</td><td>53.6</td><td>27.9</td><td>66.0</td><td>19.6</td><td>37.4</td></tr><tr><th>Gemini1.5-Pro <sup><a href="#fn:60">60</a></sup></th><td>-</td><td>71.4</td><td>63.8</td><td>30.5</td><td>82.0</td><td>29.9</td><td>55.5</td></tr><tr><th>Claude3.5-sonnet <sup><a href="#fn:61">61</a></sup></th><td>-</td><td>34.2</td><td>62.5</td><td>35.2</td><td>78.0</td><td>32.2</td><td>48.4</td></tr><tr><th>Step-1V <sup><a href="#fn:62">62</a></sup></th><td>-</td><td>65.2</td><td>64.9</td><td>33.1</td><td>78.0</td><td>25.5</td><td>53.4</td></tr></tbody></table>

Table 13: Evaluation of existing LMMs on Chinese tasks of OCRBench v2’s private data.

### A.9 Potential Factors Affecting OCR Capabilities

High-Res Visual Encoders. Since text often appears small in images, the resolution setting of the visual encoder could be a key factor affecting the text perception ability [^51]. Here we change the input resolution of the LMMs and observe the performance changes. In particular, InternVL2-8B is chosen, and the resolution setting includes $448$, $896$, and dynamic. Tab. 14 lists the results. Indeed, when the input resolution increases from $448$ to $896$, the performance increases by $4.1\%$.

Pre-provided OCR Information. To study the impact of OCR information, we use PaddleOCR <sup>15</sup> to pre-extract OCR results and incorporate them with prompts. Tab. 15 shows the results. We observe that adding OCR information does not help much. This suggests that OCRBench v2 evaluates LMMs capabilities across multiple dimensions, rather than solely focusing on text recognition abilities.

Connection Between OCR and LLMs. We further explore a direct pipeline by first extracting OCR information and then by feeding it directly into Qwen2.5. Unlike LMMs, this pipeline separates OCR and language modeling into distinct stages. The results shown in Tab. 15 suggest that Qwen2-VL-7B outperforms Qwen2.5 with OCR information, demonstrating LMMs’ remarkable ability to incorporate both textual and visual features efficiently.

Method Resolition Recognition Referring Spotting Extraction Parsing Calculation Understanding Reasoning Average InternVL2-8B [^144] 448 47.3 19.1 0.1 52.8 27.3 25.4 61.1 49.1 35.3 896 48.7 23.0 0.5 66.2 26.2 25.9 73.2 51.9 39.4 dynamic 49.9 23.1 0.5 65.2 24.8 26.7 73.5 52.9 39.6

Table 14: Evaluation of InternVL2-8B with different resolution settings on the English tasks of OCRBench v2’s public data.

Method Recognition Referring Spotting Extraction Parsing Calculation Understanding Reasoning Average Qwen2-VL-7B [^66] 72.1 47.9 17.5 82.5 25.5 25.4 78.4 61.5 51.4 Qwen2-VL-7B+OCR 69.8 50.4 20.1 79.1 29.4 28.0 77.7 60.0 51.8 Qwen2.5-8B+OCR 28.6 13.8 0 45.9 24.2 31.3 61.1 40.5 30.7

Table 15: Evaluation of Qwen2-VL-7B and Qwen2.5-7B with pre-provided OCR information on English tasks of OCRBench v2’s public data.

### A.10 Samples for Each Task

As show in Fig. 10 to Fig. 18, there are $23$ OCR tasks included in OCRBench v2. Among them, Fig. 10 to Fig. 16 present examples of English tasks, including text recognition, diagram QA, text counting, formula recognition, math QA, VQA with position, ASCII art classification, reasoning VQA, text translation, APP agent, table parsing, cognition VQA, document classification, science QA, chart parsing, key information extraction, full-page OCR, text spotting, fine-grained text recognition, text grounding, key information mapping, and document parsing. These figures show corresponding images and QA pairs for each of the $23$ tasks. Fig. 17 to Fig. 18 provide examples of Chinese tasks, including key information extraction, text translation, formula recognition, reasoning VQA, cognition VQA, handwritten content extraction, document parsing, full-page OCR, and table parsing, along with their associated images and QA pairs.

### A.11 Samples for LMMs’ Limitations

Fig. 19 to Fig. 21 provide examples corresponding to the findings discussed in Sec. 5.3 of the main text, which show error results of GPT-4o [^1], Monkey [^51], and Qwen2VL-8B on various tasks in OCRBench v2. These examples highlight the current limitations of LLMs on OCR tasks. For instance, LLMs exhibit poor recognition of less frequently encountered texts, struggle to accurately locate text in tasks involving text and coordinates, and demonstrate insufficient perception of text in complex layouts such as rotated texts. Additionally, their logical reasoning abilities are limited when addressing mathematical problems, and their analysis of complex elements in charts remains weak. These are the capabilities of LLMs in OCR tasks that require further improvement.

### A.12 Broader Impacts

Our benchmark aims to enhance the evaluation of LMMs in text-oriented visual comprehension tasks. By establishing comprehensive benchmarks that reveal deficiencies in models’ OCR capabilities, we provide insights for improving model performance. This advancement will elevate processing efficiency across scenarios such as document automation, assisted reading tools, and complex layout analysis, thereby benefiting applications in domains like healthcare and education. However, enhanced OCR functionality also introduces risks of misuse, including unauthorized extraction of sensitive information from images, surveillance-related applications, or generation of forged documents. To mitigate these risks, we restrict the use of this benchmark solely to research purposes and urge the community to prioritize privacy and fairness considerations in future model development.

### A.13 Limitations

One challenge we encountered is that LMMs sometimes produce responses that deviate from the given instructions, making it difficult to extract the desired answers. In future work, we plan to develop a more objective assessment framework to address this issue.

Another limitation arises when evaluating commercial LMMs, as some models occasionally refuse to answer certain questions due to safety filters or unclear content policies. This can lead to incomplete or biased performance assessments compared to open-source models that do not exhibit such behavior.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x10.png)

Figure 10: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x11.png)

Figure 11: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x12.png)

Figure 12: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x13.png)

Figure 13: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x14.png)

Figure 14: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x15.png)

Figure 15: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x16.png)

Figure 16: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x17.png)

Figure 17: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x18.png)

Figure 18: Samples for each task.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x19.png)

Figure 19: Samples for LMM’S Limitations. The portion of LLM’s response marked in red is incorrect content, and the content in the red dashed box is missing information.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x20.png)

Figure 20: Samples for LMM’S Limitations. The portion of LLM’s response marked in red is incorrect content, and the content in the red dashed box is missing information.

![Refer to caption](https://arxiv.org/html/2501.00321v2/x21.png)

Figure 21: Samples for LMM’S Limitations. The portion of LLM’s response marked in red is incorrect content, and the content in the red dashed box is missing information.

[^1]: J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat *et al.*, “Gpt-4 technical report,” *arXiv preprint arXiv:2303.08774*, 2023.

[^2]: H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar *et al.*, “Llama: Open and efficient foundation language models,” *arXiv preprint arXiv:2302.13971*, 2023.

[^3]: T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell *et al.*, “Language models are few-shot learners,” *Advances in Neural Information Processing Systems*, 2020.

[^4]: J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A frontier large vision-language model with versatile abilities,” *arXiv preprint arXiv:2308.12966*, 2023.

[^5]: H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” *Advances in Neural Information Processing Systems*, vol. 36, 2024.

[^6]: D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” *Proceedings of the International Conference on Learning Representations*, 2024.

[^7]: Y. Liu, B. Yang, Q. Liu, Z. Li, Z. Ma, S. Zhang, and X. Bai, “Textmonkey: An ocr-free large multimodal model for understanding document,” *arXiv preprint arXiv:2403.04473*, 2024.

[^8]: C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun *et al.*, “MME: A comprehensive evaluation benchmark for multimodal large language models,” *arXiv preprint arXiv:2306.13394*, 2023.

[^9]: K. Ying, F. Meng, J. Wang, Z. Li, H. Lin, Y. Yang, H. Zhang, W. Zhang, Y. Lin, S. Liu *et al.*, “Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi,” *arXiv preprint arXiv:2404.16006*, 2024.

[^10]: A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019, pp. 8317–8326.

[^11]: A. F. Biten, R. Tito, A. Mafla, L. Gomez, M. Rusinol, E. Valveny, C. Jawahar, and D. Karatzas, “Scene text visual question answering,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019, pp. 4291–4301.

[^12]: X. Wang, Y. Liu, C. Shen, C. C. Ng, C. Luo, L. Jin, C. S. Chan, A. v. d. Hengel, and L. Wang, “On the general value of evidence, and bilingual scene-text visual question answering,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2020, pp. 10 126–10 135.

[^13]: L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin *et al.*, “Are We on the Right Way for Evaluating Large Vision-Language Models?” *arXiv preprint arXiv:2403.20330*, 2024.

[^14]: Y. Liu, Z. Li, B. Yang, C. Li, X. Yin, C.-l. Liu, L. Jin, and X. Bai, “On the hidden mystery of ocr in large multimodal models,” *arXiv preprint arXiv:2305.07895*, 2023.

[^15]: B. Li, Y. Ge, Y. Chen, Y. Ge, R. Zhang, and Y. Shan, “Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension,” *arXiv preprint arXiv:2404.16790*, 2024.

[^16]: R. Wadhawan, H. Bansal, K.-W. Chang, and N. Peng, “ConTextual: Evaluating Context-Sensitive Text-Rich Visual Reasoning in Large Multimodal Models,” in *Proceedings of International Conference on Machine Learning*, 2024.

[^17]: C. Liu, H. Wei, J. Chen, L. Kong, Z. Ge, Z. Zhu, L. Zhao, J. Sun, C. Han, and X. Zhang, “Focus Anywhere for Fine-grained Multi-page Document Understanding,” *arXiv preprint arXiv:2405.14295*, 2024.

[^18]: Y. Kim, M. Yim, and K. Y. Song, “TableVQA-Bench: A visual question answering benchmark on multiple table domains,” *arXiv preprint arXiv:2404.19205*, 2024.

[^19]: W. Zhao, H. Feng, Q. Liu, J. Tang, S. Wei, B. Wu, L. Liao, Y. Ye, H. Liu, H. Li *et al.*, “TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy,” *arXiv preprint arXiv:2406.01326*, 2024.

[^20]: R. Xia, B. Zhang, H. Ye, X. Yan, Q. Liu, H. Zhou, Z. Chen, M. Dou, B. Shi, J. Yan *et al.*, “Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning,” *arXiv preprint arXiv:2402.12185*, 2024.

[^21]: Q. Team, “Qwen2.5-vl,” January 2025. \[Online\]. Available: [https://qwenlm.github.io/blog/qwen2.5-vl/](https://qwenlm.github.io/blog/qwen2.5-vl/)

[^22]: M. Mathew, D. Karatzas, and C. Jawahar, “Docvqa: A dataset for vqa on document images,” in *Proceedings of the IEEE Winter Conference on Applications of Computer Vision*, 2021, pp. 2200–2209.

[^23]: Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu *et al.*, “Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling,” *arXiv preprint arXiv:2412.05271*, 2024.

[^24]: OpenAI, “Hello GPT-4o,” [https://openai.com/index/gpt-4v-system-card](https://openai.com/index/gpt-4v-system-card), 2024, accessed: 2024-12-29.

[^25]: L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao *et al.*, “Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations,” *arXiv preprint arXiv:2412.07626*, 2024.

[^26]: Z. Yang, J. Tang, Z. Li, P. Wang, J. Wan, H. Zhong, X. Liu, M. Yang, P. Wang, Y. Liu *et al.*, “Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy,” *arXiv preprint arXiv:2412.02210*, 2024.

[^27]: Y. Ma, Y. Zang, L. Chen, M. Chen, Y. Jiao, X. Li, X. Lu, Z. Liu, Y. Ma, X. Dong *et al.*, “Mmlongbench-doc: Benchmarking long-context document understanding with visualizations,” *arXiv preprint arXiv:2407.01523*, 2024.

[^28]: M. Zheng, X. Feng, Q. Si, Q. She, Z. Lin, W. Jiang, and W. Wang, “Multimodal Table Understanding,” in *Proceedings of Annual Meeting of the Association for Computational Linguistics*, L. Ku, A. Martins, and V. Srikumar, Eds. Association for Computational Linguistics, 2024, pp. 9102–9124. \[Online\]. Available: [https://doi.org/10.18653/v1/2024.acl-long.493](https://doi.org/10.18653/v1/2024.acl-long.493)

[^29]: F. Liu, X. Wang, W. Yao, J. Chen, K. Song, S. Cho, Y. Yacoob, and D. Yu, “MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning,” in *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2024, pp. 1287–1310.

[^30]: Y. Zhang, R. Zhang, J. Gu, Y. Zhou, N. Lipka, D. Yang, and T. Sun, “Llavar: Enhanced visual instruction tuning for text-rich image understanding,” *arXiv preprint arXiv:2306.17107*, 2023.

[^31]: J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu, C. Li, J. Tian *et al.*, “mplug-docowl: Modularized multimodal large language model for document understanding,” *arXiv preprint arXiv:2307.02499*, 2023.

[^32]: H. Feng, Q. Liu, H. Liu, W. Zhou, H. Li, and C. Huang, “Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding,” *arXiv preprint arXiv:2311.11810*, 2023.

[^33]: J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, G. Xu, C. Li, J. Tian, Q. Qian, J. Zhang *et al.*, “Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model,” *arXiv preprint arXiv:2310.05126*, 2023.

[^34]: C. Luo, Y. Shen, Z. Zhu, Q. Zheng, Z. Yu, and C. Yao, “LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 15 630–15 640.

[^35]: A. Hu, H. Xu, J. Ye, M. Yan, L. Zhang, B. Zhang, C. Li, J. Zhang, Q. Jin, F. Huang *et al.*, “mplug-docowl 1.5: Unified structure learning for ocr-free document understanding,” *arXiv preprint arXiv:2403.12895*, 2024.

[^36]: J. Zhang, W. Yang, S. Lai, Z. Xie, and L. Jin, “Dockylin: A large multimodal model for visual document understanding with efficient visual slimming,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 9, 2025, pp. 9923–9932.

[^37]: W. Liao, J. Wang, H. Li, C. Wang, J. Huang, and L. Jin, “Doclayllm: An efficient and effective multi-modal extension of large language models for text-rich document understanding,” *arXiv preprint arXiv:2408.15045*, 2024.

[^38]: Z. Zhu, C. Luo, Z. Shao, F. Gao, H. Xing, Q. Zheng, and J. Zhang, “A simple yet effective layout token in large language models for document understanding,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025.

[^39]: H. Xiao, Y. Xie, G. Tan, Y. Chen, R. Hu, K. Wang, A. Zhou, H. Li, H. Shao, X. Lu, P. Gao, Y. Wen, X. Chen, S. Ren, and H. Li, “Adaptive markup language generation for contextually-grounded visual document understanding,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025.

[^40]: Z. Wang, T. Guan, P. Fu, C. Duan, Q. Jiang, Z. Guo, S. Guo, J. Luo, W. Shen, and X. Yang, “Marten: Visual question answering with mask generation for multi-modal document understanding,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025.

[^41]: A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque, “Chartqa: A benchmark for question answering about charts with visual and logical reasoning,” *arXiv preprint arXiv:2203.10244*, 2022.

[^42]: M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar, “Infographicvqa,” in *Proceedings of the IEEE Winter Conference on Applications of Computer Vision*, 2022, pp. 1697–1706.

[^43]: S. Zhang, B. Yang, Z. Li, Z. Ma, Y. Liu, and X. Bai, “Exploring the Capabilities of Large Multimodal Models on Dense Text,” in *Proceedings of International Conference on Document Analysis and Recognition*. Springer, 2024, pp. 281–298.

[^44]: J. Chen, L. Kong, H. Wei, C. Liu, Z. Ge, L. Zhao, J. Sun, C. Han, and X. Zhang, “Onechart: Purify the chart structural extraction via one auxiliary token,” in *Proceedings of the ACM International Conference on Multimedia*, 2024, pp. 147–155.

[^45]: J. Van Landeghem, R. Tito, Ł. Borchmann, M. Pietruszka, P. Joziak, R. Powalski, D. Jurkiewicz, M. Coustaty, B. Anckaert, E. Valveny *et al.*, “Document understanding dataset and evaluation (dude),” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 19 528–19 540.

[^46]: W. Wang, S. Zhang, Y. Ren, Y. Duan, T. Li, S. Liu, M. Hu, Z. Chen, K. Zhang, L. Lu *et al.*, “Needle in a multimodal haystack,” *Advances in Neural Information Processing Systems*, vol. 37, pp. 20 540–20 565, 2025.

[^47]: R. Tito, D. Karatzas, and E. Valveny, “Hierarchical multimodal transformers for multipage docvqa,” *Pattern Recognition*, vol. 144, p. 109834, 2023.

[^48]: C. Deng, J. Yuan, P. Bu, P. Wang, Z.-Z. Li, J. Xu, X.-H. Li, Y. Gao, J. Song, B. Zheng *et al.*, “Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating,” *arXiv preprint arXiv:2412.18424*, 2024.

[^49]: H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llava-next: Improved reasoning, ocr, and world knowledge,” 2024.

[^50]: B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, Y. Li, Z. Liu, and C. Li, “Llava-onevision: Easy visual task transfer,” *arXiv preprint arXiv:2408.03326*, 2024.

[^51]: Z. Li, B. Yang, Q. Liu, Z. Ma, S. Zhang, J. Yang, Y. Sun, Y. Liu, and X. Bai, “Monkey: Image resolution and text label are important things for large multi-modal models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 26 763–26 773.

[^52]: M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini *et al.*, “Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models,” *arXiv preprint arXiv:2409.17146*, 2024.

[^53]: S. Tong, E. L. Brown II, P. Wu, S. Woo, A. J. IYER, S. C. Akula, S. Yang, J. Yang, M. Middepogu, Z. Wang *et al.*, “Cambrian-1: A fully open, vision-centric exploration of multimodal llms,” in *Advances in Neural Information Processing Systems*, 2024.

[^54]: P. Agrawal, S. Antoniak, E. B. Hanna, B. Bout, D. Chaplot, J. Chudnovsky, D. Costa, B. De Monicault, S. Garg, T. Gervet *et al.*, “Pixtral 12b,” *arXiv preprint arXiv:2410.07073*, 2024.

[^55]: Z. Wu, X. Chen, Z. Pan, X. Liu, W. Liu, D. Dai, H. Gao, Y. Ma, C. Wu, B. Wang *et al.*, “Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding,” *arXiv preprint arXiv:2412.10302*, 2024.

[^56]: Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He *et al.*, “Minicpm-v: A gpt-4v level mllm on your phone,” *arXiv preprint arXiv:2408.01800*, 2024.

[^57]: T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Rojas, G. Feng, H. Zhao, H. Lai *et al.*, “ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools,” *arXiv preprint arXiv:2406.12793*, 2024.

[^58]: S. Lu, Y. Li, Q.-G. Chen, Z. Xu, W. Luo, K. Zhang, and H.-J. Ye, “Ovis: Structural embedding alignment for multimodal large language model,” *arXiv preprint arXiv:2405.20797*, 2024.

[^59]: OpenAI, “GPT-4o mini: advancing cost-efficient intelligence,” [https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence), 2024, accessed: 2024-12-29.

[^60]: G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican *et al.*, “Gemini: a family of highly capable multimodal models,” *arXiv preprint arXiv:2312.11805*, 2023.

[^61]: Anthropic, “Claude 3.5 Sonnet,” [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet), 2024, accessed: 2024-12-29.

[^62]: StepFun, “Step-1V,” [https://www.stepfun.com/##step1v](https://www.stepfun.com/##step1v), 2024, accessed: 2024-12-29.

[^63]: X. Zhong, E. ShafieiBavani, and A. Jimeno Yepes, “Image-based table recognition: data, model, and evaluation,” in *Proceedings of European Conference on Computer Vision*. Springer, 2020, pp. 564–580.

[^64]: K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in *Proceedings of Annual Meeting of the Association for Computational Linguistics*, 2002, pp. 311–318.

[^65]: S. Banerjee and A. Lavie, “METEOR: An automatic metric for mt evaluation with improved correlation with human judgments,” in *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, 2005, pp. 65–72.

[^66]: P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge *et al.*, “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” *arXiv preprint arXiv:2409.12191*, 2024.

[^67]: B. Shi, X. Bai, and C. Yao, “An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 39, no. 11, pp. 2298–2304, 2016.

[^68]: S. Fang, H. Xie, Y. Wang, Z. Mao, and Y. Zhang, “Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021, pp. 7098–7107.

[^69]: B. Shi, M. Yang, X. Wang, P. Lyu, C. Yao, and X. Bai, “Aster: An attentional scene text recognizer with flexible rectification,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 41, no. 9, pp. 2035–2048, 2018.

[^70]: N. Lu, W. Yu, X. Qi, Y. Chen, P. Gong, R. Xiao, and X. Bai, “Master: Multi-aspect non-local network for scene text recognition,” *Pattern Recognition*, vol. 117, p. 107980, 2021.

[^71]: Y. Du, Z. Chen, C. Jia, X. Yin, T. Zheng, C. Li, Y. Du, and Y. Jiang, “SVTR: scene text recognition with a single visual model,” in *Proceedings of the International Joint Conference on Artificial Intelligence*, L. D. Raedt, Ed. ijcai.org, 2022, pp. 884–890. \[Online\]. Available: [https://doi.org/10.24963/ijcai.2022/124](https://doi.org/10.24963/ijcai.2022/124)

[^72]: Y. Liu, H. Chen, C. Shen, T. He, L. Jin, and L. Wang, “Abcnet: Real-time scene text spotting with adaptive bezier-curve network,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2020, pp. 9809–9818.

[^73]: Y. Liu, C. Shen, L. Jin, T. He, P. Chen, C. Liu, and H. Chen, “Abcnet v2: Adaptive bezier-curve network for real-time end-to-end text spotting,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 11, pp. 8048–8064, 2021.

[^74]: X. Zhang, Y. Su, S. Tripathi, and Z. Tu, “Text spotting transformers,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 9519–9528.

[^75]: C. K. Ch’ng and C. S. Chan, “Total-text: A comprehensive dataset for scene text detection and recognition,” in *Proceedings of International Conference on Document Analysis and Recognition*, vol. 1. IEEE, 2017, pp. 935–942.

[^76]: H. Wei, C. Liu, J. Chen, J. Wang, L. Kong, Y. Xu, Z. Ge, L. Zhao, J. Sun, Y. Peng *et al.*, “General ocr theory: Towards ocr-2.0 via a unified end-to-end model,” *arXiv preprint arXiv:2409.01704*, 2024.

[^77]: D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i. Bigorda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazàn, and L. P. de las Heras, “Icdar 2013 robust reading competition,” in *Proceedings of International Conference on Document Analysis and Recognition*, 2013, pp. 1484–1493.

[^78]: C. Shi, C. Wang, B. Xiao, S. Gao, and J. Hu, “End-to-end scene text recognition using tree-structured models,” *Pattern Recognition*, vol. 47, pp. 2853–2866, 2014. \[Online\]. Available: [https://api.semanticscholar.org/CorpusID:30201169](https://api.semanticscholar.org/CorpusID:30201169)

[^79]: A. Mishra, K. Alahari, and C. V. Jawahar, “Scene text recognition using higher order language priors,” in *British Machine Vision Conference*, 2012.

[^80]: D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R. Chandrasekhar, S. Lu *et al.*, “Icdar 2015 competition on robust reading,” in *Proceedings of International Conference on Document Analysis and Recognition*. IEEE, 2015, pp. 1156–1160.

[^81]: Y. Liu, L. Jin, S. Zhang, C. Luo, and S. Zhang, “Curved scene text detection via transverse and longitudinal sequence connection,” *Pattern Recognition*, vol. 90, no. C, p. 337–345, Jun. 2019. \[Online\]. Available: [https://doi.org/10.1016/j.patcog.2019.02.002](https://doi.org/10.1016/j.patcog.2019.02.002)

[^82]: A. Veit, T. Matera, L. Neumann, J. Matas, and S. Belongie, “Coco-text: Dataset and benchmark for text detection and recognition in natural images,” *arXiv preprint arXiv:1601.07140*, 2016.

[^83]: A. Risnumawan, P. Shivakumara, C. S. Chan, and C. L. Tan, “A robust arbitrary text detection system for natural scene images,” *Expert Systems with Applications*, vol. 41, pp. 8027–8048, 2014. \[Online\]. Available: [https://api.semanticscholar.org/CorpusID:15559857](https://api.semanticscholar.org/CorpusID:15559857)

[^84]: T. Q. Phan, P. Shivakumara, S. Tian, and C. L. Tan, “Recognizing text with perspective distortion in natural scenes,” in *Proceedings of IEEE/CVF International Conference on Computer Vision*. IEEE Computer Society, 2013, pp. 569–576. \[Online\]. Available: [https://doi.org/10.1109/ICCV.2013.76](https://doi.org/10.1109/ICCV.2013.76)

[^85]: X. Xie, L. Fu, Z. Zhang, Z. Wang, and X. Bai, “Toward understanding wordart: Corner-guided transformer for scene text recognition,” in *Proceedings of European Conference on Computer Vision*, S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner, Eds. Cham: Springer Nature Switzerland, 2022, pp. 303–321.

[^86]: U.-V. Marti and H. Bunke, “The iam-database: an english sentence database for offline handwriting recognition,” *International Journal on Document Analysis and Recognition*, vol. 5, pp. 39–46, 2002.

[^87]: M. Diem, S. Fiel, F. Kleber, R. Sablatnig, J. M. Saavedra, D. Contreras, J. M. Barrios, and L. S. Oliveira, “Proceedings of ieee international conference on frontiers in handwriting recognition,” in *2014 14th International Conference on Frontiers in Handwriting Recognition*, 2014, pp. 779–784.

[^88]: Y. Wang, H. Xie, S. Fang, J. Wang, S. Zhu, and Y. Zhang, “From two to one: A new scene text recognizer with visual language modeling network,” in *Proceedings of IEEE/CVF International Conference on Computer Vision*, 2021, pp. 14 194–14 203.

[^89]: L. Yuliang, J. Lianwen, Z. Shuaitao, and Z. Sheng, “Detecting curve text in the wild: New dataset and new solution,” *arXiv preprint arXiv:1712.02170*, 2017.

[^90]: S. Long, S. Qin, D. Panteleev, A. Bissacco, Y. Fujii, and M. Raptis, “Towards end-to-end unified scene text detection and layout analysis,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 1049–1059.

[^91]: T.-L. Yuan, Z. Zhu, K. Xu, C.-J. Li, T.-J. Mu, and S.-M. Hu, “A large chinese text dataset in the wild,” *Journal of Computer Science and Technology*, vol. 34, no. 3, pp. 509–521, 2019. \[Online\]. Available: [https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-019-1923-y](https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-019-1923-y)

[^92]: B. Shi, C. Yao, M. Liao, M. Yang, P. Xu, L. Cui, S. Belongie, S. Lu, and X. Bai, “Icdar2017 competition on reading chinese text in the wild (rctw-17),” in *Proceedings of International Conference on Document Analysis and Recognition*, vol. 1. IEEE, 2017, pp. 1429–1434.

[^93]: X. Liu, R. Zhang, Y. Zhou, Q. Jiang, Q. Song, N. Li, K. Zhou, L. Wang, D. Wang, M. Liao, M. Yang, X. Bai, B. Shi, D. Karatzas, S. Lu, and C. V. Jawahar, “Icdar 2019 robust reading challenge on reading chinese text on signboard,” 2019. \[Online\]. Available: [https://arxiv.org/abs/1912.09641](https://arxiv.org/abs/1912.09641)

[^94]: Y. Sun, J. Liu, W. Liu, J. Han, E. Ding, and J. Liu, “Chinese street view text: Large-scale chinese text reading with partially supervised learning,” 2020. \[Online\]. Available: [https://arxiv.org/abs/1909.07808](https://arxiv.org/abs/1909.07808)

[^95]: H. Cheng, P. Zhang, S. Wu, J. Zhang, Q. Zhu, Z. Xie, J. Li, K. Ding, and L. Jin, “M <sup>6</sup> doc: A large-scale multi-format, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis,” 2023. \[Online\]. Available: [https://arxiv.org/abs/2305.08719](https://arxiv.org/abs/2305.08719)

[^96]: B. Deka, Z. Huang, C. Franzen, J. Hibschman, D. Afergan, Y. Li, J. Nichols, and R. Kumar, “Rico: A mobile app dataset for building data-driven design applications,” in *Proceedings of the 30th annual ACM symposium on user interface software and technology*, 2017, pp. 845–854.

[^97]: G. Jaume, H. K. Ekenel, and J.-P. Thiran, “Funsd: A dataset for form understanding in noisy scanned documents,” in *Proceedings of International Conference on Document Analysis and Recognition Workshops*, vol. 2. IEEE, 2019, pp. 1–6.

[^98]: Z. Huang, K. Chen, J. He, X. Bai, D. Karatzas, S. Lu, and C. Jawahar, “Icdar2019 competition on scanned receipt ocr and information extraction,” in *Proceedings of International Conference on Document Analysis and Recognition*. IEEE, 2019, pp. 1516–1520.

[^99]: J. Kuang, W. Hua, D. Liang, M. Yang, D. Jiang, B. Ren, and X. Bai, “Visual information extraction in the wild: practical dataset and end-to-end solution,” in *Proceedings of International Conference on Document Analysis and Recognition*. Springer, 2023, pp. 36–53.

[^100]: Y. Xu, T. Lv, L. Cui, G. Wang, Y. Lu, D. Florencio, C. Zhang, and F. Wei, “XFUND: A benchmark dataset for multilingual visually rich form understanding,” in *Proceedings of Annual Meeting of the Association for Computational Linguistics*. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 3214–3224. \[Online\]. Available: [https://aclanthology.org/2022.findings-acl.253](https://aclanthology.org/2022.findings-acl.253)

[^101]: W. Yu, C. Zhang, H. Cao, W. Hua, B. Li, H. Chen, M. Liu, M. Chen, J. Kuang, M. Cheng, Y. Du, S. Feng, X. Hu, P. Lyu, K. Yao, Y. Yu, Y. Liu, W. Che, E. Ding, C.-L. Liu, J. Luo, S. Yan, M. Zhang, D. Karatzas, X. Sun, J. Wang, and X. Bai, “Icdar 2023 competition on structured text extraction from visually-rich document images,” 2023. \[Online\]. Available: [https://arxiv.org/abs/2306.03287](https://arxiv.org/abs/2306.03287)

[^102]: L. Rujiao, W. Wen, X. Nan, G. Feiyu, Y. Zhibo, W. Yongpan, and X. Gui-Song, “Parsing table structures in the wild,” in *Proceedings of IEEE/CVF International Conference on Computer Vision*, October 2021.

[^103]: F. Yang, L. Hu, X. Liu, S. Huang, and Z. Gu, “A large-scale dataset for end-to-end table recognition in the wild,” *Scientific Data*, vol. 10, no. 1, p. 110, 2023.

[^104]: Y. Liang, Y. Zhang, C. Ma, Z. Zhang, Y. Zhao, L. Xiang, C. Zong, and Y. Zhou, “Document image machine translation with dynamic multi-pre-trained models assembling,” in *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2024, pp. 7077–7088.

[^105]: M. Mathew, D. Karatzas, and C. V. Jawahar, “Docvqa: A dataset for VQA on document images,” in *Proceedings of the IEEE Winter Conference on Applications of Computer Vision*. IEEE, 2021, pp. 2199–2208.

[^106]: Y. Yuan, X. Liu, W. Dikubab, H. Liu, Z. Ji, Z. Wu, and X. Bai, “Syntax-aware network for handwritten mathematical expression recognition,” *arXiv preprint arXiv:2203.01601*, 2022.

[^107]: K. Wang, J. Pan, W. Shi, Z. Lu, M. Zhan, and H. Li, “Measuring multimodal mathematical reasoning with math-vision dataset,” *CoRR*, vol. abs/2402.14804, 2024.

[^108]: W. Yang, Z. Li, D. Peng, L. Jin, M. He, and C. Yao, “Read ten lines at one glance: Line-aware semi-autoregressive transformer for multi-line handwritten mathematical expression recognition,” in *Proceedings of the ACM International Conference on Multimedia*, A. El-Saddik, T. Mei, R. Cucchiara, M. Bertini, D. P. T. Vallejo, P. K. Atrey, and M. S. Hossain, Eds. ACM, 2023, pp. 2066–2077.

[^109]: P. Gervais, A. Fadeeva, and A. Maksai, “Mathwriting: A dataset for handwritten mathematical expression recognition,” *CoRR*, vol. abs/2404.10690, 2024.

[^110]: L. Ding, M. Zhao, F. Yin, S. Zeng, and C.-L. Liu, “A large-scale database for chemical structure recognition and preliminary evaluation,” in *Proceedings of the International Conference on Pattern Recognition*, 2022, pp. 1464–1470.

[^111]: D. Saxton, E. Grefenstette, F. Hill, and P. Kohli, “Analysing mathematical reasoning abilities of neural models,” in *Proceedings of the International Conference on Learning Representations*. OpenReview.net, 2019.

[^112]: R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K. Chang, Y. Qiao, P. Gao, and H. Li, “MATHVERSE: does your multi-modal LLM truly see the diagrams in visual math problems?” in *Proceedings of European Conference on Computer Vision*, ser. Lecture Notes in Computer Science, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds., vol. 15066. Springer, 2024, pp. 169–186.

[^113]: P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” in *Proceedings of the International Conference on Learning Representations*. OpenReview.net, 2024.

[^114]: A. Mishra, S. Shekhar, A. K. Singh, and A. Chakraborty, “Ocr-vqa: Visual question answering by reading text in images,” in *Proceedings of International Conference on Document Analysis and Recognition*. IEEE, 2019, pp. 947–952.

[^115]: H. Feng, W. Zhou, J. Deng, Y. Wang, and H. Li, “Geometric representation learning for document image rectification,” in *Proceedings of European Conference on Computer Vision*. Springer, 2022, pp. 475–492.

[^116]: K. Kafle, S. Cohen, B. Price, and C. Kanan, “Dvqa: Understanding data visualizations via question answering,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018.

[^117]: N. Methani, P. Ganguly, M. M. Khapra, and P. Kumar, “Plotqa: Reasoning over scientific plots,” in *Proceedings of the IEEE Winter Conference on Applications of Computer Vision*, March 2020.

[^118]: M. Mathew, V. Bagal, R. P. Tito, D. Karatzas, E. Valveny, and C. V. Jawahar, “Infographicvqa,” *CoRR*, vol. abs/2104.12756, 2021. \[Online\]. Available: [https://arxiv.org/abs/2104.12756](https://arxiv.org/abs/2104.12756)

[^119]: X. Zhong, E. ShafieiBavani, and A. J. Yepes, “Image-based table recognition: data, model, and evaluation,” *arXiv preprint arXiv:1911.10683*, 2019.

[^120]: P. Pasupat and P. Liang, “Compositional semantic parsing on semi-structured tables,” in *Proceedings of Annual Meeting of the Association for Computational Linguistics*, C. Zong and M. Strube, Eds. Beijing, China: Association for Computational Linguistics, Jul. 2015, pp. 1470–1480. \[Online\]. Available: [https://aclanthology.org/P15-1142](https://aclanthology.org/P15-1142)

[^121]: S. Park, S. Shin, B. Lee, J. Lee, J. Surh, M. Seo, and H. Lee, “Cord: a consolidated receipt dataset for post-ocr parsing,” in *Advances in Neural Information Processing Systems Workshop*, 2019.

[^122]: X. Chen, Z. Zhao, L. Chen, D. Zhang, J. Ji, A. Luo, Y. Xiong, and K. Yu, “Websrc: A dataset for web-based structural reading comprehension,” *arXiv preprint arXiv:2101.09465*, 2021.

[^123]: X. Zhong, J. Tang, and A. Jimeno-Yepes, “Publaynet: Largest dataset ever for document layout analysis,” in *Proceedings of International Conference on Document Analysis and Recognition*. IEEE, 2019, pp. 1015–1022.

[^124]: A. W. Harley, A. Ufkes, and K. G. Derpanis, “Evaluation of deep convolutional nets for document image classification and retrieval,” in *Proceedings of International Conference on Document Analysis and Recognition*, 2015.

[^125]: G. Baechler, S. Sunkara, M. Wang, F. Zubach, H. Mansoor, V. Etter, V. Cărbune, J. Lin, J. Chen, and A. Sharma, “Screenai: A vision-language model for ui and infographics understanding,” 2024.

[^126]: R. Tanaka, K. Nishida, K. Nishida, T. Hasegawa, I. Saito, and K. Saito, “Slidevqa: A dataset for document visual question answering on multiple images,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, 2023.

[^127]: A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi, “A diagram is worth a dozen images,” in *Proceedings of European Conference on Computer Vision*. Springer, 2016, pp. 235–251.

[^128]: A. Kembhavi, M. Seo, D. Schwenk, J. Choi, A. Farhadi, and H. Hajishirzi, “Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2017, pp. 4999–5007.

[^129]: S. Lee, B. Lai, F. Ryan, B. Boote, and J. M. Rehg, “Modeling multimodal social interactions: New challenges and baselines with densely aligned representations,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 14 585–14 595.

[^130]: G. Zhang, X. Du, B. Chen, Y. Liang, T. Luo, T. Zheng, K. Zhu, Y. Cheng, C. Xu, S. Guo, H. Zhang, X. Qu, J. Wang, R. Yuan, Y. Li, Z. Wang, Y. Liu, Y.-H. Tsai, F. Zhang, C. Lin, W. Huang, and J. Fu, “Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark,” 2024. \[Online\]. Available: [https://arxiv.org/abs/2401.11944](https://arxiv.org/abs/2401.11944)

[^131]: P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, “Learn to explain: Multimodal reasoning via thought chains for science question answering,” *Advances in Neural Information Processing Systems*, vol. 35, pp. 2507–2521, 2022.

[^132]: X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang, H. Sun *et al.*, “Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark,” *arXiv preprint arXiv:2409.02813*, 2024.

[^133]: Q. Jia, X. Yue, S. Huang, Z. Qin, Y. Liu, B. Y. Lin, and Y. You, “Visual perception in text strings,” *arXiv preprint arXiv:2410.01733*, 2024.

[^134]: C. Yao, X. Bai, W. Liu, Y. Ma, and Z. Tu, “Detecting texts of arbitrary orientations in natural images,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. IEEE, 2012, pp. 1083–1090.

[^135]: M. He, Y. Liu, Z. Yang, S. Zhang, C. Luo, F. Gao, Q. Zheng, Y. Wang, X. Zhang, and L. Jin, “Icpr2018 contest on robust reading for multi-type web images,” in *Proceedings of the International Conference on Pattern Recognition*, 2018, pp. 7–12.

[^136]: B. Shi, C. Yao, M. Liao, M. Yang, P. Xu, L. Cui, S. Belongie, S. Lu, and X. Bai, “Icdar2017 competition on reading chinese text in the wild (rctw-17),” 2018. \[Online\]. Available: [https://arxiv.org/abs/1708.09585](https://arxiv.org/abs/1708.09585)

[^137]: J. Tang, Z. Yang, Y. Wang, Q. Zheng, Y. Xu, and X. Bai, “Seglink++: Detecting dense and arbitrary-shaped scene text by instance-aware component grouping,” *Pattern Recognition*, vol. 96, p. 106954, 2019. \[Online\]. Available: [https://www.sciencedirect.com/science/article/pii/S0031320319302511](https://www.sciencedirect.com/science/article/pii/S0031320319302511)

[^138]: C. K. Chng, Y. Liu, Y. Sun, C. C. Ng, C. Luo, Z. Ni, C. Fang, S. Zhang, J. Han, E. Ding *et al.*, “Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art,” in *Proceedings of International Conference on Document Analysis and Recognition*. IEEE, 2019, pp. 1571–1576.

[^139]: X. Zheng, D. Burdick, L. Popa, X. Zhong, and N. X. R. Wang, “Global table extractor (GTE): A framework for joint table identification and cell structure recognition using visual context,” in *Proceedings of the IEEE Winter Conference on Applications of Computer Vision*. IEEE, 2021, pp. 697–706. \[Online\]. Available: [https://doi.org/10.1109/WACV48630.2021.00074](https://doi.org/10.1109/WACV48630.2021.00074)

[^140]: X. Dong, P. Zhang, Y. Zang, Y. Cao, B. Wang, L. Ouyang, S. Zhang, H. Duan, W. Zhang, Y. Li *et al.*, “Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd,” *arXiv preprint arXiv:2404.06512*, 2024.

[^141]: Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang, “Generative multimodal models are in-context learners,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 14 398–14 409.

[^142]: J. Ye, H. Xu, H. Liu, A. Hu, M. Yan, Q. Qian, J. Zhang, F. Huang, and J. Zhou, “mplug-owl3: Towards long image-sequence understanding in multi-modal large language models,” *arXiv preprint arXiv:2408.04840*, 2024.

[^143]: W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song *et al.*, “Cogvlm: Visual expert for pretrained language models,” *arXiv preprint arXiv:2311.03079*, 2023.

[^144]: Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu *et al.*, “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 24 185–24 198.

[^145]: H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, H. Yang *et al.*, “Deepseek-vl: towards real-world vision-language understanding,” *arXiv preprint arXiv:2403.05525*, 2024.

[^146]: Z. Liu, L. Zhu, B. Shi, Z. Zhang, Y. Lou, S. Yang, H. Xi, S. Cao, Y. Gu, D. Li *et al.*, “Nvila: Efficient frontier visual language models,” *arXiv preprint arXiv:2412.04468*, 2024.

[^147]: A. Hu, H. Xu, L. Zhang, J. Ye, M. Yan, J. Zhang, Q. Jin, F. Huang, and J. Zhou, “mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding,” *arXiv preprint arXiv:2409.03420*, 2024.

[^148]: A. Young, B. Chen, C. Li, C. Huang, G. Zhang, G. Zhang, H. Li, J. Zhu, J. Chen, J. Chang *et al.*, “Yi: Open foundation models by 01. ai,” *arXiv preprint arXiv:2403.04652*, 2024.

[^149]: C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan *et al.*, “Janus: Decoupling visual encoding for unified multimodal understanding and generation,” *arXiv preprint arXiv:2410.13848*, 2024.

[^150]: M. Shi, F. Liu, S. Wang, S. Liao, S. Radhakrishnan, D.-A. Huang, H. Yin, K. Sapra, Y. Yacoob, H. Shi *et al.*, “Eagle: Exploring the design space for multimodal llms with mixture of encoders,” *arXiv preprint arXiv:2408.15998*, 2024.

[^151]: H. Laurençon, A. Marafioti, V. Sanh, and L. Tronchon, “Building and better understanding vision-language models: insights and future directions,” in *Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models*, 2024.

[^152]: A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen *et al.*, “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras,” *arXiv preprint arXiv:2503.01743*, 2025.

[^153]: H. Duan, J. Yang, Y. Qiao, X. Fang, L. Chen, Y. Liu, X. Dong, Y. Zang, P. Zhang, J. Wang *et al.*, “Vlmevalkit: An open-source toolkit for evaluating large multi-modality models,” in *Proceedings of the ACM International Conference on Multimedia*, 2024, pp. 11 198–11 201.

[^154]: K. Team, A. Du, B. Yin, B. Xing, B. Qu, B. Wang, C. Chen, C. Zhang, C. Du, C. Wei, C. Wang, D. Zhang, D. Du, D. Wang, E. Yuan, E. Lu, F. Li, F. Sung, G. Wei, G. Lai, H. Zhu, H. Ding, H. Hu, H. Yang, H. Zhang, H. Wu, H. Yao, H. Lu, H. Wang, H. Gao, H. Zheng, J. Li, J. Su, J. Wang, J. Deng, J. Qiu, J. Xie, J. Wang, J. Liu, J. Yan, K. Ouyang, L. Chen, L. Sui, L. Yu, M. Dong, M. Dong, N. Xu, P. Cheng, Q. Gu, R. Zhou, S. Liu, S. Cao, T. Yu, T. Song, T. Bai, W. Song, W. He, W. Huang, W. Xu, X. Yuan, X. Yao, X. Wu, X. Zu, X. Zhou, X. Wang, Y. Charles, Y. Zhong, Y. Li, Y. Hu, Y. Chen, Y. Wang, Y. Liu, Y. Miao, Y. Qin, Y. Chen, Y. Bao, Y. Wang, Y. Kang, Y. Liu, Y. Du, Y. Wu, Y. Wang, Y. Yan, Z. Zhou, Z. Li, Z. Jiang, Z. Zhang, Z. Yang, Z. Huang, Z. Huang, Z. Zhao, and Z. Chen, “Kimi-VL technical report,” 2025. \[Online\]. Available: [https://arxiv.org/abs/2504.07491](https://arxiv.org/abs/2504.07491)