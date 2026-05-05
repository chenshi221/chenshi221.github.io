---
title: "OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations"
source: https://arxiv.org/html/2412.07626v2
author:
published:
created: 2026-04-30
description:
tags:
  - clippings
pdf: "file:///Users/chenshi/Desktop/paper/02_%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B/OmniDocBench%2C%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations%2C%20Linke%20Ouyang%20et%20al.%2C%202024.no_watermark.zh-CN.dual.pdf"
---
Linke Ouyang <sup>1∗</sup>  Yuan Qu <sup>1∗</sup>  Hongbin Zhou <sup>1∗</sup>  Jiawei Zhu <sup>1∗</sup>  Rui Zhang <sup>1∗</sup>  Qunshu Lin <sup>2∗</sup>    
Bin Wang <sup>1∗†</sup>  Zhiyuan Zhao <sup>1</sup> Man Jiang <sup>1</sup> Xiaomeng Zhao <sup>1</sup> Jin Shi <sup>1</sup> Fan Wu <sup>1</sup> Pei Chu <sup>1</sup> Minghao Liu <sup>3</sup>  
Zhenxiang Li <sup>1</sup> Chao Xu <sup>1</sup> Bo Zhang <sup>1</sup> Botian Shi <sup>1</sup> Zhongying Tu <sup>1</sup> Conghui He <sup>1‡</sup>  
<sup>1</sup> Shanghai AI Laboratory  <sup>2</sup> Abaka AI  <sup>3</sup> 2077AI 

###### Abstract

Document content extraction is a critical task in computer vision, underpinning the data needs of large language models (LLMs) and retrieval-augmented generation (RAG) systems. Despite recent progress, current document parsing methods have not been fairly and comprehensively evaluated due to the narrow coverage of document types and the simplified, unrealistic evaluation procedures in existing benchmarks. To address these gaps, we introduce OmniDocBench, a novel benchmark featuring high-quality annotations across nine document sources, including academic papers, textbooks, and more challenging cases such as handwritten notes and densely typeset newspapers. OmniDocBench supports flexible, multi-level evaluations—ranging from an end-to-end assessment to the task-specific and attribute-based analysis—using 19 layout categories and 15 attribute labels. We conduct a thorough evaluation of both pipeline-based methods and end-to-end vision-language models, revealing their strengths and weaknesses across different document types. OmniDocBench sets a new standard for the fair, diverse, and fine-grained evaluation in document parsing. Dataset and code are available at [https://github.com/opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench).

<sup>†</sup> <sup>†</sup> <sup>†</sup>

## 1 Introduction

![Refer to caption](https://arxiv.org/html/2412.07626v2/x1.png)

Figure 1: Results of End-to-End Text Recognition on OmniDocBench across 9 PDF page types.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x2.png)

Figure 2: Overview of OmniDocBench Data Diversity. The benchmark includes 9 diverse PDF document types. It supports rich annotation types, including layout annotations (e.g., title, table, figure) and recognition annotations (e.g., text spans, equations, tables). Each page is annotated with 6 page-level attributes (e.g., PDF type, layout type), along with fine-grained 3 text attributes (e.g., language) and 6 tables attributes (Items under “ special issues ” are treated as individual binary attributes (yes/no)), enabling detailed and robust evaluation.

As large language models [^1] [^39] [^44] [^28] increasingly rely on high-quality, knowledge-rich data, the importance of accurate document parsing has grown substantially. Document parsing, a core task in computer vision and document intelligence, aims to extract structured, machine-readable content from unstructured documents such as PDFs. This task is particularly critical for ingesting academic papers, technical reports, textbooks, and other rich textual sources into large language models, thereby enhancing their factual accuracy and knowledge grounding [^19] [^42] [^45] [^47] [^52]. Moreover, with the emergence of retrieval-augmented generation (RAG) systems [^22] [^12], which retrieve and generate answers conditionally with external documents, the demand for precise document understanding has further intensified.

To address this challenging task, two main paradigms have emerged: 1) Pipeline-based approaches that decompose the task into layout analysis, OCR, formula/table recognition, and reading order estimation [^42] [^34]; and 2) End-to-end vision-language models (VLMs) that directly output structured representations (e.g., Markdown) [^7] [^46] [^29] [^45] [^3] [^8] [^48]. Although both approaches have demonstrated promising results, conducting a broad comparison of their effectiveness remains challenging due to the absence of a comprehensive and unified evaluation benchmark.

As shown in Table 1, for pipeline-based document parsing systems, dedicated benchmarks [^26] [^54] [^10] have been developed to target specific sub-tasks. For end-to-end evaluation, works like Nougat [^7] and GOT-OCR [^45] provide relatively small validation sets and assess predictions using page-level metrics such as Edit Distance [^21].

However, these benchmarks present several key limitations: 1) Limited document diversity: Existing datasets primarily focus on academic papers, overlooking other real-world document types such as textbooks, exams, financial reports, and newspapers; 2) Inconsistent evaluation metrics: Current benchmarks rely heavily on generic text similarity metrics (e.g., Edit Distance [^21] and BLEU [^33]), which fail to fairly assess the accuracy of formulas and tables in LaTeX or HTML formats that allow for diverse syntactic expressions; and 3) Lack of fine-grained evaluation: Most evaluations report only an overall score, lacking insights into specific weaknesses, such as element-level score (e.g., text vs. formula) or per document-type performance (e.g., magazine or notes).

To address these limitations, we introduce OmniDocBench, a new benchmark designed to provide a rigorous and comprehensive evaluation for document parsing models across both pipeline-based and end-to-end paradigms. In summary, our benchmark introduces the following key contributions:

- High-quality, diverse evaluation set: We include pages from 9 distinct document types, ranging from textbooks to newspapers, annotated using a combination of automated tools, manual verification, and expert review.
- Flexible, multi-dimensional evaluation: We support comprehensive evaluation at three levels—end-to-end, task-specific, and attribute-based. End-to-end evaluation measures the overall quality of full-page parsing results. Task-specific evaluation allows users to assess individual components such as layout detection, OCR, table recognition, or formula parsing. Attribute-based evaluation provides fine-grained analysis across 9 document types, 6 page-level attributes and 9 bbox-level attributes.
- Comprehensive benchmarking of state-of-the-art methods: We systematically evaluate a suite of representative document parsing systems, including both pipeline-based tools and VLMs, providing the most comprehensive comparison and identifying performance bottlenecks across document types and content structures.

## 2 Related Work

<table><tbody><tr><td rowspan="2">Benchmark</td><td rowspan="2">Document Domain</td><td colspan="5">Annotaion Type</td><td colspan="4">Single-Task Eval</td><td colspan="4">End-to-End Eval</td></tr><tr><td>BBox</td><td>Text</td><td>Table</td><td>Formula</td><td>Attributes</td><td>OCR</td><td>DLA</td><td>TR</td><td>MFR</td><td>OCR</td><td>TR</td><td>MFR</td><td>ROD</td></tr><tr><td colspan="15">Single-Task Eval Benchmark</td></tr><tr><td>Robust Reading <sup><a href="#fn:20">20</a></sup></td><td>1</td><td>✔</td><td>✔</td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PubLayNet <sup><a href="#fn:49">49</a></sup>, DocBank <sup><a href="#fn:26">26</a></sup>, DocLayNet <sup><a href="#fn:35">35</a></sup>, M <sup>6</sup> Doc <sup><a href="#fn:9">9</a></sup></td><td>1, 1, 5, 6</td><td>✔</td><td></td><td></td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PubTabNet <sup><a href="#fn:54">54</a></sup>,TableX <sup><a href="#fn:11">11</a></sup></td><td>1, 1</td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TableBank <sup><a href="#fn:25">25</a></sup></td><td>1</td><td>✔</td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Im2Latex-100K <sup><a href="#fn:10">10</a></sup>,UniMER-Test <sup><a href="#fn:40">40</a></sup></td><td>1</td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="15">End-to-end Eval Benchmarks</td></tr><tr><td>Fox <sup><a href="#fn:29">29</a></sup></td><td>2</td><td>✔</td><td>✔</td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td><td>✔</td><td></td><td></td><td></td></tr><tr><td>Nougat <sup><a href="#fn:7">7</a></sup></td><td>1</td><td></td><td>✔</td><td>✔</td><td>✔</td><td></td><td></td><td></td><td></td><td></td><td>✔</td><td>✔</td><td>✔</td><td></td></tr><tr><td>GOT OCR 2.0 <sup><a href="#fn:45">45</a></sup></td><td>2</td><td></td><td>✔</td><td>✔</td><td>✔</td><td></td><td></td><td></td><td></td><td></td><td>✔</td><td>✔</td><td>✔</td><td></td></tr><tr><td>OmniDocBench</td><td>9</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td></tr></tbody></table>

Table 1: A Comparison between OmniDocBench and existing benchmarks. BBox: Bounding boxes. Text: Text in Unicode. Table: Table in LaTeX/HTML/Markdown. Formula: Formula in LaTeX. Attributes: Page- and BBox-Level Attributes. OCR: Optical Character Recognition; DLA: Document Layout Analysis; TR: Table Recognition; MFR: Math Formula Recognition; ROD: Reading Order Detection

### 2.1 Pipeline-based Document Content Extraction

Pipeline-based methods treat the document content extraction task as a collection of single modules, such as document layout detection [^17] [^53] [^36] [^13], optical character recognition [^38] [^23] [^30] [^15] [^43], formula recognition [^51] [^27] [^6] [^40], and table recognition [^16] [^18] [^23]. In this sense, such methods can utilize different expert models to address each specific task. Marker [^34] integrates open-source models to parse documents into structured formats such as Markdown, JSON, and HTML. To get higher accuracy, an optional LLM-enabled version can also be integrated to merge tables across pages, handle inline math, and so on. Similarly, MinerU [^42] first utilizes a layout detection model to segment the document page into different regions, then applies task-specific models for corresponding regions. Finally, it outputs the complete content in Markdown format with a reading order algorithm. By leveraging lightweight models and parallelized operations, pipeline-based methods can achieve efficient parsing speeds.

### 2.2 VLM-based Document Content Extraction

Document understanding and optical character recognition (OCR) are crucial tasks for evaluating the perception capabilities of vision-language models (VLMs). By incorporating extensive OCR corpus into the pretraining stage, VLMs like GPT4o [^2] and Qwen2-VL [^3] have demonstrated comparable performance in document content extraction tasks. Unlike pipeline-based methods, VLMs perform document parsing in an end-to-end manner. Furthermore, without requiring specialized data fine-tuning, these models are able to deal with diverse and even unseen document types for their generalization capabilities.

To integrate the efficiency of lightweight models and the generalizability of VLMs, many works [^7] [^46] [^29] [^45] [^14] [^32] have focus on training specialized end-to-end expert models for document parsing. These VLM-driven models excel at comprehending both visual layouts and textual contents, balancing a trade-off between accuracy and efficiency.

### 2.3 Benchmarks for Document Content Extraction

Document content extraction requires the ability to understand document layouts and recognize various types of content. However, current benchmarks fall short of a comprehensive page-level evaluation, as they focus solely on evaluating the model’s performance on module-level recognition. PubLayNet [^49] and concurrent benchmarks [^26] [^35] [^9] specialize in evaluating a model’s ability to detect document page layouts. OCRBench [^31] proposes five OCR-related tasks with a greater emphasis on evaluating the model’s visual understanding and reasoning capabilities. Only line-level assessments are provided for text recognition and handwritten mathematical expression recognition (HMER). Similarly, single-module benchmarks [^20] [^54] [^26] [^40] disentangle the task into different dimensions and focus narrowly on specific parts. Such paradigm overlooks the importance of structural and semantic information like the reading order and fails to evaluate the model’s overall ability when processing the full-page documents as a whole.

Page-level benchmarks have been proposed alongside some recent VLM-driven expert models [^29] [^7] [^45]. However, the robustness of these benchmarks is compromised by limitations in data size, language, document type, and annotation. For example, Nougat [^7] evaluates models using only printed English documents collected from arXiv while the page-level benchmark introduced by GOT-OCR [^45] consists of only 90 pages of Chinese and English documents in total. Commonly-seen document types like handwritten notes, newspapers, and exam papers are further neglected. Lacking detailed annotations, the benchmarks can only conduct naive evaluation between the full-page results of Ground Truths and predictions without special handling for different output formats and specialized metrics for different content types. The evaluation of the model performance can be severely biased due to limited document domains, unaligned output format and mismatched metrics. Therefore, there is an urgent need for a more finely annotated, diverse, and reasonable page-level document content extraction benchmark.

## 3 OmniDocBench Dataset

Constructing a diverse and comprehensive document parsing benchmark with precise annotations is a significant challenge. As illustrated in Figure 3, we have designed a systematic and professional annotation framework for OmniDocBench, encompassing data acquisition, intelligent pre-annotation, and manual refinement. This ensures that OmniDocBench possesses the following key attributes:

- Page Diversity. We sourced document pages from a variety of origins to ensure a wide range of document types.
- Comprehensive Annotation. We meticulously annotated all elements on the pages, including bounding boxes, specific contents, and various potential attributes.
- Annotation Accuracy. By integrating semi-automated annotation processes, annotator corrections, and expert quality checks, we ensure the reliability of all annotations.

The following sections detail the data acquisition process, the annotation methodology, and a statistical analysis of the final annotated dataset.

### 3.1 Data Acquisition

During the data acquisition phase, we sourced document pages from diverse origins and used clustering algorithms to initially select visually diverse pages, followed by manual annotation of page attributes to finalize the OmniDocBench pages. Specifically, we collected over 200,000 initial PDF documents from Common Crawl, Google, Baidu search engines, and internal data. Subsequently, we extracted visual features from these document pages using ResNet-50 and performed clustering using Faiss <sup>1</sup>, sampling 6,000 visually diverse pages from 10 cluster centers. Finally, annotators provided page-level attribute annotations, including page type, layout type, and language type, and further balanced the selection to 981 samples for the final dataset. The OmniDocBench dataset includes pages from nine distinct types, multiple layout categories, and various attribute annotations, covering a wide range of real-world scenarios.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x3.png)

Figure 3: Overview of the OmniDocBench dataset construction.

### 3.2 Data Annotation

To ensure the comprehensiveness of OmniDocBench’s annotations, we conducted detailed annotations for layout detection and content recognition.

#### 3.2.1 Annotation Types

Layout Detection Annotations: Unlike typical layout detection tasks, OmniDocBench includes four comprehensive types of annotations: (1) Layout Bounding Box Annotations: Positioanl information for 19 distinct region categories such as titles, text paragraphs, tables, and images. (2) Layout Attribute Annotations: Detailed attribute annotations for detected boxes, including 3 text box attribute categories, 6 table attribute categories, 9 bbox-level attribute labels in total. (3) Reading Order Annotations: Annotating the reading sequence of detected boxes. (4) Affiliation Annotations: For images, tables, formulas, and code blocks, we annotate captions and titles to distinguish them from main text. Similarly, for cross-page paragraphs, we annotate affiliation relationships.

Content Recognition Annotations: Based on the content type within each region, we conduct the following three types of annotations: (1) Text Annotations: Pure text annotations for titles, text paragraphs, and other plain text content. (2) Formula Annotations: LaTeX format annotations for inline formulas, display formulas, and subscripts. (3) Table Annotations: Providing both HTML and LaTeX annotations for table data.

#### 3.2.2 Annotation Process

For these annotation tasks on diverse pages, we design a standardized process to ensure quality and efficiency, comprising intelligent automatic annotation, annotator correction, and expert quality inspection.

Automatic Annotation. Manually annotating entire documents is time-consuming and costly. To enhance efficiency, we employ state-of-the-art detection and recognition models for pre-annotation of layout detection and content recognition. Specifically, we use fine-tuned LayoutLMv3 [^17] for layout detection annotations and PaddleOCR [^23], UniMERNet [^40], and GPT-4o [^2] for text, formula, and table annotations, respectively.

Annotator Correction. After the layout detection phase, annotators refine the detection boxes and enhance annotations with reading order and affiliation details. Each character is verified to ensure accuracy in content recognition. For complex annotations of tables and formulas, requiring LaTeX and HTML formats, annotators use tools like Tables Generator <sup>2</sup> and latexlive <sup>3</sup> for verification and correction.

Expert Quality Inspection. Despite thorough annotator corrections, the complexity of formulas and tables may result in residual issues. To address these, we use CDM’s rendering techniques [^41] to identify unrenderable elements. These elements are then reviewed and corrected by three researchers to ensure accuracy in the final annotations.

### 3.3 Dataset Statistics

Page Diversity. OmniDocBench comprises a total of 981 PDF pages across 9 distinct types. Each page is annotated with global attributes, including text language, column layout type, and indicators for blurred scans, watermarks, and colored backgrounds.

Annotation Diversity: OmniDocBench contains over 100,000 annotations for page detection and recognition: (1) More than 20,000 block-level annotations across 15 categories, including over 15,979 text paragraphs, 989 image boxes, 428 table boxes, and so on. All document components except headers, footers, and page notes are labeled with reading order information, totaling over 16,000 annotations. (2) The dataset also includes more than 70,000 span-level annotations across 4 categories, with 4,009 inline formulas and 357 footnote markers represented in LaTeX format, while the remaining annotations are in text format.

Annotation Attribute Diversity: (1) Text Attributes: All block-level annotations, except for tables and images, include text attribute tags. In addition to standard Chinese and English text, there are over 2,000 blocks with complex backgrounds and 493 with rotated text. (2) Table Attributes: In addition to standard Chinese and English tables, there are 142 tables with complex backgrounds, 81 containing formulas, 150 with merged cells, and 7 vertical tables.

## 4 OmniDocBench Evaluation Methodology

To provide a fair and comprehensive evaluation for various models, we proposed an end-to-end evaluation pipeline consisting of several modules, including extraction, matching algorithm, and metric calculation, as shown in Figure 4. It ensures that OmniDocBench automatically performs unified evaluation on document parsing, thereby producing reliable and effective evaluation results.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x4.png)

Figure 4: OmniDocBench Evaluation Pipeline.

### 4.1 Extraction

Preprocessing. The model-generated markdown text should be preprocessed, which includes removing images, eliminating markdown tags at the beginning of the document, and standardizing the number of repeated characters.

Elements Extraction. Extraction is primarily carried out using regular expression matching. To ensure that the extraction of elements does not interfere with each other, it is necessary to follow a specific order. The extraction sequence is as follows: LaTeX tables, HTML tables, display formulas, markdown tables (which are then converted into HTML format), and code blocks.

Pure Text Extraction. After extracting special components, the remaining content is considered pure text. Paragraphs are separated by double line breaks, allowing them to participate in subsequent matching processes, thus aligning with reading order annotation units in the GTs. If no double line break exists, single line breaks are used for paragraph separation. Additionally, previously extracted code blocks are merged into the text category for processing.

Inline Formula Format Converting. We standardized inline formulas within paragraphs to Unicode format. This was necessary because different models produce inconsistent outputs for inline formulas. For formulas originally written in Unicode, it is hard to extract them using regular expressions. Therefore, to ensure a fair comparison, we do not extract inline formulas for separate evaluation. Instead, we include them in their Unicode format alongside the text paragraphs for evaluation.

Reading Order Extraction. Upon completion of the extraction, the start and end positions of the extracted content in the original markdown are recorded for subsequent reading order calculation.

<table><tbody><tr><td rowspan="2">Method Type</td><td rowspan="2">Methods</td><td colspan="2">Text <sup>Edit</sup> <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td colspan="2">Formula <sup>Edit</sup> <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td colspan="2">Formula <sup>CDM</sup> <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td colspan="2">Table <sup>TEDS</sup> <math><semantics><mo>↑</mo> <ci>↑</ci> <annotation>\uparrow</annotation> <annotation>↑</annotation></semantics></math></td><td colspan="2">Table <sup>Edit</sup> <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td colspan="2">Read Order <sup>Edit</sup> <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td><td colspan="2">Overall <sup>Edit</sup> <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation> <annotation>↓</annotation></semantics></math></td></tr><tr><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td><td>EN</td><td>ZH</td></tr><tr><td rowspan="3">Pipeline Tools</td><td>MinerU <sup><a href="#fn:42">42</a></sup></td><td>0.061</td><td>0.215</td><td>0.278</td><td>0.577</td><td>57.3</td><td>42.9</td><td>78.6</td><td>62.1</td><td>0.18</td><td>0.344</td><td>0.079</td><td>0.292</td><td>0.15</td><td>0.357</td></tr><tr><td>Marker <sup><a href="#fn:34">34</a></sup></td><td>0.08</td><td>0.315</td><td>0.53</td><td>0.883</td><td>17.6</td><td>11.7</td><td>67.6</td><td>49.2</td><td>0.619</td><td>0.685</td><td>0.114</td><td>0.34</td><td>0.336</td><td>0.556</td></tr><tr><td>Mathpix <sup>4</sup></td><td>0.105</td><td>0.384</td><td>0.306</td><td>0.454</td><td>62.7</td><td>62.1</td><td>77.0</td><td>67.1</td><td>0.243</td><td>0.32</td><td>0.108</td><td>0.304</td><td>0.191</td><td>0.365</td></tr><tr><td rowspan="2">Expert VLMs</td><td>GOT-OCR <sup><a href="#fn:45">45</a></sup></td><td>0.189</td><td>0.315</td><td>0.360</td><td>0.528</td><td>74.3</td><td>45.3</td><td>53.2</td><td>47.2</td><td>0.459</td><td>0.52</td><td>0.141</td><td>0.28</td><td>0.287</td><td>0.411</td></tr><tr><td>Nougat <sup><a href="#fn:7">7</a></sup></td><td>0.365</td><td>0.998</td><td>0.488</td><td>0.941</td><td>15.1</td><td>16.8</td><td>39.9</td><td>0.0</td><td>0.572</td><td>1.000</td><td>0.382</td><td>0.954</td><td>0.452</td><td>0.973</td></tr><tr><td rowspan="3">General VLMs</td><td>GPT4o <sup><a href="#fn:2">2</a></sup></td><td>0.144</td><td>0.409</td><td>0.425</td><td>0.606</td><td>72.8</td><td>42.8</td><td>72.0</td><td>62.9</td><td>0.234</td><td>0.329</td><td>0.128</td><td>0.251</td><td>0.233</td><td>0.399</td></tr><tr><td>Qwen2-VL-72B <sup><a href="#fn:44">44</a></sup></td><td>0.096</td><td>0.218</td><td>0.404</td><td>0.487</td><td>82.2</td><td>61.2</td><td>76.8</td><td>76.4</td><td>0.387</td><td>0.408</td><td>0.119</td><td>0.193</td><td>0.252</td><td>0.327</td></tr><tr><td>InternVL2-76B <sup><a href="#fn:8">8</a></sup></td><td>0.353</td><td>0.290</td><td>0.543</td><td>0.701</td><td>67.4</td><td>44.1</td><td>63.0</td><td>60.2</td><td>0.547</td><td>0.555</td><td>0.317</td><td>0.228</td><td>0.44</td><td>0.443</td></tr></tbody></table>

Table 2: Comprehensive evaluation of document parsing algorithms on OmniDocBench: performance metrics for text, formula, table, and reading order extraction, with overall scores derived from ground truth comparisons.

<table><tbody><tr><td>Model Type</td><td>Models</td><td>Book</td><td>Slides</td><td>Financial Report</td><td>Textbook</td><td>Exam Paper</td><td>Magazine</td><td>Academic Papers</td><td>Notes</td><td>Newspaper</td><td>Overall</td></tr><tr><td rowspan="3">Pipeline Tools</td><td>MinerU <sup><a href="#fn:42">42</a></sup></td><td>0.055</td><td>0.124</td><td>0.033</td><td>0.102</td><td>0.159</td><td>0.072</td><td>0.025</td><td>0.984</td><td>0.171</td><td>0.206</td></tr><tr><td>Marker <sup><a href="#fn:34">34</a></sup></td><td>0.074</td><td>0.34</td><td>0.089</td><td>0.319</td><td>0.452</td><td>0.153</td><td>0.059</td><td>0.651</td><td>0.192</td><td>0.274</td></tr><tr><td>Mathpix <sup>4</sup></td><td>0.131</td><td>0.22</td><td>0.202</td><td>0.216</td><td>0.278</td><td>0.147</td><td>0.091</td><td>0.634</td><td>0.69</td><td>0.3</td></tr><tr><td rowspan="2">Expert VLMs</td><td>GOT-OCR <sup><a href="#fn:45">45</a></sup></td><td>0.111</td><td>0.222</td><td>0.067</td><td>0.132</td><td>0.204</td><td>0.198</td><td>0.179</td><td>0.388</td><td>0.771</td><td>0.267</td></tr><tr><td>Nougat <sup><a href="#fn:7">7</a></sup></td><td>0.734</td><td>0.958</td><td>1.000</td><td>0.820</td><td>0.930</td><td>0.83</td><td>0.214</td><td>0.991</td><td>0.871</td><td>0.806</td></tr><tr><td rowspan="3">General VLMs</td><td>GPT4o <sup><a href="#fn:2">2</a></sup></td><td>0.157</td><td>0.163</td><td>0.348</td><td>0.187</td><td>0.281</td><td>0.173</td><td>0.146</td><td>0.607</td><td>0.751</td><td>0.316</td></tr><tr><td>Qwen2-VL-72B <sup><a href="#fn:44">44</a></sup></td><td>0.096</td><td>0.061</td><td>0.047</td><td>0.149</td><td>0.195</td><td>0.071</td><td>0.085</td><td>0.168</td><td>0.676</td><td>0.179</td></tr><tr><td>InternVL2-76B <sup><a href="#fn:8">8</a></sup></td><td>0.216</td><td>0.098</td><td>0.162</td><td>0.184</td><td>0.247</td><td>0.150</td><td>0.419</td><td>0.226</td><td>0.903</td><td>0.3</td></tr></tbody></table>

Table 3: End-to-end text recognition performance on OmniDocBench: evaluation using edit distance across 9 PDF page types.

| Models | Fuzzy | Water | Color | None |
| --- | --- | --- | --- | --- |
| MinerU [^42] | 0.15/0.048 | 0.151/0.031 | 0.107/0.052 | 0.079/0.035 |
| Marker [^34] | 0.333/0.092 | 0.484/0.126 | 0.319/0.127 | 0.062/0.125 |
| Mathpix <sup>4</sup> | 0.294/0.064 | 0.290/0.059 | 0.216/0.09 | 0.135/0.043 |
| GOT-OCR [^45] | 0.175/0.05 | 0.190/0.056 | 0.186/0.097 | 0.177/0.081 |
| Nougat [^7] | 0.934/0.051 | 0.915/0.071 | 0.873/0.096 | 0.615/0.208 |
| GPT4o [^2] | 0.263/0.078 | 0.195/0.057 | 0.184/0.078 | 0.186/0.072 |
| Qwen2-VL-72B [^44] | 0.082 /0.01 | 0.172/ 0.078 | 0.104/0.05 | 0.084/0.042 |
| InternVL2-76B [^8] | 0.120/0.013 | 0.197/0.042 | 0.155/0.059 | 0.261/0.082 |

Table 4: End-to-end text recognition on OmniDocBench: evaluation under various page attributes using the edit distance metric. The value is Mean/Variance of scores in the attribute group. Columns represent: Fuzzy (Fuzzy scan), Water (Watermark), Color (Colorful background). None (No special issue)

| Models | Single | Double | Three | Complex |
| --- | --- | --- | --- | --- |
| MinerU [^42] | 0.311/0.187 | 0.101/0.013 | 0.117/0.046 | 0.385/0.057 |
| Marker [^34] | 0.299/0.143 | 0.299/0.299 | 0.149/0.063 | 0.363/0.086 |
| Mathpix <sup>4</sup> | 0.207/0.123 | 0.188/0.07 | 0.225/0.029 | 0.452/0.177 |
| GOT-OCR [^45] | 0.163/0.106 | 0.145/0.059 | 0.257/0.072 | 0.468/0.185 |
| Nougat [^7] | 0.852/0.084 | 0.601/0.224 | 0.662/0.093 | 0.873/0.09 |
| GPT4o [^2] | 0.109/0.112 | 0.204/0.076 | 0.254/0.046 | 0.426/0.188 |
| Qwen2-VL-72B [^44] | 0.066/0.048 | 0.145/0.049 | 0.204/0.055 | 0.394/0.203 |
| InternVL2-76B [^8] | 0.082/0.052 | 0.312/0.069 | 0.682/0.098 | 0.444/0.174 |

Table 5: End-to-end reading order evaluation on OmniDocBench: results across different column layout types using Normalized Edit Distance. The value is Mean/Variance of scores in the attribute group.

### 4.2 Matching Algorithm

Adjacency Search Match. To avoid the impact of paragraph splitting on the final results, we proposed Adjacency Search Match, that merges and splits paragraphs in both GTs and Preds to achieve the best possible match. The specific strategy involves: i) Calculate a metrix of Normalized Edit Distance between GTs and Preds. The Pred and GT pairs whose similarity exceeds a specific threshold are considered as successful match. ii) For the rest, we apply fuzzy matching to determine whether one string is a subset of another string. If so, we further apply the merging algorithm which would try to merge adjacent paragraph. This process would continue to merge more paragraph until the Normalized Edit Distance starts to decrease. After this process, the best match will be found for GTs and Preds.

Ignore Handling. We implement an ignore logic for certain components in PDF page content, meaning they participate in matching but are excluded from metric calculations. This is mainly because of inconsistent output standards among models, which should not affect the validation results. For fairness, we ignore: (1) Headers, footers, page numbers, and page footnotes, which are handled inconsistently by different models. (2) Captions for figures, tables, and footnotes often have uncertain placements, thus complicating the reading order. Additionally, some models embed table captions in HTML or LaTeX tables, while others treat them as plain text.

| Model | Backbone | Params | Book | Slides | Research Report | Textbook | Exam Paper | Magazine | Academic Literature | Notes | Newspaper | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiT-L [^24] | ViT-L | 361.6M | 43.44 | 13.72 | 45.85 | 15.45 | 3.40 | 29.23 | 66.13 | 0.21 | 23.65 | 26.90 |
| LayoutLMv3 [^17] | RoBERTa-B | 138.4M | 42.12 | 13.63 | 43.22 | 21.00 | 5.48 | 31.81 | 64.66 | 0.80 | 30.84 | 28.84 |
| DocLayout-YOLO [^53] | v10m | 19.6M | 43.71 | 48.71 | 72.83 | 42.67 | 35.40 | 51.44 | 64.64 | 9.54 | 57.54 | 47.38 |
| SwinDocSegmenter [^4] | Swin-L | 223M | 42.91 | 28.20 | 47.29 | 32.44 | 20.81 | 52.35 | 48.54 | 12.38 | 38.06 | 35.89 |
| GraphKD [^5] | R101 | 44.5M | 39.03 | 16.18 | 39.92 | 22.82 | 14.31 | 37.61 | 44.43 | 5.71 | 23.86 | 27.10 |
| DOCX-Chain [^50] | \- | \- | 30.86 | 11.71 | 39.62 | 19.23 | 10.67 | 23.00 | 41.60 | 1.80 | 16.96 | 21.27 |

Table 6: Component-level layout detection evaluation on OmniDocBench layout subset: mAP results by PDF page type.

<table><tbody><tr><td rowspan="2">Model Type</td><td rowspan="2">Model</td><td colspan="3">Language</td><td colspan="4">Table Frame Type</td><td colspan="4">Special Situation</td><td rowspan="2">Overall</td></tr><tr><td>EN</td><td>ZH</td><td>Mixed</td><td>Full</td><td>Omission</td><td>Three</td><td>Zero</td><td>Merge Cell(+/-)</td><td>Formula(+/-)</td><td>Colorful (+/-)</td><td>Rotate(+/-)</td></tr><tr><td rowspan="2">OCR-based Models</td><td>PaddleOCR <sup><a href="#fn:23">23</a></sup></td><td>76.8</td><td>71.8</td><td>80.1</td><td>67.9</td><td>74.3</td><td>81.1</td><td>74.5</td><td>70.6/75.2</td><td>71.3/74.1</td><td>72.7/74.0</td><td>23.3/74.6</td><td>73.6</td></tr><tr><td>RapidTable <sup><a href="#fn:37">37</a></sup></td><td>80.0</td><td>83.2</td><td>91.2</td><td>83.0</td><td>79.7</td><td>83.4</td><td>78.4</td><td>77.1/85.4</td><td>76.7/83.9</td><td>77.6/84.9</td><td>25.2/83.7</td><td>82.5</td></tr><tr><td rowspan="2">Expert VLMs</td><td>StructEqTable <sup><a href="#fn:55">55</a></sup></td><td>72.8</td><td>75.9</td><td>83.4</td><td>72.9</td><td>76.2</td><td>76.9</td><td>88.0</td><td>64.5/81.0</td><td>69.2/76.6</td><td>72.8/76.4</td><td>30.5/76.2</td><td>75.8</td></tr><tr><td>GOT-OCR <sup><a href="#fn:45">45</a></sup></td><td>72.2</td><td>75.5</td><td>85.4</td><td>73.1</td><td>72.7</td><td>78.2</td><td>75.7</td><td>65.0/80.2</td><td>64.3/77.3</td><td>70.8/76.9</td><td>8.5/76.3</td><td>74.9</td></tr><tr><td rowspan="2">General VLMs</td><td>Qwen2-VL-7B <sup><a href="#fn:44">44</a></sup></td><td>70.2</td><td>70.7</td><td>82.4</td><td>70.2</td><td>62.8</td><td>74.5</td><td>80.3</td><td>60.8/76.5</td><td>63.8/72.6</td><td>71.4/70.8</td><td>20.0/72.1</td><td>71.0</td></tr><tr><td>InternVL2-8B <sup><a href="#fn:8">8</a></sup></td><td>70.9</td><td>71.5</td><td>77.4</td><td>69.5</td><td>69.2</td><td>74.8</td><td>75.8</td><td>58.7/78.4</td><td>62.4/73.6</td><td>68.2/73.1</td><td>20.4/72.6</td><td>71.5</td></tr></tbody></table>

Table 7: Component-level Table Recognition evaluation on OmniDocBench table subset. (+/-) means with/without special situation.

<table><tbody><tr><td rowspan="2">Model Type</td><td rowspan="2">Model</td><td colspan="3">Language</td><td colspan="3">Text background</td><td colspan="4">Text Rotate</td></tr><tr><td>EN</td><td>ZH</td><td>Mixed</td><td>White</td><td>Single</td><td>Multi</td><td>Normal</td><td>Rotate90</td><td>Rotate270</td><td>Horizontal</td></tr><tr><td rowspan="5">Expert Vision Models</td><td>PaddleOCR <sup><a href="#fn:23">23</a></sup></td><td>0.071</td><td>0.055</td><td>0.118</td><td>0.060</td><td>0.038</td><td>0.085</td><td>0.060</td><td>0.015</td><td>0.285</td><td>0.021</td></tr><tr><td>Tesseract OCR <sup>5</sup></td><td>0.179</td><td>0.553</td><td>0.553</td><td>0.453</td><td>0.463</td><td>0.394</td><td>0.448</td><td>0.369</td><td>0.979</td><td>0.982</td></tr><tr><td>Surya <sup>6</sup></td><td>0.057</td><td>0.123</td><td>0.164</td><td>0.093</td><td>0.186</td><td>0.235</td><td>0.104</td><td>0.634</td><td>0.767</td><td>0.255</td></tr><tr><td>GOT-OCR <sup><a href="#fn:45">45</a></sup></td><td>0.041</td><td>0.112</td><td>0.135</td><td>0.092</td><td>0.052</td><td>0.155</td><td>0.091</td><td>0.562</td><td>0.966</td><td>0.097</td></tr><tr><td>Mathpix <sup>4</sup></td><td>0.033</td><td>0.240</td><td>0.261</td><td>0.185</td><td>0.121</td><td>0.166</td><td>0.180</td><td>0.038</td><td>0.185</td><td>0.638</td></tr><tr><td rowspan="3">Vision Language Models</td><td>Qwen2-VL-72B <sup><a href="#fn:44">44</a></sup></td><td>0.072</td><td>0.274</td><td>0.286</td><td>0.234</td><td>0.155</td><td>0.148</td><td>0.223</td><td>0.273</td><td>0.721</td><td>0.067</td></tr><tr><td>InternVL2-76B <sup><a href="#fn:8">8</a></sup></td><td>0.074</td><td>0.155</td><td>0.242</td><td>0.113</td><td>0.352</td><td>0.269</td><td>0.132</td><td>0.610</td><td>0.907</td><td>0.595</td></tr><tr><td>GPT4o <sup><a href="#fn:2">2</a></sup></td><td>0.020</td><td>0.224</td><td>0.125</td><td>0.167</td><td>0.140</td><td>0.220</td><td>0.168</td><td>0.115</td><td>0.718</td><td>0.132</td></tr></tbody></table>

Table 8: Component-level evaluation on OmniDocBench OCR subset: results grouped by text attributes using the edit distance metric.

| Models | CDM | ExpRate@CDM | BLEU | Norm Edit |
| --- | --- | --- | --- | --- |
| GOT-OCR [^45] | 74.1 | 28.0 | 55.07 | 0.290 |
| Mathpix <sup>4</sup> | 86.6 | 2.8 | 66.56 | 0.322 |
| Pix2Tex <sup>7</sup> | 73.9 | 39.5 | 46.00 | 0.337 |
| UniMERNet-B [^40] | 85.0 | 60.2 | 60.84 | 0.238 |
| GPT4o [^2] | 86.8 | 65.5 | 45.17 | 0.282 |
| InternVL2-76B [^8] | 67.4 | 54.5 | 47.63 | 0.308 |
| Qwen2-VL-72B [^44] | 83.8 | 55.4 | 53.71 | 0.285 |

Table 9: Component-level formula recognition evaluation on OmniDocBench formula subset.

### 4.3 Metric Calculation

Pure Text. We calculate Normalized Edit Distance [^21], averaging these metrics at the sample level to obtain the final scores.

Tables. All tables are converted to HTML format before calculating the Tree-Edit-Distance-based Similarity (TEDS) [^54] metric and Normalized Edit Distance.

Formulas. Formulas are currently evaluated using the Character Detection Matching (CDM) metric [^41], Normalized Edit Distance, and BLEU [^33].

Reading Order. Reading order is evaluated using the Normalized Edit Distance as metric. It only involves text components, with tables, images, and ignored components excluded from the final reading order calculation.

## 5 Benchmarks

Based on the distinct characteristics of these algorithms, we categorize document content extraction methods into three main classes:

- Pipeline Tools: These methods integrate layout detection and various content recognition tasks (such as OCR, table recognition, and formula recognition) into a document parsing pipeline for content extraction. Prominent examples include MinerU [^42] (v0.9.3), Marker [^34] (v1.2.3), and Mathpix <sup>4</sup>.
- Expert VLMs: These are large multimodal models specifically trained for document parsing tasks. Representative models include GOT-OCR2.0 [^45] and Nougat [^7].
- General VLMs: These are general-purpose large multimodal models inherently capable of document parsing. Leading models in this category include GPT-4o [^2], Qwen2-VL-72B [^44], and InternVL2-76B [^8].

### 5.1 End-to-End Evaluation Results

Overall Evaluation Results. As illustrated in Table 2, pipeline tools such as MinerU and Mathpix, demonstrate superior performance across sub-tasks like text recognition, formula recognition, and table recognition. Moreover, the general Vision Language Models (VLMs), Qwen2-VL, and GPT4o, also exhibit competitive performance. Almost all algorithms score higher on English than on Chinese pages.

Performance Across Diverse Page Types. To gain deeper insights into model performance on diverse document types, we evaluated text recognition tasks across different page types. Intriguingly, as shown in Table 3, pipeline tools perform well for commonly used data, such as academic papers and financial reports. Meanwhile, for more specialized data, such as slides and handwritten notes, general VLMs demonstrate stronger generalization. Notably, most VLMs fail to recognize when dealing with the Newspapers, while pipeline tools achieve significantly better performance.

Performance on Pages with Visual Degradations. In Table 4, we further analyze performance on pages containing common document-specific challenges, including fuzzy scans, watermarks, and colorful backgrounds. VLMs like InternVL2 and Qwen2-VL exhibit higher robustness in these scenarios despite visual noise. Among pipeline tools, MinerU remains competitive due to its strong layout segmentation and preprocessing capabilities.

Performance on Different Layout Types. Page layout is a critical factor in document understanding, especially for tasks involving reading order. OmniDocBench annotates layout attributes such as single-column, multi-column, and complex custom formats. Across all models, we observe a clear drop in accuracy on multi-column and complex layouts. MinerU shows the most consistent reading order prediction, though its performance dips on handwritten single-column pages due to recognition noise.

Discussion on End-to-End Results. 1) While general VLMs often lag behind specialized pipelines and expert models on standard documents (e.g., academic papers), they generalize better to unconventional formats (e.g., notes) and perform more robustly under degraded conditions (e.g., fuzzy scans). This is largely due to their broader training data, enabling better handling of long-tail scenarios compared to models trained on narrow domains. 2) VLMs, however, struggle with high-density documents like newspapers due to limitations in input resolution and token length. In contrast, pipeline tools leverage layout-based segmentation to process components individually, maintaining accuracy in complex layouts. Enhancing VLMs with layout-aware designs and domain-specific fine-tuning offers a promising path forward. OmniDocBench facilitates this by providing detailed annotations for layout, text, formulas, and tables, enabling comprehensive benchmarking and modular tool development for diverse document parsing tasks.

### 5.2 Single Task Evaluation Results

Layout Detection Results. Layout detection is the first step in document parsing using pipeline tools. A robust layout detection algorithm should perform well across a variety of document types. Table 6 presents an evaluation of leading layout detection models. The DocLayout-YOLO method, which is pre-trained on diverse synthetic document data, significantly outperforms other approaches. This superiority is a key factor in MinerU’s integration of DocLayout-YOLO, contributing to its outstanding overall performance. Other methods perform well on books and academic literature but struggle with more diverse formats due to limited training data.

Table Recognition Results. In Table 7, We evaluate table recognition models across three dimensions on our OmniDocBench table subset: language diversity, table frame types, and special situations. Among all models, OCR-based models demonstrate superior overall performance, with RapidTable achieving the highest scores in language diversity and maintaining stable performance across different frame types. Expert VLMs show competitive results in specific scenarios, with StructEqTable [^55] excelling in no-frame tables and showing better rotation robustness. General VLMs (Qwen2-VL-7B and InternVL2-8B) exhibit relatively lower but consistent performance, suggesting that while general-purpose VLMs have made progress in table understanding, they still lag behind specialized solutions.

Text Recognition Results. Table 8 compares OCR tools across languages, backgrounds, and rotations using Edit Distance. PaddleOCR outperforms all competitors, followed by GOT-OCR and Mathpix. General VLMs struggle to handle text rotation or mixed-language scenarios.

Formula Recognition Results. Table 9 presents results on formula parsing, using CDM, BLEU, and normalized Edit Distance. GPT-4o, Mathpix, and UniMERNet achieve results of 86.8%, 86.6%, and 85.0%, respectively. Notably, GPT-4o excels with a recall rate of 65.5% under strict conditions requiring perfect character accuracy. Although Mathpix shows high character-level precision, it occasionally omits punctuation, such as commas, leading to a lower overall correctness rate. Nonetheless, all three models are strong candidates for formula recognition tasks.

## 6 Conclusion

This paper addresses the lack of diverse and realistic benchmarks in document parsing research by introducing OmniDocBench, a dataset featuring a variety of page types with comprehensive annotations, along with a flexible and reliable evaluation framework. OmniDocBench enables systematic and fair assessments of document parsing methods, providing crucial insights for advancing the field. Its task-specific and attribute-level evaluations facilitate targeted model optimization, promoting more robust and effective parsing solutions.

## References

Supplementary Material

## I More End-to-End Evaluation Results

Table S1 presents the evaluation results of End2End Tables grouped by Table Attributes. As it shows, most of the models perform better in English Tables rather than Chinese ones. Most models perform relatively poorly with Full Frame and No Frame tables. The accuracy of most models is affected by special conditions. Merged cells and formulas mainly test the breadth of data the model can recognize, while colored backgrounds and table rotation test their robustness. The results show that table rotation significantly impacts the accuracy of all models. Pipeline Tools’ performance would not be affected by more challenging tables (e.g., merge cell), but colored backgrounds can affect recognition accuracy. Several Vision Language Models (VLMs) tend to perform worse on tables with merged cells, but colored backgrounds do not significantly impact table recognition accuracy.

Table S2 shows the evaluation results of End2End Text blocks grouped by Text Attributes. Almost all models have lower recognition accuracy in Chinese compared to English. Some models, such as MinerU and Marker, experience a further decrease in accuracy when recognizing mixed Chinese and English content. The main reason is that minerU’s text recognition module is PaddleOCR model. According to the performance of the PaddleOCR model in text recognition module, its accuracy will decline in the case of mixed language. Moreover, complex background colors significantly affect the recognition accuracy of pipeline tools, but it has only little impact on accuracy for VLMs.

<table><tbody><tr><td rowspan="2">Model Type</td><td rowspan="2">Model</td><td colspan="3">Language</td><td colspan="4">Table Frame Type</td><td colspan="4">Special Situation</td></tr><tr><td>EN</td><td>ZH</td><td>Mixed</td><td>Full</td><td>Omission</td><td>Three</td><td>Zero</td><td>Merge Cell(+/-)</td><td>Formula(+/-)</td><td>Colorful (+/-)</td><td>Rotate(+/-)</td></tr><tr><td rowspan="3">Pipeline Tools</td><td>MinerU</td><td>75.1</td><td>59.3</td><td>79.1</td><td>59.4</td><td>71.6</td><td>69.7</td><td>60.0</td><td>63.6/65.3</td><td>66.0/64.4</td><td>59.2/67.5</td><td>3.0/65.8</td></tr><tr><td>Marker</td><td>64.9</td><td>47.3</td><td>49.8</td><td>44.5</td><td>61.8</td><td>59.0</td><td>63.6</td><td>52.6/52.7</td><td>53.2/52.5</td><td>48.0/54.9</td><td>35.5/52.9</td></tr><tr><td>Mathpix</td><td>75.4</td><td>63.2</td><td>71.3</td><td>67.4</td><td>77.3</td><td>66.3</td><td>25.5</td><td>70.3/65.4</td><td>68.7/66.7</td><td>59.7/70.8</td><td>19.2/67.9</td></tr><tr><td rowspan="2">Expert Vision Models</td><td>GOT-OCR</td><td>51.7</td><td>46.2</td><td>49.0</td><td>45.5</td><td>48.3</td><td>51.3</td><td>46.2</td><td>46.0/48.9</td><td>45.7/48.4</td><td>39.8/51.9</td><td>0.0/48.7</td></tr><tr><td>Nougat</td><td>36.2</td><td>0.3</td><td>0.0</td><td>6.1</td><td>3.5</td><td>22.1</td><td>0.0</td><td>15.0/8.9</td><td>21/8.7</td><td>2.6/15.2</td><td>0.0/11.2</td></tr><tr><td rowspan="3">Vision Language Models</td><td>GPT4o</td><td>71.1</td><td>58.0</td><td>57.3</td><td>62.5</td><td>68.7</td><td>61.3</td><td>31.2</td><td>56.8/64.7</td><td>60.8/62.2</td><td>61.4/62.2</td><td>14.2/62.7</td></tr><tr><td>Qwen2-VL-72B</td><td>73.2</td><td>75.1</td><td>76.1</td><td>72.0</td><td>79.0</td><td>77.5</td><td>63.2</td><td>67.9/78.1</td><td>71.6/75.3</td><td>77.9/72.9</td><td>42.7/75.1</td></tr><tr><td>InterVL2-76B</td><td>60.9</td><td>58.5</td><td>65.4</td><td>58.8</td><td>65.3</td><td>58.3</td><td>55.6</td><td>49.0/65.1</td><td>53.3/60.9</td><td>58.8/59.8</td><td>6.9/60.3</td></tr></tbody></table>

Table S1: End-to-End Table TEDS Result grouped by Table Attributes

<table><tbody><tr><td rowspan="2">Model Type</td><td rowspan="2">Model</td><td colspan="3">Language</td><td colspan="3">Text background</td></tr><tr><td>EN</td><td>ZH</td><td>Mixed</td><td>White</td><td>Single</td><td>Multi</td></tr><tr><td rowspan="3">Pipeline Tools</td><td>MinerU</td><td>0.124</td><td>0.234</td><td>0.742</td><td>0.188</td><td>0.15</td><td>0.514</td></tr><tr><td>Marker</td><td>0.163</td><td>0.379</td><td>0.747</td><td>0.303</td><td>0.396</td><td>0.594</td></tr><tr><td>Mathpix</td><td>0.175</td><td>0.793</td><td>0.538</td><td>0.698</td><td>0.587</td><td>0.583</td></tr><tr><td rowspan="2">Expert Vision Models</td><td>GOT-OCR</td><td>0.251</td><td>0.763</td><td>0.266</td><td>0.669</td><td>0.595</td><td>0.440</td></tr><tr><td>Nougat</td><td>0.587</td><td>0.991</td><td>0.983</td><td>0.874</td><td>0.935</td><td>0.972</td></tr><tr><td rowspan="3">Vision Language Models</td><td>GPT4o</td><td>0.170</td><td>0.647</td><td>0.322</td><td>0.536</td><td>0.423</td><td>0.406</td></tr><tr><td>Qwen2-VL-72B</td><td>0.128</td><td>0.582</td><td>0.209</td><td>0.494</td><td>0.388</td><td>0.217</td></tr><tr><td>InternVL2-76B</td><td>0.418</td><td>0.606</td><td>0.251</td><td>0.589</td><td>0.366</td><td>0.221</td></tr></tbody></table>

Table S2: End-to-End Text Normalized Edit Distance results grouped by Text Attributes. “Mixed” represents a mixture of Chinese and English, “Single” and “Multi” represent single color and multi color.

| Category | Attribute Name | Count |
| --- | --- | --- |
| PDF Type | Book | 104 |
|  | PPT2PDF | 133 |
|  | Research Report | 81 |
|  | Colorful Textbook | 96 |
|  | Exam Paper | 114 |
|  | Magazine | 97 |
|  | Academic Literature | 129 |
|  | Notes | 116 |
|  | Newspaper | 111 |
| Layout Type | Single Column | 477 |
|  | Double Column | 126 |
|  | Three Column | 45 |
|  | One&More Mixed | 120 |
|  | Complex Layout | 213 |
| Language | English | 290 |
|  | Simplified Chinese | 612 |
|  | Mixed | 79 |
| Special Issues | Fuzzy Scan | 28 |
|  | Watermark | 65 |
|  | Colorful Background | 246 |

Table S3: The Page Attributes Statistics of OmniDocBench.

| Attribute Category | Category Name | Count |
| --- | --- | --- |
| Language | English | 5857 |
|  | Simplified Chinese | 16073 |
|  | EN&CH Mixed | 1080 |
| Text Background | White | 19465 |
|  | Single-Colored | 1116 |
|  | Multi-Colored | 2429 |
| Text Rotate | Normal | 22865 |
|  | Rotate90 | 14 |
|  | Rotate270 | 58 |
|  | Horizontal | 421 |

Table S4: Text Attributes Statistics of OmniDocBench.

| Attribute Category | Category Name | Count |
| --- | --- | --- |
| Language | English | 128 |
|  | Simplified Chinese | 285 |
|  | EN&CH Mixed | 15 |
| Table Frame Type | Full Frame | 205 |
|  | Omission Line | 62 |
|  | Three Line | 147 |
|  | No Frame | 14 |
| Special Issues | Merge Cell | 150 |
|  | Colorful Background | 142 |
|  | Contain Formula | 81 |
|  | Rotate | 7 |

Table S5: Table Attributes Statistics of OmniDocBench.

| No. | Category Name | Explaination | Total |
| --- | --- | --- | --- |
| 1 | Title | Include main titles, chapter titles, etc. | 2972 |
| 2 | Text Block | Text paragraphs, which are usually separated by double line breaks in Markdown. | 15979 |
| 3 | Figure | Including images, visual charts, etc. | 989 |
| 4 | Figure Caption | Typically starts with ’Figure’ followed by a number, or just descriptive language below the figure. | 651 |
| 5 | Figure Footnotes | Descriptive language, apart from the figure caption, usually starts with an asterisk (\*). | 133 |
| 6 | Table | Content organized in table form usually includes borders or a clear table structure. | 428 |
| 7 | Table Caption | Typically starts with ’Table’ followed by a number, or just descriptive language above the Table. | 299 |
| 8 | Table Footnotes | Descriptive language, apart from the table caption, usually starts with an asterisk (\*). | 132 |
| 9 | Header | Information located at the top of a PDF page or in the sidebar, separate from the main content, typically includes chapter names and other details. | 1271 |
| 10 | Footer | Information located at the bottom of a PDF page, separate from the main content, typically includes the publisher’s name and other details. | 541 |
| 11 | Page Number | It is usually represented by numbers, which may be located at the top, in the sidebar, or at the bottom of the page. | 669 |
| 12 | Page Footnote | It provides further explanation of the footnotes marked within the page content. For example, information about the authors’ affiliations. | 92 |
| 13 | Code Block | In Markdown, a code block is typically defined using triple backticks (“‘). | 13 |
| 14 | Code Block Caption | Descriptive language above the Code Block. | / |
| 15 | Reference | Typically found only in academic literature. | 260 |
| 16 | Text Span | Span-Level text box, which is the plain text content can be directly written in Markdown format. | 73143 |
| 17 | Equation Inline | Formulas that need to be represented using LaTeX format and embedded within the text. | 4009 |
| 18 | Equation Ignore | Some formulas that can be displayed correctly without using LaTeX formatting, such as 15 kg. | 3685 |
| 19 | Footnote Mark | Typically embedded within the text as superscripts or subscripts, and their numbering usually corresponds to page footnotes. | 357 |
| 20 | Other Abandoned Categories | (Masked) Some uncategorizable, irrelevant page information, such as small icons, etc. | 538 |
| 21 | Masked Text Block | (Masked) Some difficult-to-recognize information that disrupts text flow, such as pinyin annotations above Chinese characters. | 34 |
| 22 | Organic Chemical Formula | (Masked) Organic chemistry formulas, which are difficult to write using Markdown and are easily recognized as Figures. | 24 |

Table S6: Annotation Explanations and Statistics.

## II Dataset Statistics and Visualization

OmniDocBench contains 981 pages, including 9 types of PDF pages, 4 types of layouts, 3 types of languages, and 3 special issues in visual degradations (e.g., watermarks). Table S3 and Figure S1 show the number of pages with each page attribute. Figures S5, S6, S7 and S8 are examples of PDF pages with different PDF types, Layout Types, and Special Issues.

Table S6 and Figure S2 show all annotation categories included in OmniDocBench. All of them are annotated by bounding boxes. There are 15 types of block-level annotations and 4 types of span-level annotations, with span-level annotations nested within the block-level ones. In addition, there are 3 types of annotations marked as page interference information (No.20-22), whose bounding boxes are used to mask the specific regions of the PDF pages to avoid affecting the evaluation results. The recognition annotations are also provided for each annotation category except for Figures. Formulas is written in LaTeX format and Table is annotated in both HTML and LaTeX formats. Others are annotated in plain text.

Furthermore, the Text Attributes are also annotated for each block-level category that contains text. There are 3 types of Text Attributes that might influent OCR accuracy: Language, Text Background Color, and Text Rotation. Table S5 shows the statistics of annotations with specific text attributes. There are 23,010 block-level annotations are labeled with text attributes.

Tables are also annotated with Table Attributes. There are 6 types of Table Attributes that might influent the Table Recognition accuracy: Language, Table Frame Type, Merge Cell, Colorful Background, Contain Formula, and Rotation. Table S5 shows the numbers of annotations with specific table attributes. Figures S9 and S10 are the examples of Tables with different Frames and Special Issues.

## III Discussion on Model Predictions

Conclusion Combining scattered results from tasks and sub-attributes, it can be concluded that pipeline tools and expert models have better performance on common data like academic papers and challenging cases such as tables with merged cells compared to VLMs. However, VLMs demonstrate stronger generalization on uncommon PDF types like slides and exam papers, and they show greater robustness in special page situations, such as fuzzy scans. The low accuracy of VLMs is mainly due to:1) Missing Content in dense pages(Figure S11); 2) Hallucinations in hard-to-recognize pages(Figure S30). The low accuracy of Pipeline tools mainly due to: 1) Lower robustness in special page situations, e.g., watermark(Figure S21); 2) Weak generalization on uncommon PDF types, e.g., handwriting notes(Figure S16).

Figures S12, S13, S14, S15, S16, S11, S17, S18 and S19 show the examples of Good model outputs and Bad model outputs of Document Parsing among different PDF types. As it shown, different models exhibit varying performance across different PDF types. For example, MinerU detects all handwritten notes as figures, resulting in very low recognition accuracy in Notes. Marker and InternVL2 experience missed detections, leading to lower scores. InternVL2 and Qwen2-VL, in specific PDF types (such as slides or financial reports), tend to merge multi-column text.

Figures S22, S20 and S21 show the examples of Good model outputs and Bad model outputs under special issues of the PDF pages. It shows that Marker tends to generate typos when the PDF pages are fuzzy scanned or with watermarks, while GOT-OCR fails to recognize content on pages with colored backgrounds. MinerU performs well under special situations, while Mathpix occasionally generates typos.

Figures S23, S24, S25 and S26 show examples of Good model outputs and Bad model outputs for PDF pages with different layouts. MinerU has a low reading order score for single-column layouts primarily because most notes are single-column, and MinerU performs poorly in recognizing Notes, leading to a low reading order score accordingly. InternVL2 scores high in Single-Column layouts but scores poorly on Double-Column and Three-Column layouts. It is mainly due to frequent missed content recognition and errors in reading order judgment in multi-column layouts pages. MinerU’s reading order and recognition accuracy decrease with complex layouts, primarily because it incorrectly merges multiple columns during recognition.

Figures S29 and S30 show the model’s recognition ability under special issues of text. In text recognition with complex background colors, Marker may produce errors or miss content, whereas Qwen2-VL still performs well. Most models fail to recognize text when it is rotated 270 degrees. Some vision language models generate hallucinated information based on the content they can recognize.

Figures S31, S32, S33 and S34 show the examples of good and bad model results for tables with different attributes. For three-line tables, RapidTable demonstrates a good performance with accurate structure recognition, while PaddleOCR shows limitations by missing the last column in its outputs. Interestingly, in tables without frames, PaddleOCR performs well with accurate table predictions, while Qwen2-VL-7B exhibits errors in the last two columns. This indicates that the presence or absence of table frames can significantly impact different models’ performance in different ways. Rotated tables prove to be particularly challenging, with most models, including GOT-OCR, failing to recognize the table structure. However, StructEqTable shows promising results by correctly identifying most of the table content, though with a few detail errors. For tables containing formula, Qwen2-VL-7B shows more accurate table structure recognition compared to InternVL2-8B.

## IV Model Settings

For pipeline tools such as MinerU, Marker, and Mathpix, default settings are used for evaluation. Specifically, MinerU with Version 0.9.3 <sup>8</sup> is employed. For Marker, Version 1.2.3 <sup>9</sup> is evaluated. For Nougat, we utilize its 0.1.0-base model (350M). For GOT-OCR, we employ its format OCR mode to output structured data.

For general VLMs, we used the GPT4o, Qwen2-VL-72B, and InternVL2-Llama3-76B by setting the do\_sample $=$ False to ensure the reproducibility. After testing the different setting of max\_token, the best setting is chosen for each VLMs. Specifically, max\_token $=$ 32000 is set for Qwen2-VL-72B, and max\_token $=$ 4096 is set for InternVL2-Llama3-76B. For GPT-4o, the default setting is used.

## V More Details on Methods

Ignore handling. The purpose of this process is to avoid fluctuations in accuracy caused by the lack of uniformity in the output standards among document parsing algorithm. (1) Some algorithm (e.g., GPT-OCR, Qwen2-VL) tends to remove headers and footers, while others (e.g., GPT4o) prefers to retain them ( Figure S3). (2) Moreover, the reading order mismatch cause by captions and footnotes is also considered. For example, Nougat would put the image captions in the end of the page content( Figure S4), while others tend to put the image captions in human reading order.

Ignore handling is to minimize the impact of varying standards of document parsing on evaluation. Our evaluation dataset aims to more fairly assess the parsing accuracy of various algorithms, and these trivial issues regarding standards are not within our scope of consideration.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x5.png)

Figure S1: The Data Proportion of Pages for each Attribute in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x6.png)

Figure S2: The Visualization of vary Annotations in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x7.png)

Figure S3: The Vary Standards in parsing Header, Footers, and so on.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x8.png)

Figure S4: The Vary Standards in parsing Captions.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x9.png)

Figure S5: The Examples of Academic Papers, Books, Textbooks, Notes, and Magazines in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x10.png)

Figure S6: The Examples of Finacial Reports, Newspapers, Example Papers, and Slides in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x11.png)

Figure S7: The Examples of PDF pages with different Layout Types in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x12.png)

Figure S8: The Examples of PDF pages under Special Issues in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x13.png)

Figure S9: The Examples of Tables with different Frame in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x14.png)

Figure S10: The Examples of Tables under Special Issues in OmniDocBench.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x15.png)

Figure S11: The Good Model Result and Bad Model Result for Academic Papers.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x16.png)

Figure S12: The Good Model Result and Bad Model Result for Books.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x17.png)

Figure S13: The Good Model Result and Bad Model Result for Exam Papers.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x18.png)

Figure S14: The Good Model Result and Bad Model Result for Magazines.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x19.png)

Figure S15: The Good Model Result and Bad Model Result for Newspaper.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x20.png)

Figure S16: The Good Model Result and Bad Model Result for Handwriting Notes.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x21.png)

Figure S17: The Good Model Result and Bad Model Result for Financial Reports.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x22.png)

Figure S18: The Good Model Result and Bad Model Result for Slides.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x23.png)

Figure S19: The Good Model Result and Bad Model Result for Textbooks.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x24.png)

Figure S20: The Good Model Result and Bad Model Result for Fuzzy Scan Pages.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x25.png)

Figure S21: The Good Model Result and Bad Model Result for Pages with Watermark.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x26.png)

Figure S22: The Good Model Result and Bad Model Result for Colorful Background Pages.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x27.png)

Figure S23: The Good Model Result and Bad Model Result for Single Column Pages.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x28.png)

Figure S24: The Good Model Result and Bad Model Result for Double Column Pages.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x29.png)

Figure S25: The Good Model Result and Bad Model Result for Three Column Pages.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x30.png)

Figure S26: The Good Model Result and Bad Model Result for Complex Layout Pages.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x31.png)

Figure S27: The Good Model Result and Bad Model Result for Text Language in Chinese.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x32.png)

Figure S28: The Good Model Result and Bad Model Result for Text Language in English.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x33.png)

Figure S29: The Good Model Result and Bad Model Result for Text with Colorful Background.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x34.png)

Figure S30: The Bad Model Result for Text with Rotation.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x35.png)

Figure S31: The Good Model Result and Bad Model Result for Three Line Frame Table.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x36.png)

Figure S32: The Good Model Result and Bad Model Result for No Frame Table.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x37.png)

Figure S33: The Good Model Result and Bad Model Result for Rotated Table.

![Refer to caption](https://arxiv.org/html/2412.07626v2/x38.png)

Figure S34: The Good Model Result and Bad Model Result for Table with Formula.

[^1]: Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv:2303.08774*, 2023.

[^2]: Open AI. Hello gpt 4o, 2024. Accessed July 24, 2024.

[^3]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv:2308.12966*, 2024.

[^4]: Ayan Banerjee, Sanket Biswas, Josep Lladós, and Umapada Pal. Swindocsegmenter: An end-to-end unified domain adaptive transformer for document instance segmentation. In *ICDAR*, 2023.

[^5]: Ayan Banerjee, Sanket Biswas, Josep Lladós, and Umapada Pal. Graphkd: Exploring knowledge distillation towards document object detection with structured graph creation. In *ICDAR*, 2024.

[^6]: Lukas Blecher. pix2tex - latex ocr. [https://github.com/lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR), 2022. Accessed: 2024-2-29.

[^7]: Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. *arXiv:2308.13418*, 2024.

[^8]: Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 24185–24198, 2024.

[^9]: Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, Zecheng Xie, Jing Li, Kai Ding, and Lianwen Jin. M6doc: A large-scale multi-format, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15138–15147, 2023.

[^10]: Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M Rush. Image-to-markup generation with coarse-to-fine attention. In *International Conference on Machine Learning*, pages 980–989. PMLR, 2017.

[^11]: Harsh Desai, Pratik Kayal, and Mayank Singh. Tablex: a benchmark dataset for structure and content information extraction from scientific tables. In *Document Analysis and Recognition–ICDAR 2021: 16th International Conference*, pages 554–569, 2021.

[^12]: Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. *arXiv:2312.10997*, 2023.

[^13]: Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Nikolaos Barmpalios, Ani Nenkova, and Tong Sun. Unidoc: Unified pretraining framework for document understanding. *Advances in Neural Information Processing Systems*, 34:39–50, 2021.

[^14]: Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding. *arXiv preprint arXiv:2409.03420*, 2024.

[^15]: Mingxin Huang, Yuliang Liu, Zhenghao Peng, Chongyu Liu, Dahua Lin, Shenggao Zhu, Nicholas Yuan, Kai Ding, and Lianwen Jin. Swintextspotter: Scene text spotting via better synergy between text detection and text recognition. In *proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4593–4603, 2022a.

[^16]: Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. Tabtransformer: Tabular data modeling using contextual embeddings. arxiv 2020. *arXiv preprint arXiv:2012.06678*, 2012.

[^17]: Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking, 2022b.

[^18]: Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, and Wei Peng. Improving table structure recognition with visual-alignment sequential coordinate modeling. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11134–11143, 2023.

[^19]: Wonseok Hwang, Jinyeong Yim, Seunghyun Park, Sohee Yang, and Minjoon Seo. Spatial dependency parsing for semi-structured document information extraction. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP*, pages 330–343. Association for Computational Linguistics (ACL), 2021.

[^20]: Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, and Ernest Valveny. Icdar 2015 competition on robust reading. In *2015 13th International Conference on Document Analysis and Recognition*, pages 1156–1160, 2015.

[^21]: Vladimir I Levenshtein et al. Binary codes capable of correcting deletions, insertions, and reversals. In *Doklady Physics*, pages 707–710. Soviet Union, 1966.

[^22]: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020.

[^23]: Chenxia Li, Weiwei Liu, Ruoyu Guo, Xiaoting Yin, Kaitao Jiang, Yongkun Du, Yuning Du, Lingfeng Zhu, Baohua Lai, Xiaoguang Hu, Dianhai Yu, and Yanjun Ma. Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system, 2022a.

[^24]: Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. Dit: Self-supervised pre-training for document image transformer. In *ACMMM*, 2022b.

[^25]: Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. Tablebank: Table benchmark for image-based table detection and recognition. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 1918–1925, 2020a.

[^26]: Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. *arXiv:2006.01038*, 2020b.

[^27]: Zhe Li, Lianwen Jin, Songxuan Lai, and Yecheng Zhu. Improving attention-based handwritten mathematical expression recognition with scale augmentation and drop attention. In *2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR)*, pages 175–180. IEEE, 2020c.

[^28]: Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024a.

[^29]: Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for fine-grained multi-page document understanding. *arXiv:2405.14295*, 2024b.

[^30]: Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin, and Liangwei Wang. Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In *proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 9809–9818, 2020.

[^31]: Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. *Science China Information Sciences*, 67(12), 2024c.

[^32]: Tengchao Lv, Yupan Huang, Jingye Chen, Yuzhong Zhao, Yilin Jia, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model, 2024.

[^33]: Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. pages 311–318, 2002.

[^34]: Vik Paruchuri. Marker, 2024.

[^35]: Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. Doclaynet: A large human-annotated dataset for document-layout segmentation. In *Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining*, pages 3743–3751, 2022.

[^36]: Subhojeet Pramanik, Shashank Mujumdar, and Hima Patel. Towards a multi-modal, multi-task learning based pre-training framework for document representation learning. *arXiv preprint arXiv:2009.14457*, 2020.

[^37]: RapidAI. Rapidtable. [https://github.com/RapidAI/RapidTable](https://github.com/RapidAI/RapidTable), 2023.

[^38]: Ray Smith, Daria Antonova, and Dar-Shyang Lee. Adapting the tesseract open source ocr engine for multilingual ocr. In *Proceedings of the International Workshop on Multilingual OCR*, 2009.

[^39]: Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

[^40]: Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A universal network for real-world mathematical expression recognition, 2024a.

[^41]: Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He. Cdm: A reliable metric for fair and accurate formula recognition evaluation. *arXiv:2409.03643*, 2024b.

[^42]: Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction. *arXiv:2409.18839*, 2024c.

[^43]: Pengfei Wang, Chengquan Zhang, Fei Qi, Shanshan Liu, Xiaoqiang Zhang, Pengyuan Lyu, Junyu Han, Jingtuo Liu, Errui Ding, and Guangming Shi. Pgnet: Real-time arbitrarily-shaped text spotting with point gathering network. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 2782–2790, 2021.

[^44]: Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024d.

[^45]: Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. *arXiv:2409.01704*, 2024.

[^46]: Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language model. In *European Conference on Computer Vision*, pages 408–424. Springer, 2025.

[^47]: Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al. Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models. *arXiv preprint arXiv:2406.11633*, 2024a.

[^48]: Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, et al. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. *arXiv preprint arXiv:2402.12185*, 2024b.

[^49]: Zhong Xu, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In *2019 International conference on document analysis and recognition*, pages 1015–1022, 2019.

[^50]: Cong Yao. DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond. *ArXiv*, 2023.

[^51]: Jianshu Zhang, Jun Du, and Lirong Dai. Multi-scale attention with dense encoder for handwritten mathematical expression recognition. In *2018 24th international conference on pattern recognition (ICPR)*, pages 2245–2250. IEEE, 2018.

[^52]: Qintong Zhang, Victor Shea-Jay Huang, Bin Wang, Junyuan Zhang, Zhengren Wang, Hao Liang, Shawn Wang, Matthieu Lin, Wentao Zhang, and Conghui He. Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction. *arXiv preprint arXiv:2410.21169*, 2024.

[^53]: Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception, 2024.

[^54]: Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In *European conference on computer vision*, pages 564–580, 2020.

[^55]: Hongbin Zhou, Xiangchao Yan, and Bo Zhang. Structeqtable-deploy: A high-efficiency open-source toolkit for table-to-latex transformation. [https://github.com/UniModal4Reasoning/StructEqTable-Deploy](https://github.com/UniModal4Reasoning/StructEqTable-Deploy), 2024.