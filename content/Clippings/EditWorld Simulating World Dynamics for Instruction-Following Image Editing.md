---
title: "EditWorld: Simulating World Dynamics for Instruction-Following Image Editing"
source: "https://arxiv.org/html/2405.14785v1"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/04_%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E4%B8%8E%E7%BC%96%E8%BE%91/EditWorld%2C%20Simulating%20World%20Dynamics%20for%20Instruction-Following%20Image%20Editing%2C%20Ling%20Yang%20et%20al.%2C%202024.no_watermark.zh-CN.dual.pdf"
---
Ling Yang <sup>1</sup>, Bohan Zeng <sup>1</sup> <sup>1</sup>, Jiaming Liu <sup>2</sup>, Hong Li <sup>1</sup>,  
Minghao Xu <sup>4</sup>, Wentao Zhang <sup>1†</sup>, Shuicheng Yan <sup>3</sup>  
<sup>1</sup> Peking University, <sup>2</sup> Tiamat AI, <sup>3</sup> Skywork AI, <sup>4</sup> Mila - Québec AI Institute These authors contributed equally.Correspondence to: yangling0818@163.com, wentao.zhang@pku.edu.cn.

###### Abstract

Diffusion models have significantly improved the performance of image editing. Existing methods realize various approaches to achieve high-quality image editing, including but not limited to text control, dragging operation, and mask-and-inpainting. Among these, instruction-based editing stands out for its convenience and effectiveness in following human instructions across diverse scenarios. However, it still focuses on simple editing operations like adding, replacing, or deleting, and falls short of understanding aspects of world dynamics that convey the realistic dynamic nature in the physical world. Therefore, this work, EditWorld, introduces a new editing task, namely world-instructed image editing, which defines and categorizes the instructions grounded by various world scenarios. We curate a new image editing dataset with world instructions using a set of large pretrained models (e.g., GPT-3.5, Video-LLava and SDXL). To enable sufficient simulation of world dynamics for image editing, our EditWorld trains model in the curated dataset, and improves instruction-following ability with designed post-edit strategy. Extensive experiments demonstrate our method significantly outperforms existing editing methods in this new task. Our dataset and code will be available at [https://github.com/YangLing0818/EditWorld](https://github.com/YangLing0818/EditWorld)

## 1 Introduction

Text-to-image models have achieved great success with the significant development of diffusion models [^1] [^2] [^3], such as Stable Diffusion [^4], DALL-E 2/3 [^5] [^6], and RPG [^7]. In order to improve controllability, text-guided image editing has gained remarkable improvements [^8] [^9] [^10] [^7]. Early researches such as StyleGAN [^9] and CycleGAN [^11] are able to achieve reasonable image style transfer. Subsequent models [^12] [^13] [^14] [^15] [^16] mainly extract key features from images, which can be further incorporated into the text-to-image generation process to generate images with the style and content of reference images.

Aiming to make visual editing more precise, some methods try to enable dragging operations [^17] [^18] [^19] or condition on additional masks [^20] [^14] [^15] for localized image manipulation. Thanks to the quick development of multimodal pretraining methods like CLIP models [^21], numerous methods [^22] [^23] [^24] [^25] [^26] leverage expressive cross-modal semantic controls to guide the text-based image editing process without additional manual operations or masks.

Recently, instruction-based image editing [^27] [^28] [^29] [^30] [^31] teaches the image editing model to follow human instructions, allowing ordinary users to conveniently and effortlessly manipulate images through natural commands. For instance, InstructPix2Pix [^27] leverages two large pretrained models (i.e., GPT-3 and Stable Diffusion) to generate a large dataset of input-goal-instruction triplet examples, and trains an instruction-following image editing model on the dataset. MGIE [^31] further incorporates multimodal large language models (MLLMs) to facilitate editing instructions by providing more expressive visual-aware instructions.

Despite the high-quality results achieved by existing text- and instruction-based methods, they ignore and also struggle to deal with the world dynamics that convey the realistic visual dynamic of the physical world [^32] in image editing. For example, they can achieve reasonable results when instructed to "put a hat on the man", but they may fail to edit with the instruction "what happens if the man slips". As shown in Fig. 1, neither InstructPix2pix [^27] nor MagicBrush [^28] can generate reasonable editing results.

Figure 1: EditWorld offers 10k sets of input and output images with instructions for world-instructed image editing task. As shown in the figure, our EditWorld performs better than both InstructPix2Pix [^27] and MagicBrush [^28] in image editing involving world dynamics.

To enable image editing to reflect complex world dynamics from both real physical world and virtual media, this work EditWorld, introduces a new task named world-instructed image editing, as the data examples presented in Fig. 1. Specifically, we define and categorize diverse world instructions, and based on them curate a new multimodal dataset that consists of a large number of input-instruction-output triplets. The construction of this dataset has two branches: (1) We utilize GPT-3.5 to generate world instructions for different scenarios and employ text-to-image diffusion models such as SDXL [^33] and ControlNet [^34] to produce high-quality input-output image pairs that strictly follow the world instructions; (2) We select two frames from video data as paired image data, which contains consistent identity between consecutive frames but has large spatial variances or dynamics caused by video motions. Then we utilize Video-LLava [^35] and GPT-3.5 to generate corresponding instructions.

Finally, we finetune a instructPix2Pix model using our curated dataset and propose a zero-shot image manipulation strategy post-edit to improve the ability of instruction following and enhance the appearance consistency of non-editing areas. Extensive experiments demonstrate our method achieve state-of-the-art (SOTA) results in the world-instructed image editing task compared to previous methods. We summarize our main contributions as follows:

- We propose a new fundamental task named world-instructed image editing, which reflects the real dynamics in both physical world and virtual media. We define and categorize world instructions according to different scenarios.
- We curate a new multimodal dataset for world-instructed image editing by leveraging a set of large pretrained models, including GPTs, MLLMs, and text-to-image diffusion models. This curated dataset can also serve as a new benchmark for our proposed editing task.
- We train and improve our diffusion-based image editing model on the dataset, outperforming previous state-of-the-art (SOTA) image editing methods in context of world-instructed image editing while maintaining competitive results in traditional editing tasks.

## 2 Related Works and Discussions

### 2.1 Diffusion Models for Text-Guided Image Editing

Recent advancements in Diffusion models (DMs) [^2] [^36] [^37] [^38] [^39] [^4] [^40] [^7] [^26] [^41] [^42] [^43] and multimodal frameworks such as CLIP [^21] have significantly enhanced text-to-image generation, and demonstrate impressive generation results [^44] [^27] [^45] [^46] [^47] [^5] [^48] [^49] [^50] [^51] [^52]. Building upon these foundations, some works [^53] [^54] [^55] [^20] [^10] [^56] [^24] realize text-guided image editing using conditional diffusion models [^57]. For example, Blend Diffusion [^55] introduces a mask-guided editing method for precise and seamless image manipulation. Prompt-to-Prompt [^10] manipulates an image to align with the words in the prompt by modifying cross-attention, allowing for localized image editing through changing textual prompts. PAIR-diffusion [^58] provides a way to independently edit both the structure and appearance within each masked area of the input image. Although these methods perform impressively, they highly depend on elaborate textual descriptions or precise regional masks. In contrast, providing direct instructions to modify specific regions or attributes, such as ’make the tie blue’, offers a more straightforward and user-friendly way for ordinary users.

### 2.2 Instruction-Following Image Editing

Instruction-based image editing [^59] [^60] [^61] [^62] [^27] [^28] [^29] aims to teach a generative model to follow human-written instructions for image editing, and has made some progress. For instance, InstructPix2Pix [^27] employs a large language model GPT-3 [^63] and Prompt-to-Prompt [^10] to synthesize a training dataset for obtaining a diffusion model specialized in instruction-following image editing. MagicBrush [^28] enhances the capabilities of InstructPix2Pix by fine-tuning it with a real collected image dataset. Another line of researches utilizes multimodal large language models (MLLMs) to enhance the cross-modal understanding for instruction-based image editing [^31] [^30]. For example, MGIE [^31] jointly learns the MLLM and editing model with vision-aware expressive instructions to provide explicit guidance. Although existing image editing methods, especially instruction-based image editing methods, can already make simple editing operations in images, they still struggle to deal with the instructions from real physical world and also lack a dataset that contains image pairs reflecting world dynamics. To address this issue, we formulate a new task setting, namely world-instructed image editing, and achieve SOTA performance under this setting.

### 2.3 Leveraging Large Pretrained Models for Multimodal Task

With the development of large-scale pretraining techniques, many works leverage the extensive knowledge embedded in pretrained large models (e.g., (M)LLM, SAM or Stable Diffusion) [^64] [^4] to more effectively addressing various tasks. Notable achievements have been made in some areas, such as image editing [^10] [^29] [^31] [^27] [^28], video editing [^65] [^66] [^67] and text-to-3D generation [^68] [^69] [^70]. Our EditWorld leverages a set of large pretrained models to process both image and video data, and construct a multimodal dataset that contains diverse input-instruction-output triplet examples for enhancing world-instructed image editing.

### 2.4 Data Generation with Generative Models

Deep models generally need vast amounts of data for training. While internet offers a rich data resource, it often lacks the structured, paired data needed for supervised learning. As generative models advance, they are increasingly seen as a cost-effective way to produce large quantities of training data for various applications [^71] [^72] [^73] [^74] [^27] [^28] [^75] [^76]. The paired data required for our proposed task setting is rarely found on the internet. In this work, we design two synthesis branches, i.e., text-to-image generation and video frame extraction, and utilize a combination of generative models to generate complex and rich data samples for world-instructed image editing.

## 3 Method

### 3.1 Definition of World Instructions

Instruction-guided image editing aims to edit the input images according to the instruction. Existing methods [^27] [^28] have effectively achieved object replacing, removing, and adding. However, it remains a challenge for existing methods to simulate the real visual dynamics in physical world or virtual media. Therefore, in this section, we will introduce how we generate a multimodal dataset containing editing triples that can reflect various world dynamics. We first define and categorize our proposed world instructions. The images we encounter typically originate from either physical world or virtual media. Consequently, we categorize world instructions into two main types: real-world and virtual-world. We further refine the two types of world instructions into seven specific categories based on the different scenarios, as shown in Table 1. To enhance the comprehension of different world instructions, we provide an example input-instruction-output triplet to instantiate each type.

![[Uncaptioned image]](https://arxiv.org/html/2405.14785v1/extracted/2405.14785v1/images/data_example/LTT_ori.png)

Table 1: Definition and examples of different world instructions in our generated dataset.

### 3.2 Generating A Dataset with World Instructions

![Refer to caption](https://arxiv.org/html/2405.14785v1/x2.png)

Figure 2: Generating a dataset of world-instructed image editing from two different branches.

#### 3.2.1 Text-to-Image Generation for Diverse Scenarios

In this part, we introduce how to leverage LLMs and text-to-image generation models to produce a large amount of input-instruction-output triplets. As shown in Fig. 2(a), we design a template to prompt the GPT-3.5 to synthesize a series of textual quadruples, (prompt $y_{ori}$ of input image, instruction $y_{instr}$, prompt $y_{tar}$ of output image, and keywords $\{k_{1}$, $k_{2}$,…, $k_{n}\}$), which are used for later generation of input-output paired images. Among them, keywords are used to denote the parts to be edited.

##### Combinational T2I Generation Procedure.

We incorporate a pretrained text-to-image diffusion denoiser $\epsilon_{\theta}$ of SDXL [^33] to synthesize input image $I_{ori}$ and output image $I_{tar}$ based on the textual quadruple. Firstly, to localize the editing areas, we collect the cross-attention maps $M_{1}$, $M_{2}$,…, $M_{n}$ of $k_{1}$, $k_{2}$,…, $k_{n}$ during the generation of $I_{ori}$. We binarize each pixel $(m,n)$ of these attention masks using a threshold to create binary masks $M^{b}$. The binarization process is defined as follows:

$$
\displaystyle\begin{split}M^{b}_{i}(m,n)&=\begin{cases}0,&\text{if }M_{i}(m,n)%
\leq 0.8125*\mathrm{mean}(M_{i}),\\
1,&\text{if }M_{i}(m,n)>0.8125*\mathrm{mean}(M_{i}),\end{cases}\end{split}
$$
 
$$
\displaystyle\begin{split}M^{b}&=M^{b}_{1}\cup M^{b}_{2}\cup...\cup M^{b}_{n}.%
\end{split}
$$

Then, to preserve the non-editing areas, we conduct image inpainting within the masked areas $M^{b}$ of $I_{ori}$ to synthesize $I_{tar}$ with the textual guidance $y_{tar}$. The $I_{ori}$ is encoded into latent space using the VAE encoder $\mathcal{E}$ of the LDM [^4] $z_{ori}\sim\mathcal{E}(I_{ori})$, and the denoising-based inpainting process can be defined as:

$$
\displaystyle z_{ori,t}
$$
 
$$
\displaystyle=\mathcal{N}(\sqrt{\alpha_{t}}z_{ori},(1-\alpha_{t})\mathbf{I}),
$$
$$
\displaystyle z^{*}_{t}=z_{t}\cdot M^{b}+z_{ori,t}\cdot(1-M^{b}),
$$

where $z_{t}$ denotes the denoised latent. To further ensure the identity consistency between $I_{ori}$ and $I_{tar}$, we employ IP-Adapter [^14] and ControlNet [^34] $\mathcal{F}_{ctrl}$ for additional refinement of $I_{tar}$. Specifically, IP-Adapter extracts visual feature $f_{ori}$ from the original image $I_{ori}$, which guides the refinement process to maintain the identity consistency between $I_{ori}$ and $I_{tar}$. Simultaneously, ControlNet utilizes the canny map $m_{canny}$ of $I_{tar}$ to guide the refinement, thereby preserving the structure of $I_{tar}$. The predicted noise of the refinement of $I_{tar}$ is denoted as:

$$
\displaystyle\epsilon=\epsilon_{ip}(z_{t},t,y_{tar},f_{ori},\mathcal{F}_{ctrl}%
(z_{t},m_{canny})),
$$

where $\epsilon_{ip}$ is the denoiser of IP-Adapter. Finally, we employ a MLLM as the discriminator, using semantic alignment, identity consistency, and image quality as criteria to select the most reasonable and high-quality generated image pairs as our training data. The types of world instructions that can be obtained with this text-to-image generation branch include Long-Term, Physical-Trans, Implicit-Logic, Story-Type, and Real-to-Virtual.

##### Expressiveness of Our Generation Procedure.

It is worth mentioning that, compared to InstructPix2Pix [^27] which generates training data using simple cross-attention manipulation in Prompt-to-Prompt [^10], our text-to-image generation branch can handle more complex image changes by the combinational utilization of specific pretrained models. For example, in Prompt-to-Prompt, the input-output text modifications only allow simple word changes, such as "A snowman" to "A melted snowman". In contrast, our method allow a more complex change from "A snowman stands in a winter wonderland surrounded by glistening snowflakes" to "A melted snowman, leaving behind a puddle of water and a few remnants of coal and carrot on the ground". This means that the dataset generation of EditWorld is more flexible and expressive, and the generated data can include more complex editing scenarios.

#### 3.2.2 Extracting Paired Data from Realistic Video Frames

In addition to the rich generated data from text-to-image branch, we also extract paired data from a video dataset InternVid [^77] to add more realistic data into our dataset. More concretely, we select two video frames that have strong identity consistency but the greatest spatial/visual variances or dynamics caused by certain motions. The two frames usually are the starting and ending points of the video storyline. Then, as shown in Fig. 2(b), we use a pretrained video-language model Video-LLava [^35] to obtain a description of this video storyline. Subsequently, we employ GPT-3.5 to transform this description into a world instruction according to a predefined format. In this way, the extracted first and last frames can serve as the input image and output image, respectively. Combined with the produced instructions, they can form a series of input-instruction-output triplets for augmenting our editing dataset. This branch of triplet examples extracted from video data include following types of world instructions: Spatial-Trans, physical-Trans, Story-Type, and Exaggeration.

#### 3.2.3 Statistics for The EditWorld Dataset

![Refer to caption](https://arxiv.org/html/2405.14785v1/x3.png)

Figure 3: Statistics of our generated EditWorld dataset.

To provide a more intuitive understanding, we visualize the details of our dataset. Fig. 3(a) shows the distribution of EditWorld dataset based on different branches and categories of instructions. Since the quantities of distinct types of changes in world dynamics vary (for instance, changes in the physical world are more abundant than in the virtual world), the corresponding data volumes for various categories of world instruction in EditWorld dataset differ accordingly. Additionally, to more comprehensively illustrate the contents of our dataset, Fig. 3(b) presents the 20 most frequently occurring keywords. Besides, some generated images of these two branches may have low resolution and lack reasonableness, we provide the specific recheck process in section A.1.

![Refer to caption](https://arxiv.org/html/2405.14785v1/x4.png)

Figure 4: Illustrate our image manipulation method post-edit for instruction-guided image editing.

### 3.3 Image Editing Model of EditWorld

We use the constructed dataset from the two branches to train an instruction-based image editing diffusion model (DM). The input conditions for the DM include the instruction $y_{instr}$ and the latent code $z_{ori}$ of input image $I_{ori}$. The supervision function for training the denoiser $\epsilon_{\theta}$ is as follows:

$$
\displaystyle L=\mathbb{E}_{z_{t},t,z_{ori},y_{instr},\epsilon\sim\mathcal{N}(%
0,1)}[||\epsilon-\epsilon_{\theta}(z_{t},z_{ori},y_{instr},t)||^{2}_{2}].
$$

We define the generated result of the trained diffusion model as $I_{gen}$. Following the classifier-free guidance described in InstructPix2Pix [^27], we modify the noise estimation based on the $I_{ori}$ and the $y_{instr}$ as follows:

$$
\displaystyle\hat{\epsilon}_{\theta}(z_{t},I_{ori},y_{instr})=
$$
 
$$
\displaystyle\ \epsilon_{\theta}(z_{t},\varnothing,\varnothing)+s_{I}\cdot(%
\epsilon_{\theta}(z_{t},I_{ori},\varnothing)-\epsilon_{\theta}(z_{t},%
\varnothing,\varnothing))
$$
 
$$
\displaystyle+s_{T}\cdot(\epsilon_{\theta}(z_{t},I_{ori},y_{instr})-\epsilon_{%
\theta}(z_{t},I_{ori},\varnothing)),
$$

where $s_{I}$ and $s_{T}$ denote the guidance scales.

##### Post-Edit Method

The fine-tuned DM has been able to make effective world-instructed image editing. To further refine the generated results, and maintain the non-edited areas of the input image as much as possible, as shown in Fig. 4, we propose a new image manipulation method post-edit to optimize the editing process. Specifically, we extract a binary attention mask $M^{b}_{instr}$ from $\epsilon_{\theta}(z_{t},I_{ori},\varnothing)$ during the inference process, providing approximate localization of editing position. Then, we use SAM [^64] to segment both $I_{gen}$ and $I_{ori}$, and calculate the precise masks $M_{gen}$ and $M_{ori}$ which have the maximum overlaps with $M^{b}_{instr}$. Subsequently, we define the union of $M_{gen}$ and $M_{ori}$ as the mask $M_{edit}$, which enhances the precision of the editing localization. While $I_{gen}$ has already offered a reasonable editing results, putting $M_{edit}$, canny map of $I_{gen}$, and visual features of $I_{ori}$ and $I_{ori}$ into an image inpainting process, our method can further boost the quality of $I_{gen}$ while preserving the non-edited areas well.

## 4 Experiment

### 4.1 Implementation Details

##### Evaluation

We randomly selected 300 data triples from the text-to-image branch and 200 data samples from the videos branch for testing. The remaining data were utilized to train the image editing model of EditWorld. Each test data triple from the testing dataset includes an input image, instruction, and an output image description. We employ the CLIP score as a metric to assess whether the generated editing results align with the output semantics, thus judging the effectiveness of the method in world-instructed editing. Additionally, due to the complexity of the world-instructed editing task, the CLIP score may be insufficient to accurately evaluate the logical accuracy of the image editing in some cases. Therefore, we introduce a new metric, termed as MLLM score, which employs multimodal large language models to assess the instruction-following ability of various methods in world-instructed editing task. Comprehensive evaluation details and a full list of validation results regarding MLLM score can be found in section A.2.

##### Training Details

Data generation and model training of EditWorld are completed on 4 $\times$ 80GB NVIDIA A100 GPUs. The model, initialized with the weights of InstructPix2Pix [^27], is fine-tuned on our proposed dataset for a total of 100 epochs. We use a batch size of 64 and an image resolution of 512×512. The training employs the Adam optimizer [^78] with a learning rate of 5e-6.

### 4.2 Quantitative Evaluation

To more accurately assess the editing capabilities of various models across different scenarios in world-instructed image editing, this study tests image editing abilities under different data types, categorized by seven types of instructions and two data branches. We compare our EditWorld with five SOTA methods [^56] [^25] [^27] [^28] [^31] under different types of instruction in Table 2.

##### Comparing Editing Methods Between Text-to-Image and Video Branches

Due to the different motivations for data acquisition in our text-to-image and video branches, we evaluate the performance of editing methods on the testing datasets of these branches separately. Our method consistently outperforms others in terms of CLIP scores and MLLM score metrics in two datasets, demonstrating superior editing capabilities and stability in world-instructed editing tasks. Remarkably, our method maintains excellent performance without post-editing, which highlights the effectiveness of our curated dataset. The comparison results reveal distinct variations in the performance enhancements throughout two testing datasets: within the text-to-image branch’s testing dataset, our method generally signally outperform other methods, indicating that the sophisticated editing scenarios remain a challenge for traditional editing methods. Meanwhile, in the video branch’s testing dataset, the improvement of our method is less significant than in the text-to-image branch’s testing dataset. We attribute this difference to the fact that extracted frames from videos often contain complex scenes where the main subject is not prominent. Consequently, in the video branch, it is challenging to achieve ideal edits. Besides, as shown in Table 5, while InstructPix2Pix achieves a higher CLIP score than other baseline methods, it indicates that although InstructPix2Pix responds effectively to the instructions, it still struggles to maintain the visual quality of the input images and realize desirable image editing in world-instructed editing.

Table 2: Quantitative comparison of CLIP score and MLLM score. IP2P: InstructPix2Pix [^27]; MB: MagicBrush [^28]. Bold results are the best, and underlined ones are the second best.

<table><tbody><tr><th rowspan="2">Category</th><td colspan="2">Text-based Method</td><td colspan="5">Instruction-based Method</td></tr><tr><td>Pix2Pix-Zero</td><td>DiffEdit</td><td>IP2P</td><td>MB</td><td>MGIE</td><td>EditWorld</td><td>w/o post-edit</td></tr><tr><th colspan="8">CLIP Score of Text-to-image Branch</th></tr><tr><th>Long-Term</th><td>0.1831</td><td>0.1952</td><td>0.2140</td><td>0.1870</td><td>0.1863</td><td>0.2244</td><td>0.2294</td></tr><tr><th>Physical-Trans</th><td>0.2266</td><td>0.2283</td><td>0.2186</td><td>0.2101</td><td>0.2286</td><td>0.2385</td><td>0.2467</td></tr><tr><th>Implicit-Logic</th><td>0.2356</td><td>0.2307</td><td>0.2390</td><td>0.2432</td><td>0.2350</td><td>0.2542</td><td>0.2440</td></tr><tr><th>Story-Type</th><td>0.2111</td><td>0.1980</td><td>0.2063</td><td>0.2070</td><td>0.2084</td><td>0.2534</td><td>0.2354</td></tr><tr><th>Real-to-Virtual</th><td>0.2294</td><td>0.2419</td><td>0.2285</td><td>0.2344</td><td>0.2305</td><td>0.2524</td><td>0.2435</td></tr><tr><th colspan="8">CLIP Score of Video Branch</th></tr><tr><th>Spatial-Trans</th><td>0.2102</td><td>0.2058</td><td>0.2175</td><td>0.1997</td><td>0.2157</td><td>0.2420</td><td>0.2286</td></tr><tr><th>Physical-Trans</th><td>0.1899</td><td>0.2407</td><td>0.2315</td><td>0.2278</td><td>0.2277</td><td>0.2467</td><td>0.2483</td></tr><tr><th>Story-Type</th><td>0.1724</td><td>0.2114</td><td>0.2318</td><td>0.2262</td><td>0.2155</td><td>0.2365</td><td>0.2399</td></tr><tr><th>Exaggeration</th><td>0.2164</td><td>0.2275</td><td>0.2416</td><td>0.2328</td><td>0.2208</td><td>0.2443</td><td>0.2433</td></tr><tr><th colspan="8">MLLM Score of Both Branches</th></tr><tr><th>Text-to-image</th><td>0.8465</td><td>0.8517</td><td>0.8763</td><td>0.8455</td><td>0.8173</td><td>0.8958</td><td>0.9060</td></tr><tr><th>Video</th><td>0.8706</td><td>0.94</td><td>0.9493</td><td>0.9715</td><td>0.9474</td><td>0.9920</td><td>0.9891</td></tr></tbody></table>

##### Evaluation of Editing Methods in Various Instruction Categories

To investigate the editing capabilities of various methods across different scenarios in world-instructed editing tasks, we evaluated their performance across seven distinct categories of instructions. Our method demonstrated superior performance in all categories. Particularly in categories such as Long-Term, Physical-Trans, Spatial-Trans, Story-Type, and Exaggeration, our method excels as these involve significant differences between the input and output images while maintaining a relevant connection between them. The models used for traditional editing, lacking the knowledge of world dynamics, fail to generate plausible results in these scenarios, thus highlighting the effectiveness of our approach. For categories Implicit-Logic and Real-to-Virtual, where instructions are complex or abstract and the changes between the input and output images are subtle, some changes are akin to those encountered in traditional editing tasks. Existing text-based editing methods [^56] [^25] perform well since they are directly provided with output text, simplifying their task. Conversely, instruction-based methods like InstructPix2Pix, which are trained with datasets featuring instructions generated by LLMs, excel in managing these intricate instructions. Despite this, our method consistently outperforms these baseline methods in these two categories.

### 4.3 Qualitative Analysis

Figure 5: Qualitative comparison of world-instructed image editing.

To demonstrate the editing capabilities of the EditWorld model, we randomly selected several examples from different instructions for visualization, as shown in figs. 5 and 6. Previous editing methods were generally unable to correctly edit images based on the given text or instructions, resulting in either no changes to the image or significant alterations that did not meet the editing requirements. In contrast, our method can effectively edit images according to the given instructions, producing high-quality results. Notably, while DiffEdit [^25] and MGIE [^31] maintain the quality of the input images, their visualizations reveal a lack of accurate reflection of the editing instructions. We provide more qualitative results in appendix B.

Figure 6: Qualitative comparison of world-instructed image editing.

### 4.4 Ablation Study

To validate that our proposed post-edit method effectively preserves non-edited areas while maintaining high-quality editing results, and performs well in traditional editing tasks, we conducted an ablation study. We selected 50 traditional editing test datasets, including 18 input-instruction-output triples for ’add’, 25 triples for ’change’, and 7 triples for ’remove’. As presented in Table 3, the LPIPS metric [^79] significantly improves with post-edit, whereas the CLIP score remains consistent, This demonstrates that our post-edit can preserve the non-edited areas without degrading the editing quality. At the same time, our EditWorld achieves comparable results to [^27] [^28] on the CLIP score, confirming its effectiveness in traditional editing tasks.

Table 3: Ablation study, using the LPIPS distance between the generated edited results and the input image, tests the effectiveness of post-edit in three methods: instructPix2Pix (IP2P), MagicBrush (MB), and our EditWorld.

| Metrics | IP2P | IP2P w/ post-edit | MB | MB w/ post-edit | Ours | Ours w/o post-edit |
| --- | --- | --- | --- | --- | --- | --- |
| LPIPS | 0.3929 | 0.3465 | 0.2397 | 0.2358 | 0.3082 | 0.3778 |
| CLIP score | 0.2317 | 0.2332 | 0.2383 | 0.2357 | 0.2344 | 0.2334 |

## 5 Conclusion

We propose a new image editing task called world-instructed image editing, which uses different scenarios in the real and virtual worlds to provide editing instructions for image editing. We classify, define, and exemplify these instructions, and use GPT, large-scale multimodal models, and text-to-image generation models to obtain image editing data from a large number of videos and texts. Using this data, we construct quantitative evaluation metrics for the world-instructed image editing task, and use the collected data to train and improve the image editing model, achieving state-of-the-art (SOTA) performance in this new world-instructed image editing task.

## References

## Appendix

## Appendix A EditWorld Dataset

### A.1 Human Selection

Utilizing text-to-image generation and video branches to obtain our dataset, some data triples may lack reasonableness despite the rich content and diverse scenarios. To further enhance the data quality, we employ some workers to recheck our curated dataset. Their specific tasks include: 1) For triples where the text-to-image generation results are unsatisfactory, adjust the prompt to improve the quality of the synthesis results; 2) Remove some editing data from the video branch whose images are unclear and blurry; 3) Revise unreasonable instructions in the data triples obtained from videos. At the same time, we discover that for some prompts, SDXL cannot generate desirable results, such as "balloon bursting" and "sandcastle being washed away by the sea". To address this, we adopt DALL·E 3 to generate and select higher quality images to replace the triples generated by SDXL.

### A.2 MLLM Score

Table 4: Quantitative comparison of MLLM score. IP2P: InstructPix2Pix [^27]; MB: MagicBrush [^28].

<table><tbody><tr><th rowspan="2">Category</th><td colspan="2">Text-based Editing</td><td colspan="5">Instruction-based Editing</td></tr><tr><td>Pix2Pix-Zero</td><td>DiffEdit</td><td>IP2P</td><td>MB</td><td>MGIE</td><td>EditWorld</td><td>w/o post-edit</td></tr><tr><th colspan="8">Text-to-image Generation Branch</th></tr><tr><th>Long-Term</th><td>0.9871</td><td>0.9846</td><td>0.9846</td><td>0.9874</td><td>0.9743</td><td>1.0</td><td>0.9871</td></tr><tr><th>Physical-Trans</th><td>0.8381</td><td>0.8476</td><td>0.8571</td><td>0.8452</td><td>0.8357</td><td>0.9095</td><td>0.9023</td></tr><tr><th>Implicit-Logic</th><td>0.8047</td><td>0.8323</td><td>0.8809</td><td>0.8238</td><td>0.7857</td><td>0.8810</td><td>0.9048</td></tr><tr><th>Story-Type</th><td>0.7652</td><td>0.7857</td><td>0.8047</td><td>0.7714</td><td>0.6619</td><td>0.8429</td><td>0.8762</td></tr><tr><th>Real-to-Virtual</th><td>0.8375</td><td>0.8083</td><td>0.8542</td><td>0.8</td><td>0.8292</td><td>0.8658</td><td>0.8583</td></tr><tr><th colspan="8">Video Extraction Branch</th></tr><tr><th>Spatial-Trans</th><td>0.9271</td><td>0.9152</td><td>0.9878</td><td>0.9608</td><td>0.983</td><td>0.997</td><td>1.0</td></tr><tr><th>Physical-Trans</th><td>0.8435</td><td>0.9791</td><td>0.9875</td><td>0.9792</td><td>0.9458</td><td>0.9875</td><td>0.9846</td></tr><tr><th>Story-Type</th><td>0.8175</td><td>0.9382</td><td>0.9448</td><td>0.989</td><td>0.961</td><td>1.0</td><td>0.994</td></tr><tr><th>Exaggeration</th><td>0.8944</td><td>0.9277</td><td>0.8774</td><td>0.957</td><td>0.9</td><td>0.9834</td><td>0.9778</td></tr></tbody></table>

Due to the limitations of the CLIP score in demonstrating the capability of world-instructed editing, in this section, we introduce the use of the MLLM Video-LLava [^35] and propose a new metric, MLLM score, to validate the effectiveness of world-instructed editing. Specifically, we present edited images, input and output texts, and instructions for the Video-LLava model. The specific prompts provided for Video-LLava for scoring the edited images are as follows: "The input description <input text>, the editing instruction <instruction>, and the output description <output text>. Please evaluate if the given edited image has been successfully edited. if you think editing is successful, just give me 1, else if you think editing fails, just give me 0". As shown in Table 4, our method continues to perform best under the MLLM score. Simultaneously, the performance of InstructPix2Pix [^27] in the MLLM score is significantly worse than that in the CLIP score. Qualitative comparisons suggest that InstructPix2Pix often produces unreasonable results, further validating the superiority of the MLLM score over the CLIP score. Additionally, while the MLLM score is a reasonable metric for world-instructed editing, it struggles to differentiate subtle variations. Therefore, if a more effective metric is desired to evaluate the outcomes of world-instructed editing, it is necessary for the MLLM to more accurately distinguish differences between images or videos."

## Appendix B More Qualitative Results

To further demonstrate the generation quality of EditWorld, we present some high-resolution editing results in Fig. 7 and Fig. 8.

Figure 7: More editing results of EditWorld

Figure 8: More editing results of EditWorld

## Appendix C Limitation and Discussion

##### Limitation.

Although our proposed world-instructed image editing task is of great practical significance, the scenarios in the real and virtual worlds are too complex and rich, and the amount of data we have collected is far from sufficient for true world editing. While the integration of MLLM with our method realizes world-instructed editing in a broader range of scenarios, efficient editing still requires a substantial amount of high-quality image editing training data. In addition, our data lacks a large number of precise editing examples. For instance, in a picture with eight candles arranged, given the instruction "blow out the third candle", accurate implementation of such image editing is almost non-existent in the data we have collected. Therefore, precise world-instructed image editing in complex pictures is still a challenging problem. Hence, our future work will focus on further enriching the EditWorld dataset and increasing the number of precise editing data samples.

##### Discussion.

Current large-scale multimodal models, such as LLava and GPT-4V, have demonstrated efficacy in answering questions based on content derived from videos and images. However, these models encounter difficulties when tasked with distinguishing differences between images or videos. For instance, they struggle to describe the differences or dynamics between input and output images within the dataset we have proposed. This issue primarily stems from the need for large-scale data to analyze differences between images or videos. The task and data proposed in EditWorld can contribute to addressing the challenge of multimodal difference recognition. In summary, both world-instructed image editing task and the recognition of differences in images or videos are indispensable parts in the pursuit of Artificial General Intelligence.

[^1]: Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in ICLR, 2020.

[^2]: J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[^3]: L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, and M.-H. Yang, “Diffusion models: A comprehensive survey of methods and applications,” ACM Computing Surveys, vol. 56, no. 4, pp. 1–39, 2023.

[^4]: R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.

[^5]: A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022.

[^6]: J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al., “Improving image generation with better captions,” Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, vol. 2, no. 3, p. 8, 2023.

[^7]: L. Yang, Z. Yu, C. Meng, M. Xu, S. Ermon, and B. Cui, “Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms,” ICML, 2024.

[^8]: K. Crowson, S. Biderman, D. Kornis, D. Stander, E. Hallahan, L. Castricato, and E. Raff, “Vqgan-clip: Open domain image generation and editing with natural language guidance,” in ECCV, 2022.

[^9]: R. Gal, O. Patashnik, H. Maron, A. H. Bermano, G. Chechik, and D. Cohen-Or, “Stylegan-nada: Clip-guided domain adaptation of image generators,” TOG, 2022.

[^10]: A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, “Prompt-to-prompt image editing with cross attention control,” arXiv preprint arXiv:2208.01626, 2022.

[^11]: J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in ICCV, 2017.

[^12]: K. Sohn, N. Ruiz, K. Lee, D. C. Chin, I. Blok, H. Chang, J. Barber, L. Jiang, G. Entis, Y. Li, et al., “Styledrop: Text-to-image generation in any style,” arXiv preprint arXiv:2306.00983, 2023.

[^13]: Y. Zhang, J. Xing, E. Lo, and J. Jia, “Real-world image variation by aligning diffusion inversion chain,” in NeurIPS, 2024.

[^14]: H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, “Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models,” arXiv preprint arXiv:2308.06721, 2023.

[^15]: Y. Zhang, J. Liu, Y. Song, R. Wang, H. Tang, J. Yu, H. Li, X. Tang, Y. Hu, H. Pan, et al., “Ssr-encoder: Encoding selective subject representation for subject-driven generation,” in CVPR, 2024.

[^16]: J. Nam, H. Kim, D. Lee, S. Jin, S. Kim, and S. Chang, “Dreammatcher: Appearance matching self-attention for semantically-consistent text-to-image personalization,” in CVPR, 2024.

[^17]: X. Pan, A. Tewari, T. Leimkühler, L. Liu, A. Meka, and C. Theobalt, “Drag your gan: Interactive point-based manipulation on the generative image manifold,” in SIGGRAPH, 2023.

[^18]: Y. Shi, C. Xue, J. Pan, W. Zhang, V. Y. Tan, and S. Bai, “Dragdiffusion: Harnessing diffusion models for interactive point-based image editing,” arXiv preprint arXiv:2306.14435, 2023.

[^19]: C. Mou, X. Wang, J. Song, Y. Shan, and J. Zhang, “Dragondiffusion: Enabling drag-style manipulation on diffusion models,” in ICLR, 2024.

[^20]: O. Avrahami, O. Fried, and D. Lischinski, “Blended latent diffusion,” TOG, 2023.

[^21]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021.

[^22]: O. Bar-Tal, D. Ofri-Amar, R. Fridman, Y. Kasten, and T. Dekel, “Text2live: Text-driven layered image and video editing,” in ECCV, 2022.

[^23]: R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or, “Null-text inversion for editing real images using guided diffusion models,” arXiv preprint arXiv:2211.09794, 2022.

[^24]: N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel, “Plug-and-play diffusion features for text-driven image-to-image translation,” in CVPR, 2023.

[^25]: G. Couairon, J. Verbeek, H. Schwenk, and M. Cord, “Diffedit: Diffusion-based semantic image editing with mask guidance,” arXiv preprint arXiv:2210.11427, 2022.

[^26]: L. Yang, Z. Zhang, Z. Yu, J. Liu, M. Xu, S. Ermon, and B. CUI, “Cross-modal contextualized diffusion models for text-guided visual generation and editing,” in ICLR, 2024.

[^27]: T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in CVPR, 2023.

[^28]: K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su, “Magicbrush: A manually annotated dataset for instruction-guided image editing,” arXiv preprint arXiv:2306.10012, 2023.

[^29]: S. Li, B. Zeng, Y. Feng, S. Gao, X. Liu, J. Liu, L. Lin, X. Tang, Y. Hu, J. Liu, et al., “Zone: Zero-shot instruction-guided local editing,” arXiv preprint arXiv:2312.16794, 2023.

[^30]: Y. Huang, L. Xie, X. Wang, Z. Yuan, X. Cun, Y. Ge, J. Zhou, C. Dong, R. Huang, R. Zhang, et al., “Smartedit: Exploring complex instruction-based image editing with multimodal large language models,” arXiv preprint arXiv:2312.06739, 2023.

[^31]: T.-J. Fu, W. Hu, X. Du, W. Y. Wang, Y. Yang, and Z. Gan, “Guiding instruction-based image editing via multimodal large language models,” arXiv preprint arXiv:2309.17102, 2023.

[^32]: D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, 2018.

[^33]: D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” in ICLR, 2024.

[^34]: L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in ICCV, 2023.

[^35]: B. Lin, B. Zhu, Y. Ye, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” arXiv preprint arXiv:2311.10122, 2023.

[^36]: J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[^37]: P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” in NeurIPS, 2021.

[^38]: A. Vahdat, K. Kreis, and J. Kautz, “Score-based generative modeling in latent space,” in NeurIPS, 2021.

[^39]: J. Ho, C. Saharia, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans, “Cascaded diffusion models for high fidelity image generation,” JMLR, 2022.

[^40]: W. Peebles and S. Xie, “Scalable diffusion models with transformers,” arXiv:2212.09748, 2022.

[^41]: L. Yang, H. Qian, Z. Zhang, J. Liu, and B. Cui, “Structure-guided adversarial training of diffusion models,” in CVPR, 2024.

[^42]: X. Zhang, L. Yang, Y. Cai, Z. Yu, J. Xie, Y. Tian, M. Xu, Y. Tang, Y. Yang, and B. Cui, “Realcompo: Dynamic equilibrium between realism and compositionality improves text-to-image diffusion models,” arXiv preprint arXiv:2402.12908, 2024.

[^43]: L. Yang, J. Liu, S. Hong, Z. Zhang, Z. Huang, Z. Cai, W. Zhang, and C. Bin, “Improving diffusion-based image synthesis with context prediction,” in NeurIPS, 2023.

[^44]: Y. Huang, J. Huang, Y. Liu, M. Yan, J. Lv, J. Liu, W. Xiong, H. Zhang, S. Chen, and L. Cao, “Diffusion model-based image editing: A survey,” arXiv preprint arXiv:2402.17525, 2024.

[^45]: B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani, “Imagic: Text-based real image editing with diffusion models,” in CVPR, 2023.

[^46]: J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan, et al., “Scaling autoregressive models for content-rich text-to-image generation,” arXiv preprint arXiv:2206.10789, 2022.

[^47]: C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon, “Sdedit: Guided image synthesis and editing with stochastic differential equations,” arXiv preprint arXiv:2108.01073, 2021.

[^48]: C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al., “Photorealistic text-to-image diffusion models with deep language understanding,” in NeurIPS, 2022.

[^49]: A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” in ICML, 2022.

[^50]: N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation,” in CVPR, 2023.

[^51]: B. Zeng, S. Li, X. Liu, S. Gao, X. Jiang, X. Tang, Y. Hu, J. Liu, and B. Zhang, “Controllable mind visual diffusion model,” in AAAI, 2024.

[^52]: S. Gao, X. Liu, B. Zeng, S. Xu, Y. Li, X. Luo, J. Liu, X. Zhen, and B. Zhang, “Implicit diffusion models for continuous super-resolution,” in CVPR, 2023.

[^53]: G. Kim, T. Kwon, and J. C. Ye, “Diffusionclip: Text-guided diffusion models for robust image manipulation,” in CVPR, 2022.

[^54]: G. Kwon and J. C. Ye, “Clipstyler: Image style transfer with a single text condition,” in CVPR, 2022.

[^55]: O. Avrahami, D. Lischinski, and O. Fried, “Blended diffusion for text-driven editing of natural images,” in CVPR, 2022.

[^56]: G. Parmar, K. Kumar Singh, R. Zhang, Y. Li, J. Lu, and J.-Y. Zhu, “Zero-shot image-to-image translation,” in SIGGRAPH, 2023.

[^57]: J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[^58]: V. Goel, E. Peruzzo, Y. Jiang, D. Xu, N. Sebe, T. Darrell, Z. Wang, and H. Shi, “Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models,” arXiv preprint arXiv:2303.17546, 2023.

[^59]: A. El-Nouby, S. Sharma, H. Schulz, D. Hjelm, L. E. Asri, S. E. Kahou, Y. Bengio, and G. W. Taylor, “Tell, draw, and repeat: Generating and modifying images based on continual linguistic instruction,” in CVPR, 2019.

[^60]: T. Zhang, H.-Y. Tseng, L. Jiang, W. Yang, H. Lee, and I. Essa, “Text as neural operator: Image manipulation by text instruction,” in ACMMM, 2021.

[^61]: L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al., “Training language models to follow instructions with human feedback,” in NeurIPS, 2022.

[^62]: S. Zhang, X. Yang, Y. Feng, C. Qin, C.-C. Chen, N. Yu, Z. Chen, H. Wang, S. Savarese, S. Ermon, et al., “Hive: Harnessing human feedback for instructional visual editing,” arXiv preprint arXiv:2303.09618, 2023.

[^63]: T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., “Language models are few-shot learners,” in NeurIPS, 2020.

[^64]: A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., “Segment anything,” arXiv preprint arXiv:2304.02643, 2023.

[^65]: J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, Y. Shi, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou, “Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation,” in CVPR, 2023.

[^66]: M. Geyer, O. Bar-Tal, S. Bagon, and T. Dekel, “Tokenflow: Consistent diffusion features for consistent video editing,” in ICLR, 2023.

[^67]: B. Zeng, X. Liu, S. Gao, B. Liu, H. Li, J. Liu, and B. Zhang, “Face animation with an attribute-guided diffusion model,” arXiv preprint arXiv:2304.03199, 2023.

[^68]: B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-to-3d using 2d diffusion,” in ICLR, 2022.

[^69]: C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3d: High-resolution text-to-3d content creation,” in CVPR, 2023.

[^70]: B. Zeng, S. Li, Y. Feng, H. Li, S. Gao, J. Liu, H. Li, X. Tang, J. Liu, and B. Zhang, “Ipdreamer: Appearance-controllable 3d object generation with image prompts,” arXiv preprint arXiv:2310.05375, 2023.

[^71]: D. Li, H. Ling, S. W. Kim, K. Kreis, S. Fidler, and A. Torralba, “Bigdatasetgan: Synthesizing imagenet with pixel-wise annotations,” in CVPR, 2022.

[^72]: W. Peebles, J.-Y. Zhu, R. Zhang, A. Torralba, A. A. Efros, and E. Shechtman, “Gan-supervised dense visual alignment,” in CVPR, 2022.

[^73]: N. Tritrong, P. Rewatbowornwong, and S. Suwajanakorn, “Repurposing gans for one-shot semantic part segmentation,” in CVPR, 2021.

[^74]: Y. Viazovetskyi, V. Ivashkin, and E. Kashin, “Stylegan2 distillation for feed-forward image manipulation,” in ECCV, 2020.

[^75]: Y. Jin, M. Chandra, G. Verma, Y. Hu, M. De Choudhury, and S. Kumar, “Better to ask in english: Cross-lingual evaluation of large language models for healthcare queries,” in Proceedings of the ACM on Web Conference, 2024.

[^76]: Y. Jin, M. Choi, G. Verma, J. Wang, and S. Kumar, “Mm-soc: Benchmarking multimodal large language models in social media platforms,” in ACL, 2024.

[^77]: Y. Wang, Y. He, Y. Li, K. Li, J. Yu, X. Ma, X. Li, G. Chen, X. Chen, Y. Wang, et al., “Internvid: A large-scale video-text dataset for multimodal understanding and generation,” in ICLR, 2024.

[^78]: D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[^79]: R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.