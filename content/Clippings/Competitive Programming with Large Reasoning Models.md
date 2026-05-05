---
title: "Competitive Programming with Large Reasoning Models"
source: "https://arxiv.org/html/2502.06807v2"
author:
published:
created: 2026-04-30
description:
tags:
  - "clippings"
pdf: "file:///Users/chenshi/Desktop/paper/03_Agent%E4%B8%8E%E6%8E%A8%E7%90%86/Competitive%20Programming%20with%20Large%20Reasoning%20Models%2C%20%20OpenAI%20et%20al.%2C%202025.no_watermark.zh-CN.dual.pdf"
---
OpenAI Contributions listed in Appendix A

###### Abstract

We show that reinforcement learning applied to large language models (LLMs) significantly boosts performance on complex coding and reasoning tasks. Additionally, we compare two general-purpose reasoning models — OpenAI o1 and an early checkpoint of o3 — with a domain-specific system, o1-ioi, which uses hand-engineered inference strategies designed for competing in the 2024 International Olympiad in Informatics (IOI). We competed live at IOI 2024 with o1-ioi and, using hand-crafted test-time strategies, placed in the 49th percentile. Under relaxed competition constraints, o1-ioi achieved a gold medal. However, when evaluating later models such as o3, we find that o3 achieves gold without hand-crafted domain-specific strategies or relaxed constraints. Our findings show that although specialized pipelines such as o1-ioi yield solid improvements, the scaled-up, general-purpose o3 model surpasses those results without relying on hand-crafted inference heuristics. Notably, o3 achieves a gold medal at the 2024 IOI and obtains a CodeForces rating on par with elite human competitors. Overall, these results indicate that scaling general-purpose reinforcement learning, rather than relying on domain-specific techniques, offers a robust path toward state-of-the-art AI in reasoning domains, such as competitive programming.

## 1 Introduction

Competitive programming is widely recognized as a challenging benchmark for evaluating reasoning and coding proficiency [^2]. Solving complex algorithmic problems demands advanced computational thinking and problem solving skills. Moreover, these problems are also objectively gradable, making it an ideal testbed to assess the reasoning capabilities of AI systems.

Recent work on program synthesis with large language models [^1] has demonstrated that even relatively general models, ranging from 244M to 137B parameters, can generate short Python scripts from natural language instructions. Importantly, performance improves log-linearly with model size, and fine-tuning significantly boosts accuracy. Concurrently, Codex [^2], an early code-focused LLM, excelled at Python program generation and powered GitHub Copilot. Further progress came from AlphaCode [^7], which tackled competitive programming tasks using large-scale code generation and heuristics at inference, and the subsequent AlphaCode2 [^6], whose improvements nearly doubled AlphaCode’s solved problems and placed it in the 85th percentile on the CodeForces platform. Both AlphaCode systems used large-scale sampling of up to a million candidate solutions per problem before selecting their top 10 submissions with a hand-engineered test-time strategy.

Since then, significant progress has been made in harnessing reinforcement learning to improve LLMs’ reasoning skills. This has led to the emergence of large reasoning models (LRMs): language models trained via reinforcement learning to “reason” and “think through” extended chains of thought. In particular, OpenAI’s o1 [^4] [^12] and its soon-to-be-released successor o3 [^13] use chain-of-thought reasoning to tackle intricate tasks such as mathematics and coding. Work by DeepSeek-R1 [^3] and Kimi k1.5 [^15] independently illustrates how learning chain-of-thought boosts performance on both mathematical and programming challenges.

An open question is how domain-specific, hand-engineered inference strategies compare to learned approaches that models generate and execute on their own. We have three systems available that can shed light on this question: o1, o1-ioi, and early checkpoints of o3. OpenAI o1 was the first large reasoning model and used general purpose methods to improve programming performance. Building on this foundation, o1-ioi was a fine-tuned system tailored to compete in the 2024 International Olympiad in Informatics (IOI) and used test-time strategies similar to those used in the AlphaCode system. This specialization led to strong performance improvements on both the 2024 IOI and competitive programming platforms such as CodeForces. Subsequent advances led to the development of o3, which has significantly advanced the reasoning capabilities of AI models. Unlike o1-ioi or AlphaCode, o3 does not depend on coding-specific test-time strategies defined by humans. Instead, we found that complex test-time reasoning strategies emerged naturally from end-to-end RL, leading to unprecedented performance on competitive programming benchmarks.

This report provides a high-level overview of the importance of reasoning in coding tasks such as competitive programming, the progress of OpenAI’s large reasoning models in programming ability, and our evaluation methodology and results on various competitive programming and coding benchmarks.

## 2 OpenAI o1

We start with OpenAI o1, a large language model trained with reinforcement learning to tackle complex reasoning tasks. By generating an extended internal chain of thought before answering [^16], o1 resembles a human who methodically works through a challenging problem step by step. Reinforcement learning refines this chain-of-thought process, helping the model identify and correct errors, break down complex tasks into manageable parts, and explore alternate solution paths when an approach fails. These in-context reasoning capabilities substantially boost o1’s overall performance on a wide range of tasks.

Additionally, OpenAI o1 is trained to use external tools [^14], especially for writing and executing code in a secure environment.<sup>1</sup> This capability lets o1 verify whether its generated code compiles, passes provided test cases, and meets other correctness checks. By testing and refining its outputs, o1 iteratively improves its solutions over the course of a single sample.

### 2.1 CodeForces Benchmark

CodeForces is a programming competition website that hosts live contests. It is internationally competitive and frequented by some of the best competitive programmers in the world.

To assess our models’ competitive programming abilities, we simulated CodeForces contests under conditions that closely mirrored real competitions. This included using the full test suite for each problem and enforcing appropriate time and memory constraints for solutions.

Our evaluation focused on Division 1 contests from 2024 and December 2023, ensuring all test contests occurred after the data cut-off for both pretraining and RL. Additionally, we conducted a contamination check as a sanity measure, leveraging the OpenAI embedding API to verify that test problems had not been seen during training.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x1.png)

Figure 1: Comparing reasoning LLMs OpenAI o1-preview and o1 to gpt-4o on CodeForces.

We compared o1 against a non-reasoning LLM (gpt-4o) and an earlier reasoning model (o1-preview). Figure 1 shows how both o1-preview and o1 dramatically outperform gpt-4o, highlighting the effectiveness of reinforcement learning for complex reasoning. The o1-preview model achieved a CodeForces rating of 1258 (62nd percentile) — up from gpt-4o’s 808 (11th percentile). Further training pushed o1’s rating to 1673 (89th percentile), establishing a new milestone for AI performance in competitive programming.

In Appendix B we provide additional details of which problems our models can solve and how ratings were calculated.

## 3 OpenAI o1-ioi

During our development and evaluation of OpenAI o1, we found that increasing both the amount of reinforcement learning (RL) compute and test-time inference compute consistently improved model performance.

![Refer to caption](https://arxiv.org/html/2502.06807v2/extracted/6215171/figures/compute.png)

Figure 2: Additional RL training and additional test-time compute improves competitive mathematics performance.

As shown in Figure 2, scaling RL training and extending test-time inference led to marked gains, highlighting the importance of optimizing these two compute dimensions to push performance beyond conventional LLM pretraining.

Building on these insights, we created the o1-ioi system for competing at the 2024 International Olympiad in Informatics (IOI). In addition to continued RL training targeted at coding tasks, o1-ioi incorporates specialized test-time inference strategies engineered for competitive programming.

### 3.1 Coding RL Fine-tuning

Our first step extended the reinforcement learning phase of OpenAI o1, focusing on coding tasks. By dedicating additional training compute to programming problems, we bolstered the model’s ability to plan, implement, and debug more involved solutions. Concretely:

1. We resumed RL training from the OpenAI o1 checkpoint.
2. We specifically emphasized challenging programming problems, helping the model improve C++ generation and runtime checks.
3. We guided the model to produce outputs in the IOI submission format.

This added focus on coding allowed o1-ioi to write and execute C++ programs during inference. The model improved its reasoning by iteratively running and refining solutions, thereby strengthening both its coding and problem-solving skills.

### 3.2 o1-ioi Test-time Strategy

At a high level, we divided each IOI problem into its constituent subtasks, sampled 10,000 solutions from o1-ioi for each subtask, and then employed a clustering- and reranking-based approach to decide which solutions from this set to submit.

##### Problem formulation

For o1-ioi we chose to attempt to solve the individual subtasks of each problem separately, as the scoring for IOI is done on a subtask-by-subtask basis and gives each competitor the maximum score over all of their attempts on each subtask. To do this, we divided each IOI problem into its composite subtasks (using the divisions laid out in the scoring guide for each problem). This was done simply by creating one version of the document for each subtask with the information about the other subtasks removed.

##### Clustering

We clustered the generated solutions based on their outputs on model-generated test inputs. For each subtask, we first prompted the model to write random test input generators in C++ given the problem specification and subtask. We used these generators to generate 256 random test inputs. To ensure the validity of these test inputs, we then prompted the model to write test input validators in C++ that check, given a test input, whether it satisfies the subtask constraints. Finally, we accepted each test input that passes at least 75% of the validators. For each subtask, we generated 256 of these random test case inputs, and then clustered based on their outputs for these test cases. Any programs that matched each other’s outputs on all test inputs would be placed in the same cluster.

##### Reranking

We then implemented the reranking core of our test-time compute strategy. We scored each solution based on:

- The quality of the solution according to a learned scoring function.
- Errors on model-generated test inputs.
- Failing the provided public test cases.

Each cluster was given a score defined as the average score of the samples it contained minus a penalty for each time a sample submission was attempted from that cluster. The weights of all of these penalties were tuned by random search on solutions to previous years’ IOI problems, by directly simulating the submission process.

##### Submission

We then submitted up to 50 (the maximum number allowed for human competitors) of these solutions in a round-robin fashion over subtasks, starting from the hardest. We selected the top-ranked solution in the top-ranked cluster for each given subtask. When a subtask was solved (meaning that the maximum score was attained), we ceased sampling on that subtask. When submitting solutions to any subtask that was a strict superset of a solved subtask, we would filter out any solutions that did not match the outputs on test inputs of the solved constituent subtasks, allowing us to rapidly narrow down candidate solutions on harder subtasks by rejecting those that would almost certainly have failed easier subtasks.

### 3.3 CodeForces Benchmark

Once again, we simulated CodeForces contests to evaluate o1-ioi’s coding abilities, closely mirroring contest conditions with the complete test suite for each problem and appropriate time and memory restrictions for solutions.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x2.png)

Figure 3: Further training OpenAI o1 on coding tasks and incorporating test-time strategies improves performance.

Figure 3 shows that o1-ioi reached a CodeForces rating of 1807, outperforming 93% of competitors — demonstrating clear improvements from additional RL training on coding tasks. When we applied a simple filter rejecting any solution that failed public tests, the rating rose to 2092 (96th percentile). Our complete test-time strategy pushed performance even further, attaining a rating of 2214 (98th percentile). These results confirm that domain-specific RL fine-tuning paired with advanced selection heuristics can significantly boost competitive programming outcomes.

### 3.4 IOI 2024 Live Competition

![Refer to caption](https://arxiv.org/html/2502.06807v2/x3.png)

Figure 4: Performance of o1-ioi competing at IOI 2024.

The o1-ioi system participated in the 2024 International Olympiad in Informatics (IOI) under the same conditions as human contestants. It had ten hours to solve six challenging algorithmic problems and was allowed up to 50 submissions per problem. We show the results in Figure 4.

During the competition, our system generated 10,000 candidate solutions for each problem, and selected 50 submissions using our test-time selection strategy. This strategy prioritized submissions based on their performance on the IOI public test cases, model-generated test cases, and a learned scoring function. The model scored 213 points, placing it in the 49th percentile of the competition.

In comparison, selecting 50 random submissions would have yielded an average score of only 156 points, indicating that the selection strategy contributed nearly 60 additional points under the competition’s constraints.

When the submission limit was relaxed to 10,000 per problem, the model’s performance improved dramatically. Without employing any test-time selection strategy, it achieved a score of 362.14, surpassing the gold medal threshold and demonstrating the model’s potential. We show samples that yielded the 362.14 score in Appendix C.

## 4 OpenAI o3

Building on the insights gained from o1 and o1-ioi, we explore the limits of reinforcement learning (RL) training alone, without relying on human-engineered test-time strategies. While o1-ioi achieved strong results by combining additional RL fine-tuning with carefully designed test-time inference pipelines, its success hinged on human intervention to define and implement these strategies. We sought to explore the performance of a model even further trained with RL with the ability to autonomously develop and execute its own test-time reasoning strategies. To this end, we obtained access to early checkpoints of o3 [^13] to evaluate on competitive programming tasks.

### 4.1 CodeForces Benchmark

We evaluate an early checkpoint of the o3 model on our CodeForces benchmark set, where each prompt includes the problem description, constraints, and any available sample test cases.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x4.png)

Figure 5: Performance of OpenAI o3 on the CodeForces benchmark.

As shown in Figure 5, further RL training provided a significant improvement over both o1 and the full o1-ioi system. Notably, the transition from the o1-ioi model to o3 resulted in a rating increase from 2214 (98th percentile) to 2724 (99.8th percentile), reflecting a substantial leap in competitive programming performance. This improvement demonstrates o3’s ability to solve a wider range of complex algorithmic problems with higher reliability, pushing its capabilities closer to top-tier human competitors on CodeForces.

![Refer to caption](https://arxiv.org/html/2502.06807v2/extracted/6215171/figures/brute_force.png)

Figure 6: o3 testing its own solution. This reflects a sophisticated reasoning strategy that partially implements the hand-designed test-time strategy used for o1-ioi in IOI 2024.

In addition to its significantly improved problem-solving capabilities, we observe that o3 demonstrates more insightful and deliberate chains of thought. The model not only writes and executes code to validate its solutions against public test cases, it also refines its approach based on these verifications. Figure 6 shows an advanced test-time strategy discovered by o3: for problems where verification is nontrivial, it often writes simple brute-force solutions — trading efficiency for correctness — then cross-checks the outputs against its more optimized algorithmic implementations. This self-imposed validation mechanism lets o3 catch potential errors and improve the reliability of its solutions.

### 4.2 IOI 2024 Benchmark

Although we competed in IOI 2024 using o1-ioi, we retrospectively evaluated a checkpoint of o3 on the same six IOI 2024 problems to compare performance under identical conditions. As with o1-ioi, we strictly adhered to the official IOI rules, which permit a maximum of 50 submissions per problem.

The o3 results on the IOI 2024 were produced by a later version of o3 than the CodeForces results, and included additional fresher training data. IOI 2024 occurred after the training cut-off for this model, and we additionally confirmed with search that the IOI test problems are not contaminated with the training set.

##### Sampling Approach.

Unlike o1-ioi, which sampled solutions separately for each subtask, we adopted a different approach when evaluating o3: sampling from a single prompt containing the original problem statement. Additionally, while o1-ioi generated 10K solutions per subtask, for o3 we sampled only 1K solutions per problem.

Selection strategies also differed between the two models. Whereas o1-ioi relied on a complex, human-defined test-time strategy (3.2) to select solutions, o3 followed a much simpler approach. Specifically, we selected the top 50 solutions with the highest test-time compute from 1,024 samples per problem. Despite this streamlined method, o3 produced robust solutions capable of covering many, if not all, subtasks — without the need for subtask-specific prompts, manual partitioning, or intricate submission strategies.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x5.png)

Figure 7: IOI 2024 scores under different submission strategies. Even without human-engineered heuristics or relaxed submission limits, o3 outperforms o1-ioi and surpasses the gold threshold with just 50 submissions.

##### Results.

Figure 7 presents the final scores. The IOI scoring system is subtask-based, with a maximum total of 600 points in the 2024 contest. The gold medal threshold was approximately 360 points. Key results include:

- o1-ioi scored 213 points with 50 submissions, improving to 362.14 points with 10K submissions, just above the gold medal cutoff.
- o3 achieved 395.64 points, surpassing the gold threshold even under the 50-submission limit.

These results demonstrate that o3 outperforms o1-ioi without relying on IOI-specific, hand-crafted test-time strategies. Instead, the sophisticated test-time techniques that emerged during o3 training, such as generating brute-force solutions to verify outputs, served as a more than adequate replacement and eliminated the need for the hand-engineered clustering and selection pipelines required by o1-ioi.

Overall, the IOI 2024 findings confirm that large-scale RL training alone can achieve state-of-the-art coding and reasoning performance. By independently learning to generate, evaluate, and refine solutions, o3 surpasses o1-ioi without dependence on domain-specific heuristics or clustering-based methods.

## 5 Software Engineering Evaluations

We have demonstrated how reasoning significantly enhances LLM performance in competitive programming, where solving complex algorithmic challenges requires deep logical thinking. However, we also sought to evaluate the impact of reasoning on real-world coding tasks. To this end, we tested our models on two datasets: the HackerRank Astra <sup>2</sup> dataset and SWE-bench verified <sup>3</sup> [^5] [^11].

### 5.1 HackerRank Astra

The HackerRank Astra dataset is composed of 65 project-oriented coding challenges, each crafted to simulate real-world software development tasks. These challenges cover a range of frameworks, including React.js, Django, and Node.js, allowing for hands-on experience in building features and applications.

What sets this dataset apart is its focus on assessing problem-solving skills in complex, multi-file, long-context scenarios that mirror actual development environments. Unlike typical competitive programming datasets, HackerRank Astra does not provide public test cases, which prevents us from relying on hand-crafted test-time tactics. Evaluating performance with this dataset reveals whether reasoning abilities enhance success in algorithmic problem solving alone, or extend to more practical, industry-related coding tasks.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x6.png)

Figure 8: HackerRank Astra evaluation.

Figure 8 presents performance metrics such as pass@1 (the probability of successfully completing a task on the first attempt) and average scores (the mean proportion of test cases passed). The results illustrate the impact of chain-of-thought reasoning, with the o1-preview model achieving a 9.98% improvement in pass@1 and a 6.03-point gain in average score compared to GPT-4o. Further fine-tuning through reinforcement learning enhances o1’s performance, yielding a pass@1 of 63.92% and an average score of 75.80%—a 3.03% increase in pass@1 over o1-preview. These metrics demonstrate o1’s enhanced reasoning and adaptability, enabling it to address complex, industry-relevant software development tasks effectively.

### 5.2 SWE-Bench Verified

SWE-bench Verified is OpenAI’s preparedness team’s human-validated subset of SWE-bench that more reliably evaluates AI models’ ability to solve real-world software issues. This validated set of 500 tasks fixes certain issues with SWE-bench such as incorrect grading of correct solutions, under-specified problem statements, and overly specific unit tests. This helps ensure the benchmark accurately grades model capabilities.

To illustrate performance on this software task, we display the results presented in the o1 system card [^4] as well as results from an early o3 checkpoint [^13]. Because o1-preview was not trained to use code execution or file editing tools, the best-performing open-source scaffold at the time of initial implementation, Agentless was used. Unlike for IOI, no specialized test-time strategies was used for SWE-Bench verified. All models are given 5 tries to generate a candidate patch. If the model fails after 5 attempts, it is considered an incorrect attempt. All evaluations are averaged over 3 trials. We do not penalize the model for system failures (e.g., container hangs or grading failures), and we retry these rollouts until we can record a valid attempt.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x7.png)

Figure 9: SWE-bench evaluation.

As illustrated in Figure 9, o1-preview demonstrates an 8.1% performance improvement on SWE-bench compared to gpt-4o, showcasing notable advancements in reasoning capabilities. With additional reinforcement learning compute applied during training, o1 achieves a further 8.6% improvement. Notably, o3, which was trained with significantly greater compute resources than o1, delivers an impressive 22.8% improvement over o1. These results underscore that enhanced reasoning skills extend beyond competitive programming challenges, proving their applicability to real-world tasks like software engineering.

## 6 Conclusion

Through the o-series large reasoning models, we demonstrate that chain-of-thought reasoning is a powerful strategy for improving performance in coding tasks, from competitive programming benchmarks such as CodeForces and IOI to complex software engineering challenges like SWE-bench and Astra. Our findings highlight that increasing reinforcement learning training compute, coupled with enhanced test-time compute, consistently boosts model performance to nearly match the best humans in the world. Given these results, we believe o-series large reasoning models will unlock many new use cases for AI in science, coding, math, and many other fields.

## Appendix A Authorship, credit attribution, and acknowledgments

##### Data Preparation:

Borys Minaiev, Ignasi Clavera, Lorenz Kuhn, Nat McAleese, Oleg Mürk, Szymon Sidor

##### IOI Model Training:

Ahmed El-Kishky, Mostafa Rohaninejad

##### Sampling Infrastructure:

Andre Saraiva, Hunter Lightman, Vineet Kosaraju, Wenda Zhou

##### Test-time Strategy:

Alexander Wei, Daniel Selsam, David Dohan, Francis Song, Ignasi Clavera, Max Schwarzer, Rhythm Garg, Rui Shu

##### Evaluation:

Andre Saraiva, Ignasi Clavera, Lorenz Kuhn, Nat McAleese

##### Leadership:

Jakub Pachocki, Jerry Tworek, Lukasz Kaiser, Mark Chen

##### o3 Model Development

o3 contributors [^13].

##### Acknowledgments:

We are grateful to the IOI committee for allowing us to enter our model, o1-ioi, in the 2024 International Olympiad in Informatics. We also extend our thanks to Wael Ewida, a member of the IOI technical committee, for hosting a portal that enabled us to submit our solutions under the same conditions as the contestants. Additionally, we appreciate the support of those who contributed to and maintained our sandboxed code execution, including Taylor Gordon, Oleg Boiko, John Rizzo, Paul Ashbourne, Leo Liu, Alexander Prokofiev, and Scottie Yan. We also extend our gratitude to Chris Orsinger and Michelle Fradin for their contributions to data efforts. Finally, we would like to express our sincere gratitude to everyone involved in the reinforcement learning reasoning efforts for o1 and o3, whose dedication and expertise were instrumental in advancing this work.

## Appendix B Additional CodeForces Details

In order to compare our models to human competitive programmers, we simulate contests. This section provides details of how the simulation is performed, how the overall score and ratings are calculated, as well as the per-contest results.

### B.1 Data

For our test set we use “Division 1” contests from late 2023 and 2024, all of which occurred after the o3 training set data cut-off. As a redundant additional check, we used embedding search to confirm that the test problems have not been seen by the model during training. We excluded one contest that contained an interactive problem for which grading was inconvenient, but otherwise included all post-cut-off Division 1 problems to which we had access at the time. During training we used a validation set of primarily Division 2 problems; when that set indicated that performance was very strong we built and evaluated the Division 1 set presented here.

### B.2 Grading

We run the complete set of tests for each problem, and have confirmed that our test environment closely matches the official CodeForces grading service, including by manually submitting solutions for the hardest problems to the official CodeForces graders.

Following AlphaCode [^6] we allow the model to make 10 independent submissions against the full test set and mark a problem as solved if any one of those 10 passes. This is close to but not strictly the same as the human affordance, as human participants see only the results of the pre-tests during the competition. However in Division 1 contests the pre-tests are typically “strong” (highly correlated with full tests), and in our results the number of failures before a passing submission is typically small (see LABEL:tab:cf-details). We did not have access to labels for which test cases were pre-tests.

### B.3 Thinking Time

Competitors receive a higher score for submitting their solutions faster. Because models can think in parallel and simultaneously attempt all problems, they have an innate advantage over humans. We elected to reduce this advantage in our primary results by estimating o3’s score for each solved problem as the median of the scores of the human participants that solved that problem in the contest with the same number of failed attempts.

We could instead use the model’s real thinking time to compute ratings. o3 uses a learned scoring function for test-time ranking in addition to a chain of thought. This process is perfectly parallel and true model submission times therefore depend on the number of available GPU during the contest. On a very large cluster the time taken to pick the top-ranked solutions is (very slightly more than) the maximum over the thinking times for each candidate submission. Using this maximum parallelism assumption and the sequential o3 sampling speed would result in a higher estimated rating than presented here. We note that because sequential test-time compute has grown rapidly since the early language models, it was not guaranteed that models would solve problems quickly compared to humans, but in practice o3 does.

### B.4 Estimated Rating

The CodeForces rating system is described by the creator in three blog posts [^8] [^9] [^10]. Ratings are similar to the Elo system and satisfy the property that if competitor $A$ has rating $R_{A}$ and competitor $B$ has rating $R_{B}$ then the probability that $A$ ranks better than $B$ any final contest standings is estimated as

$$
\frac{1}{10^{\frac{R_{B}-R_{A}}{400}}}
$$

To find the model rating we first calculate the rank of the model in each of the test contest from the total contest score (described above) and then directly maximize the likelihood of the observed rankings and human ratings with respect to the model rating using the equation above. We average to ensure that contests with more participants are not over-weighted.

We validated that this recovers known human ratings based on their contest performance and also gives similar values to linearly predicting participant rating from their average solve rate.

### B.5 Percentile performance

Codeforces maintains a global leaderboard of active participants, and an estimated rating can be used to compare to that group. We can also directly compare the solve rate of o3 in our test contests to the other participants in those contests. Figure 10 shows both these comparisons. Each point is a person that competed in at least 8 of the test contests. We show their average solve rate over contests that they entered against their rating, as well as the rating thresholds for key performance levels. The very best human competitors remain much stronger than o3, with solve rates in excess of 85%, but both ratings and solve rates indicate that o3 would rank among the top 200 active participants worldwide.

![Refer to caption](https://arxiv.org/html/2502.06807v2/x8.png)

Figure 10: o3 would place among the best human competitive programmers in the world. Here we show the average solve rate and current rating for participants that entered at least 8 of our 12 unseen test contests. Horizontal lines show performance thresholds from the global CodeForces leaderboard of active competitors. The very best humans still solve more problems than AI, for now.

### B.6 Per Problem Breakdown

Table 1: We estimate our CodeForces rating from simulated contest participation. Here we show a detailed breakdown of o3 performance per-problem.

<table><thead><tr><th>problem</th><th>problem rating</th><th>pass@1 (no ranking)</th><th>pass@10 (no ranking)</th><th># failed submissions</th><th>pass@10 (ranking 1162)</th></tr></thead><tbody><tr><td colspan="6">Contest 1909 - 23/Dec/23 - Pinely Round 3 (Div. 1 + Div. 2) score: 7,220</td></tr><tr><td>1909 A</td><td>800</td><td>1156 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1909 B</td><td>1200</td><td>1066 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1909 C</td><td>1400</td><td>1075 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1909 D</td><td>1900</td><td>1099 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1909 E</td><td>2400</td><td>703 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1909 F1</td><td>2200</td><td>57 / 1162</td><td>0.40</td><td>0</td><td>solved</td></tr><tr><td>1909 F2</td><td>2500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1909 G</td><td>3000</td><td>3 / 1162</td><td>0.03</td><td>0</td><td>not solved</td></tr><tr><td>1909 H</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1909 I</td><td>1900</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1916 - 30/Dec/23 - Good Bye 2023 score: 8,920</td></tr><tr><td>1916 A</td><td>800</td><td>1157 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1916 B</td><td>1000</td><td>1133 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1916 C</td><td>1200</td><td>1145 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1916 D</td><td>1700</td><td>483 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1916 E</td><td>2300</td><td>6 / 1162</td><td>0.05</td><td>0</td><td>solved</td></tr><tr><td>1916 F</td><td>2900</td><td>369 / 1162</td><td>0.98</td><td>2</td><td>solved</td></tr><tr><td>1916 G</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1916 H1</td><td>2700</td><td>1059 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1916 H2</td><td>2700</td><td>1045 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td colspan="6">Contest 1919 - 06/Jan/24 - Hello 2024 score: 6,214</td></tr><tr><td>1919 A</td><td>800</td><td>1161 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1919 B</td><td>800</td><td>1141 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1919 C</td><td>1400</td><td>499 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1919 D</td><td>2100</td><td>25 / 1162</td><td>0.20</td><td>2</td><td>solved</td></tr><tr><td>1919 E</td><td>2600</td><td>6 / 1162</td><td>0.05</td><td>1</td><td>solved</td></tr><tr><td>1919 F1</td><td>2300</td><td>1090 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1919 F2</td><td>2800</td><td>227 / 1162</td><td>0.89</td><td>0</td><td>solved</td></tr><tr><td>1919 G</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1919 H</td><td>2000</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1942 - 30/Mar/24 - CodeTON Round 8 (Div. 1 + Div. 2, Rated, Prizes!) score: 8,701</td></tr><tr><td>1942 A</td><td>800</td><td>1157 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1942 B</td><td>1100</td><td>1157 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1942 C1</td><td>1300</td><td>999 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1942 C2</td><td>1700</td><td>525 / 1162</td><td>1.00</td><td>1</td><td>solved</td></tr><tr><td>1942 D</td><td>2100</td><td>1061 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1942 E</td><td>2300</td><td>347 / 1162</td><td>0.97</td><td>0</td><td>solved</td></tr><tr><td>1942 F</td><td>2700</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1942 G</td><td>2800</td><td>239 / 1162</td><td>0.90</td><td>0</td><td>solved</td></tr><tr><td>1942 H</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1943 - 16/Mar/24 - Codeforces Round 934 (Div. 1) score: 3,427</td></tr><tr><td>1943 A</td><td>1300</td><td>116 / 1162</td><td>0.65</td><td>0</td><td>solved</td></tr><tr><td>1943 B</td><td>2000</td><td>1 / 1162</td><td>0.01</td><td>0</td><td>not solved</td></tr><tr><td>1943 C</td><td>2300</td><td>160 / 1162</td><td>0.77</td><td>0</td><td>solved</td></tr><tr><td>1943 D1</td><td>2400</td><td>848 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1943 D2</td><td>2800</td><td>14 / 1162</td><td>0.11</td><td>0</td><td>solved</td></tr><tr><td>1943 E1</td><td>2900</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1943 E2</td><td>3300</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1943 F</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1951 - 06/Apr/24 - Codeforces Global Round 25 score: 9,396</td></tr><tr><td>1951 A</td><td>900</td><td>1157 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1951 B</td><td>1200</td><td>1150 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1951 C</td><td>1400</td><td>1155 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1951 D</td><td>2000</td><td>875 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1951 E</td><td>2000</td><td>1009 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1951 F</td><td>2500</td><td>53 / 1162</td><td>0.37</td><td>0</td><td>solved</td></tr><tr><td>1951 G</td><td>3100</td><td>34 / 1162</td><td>0.26</td><td>0</td><td>solved</td></tr><tr><td>1951 H</td><td>3200</td><td>1 / 1162</td><td>0.01</td><td>0</td><td>not solved</td></tr><tr><td>1951 I</td><td>3200</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1965 - 27/Apr/24 - Codeforces Round 941 (Div. 1) score: 3,891</td></tr><tr><td>1965 A</td><td>1400</td><td>1143 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1965 B</td><td>1800</td><td>1064 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1965 C</td><td>2300</td><td>313 / 1162</td><td>0.96</td><td>0</td><td>solved</td></tr><tr><td>1965 D</td><td>2900</td><td>690 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1965 E</td><td>3100</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1965 F</td><td>3300</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1967 - 30/Apr/24 - Codeforces Round 942 (Div. 1) score: 3,871</td></tr><tr><td>1967 A</td><td>1400</td><td>1088 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1967 B1</td><td>1400</td><td>1154 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1967 B2</td><td>2200</td><td>1149 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1967 C</td><td>2300</td><td>1116 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1967 D</td><td>2800</td><td>9 / 1162</td><td>0.08</td><td>0</td><td>solved</td></tr><tr><td>1967 E1</td><td>3100</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1967 E2</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1967 F</td><td>3200</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1975 - 25/May/24 - Codeforces Round 947 (Div. 1 + Div. 2) score: 5,959</td></tr><tr><td>1975 A</td><td>800</td><td>1161 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1975 B</td><td>1000</td><td>1091 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1975 C</td><td>1200</td><td>492 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1975 D</td><td>1700</td><td>9 / 1162</td><td>0.08</td><td>3</td><td>solved</td></tr><tr><td>1975 E</td><td>2100</td><td>80 / 1162</td><td>0.51</td><td>1</td><td>solved</td></tr><tr><td>1975 F</td><td>2600</td><td>12 / 1162</td><td>0.10</td><td>0</td><td>solved</td></tr><tr><td>1975 G</td><td>3000</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1975 H</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1975 I</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 1984 - 09/Jun/24 - Codeforces Global Round 26 score: 12,255</td></tr><tr><td>1984 A</td><td>800</td><td>1161 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1984 B</td><td>1100</td><td>1158 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1984 C1</td><td>1300</td><td>914 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1984 C2</td><td>1700</td><td>768 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1984 D</td><td>2000</td><td>193 / 1162</td><td>0.84</td><td>1</td><td>solved</td></tr><tr><td>1984 E</td><td>2400</td><td>849 / 1162</td><td>1.00</td><td>1</td><td>solved</td></tr><tr><td>1984 F</td><td>2500</td><td>918 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>1984 G</td><td>3200</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td>1984 H</td><td>3300</td><td>138 / 1162</td><td>0.72</td><td>3</td><td>solved</td></tr><tr><td colspan="6">Contest 2002 - 11/Aug/24 - EPIC IoT Round August 2024 (Div. 1 + Div. 2) score: 8,981</td></tr><tr><td>2002 A</td><td>800</td><td>1161 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2002 B</td><td>1000</td><td>1152 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2002 C</td><td>1200</td><td>1096 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2002 D1</td><td>1900</td><td>1067 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2002 D2</td><td>2300</td><td>805 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2002 E</td><td>2300</td><td>232 / 1162</td><td>0.89</td><td>0</td><td>solved</td></tr><tr><td>2002 F1</td><td>2600</td><td>12 / 1162</td><td>0.10</td><td>8</td><td>solved</td></tr><tr><td>2002 F2</td><td>2800</td><td>7 / 1162</td><td>0.06</td><td>0</td><td>not solved</td></tr><tr><td>2002 G</td><td>3400</td><td>143 / 1162</td><td>0.73</td><td>1</td><td>solved</td></tr><tr><td>2002 H</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr><tr><td colspan="6">Contest 2006 - 30/Aug/24 - Codeforces Round 969 (Div. 1) score: 2,757</td></tr><tr><td>2006 A</td><td>1700</td><td>945 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2006 B</td><td>1800</td><td>1149 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2006 C</td><td>2300</td><td>811 / 1162</td><td>1.00</td><td>0</td><td>solved</td></tr><tr><td>2006 D</td><td>2600</td><td>2 / 1162</td><td>0.02</td><td>0</td><td>not solved</td></tr><tr><td>2006 E</td><td>3100</td><td>4 / 1162</td><td>0.03</td><td>0</td><td>not solved</td></tr><tr><td>2006 F</td><td>3500</td><td>0 / 1162</td><td>0.00</td><td>0</td><td>not solved</td></tr></tbody></table>

## Appendix C IOI Submissions

This section presents the solutions generated by o1-ioi during the 2024 International Olympiad in Informatics.

### C.1 Nile

All 100 possible points for Nile were scored in a single submission.

[⬇](data:text/plain;base64,I2luY2x1ZGUgIm5pbGUuaCIKI2luY2x1ZGUgPGJpdHMvc3RkYysrLmg+CnVzaW5nIG5hbWVzcGFjZSBzdGQ7CgpzdHJ1Y3QgVUZ7CiAgICB2ZWN0b3I8aW50PiBwYXJlbnQsIHNpemVlLCBtaW5JbmRleDsKICAgIHZlY3RvcjxhcnJheTxsb25nIGxvbmcsIDI+PiBzaW5nbGVkUGF0OyAvLyBzaW5nbGVkUGF0WzBdID0gbWluIGRbaV0gZm9yIGkgbW9kIDIgPSAwIGluIGJsb2NrLCBzaW5nbGVkUGF0WzFdIGZvciBpIG1vZCAyID0gMSBpbiBibG9jawogICAgdmVjdG9yPGxvbmcgbG9uZz4gc2luZ2xlZEJyOyAvLyBzaW5nbGVkIGJyaWRnaW5nCiAgICB2ZWN0b3I8bG9uZyBsb25nPiBzaW5nbGVkOyAvLyBzaW5nbGVkIGluIGJsb2NrCn07Cgpsb25nIGxvbmcgc3VtU2luZ2xlZEdsb2JhbCA9IDA7CgppbnQgZmluZFJvb3QoVUYgJnUsIGludCB4KXsKICAgIGlmKHUucGFyZW50W3hdID09IHgpIHJldHVybiB4OwogICAgdS5wYXJlbnRbeF0gPSBmaW5kUm9vdCh1LCB1LnBhcmVudFt4XSk7CiAgICByZXR1cm4gdS5wYXJlbnRbeF07Cn0KCnZvaWQgdXBkYXRlU2luZ2xlZChVRiAmdSwgaW50IHIpewogICAgaWYodS5zaXplZVtyXSAlCiAgICAgICAgdS5zaW5nbGVkW3JdID0gMDsKICAgIH0gZWxzZSB7CiAgICAgICAgbG9uZyBsb25nIHZhbCA9IG1pbih1LnNpbmdsZWRQYXRbcl1bdS5taW5JbmRleFtyXSAlCiAgICAgICAgaWYodmFsID09IExMT05HX01BWCkgdmFsID0gMDsgLy8gaWYgbm8gc3VjaCBub2RlIGZvdW5kLCBzaW5nbGVkID0gMD8KICAgICAgICB1LnNpbmdsZWRbcl0gPSB2YWw7CiAgICB9Cn0KCnZvaWQgdW5pb25Ob2RlcyhVRiAmdSwgaW50IGEsIGludCBiKXsKICAgIGEgPSBmaW5kUm9vdCh1LCBhKTsKICAgIGIgPSBmaW5kUm9vdCh1LCBiKTsKICAgIGlmKGEgPT0gYikgcmV0dXJuOyAvLyBhbHJlYWR5IGluIHRoZSBzYW1lIGNvbXBvbmVudAoKICAgIHN1bVNpbmdsZWRHbG9iYWwgLT0gdS5zaW5nbGVkW2FdOwogICAgc3VtU2luZ2xlZEdsb2JhbCAtPSB1LnNpbmdsZWRbYl07CgogICAgLy8gdW5pb24gYnkgc2l6ZQogICAgaWYodS5zaXplZVthXSA8IHUuc2l6ZWVbYl0pIHN3YXAoYSwgYik7CiAgICB1LnBhcmVudFtiXSA9IGE7CgogICAgaW50IG5ld1NpemUgPSB1LnNpemVlW2FdICsgdS5zaXplZVtiXTsKICAgIGludCBuZXdNaW5JbmRleCA9IG1pbih1Lm1pbkluZGV4W2FdLCB1Lm1pbkluZGV4W2JdKTsKCiAgICBhcnJheTxsb25nIGxvbmcsIDI+IG5ld1NpbmdsZWRQYXQ7CiAgICBuZXdTaW5nbGVkUGF0WzBdID0gbWluKHUuc2luZ2xlZFBhdFthXVswXSwgdS5zaW5nbGVkUGF0W2JdWzBdKTsKICAgIG5ld1NpbmdsZWRQYXRbMV0gPSBtaW4odS5zaW5nbGVkUGF0W2FdWzFdLCB1LnNpbmdsZWRQYXRbYl1bMV0pOwoKICAgIGxvbmcgbG9uZyBuZXdTaW5nbGVkQnIgPSBtaW4odS5zaW5nbGVkQnJbYV0sIHUuc2luZ2xlZEJyW2JdKTsKCiAgICB1LnNpemVlW2FdID0gbmV3U2l6ZTsKICAgIHUubWluSW5kZXhbYV0gPSBuZXdNaW5JbmRleDsKICAgIHUuc2luZ2xlZFBhdFthXSA9IG5ld1NpbmdsZWRQYXQ7CiAgICB1LnNpbmdsZWRCclthXSA9IG5ld1NpbmdsZWRCcjsKCiAgICB1cGRhdGVTaW5nbGVkKHUsIGEpOwoKICAgIHN1bVNpbmdsZWRHbG9iYWwgKz0gdS5zaW5nbGVkW2FdOwp9Cgp2b2lkIGFkZEJyaWRnaW5nKFVGICZ1LCBpbnQgeCwgbG9uZyBsb25nIHZhbCl7CiAgICBpbnQgciA9IGZpbmRSb290KHUsIHgpOwogICAgc3VtU2luZ2xlZEdsb2JhbCAtPSB1LnNpbmdsZWRbcl07CiAgICBpZih1LnNpemVlW3JdICUKICAgICAgICB1LnNpbmdsZWRCcltyXSA9IG1pbih1LnNpbmdsZWRCcltyXSwgdmFsKTsKICAgICAgICB1cGRhdGVTaW5nbGVkKHUsIHIpOwogICAgfSBlbHNlIHsKICAgICAgICAvLyBibG9jayBzaXplIGlzIGV2ZW4sIHNpbmdsZWQgPSAwIGFueXdheSwgYnJpZGdpbmcgd29uJ3QgbWF0dGVyIGJlY2F1c2Ugc2luZ2xlZEJyIGlzIGZvciBvZGQgc2l6ZWQgYmxvY2suCiAgICAgICAgLy8gYnV0IGJyaWRnaW5nIG1pZ2h0IG1hdHRlciBpZiBibG9jayBzaXplIGJlY29tZXMgb2RkIGluIGZ1dHVyZSBtZXJnZXMsIHNvIHN0b3JlIHNpbmdsZWRCciBhcyB3ZWxsLgogICAgICAgIHUuc2luZ2xlZEJyW3JdID0gbWluKHUuc2luZ2xlZEJyW3JdLCB2YWwpOwogICAgICAgIC8vIHNpbmdsZWQgaXMgMCBhbnl3YXkKICAgIH0KICAgIHN1bVNpbmdsZWRHbG9iYWwgKz0gdS5zaW5nbGVkW3JdOwp9CgpzdGQ6OnZlY3Rvcjxsb25nIGxvbmc+IGNhbGN1bGF0ZV9jb3N0cygKICAgIHN0ZDo6dmVjdG9yPGludD4gVywgc3RkOjp2ZWN0b3I8aW50PiBBLAogICAgc3RkOjp2ZWN0b3I8aW50PiBCLCBzdGQ6OnZlY3RvcjxpbnQ+IEUpewoKICAgIGludCBOID0gVy5zaXplKCk7CiAgICBpbnQgUSA9IEUuc2l6ZSgpOwoKICAgIHZlY3RvcjxpbnQ+IGlkeChOKTsgaW90YShpZHguYmVnaW4oKSwgaWR4LmVuZCgpLCAwKTsKICAgIHNvcnQoaWR4LmJlZ2luKCksIGlkeC5lbmQoKSwgWyZdKGludCBhLCBpbnQgYil7cmV0dXJuIFdbYV0gPCBXW2JdO30pOwoKICAgIHZlY3Rvcjxsb25nIGxvbmc+IHNvcnRlZFcoTiksIGQoTik7CiAgICBmb3IoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSB7CiAgICAgICAgc29ydGVkV1tpXSA9IFdbaWR4W2ldXTsKICAgICAgICBkW2ldID0gKGxvbmcgbG9uZylBW2lkeFtpXV0gLSAobG9uZyBsb25nKUJbaWR4W2ldXTsKICAgIH0KCiAgICBsb25nIGxvbmcgc3VtQiA9IDA7CiAgICBmb3IoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSBzdW1CICs9IEJbaWR4W2ldXTsKCiAgICBzdHJ1Y3QgRWRnZXsgbG9uZyBsb25nIGRpZmY7IGludCBpZHg7fTsKICAgIHZlY3RvcjxFZGdlPiBlZGdlczsKICAgIGVkZ2VzLnJlc2VydmUoTi0xKTsKICAgIGZvcihpbnQgaSA9IDA7IGkgPCBOLTE7IGkrKykgewogICAgICAgIGVkZ2VzLnB1c2hfYmFjayh7c29ydGVkV1tpKzFdIC0gc29ydGVkV1tpXSwgaX0pOwogICAgfQogICAgc29ydChlZGdlcy5iZWdpbigpLCBlZGdlcy5lbmQoKSwgWyZdKGNvbnN0IEVkZ2UgJmEsIGNvbnN0IEVkZ2UgJmIpe3JldHVybiBhLmRpZmYgPCBiLmRpZmY7fSk7CgogICAgdmVjdG9yPHBhaXI8bG9uZyBsb25nLCBpbnQ+PiBicmlkZ2luZzsKICAgIGJyaWRnaW5nLnJlc2VydmUoTi0yKTsKICAgIGZvcihpbnQgaSA9IDE7IGkgPCBOLTE7IGkrKykgewogICAgICAgIGxvbmcgbG9uZyB2YWwgPSBzb3J0ZWRXW2krMV0gLSBzb3J0ZWRXW2ktMV07CiAgICAgICAgYnJpZGdpbmcucHVzaF9iYWNrKHt2YWwsIGl9KTsKICAgIH0KICAgIHNvcnQoYnJpZGdpbmcuYmVnaW4oKSwgYnJpZGdpbmcuZW5kKCksIFsmXShhdXRvICZhLCBhdXRvICZiKXtyZXR1cm4gYS5maXJzdCA8IGIuZmlyc3Q7fSk7CgogICAgdmVjdG9yPHBhaXI8bG9uZyBsb25nLCBpbnQ+PiBxdWVyaWVzVmVjOwogICAgcXVlcmllc1ZlYy5yZXNlcnZlKFEpOwogICAgZm9yKGludCBpID0gMDsgaSA8IFE7IGkrKykgcXVlcmllc1ZlYy5wdXNoX2JhY2soeyhsb25nIGxvbmcpRVtpXSwgaX0pOwogICAgc29ydChxdWVyaWVzVmVjLmJlZ2luKCksIHF1ZXJpZXNWZWMuZW5kKCksIFsmXShhdXRvICZhLCBhdXRvICZiKXtyZXR1cm4gYS5maXJzdCA8IGIuZmlyc3Q7fSk7CgogICAgVUYgdTsKICAgIHUucGFyZW50LnJlc2l6ZShOKTsKICAgIHUuc2l6ZWUucmVzaXplKE4pOwogICAgdS5taW5JbmRleC5yZXNpemUoTik7CiAgICB1LnNpbmdsZWRQYXQucmVzaXplKE4pOwogICAgdS5zaW5nbGVkQnIucmVzaXplKE4pOwogICAgdS5zaW5nbGVkLnJlc2l6ZShOKTsKCiAgICBmb3IoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSB7CiAgICAgICAgdS5wYXJlbnRbaV0gPSBpOwogICAgICAgIHUuc2l6ZWVbaV0gPSAxOwogICAgICAgIHUubWluSW5kZXhbaV0gPSBpOwogICAgICAgIHUuc2luZ2xlZFBhdFtpXVswXSA9IChpICUKICAgICAgICB1LnNpbmdsZWRQYXRbaV1bMV0gPSAoaSAlCiAgICAgICAgdS5zaW5nbGVkQnJbaV0gPSBMTE9OR19NQVg7CiAgICAgICAgdS5zaW5nbGVkW2ldID0gZFtpXTsgLy8gc2luZ2xlZCA9IGRbaV0gYmVjYXVzZSBibG9jayBzaXplIDEgaXMgb2RkCiAgICB9CgogICAgc3VtU2luZ2xlZEdsb2JhbCA9IDA7CiAgICBmb3IoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSB7CiAgICAgICAgc3VtU2luZ2xlZEdsb2JhbCArPSB1LnNpbmdsZWRbaV07CiAgICB9CgogICAgdmVjdG9yPGxvbmcgbG9uZz4gcmVzdWx0KFEpOwoKICAgIGludCBwb3NFID0gMDsgLy8gaW5kZXggaW4gZWRnZXMKICAgIGludCBwb3NCID0gMDsgLy8gaW5kZXggaW4gYnJpZGdpbmcKCiAgICBmb3IoaW50IGkgPSAwOyBpIDwgUTsgaSsrKSB7CiAgICAgICAgbG9uZyBsb25nIEQgPSBxdWVyaWVzVmVjW2ldLmZpcnN0OwogICAgICAgIGludCBxaWR4ID0gcXVlcmllc1ZlY1tpXS5zZWNvbmQ7CgogICAgICAgIHdoaWxlKHBvc0UgPCAoaW50KWVkZ2VzLnNpemUoKSAmJiBlZGdlc1twb3NFXS5kaWZmIDw9IEQpIHsKICAgICAgICAgICAgdW5pb25Ob2Rlcyh1LCBlZGdlc1twb3NFXS5pZHgsIGVkZ2VzW3Bvc0VdLmlkeCsxKTsKICAgICAgICAgICAgcG9zRSsrOwogICAgICAgIH0KCiAgICAgICAgd2hpbGUocG9zQiA8IChpbnQpYnJpZGdpbmcuc2l6ZSgpICYmIGJyaWRnaW5nW3Bvc0JdLmZpcnN0IDw9IEQpIHsKICAgICAgICAgICAgaW50IHggPSBicmlkZ2luZ1twb3NCXS5zZWNvbmQ7CiAgICAgICAgICAgIGFkZEJyaWRnaW5nKHUsIHgsIGRbeF0pOwogICAgICAgICAgICBwb3NCKys7CiAgICAgICAgfQoKICAgICAgICBsb25nIGxvbmcgY29zdFZhbCA9IHN1bUIgKyBzdW1TaW5nbGVkR2xvYmFsOwogICAgICAgIHJlc3VsdFtxaWR4XSA9IGNvc3RWYWw7CiAgICB9CgogICAgcmV0dXJuIHJlc3VsdDsKfQ==)

#include "nile.h"

#include <bits/stdc++.h>

using namespace std;

struct UF{

vector<int> parent, sizee, minIndex;

vector<array<long long, 2>> singledPat; // singledPat\[0\] = min d\[i\] for i mod 2 = 0 in block, singledPat\[1\] for i mod 2 = 1 in block

vector<long long> singledBr; // singled bridging

vector<long long> singled; // singled in block

};

long long sumSingledGlobal = 0;

int findRoot(UF &u, int x){

if(u.parent\[x\] == x) return x;

u.parent\[x\] = findRoot(u, u.parent\[x\]);

return u.parent\[x\];

}

void updateSingled(UF &u, int r){

if(u.sizee\[r\] %

u.singled\[r\] = 0;

} else {

long long val = min(u.singledPat\[r\]\[u.minIndex\[r\] %

if(val == LLONG\_MAX) val = 0; // if no such node found, singled = 0?

u.singled\[r\] = val;

}

}

void unionNodes(UF &u, int a, int b){

a = findRoot(u, a);

b = findRoot(u, b);

if(a == b) return; // already in the same component

sumSingledGlobal -= u.singled\[a\];

sumSingledGlobal -= u.singled\[b\];

// union by size

if(u.sizee\[a\] < u.sizee\[b\]) swap(a, b);

u.parent\[b\] = a;

int newSize = u.sizee\[a\] + u.sizee\[b\];

int newMinIndex = min(u.minIndex\[a\], u.minIndex\[b\]);

array<long long, 2> newSingledPat;

newSingledPat\[0\] = min(u.singledPat\[a\]\[0\], u.singledPat\[b\]\[0\]);

newSingledPat\[1\] = min(u.singledPat\[a\]\[1\], u.singledPat\[b\]\[1\]);

long long newSingledBr = min(u.singledBr\[a\], u.singledBr\[b\]);

u.sizee\[a\] = newSize;

u.minIndex\[a\] = newMinIndex;

u.singledPat\[a\] = newSingledPat;

u.singledBr\[a\] = newSingledBr;

updateSingled(u, a);

sumSingledGlobal += u.singled\[a\];

}

void addBridging(UF &u, int x, long long val){

int r = findRoot(u, x);

sumSingledGlobal -= u.singled\[r\];

if(u.sizee\[r\] %

u.singledBr\[r\] = min(u.singledBr\[r\], val);

updateSingled(u, r);

} else {

// block size is even, singled = 0 anyway, bridging won’t matter because singledBr is for odd sized block.

// but bridging might matter if block size becomes odd in future merges, so store singledBr as well.

u.singledBr\[r\] = min(u.singledBr\[r\], val);

// singled is 0 anyway

}

sumSingledGlobal += u.singled\[r\];

}

std::vector<long long> calculate\_costs(

std::vector<int> W, std::vector<int> A,

std::vector<int> B, std::vector<int> E){

int N = W.size();

int Q = E.size();

vector<int> idx(N); iota(idx.begin(), idx.end(), 0);

sort(idx.begin(), idx.end(), \[&\](int a, int b){return W\[a\] < W\[b\];});

vector<long long> sortedW(N), d(N);

for(int i = 0; i < N; i++) {

sortedW\[i\] = W\[idx\[i\]\];

d\[i\] = (long long)A\[idx\[i\]\] - (long long)B\[idx\[i\]\];

}

long long sumB = 0;

for(int i = 0; i < N; i++) sumB += B\[idx\[i\]\];

struct Edge{ long long diff; int idx;};

vector<Edge> edges;

edges.reserve(N-1);

for(int i = 0; i < N-1; i++) {

edges.push\_back({sortedW\[i+1\] - sortedW\[i\], i});

}

sort(edges.begin(), edges.end(), \[&\](const Edge &a, const Edge &b){return a.diff < b.diff;});

vector<pair<long long, int>> bridging;

bridging.reserve(N-2);

for(int i = 1; i < N-1; i++) {

long long val = sortedW\[i+1\] - sortedW\[i-1\];

bridging.push\_back({val, i});

}

sort(bridging.begin(), bridging.end(), \[&\](auto &a, auto &b){return a.first < b.first;});

vector<pair<long long, int>> queriesVec;

queriesVec.reserve(Q);

for(int i = 0; i < Q; i++) queriesVec.push\_back({(long long)E\[i\], i});

sort(queriesVec.begin(), queriesVec.end(), \[&\](auto &a, auto &b){return a.first < b.first;});

UF u;

u.parent.resize(N);

u.sizee.resize(N);

u.minIndex.resize(N);

u.singledPat.resize(N);

u.singledBr.resize(N);

u.singled.resize(N);

for(int i = 0; i < N; i++) {

u.parent\[i\] = i;

u.sizee\[i\] = 1;

u.minIndex\[i\] = i;

u.singledPat\[i\]\[0\] = (i %

u.singledPat\[i\]\[1\] = (i %

u.singledBr\[i\] = LLONG\_MAX;

u.singled\[i\] = d\[i\]; // singled = d\[i\] because block size 1 is odd

}

sumSingledGlobal = 0;

for(int i = 0; i < N; i++) {

sumSingledGlobal += u.singled\[i\];

}

vector<long long> result(Q);

int posE = 0; // index in edges

int posB = 0; // index in bridging

for(int i = 0; i < Q; i++) {

long long D = queriesVec\[i\].first;

int qidx = queriesVec\[i\].second;

while(posE < (int)edges.size() && edges\[posE\].diff <= D) {

unionNodes(u, edges\[posE\].idx, edges\[posE\].idx+1);

posE++;

}

while(posB < (int)bridging.size() && bridging\[posB\].first <= D) {

int x = bridging\[posB\].second;

addBridging(u, x, d\[x\]);

posB++;

}

long long costVal = sumB + sumSingledGlobal;

result\[qidx\] = costVal;

}

return result;

}

### C.2 Message

This solution achieved a score of 79.64 out of 100, with full marks awarded for subtask 1 and partial credit received for subtask 2.

[⬇](data:text/plain;base64,I2luY2x1ZGUgIm1lc3NhZ2UuaCIKI2luY2x1ZGUgPGJpdHMvc3RkYysrLmg+CnVzaW5nIG5hbWVzcGFjZSBzdGQ7CgpleHRlcm4gc3RkOjp2ZWN0b3I8Ym9vbD4gc2VuZF9wYWNrZXQoc3RkOjp2ZWN0b3I8Ym9vbD4gQSk7Cgp2b2lkIHNlbmRfbWVzc2FnZShzdGQ6OnZlY3Rvcjxib29sPiBNLCBzdGQ6OnZlY3Rvcjxib29sPiBDKQp7CiAgICB2ZWN0b3I8aW50PiBzYWZlX2NvbHM7CiAgICB2ZWN0b3I8aW50PiBzYWJvdGFnZV9jb2xzOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCAzMTsgaSsrKSB7CiAgICAgICAgaWYgKCFDW2ldKSBzYWZlX2NvbHMucHVzaF9iYWNrKGkpOwogICAgICAgIGVsc2Ugc2Fib3RhZ2VfY29scy5wdXNoX2JhY2soaSk7CiAgICB9CgogICAgLy8gU3RlcCAxOiBzZW5kIHRoZSBmaXJzdCA0IHBhY2tldHMgd2l0aCBkaXN0aW5jdCBzZXF1ZW5jZXMuCiAgICB2ZWN0b3I8dmVjdG9yPGJvb2w+PiBmaXJzdDRyZXN1bHRzOwogICAgZmlyc3Q0cmVzdWx0cy5yZXNlcnZlKDQpOwogICAgZm9yIChpbnQgcCA9IDA7IHAgPCA0OyBwKyspIHsKICAgICAgICB2ZWN0b3I8Ym9vbD4gQSgzMSwgZmFsc2UpOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgMTY7IGkrKykgewogICAgICAgICAgICBpbnQgY29sID0gc2FmZV9jb2xzW2ldOwogICAgICAgICAgICBpbnQgYml0ID0gKGkgPj4gcCkgJiAxOyAvLyBwLXRoIGJpdCBvZiBpLCBpIGlzIHRoZSBpbmRleCBpbiBzYWZlIHN1YnNldAogICAgICAgICAgICBBW2NvbF0gPSBiaXQ7CiAgICAgICAgfQogICAgICAgIGZvciAoaW50IGNvbCA6IHNhYm90YWdlX2NvbHMpIHsKICAgICAgICAgICAgQVtjb2xdID0gZmFsc2U7CiAgICAgICAgfQogICAgICAgIHN0ZDo6dmVjdG9yPGJvb2w+IEIgPSBzZW5kX3BhY2tldChBKTsKICAgICAgICBmaXJzdDRyZXN1bHRzLnB1c2hfYmFjayhCKTsKICAgIH0KCiAgICAvLyBOb3cgaWRlbnRpZnkga25vd24gc2FmZSBjb2x1bW4gZnJvbSB0aGUgcmVzdWx0cyBvZiB0aGVzZSA0IGNhbGxzLgogICAgdmVjdG9yPHN0cmluZz4gY29sU2VxdWVuY2VzKDMxLCBzdHJpbmcoNCwgJzAnKSk7CiAgICBmb3IgKGludCBjID0gMDsgYyA8IDMxOyBjKyspIHsKICAgICAgICBmb3IgKGludCByID0gMDsgciA8IDQ7IHIrKykgewogICAgICAgICAgICBjb2xTZXF1ZW5jZXNbY11bcl0gPSBmaXJzdDRyZXN1bHRzW3JdW2NdID8gJzEnIDogJzAnOwogICAgICAgIH0KICAgIH0KCiAgICBtYXA8c3RyaW5nLCB2ZWN0b3I8aW50Pj4gc2VxVG9Db2xzOwogICAgZm9yIChpbnQgYyA9IDA7IGMgPCAzMTsgYysrKSB7CiAgICAgICAgc2VxVG9Db2xzW2NvbFNlcXVlbmNlc1tjXV0ucHVzaF9iYWNrKGMpOwogICAgfQogICAgaW50IGtub3duU2FmZUNvbCA9IC0xOwogICAgZm9yIChhdXRvICZrdiA6IHNlcVRvQ29scykgewogICAgICAgIGlmIChrdi5zZWNvbmQuc2l6ZSgpID09IDEpIHsKICAgICAgICAgICAgaW50IGNvbCA9IGt2LnNlY29uZFswXTsKICAgICAgICAgICAgLy8gY2hlY2sgaWYgY29sIGlzIGluIHNhZmUgc3Vic2V0LiBCZWNhdXNlIGtub3duIHNhZmUgY29sdW1uIG11c3QgYmUgYSBzYWZlIGNvbHVtbi4KICAgICAgICAgICAgaWYgKGZpbmQoc2FmZV9jb2xzLmJlZ2luKCksIHNhZmVfY29scy5lbmQoKSwgY29sKSAhPSBzYWZlX2NvbHMuZW5kKCkpIHsKICAgICAgICAgICAgICAgIC8vIGFsc28gY2hlY2sgaWYgdGhlIHNlcXVlbmNlIGlzIG9uZSBvZiB0aGUgZGlzdGluY3Qgc2VxdWVuY2VzIGZvciBzYWZlIHN1YnNldC4gVGhlIGluZGV4IGluIHRoZSBzYWZlIHN1YnNldCBpcyBpLCB0aGUgc2VxdWVuY2UgaXMgNCBiaXRzIHJlcHJlc2VudGluZyBpLgogICAgICAgICAgICAgICAgc3RyaW5nIHNlcSA9IGt2LmZpcnN0OwogICAgICAgICAgICAgICAgaW50IHZhbCA9IDA7CiAgICAgICAgICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8IDQ7IGkrKykgewogICAgICAgICAgICAgICAgICAgIGlmIChzZXFbaV0gPT0gJzEnKSB2YWwgfD0gKDEgPDwgaSk7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAvLyBjaGVjayBpZiB2YWwgaXMgPCAxNi4KICAgICAgICAgICAgICAgIGlmICh2YWwgPCAxNikgewogICAgICAgICAgICAgICAgICAgIGtub3duU2FmZUNvbCA9IGNvbDsKICAgICAgICAgICAgICAgICAgICBicmVhazsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0KICAgIGlmIChrbm93blNhZmVDb2wgPT0gLTEpIHsKICAgICAgICAvLyBJZiBub3QgZm91bmQsIHBpY2sgYSBzYWZlIGNvbHVtbiB0aGF0IGhhcyBhIGRpc3RpbmN0IHNlcXVlbmNlIGFueXdheS4KICAgICAgICAvLyBCdXQgZm9yIGNvcnJlY3RuZXNzLCB3ZSByZXF1aXJlIGtub3duU2FmZUNvbCB0byBiZSBmb3VuZC4gSWYgbm90IGZvdW5kLCBvdXIgbWV0aG9kIGhhcyBhIHByb2JsZW0uCiAgICB9CgogICAgLy8gU3RlcCAyOiBzdG9yZSBzYWJvdGFnZSBzdWJzZXQncyBiaXRzIGluIHRoZSBrbm93biBzYWZlIGNvbHVtbiBpbiBuZXh0IDMxIGNhbGxzLiBUaGUgb3RoZXIgMTUgc2FmZSBjb2x1bW5zIHN0b3JlIG1lc3NhZ2UgYml0cyBpZiBhdmFpbGFibGUuCiAgICBpbnQgUyA9IE0uc2l6ZSgpOyAvLyBtZXNzYWdlIGxlbmd0aC4KICAgIGludCBvZmZzZXQgPSAwOyAvLyBvZmZzZXQgaW4gTS4KICAgIGZvciAoaW50IGkgPSAwOyBpIDwgMzE7IGkrKykgewogICAgICAgIHZlY3Rvcjxib29sPiBBKDMxLCBmYWxzZSk7CiAgICAgICAgaW50IHNhYm90YWdlQml0ID0gQ1tpXTsKICAgICAgICBBW2tub3duU2FmZUNvbF0gPSBzYWJvdGFnZUJpdDsgLy8gc3RvcmUgc2Fib3RhZ2Ugc3Vic2V0J3MgYml0IGkgaW4ga25vd24gc2FmZSBjb2x1bW4uCiAgICAgICAgZm9yIChpbnQgYyA6IHNhZmVfY29scykgewogICAgICAgICAgICBpZiAoYyA9PSBrbm93blNhZmVDb2wpIGNvbnRpbnVlOyAvLyBza2lwIGtub3duIHNhZmUgY29sdW1uIGJlY2F1c2Ugd2UgdXNlZCBpdCBmb3Igc2Fib3RhZ2Ugc3Vic2V0LgogICAgICAgICAgICBpbnQgY29sID0gYzsKICAgICAgICAgICAgaWYgKG9mZnNldCA8IFMpIHsKICAgICAgICAgICAgICAgIEFbY29sXSA9IE1bb2Zmc2V0XTsKICAgICAgICAgICAgICAgIG9mZnNldCsrOwogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgQVtjb2xdID0gZmFsc2U7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgZm9yIChpbnQgc2MgOiBzYWJvdGFnZV9jb2xzKSB7CiAgICAgICAgICAgIEFbc2NdID0gZmFsc2U7CiAgICAgICAgfQogICAgICAgIHN0ZDo6dmVjdG9yPGJvb2w+IEIgPSBzZW5kX3BhY2tldChBKTsKICAgICAgICAvLyBXZSBkbyBub3Qgc3RvcmUgQiBiZWNhdXNlIHdlIGRvIG5vdCBuZWVkIHRvIGFkYXB0IGZ1cnRoZXIuIEFjdHVhbGx5LCB3ZSBjYW4gc3RvcmUgQiBpZiB3ZSB3YW50IHRvIGFkYXB0IGJ1dCB0aGUgc29sdXRpb24gZG9lcyBub3QgcmVxdWlyZSBmdXJ0aGVyIGFkYXB0YXRpb24uCiAgICB9CgogICAgLy8gU3RlcCAzOiBzdG9yZSBtZXNzYWdlIGxlbmd0aCBpbiB0aGUgbmV4dCAxMSBjYWxscyBpbiB0aGUga25vd24gc2FmZSBjb2x1bW4uIFRoZSBvdGhlciAxNSBzYWZlIGNvbHVtbnMgc3RvcmUgbW9yZSBtZXNzYWdlIGJpdHMgaWYgYXZhaWxhYmxlLgogICAgaW50IEwgPSBTOyAvLyBsZW5ndGggaW4gaW50ZWdlciBmb3JtLgogICAgdmVjdG9yPGludD4gbGVuZ3RoQmluYXJ5KDExLCAwKTsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgMTE7IGkrKykgewogICAgICAgIGxlbmd0aEJpbmFyeVtpXSA9IChMID4+IGkpICYgMTsKICAgIH0KCiAgICBmb3IgKGludCBpID0gMDsgaSA8IDExOyBpKyspIHsKICAgICAgICB2ZWN0b3I8Ym9vbD4gQSgzMSwgZmFsc2UpOwogICAgICAgIGludCBiaXQgPSBsZW5ndGhCaW5hcnlbaV07CiAgICAgICAgQVtrbm93blNhZmVDb2xdID0gYml0OyAvLyBzdG9yZSBsZW5ndGggYml0IGkgaW4ga25vd24gc2FmZSBjb2x1bW4uCiAgICAgICAgZm9yIChpbnQgYyA6IHNhZmVfY29scykgewogICAgICAgICAgICBpZiAoYyA9PSBrbm93blNhZmVDb2wpIGNvbnRpbnVlOyAvLyBza2lwIGtub3duIHNhZmUgY29sdW1uLgogICAgICAgICAgICBpbnQgY29sID0gYzsKICAgICAgICAgICAgaWYgKG9mZnNldCA8IFMpIHsKICAgICAgICAgICAgICAgIEFbY29sXSA9IE1bb2Zmc2V0XTsKICAgICAgICAgICAgICAgIG9mZnNldCsrOwogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgQVtjb2xdID0gZmFsc2U7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgZm9yIChpbnQgc2MgOiBzYWJvdGFnZV9jb2xzKSB7CiAgICAgICAgICAgIEFbc2NdID0gZmFsc2U7CiAgICAgICAgfQogICAgICAgIHN0ZDo6dmVjdG9yPGJvb2w+IEIgPSBzZW5kX3BhY2tldChBKTsKICAgIH0KCiAgICAvLyBTdGVwIDQ6IHN0b3JlIHRoZSByZXN0IG9mIHRoZSBtZXNzYWdlIGluIHN1YnNlcXVlbnQgY2FsbHMgdXNpbmcgYWxsIDE2IHNhZmUgY29sdW1ucy4KICAgIGludCBtZXNzYWdlTGVmdCA9IFMgLSBvZmZzZXQ7IC8vIGhvdyBtYW55IGJpdHMgYXJlIGxlZnQgdG8gc3RvcmUuCiAgICBpbnQgZnVsbENvbHVtbnMgPSAxNjsgLy8gbnVtYmVyIG9mIHNhZmUgY29sdW1ucy4KICAgIGludCBtZXNzYWdlUm93cyA9IChtZXNzYWdlTGVmdCArIGZ1bGxDb2x1bW5zIC0gMSkgLyBmdWxsQ29sdW1uczsgLy8gaG93IG1hbnkgcm93cyBuZWVkZWQuCgogICAgZm9yIChpbnQgciA9IDA7IHIgPCBtZXNzYWdlUm93czsgcisrKSB7CiAgICAgICAgdmVjdG9yPGJvb2w+IEEoMzEsIGZhbHNlKTsKICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8IGZ1bGxDb2x1bW5zICYmIG9mZnNldCArIGkgPCBTOyBpKyspIHsKICAgICAgICAgICAgaW50IGNvbCA9IHNhZmVfY29sc1tpXTsKICAgICAgICAgICAgQVtjb2xdID0gTVtvZmZzZXQgKyBpXTsKICAgICAgICB9CiAgICAgICAgb2Zmc2V0ICs9IGZ1bGxDb2x1bW5zOwogICAgICAgIGZvciAoaW50IHNjIDogc2Fib3RhZ2VfY29scykgewogICAgICAgICAgICBBW3NjXSA9IGZhbHNlOwogICAgICAgIH0KICAgICAgICBzdGQ6OnZlY3Rvcjxib29sPiBCID0gc2VuZF9wYWNrZXQoQSk7CiAgICB9Cgp9CgpzdGQ6OnZlY3Rvcjxib29sPiByZWNlaXZlX21lc3NhZ2Uoc3RkOjp2ZWN0b3I8c3RkOjp2ZWN0b3I8Ym9vbD4+IFIpCnsKICAgIGludCBuID0gUi5zaXplKCk7CiAgICB2ZWN0b3I8Ym9vbD4gcmVzdWx0OyAvLyBtZXNzYWdlIHRvIHJldHVybi4KICAgIGlmIChuIDwgNCkgewogICAgICAgIHJldHVybiByZXN1bHQ7CiAgICB9CgogICAgLy8gU3RlcCAxOiBmcm9tIHRoZSBmaXJzdCA0IHJvd3MgaW4gUiwgZmluZCB0aGUga25vd24gc2FmZSBjb2x1bW4uCiAgICB2ZWN0b3I8c3RyaW5nPiBjb2xTZXF1ZW5jZXMoMzEsIHN0cmluZyg0LCAnMCcpKTsKICAgIGZvciAoaW50IGMgPSAwOyBjIDwgMzE7IGMrKykgewogICAgICAgIGZvciAoaW50IHIgPSAwOyByIDwgNCAmJiByIDwgbjsgcisrKSB7CiAgICAgICAgICAgIGlmIChyIDwgbikgewogICAgICAgICAgICAgICAgY29sU2VxdWVuY2VzW2NdW3JdID0gUltyXVtjXSA/ICcxJyA6ICcwJzsKICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0KCiAgICBtYXA8c3RyaW5nLCB2ZWN0b3I8aW50Pj4gc2VxVG9Db2xzOwogICAgZm9yIChpbnQgYyA9IDA7IGMgPCAzMTsgYysrKSB7CiAgICAgICAgc2VxVG9Db2xzW2NvbFNlcXVlbmNlc1tjXV0ucHVzaF9iYWNrKGMpOwogICAgfQogICAgaW50IGtub3duU2FmZUNvbCA9IC0xOwogICAgZm9yIChhdXRvICZrdiA6IHNlcVRvQ29scykgewogICAgICAgIGlmIChrdi5zZWNvbmQuc2l6ZSgpID09IDEpIHsKICAgICAgICAgICAgaW50IGNvbCA9IGt2LnNlY29uZFswXTsKICAgICAgICAgICAgLy8gY2hlY2sgaWYgdGhlIHNlcXVlbmNlIGlzIG9uZSBvZiB0aGUgYmluYXJ5IHJlcHJlc2VudGF0aW9ucyBmcm9tIDAuLjE1IGJlY2F1c2Ugc2FmZSBzdWJzZXQncyBzZXF1ZW5jZXMgYXJlIGRpc3RpbmN0IGFuZCBpbiB0aGF0IHJhbmdlLgogICAgICAgICAgICBzdHJpbmcgc2VxID0ga3YuZmlyc3Q7CiAgICAgICAgICAgIGludCB2YWwgPSAwOwogICAgICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8IDQ7IGkrKykgewogICAgICAgICAgICAgICAgaWYgKHNlcVtpXSA9PSAnMScpIHZhbCB8PSAoMSA8PCBpKTsKICAgICAgICAgICAgfQogICAgICAgICAgICBpZiAodmFsIDwgMTYpIHsKICAgICAgICAgICAgICAgIGtub3duU2FmZUNvbCA9IGNvbDsKICAgICAgICAgICAgICAgIGJyZWFrOwogICAgICAgICAgICB9CiAgICAgICAgfQogICAgfQogICAgaWYgKGtub3duU2FmZUNvbCA9PSAtMSkgewogICAgICAgIC8vIElmIG5vbmUgZm91bmQsIHRoYXQgbWVhbnMgc2Fib3RhZ2Ugb3ZlcmNhbWUgdGhlIG1ldGhvZCBvciB3ZSBtYWRlIGFuIGVycm9yLgogICAgfQoKICAgIC8vIFN0ZXAgMjogZnJvbSBuZXh0IDMxIHJvd3MgaW4ga25vd24gc2FmZSBjb2x1bW4sIHJlYWQgc2Fib3RhZ2Ugc3Vic2V0LgogICAgdmVjdG9yPGJvb2w+IHNhYm90YWdlU3Vic2V0KDMxLCBmYWxzZSk7CiAgICBpbnQgc2Fib3RhZ2VTdGFydCA9IDQ7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IDMxICYmIHNhYm90YWdlU3RhcnQgKyBpIDwgbjsgaSsrKSB7CiAgICAgICAgc2Fib3RhZ2VTdWJzZXRbaV0gPSBSW3NhYm90YWdlU3RhcnQgKyBpXVtrbm93blNhZmVDb2xdOwogICAgfQoKICAgIC8vIFN0ZXAgMzogZnJvbSBuZXh0IDExIHJvd3MgaW4ga25vd24gc2FmZSBjb2x1bW4sIHJlYWQgbGVuZ3RoLgogICAgdmVjdG9yPGJvb2w+IGxlbmd0aEJpdHMoMTEsIGZhbHNlKTsKICAgIGludCBsZW5ndGhTdGFydCA9IHNhYm90YWdlU3RhcnQgKyAzMTsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgMTEgJiYgbGVuZ3RoU3RhcnQgKyBpIDwgbjsgaSsrKSB7CiAgICAgICAgbGVuZ3RoQml0c1tpXSA9IFJbbGVuZ3RoU3RhcnQgKyBpXVtrbm93blNhZmVDb2xdOwogICAgfQogICAgaW50IG1lc3NhZ2VMZW5ndGggPSAwOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCAxMTsgaSsrKSB7CiAgICAgICAgbWVzc2FnZUxlbmd0aCB8PSAobGVuZ3RoQml0c1tpXSA8PCBpKTsKICAgIH0KCiAgICAvLyBTdGVwIDQ6IGlkZW50aWZ5IHNhZmUgY29sdW1ucyBmcm9tIHNhYm90YWdlIHN1YnNldC4KICAgIHZlY3RvcjxpbnQ+IHNhZmVfY29sczsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgMzE7IGkrKykgewogICAgICAgIGlmICghc2Fib3RhZ2VTdWJzZXRbaV0pIHNhZmVfY29scy5wdXNoX2JhY2soaSk7CiAgICB9CgogICAgLy8gU3RlcCA1OiByZWFkIG1lc3NhZ2UgZnJvbSB0aGUgbmV4dCAoMzEgKyAxMSkgcm93cycgb3RoZXIgc2FmZSBjb2x1bW5zIGFuZCBmcm9tIHN1YnNlcXVlbnQgcm93cyB1c2luZyBhbGwgc2FmZSBjb2x1bW5zLgogICAgdmVjdG9yPGJvb2w+IG1lc3NhZ2U7CiAgICBpbnQgb2Zmc2V0ID0gMDsgLy8gb2Zmc2V0IGluIG1lc3NhZ2UuCiAgICBpbnQgc2Fib3RhZ2VSb3dTdGFydCA9IDQ7CiAgICBpbnQgc2Fib3RhZ2VSb3dzID0gMzE7CiAgICBmb3IgKGludCByID0gc2Fib3RhZ2VSb3dTdGFydDsgciA8IHNhYm90YWdlUm93U3RhcnQgKyBzYWJvdGFnZVJvd3MgJiYgciA8IG47IHIrKykgewogICAgICAgIGZvciAoaW50IGMgOiBzYWZlX2NvbHMpIHsKICAgICAgICAgICAgaWYgKGMgPT0ga25vd25TYWZlQ29sKSBjb250aW51ZTsgLy8gc2tpcCBrbm93biBzYWZlIGNvbHVtbi4KICAgICAgICAgICAgaWYgKG9mZnNldCA8IG1lc3NhZ2VMZW5ndGgpIHsKICAgICAgICAgICAgICAgIG1lc3NhZ2UucHVzaF9iYWNrKFJbcl1bY10pOwogICAgICAgICAgICAgICAgb2Zmc2V0Kys7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICB9CgogICAgaW50IGxlbmd0aFJvd1N0YXJ0ID0gc2Fib3RhZ2VSb3dTdGFydCArIHNhYm90YWdlUm93czsgLy8gNCArIDMxID0gMzUuCiAgICBpbnQgbGVuZ3RoUm93cyA9IDExOwogICAgZm9yIChpbnQgciA9IGxlbmd0aFJvd1N0YXJ0OyByIDwgbGVuZ3RoUm93U3RhcnQgKyBsZW5ndGhSb3dzICYmIHIgPCBuOyByKyspIHsKICAgICAgICBmb3IgKGludCBjIDogc2FmZV9jb2xzKSB7CiAgICAgICAgICAgIGlmIChjID09IGtub3duU2FmZUNvbCkgY29udGludWU7IC8vIHNraXAga25vd24gc2FmZSBjb2x1bW4uCiAgICAgICAgICAgIGlmIChvZmZzZXQgPCBtZXNzYWdlTGVuZ3RoKSB7CiAgICAgICAgICAgICAgICBtZXNzYWdlLnB1c2hfYmFjayhSW3JdW2NdKTsKICAgICAgICAgICAgICAgIG9mZnNldCsrOwogICAgICAgICAgICB9CiAgICAgICAgfQogICAgfQoKICAgIGludCByZXN0Um93U3RhcnQgPSBsZW5ndGhSb3dTdGFydCArIGxlbmd0aFJvd3M7IC8vIDQgKyAzMSArIDExID0gNDYuCiAgICBpbnQgZnVsbENvbHVtbnMgPSAxNjsgLy8gbnVtYmVyIG9mIHNhZmUgY29sdW1ucy4KICAgIGludCBtZXNzYWdlTGVmdCA9IG1lc3NhZ2VMZW5ndGggLSBvZmZzZXQ7CgogICAgaW50IHJvd3NGb3JNZXNzYWdlID0gKG1lc3NhZ2VMZWZ0ICsgZnVsbENvbHVtbnMgLSAxKSAvIGZ1bGxDb2x1bW5zOyAvLyBudW1iZXIgb2Ygcm93cyBuZWVkZWQuCiAgICBmb3IgKGludCByID0gcmVzdFJvd1N0YXJ0OyByIDwgcmVzdFJvd1N0YXJ0ICsgcm93c0Zvck1lc3NhZ2UgJiYgciA8IG47IHIrKykgewogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgZnVsbENvbHVtbnMgJiYgb2Zmc2V0IDwgbWVzc2FnZUxlbmd0aDsgaSsrKSB7CiAgICAgICAgICAgIGludCBjb2wgPSBzYWZlX2NvbHNbaV07CiAgICAgICAgICAgIG1lc3NhZ2UucHVzaF9iYWNrKFJbcl1bY29sXSk7CiAgICAgICAgICAgIG9mZnNldCsrOwogICAgICAgIH0KICAgIH0KCiAgICAvLyBUaGUgbWVzc2FnZSBpcyBub3cgaW4gdmVjdG9yIG1lc3NhZ2UuIElmIG9mZnNldCA8IG1lc3NhZ2VMZW5ndGgsIHdlIGRpZCBub3QgZ2V0IGFsbCBiaXRzLiBidXQgd2Ugc2hvdWxkLgogICAgcmV0dXJuIG1lc3NhZ2U7Cn0=)

#include "message.h"

#include <bits/stdc++.h>

using namespace std;

extern std::vector<bool> send\_packet(std::vector<bool> A);

void send\_message(std::vector<bool> M, std::vector<bool> C)

{

vector<int> safe\_cols;

vector<int> sabotage\_cols;

for (int i = 0; i < 31; i++) {

if (!C\[i\]) safe\_cols.push\_back(i);

else sabotage\_cols.push\_back(i);

}

// Step 1: send the first 4 packets with distinct sequences.

vector<vector<bool>> first4results;

first4results.reserve(4);

for (int p = 0; p < 4; p++) {

vector<bool> A(31, false);

for (int i = 0; i < 16; i++) {

int col = safe\_cols\[i\];

int bit = (i >> p) & 1; // p-th bit of i, i is the index in safe subset

A\[col\] = bit;

}

for (int col: sabotage\_cols) {

A\[col\] = false;

}

std::vector<bool> B = send\_packet(A);

first4results.push\_back(B);

}

// Now identify known safe column from the results of these 4 calls.

vector<string> colSequences(31, string(4, ’0’));

for (int c = 0; c < 31; c++) {

for (int r = 0; r < 4; r++) {

colSequences\[c\]\[r\] = first4results\[r\]\[c\]? ’1’: ’0’;

}

}

map<string, vector<int>> seqToCols;

for (int c = 0; c < 31; c++) {

seqToCols\[colSequences\[c\]\].push\_back(c);

}

int knownSafeCol = -1;

for (auto &kv: seqToCols) {

if (kv.second.size() == 1) {

int col = kv.second\[0\];

// check if col is in safe subset. Because known safe column must be a safe column.

if (find(safe\_cols.begin(), safe\_cols.end(), col)!= safe\_cols.end()) {

// also check if the sequence is one of the distinct sequences for safe subset. The index in the safe subset is i, the sequence is 4 bits representing i.

string seq = kv.first;

int val = 0;

for (int i = 0; i < 4; i++) {

if (seq\[i\] == ’1’) val |= (1 << i);

}

// check if val is < 16.

if (val < 16) {

knownSafeCol = col;

break;

}

}

}

}

if (knownSafeCol == -1) {

// If not found, pick a safe column that has a distinct sequence anyway.

// But for correctness, we require knownSafeCol to be found. If not found, our method has a problem.

}

// Step 2: store sabotage subset’s bits in the known safe column in next 31 calls. The other 15 safe columns store message bits if available.

int S = M.size(); // message length.

int offset = 0; // offset in M.

for (int i = 0; i < 31; i++) {

vector<bool> A(31, false);

int sabotageBit = C\[i\];

A\[knownSafeCol\] = sabotageBit; // store sabotage subset’s bit i in known safe column.

for (int c: safe\_cols) {

if (c == knownSafeCol) continue; // skip known safe column because we used it for sabotage subset.

int col = c;

if (offset < S) {

A\[col\] = M\[offset\];

offset++;

} else {

A\[col\] = false;

}

}

for (int sc: sabotage\_cols) {

A\[sc\] = false;

}

std::vector<bool> B = send\_packet(A);

// We do not store B because we do not need to adapt further. Actually, we can store B if we want to adapt but the solution does not require further adaptation.

}

// Step 3: store message length in the next 11 calls in the known safe column. The other 15 safe columns store more message bits if available.

int L = S; // length in integer form.

vector<int> lengthBinary(11, 0);

for (int i = 0; i < 11; i++) {

lengthBinary\[i\] = (L >> i) & 1;

}

for (int i = 0; i < 11; i++) {

vector<bool> A(31, false);

int bit = lengthBinary\[i\];

A\[knownSafeCol\] = bit; // store length bit i in known safe column.

for (int c: safe\_cols) {

if (c == knownSafeCol) continue; // skip known safe column.

int col = c;

if (offset < S) {

A\[col\] = M\[offset\];

offset++;

} else {

A\[col\] = false;

}

}

for (int sc: sabotage\_cols) {

A\[sc\] = false;

}

std::vector<bool> B = send\_packet(A);

}

// Step 4: store the rest of the message in subsequent calls using all 16 safe columns.

int messageLeft = S - offset; // how many bits are left to store.

int fullColumns = 16; // number of safe columns.

int messageRows = (messageLeft + fullColumns - 1) / fullColumns; // how many rows needed.

for (int r = 0; r < messageRows; r++) {

vector<bool> A(31, false);

for (int i = 0; i < fullColumns && offset + i < S; i++) {

int col = safe\_cols\[i\];

A\[col\] = M\[offset + i\];

}

offset += fullColumns;

for (int sc: sabotage\_cols) {

A\[sc\] = false;

}

std::vector<bool> B = send\_packet(A);

}

}

std::vector<bool> receive\_message(std::vector<std::vector<bool>> R)

{

int n = R.size();

vector<bool> result; // message to return.

if (n < 4) {

return result;

}

// Step 1: from the first 4 rows in R, find the known safe column.

vector<string> colSequences(31, string(4, ’0’));

for (int c = 0; c < 31; c++) {

for (int r = 0; r < 4 && r < n; r++) {

if (r < n) {

colSequences\[c\]\[r\] = R\[r\]\[c\]? ’1’: ’0’;

}

}

}

map<string, vector<int>> seqToCols;

for (int c = 0; c < 31; c++) {

seqToCols\[colSequences\[c\]\].push\_back(c);

}

int knownSafeCol = -1;

for (auto &kv: seqToCols) {

if (kv.second.size() == 1) {

int col = kv.second\[0\];

// check if the sequence is one of the binary representations from 0..15 because safe subset’s sequences are distinct and in that range.

string seq = kv.first;

int val = 0;

for (int i = 0; i < 4; i++) {

if (seq\[i\] == ’1’) val |= (1 << i);

}

if (val < 16) {

knownSafeCol = col;

break;

}

}

}

if (knownSafeCol == -1) {

// If none found, that means sabotage overcame the method or we made an error.

}

// Step 2: from next 31 rows in known safe column, read sabotage subset.

vector<bool> sabotageSubset(31, false);

int sabotageStart = 4;

for (int i = 0; i < 31 && sabotageStart + i < n; i++) {

sabotageSubset\[i\] = R\[sabotageStart + i\]\[knownSafeCol\];

}

// Step 3: from next 11 rows in known safe column, read length.

vector<bool> lengthBits(11, false);

int lengthStart = sabotageStart + 31;

for (int i = 0; i < 11 && lengthStart + i < n; i++) {

lengthBits\[i\] = R\[lengthStart + i\]\[knownSafeCol\];

}

int messageLength = 0;

for (int i = 0; i < 11; i++) {

messageLength |= (lengthBits\[i\] << i);

}

// Step 4: identify safe columns from sabotage subset.

vector<int> safe\_cols;

for (int i = 0; i < 31; i++) {

if (!sabotageSubset\[i\]) safe\_cols.push\_back(i);

}

// Step 5: read message from the next (31 + 11) rows’ other safe columns and from subsequent rows using all safe columns.

vector<bool> message;

int offset = 0; // offset in message.

int sabotageRowStart = 4;

int sabotageRows = 31;

for (int r = sabotageRowStart; r < sabotageRowStart + sabotageRows && r < n; r++) {

for (int c: safe\_cols) {

if (c == knownSafeCol) continue; // skip known safe column.

if (offset < messageLength) {

message.push\_back(R\[r\]\[c\]);

offset++;

}

}

}

int lengthRowStart = sabotageRowStart + sabotageRows; // 4 + 31 = 35.

int lengthRows = 11;

for (int r = lengthRowStart; r < lengthRowStart + lengthRows && r < n; r++) {

for (int c: safe\_cols) {

if (c == knownSafeCol) continue; // skip known safe column.

if (offset < messageLength) {

message.push\_back(R\[r\]\[c\]);

offset++;

}

}

}

int restRowStart = lengthRowStart + lengthRows; // 4 + 31 + 11 = 46.

int fullColumns = 16; // number of safe columns.

int messageLeft = messageLength - offset;

int rowsForMessage = (messageLeft + fullColumns - 1) / fullColumns; // number of rows needed.

for (int r = restRowStart; r < restRowStart + rowsForMessage && r < n; r++) {

for (int i = 0; i < fullColumns && offset < messageLength; i++) {

int col = safe\_cols\[i\];

message.push\_back(R\[r\]\[col\]);

offset++;

}

}

// The message is now in vector message. If offset < messageLength, we did not get all bits. but we should.

return message;

}

### C.3 Tree

A total of 30 points were scored on Tree across two separate submissions.

#### C.3.1 Submission 1

The first submission achieved a score of 17 out of 100, with points earned from subtasks 1 and 4.

[⬇](data:text/plain;base64,I2luY2x1ZGUgInRyZWUuaCIKI2luY2x1ZGUgPGJpdHMvc3RkYysrLmg+CnVzaW5nIG5hbWVzcGFjZSBzdGQ7CgpzdGF0aWMgaW50IE5HbG9iYWw7CnN0YXRpYyB2ZWN0b3I8aW50PiBwYXJlbnRHbG9iYWw7CnN0YXRpYyB2ZWN0b3I8aW50PiB3ZWlnaHRHbG9iYWw7CnN0YXRpYyB2ZWN0b3I8bG9uZyBsb25nPiBkZWx0YUdsb2JhbDsKc3RhdGljIHZlY3RvcjxpbnQ+IGxlYWZDb3VudEdsb2JhbDsKc3RhdGljIHZlY3RvcjxpbnQ+IHNvcnRlZEJ5TGVhZkNvdW50OyAvLyBXZSdsbCBzdG9yZSBub2RlIGluZGljZXMgc29ydGVkIGJ5IGxlYWYgY291bnQuCnN0YXRpYyB2ZWN0b3I8bG9uZyBsb25nPiBwcmVmaXhEZWx0YSwgcHJlZml4RGVsdGFUaW1lc0xlYWZDb3VudDsgLy8gcHJlZml4IHN1bXMgaW4gc29ydGVkIG9yZGVyIGJ5IGxlYWYgY291bnQuCnN0YXRpYyBsb25nIGxvbmcgc3VtQWxsRGVsdGE7CnN0YXRpYyB2ZWN0b3I8dmVjdG9yPGludD4+IGNoaWxkcmVuR2xvYmFsOwoKaW50IGNvbXB1dGVMZWFmQ291bnRSZWMoaW50IHYpewogICAgaWYgKGNoaWxkcmVuR2xvYmFsW3ZdLmVtcHR5KCkpeyAvLyBsZWFmCiAgICAgICAgbGVhZkNvdW50R2xvYmFsW3ZdID0gMTsKICAgICAgICByZXR1cm4gMTsKICAgIH0gZWxzZSB7CiAgICAgICAgaW50IHN1bSA9IDA7CiAgICAgICAgZm9yIChpbnQgYyA6IGNoaWxkcmVuR2xvYmFsW3ZdKSBzdW0gKz0gY29tcHV0ZUxlYWZDb3VudFJlYyhjKTsKICAgICAgICBsZWFmQ291bnRHbG9iYWxbdl0gPSBzdW07CiAgICAgICAgcmV0dXJuIHN1bTsKICAgIH0KfQoKdm9pZCBpbml0KHN0ZDo6dmVjdG9yPGludD4gUCwgc3RkOjp2ZWN0b3I8aW50PiBXKXsKICAgIGludCBOID0gKGludClQLnNpemUoKTsKICAgIE5HbG9iYWwgPSBOOwogICAgcGFyZW50R2xvYmFsID0gUDsKICAgIHdlaWdodEdsb2JhbCA9IFc7CiAgICBpZiAoTiA9PSAwKSByZXR1cm47CgogICAgY2hpbGRyZW5HbG9iYWwuYXNzaWduKE4sIHt9KTsKICAgIGZvciAoaW50IGk9MTsgaSA8IE47IGkrKykgY2hpbGRyZW5HbG9iYWxbcGFyZW50R2xvYmFsW2ldXS5wdXNoX2JhY2soaSk7CgogICAgbGVhZkNvdW50R2xvYmFsLnJlc2l6ZShOKTsKICAgIGNvbXB1dGVMZWFmQ291bnRSZWMoMCk7CgogICAgdmVjdG9yPGludD4gZGVnKE4sMCk7CiAgICBmb3IgKGludCBpPTE7IGkgPCBOOyBpKyspIGRlZ1twYXJlbnRHbG9iYWxbaV1dKys7CiAgICBkZWx0YUdsb2JhbC5yZXNpemUoTik7CiAgICBmb3IgKGludCBpPTA7IGkgPCBOOyBpKyspewogICAgICAgIGlmIChwYXJlbnRHbG9iYWxbaV0gPT0gLTEpeyAvLyByb290CiAgICAgICAgICAgIGlmIChkZWdbaV0gPT0gMCkgeyAvLyBsZWFmIHJvb3QKICAgICAgICAgICAgICAgIGRlbHRhR2xvYmFsW2ldID0gd2VpZ2h0R2xvYmFsW2ldOwogICAgICAgICAgICB9IGVsc2UgeyAvLyBpbnRlcm5hbCByb290CiAgICAgICAgICAgICAgICBkZWx0YUdsb2JhbFtpXSA9IC0gKGxvbmcgbG9uZyl3ZWlnaHRHbG9iYWxbaV07CiAgICAgICAgICAgIH0KICAgICAgICB9IGVsc2UgewogICAgICAgICAgICBpZiAoZGVnW2ldID09IDApIHsgLy8gbGVhZgogICAgICAgICAgICAgICAgZGVsdGFHbG9iYWxbaV0gPSAobG9uZyBsb25nKXdlaWdodEdsb2JhbFtwYXJlbnRHbG9iYWxbaV1dICsgKGxvbmcgbG9uZyl3ZWlnaHRHbG9iYWxbaV07CiAgICAgICAgICAgIH0gZWxzZSB7IC8vIGludGVybmFsCiAgICAgICAgICAgICAgICBkZWx0YUdsb2JhbFtpXSA9IChsb25nIGxvbmcpd2VpZ2h0R2xvYmFsW3BhcmVudEdsb2JhbFtpXV0gLSAobG9uZyBsb25nKXdlaWdodEdsb2JhbFtpXTsKICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0KCiAgICBzb3J0ZWRCeUxlYWZDb3VudC5yZXNpemUoTik7CiAgICBpb3RhKHNvcnRlZEJ5TGVhZkNvdW50LmJlZ2luKCksIHNvcnRlZEJ5TGVhZkNvdW50LmVuZCgpLCAwKTsKICAgIHNvcnQoc29ydGVkQnlMZWFmQ291bnQuYmVnaW4oKSwgc29ydGVkQnlMZWFmQ291bnQuZW5kKCksIFtdKGludCBhLCBpbnQgYil7CiAgICAgICAgcmV0dXJuIGxlYWZDb3VudEdsb2JhbFthXSA8IGxlYWZDb3VudEdsb2JhbFtiXTsKICAgIH0pOwogICAgcHJlZml4RGVsdGEucmVzaXplKE4sMCk7CiAgICBwcmVmaXhEZWx0YVRpbWVzTGVhZkNvdW50LnJlc2l6ZShOLDApOwogICAgc3VtQWxsRGVsdGEgPSAwOwogICAgZm9yIChpbnQgaT0wOyBpIDwgTjsgaSsrKSBzdW1BbGxEZWx0YSArPSBkZWx0YUdsb2JhbFtpXTsKICAgIGxvbmcgbG9uZyBydW5uaW5nRGVsdGFUaW1lc0xlYWYgPSAwOwogICAgbG9uZyBsb25nIHJ1bm5pbmdEZWx0YSA9IDA7CiAgICBmb3IgKGludCBpPTA7IGkgPCBOOyBpKyspewogICAgICAgIGludCBpZHggPSBzb3J0ZWRCeUxlYWZDb3VudFtpXTsKICAgICAgICBsb25nIGxvbmcgdmFsRGVsdGFUaW1lc0xlYWYgPSBkZWx0YUdsb2JhbFtpZHhdICogKGxvbmcgbG9uZylsZWFmQ291bnRHbG9iYWxbaWR4XTsKICAgICAgICBydW5uaW5nRGVsdGFUaW1lc0xlYWYgKz0gdmFsRGVsdGFUaW1lc0xlYWY7CiAgICAgICAgcnVubmluZ0RlbHRhICs9IGRlbHRhR2xvYmFsW2lkeF07CiAgICAgICAgcHJlZml4RGVsdGFUaW1lc0xlYWZDb3VudFtpXSA9IHJ1bm5pbmdEZWx0YVRpbWVzTGVhZjsKICAgICAgICBwcmVmaXhEZWx0YVtpXSA9IHJ1bm5pbmdEZWx0YTsKICAgIH0KfQoKbG9uZyBsb25nIHF1ZXJ5KGludCBMLCBpbnQgUil7CiAgICBpZiAoTkdsb2JhbCA9PSAwKSByZXR1cm4gMDsgLy8gbm8gbm9kZXMKICAgIGxvbmcgbG9uZyBhID0gUiAvIEw7IC8vIGZsb29yIG9mIFIvTC4KCiAgICBpbnQgbGVmdCA9IC0xOwogICAgaW50IHJpZ2h0ID0gTkdsb2JhbDsKICAgIHdoaWxlIChyaWdodCAtIGxlZnQgPiAxKXsKICAgICAgICBpbnQgbWlkID0gKGxlZnQgKyByaWdodCkvMjsKICAgICAgICBpbnQgbm9kZSA9IHNvcnRlZEJ5TGVhZkNvdW50W21pZF07CiAgICAgICAgaWYgKChsb25nIGxvbmcpbGVhZkNvdW50R2xvYmFsW25vZGVdIDw9IGEpIGxlZnQgPSBtaWQ7IGVsc2UgcmlnaHQgPSBtaWQ7CiAgICB9CiAgICBpbnQgaWR4ID0gbGVmdDsgLy8gaWR4IGlzIHRoZSBsYXN0IGluZGV4IHdoZXJlIGJbaV0gPD0gYS4KCiAgICBsb25nIGxvbmcgc3VtQSA9IDA7IC8vIHN1bSBvZiBkZWx0YVtpXSpiW2ldIGZvciBub2RlcyB3aXRoIGJbaV0gPD0gYS4KICAgIGlmIChpZHggPj0gMCkgewogICAgICAgIHN1bUEgPSBwcmVmaXhEZWx0YVRpbWVzTGVhZkNvdW50W2lkeF07IC8vIHN1bV97aTogYltpXSA8PSBhfSBkZWx0YVtpXSpiW2ldCiAgICB9CgogICAgbG9uZyBsb25nIHN1bVByZWZpeERlbHRhV2hlcmVMZXNzT3JFcXVhbCA9IDA7CiAgICBpZiAoaWR4ID49IDApIHN1bVByZWZpeERlbHRhV2hlcmVMZXNzT3JFcXVhbCA9IHByZWZpeERlbHRhW2lkeF07IGVsc2Ugc3VtUHJlZml4RGVsdGFXaGVyZUxlc3NPckVxdWFsID0gMDsKICAgIGxvbmcgbG9uZyBzdW1EZWx0YVdoZXJlQkdyZWF0ZXIgPSBzdW1BbGxEZWx0YSAtIHN1bVByZWZpeERlbHRhV2hlcmVMZXNzT3JFcXVhbDsKCiAgICBsb25nIGxvbmcgTExMID0gKGxvbmcgbG9uZylMOwogICAgbG9uZyBsb25nIFJSUiA9IChsb25nIGxvbmcpUjsKCiAgICBsb25nIGxvbmcgcGFydEEgPSBMTEwgKiBzdW1BOwogICAgbG9uZyBsb25nIHBhcnRCID0gUlJSICogc3VtRGVsdGFXaGVyZUJHcmVhdGVyOwoKICAgIGxvbmcgbG9uZyBjb3N0ID0gcGFydEEgKyBwYXJ0QjsKCiAgICByZXR1cm4gY29zdDsKfQ==)

#include "tree.h"

#include <bits/stdc++.h>

using namespace std;

static int NGlobal;

static vector<int> parentGlobal;

static vector<int> weightGlobal;

static vector<long long> deltaGlobal;

static vector<int> leafCountGlobal;

static vector<int> sortedByLeafCount; // We’ll store node indices sorted by leaf count.

static vector<long long> prefixDelta, prefixDeltaTimesLeafCount; // prefix sums in sorted order by leaf count.

static long long sumAllDelta;

static vector<vector<int>> childrenGlobal;

int computeLeafCountRec(int v){

if (childrenGlobal\[v\].empty()){ // leaf

leafCountGlobal\[v\] = 1;

return 1;

} else {

int sum = 0;

for (int c: childrenGlobal\[v\]) sum += computeLeafCountRec(c);

leafCountGlobal\[v\] = sum;

return sum;

}

}

void init(std::vector<int> P, std::vector<int> W){

int N = (int)P.size();

NGlobal = N;

parentGlobal = P;

weightGlobal = W;

if (N == 0) return;

childrenGlobal.assign(N, {});

for (int i=1; i < N; i++) childrenGlobal\[parentGlobal\[i\]\].push\_back(i);

leafCountGlobal.resize(N);

computeLeafCountRec(0);

vector<int> deg(N,0);

for (int i=1; i < N; i++) deg\[parentGlobal\[i\]\]++;

deltaGlobal.resize(N);

for (int i=0; i < N; i++){

if (parentGlobal\[i\] == -1){ // root

if (deg\[i\] == 0) { // leaf root

deltaGlobal\[i\] = weightGlobal\[i\];

} else { // internal root

deltaGlobal\[i\] = - (long long)weightGlobal\[i\];

}

} else {

if (deg\[i\] == 0) { // leaf

deltaGlobal\[i\] = (long long)weightGlobal\[parentGlobal\[i\]\] + (long long)weightGlobal\[i\];

} else { // internal

deltaGlobal\[i\] = (long long)weightGlobal\[parentGlobal\[i\]\] - (long long)weightGlobal\[i\];

}

}

}

sortedByLeafCount.resize(N);

iota(sortedByLeafCount.begin(), sortedByLeafCount.end(), 0);

sort(sortedByLeafCount.begin(), sortedByLeafCount.end(), \[\](int a, int b){

return leafCountGlobal\[a\] < leafCountGlobal\[b\];

});

prefixDelta.resize(N,0);

prefixDeltaTimesLeafCount.resize(N,0);

sumAllDelta = 0;

for (int i=0; i < N; i++) sumAllDelta += deltaGlobal\[i\];

long long runningDeltaTimesLeaf = 0;

long long runningDelta = 0;

for (int i=0; i < N; i++){

int idx = sortedByLeafCount\[i\];

long long valDeltaTimesLeaf = deltaGlobal\[idx\] \* (long long)leafCountGlobal\[idx\];

runningDeltaTimesLeaf += valDeltaTimesLeaf;

runningDelta += deltaGlobal\[idx\];

prefixDeltaTimesLeafCount\[i\] = runningDeltaTimesLeaf;

prefixDelta\[i\] = runningDelta;

}

}

long long query(int L, int R){

if (NGlobal == 0) return 0; // no nodes

long long a = R / L; // floor of R/L.

int left = -1;

int right = NGlobal;

while (right - left > 1){

int mid = (left + right)/2;

int node = sortedByLeafCount\[mid\];

if ((long long)leafCountGlobal\[node\] <= a) left = mid; else right = mid;

}

int idx = left; // idx is the last index where b\[i\] <= a.

long long sumA = 0; // sum of delta\[i\]\*b\[i\] for nodes with b\[i\] <= a.

if (idx >= 0) {

sumA = prefixDeltaTimesLeafCount\[idx\]; // sum\_{i: b\[i\] <= a} delta\[i\]\*b\[i\]

}

long long sumPrefixDeltaWhereLessOrEqual = 0;

if (idx >= 0) sumPrefixDeltaWhereLessOrEqual = prefixDelta\[idx\]; else sumPrefixDeltaWhereLessOrEqual = 0;

long long sumDeltaWhereBGreater = sumAllDelta - sumPrefixDeltaWhereLessOrEqual;

long long LLL = (long long)L;

long long RRR = (long long)R;

long long partA = LLL \* sumA;

long long partB = RRR \* sumDeltaWhereBGreater;

long long cost = partA + partB;

return cost;

}

#### C.3.2 Submission 2

Submission 2 achieved 13 of 100 points on subtask 2.

[⬇](data:text/plain;base64,I2luY2x1ZGUgInRyZWUuaCIKI2luY2x1ZGUgPGJpdHMvc3RkYysrLmg+CnVzaW5nIG5hbWVzcGFjZSBzdGQ7CnN0cnVjdCBQdCB7CiAgICBsb25nIGxvbmcgeDsKICAgIGxvbmcgbG9uZyB5Owp9OwpzdHJ1Y3QgUFdMIHsKICAgIGxvbmcgbG9uZyBMLCBSOwogICAgdmVjdG9yPFB0PiBwdHM7IC8vIHNvcnRlZCBieSB4Cn07CmludCBOR2xvYmFsOwppbnQgUEdsb2JhbFsyMDA1XTsKaW50IHdHbG9iYWxbMjAwNV07CnZlY3RvcjxpbnQ+IGNoaWxkcmVuR2xvYmFsWzIwMDVdOwpQV0wgR0dsb2JhbFsyMDA1XTsKCi8vIGRlZmluZSB0aGUgaGVscGVyIGZ1bmN0aW9ucyBhcyBzdGF0aWMuCgpzdGF0aWMgbG9uZyBsb25nIGV2YWxQV0woY29uc3QgUFdMICZmLCBsb25nIGxvbmcgeCkgewogICAgaWYgKGYucHRzLmVtcHR5KCkpIHJldHVybiBMTE9OR19NQVgvMjsgLy8gbm8gZnVuY3Rpb24KICAgIGlmICh4IDw9IGYucHRzLmZyb250KCkueCkgewogICAgICAgIHJldHVybiBmLnB0cy5mcm9udCgpLnk7IC8vIGRvbWFpbiBvdXQsIGJ1dCB3ZSBwcmVmZXIgZnJvbnQuCiAgICB9CiAgICBpZiAoeCA+PSBmLnB0cy5iYWNrKCkueCkgewogICAgICAgIHJldHVybiBmLnB0cy5iYWNrKCkueTsgLy8gZG9tYWluIG91dCwgYnV0IHByZWZlciBiYWNrLgogICAgfQoKICAgIGludCBsZWZ0ID0gMCwgcmlnaHQgPSBmLnB0cy5zaXplKCktMTsKICAgIHdoaWxlIChsZWZ0KzEgPCByaWdodCkgewogICAgICAgIGludCBtaWQgPSAobGVmdCArIHJpZ2h0KSAvIDI7CiAgICAgICAgaWYgKGYucHRzW21pZF0ueCA9PSB4KSByZXR1cm4gZi5wdHNbbWlkXS55OwogICAgICAgIGlmIChmLnB0c1ttaWRdLnggPCB4KSBsZWZ0ID0gbWlkOyBlbHNlIHJpZ2h0ID0gbWlkOwogICAgfQogICAgaWYgKGYucHRzW2xlZnRdLnggPD0geCAmJiB4IDw9IGYucHRzW3JpZ2h0XS54KSB7CiAgICAgICAgbG9uZyBsb25nIHgxID0gZi5wdHNbbGVmdF0ueDsKICAgICAgICBsb25nIGxvbmcgeTEgPSBmLnB0c1tsZWZ0XS55OwogICAgICAgIGxvbmcgbG9uZyB4MiA9IGYucHRzW3JpZ2h0XS54OwogICAgICAgIGxvbmcgbG9uZyB5MiA9IGYucHRzW3JpZ2h0XS55OwogICAgICAgIGlmICh4MSA9PSB4MikgcmV0dXJuIHkxOwogICAgICAgIGxvbmcgZG91YmxlIHNsb3BlID0gKGxvbmcgZG91YmxlKSh5MiAtIHkxKS8obG9uZyBkb3VibGUpKHgyIC0geDEpOwogICAgICAgIGxvbmcgZG91YmxlIHZhbGYgPSAobG9uZyBkb3VibGUpIHkxICsgc2xvcGUqKHggLSB4MSk7CiAgICAgICAgbG9uZyBsb25nIHZhbGkgPSAobG9uZyBsb25nKSBsbHJvdW5kKHZhbGYpOwogICAgICAgIHJldHVybiB2YWxpOwogICAgfSBlbHNlIHsKICAgICAgICByZXR1cm4gTExPTkdfTUFYLzI7CiAgICB9Cn0Kc3RydWN0IENoaWxkU3RhdGUgewogICAgaW50IGlkeDsgLy8gY2hpbGQgaW5kZXggaW4gbG9jYWwgYXJyYXkKICAgIGxvbmcgbG9uZyB4OyAvLyBjdXJyZW50IHggaW4gY2hpbGQncyBkb21haW4KICAgIGludCBzZWc7IC8vIHNlZ21lbnQgaW4gY2hpbGQncyBmdW5jdGlvbgogICAgbG9uZyBkb3VibGUgc2xvcGU7CiAgICBsb25nIGxvbmcgbmV4dEJvdW5kOwp9OwoKc3RhdGljIFBXTCBjb21iaW5lQ2hpbGRyZW4oY29uc3QgdmVjdG9yPFBXTD4gJmNoaWxkRiwgbG9uZyBsb25nIEwsIGxvbmcgbG9uZyBSKSB7CiAgICBpbnQgZCA9IGNoaWxkRi5zaXplKCk7CiAgICBpZiAoZCA9PSAwKSB7CiAgICAgICAgUFdMIGY7IGYuTCA9IEw7IGYuUiA9IFI7IGYucHRzID0ge3tMLCAwfSwge1IsIDB9fTsgcmV0dXJuIGY7CiAgICB9CiAgICBsb25nIGxvbmcgc3VtRG9tYWluTCA9IChsb25nIGxvbmcpZCAqIEw7CiAgICBsb25nIGxvbmcgc3VtRG9tYWluUiA9IChsb25nIGxvbmcpZCAqIFI7CgogICAgdmVjdG9yPENoaWxkU3RhdGU+IGNzKGQpOwogICAgcHJpb3JpdHlfcXVldWU8cGFpcjxsb25nIGRvdWJsZSwgaW50PiwgdmVjdG9yPHBhaXI8bG9uZyBkb3VibGUsIGludD4+LCBncmVhdGVyPHBhaXI8bG9uZyBkb3VibGUsIGludD4+PiBwcTsKCiAgICBmb3IgKGludCBpID0gMDsgaSA8IGQ7IGkrKykgewogICAgICAgIGNzW2ldLmlkeCA9IGk7CiAgICAgICAgY3NbaV0ueCA9IEw7CiAgICAgICAgaW50IHNlZyA9IDA7IGludCBuID0gY2hpbGRGW2ldLnB0cy5zaXplKCk7CiAgICAgICAgd2hpbGUgKHNlZyA8IG4tMSAmJiBjaGlsZEZbaV0ucHRzW3NlZysxXS54IDw9IEwpIHNlZysrOwogICAgICAgIGNzW2ldLnNlZyA9IHNlZzsKICAgICAgICBsb25nIGRvdWJsZSBzbG9wZSA9IDA7CiAgICAgICAgaWYgKHNlZysxIDwgbikgewogICAgICAgICAgICBsb25nIGxvbmcgZHggPSBjaGlsZEZbaV0ucHRzW3NlZysxXS54IC0gY2hpbGRGW2ldLnB0c1tzZWddLng7CiAgICAgICAgICAgIGxvbmcgbG9uZyBkeSA9IGNoaWxkRltpXS5wdHNbc2VnKzFdLnkgLSBjaGlsZEZbaV0ucHRzW3NlZ10ueTsKICAgICAgICAgICAgc2xvcGUgPSBkeCE9MD8obG9uZyBkb3VibGUpZHkvZHg6MDsKICAgICAgICB9IGVsc2UgewogICAgICAgICAgICBzbG9wZSA9IDA7CiAgICAgICAgfQogICAgICAgIGNzW2ldLnNsb3BlID0gc2xvcGU7CiAgICAgICAgaWYgKHNlZysxIDwgbikgY3NbaV0ubmV4dEJvdW5kID0gbWluKChsb25nIGxvbmcpY2hpbGRGW2ldLnB0c1tzZWcrMV0ueCwgUik7CiAgICAgICAgZWxzZSBjc1tpXS5uZXh0Qm91bmQgPSBSOwogICAgICAgIHBxLnB1c2goe3Nsb3BlLCBpfSk7CiAgICB9CgogICAgbG9uZyBsb25nIHN1bSA9IHN1bURvbWFpbkw7CiAgICBsb25nIGxvbmcgY29zdCA9IDA7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IGQ7IGkrKykgewogICAgICAgIGNvc3QgKz0gZXZhbFBXTChjaGlsZEZbaV0sIEwpOwogICAgfQoKICAgIHZlY3RvcjxQdD4gZnB0czsKICAgIGZwdHMucHVzaF9iYWNrKHtzdW0sIGNvc3R9KTsKCiAgICB3aGlsZSAoIXBxLmVtcHR5KCkgJiYgc3VtIDwgc3VtRG9tYWluUikgewogICAgICAgIGF1dG8gdG9wID0gcHEudG9wKCk7CiAgICAgICAgbG9uZyBkb3VibGUgc2xvcGUgPSB0b3AuZmlyc3Q7CiAgICAgICAgdmVjdG9yPGludD4gZ3JvdXA7CiAgICAgICAgd2hpbGUgKCFwcS5lbXB0eSgpICYmIGFicyhwcS50b3AoKS5maXJzdCAtIHNsb3BlKSA8IDFlLTkpIHsKICAgICAgICAgICAgZ3JvdXAucHVzaF9iYWNrKHBxLnRvcCgpLnNlY29uZCk7CiAgICAgICAgICAgIHBxLnBvcCgpOwogICAgICAgIH0KICAgICAgICBpZiAoZ3JvdXAuZW1wdHkoKSkgYnJlYWs7CgogICAgICAgIGxvbmcgbG9uZyBpbmMgPSBMTE9OR19NQVg7CiAgICAgICAgZm9yIChpbnQgaWR4IDogZ3JvdXApIHsKICAgICAgICAgICAgbG9uZyBsb25nIG5ld0luYyA9IGNzW2lkeF0ubmV4dEJvdW5kIC0gY3NbaWR4XS54OwogICAgICAgICAgICBpZiAobmV3SW5jIDwgaW5jKSBpbmMgPSBuZXdJbmM7CiAgICAgICAgfQogICAgICAgIGlmIChpbmMgPD0gMCkgewogICAgICAgICAgICBmb3IgKGludCBpZHggOiBncm91cCkgewogICAgICAgICAgICAgICAgcHEucHVzaCh7Y3NbaWR4XS5zbG9wZSwgaWR4fSk7CiAgICAgICAgICAgIH0KICAgICAgICAgICAgYnJlYWs7CiAgICAgICAgfQogICAgICAgIGxvbmcgbG9uZyBmZWFzaWJsZUluYyA9IGluYzsKICAgICAgICBpZiAoc3VtICsgKGxvbmcgbG9uZylncm91cC5zaXplKCkgKiBpbmMgPiBzdW1Eb21haW5SKSB7CiAgICAgICAgICAgIGZlYXNpYmxlSW5jID0gKHN1bURvbWFpblIgLSBzdW0pIC8gKGxvbmcgbG9uZykgZ3JvdXAuc2l6ZSgpOwogICAgICAgIH0KICAgICAgICBpZiAoZmVhc2libGVJbmMgPD0gMCkgewogICAgICAgICAgICBmb3IgKGludCBpZHggOiBncm91cCkgewogICAgICAgICAgICAgICAgcHEucHVzaCh7Y3NbaWR4XS5zbG9wZSwgaWR4fSk7CiAgICAgICAgICAgIH0KICAgICAgICAgICAgYnJlYWs7CiAgICAgICAgfQoKICAgICAgICBzdW0gPSBzdW0gKyBncm91cC5zaXplKCkgKiBmZWFzaWJsZUluYzsKICAgICAgICBsb25nIGRvdWJsZSBkY29zdCA9IHNsb3BlICogZmVhc2libGVJbmMgKiBncm91cC5zaXplKCk7CiAgICAgICAgY29zdCA9IChsb25nIGxvbmcpIGxscm91bmQoKGxvbmcgZG91YmxlKWNvc3QgKyBkY29zdCk7CgogICAgICAgIGZwdHMucHVzaF9iYWNrKHtzdW0sIGNvc3R9KTsKCiAgICAgICAgZm9yIChpbnQgaWR4IDogZ3JvdXApIHsKICAgICAgICAgICAgY3NbaWR4XS54ICs9IGZlYXNpYmxlSW5jOwogICAgICAgICAgICBpZiAoY3NbaWR4XS54ID09IGNzW2lkeF0ubmV4dEJvdW5kKSB7CiAgICAgICAgICAgICAgICBjb25zdCBQV0wgJmcgPSBjaGlsZEZbaWR4XTsKICAgICAgICAgICAgICAgIGludCBzZWcgPSBjc1tpZHhdLnNlZzsKICAgICAgICAgICAgICAgIGlmIChjc1tpZHhdLnggPT0gUikgewogICAgICAgICAgICAgICAgICAgIGNzW2lkeF0uc2xvcGUgPSAxZTk7IC8vIHNhdHVyYXRlCiAgICAgICAgICAgICAgICAgICAgY3NbaWR4XS5uZXh0Qm91bmQgPSBSOwogICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICBzZWcrKzsKICAgICAgICAgICAgICAgICAgICBjc1tpZHhdLnNlZyA9IHNlZzsKICAgICAgICAgICAgICAgICAgICBpbnQgbiA9IGcucHRzLnNpemUoKTsKICAgICAgICAgICAgICAgICAgICBpZiAoc2VnKzEgPCBuKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGxvbmcgbG9uZyBkeCA9IGcucHRzW3NlZysxXS54IC0gZy5wdHNbc2VnXS54OwogICAgICAgICAgICAgICAgICAgICAgICBsb25nIGxvbmcgZHkgPSBnLnB0c1tzZWcrMV0ueSAtIGcucHRzW3NlZ10ueTsKICAgICAgICAgICAgICAgICAgICAgICAgbG9uZyBkb3VibGUgc2xvcGVDID0gZHghPTA/KGxvbmcgZG91YmxlKWR5L2R4OjA7CiAgICAgICAgICAgICAgICAgICAgICAgIGNzW2lkeF0uc2xvcGUgPSBzbG9wZUM7CiAgICAgICAgICAgICAgICAgICAgICAgIGNzW2lkeF0ubmV4dEJvdW5kID0gbWluKChsb25nIGxvbmcpZy5wdHNbc2VnKzFdLngsIFIpOwogICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGNzW2lkeF0uc2xvcGUgPSAwOwogICAgICAgICAgICAgICAgICAgICAgICBjc1tpZHhdLm5leHRCb3VuZCA9IFI7CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9CiAgICAgICAgfQogICAgICAgIGZvciAoaW50IGlkeCA6IGdyb3VwKSB7CiAgICAgICAgICAgIGlmIChjc1tpZHhdLnggPCBSKSB7CiAgICAgICAgICAgICAgICBwcS5wdXNoKHtjc1tpZHhdLnNsb3BlLCBpZHh9KTsKICAgICAgICAgICAgfQogICAgICAgIH0KICAgIH0KCiAgICB2ZWN0b3I8UHQ+IHVuaWZ5OwogICAgdW5pZnkucHVzaF9iYWNrKGZwdHNbMF0pOwogICAgZm9yIChpbnQgaSA9IDE7IGkgPCBmcHRzLnNpemUoKTsgaSsrKSB7CiAgICAgICAgaWYgKGZwdHNbaV0ueCA9PSB1bmlmeS5iYWNrKCkueCkgewogICAgICAgICAgICB1bmlmeS5iYWNrKCkueSA9IG1pbih1bmlmeS5iYWNrKCkueSwgZnB0c1tpXS55KTsKICAgICAgICB9IGVsc2UgdW5pZnkucHVzaF9iYWNrKGZwdHNbaV0pOwogICAgfQoKICAgIHZlY3RvcjxQdD4gZmluYWw7CiAgICBpZiAoIXVuaWZ5LmVtcHR5KCkpIGZpbmFsLnB1c2hfYmFjayh1bmlmeVswXSk7CiAgICBmb3IgKGludCBpID0gMTsgaSA8IHVuaWZ5LnNpemUoKS0xOyBpKyspIHsKICAgICAgICBsb25nIGxvbmcgeDEgPSBmaW5hbC5iYWNrKCkueDsKICAgICAgICBsb25nIGxvbmcgeTEgPSBmaW5hbC5iYWNrKCkueTsKICAgICAgICBsb25nIGxvbmcgeDIgPSB1bmlmeVtpXS54OwogICAgICAgIGxvbmcgbG9uZyB5MiA9IHVuaWZ5W2ldLnk7CiAgICAgICAgbG9uZyBsb25nIHgzID0gdW5pZnlbaSsxXS54OwogICAgICAgIGxvbmcgbG9uZyB5MyA9IHVuaWZ5W2krMV0ueTsKICAgICAgICBsb25nIGRvdWJsZSBzbG9wZTEgPSAoeDIhPXgxKT8gKGxvbmcgZG91YmxlKSh5Mi15MSkvKHgyLXgxKSA6IDFlOTsKICAgICAgICBsb25nIGRvdWJsZSBzbG9wZTIgPSAoeDMhPXgyKT8gKGxvbmcgZG91YmxlKSh5My15MikvKHgzLXgyKSA6IDFlOTsKICAgICAgICBpZiAoYWJzKHNsb3BlMSAtIHNsb3BlMikgPCAxZS05KSB7CiAgICAgICAgICAgIC8vIHVuaWZ5IGxpbmVhciwgc2tpcCB1bmlmeVtpXQogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIGZpbmFsLnB1c2hfYmFjayh1bmlmeVtpXSk7CiAgICAgICAgfQogICAgfQogICAgaWYgKCF1bmlmeS5lbXB0eSgpKSBmaW5hbC5wdXNoX2JhY2sodW5pZnkuYmFjaygpKTsKCiAgICBQV0wgcmVzdWx0OwogICAgcmVzdWx0LkwgPSBzdW1Eb21haW5MOyByZXN1bHQuUiA9IHN1bURvbWFpblI7CiAgICByZXN1bHQucHRzID0gZmluYWw7CgogICAgcmV0dXJuIHJlc3VsdDsKfQoKc3RhdGljIFBXTCBwYXJlbnRGb3JtdWxhKGNvbnN0IFBXTCAmZiwgbG9uZyBsb25nIHdWYWwsIGxvbmcgbG9uZyBMLCBsb25nIGxvbmcgUikgewogICAgdmVjdG9yPGxvbmcgbG9uZz4gY2FuZFZhbDsKICAgIGNhbmRWYWwucHVzaF9iYWNrKEwpOwogICAgY2FuZFZhbC5wdXNoX2JhY2soUik7CiAgICBmb3IgKGF1dG8gJnAgOiBmLnB0cykgewogICAgICAgIGlmIChwLnggPj0gTCAmJiBwLnggPD0gUikgY2FuZFZhbC5wdXNoX2JhY2socC54KTsKICAgIH0KICAgIGlmIChmLkwgPj0gTCAmJiBmLkwgPD0gUikgY2FuZFZhbC5wdXNoX2JhY2soZi5MKTsKICAgIGlmIChmLlIgPj0gTCAmJiBmLlIgPD0gUikgY2FuZFZhbC5wdXNoX2JhY2soZi5SKTsKCiAgICBmb3IgKGludCBpID0gMDsgaSA8IChpbnQpZi5wdHMuc2l6ZSgpLTE7IGkrKykgewogICAgICAgIGxvbmcgbG9uZyBzeCA9IGYucHRzW2ldLng7CiAgICAgICAgbG9uZyBsb25nIHN5ID0gZi5wdHNbaSsxXS54OwogICAgICAgIGxvbmcgbG9uZyBkeCA9IHN5IC0gc3g7CiAgICAgICAgbG9uZyBsb25nIGR5ID0gZi5wdHNbaSsxXS55IC0gZi5wdHNbaV0ueTsKICAgICAgICBsb25nIGRvdWJsZSBhID0gZHghPTA/IChsb25nIGRvdWJsZSlkeS9keDowOyAvLyBzbG9wZSBpbiBzIGRvbWFpbgogICAgICAgIGlmIChhIDw9IHdWYWwgJiYgYSA+PSAtd1ZhbCkgewogICAgICAgICAgICBsb25nIGxvbmcgaW5MID0gbWF4KHN4LCAobG9uZyBsb25nKUwpOwogICAgICAgICAgICBsb25nIGxvbmcgaW5SID0gbWluKHN5LCAobG9uZyBsb25nKVIpOwogICAgICAgICAgICBpZiAoaW5MIDw9IGluUikgewogICAgICAgICAgICAgICAgY2FuZFZhbC5wdXNoX2JhY2soaW5MKTsKICAgICAgICAgICAgICAgIGNhbmRWYWwucHVzaF9iYWNrKGluUik7CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICB9CgogICAgc29ydChjYW5kVmFsLmJlZ2luKCksIGNhbmRWYWwuZW5kKCkpOwogICAgY2FuZFZhbC5lcmFzZSh1bmlxdWUoY2FuZFZhbC5iZWdpbigpLCBjYW5kVmFsLmVuZCgpKSwgY2FuZFZhbC5lbmQoKSk7CgogICAgdmVjdG9yPFB0PiBwdHM7CiAgICBmb3IgKGF1dG8gdmFsIDogY2FuZFZhbCkgewogICAgICAgIGxvbmcgbG9uZyBiZXN0Q29zdCA9IExMT05HX01BWC8yOwogICAgICAgIGlmICh2YWwgPj0gZi5MICYmIHZhbCA8PSBmLlIpIHsKICAgICAgICAgICAgaW50IGxlZnQgPSAwLCByaWdodCA9IGYucHRzLnNpemUoKS0xOwogICAgICAgICAgICB3aGlsZSAobGVmdCsxIDwgcmlnaHQpIHsKICAgICAgICAgICAgICAgIGludCBtaWQgPSAobGVmdCArIHJpZ2h0KSAvIDI7CiAgICAgICAgICAgICAgICBpZiAoZi5wdHNbbWlkXS54IDw9IHZhbCkgbGVmdCA9IG1pZDsgZWxzZSByaWdodCA9IG1pZDsKICAgICAgICAgICAgfQogICAgICAgICAgICBpZiAoZi5wdHNbbGVmdF0ueCA8PSB2YWwgJiYgdmFsIDw9IGYucHRzW3JpZ2h0XS54KSB7CiAgICAgICAgICAgICAgICBsb25nIGxvbmcgeDEgPSBmLnB0c1tsZWZ0XS54OwogICAgICAgICAgICAgICAgbG9uZyBsb25nIHkxID0gZi5wdHNbbGVmdF0ueTsKICAgICAgICAgICAgICAgIGxvbmcgbG9uZyB4MiA9IGYucHRzW3JpZ2h0XS54OwogICAgICAgICAgICAgICAgbG9uZyBsb25nIHkyID0gZi5wdHNbcmlnaHRdLnk7CiAgICAgICAgICAgICAgICBsb25nIGRvdWJsZSBzbG9wZSA9ICh4MiE9eDE/IChsb25nIGRvdWJsZSkoeTIgLSB5MSkvKGxvbmcgZG91YmxlKSh4MiAtIHgxKSA6IDApOwogICAgICAgICAgICAgICAgaWYgKHNsb3BlIDw9IHdWYWwgJiYgc2xvcGUgPj0gLXdWYWwgJiYgdmFsID49IHgxICYmIHZhbCA8PSB4MikgewogICAgICAgICAgICAgICAgICAgIGxvbmcgZG91YmxlIGNvc3RmID0gKGxvbmcgZG91YmxlKSB5MSArIHNsb3BlICogKGxvbmcgZG91YmxlKSh2YWwgLSB4MSk7CiAgICAgICAgICAgICAgICAgICAgbG9uZyBsb25nIGMgPSAobG9uZyBsb25nKSBsbHJvdW5kKGNvc3RmKTsKICAgICAgICAgICAgICAgICAgICBiZXN0Q29zdCA9IG1pbihiZXN0Q29zdCwgYyk7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgZm9yIChhdXRvICZwIDogZi5wdHMpIHsKICAgICAgICAgICAgbG9uZyBsb25nIHMgPSBwLng7CiAgICAgICAgICAgIGxvbmcgbG9uZyBjID0gcC55ICsgd1ZhbCAqIGxsYWJzKHZhbCAtIHMpOwogICAgICAgICAgICBpZiAoYyA8IGJlc3RDb3N0KSBiZXN0Q29zdCA9IGM7CiAgICAgICAgfQogICAgICAgIHB0cy5wdXNoX2JhY2soe3ZhbCwgYmVzdENvc3R9KTsKICAgIH0KCiAgICBzb3J0KHB0cy5iZWdpbigpLCBwdHMuZW5kKCksIFtdKGNvbnN0IFB0ICZhLCBjb25zdCBQdCAmYil7cmV0dXJuIGEueCA8IGIueDt9KTsKCiAgICB2ZWN0b3I8UHQ+IHVuaWZ5OwogICAgdW5pZnkucHVzaF9iYWNrKHB0c1swXSk7CiAgICBmb3IgKGludCBpID0gMTsgaSA8IHB0cy5zaXplKCk7IGkrKykgewogICAgICAgIGlmIChwdHNbaV0ueCA9PSB1bmlmeS5iYWNrKCkueCkgewogICAgICAgICAgICB1bmlmeS5iYWNrKCkueSA9IG1pbih1bmlmeS5iYWNrKCkueSwgcHRzW2ldLnkpOwogICAgICAgIH0gZWxzZSB1bmlmeS5wdXNoX2JhY2socHRzW2ldKTsKICAgIH0KCiAgICB2ZWN0b3I8UHQ+IGZpbmFsOwogICAgaWYgKCF1bmlmeS5lbXB0eSgpKSBmaW5hbC5wdXNoX2JhY2sodW5pZnlbMF0pOwogICAgZm9yIChpbnQgaSA9IDE7IGkgPCB1bmlmeS5zaXplKCktMTsgaSsrKSB7CiAgICAgICAgbG9uZyBsb25nIHgxID0gZmluYWwuYmFjaygpLng7CiAgICAgICAgbG9uZyBsb25nIHkxID0gZmluYWwuYmFjaygpLnk7CiAgICAgICAgbG9uZyBsb25nIHgyID0gdW5pZnlbaV0ueDsKICAgICAgICBsb25nIGxvbmcgeTIgPSB1bmlmeVtpXS55OwogICAgICAgIGxvbmcgbG9uZyB4MyA9IHVuaWZ5W2krMV0ueDsKICAgICAgICBsb25nIGxvbmcgeTMgPSB1bmlmeVtpKzFdLnk7CiAgICAgICAgbG9uZyBkb3VibGUgc2xvcGUxID0geDIhPXgxPyAobG9uZyBkb3VibGUpKHkyLXkxKS8oeDIteDEpIDogMWU5OwogICAgICAgIGxvbmcgZG91YmxlIHNsb3BlMiA9IHgzIT14Mj8gKGxvbmcgZG91YmxlKSh5My15MikvKHgzLXgyKSA6IDFlOTsKICAgICAgICBpZiAoYWJzKHNsb3BlMSAtIHNsb3BlMikgPCAxZS05KSB7CiAgICAgICAgICAgIC8vIHVuaWZ5IGxpbmVhciwgc2tpcCB1bmlmeVtpXQogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIGZpbmFsLnB1c2hfYmFjayh1bmlmeVtpXSk7CiAgICAgICAgfQogICAgfQogICAgaWYgKCF1bmlmeS5lbXB0eSgpKSBmaW5hbC5wdXNoX2JhY2sodW5pZnkuYmFjaygpKTsKCiAgICBQV0wgcmVzdWx0OwogICAgcmVzdWx0LkwgPSBMOyByZXN1bHQuUiA9IFI7CiAgICByZXN1bHQucHRzID0gZmluYWw7CgogICAgcmV0dXJuIHJlc3VsdDsKfQoKc3RhdGljIHZvaWQgY29tcHV0ZUcoaW50IG5vZGUsIGxvbmcgbG9uZyBMLCBsb25nIGxvbmcgUikgewogICAgZm9yIChpbnQgYyA6IGNoaWxkcmVuR2xvYmFsW25vZGVdKSB7CiAgICAgICAgY29tcHV0ZUcoYywgTCwgUik7CiAgICB9CiAgICBpZiAoY2hpbGRyZW5HbG9iYWxbbm9kZV0uZW1wdHkoKSkgewogICAgICAgIC8vIGxlYWYKICAgICAgICBQV0wgcmVzdWx0OwogICAgICAgIHJlc3VsdC5MID0gTDsgcmVzdWx0LlIgPSBSOwogICAgICAgIGlmIChMIDw9IDAgJiYgMCA8PSBSKSB7CiAgICAgICAgICAgIGxvbmcgbG9uZyB2YWxMID0gd0dsb2JhbFtub2RlXSAqIGxsYWJzKEwpOwogICAgICAgICAgICBsb25nIGxvbmcgdmFsMCA9IHdHbG9iYWxbbm9kZV0gKiAwOyAvLyAwCiAgICAgICAgICAgIGxvbmcgbG9uZyB2YWxSID0gd0dsb2JhbFtub2RlXSAqIGxsYWJzKFIpOwogICAgICAgICAgICByZXN1bHQucHRzLmNsZWFyKCk7CiAgICAgICAgICAgIHJlc3VsdC5wdHMucHVzaF9iYWNrKHtMLCB2YWxMfSk7CiAgICAgICAgICAgIHJlc3VsdC5wdHMucHVzaF9iYWNrKHswLCB2YWwwfSk7CiAgICAgICAgICAgIHJlc3VsdC5wdHMucHVzaF9iYWNrKHtSLCB2YWxSfSk7CiAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgbG9uZyBsb25nIHZhbEwgPSB3R2xvYmFsW25vZGVdICogbGxhYnMoTCk7CiAgICAgICAgICAgIGxvbmcgbG9uZyB2YWxSID0gd0dsb2JhbFtub2RlXSAqIGxsYWJzKFIpOwogICAgICAgICAgICByZXN1bHQucHRzLmNsZWFyKCk7CiAgICAgICAgICAgIHJlc3VsdC5wdHMucHVzaF9iYWNrKHtMLCB2YWxMfSk7CiAgICAgICAgICAgIHJlc3VsdC5wdHMucHVzaF9iYWNrKHtSLCB2YWxSfSk7CiAgICAgICAgfQogICAgICAgIEdHbG9iYWxbbm9kZV0gPSByZXN1bHQ7CiAgICB9IGVsc2UgewogICAgICAgIHZlY3RvcjxQV0w+IGNoaWxkRjsKICAgICAgICBmb3IgKGludCBjIDogY2hpbGRyZW5HbG9iYWxbbm9kZV0pIHsKICAgICAgICAgICAgY2hpbGRGLnB1c2hfYmFjayhHR2xvYmFsW2NdKTsKICAgICAgICB9CiAgICAgICAgUFdMIGYgPSBjb21iaW5lQ2hpbGRyZW4oY2hpbGRGLCBMLCBSKTsKICAgICAgICBQV0wgcCA9IHBhcmVudEZvcm11bGEoZiwgd0dsb2JhbFtub2RlXSwgTCwgUik7CiAgICAgICAgR0dsb2JhbFtub2RlXSA9IHA7CiAgICB9Cn0KCnZvaWQgaW5pdChzdGQ6OnZlY3RvcjxpbnQ+IFAsIHN0ZDo6dmVjdG9yPGludD4gVykgewogICAgTkdsb2JhbCA9IFcuc2l6ZSgpOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCBOR2xvYmFsOyBpKyspIHsKICAgICAgICBQR2xvYmFsW2ldID0gUFtpXTsKICAgICAgICB3R2xvYmFsW2ldID0gV1tpXTsKICAgIH0KICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTkdsb2JhbDsgaSsrKSBjaGlsZHJlbkdsb2JhbFtpXS5jbGVhcigpOwogICAgZm9yIChpbnQgaSA9IDE7IGkgPCBOR2xvYmFsOyBpKyspIHsKICAgICAgICBjaGlsZHJlbkdsb2JhbFtQW2ldXS5wdXNoX2JhY2soaSk7CiAgICB9Cn0KCmxvbmcgbG9uZyBxdWVyeShpbnQgTCwgaW50IFIpIHsKICAgIGNvbXB1dGVHKDAsIEwsIFIpOwogICAgbG9uZyBsb25nIGFucyA9IExMT05HX01BWDsKICAgIGZvciAoYXV0byAmcCA6IEdHbG9iYWxbMF0ucHRzKSB7CiAgICAgICAgaWYgKHAueSA8IGFucykgYW5zID0gcC55OwogICAgfQogICAgcmV0dXJuIGFuczsKfQ==)

#include "tree.h"

#include <bits/stdc++.h>

using namespace std;

struct Pt {

long long x;

long long y;

};

struct PWL {

long long L, R;

vector<Pt> pts; // sorted by x

};

int NGlobal;

int PGlobal\[2005\];

int wGlobal\[2005\];

vector<int> childrenGlobal\[2005\];

PWL GGlobal\[2005\];

// define the helper functions as static.

static long long evalPWL(const PWL &f, long long x) {

if (f.pts.empty()) return LLONG\_MAX/2; // no function

if (x <= f.pts.front().x) {

return f.pts.front().y; // domain out, but we prefer front.

}

if (x >= f.pts.back().x) {

return f.pts.back().y; // domain out, but prefer back.

}

int left = 0, right = f.pts.size()-1;

while (left+1 < right) {

int mid = (left + right) / 2;

if (f.pts\[mid\].x == x) return f.pts\[mid\].y;

if (f.pts\[mid\].x < x) left = mid; else right = mid;

}

if (f.pts\[left\].x <= x && x <= f.pts\[right\].x) {

long long x1 = f.pts\[left\].x;

long long y1 = f.pts\[left\].y;

long long x2 = f.pts\[right\].x;

long long y2 = f.pts\[right\].y;

if (x1 == x2) return y1;

long double slope = (long double)(y2 - y1)/(long double)(x2 - x1);

long double valf = (long double) y1 + slope\*(x - x1);

long long vali = (long long) llround(valf);

return vali;

} else {

return LLONG\_MAX/2;

}

}

struct ChildState {

int idx; // child index in local array

long long x; // current x in child’s domain

int seg; // segment in child’s function

long double slope;

long long nextBound;

};

static PWL combineChildren(const vector<PWL> &childF, long long L, long long R) {

int d = childF.size();

if (d == 0) {

PWL f; f.L = L; f.R = R; f.pts = {{L, 0}, {R, 0}}; return f;

}

long long sumDomainL = (long long)d \* L;

long long sumDomainR = (long long)d \* R;

vector<ChildState> cs(d);

priority\_queue<pair<long double, int>, vector<pair<long double, int>>, greater<pair<long double, int>>> pq;

for (int i = 0; i < d; i++) {

cs\[i\].idx = i;

cs\[i\].x = L;

int seg = 0; int n = childF\[i\].pts.size();

while (seg < n-1 && childF\[i\].pts\[seg+1\].x <= L) seg++;

cs\[i\].seg = seg;

long double slope = 0;

if (seg+1 < n) {

long long dx = childF\[i\].pts\[seg+1\].x - childF\[i\].pts\[seg\].x;

long long dy = childF\[i\].pts\[seg+1\].y - childF\[i\].pts\[seg\].y;

slope = dx!=0?(long double)dy/dx:0;

} else {

slope = 0;

}

cs\[i\].slope = slope;

if (seg+1 < n) cs\[i\].nextBound = min((long long)childF\[i\].pts\[seg+1\].x, R);

else cs\[i\].nextBound = R;

pq.push({slope, i});

}

long long sum = sumDomainL;

long long cost = 0;

for (int i = 0; i < d; i++) {

cost += evalPWL(childF\[i\], L);

}

vector<Pt> fpts;

fpts.push\_back({sum, cost});

while (!pq.empty() && sum < sumDomainR) {

auto top = pq.top();

long double slope = top.first;

vector<int> group;

while (!pq.empty() && abs(pq.top().first - slope) < 1e-9) {

group.push\_back(pq.top().second);

pq.pop();

}

if (group.empty()) break;

long long inc = LLONG\_MAX;

for (int idx: group) {

long long newInc = cs\[idx\].nextBound - cs\[idx\].x;

if (newInc < inc) inc = newInc;

}

if (inc <= 0) {

for (int idx: group) {

pq.push({cs\[idx\].slope, idx});

}

break;

}

long long feasibleInc = inc;

if (sum + (long long)group.size() \* inc > sumDomainR) {

feasibleInc = (sumDomainR - sum) / (long long) group.size();

}

if (feasibleInc <= 0) {

for (int idx: group) {

pq.push({cs\[idx\].slope, idx});

}

break;

}

sum = sum + group.size() \* feasibleInc;

long double dcost = slope \* feasibleInc \* group.size();

cost = (long long) llround((long double)cost + dcost);

fpts.push\_back({sum, cost});

for (int idx: group) {

cs\[idx\].x += feasibleInc;

if (cs\[idx\].x == cs\[idx\].nextBound) {

const PWL &g = childF\[idx\];

int seg = cs\[idx\].seg;

if (cs\[idx\].x == R) {

cs\[idx\].slope = 1e9; // saturate

cs\[idx\].nextBound = R;

} else {

seg++;

cs\[idx\].seg = seg;

int n = g.pts.size();

if (seg+1 < n) {

long long dx = g.pts\[seg+1\].x - g.pts\[seg\].x;

long long dy = g.pts\[seg+1\].y - g.pts\[seg\].y;

long double slopeC = dx!=0?(long double)dy/dx:0;

cs\[idx\].slope = slopeC;

cs\[idx\].nextBound = min((long long)g.pts\[seg+1\].x, R);

} else {

cs\[idx\].slope = 0;

cs\[idx\].nextBound = R;

}

}

}

}

for (int idx: group) {

if (cs\[idx\].x < R) {

pq.push({cs\[idx\].slope, idx});

}

}

}

vector<Pt> unify;

unify.push\_back(fpts\[0\]);

for (int i = 1; i < fpts.size(); i++) {

if (fpts\[i\].x == unify.back().x) {

unify.back().y = min(unify.back().y, fpts\[i\].y);

} else unify.push\_back(fpts\[i\]);

}

vector<Pt> final;

if (!unify.empty()) final.push\_back(unify\[0\]);

for (int i = 1; i < unify.size()-1; i++) {

long long x1 = final.back().x;

long long y1 = final.back().y;

long long x2 = unify\[i\].x;

long long y2 = unify\[i\].y;

long long x3 = unify\[i+1\].x;

long long y3 = unify\[i+1\].y;

long double slope1 = (x2!=x1)? (long double)(y2-y1)/(x2-x1): 1e9;

long double slope2 = (x3!=x2)? (long double)(y3-y2)/(x3-x2): 1e9;

if (abs(slope1 - slope2) < 1e-9) {

// unify linear, skip unify\[i\]

} else {

final.push\_back(unify\[i\]);

}

}

if (!unify.empty()) final.push\_back(unify.back());

PWL result;

result.L = sumDomainL; result.R = sumDomainR;

result.pts = final;

return result;

}

static PWL parentFormula(const PWL &f, long long wVal, long long L, long long R) {

vector<long long> candVal;

candVal.push\_back(L);

candVal.push\_back(R);

for (auto &p: f.pts) {

if (p.x >= L && p.x <= R) candVal.push\_back(p.x);

}

if (f.L >= L && f.L <= R) candVal.push\_back(f.L);

if (f.R >= L && f.R <= R) candVal.push\_back(f.R);

for (int i = 0; i < (int)f.pts.size()-1; i++) {

long long sx = f.pts\[i\].x;

long long sy = f.pts\[i+1\].x;

long long dx = sy - sx;

long long dy = f.pts\[i+1\].y - f.pts\[i\].y;

long double a = dx!=0? (long double)dy/dx:0; // slope in s domain

if (a <= wVal && a >= -wVal) {

long long inL = max(sx, (long long)L);

long long inR = min(sy, (long long)R);

if (inL <= inR) {

candVal.push\_back(inL);

candVal.push\_back(inR);

}

}

}

sort(candVal.begin(), candVal.end());

candVal.erase(unique(candVal.begin(), candVal.end()), candVal.end());

vector<Pt> pts;

for (auto val: candVal) {

long long bestCost = LLONG\_MAX/2;

if (val >= f.L && val <= f.R) {

int left = 0, right = f.pts.size()-1;

while (left+1 < right) {

int mid = (left + right) / 2;

if (f.pts\[mid\].x <= val) left = mid; else right = mid;

}

if (f.pts\[left\].x <= val && val <= f.pts\[right\].x) {

long long x1 = f.pts\[left\].x;

long long y1 = f.pts\[left\].y;

long long x2 = f.pts\[right\].x;

long long y2 = f.pts\[right\].y;

long double slope = (x2!=x1? (long double)(y2 - y1)/(long double)(x2 - x1): 0);

if (slope <= wVal && slope >= -wVal && val >= x1 && val <= x2) {

long double costf = (long double) y1 + slope \* (long double)(val - x1);

long long c = (long long) llround(costf);

bestCost = min(bestCost, c);

}

}

}

for (auto &p: f.pts) {

long long s = p.x;

long long c = p.y + wVal \* llabs(val - s);

if (c < bestCost) bestCost = c;

}

pts.push\_back({val, bestCost});

}

sort(pts.begin(), pts.end(), \[\](const Pt &a, const Pt &b){return a.x < b.x;});

vector<Pt> unify;

unify.push\_back(pts\[0\]);

for (int i = 1; i < pts.size(); i++) {

if (pts\[i\].x == unify.back().x) {

unify.back().y = min(unify.back().y, pts\[i\].y);

} else unify.push\_back(pts\[i\]);

}

vector<Pt> final;

if (!unify.empty()) final.push\_back(unify\[0\]);

for (int i = 1; i < unify.size()-1; i++) {

long long x1 = final.back().x;

long long y1 = final.back().y;

long long x2 = unify\[i\].x;

long long y2 = unify\[i\].y;

long long x3 = unify\[i+1\].x;

long long y3 = unify\[i+1\].y;

long double slope1 = x2!=x1? (long double)(y2-y1)/(x2-x1): 1e9;

long double slope2 = x3!=x2? (long double)(y3-y2)/(x3-x2): 1e9;

if (abs(slope1 - slope2) < 1e-9) {

// unify linear, skip unify\[i\]

} else {

final.push\_back(unify\[i\]);

}

}

if (!unify.empty()) final.push\_back(unify.back());

PWL result;

result.L = L; result.R = R;

result.pts = final;

return result;

}

static void computeG(int node, long long L, long long R) {

for (int c: childrenGlobal\[node\]) {

computeG(c, L, R);

}

if (childrenGlobal\[node\].empty()) {

// leaf

PWL result;

result.L = L; result.R = R;

if (L <= 0 && 0 <= R) {

long long valL = wGlobal\[node\] \* llabs(L);

long long val0 = wGlobal\[node\] \* 0; // 0

long long valR = wGlobal\[node\] \* llabs(R);

result.pts.clear();

result.pts.push\_back({L, valL});

result.pts.push\_back({0, val0});

result.pts.push\_back({R, valR});

} else {

long long valL = wGlobal\[node\] \* llabs(L);

long long valR = wGlobal\[node\] \* llabs(R);

result.pts.clear();

result.pts.push\_back({L, valL});

result.pts.push\_back({R, valR});

}

GGlobal\[node\] = result;

} else {

vector<PWL> childF;

for (int c: childrenGlobal\[node\]) {

childF.push\_back(GGlobal\[c\]);

}

PWL f = combineChildren(childF, L, R);

PWL p = parentFormula(f, wGlobal\[node\], L, R);

GGlobal\[node\] = p;

}

}

void init(std::vector<int> P, std::vector<int> W) {

NGlobal = W.size();

for (int i = 0; i < NGlobal; i++) {

PGlobal\[i\] = P\[i\];

wGlobal\[i\] = W\[i\];

}

for (int i = 0; i < NGlobal; i++) childrenGlobal\[i\].clear();

for (int i = 1; i < NGlobal; i++) {

childrenGlobal\[P\[i\]\].push\_back(i);

}

}

long long query(int L, int R) {

computeG(0, L, R);

long long ans = LLONG\_MAX;

for (auto &p: GGlobal\[0\].pts) {

if (p.y < ans) ans = p.y;

}

return ans;

}

### C.4 Hieroglyphs

A total of 44 points was scored on Hieroglyphs across two separate submissions.

#### C.4.1 Submission 1

In the first submission, a score of 34 out of 100 points was achieved, distributed across subtasks 1, 2, and 4.

[⬇](data:text/plain;base64,I2luY2x1ZGUgImhpZXJvZ2x5cGhzLmgiCiNpbmNsdWRlIDxiaXRzL3N0ZGMrKy5oPgp1c2luZyBuYW1lc3BhY2Ugc3RkOwpzdGQ6OnZlY3RvcjxpbnQ+IHVjcyhzdGQ6OnZlY3RvcjxpbnQ+IEEsIHN0ZDo6dmVjdG9yPGludD4gQikKewogICAgaW50IE4gPSAoaW50KSBBLnNpemUoKTsKICAgIGludCBNID0gKGludCkgQi5zaXplKCk7CiAgICBpZiAoTiA9PSAwIHx8IE0gPT0gMCkgewogICAgICAgIHN0ZDo6dmVjdG9yPGludD4gcmV0OwogICAgICAgIHJldHVybiByZXQ7CiAgICB9CiAgICBjb25zdCBpbnQgTUFYX1ZBTCA9IDIwMDAwMTsKCiAgICBzdGF0aWMgaW50IGNudEFbMjAwMDAyXSwgY250QlsyMDAwMDJdOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCAyMDAwMDI7IGkrKykgewogICAgICAgIGNudEFbaV0gPSAwOwogICAgICAgIGNudEJbaV0gPSAwOwogICAgfQoKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSBjbnRBW0FbaV1dKys7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IE07IGkrKykgY250QltCW2ldXSsrOwoKICAgIHN0YXRpYyBpbnQgY1ZhbFsyMDAwMDJdOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCAyMDAwMDI7IGkrKykgewogICAgICAgIGlmIChjbnRBW2ldID4gMCAmJiBjbnRCW2ldID4gMCkgewogICAgICAgICAgICBjVmFsW2ldID0gbWluKGNudEFbaV0sIGNudEJbaV0pOwogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIGNWYWxbaV0gPSAwOwogICAgICAgIH0KICAgIH0KCiAgICBsb25nIGxvbmcgc3VtQyA9IDA7CiAgICBmb3IgKGludCB4ID0gMDsgeCA8IDIwMDAwMjsgeCsrKSB7CiAgICAgICAgc3VtQyArPSBjVmFsW3hdOwogICAgfQoKICAgIGlmIChzdW1DID09IDApIHsKICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IHJldDsKICAgICAgICByZXR1cm4gcmV0OwogICAgfQoKICAgIC8vIFByZWNvbXB1dGUgZWFybGllc3QgYW5kIGxhdGVzdCBvY2N1cnJlbmNlcyBpbiBBIGFuZCBCIGZvciBhbGwgeC4KICAgIHN0YXRpYyBpbnQgZWFybGllc3RBWzIwMDAwMl0sIGxhdGVzdEFbMjAwMDAyXSwgZWFybGllc3RCWzIwMDAwMl0sIGxhdGVzdEJbMjAwMDAyXTsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgMjAwMDAyOyBpKyspIHsKICAgICAgICBlYXJsaWVzdEFbaV0gPSAtMTsKICAgICAgICBsYXRlc3RBW2ldID0gLTE7CiAgICAgICAgZWFybGllc3RCW2ldID0gLTE7CiAgICAgICAgbGF0ZXN0QltpXSA9IC0xOwogICAgfQoKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSB7CiAgICAgICAgaW50IGxldHRlciA9IEFbaV07CiAgICAgICAgaWYgKGVhcmxpZXN0QVtsZXR0ZXJdID09IC0xKSBlYXJsaWVzdEFbbGV0dGVyXSA9IGk7CiAgICAgICAgbGF0ZXN0QVtsZXR0ZXJdID0gaTsKICAgIH0KICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTTsgaSsrKSB7CiAgICAgICAgaW50IGxldHRlciA9IEJbaV07CiAgICAgICAgaWYgKGVhcmxpZXN0QltsZXR0ZXJdID09IC0xKSBlYXJsaWVzdEJbbGV0dGVyXSA9IGk7CiAgICAgICAgbGF0ZXN0QltsZXR0ZXJdID0gaTsKICAgIH0KCiAgICAvLyBOb3cgaW50ZXJ2YWxzIGZvciBjVmFsW3hdID0gMS4KICAgIHN0cnVjdCBJbnRlcnZhbCB7CiAgICAgICAgaW50IGxldHRlcjsKICAgICAgICBpbnQgZUEsIGxBLCBlQiwgbEI7CiAgICB9OwoKICAgIHZlY3RvcjxJbnRlcnZhbD4gaW50ZXJ2YWxzOwogICAgZm9yIChpbnQgeCA9IDA7IHggPCAyMDAwMDI7IHgrKykgewogICAgICAgIGlmIChjVmFsW3hdID09IDEpIHsKICAgICAgICAgICAgSW50ZXJ2YWwgSTsKICAgICAgICAgICAgSS5sZXR0ZXIgPSB4OwogICAgICAgICAgICBJLmVBID0gZWFybGllc3RBW3hdOyBJLmxBID0gbGF0ZXN0QVt4XTsKICAgICAgICAgICAgSS5lQiA9IGVhcmxpZXN0Qlt4XTsgSS5sQiA9IGxhdGVzdEJbeF07CiAgICAgICAgICAgIGludGVydmFscy5wdXNoX2JhY2soSSk7CiAgICAgICAgfQogICAgfQoKICAgIC8vIHNvcnQgaW50ZXJ2YWxzIGJ5IGVBLgogICAgc3RkOjpzb3J0KGludGVydmFscy5iZWdpbigpLCBpbnRlcnZhbHMuZW5kKCksIFtdKGNvbnN0IEludGVydmFsICZhLCBjb25zdCBJbnRlcnZhbCAmYil7CiAgICAgICAgcmV0dXJuIGEuZUEgPCBiLmVBOwogICAgfSk7CgogICAgLy8gQmFsYW5jZWQgdHJlZSBrZXllZCBieSBlQiwgc3RvcmUgbEIgYXMgd2VsbC4KICAgIHZlY3RvcjxpbnQ+IHNlZ21lbnRUcmVlKDQqKE0rNSksIC0xKTsKICAgIGF1dG8gdXBkYXRlID0gWyZdKGludCBpZHgsIGludCBzdGFydCwgaW50IGVuZCwgaW50IHBvcywgaW50IHZhbCwgYXV0byAmJmYpIC0+IHZvaWQgewogICAgICAgIGlmIChzdGFydCA9PSBlbmQpIHsKICAgICAgICAgICAgc2VnbWVudFRyZWVbaWR4XSA9IHZhbDsKICAgICAgICAgICAgcmV0dXJuOwogICAgICAgIH0KICAgICAgICBpbnQgbWlkID0gKHN0YXJ0K2VuZCkvMjsKICAgICAgICBpZiAocG9zIDw9IG1pZCkgZihpZHgqMiwgc3RhcnQsIG1pZCwgcG9zLCB2YWwsIGYpOwogICAgICAgIGVsc2UgZihpZHgqMisxLCBtaWQrMSwgZW5kLCBwb3MsIHZhbCwgZik7CiAgICAgICAgc2VnbWVudFRyZWVbaWR4XSA9IG1heChzZWdtZW50VHJlZVtpZHgqMl0sIHNlZ21lbnRUcmVlW2lkeCoyKzFdKTsKICAgIH07CiAgICBhdXRvIHF1ZXJ5ID0gWyZdKGludCBpZHgsIGludCBzdGFydCwgaW50IGVuZCwgaW50IGwsIGludCByLCBhdXRvICYmZikgLT4gaW50IHsKICAgICAgICBpZiAociA8IHN0YXJ0IHx8IGVuZCA8IGwpIHJldHVybiAtMTsKICAgICAgICBpZiAobCA8PSBzdGFydCAmJiBlbmQgPD0gcikgewogICAgICAgICAgICByZXR1cm4gc2VnbWVudFRyZWVbaWR4XTsKICAgICAgICB9CiAgICAgICAgaW50IG1pZCA9IChzdGFydCtlbmQpLzI7CiAgICAgICAgaW50IGxlZnRWYWwgPSBmKGlkeCoyLCBzdGFydCwgbWlkLCBsLCByLCBmKTsKICAgICAgICBpbnQgcmlnaHRWYWwgPSBmKGlkeCoyKzEsIG1pZCsxLCBlbmQsIGwsIHIsIGYpOwogICAgICAgIHJldHVybiBtYXgobGVmdFZhbCwgcmlnaHRWYWwpOwogICAgfTsKCiAgICB2ZWN0b3I8SW50ZXJ2YWw+IGludGVydmFsc1NvcnRlZEJ5TEEgPSBpbnRlcnZhbHM7CiAgICBzb3J0KGludGVydmFsc1NvcnRlZEJ5TEEuYmVnaW4oKSwgaW50ZXJ2YWxzU29ydGVkQnlMQS5lbmQoKSwgW10oY29uc3QgSW50ZXJ2YWwgJmEsIGNvbnN0IEludGVydmFsICZiKXsKICAgICAgICByZXR1cm4gYS5sQSA8IGIubEE7CiAgICB9KTsKCiAgICBpbnQgaiA9IDA7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IGludGVydmFscy5zaXplKCk7IGkrKykgewogICAgICAgIGF1dG8gJnggPSBpbnRlcnZhbHNbaV07CiAgICAgICAgaW50IGVBeCA9IHguZUE7CiAgICAgICAgLy8gcmVtb3ZlIGludGVydmFscyBmcm9tIGRhdGEgc3RydWN0dXJlIHdoZXJlIGxBW3ldIDwgZUF4LgogICAgICAgIHdoaWxlIChqIDwgaW50ZXJ2YWxzU29ydGVkQnlMQS5zaXplKCkgJiYgaW50ZXJ2YWxzU29ydGVkQnlMQVtqXS5sQSA8IGVBeCkgewogICAgICAgICAgICBhdXRvICZ5ID0gaW50ZXJ2YWxzU29ydGVkQnlMQVtqXTsKICAgICAgICAgICAgLy8gcmVtb3ZlIHkgZnJvbSBzZWdtZW50IHRyZWUga2V5ZWQgYnkgZUJbeV0uCiAgICAgICAgICAgIHVwZGF0ZSgxLCAwLCBNLTEsIHkuZUIsIC0xLCB1cGRhdGUpOwogICAgICAgICAgICBqKys7CiAgICAgICAgfQoKICAgICAgICAvLyBxdWVyeSBpbiBCIGRpbWVuc2lvbjogZmluZCBpZiB0aGVyZSdzIGFuIGludGVydmFsIHkgd2l0aCBlQlt5XSA8PSBsQlt4XSBhbmQgbEJbeV0gPj0gZUJbeF0uCiAgICAgICAgaW50IG1pbkIgPSB4LmVCOyAvLyBlQlt4XQogICAgICAgIGludCBtYXhCID0geC5sQjsgLy8gbEJbeF0KICAgICAgICBpZiAobWluQiA+IG1heEIpIHsKICAgICAgICAgICAgLy8gSWYgZUJbeF0gPiBsQlt4XSwgbm8gaW50ZXJzZWN0aW9uIHBvc3NpYmxlLgogICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgIC8vIHF1ZXJ5IGluIHRoZSBzZWdtZW50IHRyZWUgcmFuZ2UgWzAsIG1heEJdIHRvIGZpbmQgdGhlIG1heGltdW0gbEJbeV0uCiAgICAgICAgICAgIGludCBtYXhWYWwgPSBxdWVyeSgxLCAwLCBNLTEsIDAsIG1heEIsIHF1ZXJ5KTsKICAgICAgICAgICAgaWYgKG1heFZhbCA+PSBtaW5CKSB7CiAgICAgICAgICAgICAgICAvLyBmb3VuZCBpbnRlcnNlY3Rpb24gd2l0aCBhIGxldHRlciB5IHdoZXJlIGNWYWxbeV0gPSAxLgogICAgICAgICAgICAgICAgc3RkOjp2ZWN0b3I8aW50PiByZXQ7CiAgICAgICAgICAgICAgICByZXQucHVzaF9iYWNrKC0xKTsKICAgICAgICAgICAgICAgIHJldHVybiByZXQ7CiAgICAgICAgICAgIH0KICAgICAgICB9CgogICAgICAgIC8vIGFkZCB4IHRvIGRhdGEgc3RydWN0dXJlIGtleWVkIGJ5IGVCW3hdLgogICAgICAgIHVwZGF0ZSgxLCAwLCBNLTEsIHguZUIsIHgubEIsIHVwZGF0ZSk7CiAgICB9CgogICAgLy8gSWYgbm8gaW50ZXJzZWN0aW9uIGZvdW5kIGFtb25nIGNWYWw9MSBpbnRlcnZhbHMsIHByb2NlZWQgd2l0aCBjaGFpbiBtZXRob2QuCgogICAgLy8gVGhlIGNoYWluIG1ldGhvZCBjb2RlLgogICAgc3RhdGljIHZlY3RvcjxpbnQ+IHBvc0FhcnJbMjAwMDAyXTsKICAgIHN0YXRpYyB2ZWN0b3I8aW50PiBwb3NCYXJyWzIwMDAwMl07CiAgICBmb3IgKGludCBpID0gMDsgaSA8IDIwMDAwMjsgaSsrKSB7CiAgICAgICAgcG9zQWFycltpXS5jbGVhcigpOwogICAgICAgIHBvc0JhcnJbaV0uY2xlYXIoKTsKICAgIH0KCiAgICBmb3IgKGludCBpID0gMDsgaSA8IE47IGkrKykgewogICAgICAgIGlmIChjVmFsW0FbaV1dID4gMCkgcG9zQWFycltBW2ldXS5wdXNoX2JhY2soaSk7CiAgICB9CiAgICBmb3IgKGludCBpID0gMDsgaSA8IE07IGkrKykgewogICAgICAgIGlmIChjVmFsW0JbaV1dID4gMCkgcG9zQmFycltCW2ldXS5wdXNoX2JhY2soaSk7CiAgICB9CgogICAgc3RydWN0IFFJdGVtIHsKICAgICAgICBpbnQgbGV0dGVyOwogICAgICAgIGludCBwb3M7CiAgICB9OwoKICAgIHN0cnVjdCBRQ29tcCB7CiAgICAgICAgYm9vbCBvcGVyYXRvcigpKGNvbnN0IFFJdGVtICZhLCBjb25zdCBRSXRlbSAmYikgY29uc3QgewogICAgICAgICAgICBpZiAoYS5wb3MgPT0gYi5wb3MpIHJldHVybiBhLmxldHRlciA+IGIubGV0dGVyOwogICAgICAgICAgICByZXR1cm4gYS5wb3MgPiBiLnBvczsKICAgICAgICB9CiAgICB9OwoKICAgIHByaW9yaXR5X3F1ZXVlPFFJdGVtLCB2ZWN0b3I8UUl0ZW0+LCBRQ29tcD4gcHE7CgogICAgc3RhdGljIGludCBwb3NBYXJySWR4WzIwMDAwMl07CiAgICBzdGF0aWMgaW50IG9jY0luQWFyclsyMDAwMDJdOwoKICAgIGZvciAoaW50IHggPSAwOyB4IDwgMjAwMDAyOyB4KyspIHsKICAgICAgICBwb3NBYXJySWR4W3hdID0gMDsKICAgICAgICBvY2NJbkFhcnJbeF0gPSAwOwogICAgICAgIGlmIChjVmFsW3hdID4gMCAmJiBwb3NBYXJyW3hdLnNpemUoKSA+IDApIHsKICAgICAgICAgICAgUUl0ZW0gaXRlbTsKICAgICAgICAgICAgaXRlbS5sZXR0ZXIgPSB4OwogICAgICAgICAgICBpdGVtLnBvcyA9IHBvc0FhcnJbeF1bMF07CiAgICAgICAgICAgIHBxLnB1c2goaXRlbSk7CiAgICAgICAgfQogICAgfQoKICAgIHN0YXRpYyBpbnQgVEFscGhhWzIwMDAwMl07CiAgICBmb3IgKGludCB4ID0gMDsgeCA8IDIwMDAwMjsgeCsrKSB7CiAgICAgICAgaWYgKGNWYWxbeF0gPiAwKSBUQWxwaGFbeF0gPSBjbnRCW3hdIC0gY1ZhbFt4XTsgZWxzZSBUQWxwaGFbeF0gPSAwOwogICAgfQoKICAgIHN0YXRpYyBpbnQgZnJlcUFscGhhWzIwMDAwMl07IC8vIGZyZXEgaW4gQiB1cCB0byBwb3NCMC4KICAgIGZvciAoaW50IHggPSAwOyB4IDwgMjAwMDAyOyB4KyspIGZyZXFBbHBoYVt4XSA9IDA7CgogICAgaW50IHBvc0IwQWxwaGEgPSAtMTsKICAgIHsKICAgICAgICBpbnQgaSA9IDA7CiAgICAgICAgZm9yICg7IGkgPCBNOyBpKyspIHsKICAgICAgICAgICAgaW50IGxldHRlciA9IEJbaV07CiAgICAgICAgICAgIGlmIChjVmFsW2xldHRlcl0gPiAwKSB7CiAgICAgICAgICAgICAgICBmcmVxQWxwaGFbbGV0dGVyXSsrOwogICAgICAgICAgICAgICAgaWYgKGZyZXFBbHBoYVtsZXR0ZXJdID4gVEFscGhhW2xldHRlcl0pIHsKICAgICAgICAgICAgICAgICAgICBmcmVxQWxwaGFbbGV0dGVyXS0tOwogICAgICAgICAgICAgICAgICAgIGJyZWFrOyAvLyBjYXVzZSBwb3NCMEFscGhhID0gaS0xCiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICAgICAgcG9zQjBBbHBoYSA9IGktMTsKICAgICAgICBpZiAoaSA9PSBNKSBwb3NCMEFscGhhID0gTS0xOwogICAgfQoKICAgIHZlY3RvcjxpbnQ+IFU7CiAgICBVLnJlc2VydmUoc3VtQyk7CgogICAgaW50IHBvc0IgPSAtMTsKICAgIGludCB1c2VkQ291bnQgPSAwOwoKICAgIGF1dG8gdXBkYXRlUG9zQjBBbHBoYSA9IFsmXShpbnQgbGV0dGVyLCBhdXRvICZmcmVxQWxwaGEsIGF1dG8gJlRBbHBoYSwgaW50ICZwb3NCMEFscGhhLCBpbnQgTSkgewogICAgICAgIGludCBpID0gcG9zQjBBbHBoYSArIDE7CiAgICAgICAgd2hpbGUgKGkgPCBNKSB7CiAgICAgICAgICAgIGludCBsID0gQltpXTsKICAgICAgICAgICAgaWYgKGNWYWxbbF0gPiAwKSB7CiAgICAgICAgICAgICAgICBmcmVxQWxwaGFbbF0rKzsKICAgICAgICAgICAgICAgIGlmIChmcmVxQWxwaGFbbF0gPiBUQWxwaGFbbF0pIHsKICAgICAgICAgICAgICAgICAgICBmcmVxQWxwaGFbbF0tLTsKICAgICAgICAgICAgICAgICAgICBicmVhazsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgfQogICAgICAgICAgICBpKys7CiAgICAgICAgfQogICAgICAgIHBvc0IwQWxwaGEgPSBpLTE7CiAgICB9OwoKICAgIHdoaWxlICghcHEuZW1wdHkoKSAmJiB1c2VkQ291bnQgPCBzdW1DKSB7CiAgICAgICAgUUl0ZW0gaXRlbSA9IHBxLnRvcCgpOwogICAgICAgIHBxLnBvcCgpOwoKICAgICAgICBpbnQgbGV0dGVyID0gaXRlbS5sZXR0ZXI7CiAgICAgICAgaW50IHBvc0FWYWwgPSBwb3NBYXJyW2xldHRlcl1bcG9zQWFycklkeFtsZXR0ZXJdXTsKCiAgICAgICAgYXV0byAmYmFyciA9IHBvc0JhcnJbbGV0dGVyXTsKICAgICAgICBpbnQgaWR4ID0gKGludCkgKHN0ZDo6bG93ZXJfYm91bmQoYmFyci5iZWdpbigpLCBiYXJyLmVuZCgpLCBwb3NCICsgMSkgLSBiYXJyLmJlZ2luKCkpOwogICAgICAgIGlmIChpZHggPT0gKGludCkgYmFyci5zaXplKCkpIHsKICAgICAgICAgICAgcG9zQWFycklkeFtsZXR0ZXJdKys7CiAgICAgICAgICAgIGlmIChwb3NBYXJySWR4W2xldHRlcl0gPCBwb3NBYXJyW2xldHRlcl0uc2l6ZSgpKSB7CiAgICAgICAgICAgICAgICBRSXRlbSBuZXdJdGVtOwogICAgICAgICAgICAgICAgbmV3SXRlbS5sZXR0ZXIgPSBsZXR0ZXI7CiAgICAgICAgICAgICAgICBuZXdJdGVtLnBvcyA9IHBvc0FhcnJbbGV0dGVyXVtwb3NBYXJySWR4W2xldHRlcl1dOwogICAgICAgICAgICAgICAgcHEucHVzaChuZXdJdGVtKTsKICAgICAgICAgICAgfQogICAgICAgICAgICBjb250aW51ZTsKICAgICAgICB9IGVsc2UgewogICAgICAgICAgICBpbnQgcG9zQlZhbCA9IGJhcnJbaWR4XTsKCiAgICAgICAgICAgIFRBbHBoYVtsZXR0ZXJdKys7CiAgICAgICAgICAgIGludCBvbGRQb3NCMEFscGhhID0gcG9zQjBBbHBoYTsKICAgICAgICAgICAgdmVjdG9yPHBhaXI8aW50LGludD4+IGZyZXFDaGFuZ2VzOwoKICAgICAgICAgICAgaW50IHN0YXJ0SW5kZXggPSBwb3NCMEFscGhhICsgMTsKICAgICAgICAgICAgaW50IGkgPSBzdGFydEluZGV4OwogICAgICAgICAgICB3aGlsZSAoaSA8IE0pIHsKICAgICAgICAgICAgICAgIGludCBsID0gQltpXTsKICAgICAgICAgICAgICAgIGlmIChjVmFsW2xdID4gMCkgewogICAgICAgICAgICAgICAgICAgIGZyZXFBbHBoYVtsXSsrOwogICAgICAgICAgICAgICAgICAgIGZyZXFDaGFuZ2VzLnB1c2hfYmFjayh7bCwgaX0pOwogICAgICAgICAgICAgICAgICAgIGlmIChmcmVxQWxwaGFbbF0gPiBUQWxwaGFbbF0pIHsKICAgICAgICAgICAgICAgICAgICAgICAgZnJlcUFscGhhW2xdLS07CiAgICAgICAgICAgICAgICAgICAgICAgIGZyZXFDaGFuZ2VzLnBvcF9iYWNrKCk7CiAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrOwogICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIGkrKzsKICAgICAgICAgICAgfQogICAgICAgICAgICBpbnQgbmV3UG9zQjBBbHBoYSA9IGktMTsKCiAgICAgICAgICAgIGlmIChwb3NCVmFsIDw9IG5ld1Bvc0IwQWxwaGEpIHsKICAgICAgICAgICAgICAgIFUucHVzaF9iYWNrKGxldHRlcik7CiAgICAgICAgICAgICAgICB1c2VkQ291bnQrKzsKICAgICAgICAgICAgICAgIHBvc0IgPSBwb3NCVmFsOwoKICAgICAgICAgICAgICAgIG9jY0luQWFycltsZXR0ZXJdKys7CiAgICAgICAgICAgICAgICBwb3NBYXJySWR4W2xldHRlcl0rKzsKICAgICAgICAgICAgICAgIGlmIChvY2NJbkFhcnJbbGV0dGVyXSA8IGNWYWxbbGV0dGVyXSAmJiBwb3NBYXJySWR4W2xldHRlcl0gPCBwb3NBYXJyW2xldHRlcl0uc2l6ZSgpKSB7CiAgICAgICAgICAgICAgICAgICAgUUl0ZW0gbmV3SXRlbTsKICAgICAgICAgICAgICAgICAgICBuZXdJdGVtLmxldHRlciA9IGxldHRlcjsKICAgICAgICAgICAgICAgICAgICBuZXdJdGVtLnBvcyA9IHBvc0FhcnJbbGV0dGVyXVtwb3NBYXJySWR4W2xldHRlcl1dOwogICAgICAgICAgICAgICAgICAgIHBxLnB1c2gobmV3SXRlbSk7CiAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAgICAgcG9zQjBBbHBoYSA9IG5ld1Bvc0IwQWxwaGE7CgogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgVEFscGhhW2xldHRlcl0tLTsKICAgICAgICAgICAgICAgIGZvciAoYXV0byAmZmMgOiBmcmVxQ2hhbmdlcykgewogICAgICAgICAgICAgICAgICAgIGZyZXFBbHBoYVtmYy5maXJzdF0tLTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIHBvc0IwQWxwaGEgPSBvbGRQb3NCMEFscGhhOwoKICAgICAgICAgICAgICAgIHBvc0FhcnJJZHhbbGV0dGVyXSsrOwogICAgICAgICAgICAgICAgaWYgKHBvc0FhcnJJZHhbbGV0dGVyXSA8IHBvc0FhcnJbbGV0dGVyXS5zaXplKCkpIHsKICAgICAgICAgICAgICAgICAgICBRSXRlbSBuZXdJdGVtOwogICAgICAgICAgICAgICAgICAgIG5ld0l0ZW0ubGV0dGVyID0gbGV0dGVyOwogICAgICAgICAgICAgICAgICAgIG5ld0l0ZW0ucG9zID0gcG9zQWFycltsZXR0ZXJdW3Bvc0FhcnJJZHhbbGV0dGVyXV07CiAgICAgICAgICAgICAgICAgICAgcHEucHVzaChuZXdJdGVtKTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIGNvbnRpbnVlOwogICAgICAgICAgICB9CgogICAgICAgIH0KICAgIH0KCiAgICBpZiAodXNlZENvdW50IDwgc3VtQykgewogICAgICAgIHN0ZDo6dmVjdG9yPGludD4gcmV0OwogICAgICAgIHJldC5wdXNoX2JhY2soLTEpOwogICAgICAgIHJldHVybiByZXQ7CiAgICB9CgogICAgcmV0dXJuIFU7Cn0=)

#include "hieroglyphs.h"

#include <bits/stdc++.h>

using namespace std;

std::vector<int> ucs(std::vector<int> A, std::vector<int> B)

{

int N = (int) A.size();

int M = (int) B.size();

if (N == 0 || M == 0) {

std::vector<int> ret;

return ret;

}

const int MAX\_VAL = 200001;

static int cntA\[200002\], cntB\[200002\];

for (int i = 0; i < 200002; i++) {

cntA\[i\] = 0;

cntB\[i\] = 0;

}

for (int i = 0; i < N; i++) cntA\[A\[i\]\]++;

for (int i = 0; i < M; i++) cntB\[B\[i\]\]++;

static int cVal\[200002\];

for (int i = 0; i < 200002; i++) {

if (cntA\[i\] > 0 && cntB\[i\] > 0) {

cVal\[i\] = min(cntA\[i\], cntB\[i\]);

} else {

cVal\[i\] = 0;

}

}

long long sumC = 0;

for (int x = 0; x < 200002; x++) {

sumC += cVal\[x\];

}

if (sumC == 0) {

std::vector<int> ret;

return ret;

}

// Precompute earliest and latest occurrences in A and B for all x.

static int earliestA\[200002\], latestA\[200002\], earliestB\[200002\], latestB\[200002\];

for (int i = 0; i < 200002; i++) {

earliestA\[i\] = -1;

latestA\[i\] = -1;

earliestB\[i\] = -1;

latestB\[i\] = -1;

}

for (int i = 0; i < N; i++) {

int letter = A\[i\];

if (earliestA\[letter\] == -1) earliestA\[letter\] = i;

latestA\[letter\] = i;

}

for (int i = 0; i < M; i++) {

int letter = B\[i\];

if (earliestB\[letter\] == -1) earliestB\[letter\] = i;

latestB\[letter\] = i;

}

// Now intervals for cVal\[x\] = 1.

struct Interval {

int letter;

int eA, lA, eB, lB;

};

vector<Interval> intervals;

for (int x = 0; x < 200002; x++) {

if (cVal\[x\] == 1) {

Interval I;

I.letter = x;

I.eA = earliestA\[x\]; I.lA = latestA\[x\];

I.eB = earliestB\[x\]; I.lB = latestB\[x\];

intervals.push\_back(I);

}

}

// sort intervals by eA.

std::sort(intervals.begin(), intervals.end(), \[\](const Interval &a, const Interval &b){

return a.eA < b.eA;

});

// Balanced tree keyed by eB, store lB as well.

vector<int> segmentTree(4\*(M+5), -1);

auto update = \[&\](int idx, int start, int end, int pos, int val, auto &&f) -> void {

if (start == end) {

segmentTree\[idx\] = val;

return;

}

int mid = (start+end)/2;

if (pos <= mid) f(idx\*2, start, mid, pos, val, f);

else f(idx\*2+1, mid+1, end, pos, val, f);

segmentTree\[idx\] = max(segmentTree\[idx\*2\], segmentTree\[idx\*2+1\]);

};

auto query = \[&\](int idx, int start, int end, int l, int r, auto &&f) -> int {

if (r < start || end < l) return -1;

if (l <= start && end <= r) {

return segmentTree\[idx\];

}

int mid = (start+end)/2;

int leftVal = f(idx\*2, start, mid, l, r, f);

int rightVal = f(idx\*2+1, mid+1, end, l, r, f);

return max(leftVal, rightVal);

};

vector<Interval> intervalsSortedByLA = intervals;

sort(intervalsSortedByLA.begin(), intervalsSortedByLA.end(), \[\](const Interval &a, const Interval &b){

return a.lA < b.lA;

});

int j = 0;

for (int i = 0; i < intervals.size(); i++) {

auto &x = intervals\[i\];

int eAx = x.eA;

// remove intervals from data structure where lA\[y\] < eAx.

while (j < intervalsSortedByLA.size() && intervalsSortedByLA\[j\].lA < eAx) {

auto &y = intervalsSortedByLA\[j\];

// remove y from segment tree keyed by eB\[y\].

update(1, 0, M-1, y.eB, -1, update);

j++;

}

// query in B dimension: find if there’s an interval y with eB\[y\] <= lB\[x\] and lB\[y\] >= eB\[x\].

int minB = x.eB; // eB\[x\]

int maxB = x.lB; // lB\[x\]

if (minB > maxB) {

// If eB\[x\] > lB\[x\], no intersection possible.

} else {

// query in the segment tree range \[0, maxB\] to find the maximum lB\[y\].

int maxVal = query(1, 0, M-1, 0, maxB, query);

if (maxVal >= minB) {

// found intersection with a letter y where cVal\[y\] = 1.

std::vector<int> ret;

ret.push\_back(-1);

return ret;

}

}

// add x to data structure keyed by eB\[x\].

update(1, 0, M-1, x.eB, x.lB, update);

}

// If no intersection found among cVal=1 intervals, proceed with chain method.

// The chain method code.

static vector<int> posAarr\[200002\];

static vector<int> posBarr\[200002\];

for (int i = 0; i < 200002; i++) {

posAarr\[i\].clear();

posBarr\[i\].clear();

}

for (int i = 0; i < N; i++) {

if (cVal\[A\[i\]\] > 0) posAarr\[A\[i\]\].push\_back(i);

}

for (int i = 0; i < M; i++) {

if (cVal\[B\[i\]\] > 0) posBarr\[B\[i\]\].push\_back(i);

}

struct QItem {

int letter;

int pos;

};

struct QComp {

bool operator()(const QItem &a, const QItem &b) const {

if (a.pos == b.pos) return a.letter > b.letter;

return a.pos > b.pos;

}

};

priority\_queue<QItem, vector<QItem>, QComp> pq;

static int posAarrIdx\[200002\];

static int occInAarr\[200002\];

for (int x = 0; x < 200002; x++) {

posAarrIdx\[x\] = 0;

occInAarr\[x\] = 0;

if (cVal\[x\] > 0 && posAarr\[x\].size() > 0) {

QItem item;

item.letter = x;

item.pos = posAarr\[x\]\[0\];

pq.push(item);

}

}

static int TAlpha\[200002\];

for (int x = 0; x < 200002; x++) {

if (cVal\[x\] > 0) TAlpha\[x\] = cntB\[x\] - cVal\[x\]; else TAlpha\[x\] = 0;

}

static int freqAlpha\[200002\]; // freq in B up to posB0.

for (int x = 0; x < 200002; x++) freqAlpha\[x\] = 0;

int posB0Alpha = -1;

{

int i = 0;

for (; i < M; i++) {

int letter = B\[i\];

if (cVal\[letter\] > 0) {

freqAlpha\[letter\]++;

if (freqAlpha\[letter\] > TAlpha\[letter\]) {

freqAlpha\[letter\]--;

break; // cause posB0Alpha = i-1

}

}

}

posB0Alpha = i-1;

if (i == M) posB0Alpha = M-1;

}

vector<int> U;

U.reserve(sumC);

int posB = -1;

int usedCount = 0;

auto updatePosB0Alpha = \[&\](int letter, auto &freqAlpha, auto &TAlpha, int &posB0Alpha, int M) {

int i = posB0Alpha + 1;

while (i < M) {

int l = B\[i\];

if (cVal\[l\] > 0) {

freqAlpha\[l\]++;

if (freqAlpha\[l\] > TAlpha\[l\]) {

freqAlpha\[l\]--;

break;

}

}

i++;

}

posB0Alpha = i-1;

};

while (!pq.empty() && usedCount < sumC) {

QItem item = pq.top();

pq.pop();

int letter = item.letter;

int posAVal = posAarr\[letter\]\[posAarrIdx\[letter\]\];

auto &barr = posBarr\[letter\];

int idx = (int) (std::lower\_bound(barr.begin(), barr.end(), posB + 1) - barr.begin());

if (idx == (int) barr.size()) {

posAarrIdx\[letter\]++;

if (posAarrIdx\[letter\] < posAarr\[letter\].size()) {

QItem newItem;

newItem.letter = letter;

newItem.pos = posAarr\[letter\]\[posAarrIdx\[letter\]\];

pq.push(newItem);

}

continue;

} else {

int posBVal = barr\[idx\];

TAlpha\[letter\]++;

int oldPosB0Alpha = posB0Alpha;

vector<pair<int,int>> freqChanges;

int startIndex = posB0Alpha + 1;

int i = startIndex;

while (i < M) {

int l = B\[i\];

if (cVal\[l\] > 0) {

freqAlpha\[l\]++;

freqChanges.push\_back({l, i});

if (freqAlpha\[l\] > TAlpha\[l\]) {

freqAlpha\[l\]--;

freqChanges.pop\_back();

break;

}

}

i++;

}

int newPosB0Alpha = i-1;

if (posBVal <= newPosB0Alpha) {

U.push\_back(letter);

usedCount++;

posB = posBVal;

occInAarr\[letter\]++;

posAarrIdx\[letter\]++;

if (occInAarr\[letter\] < cVal\[letter\] && posAarrIdx\[letter\] < posAarr\[letter\].size()) {

QItem newItem;

newItem.letter = letter;

newItem.pos = posAarr\[letter\]\[posAarrIdx\[letter\]\];

pq.push(newItem);

}

posB0Alpha = newPosB0Alpha;

} else {

TAlpha\[letter\]--;

for (auto &fc: freqChanges) {

freqAlpha\[fc.first\]--;

}

posB0Alpha = oldPosB0Alpha;

posAarrIdx\[letter\]++;

if (posAarrIdx\[letter\] < posAarr\[letter\].size()) {

QItem newItem;

newItem.letter = letter;

newItem.pos = posAarr\[letter\]\[posAarrIdx\[letter\]\];

pq.push(newItem);

}

continue;

}

}

}

if (usedCount < sumC) {

std::vector<int> ret;

ret.push\_back(-1);

return ret;

}

return U;

}

#### C.4.2 Submission 2

In the second submission, the model scored 10 points on subtask 3.

[⬇](data:text/plain;base64,I2luY2x1ZGUgImhpZXJvZ2x5cGhzLmgiCiNpbmNsdWRlIDxiaXRzL3N0ZGMrKy5oPgp1c2luZyBuYW1lc3BhY2Ugc3RkOwoKc3RkOjp2ZWN0b3I8aW50PiB1Y3Moc3RkOjp2ZWN0b3I8aW50PiBBLCBzdGQ6OnZlY3RvcjxpbnQ+IEIpCnsKICAgIGludCBOID0gKGludClBLnNpemUoKTsKICAgIGludCBNID0gKGludClCLnNpemUoKTsKCiAgICBpbnQgYTAgPSAwLCBhMSA9IDA7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IE47IGkrKykgewogICAgICAgIGlmIChBW2ldID09IDApIGEwKys7CiAgICAgICAgZWxzZSBpZiAoQVtpXSA9PSAxKSBhMSsrOwogICAgfQogICAgaW50IGIwID0gMCwgYjEgPSAwOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCBNOyBpKyspIHsKICAgICAgICBpZiAoQltpXSA9PSAwKSBiMCsrOwogICAgICAgIGVsc2UgaWYgKEJbaV0gPT0gMSkgYjErKzsKICAgIH0KCiAgICBpbnQgWiA9IG1pbihhMCwgYjApOwogICAgaW50IE8gPSBtaW4oYTEsIGIxKTsKCiAgICAvLyBJZiBaPTAgYW5kIE89MCwgdW5pdmVyc2FsIHN1YnNlcXVlbmNlIGlzIHRoZSBlbXB0eSBzZXF1ZW5jZS4KICAgIGlmIChaID09IDAgJiYgTyA9PSAwKSB7CiAgICAgICAgcmV0dXJuIHt9OwogICAgfQoKICAgIHZlY3RvcjxpbnQ+IHplcm9Qb3NBOyB6ZXJvUG9zQS5yZXNlcnZlKGEwKTsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSBpZiAoQVtpXSA9PSAwKSB6ZXJvUG9zQS5wdXNoX2JhY2soaSk7CiAgICB2ZWN0b3I8aW50PiB6ZXJvUG9zQjsgemVyb1Bvc0IucmVzZXJ2ZShiMCk7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IE07IGkrKykgaWYgKEJbaV0gPT0gMCkgemVyb1Bvc0IucHVzaF9iYWNrKGkpOwoKICAgIHZlY3RvcjxpbnQ+IHByZWZpeE9uZXNBKE4rMSwgMCk7CiAgICBmb3IgKGludCBpID0gMTsgaSA8PSBOOyBpKyspIHByZWZpeE9uZXNBW2ldID0gcHJlZml4T25lc0FbaS0xXSArIChBW2ktMV0gPT0gMSk7CiAgICB2ZWN0b3I8aW50PiBwcmVmaXhPbmVzQihNKzEsIDApOwogICAgZm9yIChpbnQgaSA9IDE7IGkgPD0gTTsgaSsrKSBwcmVmaXhPbmVzQltpXSA9IHByZWZpeE9uZXNCW2ktMV0gKyAoQltpLTFdID09IDEpOwoKICAgIGF1dG8gZ2V0T25lc0FmdGVyWmVybyA9IFsmXSh2ZWN0b3I8aW50PiAmIHplcm9Qb3MsIHZlY3RvcjxpbnQ+ICYgcHJlZml4T25lcywgaW50IHRvdGFsT25lcywgaW50IHopIHsKICAgICAgICBpZiAoeiA9PSAwKSByZXR1cm4gdG90YWxPbmVzOwogICAgICAgIGVsc2UgewogICAgICAgICAgICBpZiAoei0xID49IChpbnQpemVyb1Bvcy5zaXplKCkpIHJldHVybiAwOyAvLyBzYWZldHkKICAgICAgICAgICAgaW50IHBvcyA9IHplcm9Qb3Nbei0xXTsKICAgICAgICAgICAgaW50IG9uZXNVc2VkID0gcHJlZml4T25lc1twb3MrMV07CiAgICAgICAgICAgIGludCBvbmVzQWZ0ZXIgPSB0b3RhbE9uZXMgLSBvbmVzVXNlZDsKICAgICAgICAgICAgcmV0dXJuIG9uZXNBZnRlcjsKICAgICAgICB9CiAgICB9OwoKICAgIHZlY3RvcjxpbnQ+IGZBdmFsKForMSksIGZCdmFsKForMSk7CiAgICBmb3IgKGludCB6ID0gMDsgeiA8PSBaOyB6KyspIHsKICAgICAgICBmQXZhbFt6XSA9IGdldE9uZXNBZnRlclplcm8oemVyb1Bvc0EsIHByZWZpeE9uZXNBLCBhMSwgeik7CiAgICAgICAgZkJ2YWxbel0gPSBnZXRPbmVzQWZ0ZXJaZXJvKHplcm9Qb3NCLCBwcmVmaXhPbmVzQiwgYjEsIHopOwogICAgfQoKICAgIHZlY3RvcjxpbnQ+IGNaKForMSk7CiAgICBmb3IgKGludCB6ID0gMDsgeiA8PSBaOyB6KyspIGNaW3pdID0gbWluKGZBdmFsW3pdLCBmQnZhbFt6XSk7CgogICAgdmVjdG9yPGludD4gb25lUG9zQTsgb25lUG9zQS5yZXNlcnZlKGExKTsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSBpZiAoQVtpXSA9PSAxKSBvbmVQb3NBLnB1c2hfYmFjayhpKTsKICAgIHZlY3RvcjxpbnQ+IG9uZVBvc0I7IG9uZVBvc0IucmVzZXJ2ZShiMSk7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IE07IGkrKykgaWYgKEJbaV0gPT0gMSkgb25lUG9zQi5wdXNoX2JhY2soaSk7CgogICAgdmVjdG9yPGludD4gcHJlZml4WmVyb3NBKE4rMSwgMCk7CiAgICBmb3IgKGludCBpID0gMTsgaSA8PSBOOyBpKyspIHByZWZpeFplcm9zQVtpXSA9IHByZWZpeFplcm9zQVtpLTFdICsgKEFbaS0xXSA9PSAwKTsKICAgIHZlY3RvcjxpbnQ+IHByZWZpeFplcm9zQihNKzEsIDApOwogICAgZm9yIChpbnQgaSA9IDE7IGkgPD0gTTsgaSsrKSBwcmVmaXhaZXJvc0JbaV0gPSBwcmVmaXhaZXJvc0JbaS0xXSArIChCW2ktMV0gPT0gMCk7CgogICAgYXV0byBnZXRaZXJvc0FmdGVyT25lID0gWyZdKHZlY3RvcjxpbnQ+ICYgb25lUG9zLCB2ZWN0b3I8aW50PiAmIHByZWZpeFplcm9zLCBpbnQgdG90YWxaZXJvcywgaW50IHcpIHsKICAgICAgICBpZiAodyA9PSAwKSByZXR1cm4gdG90YWxaZXJvczsKICAgICAgICBlbHNlIHsKICAgICAgICAgICAgaWYgKHctMSA+PSAoaW50KW9uZVBvcy5zaXplKCkpIHJldHVybiAwOyAvLyBzYWZldHkKICAgICAgICAgICAgaW50IHBvcyA9IG9uZVBvc1t3LTFdOwogICAgICAgICAgICBpbnQgemVyb3NVc2VkID0gcHJlZml4WmVyb3NbcG9zKzFdOwogICAgICAgICAgICBpbnQgemVyb3NBZnRlciA9IHRvdGFsWmVyb3MgLSB6ZXJvc1VzZWQ7CiAgICAgICAgICAgIHJldHVybiB6ZXJvc0FmdGVyOwogICAgICAgIH0KICAgIH07CgogICAgdmVjdG9yPGludD4gZ0F2YWwoTysxKSwgZ0J2YWwoTysxKTsKICAgIGZvciAoaW50IHcgPSAwOyB3IDw9IE87IHcrKykgewogICAgICAgIGdBdmFsW3ddID0gZ2V0WmVyb3NBZnRlck9uZShvbmVQb3NBLCBwcmVmaXhaZXJvc0EsIGEwLCB3KTsKICAgICAgICBnQnZhbFt3XSA9IGdldFplcm9zQWZ0ZXJPbmUob25lUG9zQiwgcHJlZml4WmVyb3NCLCBiMCwgdyk7CiAgICB9CgogICAgdmVjdG9yPGludD4gY08oTysxKTsKICAgIGZvciAoaW50IHcgPSAwOyB3IDw9IE87IHcrKykgY09bd10gPSBtaW4oZ0F2YWxbd10sIGdCdmFsW3ddKTsKCiAgICB2ZWN0b3I8aW50PiBVOwogICAgVS5yZXNlcnZlKForTyk7CgogICAgaW50IHpVc2VkID0gMDsKICAgIGludCBvVXNlZCA9IDA7CgogICAgYXV0byBjYW5QbGFjZVplcm8gPSBbJl0oaW50IHpVc2VkLCBpbnQgb1VzZWQpIHsKICAgICAgICBpZiAoelVzZWQgPj0gWikgcmV0dXJuIGZhbHNlOwogICAgICAgIGludCBjID0gY1pbelVzZWQrMV07IC8vIGNvbnN0cmFpbnQgZm9yIHRoZSBuZXh0IHplcm8gaW4gVS4KICAgICAgICBpbnQgb25lc0xlZnQgPSBPIC0gb1VzZWQ7CiAgICAgICAgcmV0dXJuIG9uZXNMZWZ0ID49IGM7CiAgICB9OwoKICAgIGF1dG8gY2FuUGxhY2VPbmUgPSBbJl0oaW50IHpVc2VkLCBpbnQgb1VzZWQpIHsKICAgICAgICBpZiAob1VzZWQgPj0gTykgcmV0dXJuIGZhbHNlOwogICAgICAgIGludCBjID0gY09bb1VzZWQrMV07IC8vIGNvbnN0cmFpbnQgZm9yIHRoZSBuZXh0IG9uZSBpbiBVLgogICAgICAgIGludCB6ZXJvc0xlZnQgPSBaIC0gelVzZWQ7CiAgICAgICAgcmV0dXJuIHplcm9zTGVmdCA+PSBjOwogICAgfTsKCiAgICBmb3IgKGludCBpID0gMDsgaSA8IForTzsgaSsrKSB7CiAgICAgICAgYm9vbCBwbGFjZVogPSBjYW5QbGFjZVplcm8oelVzZWQsIG9Vc2VkKTsKICAgICAgICBib29sIHBsYWNlTyA9IGNhblBsYWNlT25lKHpVc2VkLCBvVXNlZCk7CiAgICAgICAgaWYgKCFwbGFjZVogJiYgIXBsYWNlTykgewogICAgICAgICAgICByZXR1cm4gey0xfTsKICAgICAgICB9IGVsc2UgaWYgKHBsYWNlWiAmJiAhcGxhY2VPKSB7CiAgICAgICAgICAgIFUucHVzaF9iYWNrKDApOwogICAgICAgICAgICB6VXNlZCsrOwogICAgICAgIH0gZWxzZSBpZiAoIXBsYWNlWiAmJiBwbGFjZU8pIHsKICAgICAgICAgICAgVS5wdXNoX2JhY2soMSk7CiAgICAgICAgICAgIG9Vc2VkKys7CiAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgaW50IGNaYyA9IGNaW3pVc2VkKzFdOwogICAgICAgICAgICBpbnQgY09jID0gY09bb1VzZWQrMV07CiAgICAgICAgICAgIGJvb2wgY2FuUGlja1plcm8gPSBmYWxzZTsKICAgICAgICAgICAgYm9vbCBjYW5QaWNrT25lID0gZmFsc2U7CgogICAgICAgICAgICB7CiAgICAgICAgICAgICAgICBpbnQgbmV3WlVzZWQgPSB6VXNlZCArIDE7CiAgICAgICAgICAgICAgICBpbnQgbmV3T1VzZWQgPSBvVXNlZDsKICAgICAgICAgICAgICAgIGludCBvbmVzTGVmdCA9IE8gLSBuZXdPVXNlZDsKICAgICAgICAgICAgICAgIGludCB6ZXJvc0xlZnQgPSBaIC0gbmV3WlVzZWQ7CiAgICAgICAgICAgICAgICBib29sIGZlYXNpYmxlID0gdHJ1ZTsKICAgICAgICAgICAgICAgIGlmIChaID4gbmV3WlVzZWQpIHsgLy8gY2hlY2sgY1ogaW4gcmFuZ2UgW25ld1pVc2VkKzEuLlpdCiAgICAgICAgICAgICAgICAgICAgaW50IG1heFpDb25zdHJhaW50ID0gY1pbbmV3WlVzZWQrMV07CiAgICAgICAgICAgICAgICAgICAgaWYgKG9uZXNMZWZ0IDwgbWF4WkNvbnN0cmFpbnQpIGZlYXNpYmxlID0gZmFsc2U7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICBpZiAoTyA+IG5ld09Vc2VkKSB7IC8vIGNoZWNrIGNPIGluIHJhbmdlIFtuZXdPVXNlZCsxLi5PXQogICAgICAgICAgICAgICAgICAgIGludCBtYXhPQ29uc3RyYWludCA9IGNPW25ld09Vc2VkKzFdOwogICAgICAgICAgICAgICAgICAgIGlmICh6ZXJvc0xlZnQgPCBtYXhPQ29uc3RyYWludCkgZmVhc2libGUgPSBmYWxzZTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIGNhblBpY2taZXJvID0gZmVhc2libGU7CiAgICAgICAgICAgIH0KCiAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgIGludCBuZXdaVXNlZCA9IHpVc2VkOwogICAgICAgICAgICAgICAgaW50IG5ld09Vc2VkID0gb1VzZWQgKyAxOwogICAgICAgICAgICAgICAgaW50IG9uZXNMZWZ0ID0gTyAtIG5ld09Vc2VkOwogICAgICAgICAgICAgICAgaW50IHplcm9zTGVmdCA9IFogLSBuZXdaVXNlZDsKICAgICAgICAgICAgICAgIGJvb2wgZmVhc2libGUgPSB0cnVlOwogICAgICAgICAgICAgICAgaWYgKFogPiBuZXdaVXNlZCkgewogICAgICAgICAgICAgICAgICAgIGludCBtYXhaQ29uc3RyYWludCA9IGNaW25ld1pVc2VkKzFdOwogICAgICAgICAgICAgICAgICAgIGlmIChvbmVzTGVmdCA8IG1heFpDb25zdHJhaW50KSBmZWFzaWJsZSA9IGZhbHNlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaWYgKE8gPiBuZXdPVXNlZCkgewogICAgICAgICAgICAgICAgICAgIGludCBtYXhPQ29uc3RyYWludCA9IGNPW25ld09Vc2VkKzFdOwogICAgICAgICAgICAgICAgICAgIGlmICh6ZXJvc0xlZnQgPCBtYXhPQ29uc3RyYWludCkgZmVhc2libGUgPSBmYWxzZTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgIGNhblBpY2tPbmUgPSBmZWFzaWJsZTsKICAgICAgICAgICAgfQoKICAgICAgICAgICAgaWYgKGNhblBpY2taZXJvICYmICFjYW5QaWNrT25lKSB7CiAgICAgICAgICAgICAgICBVLnB1c2hfYmFjaygwKTsKICAgICAgICAgICAgICAgIHpVc2VkKys7CiAgICAgICAgICAgIH0gZWxzZSBpZiAoIWNhblBpY2taZXJvICYmIGNhblBpY2tPbmUpIHsKICAgICAgICAgICAgICAgIFUucHVzaF9iYWNrKDEpOwogICAgICAgICAgICAgICAgb1VzZWQrKzsKICAgICAgICAgICAgfSBlbHNlIGlmIChjYW5QaWNrWmVybyAmJiBjYW5QaWNrT25lKSB7CiAgICAgICAgICAgICAgICBpZiAoY1pjID4gY09jKSB7CiAgICAgICAgICAgICAgICAgICAgVS5wdXNoX2JhY2soMCk7CiAgICAgICAgICAgICAgICAgICAgelVzZWQrKzsKICAgICAgICAgICAgICAgIH0gZWxzZSBpZiAoY09jID4gY1pjKSB7CiAgICAgICAgICAgICAgICAgICAgVS5wdXNoX2JhY2soMSk7CiAgICAgICAgICAgICAgICAgICAgb1VzZWQrKzsKICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgaWYgKChaIC0gelVzZWQpIDwgKE8gLSBvVXNlZCkpIHsKICAgICAgICAgICAgICAgICAgICAgICAgVS5wdXNoX2JhY2soMCk7CiAgICAgICAgICAgICAgICAgICAgICAgIHpVc2VkKys7CiAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHsKICAgICAgICAgICAgICAgICAgICAgICAgVS5wdXNoX2JhY2soMSk7CiAgICAgICAgICAgICAgICAgICAgICAgIG9Vc2VkKys7CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgcmV0dXJuIHstMX07CiAgICAgICAgICAgIH0KICAgICAgICB9CiAgICB9CgogICAgLy8gVmFsaWRhdGUgVSB3aXRoIGNvbnN0cmFpbnRzLgogICAgdmVjdG9yPGludD4gemVyb1Bvc2l0aW9ucywgb25lUG9zaXRpb25zOwogICAgemVyb1Bvc2l0aW9ucy5yZXNlcnZlKFopOwogICAgb25lUG9zaXRpb25zLnJlc2VydmUoTyk7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IChpbnQpVS5zaXplKCk7IGkrKykgewogICAgICAgIGlmIChVW2ldID09IDApIHplcm9Qb3NpdGlvbnMucHVzaF9iYWNrKGkpOwogICAgICAgIGVsc2Ugb25lUG9zaXRpb25zLnB1c2hfYmFjayhpKTsKICAgIH0KCiAgICBmb3IgKGludCB6ID0gMDsgeiA8PSBaOyB6KyspIHsKICAgICAgICBpbnQgcG9zID0gKHogPT0gMCkgPyAtMSA6IHplcm9Qb3NpdGlvbnNbei0xXTsKICAgICAgICBpbnQgb25lc1JpZ2h0ID0gKGludClvbmVQb3NpdGlvbnMuc2l6ZSgpIC0gKGludCkoc3RkOjpsb3dlcl9ib3VuZChvbmVQb3NpdGlvbnMuYmVnaW4oKSwgb25lUG9zaXRpb25zLmVuZCgpLCBwb3MrMSkgLSBvbmVQb3NpdGlvbnMuYmVnaW4oKSk7CiAgICAgICAgaWYgKG9uZXNSaWdodCA8IGNaW3pdKSB7CiAgICAgICAgICAgIHJldHVybiB7LTF9OwogICAgICAgIH0KICAgIH0KCiAgICBmb3IgKGludCB3ID0gMDsgdyA8PSBPOyB3KyspIHsKICAgICAgICBpbnQgcG9zID0gKHcgPT0gMCkgPyAtMSA6IG9uZVBvc2l0aW9uc1t3LTFdOwogICAgICAgIGludCB6ZXJvc1JpZ2h0ID0gKGludCl6ZXJvUG9zaXRpb25zLnNpemUoKSAtIChpbnQpKHN0ZDo6bG93ZXJfYm91bmQoemVyb1Bvc2l0aW9ucy5iZWdpbigpLCB6ZXJvUG9zaXRpb25zLmVuZCgpLCBwb3MrMSkgLSB6ZXJvUG9zaXRpb25zLmJlZ2luKCkpOwogICAgICAgIGlmICh6ZXJvc1JpZ2h0IDwgY09bd10pIHsKICAgICAgICAgICAgcmV0dXJuIHstMX07CiAgICAgICAgfQogICAgfQoKICAgIGF1dG8gaXNTdWJzZXF1ZW5jZSA9IFsmXShjb25zdCBzdGQ6OnZlY3RvcjxpbnQ+ICZDLCBjb25zdCBzdGQ6OnZlY3RvcjxpbnQ+ICZYKSB7CiAgICAgICAgaW50IGlkeCA9IDA7CiAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCAoaW50KVguc2l6ZSgpICYmIGlkeCA8IChpbnQpQy5zaXplKCk7IGkrKykgewogICAgICAgICAgICBpZiAoWFtpXSA9PSBDW2lkeF0pIGlkeCsrOwogICAgICAgIH0KICAgICAgICByZXR1cm4gaWR4ID09IChpbnQpQy5zaXplKCk7CiAgICB9OwoKICAgIGlmICghaXNTdWJzZXF1ZW5jZShVLCBBKSB8fCAhaXNTdWJzZXF1ZW5jZShVLCBCKSkgewogICAgICAgIHJldHVybiB7LTF9OwogICAgfQoKICAgIHJldHVybiBVOwp9)

#include "hieroglyphs.h"

#include <bits/stdc++.h>

using namespace std;

std::vector<int> ucs(std::vector<int> A, std::vector<int> B)

{

int N = (int)A.size();

int M = (int)B.size();

int a0 = 0, a1 = 0;

for (int i = 0; i < N; i++) {

if (A\[i\] == 0) a0++;

else if (A\[i\] == 1) a1++;

}

int b0 = 0, b1 = 0;

for (int i = 0; i < M; i++) {

if (B\[i\] == 0) b0++;

else if (B\[i\] == 1) b1++;

}

int Z = min(a0, b0);

int O = min(a1, b1);

// If Z=0 and O=0, universal subsequence is the empty sequence.

if (Z == 0 && O == 0) {

return {};

}

vector<int> zeroPosA; zeroPosA.reserve(a0);

for (int i = 0; i < N; i++) if (A\[i\] == 0) zeroPosA.push\_back(i);

vector<int> zeroPosB; zeroPosB.reserve(b0);

for (int i = 0; i < M; i++) if (B\[i\] == 0) zeroPosB.push\_back(i);

vector<int> prefixOnesA(N+1, 0);

for (int i = 1; i <= N; i++) prefixOnesA\[i\] = prefixOnesA\[i-1\] + (A\[i-1\] == 1);

vector<int> prefixOnesB(M+1, 0);

for (int i = 1; i <= M; i++) prefixOnesB\[i\] = prefixOnesB\[i-1\] + (B\[i-1\] == 1);

auto getOnesAfterZero = \[&\](vector<int> & zeroPos, vector<int> & prefixOnes, int totalOnes, int z) {

if (z == 0) return totalOnes;

else {

if (z-1 >= (int)zeroPos.size()) return 0; // safety

int pos = zeroPos\[z-1\];

int onesUsed = prefixOnes\[pos+1\];

int onesAfter = totalOnes - onesUsed;

return onesAfter;

}

};

vector<int> fAval(Z+1), fBval(Z+1);

for (int z = 0; z <= Z; z++) {

fAval\[z\] = getOnesAfterZero(zeroPosA, prefixOnesA, a1, z);

fBval\[z\] = getOnesAfterZero(zeroPosB, prefixOnesB, b1, z);

}

vector<int> cZ(Z+1);

for (int z = 0; z <= Z; z++) cZ\[z\] = min(fAval\[z\], fBval\[z\]);

vector<int> onePosA; onePosA.reserve(a1);

for (int i = 0; i < N; i++) if (A\[i\] == 1) onePosA.push\_back(i);

vector<int> onePosB; onePosB.reserve(b1);

for (int i = 0; i < M; i++) if (B\[i\] == 1) onePosB.push\_back(i);

vector<int> prefixZerosA(N+1, 0);

for (int i = 1; i <= N; i++) prefixZerosA\[i\] = prefixZerosA\[i-1\] + (A\[i-1\] == 0);

vector<int> prefixZerosB(M+1, 0);

for (int i = 1; i <= M; i++) prefixZerosB\[i\] = prefixZerosB\[i-1\] + (B\[i-1\] == 0);

auto getZerosAfterOne = \[&\](vector<int> & onePos, vector<int> & prefixZeros, int totalZeros, int w) {

if (w == 0) return totalZeros;

else {

if (w-1 >= (int)onePos.size()) return 0; // safety

int pos = onePos\[w-1\];

int zerosUsed = prefixZeros\[pos+1\];

int zerosAfter = totalZeros - zerosUsed;

return zerosAfter;

}

};

vector<int> gAval(O+1), gBval(O+1);

for (int w = 0; w <= O; w++) {

gAval\[w\] = getZerosAfterOne(onePosA, prefixZerosA, a0, w);

gBval\[w\] = getZerosAfterOne(onePosB, prefixZerosB, b0, w);

}

vector<int> cO(O+1);

for (int w = 0; w <= O; w++) cO\[w\] = min(gAval\[w\], gBval\[w\]);

vector<int> U;

U.reserve(Z+O);

int zUsed = 0;

int oUsed = 0;

auto canPlaceZero = \[&\](int zUsed, int oUsed) {

if (zUsed >= Z) return false;

int c = cZ\[zUsed+1\]; // constraint for the next zero in U.

int onesLeft = O - oUsed;

return onesLeft >= c;

};

auto canPlaceOne = \[&\](int zUsed, int oUsed) {

if (oUsed >= O) return false;

int c = cO\[oUsed+1\]; // constraint for the next one in U.

int zerosLeft = Z - zUsed;

return zerosLeft >= c;

};

for (int i = 0; i < Z+O; i++) {

bool placeZ = canPlaceZero(zUsed, oUsed);

bool placeO = canPlaceOne(zUsed, oUsed);

if (!placeZ &&!placeO) {

return {-1};

} else if (placeZ &&!placeO) {

U.push\_back(0);

zUsed++;

} else if (!placeZ && placeO) {

U.push\_back(1);

oUsed++;

} else {

int cZc = cZ\[zUsed+1\];

int cOc = cO\[oUsed+1\];

bool canPickZero = false;

bool canPickOne = false;

{

int newZUsed = zUsed + 1;

int newOUsed = oUsed;

int onesLeft = O - newOUsed;

int zerosLeft = Z - newZUsed;

bool feasible = true;

if (Z > newZUsed) { // check cZ in range \[newZUsed+1..Z\]

int maxZConstraint = cZ\[newZUsed+1\];

if (onesLeft < maxZConstraint) feasible = false;

}

if (O > newOUsed) { // check cO in range \[newOUsed+1..O\]

int maxOConstraint = cO\[newOUsed+1\];

if (zerosLeft < maxOConstraint) feasible = false;

}

canPickZero = feasible;

}

{

int newZUsed = zUsed;

int newOUsed = oUsed + 1;

int onesLeft = O - newOUsed;

int zerosLeft = Z - newZUsed;

bool feasible = true;

if (Z > newZUsed) {

int maxZConstraint = cZ\[newZUsed+1\];

if (onesLeft < maxZConstraint) feasible = false;

}

if (O > newOUsed) {

int maxOConstraint = cO\[newOUsed+1\];

if (zerosLeft < maxOConstraint) feasible = false;

}

canPickOne = feasible;

}

if (canPickZero &&!canPickOne) {

U.push\_back(0);

zUsed++;

} else if (!canPickZero && canPickOne) {

U.push\_back(1);

oUsed++;

} else if (canPickZero && canPickOne) {

if (cZc > cOc) {

U.push\_back(0);

zUsed++;

} else if (cOc > cZc) {

U.push\_back(1);

oUsed++;

} else {

if ((Z - zUsed) < (O - oUsed)) {

U.push\_back(0);

zUsed++;

} else {

U.push\_back(1);

oUsed++;

}

}

} else {

return {-1};

}

}

}

// Validate U with constraints.

vector<int> zeroPositions, onePositions;

zeroPositions.reserve(Z);

onePositions.reserve(O);

for (int i = 0; i < (int)U.size(); i++) {

if (U\[i\] == 0) zeroPositions.push\_back(i);

else onePositions.push\_back(i);

}

for (int z = 0; z <= Z; z++) {

int pos = (z == 0)? -1: zeroPositions\[z-1\];

int onesRight = (int)onePositions.size() - (int)(std::lower\_bound(onePositions.begin(), onePositions.end(), pos+1) - onePositions.begin());

if (onesRight < cZ\[z\]) {

return {-1};

}

}

for (int w = 0; w <= O; w++) {

int pos = (w == 0)? -1: onePositions\[w-1\];

int zerosRight = (int)zeroPositions.size() - (int)(std::lower\_bound(zeroPositions.begin(), zeroPositions.end(), pos+1) - zeroPositions.begin());

if (zerosRight < cO\[w\]) {

return {-1};

}

}

auto isSubsequence = \[&\](const std::vector<int> &C, const std::vector<int> &X) {

int idx = 0;

for (int i = 0; i < (int)X.size() && idx < (int)C.size(); i++) {

if (X\[i\] == C\[idx\]) idx++;

}

return idx == (int)C.size();

};

if (!isSubsequence(U, A) ||!isSubsequence(U, B)) {

return {-1};

}

return U;

}

### C.5 Mosaic

A total of 42 points were scored on Mosaic across two separate submissions.

#### C.5.1 Submission 1

The first submission achieved a score of 22 out of 100, with points distributed across subtasks 1, 2, and 4.

[⬇](data:text/plain;base64,I2luY2x1ZGUgIm1vc2FpYy5oIgojaW5jbHVkZSA8Yml0cy9zdGRjKysuaD4KdXNpbmcgbmFtZXNwYWNlIHN0ZDsKCnN0ZDo6dmVjdG9yPGxvbmcgbG9uZz4gbW9zYWljKHN0ZDo6dmVjdG9yPGludD4gWCwgc3RkOjp2ZWN0b3I8aW50PiBZLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IFQsIHN0ZDo6dmVjdG9yPGludD4gQiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc3RkOjp2ZWN0b3I8aW50PiBMLCBzdGQ6OnZlY3RvcjxpbnQ+IFIpCnsKICAgIGludCBOID0gWC5zaXplKCk7CiAgICBpbnQgUSA9IFQuc2l6ZSgpOwogICAgaWYoTiA9PSAwKSB7CiAgICAgICAgcmV0dXJuIHZlY3Rvcjxsb25nIGxvbmc+KFEsIDBMTCk7CiAgICB9CgogICAgLy8gU3RlcCAxOiBmaWxsIEEgKE4qTiBhcnJheSwgc3RvcmUgYXMgY2hhciAwLzEpCiAgICB2ZWN0b3I8dW5zaWduZWQgY2hhcj4gQTsKICAgIEEucmVzaXplKChzaXplX3QpTiAqIChzaXplX3QpTik7CiAgICAvLyBmaWxsIHRvcCByb3cKICAgIGZvcihpbnQgaj0wOyBqPE47IGorKykgewogICAgICAgIEFbMCpOICsgal0gPSBYW2pdOwogICAgfQogICAgLy8gZmlsbCBsZWZ0IGNvbHVtbgogICAgZm9yKGludCBpPTA7IGk8TjsgaSsrKSB7CiAgICAgICAgQVtpKk4gKyAwXSA9IFlbaV07CiAgICB9CiAgICAvLyBmaWxsIHJlc3QKICAgIGZvcihpbnQgaT0xOyBpPE47IGkrKykgewogICAgICAgIGZvcihpbnQgaj0xOyBqPE47IGorKykgewogICAgICAgICAgICBBW2kqTiArIGpdID0gKDEgLSBBWyhpLTEpKk4gKyBqXSkgKiAoMSAtIEFbaSpOICsgai0xXSk7CiAgICAgICAgfQogICAgfQoKICAgIC8vIFN0ZXAgMjogY29tcHV0ZSBwcmVmaXggc3VtcyBpbiBQIChOKzEpKihOKzEpIGFycmF5CiAgICB2ZWN0b3I8aW50PiBQOwogICAgUC5yZXNpemUoKHNpemVfdCkoTisxKSAqIChzaXplX3QpKE4rMSkpOwogICAgLy8gaW5pdGlhbGl6ZSBQIHdpdGggMCdzCiAgICBzdGQ6OmZpbGwoUC5iZWdpbigpLCBQLmVuZCgpLCAwKTsKCiAgICBmb3IoaW50IGk9MDsgaTxOOyBpKyspIHsKICAgICAgICBpbnQgYmFzZVJvdyA9IChpKzEpKihOKzEpOwogICAgICAgIGludCBwcmV2Um93ID0gaSooTisxKTsKICAgICAgICBmb3IoaW50IGo9MDsgajxOOyBqKyspIHsKICAgICAgICAgICAgUFtiYXNlUm93ICsgKGorMSldID0gUFtwcmV2Um93ICsgKGorMSldICsgUFtiYXNlUm93ICsgal0gLSBQW3ByZXZSb3cgKyBqXSArIChBW2kqTiArIGpdKTsKICAgICAgICB9CiAgICB9CgogICAgLy8gU3RlcCAzOiBhbnN3ZXIgcXVlcmllcyB1c2luZyBQCiAgICB2ZWN0b3I8bG9uZyBsb25nPiBDOwogICAgQy5yZXNpemUoUSk7CiAgICBmb3IoaW50IGs9MDsgazxROyBrKyspIHsKICAgICAgICBpbnQgdG9wID0gVFtrXTsKICAgICAgICBpbnQgYm90dG9tID0gQltrXTsKICAgICAgICBpbnQgbGVmdCA9IExba107CiAgICAgICAgaW50IHJpZ2h0ID0gUltrXTsKICAgICAgICBsb25nIGxvbmcgYmxhY2tDb3VudCA9IChsb25nIGxvbmcpUFsoYm90dG9tKzEpKihOKzEpICsgKHJpZ2h0KzEpXQogICAgICAgICAgICAtIChsb25nIGxvbmcpUFsodG9wKSooTisxKSArIChyaWdodCsxKV0KICAgICAgICAgICAgLSAobG9uZyBsb25nKVBbKGJvdHRvbSsxKSooTisxKSArIGxlZnRdCiAgICAgICAgICAgICsgKGxvbmcgbG9uZylQWyh0b3ApKihOKzEpICsgbGVmdF07CiAgICAgICAgQ1trXSA9IGJsYWNrQ291bnQ7CiAgICB9CgogICAgcmV0dXJuIEM7Cn0=)

#include "mosaic.h"

#include <bits/stdc++.h>

using namespace std;

std::vector<long long> mosaic(std::vector<int> X, std::vector<int> Y,

std::vector<int> T, std::vector<int> B,

std::vector<int> L, std::vector<int> R)

{

int N = X.size();

int Q = T.size();

if(N == 0) {

return vector<long long>(Q, 0LL);

}

// Step 1: fill A (N\*N array, store as char 0/1)

vector<unsigned char> A;

A.resize((size\_t)N \* (size\_t)N);

// fill top row

for(int j=0; j<N; j++) {

A\[0\*N + j\] = X\[j\];

}

// fill left column

for(int i=0; i<N; i++) {

A\[i\*N + 0\] = Y\[i\];

}

// fill rest

for(int i=1; i<N; i++) {

for(int j=1; j<N; j++) {

A\[i\*N + j\] = (1 - A\[(i-1)\*N + j\]) \* (1 - A\[i\*N + j-1\]);

}

}

// Step 2: compute prefix sums in P (N+1)\*(N+1) array

vector<int> P;

P.resize((size\_t)(N+1) \* (size\_t)(N+1));

// initialize P with 0’s

std::fill(P.begin(), P.end(), 0);

for(int i=0; i<N; i++) {

int baseRow = (i+1)\*(N+1);

int prevRow = i\*(N+1);

for(int j=0; j<N; j++) {

P\[baseRow + (j+1)\] = P\[prevRow + (j+1)\] + P\[baseRow + j\] - P\[prevRow + j\] + (A\[i\*N + j\]);

}

}

// Step 3: answer queries using P

vector<long long> C;

C.resize(Q);

for(int k=0; k<Q; k++) {

int top = T\[k\];

int bottom = B\[k\];

int left = L\[k\];

int right = R\[k\];

long long blackCount = (long long)P\[(bottom+1)\*(N+1) + (right+1)\]

\- (long long)P\[(top)\*(N+1) + (right+1)\]

\- (long long)P\[(bottom+1)\*(N+1) + left\]

\+ (long long)P\[(top)\*(N+1) + left\];

C\[k\] = blackCount;

}

return C;

}

#### C.5.2 Submission 2

The model scored 20 points in the second submission on subtasks 1, 3, and 5.

[⬇](data:text/plain;base64,I2luY2x1ZGUgIm1vc2FpYy5oIgojaW5jbHVkZSA8Yml0cy9zdGRjKysuaD4KdXNpbmcgbmFtZXNwYWNlIHN0ZDsKCi8vIFdlJ2xsIGltcGxlbWVudCB0aGUgbW9zYWljIGZ1bmN0aW9uIGFzIHNwZWNpZmllZC4Kc3RkOjp2ZWN0b3I8bG9uZyBsb25nPiBtb3NhaWMoc3RkOjp2ZWN0b3I8aW50PiBYLCBzdGQ6OnZlY3RvcjxpbnQ+IFksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN0ZDo6dmVjdG9yPGludD4gVCwgc3RkOjp2ZWN0b3I8aW50PiBCLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IEwsIHN0ZDo6dmVjdG9yPGludD4gUikKewogICAgaW50IE4gPSBYLnNpemUoKTsKICAgIGludCBRID0gVC5zaXplKCk7CgogICAgLy8gSWYgTj0xLCB0aGUgbW9zYWljIGlzIGp1c3QgdGlsZSgwLDApLCBxdWVyaWVzIGFyZSBlYXN5LgogICAgaWYgKE4gPT0gMSkgewogICAgICAgIHZlY3Rvcjxsb25nIGxvbmc+IHJlc3VsdChRKTsKICAgICAgICBmb3IgKGludCBxID0gMDsgcSA8IFE7IHErKykgewogICAgICAgICAgICAvLyBzdWJyZWN0YW5nbGUgVC4uQiB4IEwuLlIgaXMgYWx3YXlzIFQ9MCxCPTAsTD0wLFI9MCBpZiB3ZSBoYXZlIHF1ZXJpZXMuCiAgICAgICAgICAgIGlmIChUW3FdID09IDAgJiYgQltxXSA9PSAwICYmIExbcV0gPT0gMCAmJiBSW3FdID09IDApIHsKICAgICAgICAgICAgICAgIC8vIGJsYWNrIHRpbGUgY291bnQgaXMgMSBpZiBYWzBdPTEgb3IgWVswXT0xLCBhbmQgWFswXT1ZWzBdLCBlbHNlIDAuCiAgICAgICAgICAgICAgICAvLyBUaGUgcHJvYmxlbSBzdGF0ZXMgWFswXSA9IFlbMF0sIHNvIHRpbGUoMCwwKSBpcyBibGFjayBpZiBYWzBdPTEuCiAgICAgICAgICAgICAgICBpZiAoWFswXSA9PSAxKSByZXN1bHRbcV0gPSAxOyBlbHNlIHJlc3VsdFtxXSA9IDA7CiAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAvLyBubyBvdGhlciBxdWVyaWVzIGFyZSBwb3NzaWJsZSBpZiBOPTEsIGJ1dCBpZiB0aGV5IGFyZSwgdGhlIHN1YnJlY3RhbmdsZSBpcyBlbXB0eSBvciBvdXQgb2YgcmFuZ2UuCiAgICAgICAgICAgICAgICByZXN1bHRbcV0gPSAwOwogICAgICAgICAgICB9CiAgICAgICAgfQogICAgICAgIHJldHVybiByZXN1bHQ7CiAgICB9CgogICAgLy8gUHJlY29tcHV0ZSBwcmVmaXggc3VtcyBmb3IgWCBhbmQgWS4KICAgIHZlY3Rvcjxsb25nIGxvbmc+IHByZWZpeFgoTisxLCAwKSwgcHJlZml4WShOKzEsIDApOwogICAgZm9yIChpbnQgaSA9IDE7IGkgPD0gTjsgaSsrKSB7CiAgICAgICAgcHJlZml4WFtpXSA9IHByZWZpeFhbaS0xXSArIFhbaS0xXTsKICAgICAgICBwcmVmaXhZW2ldID0gcHJlZml4WVtpLTFdICsgWVtpLTFdOwogICAgfQoKICAgIC8vIERldGVybWluZSBhbHBoYSBtb2QgMiBhbmQgYmV0YSBtb2QgMiBmcm9tIHRpbGUoMSwxKS4KICAgIC8vIENbMSwxXSA9IDEgaWYgWFsxXSBhbmQgWVsxXSBhcmUgMCwgZWxzZSAwLgogICAgaW50IGMxMSA9IDA7CiAgICBpZiAoTiA+IDEpIHsKICAgICAgICBpZiAoWC5zaXplKCkgPiAxICYmIFkuc2l6ZSgpID4gMSAmJiBYWzFdID09IDAgJiYgWVsxXSA9PSAwKSBjMTEgPSAxOyBlbHNlIGMxMSA9IDA7CiAgICB9IGVsc2UgewogICAgICAgIGMxMSA9IDA7IC8vIGJ1dCBOPTEgc2NlbmFyaW8gaXMgaGFuZGxlZCBhYm92ZS4KICAgIH0KCiAgICBpbnQgYWxwaGFNb2QyLCBiZXRhTW9kMjsKICAgIGlmIChjMTEgPT0gMSkgewogICAgICAgIGFscGhhTW9kMiA9IDA7CiAgICAgICAgYmV0YU1vZDIgPSAwOwogICAgfSBlbHNlIHsKICAgICAgICBhbHBoYU1vZDIgPSAwOwogICAgICAgIGJldGFNb2QyID0gMTsKICAgIH0KCiAgICAvLyBkZWZpbmUgZnVuY3Rpb25zIHRvIGNvdW50IGV2ZW4gYW5kIG9kZCBpbiBhIHJhbmdlLgogICAgYXV0byBjb3VudEV2ZW5JblJhbmdlID0gWyZdKGludCBhLCBpbnQgYil7CiAgICAgICAgaWYgKGIgPCBhKSByZXR1cm4gMExMOwogICAgICAgIGxvbmcgbG9uZyBBID0gYTsgbG9uZyBsb25nIEIgPSBiOwogICAgICAgIGxvbmcgbG9uZyBldmVuQ291bnQgPSAoKGxvbmcgbG9uZylCICsgMikgLyAyIC0gKChsb25nIGxvbmcpQSArIDEpIC8gMjsKICAgICAgICByZXR1cm4gZXZlbkNvdW50OwogICAgfTsKCiAgICBhdXRvIGNvdW50T2RkSW5SYW5nZSA9IFsmXShpbnQgYSwgaW50IGIpewogICAgICAgIGlmIChiIDwgYSkgcmV0dXJuIDBMTDsKICAgICAgICBsb25nIGxvbmcgQSA9IGE7IGxvbmcgbG9uZyBCID0gYjsKICAgICAgICBsb25nIGxvbmcgb2RkQ291bnQgPSAoKGxvbmcgbG9uZylCICsgMSkgLyAyIC0gKGxvbmcgbG9uZylBIC8gMjsKICAgICAgICByZXR1cm4gb2RkQ291bnQ7CiAgICB9OwoKICAgIC8vIE5vdyBmb3IgZWFjaCBxdWVyeSwgY29tcHV0ZSB0aGUgYmxhY2sgdGlsZSBjb3VudC4KICAgIHN0ZDo6dmVjdG9yPGxvbmcgbG9uZz4gcmVzdWx0KFEpOwogICAgZm9yIChpbnQgcSA9IDA7IHEgPCBROyBxKyspIHsKICAgICAgICBsb25nIGxvbmcgdCA9IFRbcV07CiAgICAgICAgbG9uZyBsb25nIGIgPSBCW3FdOwogICAgICAgIGxvbmcgbG9uZyBsID0gTFtxXTsKICAgICAgICBsb25nIGxvbmcgciA9IFJbcV07CgogICAgICAgIGxvbmcgbG9uZyBjb3VudCA9IDA7CgogICAgICAgIC8vIGlmIFQ9MCwgYWRkIGJsYWNrIHRpbGUgZnJvbSByb3cwIGFuZCBjb2x1bW5zIEwuLlIuCiAgICAgICAgaWYgKHQgPT0gMCAmJiByID49IGwpIHsKICAgICAgICAgICAgbG9uZyBsb25nIHN1bVJvdzAgPSBwcmVmaXhYW3IrMV0gLSBwcmVmaXhYWyhpbnQpbF07IC8vIHN1bSBvZiBYW0wuLlJdCiAgICAgICAgICAgIGNvdW50ICs9IHN1bVJvdzA7CiAgICAgICAgfQoKICAgICAgICAvLyBpZiBMPTAsIGFkZCBibGFjayB0aWxlIGZyb20gY29sdW1uMCBhbmQgcm93cyBULi5CLgogICAgICAgIGlmIChsID09IDAgJiYgYiA+PSB0KSB7CiAgICAgICAgICAgIGxvbmcgbG9uZyBzdW1Db2wwID0gcHJlZml4WVtiKzFdIC0gcHJlZml4WVsoaW50KXRdOyAvLyBzdW0gb2YgWVtULi5CXQogICAgICAgICAgICBjb3VudCArPSBzdW1Db2wwOwogICAgICAgIH0KCiAgICAgICAgLy8gaWYgVD0wIGFuZCBMPTAsIHN1YnRyYWN0IHRpbGUoMCwwKSBpZiBibGFjay4KICAgICAgICBpZiAodCA9PSAwICYmIGwgPT0gMCAmJiByID49IDAgJiYgYiA+PSAwKSB7CiAgICAgICAgICAgIGlmIChYWzBdID09IDEpIHsKICAgICAgICAgICAgICAgIGNvdW50IC09IDE7CiAgICAgICAgICAgIH0KICAgICAgICB9CgogICAgICAgIC8vIGFkZCBibGFjayB0aWxlIGZyb20gdGhlIGludGVyaW9yIHN1Ym1hdHJpeCBUJz1tYXgoVCwxKS4uQiB4IEwnPW1heChMLDEpLi5SLgogICAgICAgIGludCBUcHJpbWUgPSBtYXgoKGludCl0LCAxKTsKICAgICAgICBpbnQgTHByaW1lID0gbWF4KChpbnQpbCwgMSk7CiAgICAgICAgaWYgKFRwcmltZSA8PSBiICYmIExwcmltZSA8PSByKSB7CiAgICAgICAgICAgIGxvbmcgbG9uZyBldmVuSSA9IGNvdW50RXZlbkluUmFuZ2UoVHByaW1lLCAoaW50KWIpOwogICAgICAgICAgICBsb25nIGxvbmcgb2RkSSA9IGNvdW50T2RkSW5SYW5nZShUcHJpbWUsIChpbnQpYik7CiAgICAgICAgICAgIGxvbmcgbG9uZyBldmVuSiA9IGNvdW50RXZlbkluUmFuZ2UoTHByaW1lLCAoaW50KXIpOwogICAgICAgICAgICBsb25nIGxvbmcgb2RkSiA9IGNvdW50T2RkSW5SYW5nZShMcHJpbWUsIChpbnQpcik7CgogICAgICAgICAgICBsb25nIGxvbmcgaW50ZXJpb3JDb3VudCA9IDA7CiAgICAgICAgICAgIGlmIChhbHBoYU1vZDIgPT0gYmV0YU1vZDIpIHsKICAgICAgICAgICAgICAgIC8vIGJsYWNrIHRpbGUgYXBwZWFyIHdoZW4gaSUKICAgICAgICAgICAgICAgIGludGVyaW9yQ291bnQgPSBldmVuSSAqIGV2ZW5KICsgb2RkSSAqIG9kZEo7CiAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAvLyBibGFjayB0aWxlIGFwcGVhciB3aGVuIGklCiAgICAgICAgICAgICAgICBpbnRlcmlvckNvdW50ID0gZXZlbkkgKiBvZGRKICsgb2RkSSAqIGV2ZW5KOwogICAgICAgICAgICB9CiAgICAgICAgICAgIGNvdW50ICs9IGludGVyaW9yQ291bnQ7CiAgICAgICAgfQoKICAgICAgICByZXN1bHRbcV0gPSBjb3VudDsKICAgIH0KCiAgICByZXR1cm4gcmVzdWx0Owp9)

#include "mosaic.h"

#include <bits/stdc++.h>

using namespace std;

// We’ll implement the mosaic function as specified.

std::vector<long long> mosaic(std::vector<int> X, std::vector<int> Y,

std::vector<int> T, std::vector<int> B,

std::vector<int> L, std::vector<int> R)

{

int N = X.size();

int Q = T.size();

// If N=1, the mosaic is just tile(0,0), queries are easy.

if (N == 1) {

vector<long long> result(Q);

for (int q = 0; q < Q; q++) {

// subrectangle T..B x L..R is always T=0,B=0,L=0,R=0 if we have queries.

if (T\[q\] == 0 && B\[q\] == 0 && L\[q\] == 0 && R\[q\] == 0) {

// black tile count is 1 if X\[0\]=1 or Y\[0\]=1, and X\[0\]=Y\[0\], else 0.

// The problem states X\[0\] = Y\[0\], so tile(0,0) is black if X\[0\]=1.

if (X\[0\] == 1) result\[q\] = 1; else result\[q\] = 0;

} else {

// no other queries are possible if N=1, but if they are, the subrectangle is empty or out of range.

result\[q\] = 0;

}

}

return result;

}

// Precompute prefix sums for X and Y.

vector<long long> prefixX(N+1, 0), prefixY(N+1, 0);

for (int i = 1; i <= N; i++) {

prefixX\[i\] = prefixX\[i-1\] + X\[i-1\];

prefixY\[i\] = prefixY\[i-1\] + Y\[i-1\];

}

// Determine alpha mod 2 and beta mod 2 from tile(1,1).

// C\[1,1\] = 1 if X\[1\] and Y\[1\] are 0, else 0.

int c11 = 0;

if (N > 1) {

if (X.size() > 1 && Y.size() > 1 && X\[1\] == 0 && Y\[1\] == 0) c11 = 1; else c11 = 0;

} else {

c11 = 0; // but N=1 scenario is handled above.

}

int alphaMod2, betaMod2;

if (c11 == 1) {

alphaMod2 = 0;

betaMod2 = 0;

} else {

alphaMod2 = 0;

betaMod2 = 1;

}

// define functions to count even and odd in a range.

auto countEvenInRange = \[&\](int a, int b){

if (b < a) return 0LL;

long long A = a; long long B = b;

long long evenCount = ((long long)B + 2) / 2 - ((long long)A + 1) / 2;

return evenCount;

};

auto countOddInRange = \[&\](int a, int b){

if (b < a) return 0LL;

long long A = a; long long B = b;

long long oddCount = ((long long)B + 1) / 2 - (long long)A / 2;

return oddCount;

};

// Now for each query, compute the black tile count.

std::vector<long long> result(Q);

for (int q = 0; q < Q; q++) {

long long t = T\[q\];

long long b = B\[q\];

long long l = L\[q\];

long long r = R\[q\];

long long count = 0;

// if T=0, add black tile from row0 and columns L..R.

if (t == 0 && r >= l) {

long long sumRow0 = prefixX\[r+1\] - prefixX\[(int)l\]; // sum of X\[L..R\]

count += sumRow0;

}

// if L=0, add black tile from column0 and rows T..B.

if (l == 0 && b >= t) {

long long sumCol0 = prefixY\[b+1\] - prefixY\[(int)t\]; // sum of Y\[T..B\]

count += sumCol0;

}

// if T=0 and L=0, subtract tile(0,0) if black.

if (t == 0 && l == 0 && r >= 0 && b >= 0) {

if (X\[0\] == 1) {

count -= 1;

}

}

// add black tile from the interior submatrix T’=max(T,1)..B x L’=max(L,1)..R.

int Tprime = max((int)t, 1);

int Lprime = max((int)l, 1);

if (Tprime <= b && Lprime <= r) {

long long evenI = countEvenInRange(Tprime, (int)b);

long long oddI = countOddInRange(Tprime, (int)b);

long long evenJ = countEvenInRange(Lprime, (int)r);

long long oddJ = countOddInRange(Lprime, (int)r);

long long interiorCount = 0;

if (alphaMod2 == betaMod2) {

// black tile appear when i%

interiorCount = evenI \* evenJ + oddI \* oddJ;

} else {

// black tile appear when i%

interiorCount = evenI \* oddJ + oddI \* evenJ;

}

count += interiorCount;

}

result\[q\] = count;

}

return result;

}

### C.6 Sphinx

A total of 71.5 points were scored on Sphinx across two separate submissions.

#### C.6.1 Submission 1

The first submission achieved a score of 50 out of 100, with 50% partial credit earned on all subtasks.

[⬇](data:text/plain;base64,I2luY2x1ZGUgInNwaGlueC5oIgojaW5jbHVkZSA8Yml0cy9zdGRjKysuaD4KaW50IHBlcmZvcm1fZXhwZXJpbWVudChzdGQ6OnZlY3RvcjxpbnQ+IEUpOwpzdGQ6OnZlY3RvcjxpbnQ+IGZpbmRfY29sb3VycyhpbnQgTngsIHN0ZDo6dmVjdG9yPGludD4gWHgsIHN0ZDo6dmVjdG9yPGludD4gWXgpIHsKICAgIGludCBOID0gTng7CiAgICBzdGQ6OnZlY3RvcjxzdGQ6OnZlY3RvcjxpbnQ+PiBhZGooTik7CiAgICBpbnQgTSA9IFh4LnNpemUoKTsKICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTTsgaSsrKSB7CiAgICAgICAgaW50IHUgPSBYeFtpXSwgdiA9IFl4W2ldOwogICAgICAgIGFkalt1XS5wdXNoX2JhY2sodik7CiAgICAgICAgYWRqW3ZdLnB1c2hfYmFjayh1KTsKICAgIH0KCiAgICBzdHJ1Y3QgVW5pb25GaW5kIHsKICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IHBhcmVudCwgcmFuazsKICAgICAgICBVbmlvbkZpbmQoaW50IG4pIDogcGFyZW50KG4pLCByYW5rKG4sIDApIHsKICAgICAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCBuOyBpKyspIHBhcmVudFtpXSA9IGk7CiAgICAgICAgfQogICAgICAgIGludCBmaW5kcChpbnQgeCkgewogICAgICAgICAgICBpZiAocGFyZW50W3hdID09IHgpIHJldHVybiB4OwogICAgICAgICAgICBwYXJlbnRbeF0gPSBmaW5kcChwYXJlbnRbeF0pOwogICAgICAgICAgICByZXR1cm4gcGFyZW50W3hdOwogICAgICAgIH0KICAgICAgICB2b2lkIHVuaXRlKGludCB4LCBpbnQgeSkgewogICAgICAgICAgICB4ID0gZmluZHAoeCk7IHkgPSBmaW5kcCh5KTsKICAgICAgICAgICAgaWYgKHggPT0geSkgcmV0dXJuOwogICAgICAgICAgICBpZiAocmFua1t4XSA8IHJhbmtbeV0pIHBhcmVudFt4XSA9IHk7CiAgICAgICAgICAgIGVsc2UgaWYgKHJhbmtbeF0gPiByYW5rW3ldKSBwYXJlbnRbeV0gPSB4OwogICAgICAgICAgICBlbHNlIHtwYXJlbnRbeV0gPSB4OyByYW5rW3hdKys7fQogICAgICAgIH0KICAgIH0gdWYoTik7CgogICAgYXV0byBjb25uZWN0ZWRDb21wb25lbnRzSW5TdWJzZXQgPSBbJl0oY29uc3Qgc3RkOjp2ZWN0b3I8aW50PiYgc3Vic2V0KSB7CiAgICAgICAgc3RhdGljIHN0ZDo6dmVjdG9yPGludD4gdmlzaXRlZDsKICAgICAgICB2aXNpdGVkLmFzc2lnbihOLCAtMSk7CiAgICAgICAgZm9yIChpbnQgdiA6IHN1YnNldCkgdmlzaXRlZFt2XSA9IDA7CiAgICAgICAgaW50IGNvdW50ID0gMDsKICAgICAgICBmb3IgKGludCB2IDogc3Vic2V0KSBpZiAodmlzaXRlZFt2XSA9PSAwKSB7CiAgICAgICAgICAgIGNvdW50Kys7CiAgICAgICAgICAgIHN0ZDo6cXVldWU8aW50PiBxOwogICAgICAgICAgICBxLnB1c2godik7CiAgICAgICAgICAgIHZpc2l0ZWRbdl0gPSAxOwogICAgICAgICAgICB3aGlsZSghcS5lbXB0eSgpKSB7CiAgICAgICAgICAgICAgICBpbnQgdSA9IHEuZnJvbnQoKTsgcS5wb3AoKTsKICAgICAgICAgICAgICAgIGZvciAoaW50IHcgOiBhZGpbdV0pIHsKICAgICAgICAgICAgICAgICAgICBpZiAodmlzaXRlZFt3XSA9PSAwKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIHZpc2l0ZWRbd10gPSAxOwogICAgICAgICAgICAgICAgICAgICAgICBxLnB1c2godyk7CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9CiAgICAgICAgfQogICAgICAgIHJldHVybiBjb3VudDsKICAgIH07CgogICAgYXV0byBxdWVyeU1vbm9jaHJvbUNvbXBzID0gWyZdKGNvbnN0IHN0ZDo6dmVjdG9yPGludD4mIFMpIHsKICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IEVhcnIoTik7CiAgICAgICAgc3RhdGljIHN0ZDo6dmVjdG9yPGludD4gdmlzaXRlZDI7CiAgICAgICAgdmlzaXRlZDIuYXNzaWduKE4sIC0xKTsKICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IFNzZXQoTiwgMCk7CiAgICAgICAgZm9yIChpbnQgdiA6IFMpIFNzZXRbdl0gPSAxOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSBFYXJyW2ldID0gKFNzZXRbaV0gPT0gMSkgPyAtMSA6IE47CgogICAgICAgIGludCByZXMgPSBwZXJmb3JtX2V4cGVyaW1lbnQoRWFycik7CgogICAgICAgIHN0ZDo6dmVjdG9yPGludD4gY29tcElucHV0OyBjb21wSW5wdXQucmVzZXJ2ZShOIC0gUy5zaXplKCkpOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgTjsgaSsrKSBpZiAoU3NldFtpXSA9PSAwKSBjb21wSW5wdXQucHVzaF9iYWNrKGkpOwogICAgICAgIGludCBjb21wQ291bnQgPSAwOwogICAgICAgIGlmICghY29tcElucHV0LmVtcHR5KCkpIGNvbXBDb3VudCA9IGNvbm5lY3RlZENvbXBvbmVudHNJblN1YnNldChjb21wSW5wdXQpOwogICAgICAgIGVsc2UgY29tcENvdW50ID0gMDsKCiAgICAgICAgaW50IG1vbm9zID0gcmVzIC0gY29tcENvdW50OwogICAgICAgIHJldHVybiBtb25vczsKICAgIH07CgogICAgc3RkOjp2ZWN0b3I8aW50PiBjKE4rMSwgMCk7CiAgICBjWzBdID0gMDsgLy8gbm8gdmVydGljZXMsICNjb25uZWN0ZWRDb21wb25lbnRzKDApID0gMC4KICAgIGZvciAoaW50IGsgPSAxOyBrIDw9IE47IGsrKykgewogICAgICAgIHN0ZDo6dmVjdG9yPGludD4gUzsgUy5yZXNlcnZlKGspOwogICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgazsgaSsrKSBTLnB1c2hfYmFjayhpKTsKICAgICAgICBjW2tdID0gcXVlcnlNb25vY2hyb21Db21wcyhTKTsKICAgIH0KCiAgICBmb3IgKGludCBrID0gMTsgayA8PSBOOyBrKyspIHsKICAgICAgICBpbnQgdiA9IGstMTsKICAgICAgICBpbnQgbWVyZ2VzID0gY1trLTFdICsgMSAtIGNba107CiAgICAgICAgaWYgKG1lcmdlcyA8IDApIG1lcmdlcyA9IDA7IC8vIHNhZmV0eS4KCiAgICAgICAgYXV0byBmaW5kTWVyZ2VzID0gWyZdKGF1dG8gJnNlbGYsIHN0ZDo6dmVjdG9yPGludD4mIHJvb3RzLCBpbnQgbWVyZ2VzKSAtPiB2b2lkIHsKICAgICAgICAgICAgaWYgKG1lcmdlcyA9PSAwKSByZXR1cm47CiAgICAgICAgICAgIGlmIChtZXJnZXMgPT0gKGludClyb290cy5zaXplKCkpIHsKICAgICAgICAgICAgICAgIGZvciAoaW50IHIgOiByb290cykgewogICAgICAgICAgICAgICAgICAgIHVmLnVuaXRlKHYsIHIpOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgcmV0dXJuOwogICAgICAgICAgICB9IGVsc2UgewogICAgICAgICAgICAgICAgaWYgKHJvb3RzLnNpemUoKSA9PSBtZXJnZXMpIHsKICAgICAgICAgICAgICAgICAgICBmb3IgKGludCByIDogcm9vdHMpIHsKICAgICAgICAgICAgICAgICAgICAgICAgdWYudW5pdGUodiwgcik7CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgIHJldHVybjsKICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgaW50IGhhbGYgPSByb290cy5zaXplKCkvMjsKICAgICAgICAgICAgICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IGxlZnQocm9vdHMuYmVnaW4oKSwgcm9vdHMuYmVnaW4oKSArIGhhbGYpOwogICAgICAgICAgICAgICAgICAgIHN0ZDo6dmVjdG9yPGludD4gcmlnaHQocm9vdHMuYmVnaW4oKSArIGhhbGYsIHJvb3RzLmVuZCgpKTsKCiAgICAgICAgICAgICAgICAgICAgc3RkOjp2ZWN0b3I8aW50PiBTMjsgUzIucmVzZXJ2ZSgxICsgdik7CiAgICAgICAgICAgICAgICAgICAgUzIucHVzaF9iYWNrKHYpOwogICAgICAgICAgICAgICAgICAgIGZvciAoaW50IHIgOiBsZWZ0KSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGZvciAoaW50IGkgPSAwOyBpIDwgdjsgaSsrKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodWYuZmluZHAoaSkgPT0gcikgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFMyLnB1c2hfYmFjayhpKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgIH0KCiAgICAgICAgICAgICAgICAgICAgc3RkOjp2ZWN0b3I8aW50PiBFYXJyKE4pOwogICAgICAgICAgICAgICAgICAgIHN0ZDo6dmVjdG9yPGludD4gU3NldDIoTiwgMCk7CiAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgaSA6IFMyKSBTc2V0MltpXSA9IDE7CiAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCBOOyBpKyspIEVhcnJbaV0gPSAoU3NldDJbaV0gPT0gMSkgPyAtMSA6IE47CgogICAgICAgICAgICAgICAgICAgIGludCByZXMgPSBwZXJmb3JtX2V4cGVyaW1lbnQoRWFycik7CgogICAgICAgICAgICAgICAgICAgIHN0ZDo6dmVjdG9yPGludD4gY29tcElucHV0OyBjb21wSW5wdXQucmVzZXJ2ZShOIC0gUzIuc2l6ZSgpKTsKICAgICAgICAgICAgICAgICAgICBzdGF0aWMgc3RkOjp2ZWN0b3I8aW50PiB2aXNpdGVkMjsgdmlzaXRlZDIuYXNzaWduKE4sIC0xKTsKICAgICAgICAgICAgICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8IE47IGkrKykgdmlzaXRlZDJbaV0gPSAwOwogICAgICAgICAgICAgICAgICAgIGZvciAoaW50IGkgOiBTMikgdmlzaXRlZDJbaV0gPSAtMTsKICAgICAgICAgICAgICAgICAgICBjb21wSW5wdXQuY2xlYXIoKTsKICAgICAgICAgICAgICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8IE47IGkrKykgaWYgKHZpc2l0ZWQyW2ldID09IDApIGNvbXBJbnB1dC5wdXNoX2JhY2soaSk7CiAgICAgICAgICAgICAgICAgICAgaW50IGNvbXBDb3VudCA9IDA7CiAgICAgICAgICAgICAgICAgICAgaWYgKCFjb21wSW5wdXQuZW1wdHkoKSkgewogICAgICAgICAgICAgICAgICAgICAgICBmb3IgKGludCBpcCA6IGNvbXBJbnB1dCkgdmlzaXRlZDJbaXBdID0gMDsgLy8gcmUtbWFyayBmb3IgQkZTLgogICAgICAgICAgICAgICAgICAgICAgICBmb3IgKGludCBpcCA6IGNvbXBJbnB1dCkgaWYgKHZpc2l0ZWQyW2lwXSA9PSAwKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb21wQ291bnQrKzsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN0ZDo6cXVldWU8aW50PiBxOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgcS5wdXNoKGlwKTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZpc2l0ZWQyW2lwXSA9IDE7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICB3aGlsZSghcS5lbXB0eSgpKSB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW50IHUgPSBxLmZyb250KCk7IHEucG9wKCk7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgdyA6IGFkalt1XSkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodmlzaXRlZDJbd10gPT0gMCkgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmlzaXRlZDJbd10gPSAxOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcS5wdXNoKHcpOwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgfSBlbHNlIGNvbXBDb3VudCA9IDA7CgogICAgICAgICAgICAgICAgICAgIGludCBtb25vczIgPSByZXMgLSBjb21wQ291bnQ7CiAgICAgICAgICAgICAgICAgICAgaW50IG1lcmdlc0luTGVmdCA9IGxlZnQuc2l6ZSgpICsgMSAtIG1vbm9zMjsKCiAgICAgICAgICAgICAgICAgICAgc2VsZihzZWxmLCBsZWZ0LCBtZXJnZXNJbkxlZnQpOwogICAgICAgICAgICAgICAgICAgIHNlbGYoc2VsZiwgcmlnaHQsIG1lcmdlcyAtIG1lcmdlc0luTGVmdCk7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0KICAgICAgICB9OwoKICAgICAgICBzdGQ6OnNldDxpbnQ+IGRpc3RpbmN0Um9vdHM7CiAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCB2OyBpKyspIGRpc3RpbmN0Um9vdHMuaW5zZXJ0KHVmLmZpbmRwKGkpKTsKICAgICAgICBzdGQ6OnZlY3RvcjxpbnQ+IHJvb3RzVmVjKGRpc3RpbmN0Um9vdHMuYmVnaW4oKSwgZGlzdGluY3RSb290cy5lbmQoKSk7CgogICAgICAgIGZpbmRNZXJnZXMoZmluZE1lcmdlcywgcm9vdHNWZWMsIG1lcmdlcyk7CiAgICB9CgogICAgc3RkOjp2ZWN0b3I8aW50PiByb290SUQoTiwgLTEpOwogICAgc3RkOjp2ZWN0b3I8aW50PiBjb2xvcklEKE4sIC0xKTsKICAgIGludCBuZXh0SUQgPSAwOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCBOOyBpKyspIHsKICAgICAgICBpbnQgciA9IHVmLmZpbmRwKGkpOwogICAgICAgIGlmIChyb290SURbcl0gPT0gLTEpIHsKICAgICAgICAgICAgcm9vdElEW3JdID0gbmV4dElEKys7CiAgICAgICAgfQogICAgICAgIGNvbG9ySURbaV0gPSByb290SURbcl07CiAgICB9CgogICAgcmV0dXJuIGNvbG9ySUQ7Cn0=)

#include "sphinx.h"

#include <bits/stdc++.h>

int perform\_experiment(std::vector<int> E);

std::vector<int> find\_colours(int Nx, std::vector<int> Xx, std::vector<int> Yx) {

int N = Nx;

std::vector<std::vector<int>> adj(N);

int M = Xx.size();

for (int i = 0; i < M; i++) {

int u = Xx\[i\], v = Yx\[i\];

adj\[u\].push\_back(v);

adj\[v\].push\_back(u);

}

struct UnionFind {

std::vector<int> parent, rank;

UnionFind(int n): parent(n), rank(n, 0) {

for (int i = 0; i < n; i++) parent\[i\] = i;

}

int findp(int x) {

if (parent\[x\] == x) return x;

parent\[x\] = findp(parent\[x\]);

return parent\[x\];

}

void unite(int x, int y) {

x = findp(x); y = findp(y);

if (x == y) return;

if (rank\[x\] < rank\[y\]) parent\[x\] = y;

else if (rank\[x\] > rank\[y\]) parent\[y\] = x;

else {parent\[y\] = x; rank\[x\]++;}

}

} uf(N);

auto connectedComponentsInSubset = \[&\](const std::vector<int>& subset) {

static std::vector<int> visited;

visited.assign(N, -1);

for (int v: subset) visited\[v\] = 0;

int count = 0;

for (int v: subset) if (visited\[v\] == 0) {

count++;

std::queue<int> q;

q.push(v);

visited\[v\] = 1;

while(!q.empty()) {

int u = q.front(); q.pop();

for (int w: adj\[u\]) {

if (visited\[w\] == 0) {

visited\[w\] = 1;

q.push(w);

}

}

}

}

return count;

};

auto queryMonochromComps = \[&\](const std::vector<int>& S) {

std::vector<int> Earr(N);

static std::vector<int> visited2;

visited2.assign(N, -1);

std::vector<int> Sset(N, 0);

for (int v: S) Sset\[v\] = 1;

for (int i = 0; i < N; i++) Earr\[i\] = (Sset\[i\] == 1)? -1: N;

int res = perform\_experiment(Earr);

std::vector<int> compInput; compInput.reserve(N - S.size());

for (int i = 0; i < N; i++) if (Sset\[i\] == 0) compInput.push\_back(i);

int compCount = 0;

if (!compInput.empty()) compCount = connectedComponentsInSubset(compInput);

else compCount = 0;

int monos = res - compCount;

return monos;

};

std::vector<int> c(N+1, 0);

c\[0\] = 0; // no vertices, #connectedComponents(0) = 0.

for (int k = 1; k <= N; k++) {

std::vector<int> S; S.reserve(k);

for (int i = 0; i < k; i++) S.push\_back(i);

c\[k\] = queryMonochromComps(S);

}

for (int k = 1; k <= N; k++) {

int v = k-1;

int merges = c\[k-1\] + 1 - c\[k\];

if (merges < 0) merges = 0; // safety.

auto findMerges = \[&\](auto &self, std::vector<int>& roots, int merges) -> void {

if (merges == 0) return;

if (merges == (int)roots.size()) {

for (int r: roots) {

uf.unite(v, r);

}

return;

} else {

if (roots.size() == merges) {

for (int r: roots) {

uf.unite(v, r);

}

return;

} else {

int half = roots.size()/2;

std::vector<int> left(roots.begin(), roots.begin() + half);

std::vector<int> right(roots.begin() + half, roots.end());

std::vector<int> S2; S2.reserve(1 + v);

S2.push\_back(v);

for (int r: left) {

for (int i = 0; i < v; i++) {

if (uf.findp(i) == r) {

S2.push\_back(i);

}

}

}

std::vector<int> Earr(N);

std::vector<int> Sset2(N, 0);

for (int i: S2) Sset2\[i\] = 1;

for (int i = 0; i < N; i++) Earr\[i\] = (Sset2\[i\] == 1)? -1: N;

int res = perform\_experiment(Earr);

std::vector<int> compInput; compInput.reserve(N - S2.size());

static std::vector<int> visited2; visited2.assign(N, -1);

for (int i = 0; i < N; i++) visited2\[i\] = 0;

for (int i: S2) visited2\[i\] = -1;

compInput.clear();

for (int i = 0; i < N; i++) if (visited2\[i\] == 0) compInput.push\_back(i);

int compCount = 0;

if (!compInput.empty()) {

for (int ip: compInput) visited2\[ip\] = 0; // re-mark for BFS.

for (int ip: compInput) if (visited2\[ip\] == 0) {

compCount++;

std::queue<int> q;

q.push(ip);

visited2\[ip\] = 1;

while(!q.empty()) {

int u = q.front(); q.pop();

for (int w: adj\[u\]) {

if (visited2\[w\] == 0) {

visited2\[w\] = 1;

q.push(w);

}

}

}

}

} else compCount = 0;

int monos2 = res - compCount;

int mergesInLeft = left.size() + 1 - monos2;

self(self, left, mergesInLeft);

self(self, right, merges - mergesInLeft);

}

}

};

std::set<int> distinctRoots;

for (int i = 0; i < v; i++) distinctRoots.insert(uf.findp(i));

std::vector<int> rootsVec(distinctRoots.begin(), distinctRoots.end());

findMerges(findMerges, rootsVec, merges);

}

std::vector<int> rootID(N, -1);

std::vector<int> colorID(N, -1);

int nextID = 0;

for (int i = 0; i < N; i++) {

int r = uf.findp(i);

if (rootID\[r\] == -1) {

rootID\[r\] = nextID++;

}

colorID\[i\] = rootID\[r\];

}

return colorID;

}

#### C.6.2 Submission 2

Submission 2 achieved 43 points on subtasks 1, 2, and 3.

[⬇](data:text/plain;base64,I2luY2x1ZGUgInNwaGlueC5oIgojaW5jbHVkZSA8Yml0cy9zdGRjKysuaD4KdXNpbmcgbmFtZXNwYWNlIHN0ZDsKCmludCBwZXJmb3JtX2V4cGVyaW1lbnQodmVjdG9yPGludD4gRSk7Cgp2ZWN0b3I8aW50PiBmaW5kX2NvbG91cnMoaW50IE4sIHZlY3RvcjxpbnQ+IFgsIHZlY3RvcjxpbnQ+IFkpIHsKICAgIC8vIFdlJ2xsIHN0b3JlIGFkamFjZW5jeS4KICAgIHZlY3Rvcjx2ZWN0b3I8aW50Pj4gYWRqKE4pOwogICAgdmVjdG9yPHZlY3Rvcjxib29sPj4gYWRqTWF0cml4KE4sIHZlY3Rvcjxib29sPihOLCBmYWxzZSkpOwogICAgZm9yIChpbnQgaSA9IDA7IGkgPCAoaW50KVguc2l6ZSgpOyBpKyspIHsKICAgICAgICBpbnQgdSA9IFhbaV07CiAgICAgICAgaW50IHYgPSBZW2ldOwogICAgICAgIGFkalt1XS5wdXNoX2JhY2sodik7CiAgICAgICAgYWRqW3ZdLnB1c2hfYmFjayh1KTsKICAgICAgICBhZGpNYXRyaXhbdV1bdl0gPSBhZGpNYXRyaXhbdl1bdV0gPSB0cnVlOwogICAgfQoKICAgIHZlY3RvcjxpbnQ+IGZpbmFsX2NvbG9ycyhOLCAtMSk7CiAgICB2ZWN0b3I8aW50PiB1bmFzc2lnbmVkOyBmb3IgKGludCBpID0gMDsgaSA8IE47IGkrKykgdW5hc3NpZ25lZC5wdXNoX2JhY2soaSk7CgogICAgLy8gZGVmaW5lIGdyb3VwIHRlc3RpbmcgZnVuY3Rpb25zLgogICAgLy8gZnVuY3Rpb24gcXVlcnlfY291bnRfaW5fVDogaG93IG1hbnkgaW4gVCBoYXZlIGNvbG9yIHguCiAgICBhdXRvIHF1ZXJ5X2NvdW50X2luX1QgPSBbJl0oY29uc3QgdmVjdG9yPGludD4gJlQsIGludCB4KSB7CiAgICAgICAgdmVjdG9yPGludD4gRShOLCB4KTsKICAgICAgICB2ZWN0b3I8aW50PiBpblQoTiwgMCk7CiAgICAgICAgZm9yIChpbnQgdiA6IFQpIHsKICAgICAgICAgICAgRVt2XSA9IC0xOwogICAgICAgICAgICBpblRbdl0gPSAxOwogICAgICAgIH0KCiAgICAgICAgaW50IFIgPSBwZXJmb3JtX2V4cGVyaW1lbnQoRSk7CgogICAgICAgIC8vIG51bWJlciBjIG9mIGNvbm5lY3RlZCBjb21wb25lbnRzIGluIHRoZSByZXN0LgogICAgICAgIHZlY3RvcjxpbnQ+IGNvbXBsZW1lbnQ7CiAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCBOOyBpKyspIGlmICghaW5UW2ldKSBjb21wbGVtZW50LnB1c2hfYmFjayhpKTsKICAgICAgICAvLyBCRlMgZm9yIGMuCiAgICAgICAgdmVjdG9yPGludD4gdmlzaXRlZChOLCAwKTsKICAgICAgICBpbnQgY2NvdW50ID0gMDsKICAgICAgICBmb3IgKGludCB2IDogY29tcGxlbWVudCkgewogICAgICAgICAgICBpZiAoIXZpc2l0ZWRbdl0pIHsKICAgICAgICAgICAgICAgIGNjb3VudCsrOwogICAgICAgICAgICAgICAgc3RhY2s8aW50PiBzdDsKICAgICAgICAgICAgICAgIHN0LnB1c2godik7CiAgICAgICAgICAgICAgICB2aXNpdGVkW3ZdID0gMTsKICAgICAgICAgICAgICAgIHdoaWxlICghc3QuZW1wdHkoKSkgewogICAgICAgICAgICAgICAgICAgIGludCBub2RlID0gc3QudG9wKCk7IHN0LnBvcCgpOwogICAgICAgICAgICAgICAgICAgIGZvciAoaW50IG5iIDogYWRqW25vZGVdKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGlmICghaW5UW25iXSAmJiAhdmlzaXRlZFtuYl0pIHsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZpc2l0ZWRbbmJdID0gMTsKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN0LnB1c2gobmIpOwogICAgICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9CiAgICAgICAgfQoKICAgICAgICBpbnQgYyA9IGNjb3VudDsKCiAgICAgICAgaW50IGNvdW50eCA9IChpbnQpVC5zaXplKCkgLSAoUiAtIGMpOwogICAgICAgIHJldHVybiBjb3VudHg7CiAgICB9OwoKICAgIC8vIGZ1bmN0aW9uIGZpbmQgdmVydGljZXMgaW4gVCB3aXRoIGNvbG9yIHguCiAgICBhdXRvIGZpbmRfdmVydGljZXNfaW5fVF93aXRoX2NvbG9yX3ggPSBbJl0odmVjdG9yPGludD4gVCwgaW50IHgpIHsKICAgICAgICB2ZWN0b3I8aW50PiByZXN1bHQ7CiAgICAgICAgdmVjdG9yPGludD4gUyA9IFQ7CiAgICAgICAgaW50IGNvdW50ID0gcXVlcnlfY291bnRfaW5fVChTLCB4KTsKICAgICAgICB3aGlsZSAoY291bnQgPiAwKSB7CiAgICAgICAgICAgIGludCBsID0gMDsgaW50IHIgPSBTLnNpemUoKSAtIDE7CiAgICAgICAgICAgIHdoaWxlIChsIDwgcikgewogICAgICAgICAgICAgICAgaW50IG1pZCA9IChsICsgcikgLyAyOwogICAgICAgICAgICAgICAgdmVjdG9yPGludD4gcGFydChTLmJlZ2luKCkgKyBsLCBTLmJlZ2luKCkgKyBtaWQgKyAxKTsKICAgICAgICAgICAgICAgIGludCBjcGFydCA9IHF1ZXJ5X2NvdW50X2luX1QocGFydCwgeCk7CiAgICAgICAgICAgICAgICBpZiAoY3BhcnQgPiAwKSB7CiAgICAgICAgICAgICAgICAgICAgciA9IG1pZDsKICAgICAgICAgICAgICAgIH0gZWxzZSB7CiAgICAgICAgICAgICAgICAgICAgbCA9IG1pZCArIDE7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0KICAgICAgICAgICAgaW50IGZvdW5kID0gU1tsXTsKICAgICAgICAgICAgcmVzdWx0LnB1c2hfYmFjayhmb3VuZCk7CgogICAgICAgICAgICAvLyByZW1vdmUgZm91bmQgZnJvbSBTLgogICAgICAgICAgICBhdXRvIGl0ID0gZmluZChTLmJlZ2luKCksIFMuZW5kKCksIGZvdW5kKTsKICAgICAgICAgICAgaWYgKGl0ICE9IFMuZW5kKCkpIFMuZXJhc2UoaXQpOwoKICAgICAgICAgICAgY291bnQgPSBxdWVyeV9jb3VudF9pbl9UKFMsIHgpOwogICAgICAgIH0KICAgICAgICByZXR1cm4gcmVzdWx0OwogICAgfTsKCiAgICBhdXRvIGZpbmRfaW5kZXBlbmRlbnRfc2V0X2hldXJpc3RpYyA9IFsmXSh2ZWN0b3I8aW50PiAmdmVydHMpIHsKICAgICAgICB2ZWN0b3I8aW50PiBiZXN0OwogICAgICAgIC8vIFdlJ2xsIHVzZSBhIHJhbmRvbSBhcHByb2FjaC4KICAgICAgICBmb3IgKGludCByZXAgPSAwOyByZXAgPCAxMDsgcmVwKyspIHsKICAgICAgICAgICAgdmVjdG9yPGludD4gYXR0ZW1wdDsKICAgICAgICAgICAgdmVjdG9yPGludD4gbG9jYWwgPSB2ZXJ0czsKICAgICAgICAgICAgcmFuZG9tX3NodWZmbGUobG9jYWwuYmVnaW4oKSwgbG9jYWwuZW5kKCkpOwogICAgICAgICAgICBmb3IgKGludCB2IDogbG9jYWwpIHsKICAgICAgICAgICAgICAgIGJvb2wgY2FuID0gdHJ1ZTsKICAgICAgICAgICAgICAgIGZvciAoaW50IHUgOiBhdHRlbXB0KSB7CiAgICAgICAgICAgICAgICAgICAgaWYgKGFkak1hdHJpeFt1XVt2XSkgewogICAgICAgICAgICAgICAgICAgICAgICBjYW4gPSBmYWxzZTsKICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWs7CiAgICAgICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaWYgKGNhbikgYXR0ZW1wdC5wdXNoX2JhY2sodik7CiAgICAgICAgICAgIH0KICAgICAgICAgICAgaWYgKGF0dGVtcHQuc2l6ZSgpID4gYmVzdC5zaXplKCkpIGJlc3QgPSBhdHRlbXB0OwogICAgICAgIH0KICAgICAgICByZXR1cm4gYmVzdDsKICAgIH07CgogICAgdmVjdG9yPGludD4gVSA9IHVuYXNzaWduZWQ7CiAgICB3aGlsZSAoIVUuZW1wdHkoKSkgewogICAgICAgIHZlY3RvcjxpbnQ+IFQgPSBmaW5kX2luZGVwZW5kZW50X3NldF9oZXVyaXN0aWMoVSk7CgogICAgICAgIHZlY3RvcjxpbnQ+IGFzc2lnbmVkSW5UOwogICAgICAgIHZlY3RvcjxpbnQ+IFMgPSBUOwogICAgICAgIHdoaWxlICghUy5lbXB0eSgpKSB7CiAgICAgICAgICAgIGJvb2wgZm91bmRYID0gZmFsc2U7CiAgICAgICAgICAgIGZvciAoaW50IHggPSAwOyB4IDwgTjsgeCsrKSB7CiAgICAgICAgICAgICAgICBpZiAoUy5lbXB0eSgpKSBicmVhazsKICAgICAgICAgICAgICAgIGludCBjb3VudHggPSBxdWVyeV9jb3VudF9pbl9UKFMsIHgpOwogICAgICAgICAgICAgICAgaWYgKGNvdW50eCA+IDApIHsKICAgICAgICAgICAgICAgICAgICB2ZWN0b3I8aW50PiBmb3VuZCA9IGZpbmRfdmVydGljZXNfaW5fVF93aXRoX2NvbG9yX3goUywgeCk7CiAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgdiA6IGZvdW5kKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGZpbmFsX2NvbG9yc1t2XSA9IHg7CiAgICAgICAgICAgICAgICAgICAgICAgIGFzc2lnbmVkSW5ULnB1c2hfYmFjayh2KTsKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgZm9yIChpbnQgdiA6IGZvdW5kKSB7CiAgICAgICAgICAgICAgICAgICAgICAgIGF1dG8gaXQgPSBmaW5kKFMuYmVnaW4oKSwgUy5lbmQoKSwgdik7CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChpdCAhPSBTLmVuZCgpKSBTLmVyYXNlKGl0KTsKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICAgICAgZm91bmRYID0gdHJ1ZTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgfQogICAgICAgICAgICBpZiAoIWZvdW5kWCkgewogICAgICAgICAgICAgICAgLy8gSWYgbm8gY29sb3IgZm91bmQgZm9yIFMsIHdlIGNhbid0IGFzc2lnbiB0aG9zZSB2ZXJ0aWNlcy4KICAgICAgICAgICAgICAgIC8vIEJ1dCBtYXliZSBpdCdzIGltcG9zc2libGUuCiAgICAgICAgICAgICAgICBicmVhazsKICAgICAgICAgICAgfQogICAgICAgIH0KCiAgICAgICAgLy8gcmVtb3ZlIFQgZnJvbSBVLgogICAgICAgIHNldDxpbnQ+IGFzc2lnbmVkU2V0KGFzc2lnbmVkSW5ULmJlZ2luKCksIGFzc2lnbmVkSW5ULmVuZCgpKTsKICAgICAgICB2ZWN0b3I8aW50PiBuZXdVOwogICAgICAgIGZvciAoaW50IHYgOiBVKSB7CiAgICAgICAgICAgIGlmIChhc3NpZ25lZFNldC5maW5kKHYpID09IGFzc2lnbmVkU2V0LmVuZCgpKSBuZXdVLnB1c2hfYmFjayh2KTsKICAgICAgICB9CiAgICAgICAgVSA9IG5ld1U7CiAgICB9CgogICAgcmV0dXJuIGZpbmFsX2NvbG9yczsKfQ==)

#include "sphinx.h"

#include <bits/stdc++.h>

using namespace std;

int perform\_experiment(vector<int> E);

vector<int> find\_colours(int N, vector<int> X, vector<int> Y) {

// We’ll store adjacency.

vector<vector<int>> adj(N);

vector<vector<bool>> adjMatrix(N, vector<bool>(N, false));

for (int i = 0; i < (int)X.size(); i++) {

int u = X\[i\];

int v = Y\[i\];

adj\[u\].push\_back(v);

adj\[v\].push\_back(u);

adjMatrix\[u\]\[v\] = adjMatrix\[v\]\[u\] = true;

}

vector<int> final\_colors(N, -1);

vector<int> unassigned; for (int i = 0; i < N; i++) unassigned.push\_back(i);

// define group testing functions.

// function query\_count\_in\_T: how many in T have color x.

auto query\_count\_in\_T = \[&\](const vector<int> &T, int x) {

vector<int> E(N, x);

vector<int> inT(N, 0);

for (int v: T) {

E\[v\] = -1;

inT\[v\] = 1;

}

int R = perform\_experiment(E);

// number c of connected components in the rest.

vector<int> complement;

for (int i = 0; i < N; i++) if (!inT\[i\]) complement.push\_back(i);

// BFS for c.

vector<int> visited(N, 0);

int ccount = 0;

for (int v: complement) {

if (!visited\[v\]) {

ccount++;

stack<int> st;

st.push(v);

visited\[v\] = 1;

while (!st.empty()) {

int node = st.top(); st.pop();

for (int nb: adj\[node\]) {

if (!inT\[nb\] &&!visited\[nb\]) {

visited\[nb\] = 1;

st.push(nb);

}

}

}

}

}

int c = ccount;

int countx = (int)T.size() - (R - c);

return countx;

};

// function find vertices in T with color x.

auto find\_vertices\_in\_T\_with\_color\_x = \[&\](vector<int> T, int x) {

vector<int> result;

vector<int> S = T;

int count = query\_count\_in\_T(S, x);

while (count > 0) {

int l = 0; int r = S.size() - 1;

while (l < r) {

int mid = (l + r) / 2;

vector<int> part(S.begin() + l, S.begin() + mid + 1);

int cpart = query\_count\_in\_T(part, x);

if (cpart > 0) {

r = mid;

} else {

l = mid + 1;

}

}

int found = S\[l\];

result.push\_back(found);

// remove found from S.

auto it = find(S.begin(), S.end(), found);

if (it!= S.end()) S.erase(it);

count = query\_count\_in\_T(S, x);

}

return result;

};

auto find\_independent\_set\_heuristic = \[&\](vector<int> &verts) {

vector<int> best;

// We’ll use a random approach.

for (int rep = 0; rep < 10; rep++) {

vector<int> attempt;

vector<int> local = verts;

random\_shuffle(local.begin(), local.end());

for (int v: local) {

bool can = true;

for (int u: attempt) {

if (adjMatrix\[u\]\[v\]) {

can = false;

break;

}

}

if (can) attempt.push\_back(v);

}

if (attempt.size() > best.size()) best = attempt;

}

return best;

};

vector<int> U = unassigned;

while (!U.empty()) {

vector<int> T = find\_independent\_set\_heuristic(U);

vector<int> assignedInT;

vector<int> S = T;

while (!S.empty()) {

bool foundX = false;

for (int x = 0; x < N; x++) {

if (S.empty()) break;

int countx = query\_count\_in\_T(S, x);

if (countx > 0) {

vector<int> found = find\_vertices\_in\_T\_with\_color\_x(S, x);

for (int v: found) {

final\_colors\[v\] = x;

assignedInT.push\_back(v);

}

for (int v: found) {

auto it = find(S.begin(), S.end(), v);

if (it!= S.end()) S.erase(it);

}

foundX = true;

}

}

if (!foundX) {

// If no color found for S, we can’t assign those vertices.

// But maybe it’s impossible.

break;

}

}

// remove T from U.

set<int> assignedSet(assignedInT.begin(), assignedInT.end());

vector<int> newU;

for (int v: U) {

if (assignedSet.find(v) == assignedSet.end()) newU.push\_back(v);

}

U = newU;

}

return final\_colors;

}

[^1]: Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[^2]: Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[^3]: DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[^4]: Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

[^5]: Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

[^6]: Leblond, Rémi and Gimeno, Felix and Altché, Florent and Saade, Alaa and Ruddock, Anton and Tallec, Corentin and Powell, George and Grill, Jean-Bastien and Mikuła, Maciej and Lochbrunner, Matthias and others. Alphacode 2 technical report. [https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2\_Tech\_Report.pdf](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf), December 2023. Accessed: 2025-01-14.

[^7]: Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

[^8]: Mike Mirzayanov. Codeforces rating system. [https://codeforces.com/blog/entry/102](https://codeforces.com/blog/entry/102), 2010.

[^9]: Mike Mirzayanov. Open codeforces rating system. [https://codeforces.com/blog/entry/20762](https://codeforces.com/blog/entry/20762), 2016.

[^10]: Mike Mirzayanov. Codeforces: Soon we will change the rating calculation for new accounts. [https://codeforces.com/blog/entry/77890](https://codeforces.com/blog/entry/77890), 2020.

[^11]: OpenAI. Introducing swe-bench verified. [https://openai.com/index/introducing-swe-bench-verified/](https://openai.com/index/introducing-swe-bench-verified/), August 2024. Accessed: 2025-01-14.

[^12]: OpenAI. Learning to reason with llms. [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/), September 2024. Accessed: 2025-01-14.

[^13]: OpenAI. Openai o3 system card. Technical Report, 2025.

[^14]: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

[^15]: Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, et al. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

[^16]: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.