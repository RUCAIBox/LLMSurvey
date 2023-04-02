# LLMSurvey


> A collection of papers and resources related to Large Language Models. 
>
> The arrangement of papers refers to our survey [**"A Survey of Large Language Models"**]().
>
> If you find our survey useful for your research, please cite the following paper:

```
@article{LLMSurvey,
    title={A Survey of Large Language Models},
    author={Wayne Xin Zhao},
    year={2023},
    journal={coRR}
}
```


## Table of Contents

- [Resources of LLMs](#resources-of-llms)
  - [Open-source Models](#Open-source-Models)
  - [Closed-source Models](#closed-source-models)
  
- [Pre-training](#pre-training)
  - [Data Collection](#data-collection)
  - [Architecture](#architecture)
  - [Training Algorithms](#training-algorithms)
  - [Pre-training on Code](#pre-training-on-code)

- [Adaptation Tuning of LLMs](#adaptation-tuning-of-llms)
- [Utilization](#utilization)
- [Capacity Evaluation](#capacity-evaluation)

## Resources of LLMs

### Open-source Models

1. **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"** . *Colin Raffel* . JMLR. [[Paper](https://arxiv.org/abs/1910.10683)] [[Checkpoint](https://huggingface.co/t5-base)]
2. **"mT5: A massively multilingual pre-trained text-to-text transformer"** . *Linting Xue* . NAACL. [[Paper](https://arxiv.org/abs/2010.11934)] [[Checkpoint](https://huggingface.co/google/mt5-xxl/tree/main)]
3. **"PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation"** . *Wei Zeng* . arXiv. [[Paper](https://arxiv.org/abs/2104.12369)] [[Checkpoint](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)]
4. **"CPM-2: Large-scale Cost-effective Pre-trained Language Models"** . *Zhengyan Zhang* . arXiv. [[Paper](https://arxiv.org/abs/2104.12369)] [[Checkpoint](https://github.com/TsinghuaAI/CPM)]
5. **"Multitask Prompted Training Enables Zero-Shot Task Generalization"** . *Victor Sanh* . ICLR. [[Paper](https://arxiv.org/abs/2110.08207)] [[Checkpoint](https://huggingface.co/bigscience/T0)]
6. **"GPT-NeoX-20B: An Open-Source Autoregressive Language Model"** . *Sid Black* . arXiv. [[Paper](https://arxiv.org/abs/2204.06745)] [[Checkpoint](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)]
7. **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"** . *Erik Nijkamp* . arXiv. [[Paper](https://arxiv.org/abs/2203.13474)] [[Checkpoint](https://huggingface.co/Salesforce/codegen-16B-nl)]
8. **"Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"** . *Yizhong Wang* . EMNLP. [[Paper](https://arxiv.org/abs/2204.07705)] [[Checkpoint](https://huggingface.co/allenai/tk-instruct-11b-def-pos)]
9. **"UL2: Unifying Language Learning Paradigms"** . *Yi Tay* . arXiv. [[Paper](https://arxiv.org/abs/2205.05131)] [[Checkpoint](https://github.com/google-research/google-research/tree/master/ul2)]
10. **"OPT: Open Pre-trained Transformer Language Models"** . *Susan Zhang* . arXiv. [[Paper](https://arxiv.org/abs/2205.01068)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)]
11. **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"** . *BigScience Workshop* . arXiv. [[Paper](https://arxiv.org/abs/2211.05100)] [[Checkpoint](https://huggingface.co/bigscience/bloom)]
12. **"GLM-130B: An Open Bilingual Pre-trained Model"** . *Aohan Zeng* . arXiv. [[Paper](https://arxiv.org/abs/2210.02414)] [[Checkpoint](https://github.com/THUDM/GLM-130B)]
13. **"Scaling Instruction-Finetuned Language Models"** . *Hyung Won Chung* . arXiv. [[Paper](https://arxiv.org/abs/2210.11416)] [[Checkpoint](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)]
14. **"Crosslingual Generalization through Multitask Finetuning"** . *Niklas Muennighoff* . [[Paper](https://arxiv.org/abs/2211.01786)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf)]
15. **"Galactica: A Large Language Model for Science"** . *Ross Taylor* . arXiv. [[Paper](https://arxiv.org/abs/2211.09085)] [[Checkpoint](https://huggingface.co/facebook/galactica-120b)]
16. **"OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization"** . *Srinivasan* . arXiv. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://huggingface.co/facebook/opt-iml-30b)]
17. **"LLaMA: Open and Efficient Foundation Language Models"** . *Hugo Touvron* . arXiv. [[Paper](https://arxiv.org/abs/2302.13971v1)] [[Checkpoint](https://github.com/facebookresearch/llama)]

### Closed-source Models

1. **"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"** . *Dmitry Lepikhin* . ICLR. [[Paper](http://arxiv.org/abs/2006.16668v1)]
2. **"Language Models are Few-Shot Learners"** . *Tom B. Brown* . NeurIPS. [[Paper](https://arxiv.org/abs/2005.14165)]
3. **"LaMDA: Language Models for Dialog Applications"** . *Romal Thoppilan* . CoRR. [[Paper](https://arxiv.org/abs/2201.08239)]
4. **"What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers"** . *Boseop Kim* . EMNLP. [[Paper](https://arxiv.org/abs/2109.04650)]
5. **"Evaluating Large Language Models Trained on Code"** . *Mark Chen* . arXiv. [[Paper](https://arxiv.org/abs/2107.03374)]
6. **"ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation"** . *Y*u Sun . arXiv. [[Paper](https://arxiv.org/abs/2107.02137)]
7. **"Jurassic-1: Technical details and evaluation"** . Opher Lieber. [[Paper](https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)]
8. **"Finetuned Language Models Are Zero-Shot Learners"** . *Jason Wei* . ICLR. [[Paper](https://arxiv.org/abs/2109.01652)]
9. **"Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model"** . *Shaden Smith* . arXiv. [[Paper](https://arxiv.org/abs/2201.11990)]
10. **"Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning"** . *Shaohua Wu* . arXiv. [[Paper](https://arxiv.org/abs/2110.04725)]
11. **"WebGPT: Browser-assisted question-answering with human feedback"** . *Reiichiro Nakano* . arXiv. [[Paper](https://arxiv.org/abs/2112.09332)]
12. **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"** . *Jack W. Rae* . arXiv. [[Paper](http://arxiv.org/abs/2112.11446v2)]
13. **"ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation"** . *Shuohuan Wang*. arXiv. [[Paper](http://arxiv.org/abs/2112.11446v2)]
14. **"GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"** . *Nan Du* . ICML. [[Paper](https://arxiv.org/abs/2112.06905)]
15. **"Training language models to follow instructions with human feedback"** . *Long Ouyang* . arXiv. [[Paper](http://arxiv.org/abs/2203.02155v1)]
16. **"Competition-Level Code Generation with AlphaCode"** . *Yujia Li* . arXiv. [[Paper](http://arxiv.org/abs/2203.07814v1)]
17. **"Training Compute-Optimal Large Language Models"** . *Jordan Hoffmann* . arXiv. [[Paper](https://arxiv.org/abs/2203.15556)]
18. **"PaLM: Scaling Language Modeling with Pathways"** . *Aakanksha Chowdhery* . arXiv. [[Paper](https://arxiv.org/abs/2204.02311)]
19. **"AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model"** . *Saleh Soltan* . arXiv. [[Paper](https://arxiv.org/abs/2208.01448)]
20. **"Improving alignment of dialogue agents via targeted human judgements"** . *Amelia Glaese* . arXiv. [[Paper](http://arxiv.org/abs/2209.14375v1)]
21. **"Transcending Scaling Laws with 0.1\% Extra Compute"** . *Yi Tay* . arXiv. [[Paper](https://arxiv.org/abs/2210.11399)]
22. **"Scaling Instruction-Finetuned Language Models"** . *Hyung Won Chung* . arXiv. [[Paper](https://arxiv.org/abs/2210.11416)] 
23. **"GPT-4 Technical Report"** . *OpenAI* . arXiv. [[Paper](http://arxiv.org/abs/2303.08774v2)]
24. **"PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing"** . *X*iaozhe Ren . arXiv. [[Paper](https://arxiv.org/abs/2303.10845)]


## Pre-training

### Data Collection

1. **"The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset"**. *Laurençon et al*. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]
1. **"Deduplicating Training Data Makes Language Models Better"**. *Katherine Lee et al*. ACL 2022. [[paper](https://arxiv.org/abs/2107.06499)]
1. **"Deduplicating Training Data Mitigates Privacy Risks in Language Models"**. *Nikhil Kandpal et al*. ICML 2022. [[paper](https://arxiv.org/abs/2202.06539)]
1. **"Scaling Laws and Interpretability of Learning from Repeated Data"**. *Danny Hernandez et al*. arXiv 2022. [[paper](https://arxiv.org/abs/2205.10487)]

### Architecture

#### Mainstream Architectures

**Casual Decoder**
1. **"Language Models are Few-Shot Learners"**. *Tom B. Brown et al*. NeurIPS 2020. [[paper](http://arxiv.org/abs/2005.14165)]
1. **"OPT: Open Pre-trained Transformer Language Models"**. *Susan Zhang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2205.01068)]
1. **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"**. *Teven Le Scao et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2211.05100)]
1. **"Training Compute-Optimal Large Language Models"**. *Jordan Hoffmann et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2203.15556)]
1. **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"**. *Jack W. Rae et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2112.11446)]
1. **"Galactica: A Large Language Model for Science"**. *Ross Taylor et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2211.09085)]
1. **"PaLM: Scaling Language Modeling with Pathways"**. *Aakanksha Chowdhery et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2204.02311)]
1. **"Jurassic-1: Technical Details and Evaluation"**. *Opher Lieber et al*. AI21 Labs. [[paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)]
1. **"LaMDA: Language Models for Dialog Applications"**. *Romal Thoppilan et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2201.08239)]

**Prefix Decoder**
1. **"GLM-130B: An Open Bilingual Pre-trained Model"**. *Aohan Zeng et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2210.02414)]
1. **"GLM: General Language Model Pretraining with Autoregressive Blank Infilling"**. *Zhengxiao Du et al*. ACL 2022. [[paper](http://arxiv.org/abs/2103.10360)]
1. **"Transcending Scaling Laws with 0.1% Extra Compute"**. *Yi Tay et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2210.11399)]

**MoE**
1. **"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"**. *William Fedus et al*. JMLR. [[paper](http://arxiv.org/abs/2101.03961)]
1. **"Unified Scaling Laws for Routed Language Models"**. *Aidan Clark et al*. ICML 2022. [[paper](http://arxiv.org/abs/2202.01169)]

**SSM**
1. **"Pretraining Without Attention"**. *Junxiong Wang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2212.10544)]
1. **"Efficiently Modeling Long Sequences with Structured State Spaces"**. *Albert Gu et al*. ICLR 2022. [[paper](http://arxiv.org/abs/2111.00396)]
1. **"Long Range Language Modeling via Gated State Spaces"**. *Harsh Mehta et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2206.13947)]

#### Detailed Configuration

**Layer Normalization**
1. **"DeepNet: Scaling Transformers to 1,000 Layers"**. *Hongyu Wang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2203.00555)]
1. **"Root Mean Square Layer Normalization"**. *Biao Zhang et al*. NeurIPS 2019. [[paper](http://arxiv.org/abs/1910.07467)]

**Position Encoding**
1. **"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"**. *Ofir Press et al*. ICLR 2022. [[paper](http://arxiv.org/abs/2108.12409)]
1. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"**. *Jianlin Su et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2104.09864)]

#### Analysis

1. **"What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"**. *Thomas Wang et al*. ICML 2022. [[paper](http://arxiv.org/abs/2204.05832)]
1. **"What Language Model to Train if You Have One Million GPU Hours?"**. *Teven Le Scao et al*. Findings of EMNLP 2022. [[paper](http://arxiv.org/abs/2210.15424)]
1. **"Examining Scaling and Transfer of Language Model Architectures for Machine Translation"**. *Biao Zhang et al*. ICML 2022. [[paper](http://arxiv.org/abs/2202.00528)]
1. **"Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?"**. *Yi Tay et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2207.10551)]
1. **"Do Transformer Modifications Transfer Across Implementations and Applications?"**. *Sharan Narang et al*. EMNLP 2021. [[paper](http://arxiv.org/abs/2102.11972)]

### Training Algorithms

1. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**. *Mohammad Shoeybi et al*. arXiv 2019. [[paper](http://arxiv.org/abs/1909.08053)]
1. **"An Efficient 2D Method for Training Super-Large Deep Learning Models"**. *Qifan Xu et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2104.05343)]
1. **"Tesseract: Parallelize the Tensor Parallelism Efficiently"**. *Boxiang Wang et al*. ICPP 2022. [[paper](http://arxiv.org/abs/2105.14500)]
1. **"Maximizing Parallelism in Distributed Training for Huge Neural Networks"**. *Zhengda Bian et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2105.14450)]
1. **"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"**. *Yanping Huang et al*. NeurIPS 2019. [[paper](http://arxiv.org/abs/1811.06965)]
1. **"PipeDream: Fast and Efficient Pipeline Parallel DNN Training"**. *Aaron Harlap et al*. arXiv 2018. [[paper](http://arxiv.org/abs/1806.03377)]
1. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"**. *Samyam Rajbhandari et al*. SC 2020. [[paper](http://arxiv.org/abs/1910.02054)]
1. **"ZeRO-Offload: Democratizing Billion-Scale Model Training"**. *Jie Ren et al*. USENIX 2021. [[paper](http://arxiv.org/abs/2101.06840)]

### Pre-training on Code

#### LLMs for Program Synthesis

1. **"Evaluating Large Language Models Trained on Code"**. *Mark Chen et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2107.03374)]
1. **"Program Synthesis with Large Language Models"**. *Jacob Austin et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2108.07732)]
1. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models"**. *Maxwell Nye et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2112.00114)]
1. **"A Systematic Evaluation of Large Language Models of Code"**. *Frank F. Xu et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2202.13169)]
1. **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al*. Science. [[paper](http://arxiv.org/abs/2203.07814)]
1. **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"**. *Erik Nijkamp et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2203.13474)]
1. **"InCoder: A Generative Model for Code Infilling and Synthesis"**. *Daniel Fried et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2204.05999)]
1. **"CodeT: Code Generation with Generated Tests"**. *Bei Chen et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2207.10397)]

#### NLP Tasks Formatted as Code

1. **"Language Models of Code are Few-Shot Commonsense Learners"**. *Aman Madaan et al*. EMNLP 2022. [[paper](http://arxiv.org/abs/2210.07128)]
1. **"Autoformalization with Large Language Models"**. *Yuhuai Wu et al*. NeurIPS 2022. [[paper](http://arxiv.org/abs/2205.12615)]

## Adaptation Tuning of LLMs

## Utilization

1. **"An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels"** . *Taylor Sorensen* . ACL . [[Paper](https://arxiv.org/abs/2203.11364)]
2. **"What Makes Good In-Context Examples for GPT-3?"** . *Jiachang Liu* . ACL . [[Paper](https://arxiv.org/abs/2101.06804)]
3. **"Learning to retrieve prompts for in-context learning"** .  *Ohad Rubin* . NAACL . [[Paper](https://arxiv.org/abs/2112.08633)]
4. **"Diverse demonstrations improve in-context compositional generalization"** . *Itay Levy* . arXiv . [[Paper](https://arxiv.org/abs/2212.06800)]
5. **"Automatic Chain of Thought Prompting in Large Language Models"** . *Zhuosheng Zhang* . arXiv . [[Paper](https://arxiv.org/abs/2210.03493)]
6. **"Demystifying Prompts in Language Models via Perplexity Estimation"** . *Hila Gonen* . arXiv . [[Paper](https://arxiv.org/abs/2212.04037)]
7. **"Active Example Selection for In-Context Learning"** . *Yiming Zhang* . EMNLP . [[Paper](https://arxiv.org/abs/2211.04486)]
8. **"Self-adaptive In-context Learning"** . *Zhiyong Wu* . arXiv . [[Paper](https://arxiv.org/abs/2212.10375)]
9. **"Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity"** . *Yao Lu* . ACL . [[Paper](https://arxiv.org/abs/2104.08786)]
10. **"Structured Prompting: Scaling In-Context Learning to 1,000 Examples"** . *Hao, Yaru* . arXiv . [[Paper](https://arxiv.org/abs/2212.06713)]
11. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"** . *Ye, Xi* . arXiv . [[Paper](https://arxiv.org/abs/2205.03401)]
12. **"Cross-Task Generalization via Natural Language Crowdsourcing Instructions"** . *Swaroop Mishra* .  ACL . [[Paper](https://arxiv.org/abs/2104.08773)]
13. **"Prompt-Augmented Linear Probing: Scaling Beyond the Limit of Few-shot In-Context Learner"** . *Hyunsoo Cho* . arXiv . [[Paper](https://arxiv.org/abs/2212.10873)]
14. **"Self-instruct: Aligning language model with self generated instructions"** . *Yizhong Wang* . arXiv . [[Paper](https://arxiv.org/abs/2212.10560)]
15. **"An Explanation of In-context Learning as Implicit Bayesian Inference"** . S*ang Michael Xie* . ICLR . [[Paper](https://arxiv.org/abs/2111.02080)]
16. **"Calibrate Before Use: Improving Few-Shot Performance of Language Models"** . *Zihao Zhao* . ICML . [[Paper](https://arxiv.org/abs/2102.09690)]
17. **"Data distributional properties drive emergent in-context learning in transformers"** . *Stephanie C. Y. Chan* . arXiv . [[Paper](https://arxiv.org/abs/2205.05055)]
18. **"Emergent Abilities of Large Language Models"** . *Jason Wei* . arXiv . [[Paper](https://arxiv.org/abs/2206.07682)]
19. **"In-context Learning and Induction Heads"** . *Catherine Olsson* . arXiv . [[Paper](http://arxiv.org/abs/2209.11895)]
20. **"Language Models are Few-Shot Learners"** . *Tom B. Brown* . NeurIPS . [[Paper](https://arxiv.org/abs/2005.14165)]
21. **"On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model"** . *Seongjin Shin* . NAACL . [[Paper](https://arxiv.org/abs/2204.13509)]
22. **"Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"** . *Sewon Min* . EMNLP . [[Paper](https://arxiv.org/abs/2202.12837)]
23. **"Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale"** . *Hritik Bansal* . arXiv . [[Paper](https://arxiv.org/abs/2212.09095)]
24. **"Transformers as algorithms: Generalization and implicit model selection in in-context learning"** . *Yingcong Li* . arXiv . [[Paper](https://arxiv.org/abs/2301.07067)]
25. **"Transformers learn in-context by gradient descent"** . *Johannes von Oswald* . arXiv . [[Paper](https://arxiv.org/abs/2212.07677)]
26. **"What learning algorithm is in-context learning? investigations with linear models"** . *Ekin Aky{\"{u}}rek* . arXiv . [[Paper](https://arxiv.org/abs/2211.15661)]
27. **"Chain of Thought Prompting Elicits Reasoning in Large Language Models"** . *Jason Wei* . arXiv . [[Paper](https://arxiv.org/abs/2201.11903)]
28. **"STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning"** . *Zelikman* . arXiv . [[Paper](https://arxiv.org/abs/2203.14465)]
29. **"Large language models are zero-shot reasoners"** . *Takeshi Kojima* . arXiv . [[Paper](https://arxiv.org/abs/2205.11916)]
30. **"Automatic Chain of Thought Prompting in Large Language Models"** . *Zhuosheng Zhang* . arXiv . [[Paper](http://arxiv.org/abs/2210.03493)]
31. **"Complexity-Based Prompting for Multi-Step Reasoning"** . *Yao Fu* . arXiv . [[Paper](https://arxiv.org/abs/2210.00720)]
32. **"Language Models are Multilingual Chain-of-Thought Reasoners"** . *Freda Shi* . arXiv . [[Paper](https://arxiv.org/abs/2210.03057)]
33. **"Rationale-Augmented Ensembles in Language Models"** . *Xuezhi Wang* . arXiv . [[Paper](https://arxiv.org/abs/2207.00747)]
34. **"Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"** . *Denny Zhou* . arXiv . [[Paper](https://arxiv.org/abs/2205.10625)]
35. **"Multimodal Chain-of-Thought Reasoning in Language Models"** . *Zhuosheng Zhang* . arXiv . [[Paper](https://arxiv.org/abs/2302.00923)]
36. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"** . *Xuezhi Wang* . arXiv . [[Paper](https://arxiv.org/abs/2203.11171)]
37. **"Large Language Models Can Self-Improve"** . *Jiaxin Huang* . arXiv . [[Paper](https://arxiv.org/abs/2210.11610)]
38. **"Training Verifiers to Solve Math Word Problems"** . *Karl Cobbe* . arXiv . [[Paper](https://arxiv.org/abs/2110.14168)]
39. **"On the Advance of Making Language Models Better Reasoners"** . *Yifei Li* . arXiv . [[Paper](https://arxiv.org/abs/2206.02336)]
40. **"Large Language Models are reasoners with Self-Verification"** . *Yixuan Weng* . arXiv . [[Paper](https://arxiv.org/abs/2212.09561)]
41. **"Teaching small language models to reason"** . *Lucie Charlotte Magister* . arXiv . [[Paper](https://arxiv.org/abs/2212.08410)]
42. **"Large language models are reasoning teachers"** . *Namgyu Ho* . arXiv . [[Paper](https://arxiv.org/abs/2212.10071)]
43. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"** . *Ye, Xi* . arXiv . [[Paper](https://arxiv.org/abs/2205.03401)]
44. **"Scaling Instruction-Finetuned Language Models"** . *Hyung Won Chung* . arXiv . [[Paper](https://arxiv.org/abs/2210.11416)]
45. **"Solving Quantitative Reasoning Problems with Language Models"** . *Aitor Lewkowycz* . arXiv . [[Paper](https://arxiv.org/abs/2206.14858)]
46. **"Text and patterns: For effective chain of thought, it takes two to tango"** . *Aman Madaan* . arXiv . [[Paper](https://arxiv.org/abs/2209.07686)]
47. **"Challenging BIG-Bench tasks and whether chain-of-thought can solve them"** . *Mirac Suzgun* . arXiv . [[Paper](http://arxiv.org/abs/2210.09261)]
48. **"A Survey for In-context Learning"** . *Qingxiu Dong* . arXiv . [[Paper](https://arxiv.org/abs/2301.00234)]
49. **"Reasoning with Language Model Prompting: A Survey"** . *Shuofei Qiao* . arXiv . [[Paper](https://arxiv.org/abs/2212.09597)]
50. **"Towards Reasoning in Large Language Models: A Survey"** . *Jie Huang* . arXiv . [[Paper](https://arxiv.org/abs/2212.10403)]
51. **"Reward Design with Language Models"** . *Minae Kwon* . arXiv . [[Paper](https://arxiv.org/abs/2303.00001)]
52. **"Promptagator: Few-shot Dense Retrieval From 8 Examples"** . *Zhuyun Dai* . arXiv . [[Paper](https://arxiv.org/abs/2209.11755)]
53. **"On the Feasibility of Specialized Ability Stealing for Large Language Code Models"** . *Zongjie Li* . arXiv . [[Paper](https://arxiv.org/abs/2303.03012)]
54. **"MathPrompter: Mathematical Reasoning using Large Language Models"** . *Imani, Shima* . arXiv . [[Paper](https://paperswithcode.com/paper/mathprompter-mathematical-reasoning-using)]
55. **"ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction"** . *Jiabang He* . arXiv . [[Paper](https://arxiv.org/abs/2303.05063)]
56. **"Selective Annotation Makes Language Models Better Few-Shot Learners"** . *Hongjin Su* . arXiv . [[Paper](https://arxiv.org/abs/2209.01975)]


## Capacity Evaluation

