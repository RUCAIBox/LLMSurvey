# LLMSurvey


> A collection of papers and resources related to Large Language Models. 
>
> The arrangement of papers refers to our survey [**"A Survey of Large Language Models"**]().
>
> If you find our survey useful for your research, please cite the following paper:

```
@article{LLMurvey,
    title={A Survey of Large Language Models},
    author={Wayne Xin Zhao},
    year={2023},
    journal={}
}
```


## Table of Contents

- [Resources of LLMs](#resources-of-llms)
  - [Open-source Models](#Open-source-Models)

- [Pre-training](#pre-training) 
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
6. **"GPT-NeoX-20B: An Open-Source Autoregressive Language Model"** . *Sid Black* . ArXiv. [[Paper](https://arxiv.org/abs/2204.06745)] [[Checkpoint](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)]
7. **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"** . *Erik Nijkamp* . ArXiv. [[Paper](https://arxiv.org/abs/2203.13474)] [[Checkpoint](https://huggingface.co/Salesforce/codegen-16B-nl)]
8. **"Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"** . *Yizhong Wang* . EMNLP. [[Paper](https://arxiv.org/abs/2204.07705)] [[Checkpoint](https://huggingface.co/allenai/tk-instruct-11b-def-pos)]
9. **"UL2: Unifying Language Learning Paradigms"** . *Yi Tay* . ArXiv. [[Paper](https://arxiv.org/abs/2205.05131)] [[Checkpoint](https://github.com/google-research/google-research/tree/master/ul2)]
10. **"OPT: Open Pre-trained Transformer Language Models"** . *Susan Zhang* . ArXiv. [[Paper](https://arxiv.org/abs/2205.01068)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)]
11. **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"** . *BigScience Workshop* . ArXiv. [[Paper](https://arxiv.org/abs/2211.05100)] [[Checkpoint](https://huggingface.co/bigscience/bloom)]
12. **"GLM-130B: An Open Bilingual Pre-trained Model"** . *Aohan Zeng* . ArXiv. [[Paper](https://arxiv.org/abs/2210.02414)] [[Checkpoint](https://github.com/THUDM/GLM-130B)]
13. **"Scaling Instruction-Finetuned Language Models"** . *Hyung Won Chung* . ArXiv. [[Paper](https://arxiv.org/abs/2210.11416)] [[Checkpoint](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)]
14. **"Crosslingual Generalization through Multitask Finetuning"** . *Niklas Muennighoff* . [[Paper](https://arxiv.org/abs/2211.01786)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf)]
15. **"Galactica: A Large Language Model for Science"** . *Ross Taylor* . arxiv. [[Paper](https://arxiv.org/abs/2211.09085)] [[Checkpoint](https://huggingface.co/facebook/galactica-120b)]
16. **"OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization"** . *Srinivasan* . arxiv. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://huggingface.co/facebook/opt-iml-30b)]
17. **"LLaMA: Open and Efficient Foundation Language Models"** . *Hugo Touvron* . arxiv. [[Paper](https://arxiv.org/abs/2302.13971v1)] [[Checkpoint](https://github.com/facebookresearch/llama)]

### Closed-source Models

1. **"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"** . *Dmitry Lepikhin* . ICLR. [[Paper](http://arxiv.org/abs/2006.16668v1)]
2. **"Language Models are Few-Shot Learners"** . *Tom B. Brown* . NeurIPS. [[Paper](https://arxiv.org/abs/2005.14165)]
3. 




## Pre-training


## Adaptation Tuning of LLMs

## Utilization

1. **"An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels"** . *Taylor Sorensen* . ACL . [[Paper](https://arxiv.org/abs/2203.11364)]
2. **"What Makes Good In-Context Examples for GPT-3?"** . *Jiachang Liu* . ACL . [[Paper](https://arxiv.org/abs/2101.06804)]
3. **"Learning to retrieve prompts for in-context learning"** .  *Ohad Rubin* . NAACL . [[Paper](https://arxiv.org/abs/2112.08633)]
4. **"Diverse demonstrations improve in-context compositional generalization"** . *Itay Levy* . arxiv . [[Paper](https://arxiv.org/abs/2212.06800)]
5. **"Automatic Chain of Thought Prompting in Large Language Models"** . *Zhuosheng Zhang* . arxiv . [[Paper](https://arxiv.org/abs/2210.03493)]
6. **"Demystifying Prompts in Language Models via Perplexity Estimation"** . *Hila Gonen* . arxiv . [[Paper](https://arxiv.org/abs/2212.04037)]
7. **"Active Example Selection for In-Context Learning"** . *Yiming Zhang* . EMNLP . [[Paper](https://arxiv.org/abs/2211.04486)]
8. **"Self-adaptive In-context Learning"** . *Zhiyong Wu* . arxiv . [[Paper](https://arxiv.org/abs/2212.10375)]
9. **"Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity"** . *Yao Lu* . ACL . [[Paper](https://arxiv.org/abs/2104.08786)]
10. **"Structured Prompting: Scaling In-Context Learning to 1,000 Examples"** . *Hao, Yaru* . arxiv . [[Paper](https://arxiv.org/abs/2212.06713)]
11. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"** . *Ye, Xi* . arxiv . [[Paper](https://arxiv.org/abs/2205.03401)]
12. **"Cross-Task Generalization via Natural Language Crowdsourcing Instructions"** . *Swaroop Mishra* .  ACL . [[Paper](https://arxiv.org/abs/2104.08773)]
13. **"Prompt-Augmented Linear Probing: Scaling Beyond the Limit of Few-shot In-Context Learner"** . *Hyunsoo Cho* . arxiv . [[Paper](https://arxiv.org/abs/2212.10873)]
14. **"Self-instruct: Aligning language model with self generated instructions"** . *Yizhong Wang* . arxiv . [[Paper](https://arxiv.org/abs/2212.10560)]
15. **"An Explanation of In-context Learning as Implicit Bayesian Inference"** . S*ang Michael Xie* . ICLR . [[Paper](https://arxiv.org/abs/2111.02080)]
16. **"Calibrate Before Use: Improving Few-Shot Performance of Language Models"** . *Zihao Zhao* . ICML . [[Paper](https://arxiv.org/abs/2102.09690)]
17. **"Data distributional properties drive emergent in-context learning in transformers"** . *Stephanie C. Y. Chan* . arxiv . [[Paper](https://arxiv.org/abs/2205.05055)]
18. **"Emergent Abilities of Large Language Models"** . *Jason Wei* . arxiv . [[Paper](https://arxiv.org/abs/2206.07682)]
19. **"In-context Learning and Induction Heads"** . *Catherine Olsson* . arxiv . [[Paper](http://arxiv.org/abs/2209.11895)]
20. **"Language Models are Few-Shot Learners"** . *Tom B. Brown* . NeurIPS . [[Paper](https://arxiv.org/abs/2005.14165)]
21. **"On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model"** . *Seongjin Shin* . NAACL . [[Paper](https://arxiv.org/abs/2204.13509)]
22. **"Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"** . *Sewon Min* . EMNLP . [[Paper](https://arxiv.org/abs/2202.12837)]
23. **"Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale"** . *Hritik Bansal* . arxiv . [[Paper](https://arxiv.org/abs/2212.09095)]
24. **"Transformers as algorithms: Generalization and implicit model selection in in-context learning"** . *Yingcong Li* . arxiv . [[Paper](https://arxiv.org/abs/2301.07067)]
25. **"Transformers learn in-context by gradient descent"** . *Johannes von Oswald* . arxiv . [[Paper](https://arxiv.org/abs/2212.07677)]
26. **"What learning algorithm is in-context learning? investigations with linear models"** . *Ekin Aky{\"{u}}rek* . arxiv . [[Paper](https://arxiv.org/abs/2211.15661)]
27. **"Chain of Thought Prompting Elicits Reasoning in Large Language Models"** . *Jason Wei* . arxiv . [[Paper](https://arxiv.org/abs/2201.11903)]
28. **"STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning"** . *Zelikman* . arxiv . [[Paper](https://arxiv.org/abs/2203.14465)]
29. **"Large language models are zero-shot reasoners"** . *Takeshi Kojima* . arxiv . [[Paper](https://arxiv.org/abs/2205.11916)]
30. **"Automatic Chain of Thought Prompting in Large Language Models"** . *Zhuosheng Zhang* . arxiv . [[Paper](http://arxiv.org/abs/2210.03493)]
31. **"Complexity-Based Prompting for Multi-Step Reasoning"** . *Yao Fu* . arxiv . [[Paper](https://arxiv.org/abs/2210.00720)]
32. **"Language Models are Multilingual Chain-of-Thought Reasoners"** . *Freda Shi* . arxiv . [[Paper](https://arxiv.org/abs/2210.03057)]
33. **"Rationale-Augmented Ensembles in Language Models"** . *Xuezhi Wang* . arxiv . [[Paper](https://arxiv.org/abs/2207.00747)]
34. **"Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"** . *Denny Zhou* . arxiv . [[Paper](https://arxiv.org/abs/2205.10625)]
35. **"Multimodal Chain-of-Thought Reasoning in Language Models"** . *Zhuosheng Zhang* . arxiv . [[Paper](https://arxiv.org/abs/2302.00923)]
36. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"** . *Xuezhi Wang* . arxiv . [[Paper](https://arxiv.org/abs/2203.11171)]
37. **"Large Language Models Can Self-Improve"** . *Jiaxin Huang* . arxiv . [[Paper](https://arxiv.org/abs/2210.11610)]
38. **"Training Verifiers to Solve Math Word Problems"** . *Karl Cobbe* . arxiv . [[Paper](https://arxiv.org/abs/2110.14168)]
39. **"On the Advance of Making Language Models Better Reasoners"** . *Yifei Li* . arxiv . [[Paper](https://arxiv.org/abs/2206.02336)]
40. **"Large Language Models are reasoners with Self-Verification"** . *Yixuan Weng* . arxiv . [[Paper](https://arxiv.org/abs/2212.09561)]
41. **"Teaching small language models to reason"** . *Lucie Charlotte Magister* . arxiv . [[Paper](https://arxiv.org/abs/2212.08410)]
42. **"Large language models are reasoning teachers"** . *Namgyu Ho* . arxiv . [[Paper](https://arxiv.org/abs/2212.10071)]
43. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"** . *Ye, Xi* . arxiv . [[Paper](https://arxiv.org/abs/2205.03401)]
44. **"Scaling Instruction-Finetuned Language Models"** . *Hyung Won Chung* . arxiv . [[Paper](https://arxiv.org/abs/2210.11416)]
45. **"Solving Quantitative Reasoning Problems with Language Models"** . *Aitor Lewkowycz* . arxiv . [[Paper](https://arxiv.org/abs/2206.14858)]
46. **"Text and patterns: For effective chain of thought, it takes two to tango"** . *Aman Madaan* . arxiv . [[Paper](https://arxiv.org/abs/2209.07686)]
47. **"Challenging BIG-Bench tasks and whether chain-of-thought can solve them"** . *Mirac Suzgun* . arxiv . [[Paper](http://arxiv.org/abs/2210.09261)]
48. **"A Survey for In-context Learning"** . *Qingxiu Dong* . arxiv . [[Paper](https://arxiv.org/abs/2301.00234)]
49. **"Reasoning with Language Model Prompting: A Survey"** . *Shuofei Qiao* . arxiv . [[Paper](https://arxiv.org/abs/2212.09597)]
50. **"Towards Reasoning in Large Language Models: A Survey"** . *Jie Huang* . arxiv . [[Paper](https://arxiv.org/abs/2212.10403)]
51. **"Reward Design with Language Models"** . *Minae Kwon* . arxiv . [[Paper](https://arxiv.org/abs/2303.00001)]
52. **"Promptagator: Few-shot Dense Retrieval From 8 Examples"** . *Zhuyun Dai* . arxiv . [[Paper](https://arxiv.org/abs/2209.11755)]
53. **"On the Feasibility of Specialized Ability Stealing for Large Language Code Models"** . *Zongjie Li* . arxiv . [[Paper](https://arxiv.org/abs/2303.03012)]
54. **"MathPrompter: Mathematical Reasoning using Large Language Models"** . *Imani, Shima* . arxiv . [[Paper](https://paperswithcode.com/paper/mathprompter-mathematical-reasoning-using)]
55. **"ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction"** . *Jiabang He* . arxiv . [[Paper](https://arxiv.org/abs/2303.05063)]
56. **"Selective Annotation Makes Language Models Better Few-Shot Learners"** . *Hongjin Su* . arxiv . [[Paper](https://arxiv.org/abs/2209.01975)]


## Capacity Evaluation

