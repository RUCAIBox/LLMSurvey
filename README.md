# LLMSurvey


> A collection of papers and resources related to Large Language Models. 
>
> The organization of papers refers to our survey [**"A Survey of Large Language Models"**](https://arxiv.org/abs/2303.18223). [![Paper page](https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2303.18223)
>
> Please let us know if you find out a mistake or have any suggestions by e-mail: batmanfly@gmail.com
>
> (we suggest ccing another email francis_kun_zhou@163.com meanwhile, in case of any unsuccessful delivery issue.)
>
>
> If you find our survey useful for your research, please cite the following paper:

```
@article{LLMSurvey,
    title={A Survey of Large Language Models},
    author={Zhao, Wayne Xin and Zhou, Kun and Li, Junyi and Tang, Tianyi and Wang, Xiaolei and Hou, Yupeng and Min, Yingqian and Zhang, Beichen and Zhang, Junjie and Dong, Zican and Du, Yifan and Yang, Chen and Chen, Yushuo and Chen, Zhipeng and Jiang, Jinhao and Ren, Ruiyang and Li, Yifan and Tang, Xinyu and Liu, Zikang and Liu, Peiyu and Nie, Jian-Yun and Wen, Ji-Rong},
    year={2023},
    journal={arXiv preprint arXiv:2303.18223},
    url={http://arxiv.org/abs/2303.18223}
}
```

## Chinese Version

To facilitate the reading of our (English-verison) survey, we also translate a [**Chinese version**](assets/LLM_Survey_Chinese.pdf) for this survey. We will continue to update the Chinese version.



![chinese_version](assets/chinese_version.png)



## 🚀(New) The trends of the number of papers related to LLMs on arXiv

Here are the trends of the cumulative numbers of arXiv papers that contain the keyphrases “language model” (since June 2018)
and “large language model” (since October 2019), respectively.

![arxiv_llms](assets/arxiv_llms.png)

The statistics are calculated using exact match by querying the keyphrases in title or abstract by months. We set different x-axis ranges for the two keyphrases, because “language models” have been explored at an earlier time. We label the points corresponding to important landmarks in the research progress of LLMs. A sharp increase occurs after the release of ChatGPT: the average number of published arXiv papers that contain “large language model” in title or abstract goes from 0.40 per day to 8.58 per day.



## 🚀(New) Technical Evolution of GPT-series Models

A brief illustration for the technical evolution of GPT-series models. We plot this figure mainly based on the papers, blog articles and official APIs from OpenAI. Here, solid lines denote that there exists an explicit evidence (e.g., the official statement that a new model is developed based on a base model) on the evolution path between two models, while dashed lines denote a relatively weaker evolution relation.



![gpt-series](assets/gpt-series.png)



## 🚀(New) Evolutionary Graph of LLaMA Family

An evolutionary graph of the research work conducted on LLaMA. Due to the huge number, we cannot include all
the LLaMA variants in this figure, even much excellent work. 



![LLaMA_family](assets/llama-0628-final.png)



To support incremental update, **we share the source file of this figure, and welcome the readers to include the desired models by submitting the pull requests on our GitHub page. If you're instrested, please request by application.**




## 🚀(New) Prompts

We collect some useful tips for designing prompts that are collected from online notes and experiences from our authors, where we also show the related ingredients and principles (introduced in Section 8.1). 

![prompt examples](assets/prompts_main.png)

Please click [here](Prompts/README.md) to view more detailed information.

**Welcome everyone to provide us with more relevant tips in the form of [issues](https://github.com/RUCAIBox/LLMSurvey/issues/34)**. After selection, we will regularly update them on GitHub and indicate the source.



## 🚀(New) Experiments

### Instruction Tuning Experiments

We will explore the effect of different types of instructions in fine-tuning LLMs (i.e., 7B LLaMA26), as well as examine the usefulness of several instruction improvement strategies.



![instruction_tuning_table](assets/instruction_tuning_table.png)



Please click [here](Experiments/README.md) to view more detailed information.

### Ability Evaluaition Experiments

We conduct a fine-grained evaluation on the abilities discussed in Section 7.1 and Section 7.2. For each kind of ability, we select representative tasks and datasets for conducting evaluation experiments to examine the corresponding performance of LLMs. 



![ability_main](assets/ability_main.png)



Please click [here](Experiments/README.md) to view more detailed information.



**We also call for support of computing power for conducting more comprehensive experiments.**



## Table of Contents

- [LLMSurvey](#llmsurvey)
  - [Chinese Version](#chinese-version)
  - [🚀(New) The trends of the number of papers related to LLMs on arXiv](#new-the-trends-of-the-number-of-papers-related-to-llms-on-arxiv)
  - [🚀(New) Technical Evolution of GPT-series Models](#new-technical-evolution-of-gpt-series-models)
  - [🚀(New) Evolutionary Graph of LLaMA Family](#new-evolutionary-graph-of-llama-family)
  - [🚀(New) Prompts](#new-prompts)
  - [🚀(New) Experiments](#new-experiments)
    - [Instruction Tuning Experiments](#instruction-tuning-experiments)
    - [Ability Evaluaition Experiments](#ability-evaluaition-experiments)
  - [Table of Contents](#table-of-contents)
  - [Timeline of LLMs](#timeline-of-llms)
  - [List of LLMs](#list-of-llms)
  - [Paper List](#paper-list)
    - [Resources of LLMs](#resources-of-llms)
      - [Publicly Available Models](#publicly-available-models)
      - [Closed-source Models](#closed-source-models)
      - [Commonly Used Corpora](#commonly-used-corpora)
      - [Library Resource](#library-resource)
      - [Deep Learning Frameworks](#deep-learning-frameworks)
    - [Pre-training](#pre-training)
      - [Data Collection](#data-collection)
      - [Architecture](#architecture)
        - [Mainstream Architectures](#mainstream-architectures)
        - [Detailed Configuration](#detailed-configuration)
        - [Analysis](#analysis)
      - [Training Algorithms](#training-algorithms)
      - [Pre-training on Code](#pre-training-on-code)
        - [LLMs for Program Synthesis](#llms-for-program-synthesis)
        - [NLP Tasks Formatted as Code](#nlp-tasks-formatted-as-code)
    - [Adaptation Tuning](#adaptation-tuning)
      - [Instruction Tuning](#instruction-tuning)
      - [Alignment Tuning](#alignment-tuning)
      - [Parameter-Efficient Model Adaptation](#parameter-efficient-model-adaptation)
      - [Memory-Efficient Model Adaptation](#memory-efficient-model-adaptation)
    - [Utilization](#utilization)
      - [In-Context Learning (ICL)](#in-context-learning-icl)
      - [Chain-of-Thought Reasoning (CoT)](#chain-of-thought-reasoning-cot)
      - [Planning for Complex Task Solving](#planning-for-complex-task-solving)
    - [Capacity Evaluation](#capacity-evaluation)
    - [The Team](#the-team)
  - [Acknowledgments](#acknowledgments)
  - [Update Log](#update-log)

## Timeline of LLMs

![LLMs_timeline](assets/LLMs-0623-final.png)





## List of LLMs

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" align="center" rowspan="2">Category</th>
    <th class="tg-baqh" align="center" rowspan="2">model</th>
    <th class="tg-0lax" align="center" rowspan="2">Release Time</th>
    <th class="tg-baqh" align="center" rowspan="2">Size(B)</th>
    <th class="tg-0lax" align="center" rowspan="2">Link</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" align="center" rowspan="27">Publicly <br>Accessbile</td>
    <td class="tg-baqh" align="center">T5</td>
    <td class="tg-0lax" align="center">2019/10</td>
    <td class="tg-baqh" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/1910.10683">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">mT5</td>
    <td class="tg-0lax" align="center">2021/03</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2010.11934">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">PanGu-α</td>
    <td class="tg-0lax" align="center">2021/05</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2104.12369">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">CPM-2</td>
    <td class="tg-0lax" align="center">2021/05</td>
    <td class="tg-baqh" align="center">198</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2106.10715">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">T0</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2110.08207">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GPT-NeoX-20B</td>
    <td class="tg-0lax" align="center">2022/02</td>
    <td class="tg-baqh" align="center">20</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2204.06745">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">CodeGen</td>
    <td class="tg-0lax" align="center">2022/03</td>
    <td class="tg-baqh" align="center">16</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2203.13474">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Tk-Instruct</td>
    <td class="tg-0lax" align="center">2022/04</td>
    <td class="tg-baqh" align="center" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2204.07705">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">UL2</td>
    <td class="tg-0lax" align="center">2022/02</td>
    <td class="tg-baqh" align="center">20</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2205.05131">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">OPT</td>
    <td class="tg-0lax" align="center">2022/05</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2205.01068">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">YaLM</td>
    <td class="tg-0lax" align="center">2022/06</td>
    <td class="tg-baqh" align="center">100</td>
    <td class="tg-0lax" align="center"><a href="https://github.com/yandex/YaLM-100B">GitHub</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">NLLB</td>
    <td class="tg-0lax" align="center">2022/07</td>
    <td class="tg-baqh" align="center">55</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2207.04672">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">BLOOM</td>
    <td class="tg-0lax" align="center">2022/07</td>
    <td class="tg-baqh" align="center">176</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.05100">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GLM</td>
    <td class="tg-0lax" align="center">2022/08</td>
    <td class="tg-baqh" align="center">130</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.02414">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Flan-T5</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center">11</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11416">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">mT0</td>
    <td class="tg-0lax" align="center">2022/11</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.01786">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Galatica</td>
    <td class="tg-0lax" align="center" align="center" align="center">2022/11</td>
    <td class="tg-baqh" align="center" align="center">120</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.09085">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">BLOOMZ</td>
    <td class="tg-0lax" align="center">2022/11</td>
    <td class="tg-baqh" align="center">176</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.01786">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">OPT-IML</td>
    <td class="tg-0lax" align="center">2022/12</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2212.12017">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Pythia</td>
    <td class="tg-0lax" align="center">2023/01</td>
    <td class="tg-baqh" align="center">12</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2304.01373">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">LLaMA</td>
    <td class="tg-0lax" align="center">2023/02</td>
    <td class="tg-baqh" align="center">65</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2302.13971v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Vicuna</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://lmsys.org/blog/2023-03-30-vicuna/">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ChatGLM</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">6</td>
    <td class="tg-0lax" align="center"><a href="https://github.com/THUDM/ChatGLM-6B">GitHub</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">CodeGeeX</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2303.17568">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Alpaca</td>
    <td class="tg-0lax" align="center">2023/03</td>
    <td class="tg-baqh" align="center">7</td>
    <td class="tg-0lax" align="center"><a href="https://crfm.stanford.edu/2023/03/13/alpaca.html">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Koala</td>
    <td class="tg-0lax" align="center">2023/04</td>
    <td class="tg-baqh" align="center">13</td>
    <td class="tg-0lax" align="center"><a href="https://bair.berkeley.edu/blog/2023/04/03/koala/">Blog</a></td>
  </tr>
    <tr>
    <td class="tg-baqh" align="center">Mistral</td>
    <td class="tg-0lax" align="center">2023/09</td>
    <td class="tg-baqh" align="center">7</td>
    <td class="tg-0lax" align="center"><a href="https://mistral.ai/news/announcing-mistral-7b/">Blog</a></td>
  </tr>
  <tr>
    <td class="tg-nrix" align="center" rowspan="31">Closed<br>Source</td>
    <td class="tg-baqh" align="center">GShard</td>
    <td class="tg-0lax" align="center">2020/01</td>
    <td class="tg-baqh" align="center" align="center">600</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2006.16668v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GPT-3</td>
    <td class="tg-0lax" align="center">2020/05</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2005.14165">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">LaMDA</td>
    <td class="tg-0lax" align="center">2021/05</td>
    <td class="tg-baqh" align="center">137</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2201.08239">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">HyperCLOVA</td>
    <td class="tg-0lax" align="center">2021/06</td>
    <td class="tg-baqh" align="center">82</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2109.04650">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Codex</td>
    <td class="tg-0lax" align="center">2021/07</td>
    <td class="tg-baqh" align="center">12</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2107.03374">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ERNIE 3.0</td>
    <td class="tg-0lax" align="center" align="center">2021/07</td>
    <td class="tg-baqh" align="center">10</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2107.02137">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Jurassic-1</td>
    <td class="tg-0lax" align="center">2021/08</td>
    <td class="tg-baqh" align="center">178</td>
    <td class="tg-0lax" align="center"><a href="https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center" align="center">FLAN</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">137</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2109.01652">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">MT-NLG</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">530</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2201.11990">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Yuan 1.0</td>
    <td class="tg-0lax" align="center">2021/10</td>
    <td class="tg-baqh" align="center">245</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2110.04725">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Anthropic</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">52</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.00861">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">WebGPT</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.09332">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Gopher</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">280</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2112.11446v2">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ERNIE 3.0 Titan</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">260</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.12731">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GLaM</td>
    <td class="tg-0lax" align="center">2021/12</td>
    <td class="tg-baqh" align="center">1200</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2112.06905">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">InstructGPT</td>
    <td class="tg-0lax" align="center">2022/01</td>
    <td class="tg-baqh" align="center">175</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2203.02155v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">AlphaCode</td>
    <td class="tg-0lax" align="center">2022/02</td>
    <td class="tg-baqh" align="center">41</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2203.07814v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Chinchilla</td>
    <td class="tg-0lax" align="center">2022/03</td>
    <td class="tg-baqh" align="center">70</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2203.15556">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">PaLM</td>
    <td class="tg-0lax" align="center">2022/04</td>
    <td class="tg-baqh" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2204.02311">Paper</a></td>
    <tr>
    <td class="tg-baqh" align="center">Cohere</td>
    <td class="tg-0lax" align="center">2022/06</td>
    <td class="tg-baqh" align="center">54</td>
    <td class="tg-0lax" align="center"><a href="https://cohere.ai/">Homepage</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">AlexaTM</td>
    <td class="tg-0lax" align="center">2022/08</td>
    <td class="tg-baqh" align="center">20</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2208.01448">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Luminous</td>
    <td class="tg-0lax" align="center">2022/09</td>
    <td class="tg-baqh" align="center">70</td>
    <td class="tg-0lax" align="center"><a href="https://docs.aleph-alpha.com/docs/introduction/luminous/">Docs</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Sparrow</td>
    <td class="tg-0lax" align="center">2022/09</td>
    <td class="tg-baqh" align="center">70</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2209.14375v1">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">WeLM</td>
    <td class="tg-0lax" align="center">2022/09</td>
    <td class="tg-baqh" align="center">10</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2209.10372">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">U-PaLM</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11399">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Flan-PaLM</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11416">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">Flan-U-PaLM</td>
    <td class="tg-0lax" align="center">2022/10</td>
    <td class="tg-baqh" align="center">540</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2210.11416">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">GPT-4</td>
    <td class="tg-0lax" align="center">2023/3</td>
    <td class="tg-baqh" align="center">-</td>
    <td class="tg-0lax" align="center"><a href="http://arxiv.org/abs/2303.08774v2">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">PanGU-Σ</td>
    <td class="tg-0lax" align="center">2023/3</td>
    <td class="tg-baqh" align="center">1085</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2303.10845">Paper</a></td>
  </tr>
</tbody>
</table>


## Paper List

### Resources of LLMs

#### Publicly Available Models

1. <u>T5</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[Paper](https://arxiv.org/abs/1910.10683)] [[Checkpoint](https://huggingface.co/t5-11b)]
2. <u>mT5</u>: **"mT5: A massively multilingual pre-trained text-to-text transformer"**. *Linting Xue* et al. NAACL 2021. [[Paper](https://arxiv.org/abs/2010.11934)] [[Checkpoint](https://huggingface.co/google/mt5-xxl/tree/main)]
3. <u>PanGu-α</u>: **"PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation"**. *Wei Zeng et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2104.12369)] [[Checkpoint](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)]
4. <u>CPM-2</u>: **"CPM-2: Large-scale Cost-effective Pre-trained Language Models"**. *Zhengyan Zhang et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2106.10715)] [[Checkpoint](https://github.com/TsinghuaAI/CPM)]
5. <u>T0</u>: **"Multitask Prompted Training Enables Zero-Shot Task Generalization"**. *Victor Sanh et al.* ICLR 2022. [[Paper](https://arxiv.org/abs/2110.08207)] [[Checkpoint](https://huggingface.co/bigscience/T0)]
6. <u>GPT-NeoX-20B</u>: **"GPT-NeoX-20B: An Open-Source Autoregressive Language Model"**. *Sid Black et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2204.06745)] [[Checkpoint](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)]
7. <u>CodeGen</u>: **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"**. *Erik Nijkamp et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2203.13474)] [[Checkpoint](https://huggingface.co/Salesforce/codegen-16B-nl)]
8. <u>Tk-Instruct</u>: **"Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"**. *Yizhong Wang et al.* EMNLP 2022. [[Paper](https://arxiv.org/abs/2204.07705)] [[Checkpoint](https://huggingface.co/allenai/tk-instruct-11b-def-pos)]
9. <u>UL2</u>: **"UL2: Unifying Language Learning Paradigms"**. *Yi Tay et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2205.05131)] [[Checkpoint](https://github.com/google-research/google-research/tree/master/ul2)]
10. <u>OPT</u>: **"OPT: Open Pre-trained Transformer Language Models"**. *Susan Zhang et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2205.01068)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)]
11. <u>NLLB</u>: **"No Language Left Behind: Scaling Human-Centered Machine Translation"**. *NLLB Team.* arXiv 2022. [[Paper](https://arxiv.org/abs/2207.04672)] [[Checkpoint](https://github.com/facebookresearch/fairseq/tree/nllb)]
12. <u>BLOOM</u>: **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"**. *BigScience Workshop*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.05100)] [[Checkpoint](https://huggingface.co/bigscience/bloom)]
13. <u>GLM</u>: **"GLM-130B: An Open Bilingual Pre-trained Model"**. *Aohan Zeng et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.02414)] [[Checkpoint](https://github.com/THUDM/GLM-130B)]
14. <u>Flan-T5</u>: **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)] [[Checkpoint](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)]
15. <u>mT0 && BLOOMZ</u>: **"Crosslingual Generalization through Multitask Finetuning"**. *Niklas Muennighoff et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01786)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf)]
16. <u>Galactica</u>: **"Galactica: A Large Language Model for Science"**. *Ross Taylor et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09085)] [[Checkpoint](https://huggingface.co/facebook/galactica-120b)]
17. <u>OPT-IML</u>: **"OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization"**. *Srinivasan et al.* . arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://huggingface.co/facebook/opt-iml-30b)]
18. <u>CodeGeeX</u>: **"CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X"**. *Qinkai Zheng et al.* . arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17568)] [[Checkpoint](https://github.com/THUDM/CodeGeeX)]
19. <u>Pythia</u>: **"Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"**. *Stella Biderman et al.* . arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01373)] [[Checkpoint](https://github.com/EleutherAI/pythia)]
20. <u>LLaMA</u>: **"LLaMA: Open and Efficient Foundation Language Models"**. *Hugo Touvron et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13971v1)] [[Checkpoint](https://github.com/facebookresearch/llama)]

#### Closed-source Models

1. <u>GShard</u>: **"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"**. *Dmitry Lepikhin et al.* ICLR 2021. [[Paper](http://arxiv.org/abs/2006.16668v1)]
2. <u>GPT-3</u>: **"Language Models are Few-Shot Learners"**. *Tom B. Brown et al.* NeurIPS 2020. [[Paper](https://arxiv.org/abs/2005.14165)]
3. <u>LaMDA</u>: **"LaMDA: Language Models for Dialog Applications"**. *Romal Thoppilan et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2201.08239)]
4. <u>HyperCLOVA</u>: **"What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers"**. *Boseop Kim et al.* EMNLP 2021. [[Paper](https://arxiv.org/abs/2109.04650)]
5. <u>CodeX</u>: **"Evaluating Large Language Models Trained on Code"**. *Mark Chen et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2107.03374)]
6. <u>ERNIE 3.0</u>: **"ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation"**. *Yu Sun et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2107.02137)]
7. <u>Jurassic-1</u>: **"Jurassic-1: Technical details and evaluation"**. *Opher Lieber et al.* 2021. [[Paper](https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)]
8. <u>FLAN</u>: **"Finetuned Language Models Are Zero-Shot Learners"**. *Jason Wei et al.* ICLR 2021. [[Paper](https://arxiv.org/abs/2109.01652)]
9. <u>MT-NLG</u>: **"Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model"**. *Shaden Smith et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2201.11990)]
10. <u>Yuan 1.0</u>: **"Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning"**. *Shaohua Wu et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2110.04725)]
11. <u>Anthropic</u>: **"A General Language Assistant as a Laboratory for Alignment"** . *Amanda Askell et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00861)]
12. <u>WebGPT</u>: **"WebGPT: Browser-assisted question-answering with human feedback"** . *Reiichiro Nakano et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2112.09332)]
13. <u>Gopher</u>: **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"**.  *Jack W. Rae et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2112.11446v2)]
14. <u>ERNIE 3.0 Titan</u>: **"ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation"**.  *Shuohuan Wang et al. *arXiv 2021. [[Paper](https://arxiv.org/abs/2112.12731)]
15. <u>GLaM</u>: **"GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"**. *Nan Du et al.* ICML 2022. [[Paper](https://arxiv.org/abs/2112.06905)]
16. <u>InstructGPT</u>: **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2203.02155v1)]
17. <u>AlphaCode</u>: **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2203.07814v1)]
18. <u>Chinchilla</u>: **"Training Compute-Optimal Large Language Models"**. *Jordan Hoffmann et al.* arXiv. [[Paper](https://arxiv.org/abs/2203.15556)]
19. <u>PaLM</u>: **"PaLM: Scaling Language Modeling with Pathways"**. *Aakanksha Chowdhery et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02311)]
20. <u>AlexaTM</u>: **"AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model"**. *Saleh Soltan et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2208.01448)]
21. <u>Sparrow</u>: **"Improving alignment of dialogue agents via targeted human judgements"**. *Amelia Glaese et al.* . arXiv 2022. [[Paper](http://arxiv.org/abs/2209.14375v1)]
22. <u>WeLM</u>: **"WeLM: A Well-Read Pre-trained Language Model for Chinese"**. *Hui Su et al.* . arXiv 2022. [[Paper](https://arxiv.org/abs/2209.10372)]
23. <u>U-PaLM</u>: **"Transcending Scaling Laws with 0.1% Extra Compute"**. *Yi Tay et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11399)]
24. <u>Flan-PaLM && Flan-U-PaLM</u>: **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al.* arXiv. [[Paper](https://arxiv.org/abs/2210.11416)] 
25. <u>GPT-4</u>: **"GPT-4 Technical Report"**. *OpenAI*. arXiv 2023. [[Paper](http://arxiv.org/abs/2303.08774v2)]
26. <u>PanGu-Σ</u>: **"PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing"**. *Xiaozhe Ren et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10845)]

#### Commonly Used Corpora

1. <u>BookCorpus</u>: **"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books"**. *Yukun Zhu et al.*  ICCV 2015. [[Paper](http://arxiv.org/abs/1506.06724v1)] [[Source](https://huggingface.co/datasets/bookcorpus)]
2. <u>Guntenburg</u>: [[Source](https://www.gutenberg.org/)]
3. <u>CommonCrawl</u>: [[Source](https://commoncrawl.org/)]
4. <u>C4</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[Paper](http://arxiv.org/abs/1910.10683v3)] [[Source](https://www.tensorflow.org/datasets/catalog/c4)]
5. <u>CC-stories-R</u>: **"A Simple Method for Commonsense Reasoning"**. *Trieu H. Trinh el al.* arXiv 2018. [[Paper](http://arxiv.org/abs/1806.02847v2)] [[Source](https://huggingface.co/datasets/spacemanidol/cc-stories)]
6. <u>CC-NEWS</u>: **"RoBERTa: A Robustly Optimized BERT Pretraining Approach"**. *Yinhan Liu et al.* arXiv 2019. [[Paper](http://arxiv.org/abs/1907.11692v1)] [[Source](https://huggingface.co/datasets/cc_news)]
7. <u>REALNEWs</u>: **"Defending Against Neural Fake News"**. *Rowan Zellers et al.* NeurIPS 2019. [[Paper](http://arxiv.org/abs/1905.12616v3)] [[Source](https://github.com/rowanz/grover/tree/master/realnews)]
8. <u>OpenWebText</u>: [[Source](https://skylion007.github.io/OpenWebTextCorpus/)]
9. <u>Pushshift.io</u>: **"The Pushshift Reddit Dataset"**. *Jason Baumgartner et al*. AAAI 2020. [[Paper](http://arxiv.org/abs/2001.08435v1)] [[Source](https://files.pushshift.io/reddit/)]
10. <u>Wikipedia</u>: [[Source](https://dumps.wikimedia.org/)]
11. <u>BigQuery</u>:  [[Source](https://cloud.google.com/bigquery/public-data?hl=zh-cn)]
12. <u>The Pile</u>: **"The Pile: An 800GB Dataset of Diverse Text for Language Modeling"**. *Leo Gao et al*. arxiv 2021. [[Paper](http://arxiv.org/abs/2101.00027v1)] [[Source](https://pile.eleuther.ai/)]
13. <u>ROOTS</u>: **"The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset"**. *Laurençon et al*. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]

#### Library Resource

1. <u>Transformers</u>: **"Transformers: State-of-the-Art Natural Language Processing"**. *Thomas Wolf et al.* EMNLP 2020. [[Paper](https://arxiv.org/abs/1910.03771)] [[Source](https://huggingface.co/)]
2. <u>DeepSpeed</u>: **"Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters"**. *Rasley et al.* KDD 2020. [[Paper](https://dl.acm.org/doi/10.1145/3394486.3406703)] [[Source](https://github.com/microsoft/DeepSpeed)]
3. <u>Megatron-LM</u>: **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**. *Mohammad Shoeybi et al.* arXiv 2019. [[Paper](https://arxiv.org/abs/1909.08053)] [[Source](https://github.com/NVIDIA/Megatron-LM)]
4. <u>JAX</u>:  [[Source](https://github.com/google/jax)]
5. <u>Colossal-AI</u>: **"Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training"**. *Zhengda Bian et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2110.14883v2)] [[Source](https://github.com/hpcaitech/ColossalAI)]
6. <u>BMTrain</u>: [[Source](https://github.com/OpenBMB/BMTrain)]
7. <u>FastMoE</u>: **"FastMoE: A Fast Mixture-of-Expert Training System"**.  *Jiaao He et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2103.13262)] [[Source](https://github.com/laekov/fastmoe)]

#### Deep Learning Frameworks

1. <u>Pytorch</u>: **"PyTorch: An Imperative Style, High-Performance Deep Learning Library"**. *Adam Paszke el al.* NeurIPS 2019. [[Paper](https://arxiv.org/abs/1912.01703)] [[Source](https://pytorch.org/)]
2. <u>TensorFlow</u>: **"TensorFlow: A system for large-scale machine learning"**. *Martín Abadi et al.* OSDI 2016. [[Paper](https://arxiv.org/abs/1605.08695)] [[Source](https://www.tensorflow.org/)] 
3. <u>MXNet</u>: **"MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems"**. *Tianqi Chen et al.* arXiv 2015. [[Paper](https://arxiv.org/abs/1512.01274)] [[Source](https://github.com/apache/mxnet)] 
4. <u>PaddlePaddle</u>: **"PaddlePaddle: An Open-Source Deep Learning Platform from Industrial Practice"** . *Yanjun Ma et al.* Frontiers of Data and Domputing 2019.  [[Paper](http://www.jfdc.cnic.cn/EN/abstract/abstract2.shtml)] [[Source](https://github.com/PaddlePaddle/Paddle)] 
5. <u>MindSpore</u>: **"Huawei MindSpore AI Development Framework"** . *Huawei Technologies Co., Ltd.* Artificial Intelligence Technology 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-981-19-2879-6_5)] [[Source](https://github.com/mindspore-ai/mindspore)] 
6. <u>OneFlow</u>: **"OneFlow: Redesign the Distributed Deep Learning Framework from Scratch"** . *Jinhui Yuan et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2110.15032)] [[Source](https://github.com/Oneflow-Inc/oneflow)] 

### Pre-training
#### Data Collection

1. **"The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset"**. *Laurençon et al*. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]
1. **"Deduplicating Training Data Makes Language Models Better"**. *Katherine Lee et al*. ACL 2022. [[paper](https://arxiv.org/abs/2107.06499)]
1. **"Deduplicating Training Data Mitigates Privacy Risks in Language Models"**. *Nikhil Kandpal et al*. ICML 2022. [[paper](https://arxiv.org/abs/2202.06539)]
1. **"Scaling Laws and Interpretability of Learning from Repeated Data"**. *Danny Hernandez et al*. arXiv 2022. [[paper](https://arxiv.org/abs/2205.10487)]
1. **"A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity"**. *Shayne Longpre et al*. arXiv 2023. [[paper](https://arxiv.org/abs/2305.13169)]

#### Architecture

##### Mainstream Architectures

**Causal Decoder**

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
1. **"Hungry Hungry Hippos: Towards Language Modeling with State Space Models"**. *Daniel Y. Fu et al*. ICLR 2023. [[paper](https://arxiv.org/abs/2212.14052)]

##### Detailed Configuration

**Layer Normalization**
1. <u>RMSNorm</u>: **"Root Mean Square Layer Normalization"**. *Biao Zhang et al*. NeurIPS 2019. [[paper](http://arxiv.org/abs/1910.07467)]
1. <u>DeepNorm</u>: **"DeepNet: Scaling Transformers to 1,000 Layers"**. *Hongyu Wang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2203.00555)]
1. <u>Sandwich-LN</u>: **"CogView: Mastering Text-to-Image Generation via Transformers"**. *Ming Ding et al*. NeirIPS 2021. [[paper](https://arxiv.org/abs/2105.13290)]

**Position Encoding**
1. <u>T5 bias</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[paper](https://arxiv.org/abs/1910.10683)]
1. <u>ALiBi</u>: **"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"**. *Ofir Press et al*. ICLR 2022. [[paper](http://arxiv.org/abs/2108.12409)]
1. <u>RoPE</u>: **"RoFormer: Enhanced Transformer with Rotary Position Embedding"**. *Jianlin Su et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2104.09864)]
1. <u>xPos</u>: **"A Length-Extrapolatable Transformer"**. *Yutao Sun et al*. arXiv 2022. [[paper](https://arxiv.org/abs/2212.10554)]

**Attention**
1. <u>Multi-query attention</u>: **"Fast Transformer Decoding: One Write-Head is All You Need"**. *Noam Shazeer*. arXiv 2019. [[paper](https://arxiv.org/abs/1911.02150)]
1. <u>FlashAttention</u>: **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"**. *Tri Dao et al*. NeurIPS 2022. [[paper](https://arxiv.org/abs/2205.14135)]
1. <u>PagedAttention</u>: **"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"**. *Woosuk Kwon et al*.  2023.  paper(Stay Tuned) [[Offical WebSite](https://vllm.ai/)]

##### Analysis

1. **"What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"**. *Thomas Wang et al*. ICML 2022. [[paper](http://arxiv.org/abs/2204.05832)]
1. **"What Language Model to Train if You Have One Million GPU Hours?"**. *Teven Le Scao et al*. Findings of EMNLP 2022. [[paper](http://arxiv.org/abs/2210.15424)]
1. **"Examining Scaling and Transfer of Language Model Architectures for Machine Translation"**. *Biao Zhang et al*. ICML 2022. [[paper](http://arxiv.org/abs/2202.00528)]
1. **"Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?"**. *Yi Tay et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2207.10551)]
1. **"Do Transformer Modifications Transfer Across Implementations and Applications?"**. *Sharan Narang et al*. EMNLP 2021. [[paper](http://arxiv.org/abs/2102.11972)]

#### Training Algorithms

1. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**. *Mohammad Shoeybi et al*. arXiv 2019. [[paper](http://arxiv.org/abs/1909.08053)]
1. **"An Efficient 2D Method for Training Super-Large Deep Learning Models"**. *Qifan Xu et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2104.05343)]
1. **"Tesseract: Parallelize the Tensor Parallelism Efficiently"**. *Boxiang Wang et al*. ICPP 2022. [[paper](http://arxiv.org/abs/2105.14500)]
1. **"Maximizing Parallelism in Distributed Training for Huge Neural Networks"**. *Zhengda Bian et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2105.14450)]
1. **"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"**. *Yanping Huang et al*. NeurIPS 2019. [[paper](http://arxiv.org/abs/1811.06965)]
1. **"PipeDream: Fast and Efficient Pipeline Parallel DNN Training"**. *Aaron Harlap et al*. arXiv 2018. [[paper](http://arxiv.org/abs/1806.03377)]
1. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"**. *Samyam Rajbhandari et al*. SC 2020. [[paper](http://arxiv.org/abs/1910.02054)]
1. **"ZeRO-Offload: Democratizing Billion-Scale Model Training"**. *Jie Ren et al*. USENIX 2021. [[paper](http://arxiv.org/abs/2101.06840)]

#### Pre-training on Code

##### LLMs for Program Synthesis

1. **"Evaluating Large Language Models Trained on Code"**. *Mark Chen et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2107.03374)]
1. **"Program Synthesis with Large Language Models"**. *Jacob Austin et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2108.07732)]
1. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models"**. *Maxwell Nye et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2112.00114)]
1. **"A Systematic Evaluation of Large Language Models of Code"**. *Frank F. Xu et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2202.13169)]
1. **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al*. Science. [[paper](http://arxiv.org/abs/2203.07814)]
1. **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"**. *Erik Nijkamp et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2203.13474)]
1. **"InCoder: A Generative Model for Code Infilling and Synthesis"**. *Daniel Fried et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2204.05999)]
1. **"CodeT: Code Generation with Generated Tests"**. *Bei Chen et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2207.10397)]
1. **"StarCoder: may the source be with you!"**. *Raymond Li et al*. arXiv 2023. [[paper](https://arxiv.org/abs/2305.06161)]

##### NLP Tasks Formatted as Code

1. **"Language Models of Code are Few-Shot Commonsense Learners"**. *Aman Madaan et al*. EMNLP 2022. [[paper](http://arxiv.org/abs/2210.07128)]
1. **"Autoformalization with Large Language Models"**. *Yuhuai Wu et al*. NeurIPS 2022. [[paper](http://arxiv.org/abs/2205.12615)]

### Adaptation Tuning

#### Instruction Tuning

1. **"Multi-Task Deep Neural Networks for Natural Language Understanding"**. *Xiaodong Liu et al*. ACL 2019. [[Paper](https://arxiv.org/abs/1901.11504)] [[Homepage](https://github.com/namisan/mt-dnn)]
1. **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al*. JMLR 2020. [[Paper](https://arxiv.org/abs/1910.10683)] [[Checkpoint](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints)]
1. **"Muppet: Massive Multi-task Representations with Pre-Finetuning"**. *Armen Aghajanyan et al*. EMNLP 2021. [[Paper](https://arxiv.org/abs/2101.11038)] [[Checkpoint](https://huggingface.co/models?other=arxiv:2101.11038)]
1. **"Cross-Task Generalization via Natural Language Crowdsourcing Instructions"**. *Swaroop Mishra et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2104.08773)] [[Collection](https://instructions.apps.allenai.org/#data)]
1. **"Finetuned Language Models Are Zero-Shot Learners"**. *Jason Wei et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2109.01652)] [[Homepage](https://github.com/google-research/FLAN)]
1. **"Multitask Prompted Training Enables Zero-Shot Task Generalization"**. *Victor Sanh et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2110.08207)] [[Checkpoint](https://huggingface.co/bigscience/T0#how-to-use)]
1. **"PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts"**. *Stephen H. Bach et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2202.01279)] [[Collection](https://github.com/bigscience-workshop/promptsource)]
1.  **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
1. **"Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"**. *Yizhong Wang et al*. EMNLP 2022. [[Paper](https://arxiv.org/abs/2204.07705)] [[Collection](https://instructions.apps.allenai.org/#data)] [[Checkpoint](https://huggingface.co/models?search=tk-instruct-)]
1. **"MVP: Multi-task Supervised Pre-training for Natural Language Generation"**. *Tianyi Tang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.12131)] [[Collection](https://huggingface.co/RUCAIBox)] [[Checkpoint](https://huggingface.co/RUCAIBox)]
1. **"Crosslingual Generalization through Multitask Finetuning"**. *Niklas Muennighoff et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01786)] [[Collection](https://github.com/bigscience-workshop/xmtf#data)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf#models)]
1. **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)] [[Homepage](https://github.com/google-research/FLAN)]
1. **"Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor"**. *Or Honovich et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09689)] [[Homepage](https://github.com/orhonovich/unnatural-instructions)]
1. **"Self-Instruct: Aligning Language Model with Self Generated Instructions"**. *Yizhong Wang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10560)] [[Homepage](https://github.com/yizhongw/self-instruct)]
1. **"OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization"**. *Srinivasan Iyer et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML)]
1. **"The Flan Collection: Designing Data and Methods for Effective Instruction Tuning"**. *Shayne Longpre et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13688)] [[Homepage](https://github.com/google-research/FLAN)]
1. **"Is Prompt All You Need No. A Comprehensive and Broader View of Instruction Learning"**. *Renze Lou et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10475)]
1. **"Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning"**. *Hao Chen et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09246)]
1. **"LIMA: Less Is More for Alignment"**. *Chunting Zhou*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11206)]


#### Alignment Tuning

1. **"TAMER: Training an Agent Manually via Evaluative Reinforcement"**. *W. Bradley Knox et al*. ICDL 2008. [[Paper](https://www.cs.utexas.edu/~bradknox/papers/icdl08-knox.pdf)]
1. **"Interactive Learning from Policy-Dependent Human Feedback"**. *James MacGlashan et al*. ICML 2017. [[Paper](https://arxiv.org/abs/1701.06049)]
1. **"Deep Reinforcement Learning from Human Preferences"**. *Paul Christiano et al*. NIPS 2017. [[Paper](https://arxiv.org/abs/1706.03741)]
1. **"Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces"**. *Garrett Warnell et al*. AAAI 2018. [[Paper](https://arxiv.org/abs/1709.10163)]
1. **"Fine-Tuning Language Models from Human Preferences"**. *Daniel M. Ziegler et al*. arXiv 2019. [[Paper](https://arxiv.org/abs/1909.08593)]
1. **"Learning to summarize from human feedback"**. *Nisan Stiennon et al*. NeurIPS 2020. [[Paper](https://arxiv.org/abs/2009.01325)]
1. **"Alignment of Language Agents"**. *Zachary Kenton et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.14659)]
1. **"Recursively Summarizing Books with Human Feedback"**. *Jeff Wu et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2109.10862)]
1. **"A General Language Assistant as a Laboratory for Alignment"**. *Amanda Askell et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00861)]
1. **"WebGPT: Browser-assisted question-answering with human feedback"**. *Reiichiro Nakano et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.09332)]
1. **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
1. **"Teaching language models to support answers with verified quotes"**. *Jacob Menick et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.11147)]
1. **"Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"**. *Yuntao Bai et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2204.05862)]
1. **"Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning"**. *Deborah Cohen et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2208.02294)]
1. **"Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned"**. *Deep Ganguli et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.07858)]
1. **"Improving alignment of dialogue agents via targeted human judgements"**. *Amelia Glaese et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14375)]
1. **"Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization"**. *Rajkumar Ramamurthy et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.01241)]
1. **"Scaling Laws for Reward Model Overoptimization"**. *Leo Gao et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.10760)]
1. **"The Wisdom of Hindsight Makes Language Models Better Instruction Followers"**. *Tianjun Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05206)]
1. **"RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment"**. *Hanze Dong et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06767)]
1. **"Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment"**. *Rishabh Bhardwaj et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09662)]

#### Parameter-Efficient Model Adaptation
1. **"Parameter-Efficient Transfer Learning for NLP"**. *Neil Houlsby et al*. ICML 2019. [[Paper](https://arxiv.org/abs/1902.00751)] [[GitHub](https://github.com/google-research/adapter-bert)]
1. **"MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer"**. *Jonas Pfeiffer et al*. EMNLP 2020. [[Paper](https://arxiv.org/abs/2005.00052)] [[GitHub](https://github.com/Adapter-Hub/adapter-transformers)]
1. **"AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts"**. *Taylor Shin et al*. EMNLP 2020. [[Paper](https://arxiv.org/abs/2010.15980)] [[GitHub](https://ucinlp.github.io/autoprompt/)]
1. **"Prefix-Tuning: Optimizing Continuous Prompts for Generation"**. *Xiang Lisa Li et al*. ACL 2021. [[Paper](https://arxiv.org/abs/2101.00190)] [[GitHub](https://github.com/XiangLi1999/PrefixTuning)]
1. **"GPT Understands, Too"**. *Xiao Liu et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.10385)] [[GitHub](https://github.com/THUDM/P-tuning)]
1. **"The Power of Scale for Parameter-Efficient Prompt Tuning"**. *Brian Lester et al*. EMNLP 2021. [[Paper](https://arxiv.org/pdf/2104.08691)]
1. **"LoRA: Low-Rank Adaptation of Large Language Models"**. *Edward J. Hu et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2106.09685)] [[GitHub](https://github.com/microsoft/LoRA)]
1. **"Towards a Unified View of Parameter-Efficient Transfer Learning"**. *Junxian He et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2110.04366)] [[GitHub](https://github.com/jxhe/unify-parameter-efficient-tuning)]
1. **"P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks"**. *Xiao Liu et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2110.07602)] [[GitHub](https://github.com/THUDM/P-tuning-v2)]
1. **"DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation"**. *Mojtaba Valipour et al*. EACL 2023. [[Paper](https://arxiv.org/abs/2210.07558)] [[GitHub](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
1. **"Parameter-efficient fine-tuning of large-scale pre-trained language models"**. *Ning Ding et al*. Nat Mach Intell. [[Paper](https://www.nature.com/articles/s42256-023-00626-4)] [[GitHub](https://github.com/thunlp/OpenDelta)]
1. **"Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"**. *Qingru Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10512)] [[GitHub](https://github.com/QingruZhang/AdaLoRA)]
1. **"LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention"**. *Renrui Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16199)] [[GitHub](https://github.com/OpenGVLab/LLaMA-Adapter)]
1. **"LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models"**. *Zhiqiang Hu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01933)] [[GitHub](https://github.com/AGI-Edgerunners/LLM-Adapters)]


#### Memory-Efficient Model Adaptation
1. **"A Survey of Quantization Methods for Efficient Neural Network Inference"**. *Amir Gholami et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.13630)]
1. **"8-bit Optimizers via Block-wise Quantization"**. *Tim Dettmers et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2110.02861)]
1. **"Compression of Generative Pre-trained Language Models via Quantization"**. *Chaofan Tao et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2203.10705)]
1. **"ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"**. *Zhewei Yao et al*. NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.01861)] [[GitHub](https://github.com/microsoft/DeepSpeed)]
1. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"**. *Tim Dettmers et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2208.07339)] [[GitHub](https://github.com/TimDettmers/bitsandbytes)]
1. **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"**. *Elias Frantar et al*. ICLR 2023. [[Paper](https://arxiv.org/abs/2210.17323)] [[GitHub](https://github.com/IST-DASLab/gptq)]
1. **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**. *Guangxuan Xiao et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10438)] [[GitHub](https://github.com/mit-han-lab/smoothquant)]
1. **"The case for 4-bit precision: k-bit Inference Scaling Laws"**. *Tim Dettmers et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09720)]
1. **"ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation"**. *Zhewei Yao et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08302)]
1. **"QLoRA: Efficient Finetuning of Quantized LLMs"**. *Tim Dettmers et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14314)] [[GitHub](https://github.com/artidoro/qlora)]
1. **"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"**. *Zechun Liu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17888)]
1. **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"**. *Ji Lin et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00978)] [[GitHub](https://github.com/mit-han-lab/llm-awq)]


### Utilization

#### In-Context Learning (ICL)

1. **"An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels"**. *Taylor Sorensen et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2203.11364)]
2. **"What Makes Good In-Context Examples for GPT-3?"**. *Jiachang Liu et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2101.06804)]
3. **"Learning to retrieve prompts for in-context learning"**. *Ohad Rubin et al*. NAACL 2022. [[Paper](https://arxiv.org/abs/2112.08633)]
4. **"Diverse demonstrations improve in-context compositional generalization"**. *Itay Levy et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06800)]
5. **"Demystifying Prompts in Language Models via Perplexity Estimation"**. *Hila Gonen et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04037)]
6. **"Active Example Selection for In-Context Learning"**. *Yiming Zhang et al*. EMNLP 2022. [[Paper](https://arxiv.org/abs/2211.04486)]
7. **"Self-adaptive In-context Learning"**. *Zhiyong Wu et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10375)]
8. **"Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity"**. *Yao Lu et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2104.08786)]
9. **"Structured Prompting: Scaling In-Context Learning to 1,000 Examples"**. *Hao, Yaru et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06713)]
10. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"**. *Ye, Xi et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.03401)]
11. **"Cross-Task Generalization via Natural Language Crowdsourcing Instructions"**. *Swaroop Mishra et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2104.08773)]
12. **"Prompt-Augmented Linear Probing: Scaling Beyond the Limit of Few-shot In-Context Learner"**. *Hyunsoo Cho et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10873)]
13. **"An Explanation of In-context Learning as Implicit Bayesian Inference"**. S*ang Michael Xie et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2111.02080)]
14. **"Calibrate Before Use: Improving Few-Shot Performance of Language Models"**. *Zihao Zhao et al*. ICML 2021. [[Paper](https://arxiv.org/abs/2102.09690)]
15. **"Data distributional properties drive emergent in-context learning in transformers"**. *Stephanie C. Y. Chan et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.05055)]
16. **"In-context Learning and Induction Heads"**. *Catherine Olsson et al*. arXiv 2022. [[Paper](http://arxiv.org/abs/2209.11895)]
17. **"On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model"**. *Seongjin Shin et al*. NAACL 2022. [[Paper](https://arxiv.org/abs/2204.13509)]
18. **"Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"**. *Sewon Min et al*. EMNLP 2022. [[Paper](https://arxiv.org/abs/2202.12837)]
19. **"Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale"**. *Hritik Bansal et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09095)]
20. **"Transformers as algorithms: Generalization and implicit model selection in in-context learning"**. *Yingcong Li et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07067)]
21. **"Transformers learn in-context by gradient descent"**. *Johannes von Oswald et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07677)]
22. **"What learning algorithm is in-context learning? investigations with linear models"**. *Ekin Aky{\"{u}}rek et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15661)]
23. **"A Survey for In-context Learning"**. *Qingxiu Dong et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.00234)]
24. **What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning**. *Jane Pan et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09731)]
25. **The Learnability of In-Context Learning**. *Noam Wies et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07895)]
26. **Do Prompt-Based Models Really Understand the Meaning of Their Prompts?** *Albert Webson et al*. NAACL 2022. [[Paper](https://aclanthology.org/2022.naacl-main.167/)]
27. **Larger language models do in-context learning differently**. *Jerry Wei*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03846)]
28. **Meta-in-context learning in large language models**. *Julian Coda-Forno*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12907)]
29. **Symbol tuning improves in-context learning in language models**. *Jerry Wei*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08298)]

#### Chain-of-Thought Reasoning (CoT)

1. **"Automatic Chain of Thought Prompting in Large Language Models"**. *Zhuosheng Zhang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.03493)]
2. **"Chain of Thought Prompting Elicits Reasoning in Large Language Models"**. *Jason Wei et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11903)]
3. **"STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning"**. *Zelikman et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.14465)]
4. **"Large language models are zero-shot reasoners"**. *Takeshi Kojima et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.11916)]
5. **"Automatic Chain of Thought Prompting in Large Language Models"**. *Zhuosheng Zhang et al*. arXiv. [[Paper](http://arxiv.org/abs/2210.03493)]
6. **"Complexity-Based Prompting for Multi-Step Reasoning"**. *Yao Fu et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00720)]
7. **"Language Models are Multilingual Chain-of-Thought Reasoners"**. *Freda Shi et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.03057)]
8. **"Rationale-Augmented Ensembles in Language Models"**. *Xuezhi Wang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2207.00747)]
9. **"Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"**. *Denny Zhou et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.10625)]
10. **"Multimodal Chain-of-Thought Reasoning in Language Models"**. *Zhuosheng Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00923)]
11. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"**. *Xuezhi Wang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.11171)]
12. **"Large Language Models Can Self-Improve"**. *Jiaxin Huang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11610)]
13. **"Training Verifiers to Solve Math Word Problems"**. *Karl Cobbe et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2110.14168)]
14. **"On the Advance of Making Language Models Better Reasoners"**. *Yifei Li et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.02336)]
15. **"Large Language Models are reasoners with Self-Verification"**. *Yixuan Weng et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09561)]
16. **"Teaching small language models to reason"**. *Lucie Charlotte Magister et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08410)]
17. **"Large language models are reasoning teachers"**. *Namgyu Ho et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10071)]
18. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"**. *Ye, Xi et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.03401)]
19. **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)]
20. **"Solving Quantitative Reasoning Problems with Language Models"**. *Aitor Lewkowycz et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.14858)]
21. **"Text and patterns: For effective chain of thought, it takes two to tango"**. *Aman Madaan et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.07686)]
22. **"Challenging BIG-Bench tasks and whether chain-of-thought can solve them"**. *Mirac Suzgun et al*. arXiv 2022. [[Paper](http://arxiv.org/abs/2210.09261)]
23. **"Reasoning with Language Model Prompting: A Survey"**. *Shuofei Qiao et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09597)]
24. **"Towards Reasoning in Large Language Models: A Survey"**. *Jie Huang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10403)]

#### Planning for Complex Task Solving

1. **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**. *Denny Zhou et al*. ICLR 2023. [[Paper](https://openreview.net/forum?id=WZH7099tgfM)]
2. **PAL: Program-aided Language Models**. *Luyu Gao et al*. ICML 2023. [[Paper](https://openreview.net/forum?id=M1fd9Z00sj)]
3. **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**. *Lei Wang et al*. ACL 2023. [[Paper](https://arxiv.org/abs/2305.04091)]
4. **ProgPrompt: Generating Situated Robot Task Plans using Large Language Models**. *Ishika Singh et al*. ICRA 2022. [[Paper](https://arxiv.org/abs/2209.11302)]
5. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**. *Shunyu Yao et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10601)]
6. **Voyager: An Open-Ended Embodied Agent with Large Language Models**. *Guanzhi Wang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16291)]
7. **Reflexion: Language Agents with Verbal Reinforcement Learning**. *Noah Shinn et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11366)]
8. **Multimodal Procedural Planning via Dual Text-Image Prompting**. *Yujie Lu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01795)]
9. **Self-planning Code Generation with Large Language Model**. *Xue Jiang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06689)]
10. **Decomposed Prompting: A Modular Approach for Solving Complex Tasks**. *Tushar Khot et al*. ICLR 2023 [[Paper](https://openreview.net/forum?id=_nGgzQjzaRy)]
11. **Toolformer: Language Models Can Teach Themselves to Use Tools**. *Timo Schick et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04761)]
12. **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**. *Yongliang Shen et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17580)]
13. **Faithful Chain-of-Thought Reasoning**. *Qing Lyu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13379)]
14. **LLM+P: Empowering Large Language Models with Optimal Planning Proficiency**. *Bo Liu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11477)]
15. **Reasoning with Language Model is Planning with World Model**. *Shibo Hao et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14992)]
16. **Generative Agents: Interactive Simulacra of Human Behavior**. *Joon Sung Park et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03442)]
17. **ReAct: Synergizing Reasoning and Acting in Language Models**. *Shunyu Yao et al*. ICLR 2023. [[Paper](https://openreview.net/forum?id=WE_vluYUL-X)]
18. **ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models**. *Zhipeng Chen et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14323)]
19. **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents**. *Zihao Wang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01560)]
20. **AdaPlanner: Adaptive Planning from Feedback with Language Models**. *Haotian Sun et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16653)]


### Capacity Evaluation

1. **"Measuring Massive Multitask Language Understanding"**. *Dan Hendrycks et al.* ICLR 2021. [[Paper](http://arxiv.org/abs/2009.03300v3)]
2. **"Persistent Anti-Muslim Bias in Large Language Models"**. *Abubakar Abid et al.* AIES 2021. [[Paper](http://arxiv.org/abs/2101.05783v2)]
3. **"Understanding the Capabilities, Limitations, and Societal Impact of Large Language Models"**. *Alex Tamkin et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2102.02503v1)]
4. **"BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments"**. *Sanjana Srivastava et al.* CoRL 2021. [[Paper](http://arxiv.org/abs/2108.03332v1)]
5. **"Program Synthesis with Large Language Models"**. *Jacob Austin et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2108.07732v1)]
6. **"Training Verifiers to Solve Math Word Problems"**. *Karl Cobbe et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2110.14168v2)]
7. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models"**. *Maxwell I. Nye et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2112.00114v1)]
8. **"Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents"**. *Wenlong Huang et al.* ICML 2022. [[Paper](http://arxiv.org/abs/2201.07207v2)]
9. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"**. *Jason Wei et al.* NeurIPS 2022. [[Paper](http://arxiv.org/abs/2201.11903v6)]
10. **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2203.02155v1)]
11. **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al.* Science 2022. [[Paper](http://arxiv.org/abs/2203.07814v1)]
12. **"Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"**. *Michael Ahn et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2204.01691v2)]
13. **"Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"**. *Yuntao Bai et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2204.05862v1)]
14. **"Autoformalization with Large Language Models"**. *Yuhuai Wu et al.* NeurIPS 2022. [[Paper](http://arxiv.org/abs/2205.12615v1)]
15. **"Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models"**. *Aarohi Srivastava et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2206.04615)]
16. **"Exploring Length Generalization in Large Language Models"**. *Cem Anil et al.* NeurIPS 2022. [[Paper](http://arxiv.org/abs/2207.04901v2)]
17. **"Few-shot Learning with Retrieval Augmented Language Models"**. *Gautier Izacard et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2208.03299)]
18. **"Limitations of Language Models in Arithmetic and Symbolic Induction"**. *Jing Qian et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2208.05051v1)]
19. **"Code as Policies: Language Model Programs for Embodied Control"**. *Jacky Liang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2209.07753v3)]
20. **"ProgPrompt: Generating Situated Robot Task Plans using Large Language Models"**. *Ishika Singh et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2209.11302v1)]
21. **"Law Informs Code: A Legal Informatics Approach to Aligning Artificial Intelligence with Humans"**. *John J. Nay et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2209.13020v13)]
22. **"Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought"**. *Abulhair Saparov et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2210.01240v4)]
23. **"Language Models are Multilingual Chain-of-Thought Reasoners"**. *Freda Shi et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2210.03057v1)]
24. **"Re3: Generating Longer Stories With Recursive Reprompting and Revision"**. *Kevin Yang et al.* EMNLP 2022. [[Paper](http://arxiv.org/abs/2210.06774v3)]
25. **"Language Models of Code are Few-Shot Commonsense Learners"**. *Aman Madaan et al.* EMNLP 2022. [[Paper](http://arxiv.org/abs/2210.07128v3)]
26. **"Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them"**. *Mirac Suzgun et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2210.09261v1)]
27. **"Large Language Models Can Self-Improve"**. *Jiaxin Huang et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11610)]
28. **"Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs"**. *Albert Q. Jiang et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2210.12283v3)]
29. **"Holistic Evaluation of Language Models"**. *Percy Liang et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09110)]
30. **"PAL: Program-aided Language Models"**. *Luyu Gao et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10435)]
31. **"Legal Prompt Engineering for Multilingual Legal Judgement Prediction"**. *Dietrich Trautmann et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.02199v1)]
32. **"How Does ChatGPT Perform on the Medical Licensing Exams? The Implications of Large Language Models for Medical Education and Knowledge Assessment"**. *Aidan Gilson et al.* medRxiv 2022. [[Paper](https://www.medrxiv.org/content/10.1101/2022.12.23.22283901v1)]
33. **"ChatGPT: The End of Online Exam Integrity?"**. *Teo Susnjak et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.09292v1)]
34. **"Large Language Models are reasoners with Self-Verification"**. *Yixuan Weng et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09561)]
35. **"Self-Instruct: Aligning Language Model with Self Generated Instructions"**. *Yizhong Wang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.10560v1)]
36. **"ChatGPT Makes Medicine Easy to Swallow: An Exploratory Case Study on Simplified Radiology Reports"**. *Katharina Jeblick et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.14882v1)]
37. **"The End of Programming"**. *Matt Welsh et al.* ACM 2023. [[Paper](https://cacm.acm.org/magazines/2023/1/267976-the-end-of-programming/fulltext)]
38. **"Chatgpt goes to law school"**. *Choi Jonathan H et al.* SSRN 2023. [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4335905)]
39. **"How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection"**. *Biyang Guo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07597v1)]
40. **"Is ChatGPT A Good Translator? A Preliminary Study"**. *Wenxiang Jiao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08745v3)]
41. **"Could an Artificial-Intelligence agent pass an introductory physics course?"**. *Gerd Kortemeyer et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12127v2)]
42. **"Mathematical Capabilities of ChatGPT"**. *Simon Frieder et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13867v1)]
43. **"Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models"**. *Zhihong Shao et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.00618v1)]
44. **"Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning"**. *Thomas Carta et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02662v1)]
45. **"Evaluating ChatGPT as an Adjunct for Radiologic Decision-Making"**. *Arya Yao et al.* medRxiv 2023. [[Paper](https://www.medrxiv.org/content/10.1101/2023.02.02.23285399v1)]
46. **"Theory of Mind May Have Spontaneously Emerged in Large Language Models"**. *Michal Kosinski et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.02083v3)]
47. **"A Categorical Archive of ChatGPT Failures"**. *Ali Borji et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03494v7)]
48. **"A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity"**. *Yejin Bang et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.04023v2)]
49. **"Toolformer: Language Models Can Teach Themselves to Use Tools"**. *Timo Schick et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.04761v1)]
50. **"Is ChatGPT a General-Purpose Natural Language Processing Task Solver?"**. *Chengwei Qin et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.06476v2)]
51. **"How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation"**. *Hendy Amr et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.09210)]
52. **"Can ChatGPT Understand Too? A Comparative Study on ChatGPT and Fine-tuned BERT"**. *Qihuang Zhong et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10198v2)]
53. **"Zero-Shot Information Extraction via Chatting with ChatGPT"**. *Xiang Wei et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10205v1)]
54. **"ChatGPT: Jack of all trades, master of none"**. *Jan Kocon et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.10724v1)]
55. **"On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective"**. *Jindong Wang et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12095v4)]
56. **"Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback"**. *Baolin Peng et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.12813v3)]
57. **"An Independent Evaluation of ChatGPT on Mathematical Word Problems (MWP)"**. *Paulo Shakarian et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13814v2)]
58. **"How Robust is GPT-3.5 to Predecessors? A Comprehensive Study on Language Understanding Tasks"**. *Chen Xuanting et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.00293v1)]
59. **"The utility of ChatGPT for cancer treatment information"**. *Shen Chen et al.* medRxiv 2023. [[Paper](https://www.medrxiv.org/content/10.1101/2023.03.16.23287316v1)]
60. **"Can ChatGPT Assess Human Personalities? A General Evaluation Framework"**. *Haocong Rao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01248v2)]
61. **"Will Affective Computing Emerge from Foundation Models and General AI? A First Evaluation on ChatGPT."**. *Mostafa M. Amin et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03186v1)]
62. **"Exploring the Feasibility of ChatGPT for Event Extraction."**. *Jun Gao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03836v2)]
63. **"Does Synthetic Data Generation of LLMs Help Clinical Text Mining?"**. *Tang Ruixiang et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.04360v1)]
64. **"Consistency Analysis of ChatGPT"**. *Myeongjun Jang et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.06273v1)]
65. **"Self-planning Code Generation with Large Language Model"**. *Shun Zhang et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2303.06689v1)]
66. **"Evaluation of ChatGPT as a Question Answering System for Answering Complex Questions"**. *Yiming Tan et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07992)]
67. **"GPT-4 Technical Report"**. *OpenAI et al.* OpenAI 2023. [[Paper](http://arxiv.org/abs/2303.08774v3)]
68. **"A Short Survey of Viewing Large Language Models in Legal Aspect"**. *Zhongxiang Sun et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.09136v1)]
69. **"ChatGPT Participates in a Computer Science Exam"**. *Sebastian Bordt et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09461v2)]
70. **"A Comprehensive Capability Analysis of GPT-3 and GPT-3.5 Series Models"**. *Junjie Ye et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10420v1)]
71. **"On the Educational Impact of ChatGPT: Is Artificial Intelligence Ready to Obtain a University Degree?"**. *Kamil Malinka et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.11146v1)]
72. **"Sparks of Artificial General Intelligence: Early experiments with GPT-4"**. *S'ebastien Bubeck et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.12712v3)]
73. **"Is ChatGPT A Good Keyphrase Generator? A Preliminary Study"**. *Mingyang Song et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13001v1)]
74. **"Capabilities of GPT-4 on Medical Challenge Problems"**. *Harsha Nori et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13375v1)]
75. **"Can we trust the evaluation on ChatGPT?"**. *Rachith Aiyappa et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12767)]
76. **"ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks"**. *Fabrizio Gilardi et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.15056v1)]
77. **"Evaluation of ChatGPT for NLP-based Mental Health Applications"**. *Bishal Lamichhane et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15727v1)]
78. **"ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of Commonsense Problem in Large Language Models"**. *Bian Ning et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.16421v1)]
79. **"Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams"**. *Desnes Nunes et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17003v1)]
80. **"Humans in Humans Out: On GPT Converging Toward Common Sense in both Success and Failure"**. *Philipp Koralus et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17276v1)]
81. **"Yes but.. Can ChatGPT Identify Entities in Historical Documents?"**. *Carlos-Emiliano González-Gallardo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17322v1)]
82. **"Uncovering ChatGPT's Capabilities in Recommender Systems"**. *Sunhao Dai et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2305.02182)]
83. **"Editing Large Language Models: Problems, Methods, and Opportunities"**. *Yunzhi Yao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13172)]
84. **"Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity"**. *Terry Yue Zhuo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12867)]
85. **"On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex"**. *Terry Yue Zhuo et al.* EACL 2023. [[Paper](https://arxiv.org/abs/2301.12868)]
86. **"A Systematic Study and Comprehensive Evaluation of ChatGPT on Benchmark Datasets"**. Laskar et al.* ACL'23. [[Paper]](https://arxiv.org/abs/2305.18486)
87. **"Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment"**. *Rishabh Bhardwaj et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09662)]

### The Team

Here is the list of our student contributors in each section.

| Section                       | Student Contributors                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| The whole paper               | Kun Zhou, Junyi Li                                                   |
| Overview && Resources of LLMs | Yingqian Min (Lead), Chen Yang                                       |
| Pretraining                   | Yupeng Hou (Lead), Junjie Zhang, Zican Dong, Yushuo Chen             |
| Adaptaion Tuning              | Tianyi Tang (Lead), Jinhao Jiang, Ruiyang Ren, Zikang Liu, Peiyu Liu |
| Utilization                   | Xiaolei Wang (Lead), Yifan Du, Xinyu Tang                            |
| Capacity Evaluation           | Beichen Zhang (Lead), Zhipeng Chen, Yifan Li                         |

## Acknowledgments

The authors would like to thank Yankai Lin and Yutao Zhu for proofreading  this paper. Since the first release of this paper, we have received a number of valuable comments from the readers. We sincerely thank the readers who have written to us with constructive suggestions and comments: Tyler Suard, Damai Dai, Liang Ding,  Stella Biderman,  Kevin Gray,  Jay Alammar and Yubo Feng.

## Update Log

| Version                  | Time       | Update Content                                               |
| ------------------------ | ---------- | ------------------------------------------------------------ |
| V1                       | 2023/03/31 | The initial version.                                         |
| V2                       | 2023/04/09 | Add the affiliation information.<br/>Revise Figure 1 and Table 1 and clarify the <br/>corresponding selection criterion for LLMs.<br/>Improve the writing.<br/>Correct some minor errors. |
| V3                       | 2023/04/11 | Correct the errors for library resources.                    |
| V4                       | 2023/04/12 | Revise Figure 1 and Table 1 and clarify the release date of LLMs. |
| V5                       | 2023/04/16 | Add a new Section 2.2 about<br/>the technical evolution of GPT-series models. |
| V6                       | 2023/04/24 | Add some new models in Table 1 and Figure 1.<br/>Add the discussion about scaling laws.<br/>Add some explanations about the<br/>model sizes for emergent abilities (Section 2.1).<br/>Add an illustrative figure for the attention patterns <br/>for different architectures in Figure 4.<br/>Add the detailed formulas in Table 4. |
| V7                       | 2023/04/25 | Revise some copy errors in figures and tables.               |
| V8                       | 2023/04/27 | Add efficient tuning in Section 5.3                          |
| V9                       | 2023/04/28 | Revise  Section 5.3                                          |
| V10                      | 2023/05/07 | Revise Table 1, Table 2, and some minor points.              |
| V11 <br>(major revision) | 2023/06/29 | – Section 1: add Figure 1 for the trends of published<br/>LLM papers in arXiv;<br/>– Section 2: add Figure 3 for GPT’s evolution and the<br/>corresponding discussion;<br/>– Section 3: add Figure 4 for LLaMA family and the<br/>corresponding discussion;<br/>– Section 5: add latest discussion about the synthetic<br/>data formatting of instruction tuning in Section 5.1.1,<br/>the empirical analysis for instruction tuning in Sec-<br/>tion 5.1.4, parameter-efficient model adaptation in<br/>Section 5.3 and memory-efficient adaptation in Sec-<br/>tion 5.4;<br/>– Section 6: add latest discussion about the underlying<br/>mechanism of ICL 6.1.3, planning for complex task<br/>solving in Section 6.3;<br/>– Section 7: add Table 10 for representative datasets for<br/>evaluating advanced abilities of LLMs, and empirical<br/>ability evaluation in Section 7.3.2;<br/>– Section 8: add prompt design;<br/>– Section 9: add the discussions on applications of<br/>LLMs in finance and scientific research domains; |
