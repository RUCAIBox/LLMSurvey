# Prompts



> We collect some useful tips for designing prompts that are collected from online notes and experiences from our authors, where we also show the related ingredients and principles (introduced in Section 8.1). 
>
> We abbreviate principles as Prin. and list the IDs of the related principles for each prompt. 1⃝: expressing the task goal clearly; 2⃝: decomposing into easy, detailed sub-tasks; 3⃝: providing few-shot demonstrations; 4⃝: utilizing model-friendly format.
>
> **Welcome everyone to provide us with more relevant tips in the form of [issues](https://github.com/RUCAIBox/LLMSurvey/issues/34)**. After selection, we will regularly update them on GitHub and indicate the source.





<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">Ingredient</th>
    <th class="tg-0lax">Collected Prompts</th>
    <th class="tg-0lax">Prin.</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="4">Task Description</td>
    <td class="tg-0lax">T1. Make your prompt <span style="font-weight:bold;text-decoration:underline">as detailed as possible</span>, <span style="font-style:italic">e.g., "Summarize the article into a short paragraph within 50 words. The major storyline and conclusion should be included, and the unimportant details can be omitted."</span></td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">T2. It is helpful to let the LLM know that it is <span style="font-weight:bold;text-decoration:underline">an expert with a prefixed prompt</span>, <span style="font-style:italic">e.g., ''You are a sophisticated expert in the domain of compute science."</span></td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">T3. Tell the model <span style="font-weight:bold;text-decoration:underline">more what it should do</span>, but not what it should not do.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">T4. To avoid the LLM to generate too long output, you can just use the prompt: <span style="font-style:italic">"Question:  Short Answer: ''</span>. <br>Besides, you can also use the following suffixes,<span style="font-style:italic"> "in a or a few words'', "in one of two sentences''</span>.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="2">Input Data</td>
    <td class="tg-0lax">I1. For the question required factual knowledge, it is useful to first <span style="font-weight:bold;text-decoration:underline">retrieve relevant documents</span> via the search engine, and then <span style="font-weight:bold;text-decoration:underline">concatenate them into the prompt</span> as reference.</td>
    <td class="tg-0lax">4⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">I2. To highlight some important parts in your prompt, please <span style="font-weight:bold;text-decoration:underline">use special marks</span>, <br><span style="font-style:italic">e.g., quotation ("") and line break (\n)</span>. You can also use both of them for emphasizing.</td>
    <td class="tg-0lax">4⃝</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">Contextual Information</td>
    <td class="tg-0lax">C1. For complex tasks, you can <span style="font-weight:bold;text-decoration:underline">clearly describe the required intermediate steps</span> to accomplish it, <br><span style="font-style:italic">e.g., "Please answer the question step by step as: Step 1 - Decompose the question into several sub-questions, …''</span></td>
    <td class="tg-0lax">2⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">C2. If you want LLMs to provide the score for a  text, it is necessary to provide a <span style="font-weight:bold;text-decoration:underline">detailed description about the scoring standard</span> with examples as reference.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">C3. When LLMs generate text according to some context (<span style="font-style:italic">e.g.</span> making recommendations according to purchase history), instructing them with <span style="font-weight:bold;text-decoration:underline">the explanation about the generated result</span> conditioned on context is helpful to improve the quality of the generated text.</td>
    <td class="tg-0lax">2⃝</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="9">Demonstration</td>
    <td class="tg-0lax">D1. <span style="font-weight:bold;text-decoration:underline">Well-formatted in-context  exemplars</span> are very useful to guide LLMs, especially for producing the outputs with complex formats.</td>
    <td class="tg-0lax">3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D2. For few-shot chain-of-thought prompting, you can also use the prompt <span style="font-style:italic">"Let's think step-by-step''</span>, and the few-shot examples should be <span style="font-weight:bold;text-decoration:underline">separated by "\n''</span> instead of full stop.</td>
    <td class="tg-0lax">1⃝3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D3. You can also <span style="font-weight:bold;text-decoration:underline">retrieve similar examples</span> in context to supply the useful task-specific knowledge for LLMs. To retrieve more relevant examples, it is useful to <span style="font-weight:bold;text-decoration:underline">first obtain the answer</span> of the question, and then concatenate it with the question for retrieval.</td>
    <td class="tg-0lax">3⃝4⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D4. The <span style="font-weight:bold;text-decoration:underline">diversity of the in-context exemplars</span> within the prompt is also useful. If it is not easy to obtain diverse questions, you can also seek to keep the <span style="font-weight:bold;text-decoration:underline">diversity of the solutions</span> for the questions.</td>
    <td class="tg-0lax">3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D5. When using chat-based LLMs, you can <span style="font-weight:bold;text-decoration:underline">decompose in-context exemplars into multi-turn messages</span>, to better match the human-chatbot conversation format. Similarly, you can also decompose the reasoning process of an exemplars into multi-turn conversation.</td>
    <td class="tg-0lax">3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D6. <span style="font-weight:bold;text-decoration:underline">Complex and informative</span> in-context exemplars can help LLMs answer complex questions.</td>
    <td class="tg-0lax">3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D7. As a symbol sequence can typically be divided into multiple segments (e.g., <math><msub><mi>i</mi><mn>1</mn></msub><mo>,</mo> <msub><mi>i</mi><mn>2</mn></msub><mo>,</mo> <msub><mi>i</mi><mn>3</mn></msub> <mo>&rarr;</mo> <msub><mi>i</mi><mn>1</mn></msub><mo>,</mo> <msub><mi>i</mi><mn>2</mn></msub></math> and <math><msub><mi>i</mi><mn>2</mn></msub><mo>,</mo><msub><mi>i</mi><mn>3</mn></msub></math>), the preceding ones can be used <span style="font-weight:bold;text-decoration:underline">as in-context exemplars</span> to guide LLMs to predict the subsequent ones,  meanwhile providing  historical information.</td>
    <td class="tg-0lax">2⃝3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D8. <span style="font-weight:bold;text-decoration:underline">Order matters</span> for in-context exemplars and prompts components. For very long input data, the position of the question (first or last) may also affect the performance.</td>
    <td class="tg-0lax">3⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">D9. If you can not obtain the in-context exemplars from existing datasets, an alternative way is to use the <span style="font-weight:bold;text-decoration:underline">zero-shot generated ones</span> from the LLM itself.</td>
    <td class="tg-0lax">3⃝</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="8">Other Designs</td>
    <td class="tg-0lax">O1. Let the <span style="font-weight:bold;text-decoration:underline">LLM check its generated results</span> before draw the conclusion, <span style="font-style:italic">e.g., "Check whether the above solution is correct or not.''</span></td>
    <td class="tg-0lax">2⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O2. If the LLM can not well solve the task, you can <span style="font-weight:bold;text-decoration:underline">seek help from external tools</span> by prompting the LLM to manipulate them. In this way, the tools should be encapsulated into callable APIs with detailed description about their functions, to better guide the LLM to utilize the tools.</td>
    <td class="tg-0lax">4⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O3. The prompt should be <span style="font-weight:bold;text-decoration:underline">self-contained</span>, and better not include the information in the context with pronouns <span style="font-style:italic">(e.g., it and they)</span>.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O4. When using LLMs for <span style="font-weight:bold;text-decoration:underline">comparing</span> two or more examples, the order affects the performance a lot.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O5. Before the prompt, <span style="font-weight:bold;text-decoration:underline">assigning a role for the LLM</span> is useful to help it better fulfill the following task instruction, <span style="font-style:italic">e.g., "I want you to act as a lawyer'</span>'.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O6. OpenAI models can perform a task better in English than other languages. Thus, it is useful to first <span style="font-weight:bold;text-decoration:underline">translate the input into English</span> and then feed it to LLMs</td>
    <td class="tg-0lax">4⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O7. For multi-choice questions, it is useful to <span style="font-weight:bold;text-decoration:underline">constrain the output space</span> of the LLM. You can use a more detailed explanation or just imposing constraints on the logits.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
  <tr>
    <td class="tg-0lax">O8. For sorting based  tasks (<span style="font-style:italic">e.g</span>., recommendation), instead of directly outputting the complete text of each item after sorting, one can <span style="font-weight:bold;text-decoration:underline">assign indicators</span> (<span style="font-style:italic">e.g.</span>, ABCD) to the unsorted items and instruct the LLMs to directly output the sorted indicators.</td>
    <td class="tg-0lax">1⃝</td>
  </tr>
</tbody>
</table>