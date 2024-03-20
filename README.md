# Meta Prompting for AI Systems 

[![arXiv](https://img.shields.io/badge/arXiv-2311.11482-<COLOR>.svg)](https://arxiv.org/abs/2311.11482) 
![Python 3.10](https://img.shields.io/badge/python-3.9-green.svg)

Homepage: https://meta-prompting.github.io.

## Introduction

Official implementation of paper "Meta Prompting for AI Systems" (https://arxiv.org/abs/2311.11482).

**Meta Prompting (General Definition)**: Meta Prompting is a prompting technique inspired by type theory, emphasizing the structure and syntax of examples rather than their detailed content. It's an approach where the focus is on presenting the outline or framework of a problem or topic, offering a scaffold that can be filled with specific details as needed. This technique is particularly useful in situations where understanding the form and pattern of a problem or solution is more crucial than the specific content.

### Characteristics of Meta Prompting:

1. **Syntax-Oriented**: The emphasis is on the form and structure of the prompt. The syntax acts as a template or a guide that outlines how a response or solution should be structured.

2. **Abstract-Example-Based**: Abstracted examples are employed to demonstrate the structure, though they are not elaborated in the content. These examples act as frameworks for inserting specific details.

3. **Type Theory Inspiration**: Drawing from type theory, this approach focuses on the types or categories of components in a prompt, such as problem statements, solution steps, or conclusions, and how they are logically organized.

4. **Adaptability**: The approach is adaptable to various domains, from mathematical problem-solving to creative writing, where the structure of the response is a key element.

5. **Guidance for Detailed Exploration**: While it does not delve into specifics, Meta Prompting provides a clear pathway for detailed exploration, guiding users on how to approach and structure their deep dive into the topic.

In essence, the general concept of Meta Prompting is about providing a skeleton or a blueprint that outlines the structure of a response or solution, focusing more on the "how" rather than the "what" of information presentation. This method is especially useful in contexts where understanding the underlying structure is key to mastering the content or solving the problem.

## Solving Math Problems

### Experiment Settings

We evaluated our models on the MATH and GSM8K datasets:

- MATH: A competition-level math word problems benchmark with 5000 test problems.
- GSM8K: The most widely used math word problem dataset with 1319 test grade school math problems.

Models were evaluated using the vLLM framework for Qwen-14B and Qwen-72B base language models with Meta Prompt. The evaluation was performed with a rule-based evaluator and SymPy to align model responses with ground-truth solutions.

### Experimental Results

Our evaluation demonstrates superior performance of the zero-shot meta-prompted Qwen-72B base language model on both MATH and GSM8K datasets, showcasing Meta Prompting's efficacy in mathematical problem-solving.

#### MATH Dataset Performance

| Model                            | FT-Dataset | Tool Usage | Eval Method  | MATH (%) |
|----------------------------------|------------|------------|--------------|----------|
| **Proprietary Models**           |            |            |              |          |
| Claude-2                         | -          | No         | CoT          | 32.5     |
| Minerva-540B                     | Arxiv+Web  | No         | CoT          | 33.6     |
| PaLM-2                           | -          | No         | CoT          | 34.3     |
| GPT-4 (2023-0314)                | -          | No         | CoT          | 42.5     |
| **Open-source Models**           |            |            |              |          |
| Qwen-14B (base)                  | -          | No         | CoT          | 24.8     |
| Qwen-14B (base)                  | -          | No         | **MetaPrompt**| **28.9**|
| Llama-2-70B (base)               | -          | No         | CoT          | 13.5     |
| Qwen-72B (base)                  | -          | No         | CoT          | 35.2     |
| Qwen-72B-MetaMathQA              | MetaMathQA | No         | CoT          | 41.7     |
| Qwen-72B (base)                  | -          | No         | **MetaPrompt**| **46.3**|

#### GSM8K Dataset Performance

| Model                          | FT-Dataset | Tool Usage | Eval Method  | GSM8K (%) |
|--------------------------------|------------|------------|--------------|-----------|
| Llama-2-70B (base)             | -          | No         | CoT          | 13.5      |
| Qwen-14B (base)                | -          | No         | CoT          | 61.3      |
| Qwen-14B (base)                | -          | No         | **MetaPrompt**| **64.8** |
| WizardMath-70B                 | WizardMath | No         | CoT          | 81.6      |
| MetaMath-70B                   | MetaMathQA | No         | CoT          | 82.3      |
| Qwen-72B (base)                | -          | No         | CoT          | 78.9      |
| Qwen-72B (base)                | -          | No         | **MetaPrompt**| **83.5** |

### Solving Game of 24 Tasks

| Method       | LLM Sessions (per sample) | Generate/Prompt tokens | Cost     | Success Rate |
|--------------|---------------------------|------------------------|----------|--------------|
| IO (best of 100) | 100                     | 1.8k / 1.0k            | $0.13    | 33%          |
| CoT (best of 100) | 100                    | 6.7k / 2.2k            | $0.47    | 49%          |
| ToT~\citep{yao2023tree} | 61.72            | 5.5k / 1.4k            | $0.74    | 74%          |
| **MP**       | **$\frac{1}{N}$**         | **$\approx \frac{1}{N}$ (8k / 1k)** | **$\approx$ $0.0003** | **100%** |


## Meta Prompting for Complex Reasoning 

**"Meta Prompting for Complex Reasoning"** (see `/.prompts/cr-agent*` ) is a specialized adaptation of the general Meta Prompting approach, specifically tailored for tackling intricate and multifaceted problems, particularly in fields requiring in-depth analytical and logical reasoning. This version emphasizes not just the structure and syntax of the problem-solving process, but also delves deeply into the content, ensuring a thorough and comprehensive approach to each problem.

### Key Elements of Meta Prompting for Complex Reasoning:

1. **Complex Problem Decomposition**: The technique begins with a complex problem or question, which is then broken down into smaller, more manageable sub-problems or questions. This decomposition is crucial for tackling complex issues in a systematic and methodical way.

2. **Detailed Preliminary Content**: Before addressing the main problem, the AI provides extensive preliminary content, including foundational concepts, relevant theories, and useful hints. This step ensures that all necessary background information is covered to understand and solve the problem.

3. **Step-by-Step Problem Solving**:
   
   - **Intermediate Questions**: The AI formulates a series of intermediate questions, each targeting a specific aspect of the complex problem. These questions guide the problem-solving process in a structured manner.
   
   - **Answer Sketches and Code Execution**: For each question, the AI develops a detailed answer sketch, which is then tested and refined through code execution. This process not only verifies the accuracy of the answer but also deepens the understanding of the problem.
   
   - **Detailed Answers**: Based on the code execution results, the AI provides comprehensive and detailed answers for each intermediate question, gradually building towards the solution of the original complex problem.

4. **Final Solution Presentation**:
   
   - **Solution Synthesis**: After addressing all intermediate questions, the AI synthesizes the findings into a complete solution for the original complex problem.
   
   - **Code for Final Solution**: The final solution is further verified or solved using coding, ensuring accuracy and precision.
   
   - **Formatted Final Answer**: The solution is presented in a clear, concise, and formally correct format, often using LaTeX for mathematical precision and enclosed within `\boxed{}` for emphasis.

#### CR Agent Assistant v0.1 based on `Meta Prompting`

See `./prompts/cr-agent-assistant-v0.1.md` for a minimalist implementation based on OpenAI Assistant API as a System Message.

- please visit [https://chat.openai.com/g/g-L3a4ZCIHx-cr-agent-v0-1](https://chat.openai.com/g/g-L3a4ZCIHx-cr-agent-v0-1) for an online demo.

See `./prompts/cr-agent-xml-assistant-v0.1.xml` for a minimalist XML-style implementation based on OpenAI Assistant API.

- please visit [https://chat.openai.com/g/g-4ir4la2Z6-cr-agent-xml-v0-1](https://chat.openai.com/g/g-4ir4la2Z6-cr-agent-xml-v0-1) for an online demo.

See `./prompts/cr-agent-xml-assistant-v0.2.xml` for a minimalist XML-style implementation based on OpenAI Assistant API.

- please visit [https://chat.openai.com/g/g-cJV031wLP-cr-agent-xml-v0-2](https://chat.openai.com/g/g-cJV031wLP-cr-agent-xml-v0-2) for an online demo.

The prompt of CR Agent XML v0.2 is autonomously generated by CR Agent v0.1 (this process can be seen as metaprogramming).


## Meta Prompting for Prompting Tasks

In the realm of advanced machine learning and AI systems, the task of automatically generating structured prompts, termed **Meta Prompting for Prompting Tasks (MP-PT)** or simply **Meta Prompting** in this specialized case (Reynolds & McDonell, 2021; Hou et al., 2022), emerges as a critical component. This process entails utilizing language models to interpret input strings as instructions and consequently generate prompts that guide further tasks. We formalize this concept within the framework of General Meta Prompting with special tasks called prompting tasks, detailing its categorical and functorial properties.

### Prompt Revision to Enhance Reasoning Capabilities

- please visit [https://chat.openai.com/g/g-o54JV8zr7-mp-pt](https://chat.openai.com/g/g-o54JV8zr7-mp-pt) for an online demo.

See `./prompts/mp-pt-reasoning-v0.1.tex` for a minimalist latex-style implementation based on OpenAI Assistant API.

## Recursive Meta Prompting

### Meta Prompting for In-Context Prompt Design

See `./prompts/mp-icpd-v0.1.md` for a minimalist implementation based on OpenAI Assistant API.

- please visit [https://chat.openai.com/g/g-9d0iBPnzR-mp-icpd](https://chat.openai.com/g/g-9d0iBPnzR-mp-icpd) for an online demo.

### Recursive Meta Prompting for In-Context Prompt Design

```
<|User|> [Input Document]: <your_system_prompt_itself>
         \textbf{Output Prompt: [to be generated using the same latex format as this prompt]}
```

The generated is shown in `./prompts/mp-icpd-v0.2.md`. 

## References

1. Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the
few-shot paradigm. In Extended Abstracts of the 2021 CHI Conference on Human Factors in
Computing Systems, pp. 1–7, 2021.

2. Yutai Hou, Hongyuan Dong, Xinghao Wang, Bohan Li, and Wanxiang Che. Metaprompting: Learning
to learn better prompts. arXiv preprint arXiv:2209.11486, 2022.

## Citations
Please cite the paper and star this repo if you use Meta Prompting (MP) and find it interesting/useful, thanks! Feel free to contact zhangyif21@tsinghua.edu.cn or open an issue if you have any questions.

```bibtex
@inproceedings{
  zhang2024meta,
  title={Meta Prompting for AI Systems},
  author={Zhang, Yifan and Yuan, Yang and Yao, Andrew Chi-Chih},
  booktitle={ICLR 2024 Workshop on Bridging the Gap Between Practice and Theory in Deep Learning},
  year={2024},
  url={https://openreview.net/forum?id=vXRKHNYg1F}
}
```
