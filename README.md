# Meta Prompting for AI Systems

[![arXiv](https://img.shields.io/badge/arXiv-2311.11482-b31b1b.svg)](https://arxiv.org/abs/2311.11482)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

This is the official repository for the paper **"Meta Prompting for AI Systems"**.

> **Abstract:** We introduce Meta Prompting (MP), a framework that elevates the reasoning capabilities of large language models (LLMs) by focusing on the formal structure of a task rather than content-specific examples. We establish a theoretical foundation for this paradigm, formalizing MP as a functor that maps a category of tasks to a category of structured prompts, thereby guaranteeing that compositional problem-solving strategies can be systematically decomposed into modular prompt structures. We extend this concept to Recursive Meta Prompting (RMP), an automated process where an LLM can generate and refine its own prompts. We model this self-improvement loop formally as a monad, providing a principled framework for automated prompt engineering. Our claims are validated through extensive experiments demonstrating that a Qwen-72B base model, guided by a single, example-agnostic meta-prompt, achieves state-of-the-art results on MATH, GSM8K, and Game of 24. These results are achieved with substantial token efficiency gains over traditional few-shot methods.

---

### Table of Contents
- [What is Meta Prompting?](#what-is-meta-prompting)
- [Key Contributions](#key-contributions)
- [A Formal Framework](#a-formal-framework)
- [🚀 State-of-the-Art Results](#-state-of-the-art-results)
  - [MATH Benchmark](#math-benchmark)
  - [GSM8K Benchmark](#gsm8k-benchmark)
  - [Game of 24](#game-of-24)
- [Getting Started: Try Meta Prompting](#getting-started-try-meta-prompting)
- [Citation](#citation)

---

## What is Meta Prompting?

**Meta Prompting** is a technique that provides a high-level, structural template for *how to think* rather than specific examples of *what to think*. Unlike few-shot prompting, which relies on content-rich examples, a meta-prompt is a single, example-agnostic scaffold that guides the LLM's reasoning process. It focuses on the formal procedure, syntax, and compositionality of problem-solving.

This approach teaches the model a reusable, structured method for tackling an entire category of tasks, leading to greater efficiency and robustness.

### Visual Comparison

Here’s the difference in action. A **Meta Prompt** provides a general structure for solving *any* problem in a category, while a **Few-Shot Prompt** provides specific, solved examples.

#### Meta Prompt: A Structural Template
This prompt gives the LLM a reusable blueprint for solving math problems. It's completely agnostic to the actual numbers or problem details.

```json
Integrate step-by-step reasoning to solve mathematical problems under the following structure:

{
  "Problem": "[question to be answered]",
  "Solution": {
    "Step 1": "Begin the response with 'Let's think step by step.'",
    "Step 2": "Follow with the reasoning steps, ensuring the solution process is broken down clearly and logically.",
    "Step 3": "End the solution with the final answer encapsulated in a LaTeX-formatted box, $\boxed{...}$"
  },
  "Final Answer": "[final answer]"
}
````

#### Few-Shot Prompt: Concrete Examples

For comparison, a traditional few-shot prompt provides several concrete `(problem, solution)` pairs. The model is expected to learn the pattern from these specific examples.

```
Problem: Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.

Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5-x>0$, so $x<5$. Therefore, the domain is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$.

---------

Problem: If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution: We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$.

---------
...
```

-----

## Key Contributions

  - **Introduces Meta Prompting (MP):** A new, example-agnostic prompting paradigm that focuses on the formal structure of reasoning.
  - **Establishes a Theoretical Foundation:** We formalize MP using category theory, modeling it as a **functor** that guarantees compositional problem-solving.
  - **Defines Recursive Meta Prompting (RMP):** We introduce a process for automated prompt engineering and self-improvement, which we formalize as a **monad**.
  - **Achieves State-of-the-Art Performance:** Using a *single, zero-shot meta-prompt*, our method sets new SOTA results on benchmarks like MATH (46.3%) and GSM8K (83.5%) with a base Qwen-72B model, outperforming fine-tuned models and even GPT-4.

-----

## A Formal Framework

A core novelty of our work is providing a rigorous mathematical foundation for prompt engineering.

### Meta Prompting as a Functor

We model Meta Prompting as a functor $\\mathcal{M}: \\mathcal{T} \\to \\mathcal{P}$ that maps a category of tasks ($\\mathcal{T}$) to a category of prompts ($\\mathcal{P}$). This ensures that if a complex task can be broken down into simpler sub-tasks, the corresponding prompt can also be constructed by composing simpler prompts. This provides a theoretical guarantee of **modularity and compositionality**.

### Recursive Meta Prompting as a Monad

We model the process of an LLM refining its own prompts (RMP) as a monad. This framework provides a principled way to manage self-improvement, ensuring the process is stable and consistent. The workflow can be visualized as a `Meta-Meta-Prompt` guiding a "Proposer" LLM to generate a task-specific meta-prompt, which is then used by an "Executor" LLM.

-----

## 🚀 State-of-the-Art Results

Our experiments show that Meta Prompting unlocks powerful reasoning in base language models without any fine-tuning, using only a single, example-agnostic prompt.

### MATH Benchmark

With a single meta-prompt, Qwen-72B (base) surpasses all other open-source models and even the initial release of GPT-4.

| Model                 | FT-Dataset | Tool Usage | Eval Method | MATH (%)     |
| --------------------- | ---------- | ---------- | ----------- | ------------ |
| **Proprietary Models**|            |            |             |              |
| Claude-2              | -          | No         | CoT         | 32.5         |
| Minerva-540B          | Arxiv+Web  | No         | CoT         | 33.6         |
| PaLM-2                | -          | No         | CoT         | 34.3         |
| GPT-4 (2023-0314)     | -          | No         | CoT         | 42.5         |
| **Open-source Models**|            |            |             |              |
| Llama-2-70B (base)    | -          | No         | CoT         | 13.5         |
| Qwen-14B (base)       | -          | No         | **MP** | **28.9** |
| Qwen-72B-MetaMathQA   | MetaMathQA | No         | CoT         | 41.7         |
| Qwen-72B (base)       | -          | No         | **MP** | **46.3** |

### GSM8K Benchmark

Our zero-shot meta-prompting approach again shows substantial improvements over both base models with CoT and powerful fine-tuned models.

| Model               | FT-Dataset | Tool Usage | Eval Method | GSM8K (%) |
| ------------------- | ---------- | ---------- | ----------- | --------- |
| Llama-2-70B (base)  | -          | No         | CoT         | 56.8      |
| Qwen-14B (base)     | -          | No         | **MP** | **64.8** |
| WizardMath-70B      | WizardMath | No         | CoT         | 81.6      |
| MetaMath-70B        | MetaMathQA | No         | CoT         | 82.3      |
| Qwen-72B (base)     | -          | No         | **MP** | **83.5** |

### Game of 24

Meta Prompting achieves a 100% success rate by generating a single Python program to solve all puzzles, demonstrating extreme efficiency compared to iterative methods like Tree-of-Thought (ToT).

| Method             | LLM Sessions (per sample) | Generate/Prompt tokens      | Cost (per sample) | Success Rate |
| ------------------ | ------------------------- | --------------------------- | ----------------- | ------------ |
| IO (best of 100)   | 100                       | 1.8k / 1.0k                 | $0.13            | 33%          |
| CoT (best of 100)  | 100                       | 6.7k / 2.2k                 | $0.47            | 49%          |
| ToT (breadth=5)    | 61.72                     | 5.5k / 1.4k                 | $0.74            | 74%          |
| **MP** | **1/N** | **≈ 1/N (8k / 1k)** | **≈ $0.0003** | **100%** |

-----

## Getting Started: Try Meta Prompting

### Using our Prompts

The prompts used in our paper are available in the [`/prompts`](https://www.google.com/search?q=./prompts/) directory. You can use them as system prompts for LLMs to replicate our results or adapt them for your own tasks.

#### Example: MATH Meta-Prompt

Here is the meta-prompt used for the MATH benchmark, which you can use directly.

```markdown
**Problem Statement**:
- **Problem**: [question to be answered]

**Solution Structure**:

1. Begin the response with "Let's think step by step."
2. Follow with the reasoning steps, ensuring the solution process is broken down clearly and logically.
3. End the solution with the final answer encapsulated in a LaTeX-formatted box, $\boxed{...}$, for clarity and emphasis.
4. Finally, state "The answer is [final answer to the problem].", with the final answer presented in LaTeX notation.
---------
```

### Live Demos (Custom GPTs)

You can try out various versions of Meta Prompting agents via these custom GPTs:

  - **🤖 CR Agent (Complex Reasoning):** [v0.1](https://chat.openai.com/g/g-L3a4ZCIHx-cr-agent-v0-1) | [v0.2 (XML)](https://www.google.com/search?q=https://chat.openai.com/g/g-cJV031wLP-cr-agent-xml-v0-2)
  - **✍️ MP-PT (Prompt Revision):** [Online Demo](https://chat.openai.com/g/g-o54JV8zr7-mp-pt)
  - **🔄 MP-ICPD (Recursive In-Context Prompt Design):** [Online Demo](https://www.google.com/search?q=https://chat.openai.com/g/g-9d0iBPnzR-mp-icpd)

-----

## Citation

If you find our work useful for your research, please consider citing our paper and starring this repository. Thank you\!

```bibtex
@article{zhang2023meta,
  title={Meta Prompting for AI Systems},
  author={Zhang, Yifan and Yuan, Yang and Yao, Andrew Chi-Chih},
  journal={arXiv preprint arXiv:2311.11482},
  year={2023}
}
```
