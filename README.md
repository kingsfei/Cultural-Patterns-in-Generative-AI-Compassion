# Cultural Patterns in Generative AI Compassion

> **Official Research Code Repository**

This repository contains the research code accompanying the paper:

**Cultural Patterns in Generative AI Compassion**  
*(Manuscript under review)*

It implements a multi-stage experimental framework for examining how large language models (LLMs) understand emotions, select compassionate actions, and respond to cultural identity cues across languages.


## ✨ Overview

This project studies **cross-cultural patterns in generative AI compassion** using a **controlled, choice-based experimental design**.

Instead of evaluating free-form text generation, compassion is operationalized as **repeated choices over a fixed set of culturally grounded action strategies**, enabling systematic comparison across:

- 🌍 Languages  
- 🤖 Model families  
- 🧭 Cultural identity conditions  

The experimental pipeline consists of three stages:

1. **Parallel Multilingual Dataset Construction**  
   Semantically aligned translation of emotional scenarios while preserving pragmatic intent and emotional logic.

2. **Compassionate Action Preference Measurement**  
   Measurement of LLM preferences over predefined compassionate action strategies reflecting cultural orientations.

3. **Cultural Identity Priming (Causal Test)**  
   A controlled manipulation introducing minimal cultural identity cues to test causal effects on compassionate behavior.

All stages support repeated stochastic sampling and uncertainty estimation.

## 🧪 Usage Guide

This repository is intended for **research reproducibility and methodological transparency**, rather than as a turn-key executable pipeline. Each script corresponds to a distinct methodological component of the experimental framework described in the paper, and is designed to be used independently depending on the research stage of interest. Specifically, `Translate_Emobench.py` supports parallel multilingual dataset construction under strict semantic and structural constraints; `Generate_subset.py` constructs the fixed compassionate action choice space used as experimental stimuli; `Test_prefs.py` measures large language model preferences over these strategies through repeated stochastic sampling; and `Test_priming.py` extends this evaluation by introducing minimal cultural identity primes to test causal effects on compassionate action selection. Model access, API providers, and credentials are intentionally decoupled from the code logic and configured externally, allowing the framework to be applied flexibly across different models and deployment environments.




## 🔌 Supported APIs

The codebase supports multiple large language model backends through a unified interface.

**OpenAI**

- Official OpenAI API
- GPT-4–series models
  
**OpenAI-Compatible APIs**
Providers implementing the OpenAI Chat Completions schema, including:

- OpenRouter
- DeepSeek
-SiliconFlow


## 📖 Citation

If you find our work useful for your research, please kindly cite our paper as follows:

```bibtex
@article{cultural_compassion_llm,
  title   = {Cultural Patterns in Generative AI Compassion},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}

