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

## Usage Guide
1. Parallel Multilingual Dataset Construction

Translate the original EmoBench-style dataset into a target language while preserving semantic content, pragmatic intent, and label alignment.


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

