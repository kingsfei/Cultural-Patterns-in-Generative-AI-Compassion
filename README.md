# Cultural Patterns in Generative AI Compassion

This repository contains the official research code accompanying the paper:

> **Cultural Patterns in Generative AI Compassion**  
> *(Manuscript under review)*

The codebase implements a multi-stage experimental pipeline for studying how large language models (LLMs) understand emotions, select compassionate actions, and respond to cultural identity priming across languages.

---

## Overview

This project investigates **cross-cultural patterns in generative AI compassion** using a controlled, choice-based experimental framework.

Rather than evaluating free-form text generation, the framework operationalizes compassion as **repeated choices over a fixed set of culturally grounded action strategies**, allowing systematic comparison across:

- languages,
- model families,
- and cultural identity conditions.

The overall experimental pipeline consists of three stages:

1. **Parallel Multilingual Dataset Construction**  
   Construction of semantically aligned multilingual emotional scenarios while preserving pragmatic intent and emotional logic.

2. **Compassionate Action Preference Measurement**  
   Measurement of LLM preferences over four predefined compassionate action strategies reflecting distinct cultural orientations.

3. **Cultural Identity Promting**  
   Examination of whether minimal cultural identity cues causally shift compassionate action preferences under otherwise identical task conditions.

All stages support repeated stochastic sampling and uncertainty estimation.
