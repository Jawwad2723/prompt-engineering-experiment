# Prompt Engineering Experiment

## Overview
This project explores how different prompting strategies affect the quality, consistency, and accuracy of Large Language Model (LLM) responses. The experiment compares zero-shot, few-shot, and chain-of-thought prompting on the same set of questions.

## Objectives
- Understand LLM behavior under different prompt designs
- Compare response clarity and consistency
- Identify effective prompt engineering techniques

## Prompting Strategies
- **Zero-shot:** Direct question without examples
- **Few-shot:** Includes example question–answer pairs
- **Chain-of-thought:** Encourages step-by-step reasoning

## Methodology
- Used a fixed set of factual and conceptual questions
- Kept model, temperature, and inputs constant
- Recorded responses for qualitative comparison

## Results
- Few-shot prompting improved response consistency
- Chain-of-thought prompting produced more detailed explanations
- Zero-shot prompting was concise but occasionally less accurate

## Technologies Used
- Python
- OpenAI-compatible LLM API
- Pandas

## How to Run
```bash
pip install -r requirements.txt
python prompts_experiment.py
