import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"  # you can change this

# -----------------------------
# Task Definition
# -----------------------------
TASK = "Answer the question accurately."

QUESTIONS = [
    "What causes rain?",
    "Explain overfitting in machine learning.",
    "Why is the sky blue?"
]

# -----------------------------
# Prompt Templates
# -----------------------------
def zero_shot_prompt(question):
    return f"{TASK}\n\nQuestion: {question}\nAnswer:"

def few_shot_prompt(question):
    return f"""
{TASK}

Question: What is photosynthesis?
Answer: Photosynthesis is the process by which plants convert sunlight into energy.

Question: What is gravity?
Answer: Gravity is a force that attracts objects with mass toward each other.

Question: {question}
Answer:
"""

def chain_of_thought_prompt(question):
    return f"""
{TASK}

Question: {question}
Answer step by step:
"""

# -----------------------------
# LLM Call
# -----------------------------
def get_response(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Run Experiment
# -----------------------------
results = []

for question in QUESTIONS:
    zs = get_response(zero_shot_prompt(question))
    fs = get_response(few_shot_prompt(question))
    cot = get_response(chain_of_thought_prompt(question))

    results.append({
        "Question": question,
        "Zero-Shot Response": zs,
        "Few-Shot Response": fs,
        "Chain-of-Thought Response": cot
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("✅ Experiment completed. Results saved to results.csv")
