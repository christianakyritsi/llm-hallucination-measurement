# 🧠 LLM Hallucination Measurement under Emotional Framing

Investigating whether emotionally framed factual questions influence hallucination rates across five transformer-based LLMs, using a custom-built dataset of 823 validated stimuli.

> 📄 **Full paper available in this repo:** `INFOMTALC_Final_Report.pdf`

## 📌 Overview

**Research Question:** Does the emotional framing of a factual question influence hallucination rates in transformer-based LLMs?

We constructed a dataset of emotionally reframed factual questions (joy, sadness, anger, fear, neutral) derived from TriviaQA, validated them using a GoEmotions classifier, and evaluated five instruction-tuned LLMs on factual accuracy. Statistical analysis used chi-squared tests and Cramér's V effect sizes.

## 🤖 Models Evaluated

| Model | Accuracy | Hallucination Rate |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | 74.5% | 25.5% |
| Meta-Llama-3-8B-Instruct | 74.2% | 25.8% |
| Phi-3-mini-4k-instruct | 60.8% | 39.2% |
| Falcon-7b-instruct | 40.7% | 59.3% |
| Flan-T5-Large | 21.4% | 78.6% |

## 🔍 Key Findings

- 😤 **Anger** consistently *reduced* hallucination rates below neutral in 4/5 models (~3.8 pp decrease)
- 😢 **Sadness** and 😨 **fear** tended to *increase* hallucination rates
- 📊 Emotion effects were **not statistically significant** (Cramér's V: 0.020–0.058, negligible)
- 🌍 **Topic domain** was the dominant predictor (Cramér's V: 0.186–0.316, small-medium)
- Geography = lowest hallucination rates; Culture & History = highest
- Substitution errors dominated — models retrieve the wrong fact rather than fabricating

## 🗂️ Dataset

Emotionally reframed factual QA dataset (823 stimuli, 5 emotion conditions, 4 domains):

👉 [emotion-framed-factual-qa on Hugging Face](https://huggingface.co/datasets/belpekkan/emotion-framed-factual-qa)

```python
from datasets import load_dataset
df = load_dataset("belpekkan/emotion-framed-factual-qa")
```

## 🛠️ Tech Stack

`Python` `HuggingFace Transformers` `TriviaQA` `GoEmotions` `sentence-transformers` `scipy` `pandas`

## 👩‍💻 Authors

Christiana Kyritsi · Beliz Pekkan · Geanina Verestiuc · Mariyana Shishmanova

*Utrecht University — Transformers: Applications in Language and Communication, 2026*
