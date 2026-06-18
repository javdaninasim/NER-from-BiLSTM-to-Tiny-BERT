# NER: From BiLSTM to TinyBERT

> A comparative study of **Named Entity Recognition** architectures, from classical recurrent models to compact transformers.  
> **Author:** Nasim Javdani · [GitHub](https://github.com/javdaninasim) · [LinkedIn](https://linkedin.com/in/nasim-javdani-810a9932a)

---

## Overview

This project benchmarks a progression of NLP architectures on the **Named Entity Recognition (NER)** task. Starting from a classical BiLSTM baseline and culminating in TinyBERT, it investigates the accuracy/efficiency trade-offs at each step. All models are implemented and evaluated in Python using Jupyter Notebooks.

---

## Models

### 1. BiLSTM
- Bidirectional LSTM processing token sequences in both directions
- Word embeddings (GloVe / random init) + optional character-level CNN
- Output: per-token softmax over BIO tag set

### 2. BiLSTM-CRF
- Extends BiLSTM with a Conditional Random Field (CRF) output layer
- Models inter-label dependencies (e.g., `I-PER` cannot follow `B-ORG`)
- Globally optimal tag sequence via Viterbi decoding

### 3. BERT (Fine-tuned)
- Pre-trained `bert-base-cased` fine-tuned on the NER dataset
- Contextual token embeddings replace static word vectors
- Significant accuracy gain over recurrent baselines

### 4. TinyBERT
- Knowledge-distilled version of BERT: 4 layers, 4 attention heads, 312 hidden dims
- ~7× smaller and ~9× faster than BERT-base
- Evaluated on accuracy retention vs. model size trade-off

---

## Dataset

**CoNLL-2003** (English NER benchmark):
- Entity types: `PER` (person), `ORG` (organization), `LOC` (location), `MISC`
- Standard train / dev / test splits

---

## Results

| Model | F1 Score | Parameters | Inference Speed |
|---|---|---|---|
| BiLSTM | — | ~5M | Fast |
| BiLSTM-CRF | — | ~5M | Fast |
| BERT-base | — | 110M | Slow |
| TinyBERT | — | 14.5M | Fast |

*(Fill in F1 scores after running the notebooks.)*

---

## Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Core programming language |
| Jupyter Notebook | Experiments and reporting |
| PyTorch | Model implementation |
| Hugging Face Transformers | BERT and TinyBERT |
| `seqeval` | NER-specific evaluation (span-level F1) |
| Datasets (HuggingFace) | CoNLL-2003 data loading |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/javdaninasim/NER-from-BiLSTM-to-Tiny-BERT.git
   cd NER-from-BiLSTM-to-Tiny-BERT
   ```
2. Install dependencies:
   ```bash
   pip install torch transformers datasets seqeval jupyter
   ```
3. Run notebooks in order (1 → 4) for a progressive comparison.

---

## Course Info

- **Institution:** Sharif University of Technology, Department of Computer Engineering
- **Topic:** Natural Language Processing / Deep Learning
