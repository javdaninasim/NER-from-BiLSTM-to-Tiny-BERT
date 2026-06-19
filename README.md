<div align="center">

<h1 align="center">🔤 NER: From BiLSTM to Tiny BERT</h1>
<h3 align="center">Progressive NLP Architecture Exploration | Named Entity Recognition | Sharif University</h3>

[**GitHub**](https://github.com/javdaninasim/NER-from-BiLSTM-to-Tiny-BERT) &nbsp; ⬩ &nbsp; [**Dataset: OntoNotes5**](https://huggingface.co/datasets/tner/ontonotes5) &nbsp; ⬩ &nbsp; [**University**](https://sharif.edu/)

</div>

---

### 📚 Course Context

```
Instructor: Dr. Ali Sharifi-Zarchi
Institution: Sharif University of Technology, Computer Engineering Department
Course: Machine Learning (Fall 2025)
Assignment: Practical Assignment 5 - Natural Language Processing
Focus: NER Architecture Evolution & Comparative Analysis
```

---

### 🎯 Project Overview

```python
class NERArchitectureJourney:
    def __init__(self):
        self.task = "Named Entity Recognition (NER)"
        self.dataset = "OntoNotes5 (59.9K train, 8.5K val, 8.3K test)"
        self.entity_types = ["PER", "ORG", "LOC", "MISC"]
        self.evaluation_metric = "Entity-level F1-score"
        
    def learning_goal(self):
        return """
        Understand how modern Transformer architectures emerge from
        progressive architectural innovations, starting from BiLSTM
        baselines and building up to BERT-style pretraining.
        """
    
    def progression(self):
        return [
            "1. BiLSTM baseline (sequential encoder)",
            "2. Add attention (global context mixing)",
            "3. Attention-only (remove sequential bias)",
            "4. Add positional encoding (restore order)",
            "5. Multi-head attention (multiple attention patterns)",
            "6. Deeper transformer encoder (scaling depth)",
            "7. BERT-style embeddings (special tokens, segments)",
            "8. MLM pretraining (self-supervised learning)",
            "9. MLM+NSP pretraining (multi-objective learning)",
            "10. Fine-tune TinyBERT on NER",
            "11. Compare with foundation models (DistilBERT/BERT)",
            "12. Analysis & interpretability lab"
        ]
```

---

### 📂 Repository Structure

```
NER-from-BiLSTM-to-Tiny-BERT/
│
├── code.ipynb                    # Complete end-to-end practical lab
└── README.md                     # Documentation
```

---

### 🏗️ Architecture Progression

#### **Part 1: BiLSTM Baseline** 🔄
**What you learn:** Sequential processing with bidirectional context

- Bidirectional LSTM reads sequences left→right and right→left
- Token embeddings (word vectors)
- Per-token softmax classification over BIO tags
- **Baseline performance:** Strong recurrent model

#### **Part 2: BiLSTM + Attention** 👁️
**What you learn:** Adding global context mixing to sequential encoder

- Keep BiLSTM encoder
- Add single self-attention layer on top
- Allows direct token-to-token interactions
- **Expected improvement:** Captures global dependencies better

#### **Part 3: Attention-Only (No LSTM)** ⚠️
**What you learn:** Self-attention alone lacks positional awareness

- Remove LSTM, use only self-attention
- Pure content-based mixing
- **Expected issue:** Order-insensitive → degrades NER performance
- **Diagnostic:** Identifies need for position encoding

#### **Part 4: Positional Encoding** 📍
**What you learn:** How position information enables order-aware attention

- Keep attention-only design
- Add positional encoding (sine/cosine or learnable)
- Injects position information into embeddings
- **Recovery:** Performance bounces back close to Part 3 baseline
- **Ablation insight:** Part 3 vs Part 4 shows position value

#### **Part 5: Multi-Head Attention** 🧠
**What you learn:** Parallel attention patterns for specialization

- Replace single attention with multi-head (e.g., 4/8 heads)
- Different heads learn different attention patterns:
  - Some focus on local context
  - Others capture long-range dependencies
  - Some learn syntactic/semantic boundaries
- **Visualization:** Attention heatmaps reveal head specialization

#### **Part 6: Transformer Encoder Stack** 🏗️
**What you learn:** Stacking depth vs. optimization tradeoffs

- Convert attention block to standard encoder layer
- Add feed-forward networks (2-layer MLP)
- Layer normalization and residual connections
- Stack 2, 4, 6, 12 layers (depth ablation)
- **Insight:** Depth helps but not always; optimization matters

#### **Part 7: BERT-style Embedding Recipe** 🔌
**What you learn:** How BERT structures inputs for pretraining

- Special tokens: `[CLS]`, `[SEP]`, `[MASK]`
- Token embeddings + position embeddings + segment embeddings
- Sum three embedding sources
- **Purpose:** Supports masking and paired-sentence pretraining

#### **Part 8: Masked Language Modeling (MLM)** 🎭
**What you learn:** Self-supervised pretraining objective

- Randomly mask 15% of tokens
- Predict masked token from context only
- **Benefit:** Learns language structure without manual labels
- **Mechanism:** Bidirectional context → deep language understanding

#### **Part 9: Next Sentence Prediction (NSP)** 🔗
**What you learn:** Multi-task pretraining and sentence understanding

- Binary task: Is sentence B the true next sentence?
- 50% real pairs, 50% random pairs
- Train MLM + NSP jointly
- **Comparison:** MLM alone vs. MLM+NSP effectiveness
- **Note:** NSP benefits are mixed (modern models often drop it)

#### **Part 10: Fine-tune TinyBERT on NER** 🎯
**What you learn:** How pretraining initializes downstream learning

- Compare 3 conditions under same fine-tuning budget:
  - **Random init** (scratch learning)
  - **MLM-pretrained** (self-supervised only)
  - **MLM+NSP-pretrained** (multi-task pretrained)
- **Metrics:** Training speed, convergence, final entity F1
- **Insight:** Pretraining reduces data requirements

#### **Part 11: Foundation Models** 🏛️
**What you learn:** What large-scale pretraining buys

- Fine-tune real pretrained models: DistilBERT, BERT-base
- Large-scale pretraining data (Wikipedia + BookCorpus)
- Subword tokenization (BPE/WordPiece) effects
- **Performance jump:** Significant gains over smaller pretrained models
- **Trade-off:** Computational cost vs. accuracy

#### **Part 12: Analysis & Interpretability Lab** 🔬
**What you learn:** Practical debugging and efficiency analysis

- **Attention visualization:** Heatmaps from fine-tuned model
- **Failure analysis:** Entity boundaries, type confusion
- **Computational efficiency:**
  - Sequence length vs. runtime (quadratic scaling)
  - Model size vs. accuracy trade-offs
- **Comparison table:** All architectures side-by-side
- **Takeaways:** When to use which approach

---

### 📊 Expected Results

| Model | Params | F1 Score | Training Time | Notes |
| :--- | ---: | ---: | ---: | :--- |
| BiLSTM | ~5M | Baseline | Fast | Classical strong baseline |
| BiLSTM+Attention | ~5M | ↑ Minor | Fast | Modest improvement |
| Pure Attention | ~5M | ↓ Significant | Fast | Importance of position |
| + Positional Encoding | ~5M | ↑ Recover | Fast | Position matters! |
| Multi-Head (4H) | ~5M | ↑ Better | Fast | Specialization helps |
| Transformer Stack | ~5-10M | ↑↑ Good | Medium | Depth + optimization |
| TinyBERT (random) | ~14.5M | Medium | Medium | No pretraining benefit |
| TinyBERT (MLM) | ~14.5M | ↑↑ Better | Medium-Fast | Pretraining helps |
| TinyBERT (MLM+NSP) | ~14.5M | ↑↑↑ Best | Medium-Fast | Multi-task pretraining |
| DistilBERT (pretrained) | ~67M | ↑↑↑↑ Excellent | Slow | Large-scale pretraining |
| BERT-base (pretrained) | ~110M | ↑↑↑↑↑ SOTA | Slow | Full-scale pretraining |

---

### 🎓 Learning Outcomes by Part

| Part | Key Concept | Ablation Insight |
| ---: | :--- | :--- |
| 1-2 | Sequential baseline + attention | Attention ⊕ sequence awareness |
| 3-4 | Position encoding necessity | Part 3 vs 4 shows: **position is critical** |
| 5 | Multi-head specialization | Visualization reveals head roles |
| 6 | Depth & optimization | Depth ≠ always better without tuning |
| 7 | BERT embedding design | Special tokens + segments matter for pretraining |
| 8-9 | Self-supervised pretraining | MLM is powerful; NSP = mixed |
| 10 | Pretraining transfer value | Pretraining → faster downstream learning |
| 11 | Foundation models | Scale + large data = massive gains |
| 12 | Efficiency & interpretability | Quadratic cost; attention maps reveal patterns |

---

### ⚡ Tech Stack

<div align="left">
  <img src="https://img.shields.io/badge/Python-1A1A1A?style=for-the-badge&logo=python&logoColor=3776AB" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-1A1A1A?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Transformers-1A1A1A?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="Transformers" />
  <img src="https://img.shields.io/badge/Jupyter-1A1A1A?style=for-the-badge&logo=jupyter&logoColor=F37726" alt="Jupyter" />
  <img src="https://img.shields.io/badge/HuggingFace-1A1A1A?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/seqeval-1A1A1A?style=for-the-badge&logo=python&logoColor=3776AB" alt="seqeval" />
</div>

<br>

> **Core Libraries:** `torch` ⬩ `transformers` ⬩ `datasets` ⬩ `seqeval` ⬩ `matplotlib` ⬩ `tqdm`

---

### 📥 Installation & Setup

```bash
# Clone repository
git clone https://github.com/javdaninasim/NER-from-BiLSTM-to-Tiny-BERT.git
cd NER-from-BiLSTM-to-Tiny-BERT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets seqeval jupyter matplotlib tqdm

# (Optional) Install accelerate for faster training
pip install accelerate

# Launch notebook
jupyter notebook code.ipynb
```

---

### 📖 Notebook Structure

**`code.ipynb`** is organized as a **hands-on practical lab** with:

- **Part 0:** Setup, data pipeline, evaluation harness
  - Load OntoNotes5 dataset
  - Entity-level F1 metric (main score)
  - Reproducible training loop
  
- **Parts 1-6:** Progressive architectural experiments
  - Code templates with TODO markers
  - Fair comparisons under controlled conditions
  - Training curves and performance tracking
  
- **Parts 7-10:** Pretraining and fine-tuning
  - Build BERT-like pretraining pipeline
  - Compare initialization strategies
  - Foundation model fine-tuning
  
- **Part 12:** Analysis lab
  - Attention visualization
  - Failure case analysis
  - Efficiency measurements
  - Final comparison table

---

### 🎓 Grading & Evaluation Criteria

Your submission will be evaluated on:

1. **Correctness & Rerunnability** (20%)
   - Code must be correct and run end-to-end
   - No hidden dependencies or bugs
   
2. **Reasonable Training Progression** (20%)
   - Loss decreases, metrics stabilize
   - Training curves show sensible trends
   
3. **Final Performance** (20%)
   - Results should be in expected ranges
   - Not drastically below typical benchmarks
   
4. **Visualizations & Plots** (20%)
   - Clear, labeled, interpretable
   - Attention heatmaps well-formatted
   
5. **Analysis & Concept Checks** (20%)
   - Correct explanations
   - Clear reasoning for design choices

---

### 💡 Key Insights & Takeaways

| Insight | Implication |
| :--- | :--- |
| **Position is Critical** | Pure attention without position fails → Part 3 vs 4 ablation |
| **Multi-head Specialization** | Different heads learn different patterns (local vs global) |
| **Pretraining Transfers** | MLM pretraining reduces data requirements on downstream tasks |
| **Scaling Works** | More layers + larger models + more pretraining data → better performance |
| **Quadratic Cost** | Attention is O(n²) in sequence length → practical bottleneck for long sequences |
| **BERT Embeddings Matter** | Special tokens + segment IDs support paired-sentence understanding |
| **NSP is Mixed** | Joint MLM+NSP helps, but many modern models drop NSP |
| **Foundation Models Win** | Large-scale pretraining >> small pretrained models >> scratch |

---

### 📊 NER Evaluation: Why Entity F1 Matters

**Token-level accuracy is misleading** because:
- Most tokens are labeled **O** (not entities)
- A model predicting **O** everywhere appears ~90% "accurate"

**Entity-level F1 is the right metric:**
- Exact span match required (boundaries + type)
- Penalizes false positives (hallucinated entities)
- Penalizes false negatives (missed entities)
- Punishes wrong entity types

Formula:
$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad \text{F1} = \frac{2 \cdot P \cdot R}{P + R}$$

---

### 📌 Important Notes for Students

- ✅ You are encouraged to modify code for experimentation
- ✅ Notebook will be **re-run by graders** → write clean, stable code
- ✅ Keep comparisons **fair** (same data splits, same evaluation, similar budgets)
- ✅ **Do not change the core task** (keep NER on OntoNotes5)
- ✅ Explain important changes in **Markdown cells**
- ✅ Even unexpected results can earn credit if **clearly diagnosed**

---

### 🔗 Resources

- **OntoNotes5 Dataset:** [HuggingFace Datasets](https://huggingface.co/datasets/tner/ontonotes5)
- **seqeval Metrics:** [GitHub - seqeval](https://github.com/chakki-works/seqeval)
- **Transformers Library:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- **PyTorch:** [pytorch.org](https://pytorch.org)
- **BERT Paper:** [Devlin et al. (2019)](https://arxiv.org/abs/1810.04805)
- **Attention Is All You Need:** [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)

---

### ℹ️ Course Information

| Detail | Value |
| :--- | :--- |
| **Instructor** | Dr. Ali Sharifi-Zarchi |
| **Institution** | Sharif University of Technology |
| **Department** | Computer Engineering |
| **Course** | Machine Learning (Fall 2025) |
| **Assignment** | Practical Assignment 5 - NLP |
| **Dataset** | OntoNotes5 (English NER) |
| **Format** | Jupyter Notebook (hands-on lab) |

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0096FF&height=100&section=footer" width="100%"/>
</div>
