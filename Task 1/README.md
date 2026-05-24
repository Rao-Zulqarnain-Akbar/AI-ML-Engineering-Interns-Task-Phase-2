# 📰 News Topic Classifier — BERT Fine-Tuning on AG News

Fine-tune `bert-base-uncased` on the **AG News** dataset to classify news headlines  
into four categories: **World · Sports · Business · Sci/Tech**.

---

## 🗂️ Project Structure

```
news_classifier/
├── train.py           ← Fine-tune BERT (main training script)
├── evaluate.py        ← Detailed evaluation + 4 visualisation plots
├── predict.py         ← CLI inference (interactive / file / single)
├── app.py             ← Gradio web app for live demo
├── requirements.txt   ← Python dependencies
├── models/
│   └── bert_ag_news/  ← Saved fine-tuned model (created by train.py)
└── outputs/
    ├── logs/          ← Training logs
    ├── metrics.json   ← Accuracy & F1 scores
    ├── confusion_matrix.png
    └── evaluation_report.png
```

---

## ⚙️ Setup

```bash
# 1 — Clone / enter the project folder
cd news_classifier

# 2 — Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3 — Install dependencies
pip install -r requirements.txt
```

> **GPU recommended.**  Training on CPU will work but takes ~30 min/epoch.  
> With a T4 / A100 GPU the full 3-epoch run completes in ≈ 15 minutes.

---

## 🚀 Quick Start

### Step 1 — Fine-tune the model

```bash
python train.py
```

What it does:
- Downloads **AG News** (~120 k train / 7.6 k test headlines) via 🤗 Datasets
- Tokenises with `BertTokenizerFast` (max 128 tokens, dynamic padding)
- Fine-tunes `bert-base-uncased` for 3 epochs with early stopping
- Saves the best checkpoint to `models/bert_ag_news/`
- Prints **accuracy** and **F1** on the held-out test set

### Step 2 — Evaluate in detail

```bash
python evaluate.py
```

Produces:
- Full `classification_report` (precision, recall, F1 per class)
- `outputs/evaluation_report.png` with 4 subplots:
  1. Confusion matrix (%)
  2. Per-class F1 bar chart
  3. True vs predicted class distribution
  4. Confidence histogram (correct vs incorrect)

### Step 3 — Predict on new headlines

```bash
# Single headline
python predict.py "NASA discovers water on the Moon"

# Batch from file (one headline per line)
python predict.py --file my_headlines.txt

# Interactive mode (type headlines one by one)
python predict.py
```

### Step 4 — Launch the web app

```bash
python app.py
# → Open http://localhost:7860 in your browser
```

---

## 📊 Expected Results

| Metric | Value |
|---|---|
| **Accuracy** | ~94 % |
| **F1 (macro)** | ~94 % |
| **F1 (weighted)** | ~94 % |

Per-class F1 scores typically reach **0.93 – 0.96** across all four categories.

---

## 🔧 Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Tokeniser | `BertTokenizerFast` | 10× faster than slow tokeniser |
| Padding | Dynamic (`DataCollatorWithPadding`) | Saves memory vs. fixed padding |
| Scheduler | Cosine with 10 % warm-up | Stabilises early training |
| Mixed precision | `fp16=True` on CUDA | ~2× speed-up, minimal accuracy cost |
| Early stopping | patience = 2 | Prevents overfitting on small dataset |
| Metric | `f1_macro` | Balanced across unequal classes |

---

## 🧠 How Fine-Tuning Works

```
                 ┌──────────────────────────────┐
Input headline ──▶  BertTokenizerFast            │
                 │  (WordPiece, max 128 tokens)  │
                 └────────────┬─────────────────┘
                              │
                 ┌────────────▼─────────────────┐
                 │  bert-base-uncased            │
                 │  12 layers · 768 hidden dim   │
                 │  110 M parameters             │
                 └────────────┬─────────────────┘
                              │  [CLS] token
                 ┌────────────▼─────────────────┐
                 │  Linear(768 → 4)             │
                 │  + Softmax                   │
                 └────────────┬─────────────────┘
                              │
                    World / Sports / Business / Sci/Tech
```

Only the classification head is trained from scratch; the BERT encoder  
is fine-tuned with a low learning rate (2e-5) to preserve pretrained knowledge.

---

## 📚 References

- [AG News Dataset](https://huggingface.co/datasets/ag_news)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Gradio Docs](https://www.gradio.app/docs)
