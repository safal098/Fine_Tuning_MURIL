# MuRIL Fine-Tuning for Nepali / Code-Mixed Text Classification

## 📌 Project Overview

This repository contains a **complete fine-tuning pipeline for MuRIL (Multilingual Representations for Indian Languages)** using Hugging Face Transformers.  
The project is designed for **sentence-level text classification** tasks, with a strong focus on **Nepali and Nepali–English code-mixed social media text**, such as reviews and comments.

The notebook demonstrates **best practices in modern NLP engineering**, including:
- Efficient tokenization
- Robust training and evaluation
- Clean dataset handling
- Reproducible experiment configuration

---

## 🧠 Why MuRIL?

**MuRIL** (`google/muril-base-cased`) is specifically trained on:
- Indian subcontinent languages (including **Nepali**)
- Transliterated and code-mixed text

This makes it **far superior to generic multilingual models** (e.g., mBERT) for:
- Nepali sentiment analysis
- Social media text understanding
- Low-resource language NLP tasks

---

## 🏗️ Architecture & Workflow

```text
Raw Text Data (CSV / TSV)
        ↓
Sentence-Level Tokenization (MuRIL Tokenizer)
        ↓
Fine-Tuned MuRIL Transformer
        ↓
Classification Head
        ↓
Evaluation Metrics (Accuracy, Precision, Recall, F1)
📁 Repository Structure
bash
Copy code
├── MURIL_FT.ipynb        # Main training and evaluation notebook
├── data/
│   ├── train.tsv        # Training dataset
│   ├── validation.tsv  # Validation dataset
│   └── test.tsv        # Test dataset
├── outputs/
│   └── muril_finetuned/ # Saved model checkpoints
├── README.md
📊 Dataset Format
The model expects sentence-level labeled data in .tsv or .csv format.

Required Columns
Column Name	Description
text	Input sentence / comment
label	Class label (integer encoded)

Example
tsv
Copy code
text	label
यो ठाउँ धेरै राम्रो छ	1
service ramro chaina	0
⚠️ Token-level annotation is not required. This pipeline is optimized for sentence classification.

⚙️ Installation & Setup
1️⃣ Install Dependencies
bash
Copy code
pip install transformers accelerate datasets evaluate scikit-learn torch pandas
2️⃣ Environment
Python 3.9+

GPU recommended (Google Colab / CUDA)

CPU training supported for small datasets

🚀 Training Pipeline (Notebook Steps)
1. Data Loading
Loads TSV files using Pandas

Verifies column integrity

Converts data into Hugging Face Dataset objects

2. Tokenization
Uses AutoTokenizer from MuRIL

Dynamic padding for faster training

Sentence-level tokenization (no manual splitting)

3. Model Initialization
AutoModelForSequenceClassification

Custom label mappings (id2label, label2id)

Supports binary and multi-class classification

4. Training
Hugging Face Trainer API

Optimized learning rate and batch size

Epoch-based evaluation

5. Evaluation
Metrics computed using scikit-learn:

Accuracy

Precision

Recall

F1-score

6. Model Saving
Best checkpoint automatically saved

Ready for inference or deployment

📈 Metrics Used
text
Copy code
Accuracy
Precision (weighted)
Recall (weighted)
F1-score (weighted)
Weighted metrics ensure robust evaluation for class-imbalanced datasets, common in real-world social media data.

🧪 Example Results (Typical)
Metric	Score
Accuracy	~85–90%
F1-score	~0.86

Actual performance depends on dataset size, cleanliness, and label quality.

🔍 Inference Usage (After Training)
python
Copy code
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="outputs/muril_finetuned",
    tokenizer="google/muril-base-cased"
)

classifier("यो होटल धेरै राम्रो लाग्यो")
🛠️ Engineering Best Practices Followed
✅ Reproducible experiments
✅ Clear separation of training & evaluation
✅ Language-aware model selection
✅ Scalable to production inference
✅ Clean and readable pipeline

📌 Use Cases
Nepali sentiment analysis

Code-mixed (Nepali–English) text classification

Tourism review analysis

Social media opinion mining

Academic NLP research (Final Year / Master’s Thesis)

🔮 Future Improvements
Add hyperparameter tuning

Integrate class imbalance handling

Export model to ONNX

Deploy via FastAPI / Django REST

Combine with token-level LID or NER

👨‍💻 Author
Safal Sharma
NLP & AI Engineering
Focus: Low-Resource Languages, Code-Mixed NLP, Transformer Models
