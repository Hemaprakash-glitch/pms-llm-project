# PMS LLM Medical Classifier

## Overview

This project builds an AI system to automatically extract structured data from medical complaint narratives using a fine-tuned Large Language Model (Mistral-7B).

---

## Key Features

* LLM-based extraction (13 PMS fields)
* GGUF model (runs offline, CPU supported)
* Structured JSON output
* Multi-complaint handling
* Real-world PMS dataset usage

---

## Tech Stack

* Python
* Llama.cpp (GGUF inference)
* Mistral-7B
* Streamlit
* NLP / LLM

---

## Model Note

The model file is not uploaded due to large size (4GB+).

Place manually:
models/pms_mistral.q4_k_m.gguf

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Results

* 100% JSON structural accuracy
* 56% classification accuracy

---

## 👨‍💻 Author

Hemaprakash V K
