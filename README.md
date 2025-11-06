# 🧠 EEG–Eye–Based Sentence Generation  
> _Multimodal Neural–Language Decoding using EEG, Eye-Tracking & Spectral Features_

---

## 🌍 Overview

This repository presents a **Transformer-based multimodal model** that reconstructs **English sentences** from synchronized **EEG**, **eye-tracking**, and **spectral brainwave** data.  
It bridges **cognitive neuroscience** and **natural language generation (NLG)** through deep learning.

---

## 📚 Table of Contents
- [Project Summary](#-project-summary)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Model Architecture](#-model-architecture)
- [Training Setup](#️-training-setup)
- [Evaluation](#-evaluation)
- [Streamlit Web App](#-streamlit-web-app)
- [Results & Insights](#-results--insights)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🧩 Project Summary

The goal is to map **neural activity and ocular behavior** to corresponding **linguistic representations**.  
Each input sequence captures how a subject reads and processes a sentence; the model then learns to reconstruct the original text purely from brain and eye features.

---

## 📊 Dataset

The dataset integrates multimodal information:

| Modality | Features | Description |
|-----------|-----------|-------------|
| **EEG** | 24 PCA components × 256 timesteps | Brain electrical activity from multiple scalp channels |
| **Eye-tracking** | Fixation count, mean fixation duration, mean pupil size | Gaze and visual attention indicators |
| **Spectral EEG** | Mean α, β, γ, θ band amplitudes | Frequency-domain energy from filtered EEG |
| **Text** | Ground-truth sentence | Linguistic target for each recording |

**Source:** Adapted from the [ZuCo Dataset](https://osf.io/q3zws/) and extended with cleaned and normalized samples.

---

<details>
<summary>⚙️ <b>Data Preprocessing</b></summary>

1. **Signal Cleaning**
   - EEG raw signals filtered with **band-pass filters (0.1–45 Hz)**.  
   - Removal of **ocular and muscular artifacts** using **ICA decomposition**.

2. **Resampling & Synchronization**
   - EEG resampled to **256 Hz** for temporal consistency.  
   - Eye-tracking synchronized via timestamp alignment with EEG epochs.

3. **Feature Extraction**
   - **Spectral decomposition:** Welch’s method → mean power for α, β, γ, θ bands.  
   - **Eye metrics:** fixation count, mean duration, and average pupil size.

4. **Dimensionality Reduction**
   - **PCA (Principal Component Analysis)** applied to EEG → reduced from 64–128 channels to **24 principal components**, retaining >95% variance.

5. **Normalization**
   - EEG, eye, and spectral features standardized with **Z-score normalization**.  
   - Outlier values clipped to ±3 σ.

6. **Encoding**
   - Sentences tokenized with **DistilBERT tokenizer** (max length = 64 tokens).  
   - Padding and truncation ensure uniform sequence length.

</details>

---

## 🏗️ Model Architecture
<img width="638" height="435" alt="Screenshot from 2025-11-06 02-33-12" src="https://github.com/user-attachments/assets/4af62479-78c6-459e-85a4-c5105cf768d2" />

**Main components:**
- 🧠 **EEG Encoder** → Conv1D + Bi-LSTM for spatio-temporal neural encoding.  
- 👁️ **Eye Encoder** → 2-layer MLP capturing fixation and pupil metrics.  
- ⚡ **Spectral Encoder** → Dense non-linear transformation of frequency features.  
- 🔗 **Fusion Module** → Concatenates and projects all encoders into a shared space.  
- 🔄 **Transformer Decoder** → Multi-head cross-attention linking brain features to linguistic tokens.  
- 💬 **Text Generator** → Outputs token logits decoded to natural sentences.

---

## ⚙️ Training Setup

| Parameter | Value |
|------------|--------|
| Optimizer | AdamW |
| Learning Rate | 3e-5 (with cosine decay) |
| Batch Size | 8 |
| Epochs | 30 |
| Loss | Cross-Entropy + Label Smoothing (0.1) |
| Dropout | 0.3 |
| Hardware | NVIDIA RTX 3050 (4 GB) |

**Regularization techniques:**
- Teacher forcing (80% ratio)  
- Top-k / top-p sampling for inference diversity  
- Early stopping based on validation loss

---

## 📈 Evaluation

| Metric | Score |
|--------|--------|
| Final Train Loss | **1.51** |
| Validation Loss | **1.49** |
| BLEU | **0.61** |

Loss decreased smoothly across epochs, confirming stable convergence.

---

## 💬 Sample Results

| Generated Sentence | Ground Truth |
|--------------------|---------------|
| he had been educated at oxford and his closest friends and outlook on life were british. | He had been educated at Oxford and his closest friends and outlook on life were British. |
| while a printing apprentice he wrote under the pseudonym of ‘silence dogood’. | While a printing apprentice he wrote under the pseudonym of ‘Silence Dogood’. |
| simon has been married three times; he is currently married to edie brickell whom he wed on may 30 1992. | Simon has been married three times; he is currently married to Edie Brickell whom he wed on May 30 1992. |

---

## 🎨 Visualizations

<details>
<summary>🧾 <b>Plots & Interpretations</b></summary>

- **📉 Training Curves** — Both training and validation loss decrease steadily, indicating efficient generalization.    
- **📊 Feature Distributions** — EEG and spectral embeddings cluster semantically similar sentences.  
- **📈 BLEU/ROUGE Trends** — Metrics plotted per epoch show gradual linguistic improvement.

</details>

---

## 🌐 Streamlit Web App

An interactive web demo built with **Streamlit** lets users test the model on pre-saved `.npz` samples.

**Features:**
- Upload EEG + Eye + Spectral data.
- Generate predicted sentence (argmax decoding).
- Display ground truth comparison for up to 5 samples.
- Show BLEU and ROUGE-L scores for each prediction.

**Run locally:**
```bash
streamlit run app.py
