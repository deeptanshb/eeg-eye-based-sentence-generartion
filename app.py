import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer
from model import EEG2TextTransformer  # your model class
import os
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# -------------------------------------------------------
# 1️⃣ Load model + tokenizer
# -------------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEG2TextTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=384,
        n_heads=6,
        n_layers=3,
        max_len=64
    ).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    return model, tokenizer, device

# -------------------------------------------------------
# 2️⃣ Calculate BLEU Score
# -------------------------------------------------------
def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis sentences
    """
    # Tokenize sentences into words
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # BLEU expects reference as list of lists
    reference_list = [ref_tokens]
    
    # Use smoothing for short sentences
    smooth = SmoothingFunction()
    
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu1 = sentence_bleu(reference_list, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
    bleu2 = sentence_bleu(reference_list, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    bleu3 = sentence_bleu(reference_list, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1)
    bleu4 = sentence_bleu(reference_list, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }

# -------------------------------------------------------
# 3️⃣ Plot EEG Signals
# -------------------------------------------------------
def plot_eeg_signals(eeg_data):
    """
    Plot EEG signals across all channels
    eeg_data shape: (256, 24) - 256 time steps, 24 channels
    """
    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    fig.suptitle('EEG Signals Across 24 Channels', fontsize=16, fontweight='bold')
    
    for i in range(24):
        row = i // 6
        col = i % 6
        ax = axes[row, col]
        
        # Plot the signal
        ax.plot(eeg_data[:, i], linewidth=0.8, color='#2E86AB')
        ax.set_title(f'Ch {i+1}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------
# 4️⃣ Plot Eye Tracking Features
# -------------------------------------------------------
def plot_eye_features(eye_data):
    """
    Plot eye-tracking features
    eye_data shape: (3,) - 3 features
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
    colors = ['#E63946', '#F4A261', '#2A9D8F']
    
    bars = ax.bar(feature_names, eye_data, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, eye_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Eye-Tracking Features', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------
# 5️⃣ Plot Spectral Features
# -------------------------------------------------------
def plot_spectral_features(spec_data):
    """
    Plot spectral features
    spec_data shape: (8,) - 8 frequency bands
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bands = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 
             'Band 5', 'Band 6', 'Band 7', 'Band 8']
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 8))
    
    bars = ax.bar(bands, spec_data, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, spec_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Power', fontsize=12, fontweight='bold')
    ax.set_title('Spectral Features (Frequency Bands)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------
# 6️⃣ Page setup
# -------------------------------------------------------
st.set_page_config(page_title="EEG → Text Generator", layout="wide")
st.title("🧠 EEG → Text Sentence Decoder")
st.write("Generate sentences from EEG, eye-tracking, and spectral features using your trained Transformer model.")

model, tokenizer, device = load_model()

# -------------------------------------------------------
# 7️⃣ File selection or upload
# -------------------------------------------------------
sample_files = [f for f in os.listdir() if f.startswith("sample_") and f.endswith(".npz")]
choice = st.selectbox("📂 Choose a test sample", sample_files)
uploaded = st.file_uploader("Or upload a .npz test file", type=["npz"])

npz_file = uploaded if uploaded is not None else (choice if choice else None)

# -------------------------------------------------------
# 8️⃣ Run inference
# -------------------------------------------------------
if npz_file and st.button("🚀 Generate Text"):
    data = np.load(npz_file)
    eeg = torch.tensor(data["eeg"], dtype=torch.float32).unsqueeze(0).to(device)
    eye = torch.tensor(data["eye"], dtype=torch.float32).unsqueeze(0).to(device)
    spec = torch.tensor(data["spec"], dtype=torch.float32).unsqueeze(0).to(device)
    sentence = str(data["sentence"])
    
    # Store numpy versions for plotting
    eeg_np = data["eeg"]
    eye_np = data["eye"]
    spec_np = data["spec"]
    
    with torch.no_grad():
        # Tokenize the ground truth sentence
        tokenized = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        
        logits = model(eeg, eye, spec, input_ids)
        preds = torch.argmax(logits, dim=-1)[0]
        decoded = tokenizer.decode(preds.cpu().numpy(), skip_special_tokens=True).strip()
        
        if "." in decoded:
            decoded = decoded[:decoded.index(".") + 1]
    
    st.success("✅ Generation Complete!")
    
    # -------------------------------------------------------
    # Display Results
    # -------------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Generated Text")
        st.info(decoded)
    
    with col2:
        st.markdown("### 💬 Ground Truth Sentence")
        st.success(sentence)
    
    # -------------------------------------------------------
    # Calculate and Display BLEU Score
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 BLEU Score Evaluation")
    
    bleu_scores = calculate_bleu(sentence, decoded)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("BLEU-1 (Unigram)", f"{bleu_scores['BLEU-1']:.4f}")
    with col2:
        st.metric("BLEU-2 (Bigram)", f"{bleu_scores['BLEU-2']:.4f}")
    with col3:
        st.metric("BLEU-3 (Trigram)", f"{bleu_scores['BLEU-3']:.4f}")
    with col4:
        st.metric("BLEU-4 (Quadgram)", f"{bleu_scores['BLEU-4']:.4f}")
    
    # -------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📈 Signal Visualizations")
    
    # EEG Signals
    st.markdown("### 🧠 EEG Signals (24 Channels)")
    with st.spinner("Plotting EEG signals..."):
        eeg_fig = plot_eeg_signals(eeg_np)
        st.pyplot(eeg_fig)
    
    # Eye Tracking and Spectral in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👁️ Eye-Tracking Features")
        eye_fig = plot_eye_features(eye_np)
        st.pyplot(eye_fig)
    
    with col2:
        st.markdown("### 🌊 Spectral Features")
        spec_fig = plot_spectral_features(spec_np)
        st.pyplot(spec_fig)
    
    # -------------------------------------------------------
    # Additional Info
    # -------------------------------------------------------
    st.markdown("---")
    with st.expander("ℹ️ Data Shapes and Statistics"):
        st.write(f"**EEG Shape:** {eeg_np.shape}")
        st.write(f"**EEG Range:** [{eeg_np.min():.4f}, {eeg_np.max():.4f}]")
        st.write(f"**EEG Mean:** {eeg_np.mean():.4f}")
        st.write(f"**EEG Std:** {eeg_np.std():.4f}")
        st.write("---")
        st.write(f"**Eye Shape:** {eye_np.shape}")
        st.write(f"**Eye Values:** {eye_np}")
        st.write("---")
        st.write(f"**Spectral Shape:** {spec_np.shape}")
        st.write(f"**Spectral Range:** [{spec_np.min():.4f}, {spec_np.max():.4f}]")
        st.write(f"**Spectral Mean:** {spec_np.mean():.4f}")