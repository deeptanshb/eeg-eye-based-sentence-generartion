import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import random

# ====== EEG Encoder ======
class EEGEncoder(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.proj(x)
        return x  # (B, T, 256)


# ====== Eye Encoder ======
class EyeEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, x):
        return self.fc(x)  # (B, 64)


# ====== Spectral Encoder ======
class SpectralEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, x):
        return self.fc(x)  # (B, 64)


# ====== EEG2Text Transformer ======
class EEG2TextTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=384, n_heads=6, n_layers=3, max_len=64):
        super().__init__()

        # Encoders
        self.eeg_enc = EEGEncoder()
        self.eye_enc = EyeEncoder()
        self.spec_enc = SpectralEncoder()
        self.hidden_dim = hidden_dim

        # Projections to common hidden_dim
        self.eeg_proj = nn.Linear(256, hidden_dim)
        self.eye_proj = nn.Linear(64, hidden_dim)
        self.spec_proj = nn.Linear(64, hidden_dim)

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # Transformer decoder
        layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True
        )
        self.decoder = TransformerDecoder(layer, num_layers=n_layers)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    # ---- forward pass ----
    def forward(self, eeg, eye, spec, tgt_ids):
        B, seq_len = eeg.size(0), tgt_ids.size(1)
        device = eeg.device

        # encode modalities
        eeg_seq = self.eeg_enc(eeg)              # (B, 256, 256)
        eye_vec = self.eye_enc(eye)              # (B, 64)
        spec_vec = self.spec_enc(spec)           # (B, 64)

        # project to hidden space
        eeg_memory = self.eeg_proj(eeg_seq)      # (B, 256, 384)
        eye_memory = self.eye_proj(eye_vec).unsqueeze(1)   # (B, 1, 384)
        spec_memory = self.spec_proj(spec_vec).unsqueeze(1)# (B, 1, 384)

        # full memory
        memory = torch.cat([eeg_memory, eye_memory, spec_memory], dim=1)  # (B, 258, 384)

        # target embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        tgt_emb = self.token_emb(tgt_ids) + self.pos_emb(positions)

        # causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
        logits = self.output_head(out)
        return logits  # (B, 64, vocab_size)

    # ---- top-k/top-p filtering ----
    def top_k_top_p_filtering(self, logits, top_k=20, top_p=0.9):
   
        top_k = min(top_k, logits.size(-1))
    
        if top_k > 0:
        # Remove all tokens with probability less than the top k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
    
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least top_k tokens
        if top_k > 0:
            sorted_indices_to_remove[..., :top_k] = False
        
        # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, 
                index=sorted_indices, 
                src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')
    
        return logits

    # ---- autoregressive generation ----
    @torch.no_grad()
    def generate(self, eeg, eye, spec, tokenizer, max_len=50):
        device = next(self.parameters()).device
        B = eeg.size(0)
        generated = torch.full((B, 1), tokenizer.cls_token_id, device=device, dtype=torch.long)

        for _ in range(max_len):
            logits = self.forward(eeg, eye, spec, generated)
            next_token_logits = logits[:, -1, :] / 1.1   # ← temperature added
            filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=30, top_p=0.9)
            probs = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        # Decode current text and stop at period
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            if "." in decoded:
                decoded = decoded[:decoded.index(".") + 1]
                break

            if torch.all(next_token == tokenizer.sep_token_id):
                break

        return generated, decoded


