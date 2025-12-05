# Data Structure and Time Alignment

## Critical Understanding: Time Alignment in Your Model

### Data Structure from HDF5 Files

Each trial in the dataset contains:

```python
trial = {
    'input_features': [T_brain, 512],      # Neural activity over time
    'seq_class_ids': [500],                 # Phoneme sequence (padded)
    'transcription': [500],                 # Character-level text (padded)
    'n_time_steps': int,                    # Actual length of neural data (e.g., 544)
    'seq_len': int,                         # Actual length of phoneme sequence (e.g., 14)
    'sentence_label': str,                  # Ground truth text (e.g., "I will go around.")
}
```

### Example from Data:
```
input_features shape: (544, 512)  → 544 time steps of brain activity
transcription: "I will go around." (character-level, padded to 500)
n_time_steps: 544
seq_len: 14 (phoneme sequence length)
```

---

## How Time Alignment Works in Your Model

### The Challenge

**Brain data** is time-series with variable length (e.g., 544 time steps)
**Text labels** are discrete symbols without explicit time alignment

You need to align:
- Brain activity at time `t` → What was being spoken at time `t`

### Your Two-Stage Approach Solution

#### Stage 1: Brain → Audio Embedding Alignment

```
Trial at time t:

Brain pathway (TRAINABLE):
  brain_data[t] → GRU → Projector → brain_embedding[t]
  Shape: [batch, T_brain, 512] → [batch, T_brain, 1280]

Audio pathway (FROZEN - creates target):
  text_label → TTS → audio_array → Qwen AudioTower → audio_embedding[t]
  "I will go around." → audio waveform → [batch, T_audio, 1280]
```

**Key Point**: The TTS model converts text to time-aligned audio, which is then embedded. This creates a temporal target that naturally aligns with brain activity over time.

**Loss 1**: `MSE(brain_embedding[t], audio_embedding[t])`
- Both are time-series embeddings
- Forces brain encoder to learn temporal patterns that match speech audio

#### Stage 2: LLM Decoding

```
aligned_brain_embedding[t] → Qwen Projector → LLM → text_output
  [batch, T_brain, 1280] → [batch, T_brain, 4096] → "I will go around."
```

**Loss 2**: Cross-entropy between predicted text and ground truth
- This is sequence-to-sequence loss
- Brain embeddings act as "soft tokens" for the LLM

---

## Why This Works (Implicit Time Alignment)

### The TTS Bridge

Text-to-Speech (TTS) provides implicit time alignment:

```
Text: "I will go around."
         ↓ TTS (facebook/mms-tts-eng)
Audio: [0.0s----0.2s----0.4s----0.6s----0.8s----1.0s]
        I    will   go    a   round

         ↓ Qwen AudioTower
audio_embedding[t]: [T_audio, 1280]
  t=0:   embedding for "I"
  t=10:  embedding for "will"
  t=20:  embedding for "go"
  ...
```

When brain activity happens:
```
brain_data[t]: [T_brain, 512]
  t=0-100:   neural activity during "I"
  t=100-200: neural activity during "will"
  t=200-300: neural activity during "go"
  ...
```

The alignment loss forces `brain_embedding[t]` to match `audio_embedding[t]` at corresponding times.

---

## Sequence Length Handling

### Potential Mismatch Problem

```
brain_embedding: [batch, T_brain=544, 1280]
audio_embedding: [batch, T_audio=????, 1280]
```

The audio sequence length depends on:
- Text length
- TTS speaking rate
- Audio sample rate
- AudioTower downsampling

**This is a problem you need to handle!**

### Solutions

**Option 1: Padding/Truncation**
```python
# In model_complete.py, modify forward():
max_len = max(brain_embedding.shape[1], audio_embedding.shape[1])

# Pad shorter sequence
if brain_embedding.shape[1] < max_len:
    brain_embedding = F.pad(brain_embedding, (0, 0, 0, max_len - brain_embedding.shape[1]))
if audio_embedding.shape[1] < max_len:
    audio_embedding = F.pad(audio_embedding, (0, 0, 0, max_len - audio_embedding.shape[1]))
```

**Option 2: Temporal Pooling**
```python
# Pool to same length
def pool_to_length(tensor, target_len):
    # [batch, T, dim] → [batch, target_len, dim]
    return F.interpolate(
        tensor.transpose(1, 2),  # [batch, dim, T]
        size=target_len,
        mode='linear'
    ).transpose(1, 2)

audio_embedding = pool_to_length(audio_embedding, brain_embedding.shape[1])
```

**Option 3: Dynamic Time Warping (DTW)**
```python
# Align sequences using DTW before computing loss
from dtw import dtw
alignment_loss = dtw_aligned_loss(brain_embedding, audio_embedding)
```

---

## Baseline Model Approach (CTC Loss)

The baseline model uses **CTC Loss** which handles alignment differently:

```python
# Baseline uses phoneme labels, not text directly
logits = model(brain_data, day_idx)  # [batch, T, n_phonemes]
loss = CTCLoss(
    log_probs=logits,
    targets=phoneme_labels,      # [batch, seq_len]
    input_lengths=n_time_steps,  # Actual brain sequence lengths
    target_lengths=phone_seq_lens # Actual phoneme sequence lengths
)
```

**CTC Loss** automatically learns alignment between:
- Input sequence (brain activity over time)
- Output sequence (phoneme labels)

**Your approach is different**: You use the audio embedding as an intermediate representation that has explicit temporal structure.

---

## Implementation Checklist

### ✓ What's Already Handled

1. **Dataset loads time-series brain data correctly**
   - `batch['input_features']`: [batch, time, 512]
   - Padding handled by dataset

2. **BrainEncoder preserves temporal structure**
   - GRU processes sequences: [batch, T, 512] → [batch, T, 1280]
   - Time dimension maintained

3. **Training loop extracts transcriptions correctly**
   - Decodes bytes to strings
   - Passes to AudioTarget for TTS → embedding

### ⚠️ What You Need to Fix

1. **Sequence length mismatch in alignment loss**
   - Currently in `model_complete.py:103`:
   ```python
   alignment_loss = self.alignment_loss_fn(brain_embedding, audio_embedding)
   ```
   - This will fail if `brain_embedding.shape[1] != audio_embedding.shape[1]`

2. **AudioTarget batch processing**
   - Currently in `audio_target.py:102-111`:
   ```python
   for text in text_labels:
       # Processes one at a time
   ```
   - Need to ensure all outputs have same sequence length for batching

### Recommended Fix

Update `model_complete.py` to handle length mismatch:

```python
def forward(self, brain_data, day_idx, target_texts):
    # Get embeddings
    brain_embedding = self.brain_encoder(brain_data, day_idx)
    audio_embedding = self.audio_target(target_texts)

    # Handle length mismatch - use interpolation
    if brain_embedding.shape[1] != audio_embedding.shape[1]:
        target_len = min(brain_embedding.shape[1], audio_embedding.shape[1])

        brain_embedding = F.interpolate(
            brain_embedding.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        audio_embedding = F.interpolate(
            audio_embedding.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

    # Now compute loss with aligned sequences
    alignment_loss = self.alignment_loss_fn(brain_embedding, audio_embedding)

    # Rest of forward pass...
```

---

## Summary

**Your model achieves time alignment through:**

1. **TTS creates temporal structure**: Text → Time-aligned audio
2. **AudioTower creates temporal embeddings**: Audio → audio_embedding[t]
3. **BrainEncoder learns to match**: brain_data[t] → brain_embedding[t] ≈ audio_embedding[t]
4. **LLM decodes**: aligned_brain_embedding → text

The critical insight is that the **audio embedding is already time-aligned** because it comes from time-series audio, which naturally corresponds to the time-series brain activity.
