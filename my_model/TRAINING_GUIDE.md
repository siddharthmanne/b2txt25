# Complete Training Guide

### Legacy (Old Trainer)

```bash
python train_complete.py  # Old version without baseline techniques
```

## What Happens During Training

### Data Flow (One Training Step)

```
1. Load batch from dataset
   ├─ brain_data: [64, 544, 512]     # 64 trials, ~544 timesteps, 512 features
   ├─ day_idx: [64]                   # Which recording day each trial is from
   └─ transcriptions: ["I will go around.", "Hello world.", ...]

2. BrainEncoder (TRAINABLE)
   brain_data [64, 544, 512]
      ↓ Day-specific layers
      ↓ GRU (5 layers, 768 units)
      ↓ Projector
   brain_embedding [64, 544, 1280]

3. AudioTarget (FROZEN)
   "I will go around."
      ↓ TTS (facebook/mms-tts-eng)
   audio_waveform [sample_rate * duration]
      ↓ Qwen AudioTower
   audio_embedding [64, T_audio, 1280]
```
### ✅ FIXED: Patching Changes Sequence Length!
```
**IMPORTANT**: With patch_size=14, stride=4, brain embedding has FEWER timesteps (NOW TRACKED!):

3.5 Calculate adjusted length (NEWLY ADDED in train_new.py)
   adjusted_lens = (544 - 14) / 4 + 1 = 133 patches

4. Align sequences
   brain_embedding [64, 133, 1280]   ← 544 timesteps → 133 patches!
                                      ├─ Interpolate to same length
   audio_embedding [64, T_audio, 1280]─┘
      ↓
   brain_aligned [64, T_min, 1280]
   audio_aligned [64, T_min, 1280]
      ↓
   alignment_loss = Cosine(brain_aligned, audio_aligned)

**See [PATCHING_EXPLAINED.md](PATCHING_EXPLAINED.md) for full details.**

5. LLMDecoder (FROZEN)
   brain_embedding [64, 544, 1280]
      ↓ Qwen Projector (1280 → 4096)
   llm_embedding [64, 544, 4096]
      ↓ Concatenate with text prompt embeddings
      ↓ LLM forward pass
   predicted_tokens
      ↓
   llm_loss = CrossEntropy(predicted, true_text)

6. Backpropagation
   total_loss = alignment_loss + llm_loss
      ↓
   Only BrainEncoder parameters update!
   (AudioTarget and LLMDecoder are frozen)
```

## File Responsibilities

### Core Model Files

| File | Purpose | Trainable? |
|------|---------|------------|
| [brain_encoder.py](brain_encoder.py) | GRU + Projector: brain → embedding | ✓ YES |
| [audio_target.py](audio_target.py) | TTS + AudioTower: text → audio embedding | ✗ FROZEN |
| [llm_decoder_new.py](llm_decoder_new.py) | Qwen Projector + LLM: embedding → text | ✗ FROZEN |
| [model_complete.py](model_complete.py) | Combines all, computes joint loss | Container |

### Training Files

| File | Purpose |
|------|---------|
| [train_complete.py](train_complete.py) | Main training script |
| [dataset.py](dataset.py) | Data loading from HDF5 files |

### Documentation

| File | Purpose |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture overview & PyTorch concepts |
| [DATA_ALIGNMENT.md](DATA_ALIGNMENT.md) | Time alignment explanation |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | This file |

## Understanding the Dataset

### HDF5 Structure

Each session directory (e.g., `t15.2023.11.03/`) contains:
- `data_train.hdf5` - Training trials
- `data_val.hdf5` - Validation trials
- `data_test.hdf5` - Test trials

Each file contains trials:
```
trial_0000/
├── input_features: [T, 512]    # Neural activity
├── seq_class_ids: [500]        # Phoneme labels (baseline uses this)
├── transcription: [500]        # Text transcription (YOU use this)
└── attributes:
    ├── n_time_steps: int       # Actual neural data length
    ├── seq_len: int            # Actual phoneme sequence length
    ├── sentence_label: str     # Ground truth text
    ├── block_num: int
    ├── trial_num: int
    └── session: str
```

### Dataset Splits (Following Baseline)

```python
# Training data comes from data_train.hdf5 files
train_file_paths = [
    'data/hdf5_data_final/t15.2023.08.11/data_train.hdf5',
    'data/hdf5_data_final/t15.2023.08.13/data_train.hdf5',
    ...
]

# Validation data comes from data_val.hdf5 files
val_file_paths = [
    'data/hdf5_data_final/t15.2023.08.11/data_val.hdf5',
    'data/hdf5_data_final/t15.2023.08.13/data_val.hdf5',
    ...
]
```

### BrainToTextDataset Behavior

**Key insight**: `__getitem__()` returns an **entire batch**, not a single example!

```python
dataset = BrainToTextDataset(
    trial_indicies=train_trials,
    n_batches=10000,           # How many batches to generate
    split='train',
    batch_size=64,             # Samples per batch
    days_per_batch=4,          # Mix data from 4 different days
    random_seed=42,
)

# When you iterate:
for batch in dataset:
    # batch is already a full batch!
    batch['input_features']   # [64, T, 512]
    batch['day_indicies']     # [64]
    batch['transcriptions']   # [64, 500]
```

That's why DataLoader uses `batch_size=None`.

## Training Parameters

### Current Settings (in training_args.yaml)

**All hyperparameters are now centralized in [training_args.yaml](training_args.yaml)**

Key parameters:

```yaml
# Model architecture (from baseline)
model:
  n_input_features: 512     # Input features (2 per electrode × 256 electrodes)
  n_units: 768              # GRU hidden size
  n_layers: 5               # GRU depth
  rnn_dropout: 0.4          # Dropout in GRU (CRITICAL for regularization)
  patch_size: 14            # Input patching (14 timesteps)
  patch_stride: 4           # Patching stride
  audio_embedding_dim: 1280 # Qwen AudioTower output dim

# Training
dataset:
  batch_size: 64            # Trials per batch
  days_per_batch: 4         # Mix days for robustness
num_training_batches: 120000  # 120k batches (same as baseline)

# Learning rates (with warmup!)
lr_max: 0.005               # Max LR for main model
lr_min: 0.0001              # Min LR after decay
lr_warmup_steps: 1000       # Warmup steps (CRITICAL for stability)
lr_max_day: 0.005           # Separate LR for day-specific layers
lr_min_day: 0.0001

# Loss weights
alpha: 1.0                  # Alignment loss weight
beta: 1.0                   # LLM loss weight

# Data augmentation (NEW!)
dataset:
  data_transforms:
    white_noise_std: 1.0         # White noise augmentation
    constant_offset_std: 0.2     # Constant offset per trial
    random_cut: 3                # Remove 0-3 timesteps from start
    smooth_data: true            # CRITICAL: Gaussian smoothing
    smooth_kernel_std: 2         # Smoothing kernel std dev
```

### Tuning Guidance

**If alignment loss is high but LLM loss is low:**
- Brain embeddings aren't in the right space
- Increase `alpha` to prioritize alignment
- Check that audio embeddings are being created correctly

**If LLM loss is high but alignment loss is low:**
- Brain embeddings are in right space but lack semantic content
- Increase `beta` to prioritize text generation
- May need more training data

**If both losses are high:**
- Model capacity may be insufficient
- Try increasing `n_units` or `n_layers`
- Check learning rate (may be too high/low)

**If training is unstable:**
- Reduce learning rate
- Increase gradient clipping (currently 10.0)
- Reduce batch size

## Common Issues & Solutions

### 0. **CRITICAL: Missing Data Augmentation** (NEW!)

**Problem**: Model overfits to training data, poor generalization across days

**Solution**: The new trainer (`train_new.py`) includes all baseline augmentations:

```python
# Automatically applied in train_new.py
from data_augmentations import apply_data_augmentations

# In training loop (applied ON GPU for speed)
features, n_time_steps = apply_data_augmentations(
    features, n_time_steps, mode='train',
    transform_args=config['dataset']['data_transforms'],
    device=device
)
```

**What it does**:
1. ✅ **Gaussian smoothing** (ALWAYS applied, train & val) - reduces neural noise
2. ✅ **White noise** (std=1.0) - improves robustness
3. ✅ **Constant offset** (std=0.2) - handles baseline drift
4. ✅ **Random cut** (0-3 timesteps) - temporal augmentation

**Impact**: Baseline model uses this extensively. Essential for good performance!

### 1. Sequence Length Mismatch

**Problem**: `RuntimeError: The size of tensor a (544) must match the size of tensor b (312)`

**Solution**: Already handled in `model_complete.py` with `F.interpolate`

### 2. Out of Memory (OOM)

**Problem**: CUDA out of memory

**Solutions**:
```python
# Reduce batch size
batch_size = 32  # Instead of 64

# Use gradient accumulation
accumulation_steps = 2
if batch_idx % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# Reduce model size
n_units = 512  # Instead of 768
```

### 3. Audio Model Not Loading

**Problem**: Hugging Face authentication error

**Solution**: Set your HuggingFace token
```python
# In audio_target.py or llm_decoder_new.py
from huggingface_hub import login
login(token="your_hf_token_here")
```

### 4. Transcriptions Not Decoding

**Problem**: Gibberish text output

**Solution**: Check transcription decoding in training loop
```python
# Correct:
text = bytes(transcriptions_raw[i].cpu().numpy()).decode('utf-8').strip()

# If still broken, inspect raw data:
print(transcriptions_raw[0])  # Should be array of ASCII codes
```

## Monitoring Training

### What to Watch

**Every 100 batches:**
```
Batch 100/10000 | Total Loss: 4.5 | Align Loss: 2.3 | LLM Loss: 2.2
```
- Total loss should decrease
- Both component losses should decrease
- If one loss dominates, adjust `alpha`/`beta`

**Every 1000 batches (validation):**
```
Validation Results: Total Loss: 3.8 | Align Loss: 1.9 | LLM Loss: 1.9
Saved best checkpoint: checkpoints/best_model.pt
```
- Val loss should track train loss
- Large gap = overfitting (need regularization)

### Expected Training Time

**With NEW Optimizations (train_new.py):**

With 1 GPU (e.g., RTX 3090):
- ~1-2 seconds per batch (with torch.compile + AMP)
- 120,000 batches ≈ 33-67 hours (1.5-3 days)

**Without Optimizations (old train_complete.py):**
- ~3-5 seconds per batch
- 10,000 batches ≈ 8-14 hours

Bottlenecks:
1. TTS generation (slow, runs on CPU)
2. Qwen AudioTower forward pass
3. LLM forward pass

**Optimization**: Pre-compute audio embeddings for all training data, save to disk, load during training.

### Performance Improvements from Baseline Techniques

| Technique | Speedup | Memory Savings | Implemented |
|-----------|---------|----------------|-------------|
| torch.compile() | 2-3x | - | ✅ train_new.py |
| AMP (bfloat16) | 1.3-1.5x | 30-50% | ✅ train_new.py |
| Fused AdamW | 1.1-1.2x | - | ✅ train_new.py |
| Data aug on GPU | - | - | ✅ train_new.py |
| **Total** | **~3-4x faster** | **~40% less VRAM** | ✅ |

## After Training

### Generate Text from Brain Activity

```python
from model_complete import Brain2TextModel
import torch

# Load model
model = Brain2TextModel(...)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load some test data
test_batch = ...  # Get from test_loader

# Generate
with torch.no_grad():
    generated_texts = model.generate(
        brain_data=test_batch['input_features'],
        day_idx=test_batch['day_indicies'],
        max_length=50
    )

# Compare to ground truth
ground_truth = [bytes(t).decode('utf-8').strip()
                for t in test_batch['transcriptions']]

for pred, true in zip(generated_texts, ground_truth):
    print(f"Predicted: {pred}")
    print(f"True:      {true}")
    print()
```

### Evaluate Performance

Common metrics for brain-to-text:
- **Word Error Rate (WER)**: Levenshtein distance between predicted and true text
- **Character Error Rate (CER)**: Character-level accuracy
- **BLEU Score**: N-gram overlap

```python
from jiwer import wer

wer_score = wer(ground_truth, generated_texts)
print(f"Word Error Rate: {wer_score:.2%}")
```

## Next Steps

### 1. **Test the New Trainer** (RECOMMENDED FIRST STEP)

```bash
cd /Users/Siddharth/nejm-brain-to-text/my_model

# Quick test run (1000 batches)
# Edit training_args.yaml: set num_training_batches: 1000
python train_new.py
```

**What to verify**:
- ✅ Data augmentation is working (check logs)
- ✅ torch.compile doesn't crash
- ✅ Losses are decreasing
- ✅ Both alignment_loss and llm_loss decrease together
- ✅ ~1-2 seconds per batch (with GPU)

### 2. **Compare with Baseline Techniques**

Your model now has:
- ✅ Same data augmentation as baseline
- ✅ Same optimizer setup (separate param groups)
- ✅ Same LR scheduling (cosine with warmup)
- ✅ Same training duration (120k batches)
- ✅ torch.compile + AMP for speed
- ✅ YAML config for reproducibility

### 3. **Optimize Audio Embedding** (Optional but recommended)

   - Pre-compute and cache audio embeddings
   - Save to HDF5 file
   - Load during training (much faster)

### 4. **Tune Hyperparameters**

Adjust in [training_args.yaml](training_args.yaml):
   - Loss weights (alpha, beta)
   - Learning rates
   - Augmentation strengths
   - Model architecture

### 5. **Try Advanced Alignment**

   - Implement DTW loss instead of MSE/Cosine
   - Try contrastive loss
   - Experiment with different embedding spaces

### 6. **Add LoRA to LLM**

   - Currently LLM is fully frozen
   - Adding LoRA adapters allows light fine-tuning
   - May improve text generation quality

## Questions?

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for PyTorch concepts
- Check [DATA_ALIGNMENT.md](DATA_ALIGNMENT.md) for alignment details
- Inspect baseline model code in `baseline_model/` for reference
