# Google Cloud Platform Deployment Guide
## Deploy Brain-to-Text Training on GCP L4 GPU

**Your Setup:**
- VM Name: **b2txt25-vm**
- VM: g2-standard-4 (4 vCPUs, 16GB RAM, 1x NVIDIA L4 24GB)
- Deep Learning VM with PyTorch/CUDA pre-installed
- GCS Bucket: **b2txt25-data**

---

## Part 1: Upload Data to Google Cloud Storage

### Step 1: Upload Dataset to Your Bucket
```bash
# On your local machine
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Upload your 12GB dataset to b2txt25-data bucket
gsutil -m rsync -r /Users/Siddharth/nejm-brain-to-text/data/hdf5_data_final \
  gs://b2txt25-data/hdf5_data_final

# This will take ~10-20 minutes for 12GB
```

### Step 2: Verify Upload
```bash
# Check data is uploaded
gsutil ls gs://b2txt25-data/hdf5_data_final/

# Should show your session folders:
# gs://b2txt25-data/hdf5_data_final/t15.2023.08.11/
# gs://b2txt25-data/hdf5_data_final/t15.2023.08.13/
# ...
```

---

## Part 2: Create VM Instance via GCP Console

You're creating **b2txt25-vm** via the web console. Here are the settings:

### VM Configuration Checklist:
- ✅ **Name**: b2txt25-vm
- ✅ **Region**: us-central1 (or your preferred region)
- ✅ **Zone**: us-central1-a
- ✅ **Machine type**: g2-standard-4 (4 vCPU, 16 GB memory)
- ✅ **GPU**: 1x NVIDIA L4
- ✅ **Boot disk**:
  - OS: **Deep Learning on Linux**
  - Version: **PyTorch 2.x with CUDA 12.x**
  - Size: **200 GB**
  - Type: **SSD persistent disk**
- ✅ **Access scopes**:
  - Select "Allow full access to all Cloud APIs" (needed for GCS access)
- ✅ **Firewall**: Allow HTTP/HTTPS (if you want TensorBoard access)

### Click "Create" and wait 2-3 minutes for deployment

---

## Part 3: Connect to b2txt25-vm and Setup Environment

### Step 1: SSH into VM
```bash
# Connect to b2txt25-vm
gcloud compute ssh b2txt25-vm --zone=us-central1-a

# Verify GPU is available
nvidia-smi
# Should show: NVIDIA L4 with 24GB memory
```

### Step 2: Upload Your Project Code (EXCLUDE DATA)
```bash
# From your LOCAL machine (new terminal window):
# IMPORTANT: We're uploading code ONLY, not the 12GB data folder
# Data will be accessed via GCS mount instead


# Option B: Upload only the my_model directory
gcloud compute scp --recurse \
  /Users/Siddharth/nejm-brain-to-text/my_model \
  b2txt25-vm:~/nejm-brain-to-text/ \
  --zone=us-central1-a

# This uploads only your code (~few MB)
# Takes < 1 minute (no data transfer needed!)
```

### Step 3: Install Dependencies on VM
```bash
# Back on the VM (SSH session)
cd ~/nejm-brain-to-text

# Verify Python and PyTorch
python3 --version  # Should be 3.10+
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
python3 -c "import torch; print(torch.cuda.get_device_name(0))"  # Should show L4

# Install required packages
pip install -r requirements.txt

# If requirements.txt doesn't exist or is incomplete, install manually:
pip install transformers accelerate datasets
pip install h5py librosa soundfile scipy
pip install omegaconf tensorboard
pip install huggingface_hub
pip install parler-tts  # For TTS models
```

---

## Part 4: Mount Google Cloud Storage Data (NO LOCAL COPY NEEDED)

**Important:** Your data stays in GCS. We mount it like a local filesystem using gcsfuse.

### Install and Mount gcsfuse (Recommended Method)

```bash
# On the VM - Install gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

# Create mount point
mkdir -p ~/data

# Mount b2txt25-data bucket (acts like local directory)
gcsfuse b2txt25-data ~/data

# Verify data is accessible
ls ~/data/hdf5_data_final/
# Should show: t15.2023.08.11  t15.2023.08.13  t15.2023.08.18  ...

# Check a specific session
ls ~/data/hdf5_data_final/t15.2023.08.11/
# Should show: data_train.hdf5  data_val.hdf5
```

**Benefits of gcsfuse:**
- ✅ No disk space used (data stays in GCS)
- ✅ Access data like local files
- ✅ Changes sync to GCS automatically
- ✅ Multiple VMs can access same data

**Performance:** gcsfuse is fast enough for training. PyTorch DataLoader will cache data in RAM during training.

### Optional: Auto-mount on VM reboot
```bash
# Only if you plan to stop/start the VM frequently
echo "b2txt25-data /home/$(whoami)/data gcsfuse rw,allow_other,file_mode=777,dir_mode=777" | sudo tee -a /etc/fstab
```

### Alternative: Copy Data to VM Disk (NOT RECOMMENDED)

⚠️ **Don't do this unless gcsfuse is too slow:**
```bash
# This wastes 12GB of disk space and upload time
# Only use if you have performance issues with gcsfuse
mkdir -p ~/data
gsutil -m rsync -r gs://b2txt25-data/hdf5_data_final ~/data/hdf5_data_final
```

---

## Part 5: Update Config Files for Cloud Deployment

### Get Your VM Username
```bash
# On the VM
whoami
# Outputs: your_username (e.g., siddharth_gmail_com)
```

### Update Dataset Paths in Configs
```bash
cd ~/nejm-brain-to-text/my_model

# Replace local path with VM path
# Replace YOUR_USERNAME with output from `whoami` above
sed -i 's|/Users/Siddharth/nejm-brain-to-text/data/hdf5_data_final|/home/YOUR_USERNAME/data/hdf5_data_final|g' \
  training_args_quick_test.yaml \
  training_args_overfit_test.yaml \
  training_args.yaml

# Or edit manually:
nano training_args_quick_test.yaml
# Change line ~90:
#   dataset_dir: /home/YOUR_USERNAME/data/hdf5_data_final
```

**Example if your username is `siddharth_gmail_com`:**
```yaml
dataset:
  dataset_dir: /home/siddharth_gmail_com/data/hdf5_data_final
```

---

## Part 6: Authenticate HuggingFace (for model downloads)

```bash
# Install huggingface-cli if needed
pip install -U huggingface_hub

# Login (you'll need a HF token from https://huggingface.co/settings/tokens)
huggingface-cli login
# Paste your token when prompted

# Test download works
python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('openai/whisper-base')"
```

---

## Part 7: Run Quick Test (5-10 minutes)

```bash
cd ~/nejm-brain-to-text/my_model

# Run in foreground to see output directly:
python train_new.py training_args_quick_test.yaml

# Or run in background with logging:
nohup python train_new.py training_args_quick_test.yaml > quick_test.log 2>&1 &

# Monitor progress
tail -f quick_test.log
```

### Expected Output:
```
Using device: cuda:0
Initialized Brain2Text model
Brain encoder has 5,123,456 trainable parameters
Successfully initialized datasets
Data shapes: input=[4, 1500, 512] brain_emb=[4, 107, 1280] audio_emb=[4, 150, 1280]
Train batch 0: total_loss: 4.5234 align_loss: 2.1234 llm_loss: 2.4000
...
Saved checkpoint: trained_models/quick_test/checkpoint/best_checkpoint
```

### Verify Success:
```bash
# Check logs
cat trained_models/quick_test/training_log

# Check checkpoint was saved
ls -lh trained_models/quick_test/checkpoint/
```

---

## Part 8: Run Overfitting Test (30-60 min)

```bash
cd ~/nejm-brain-to-text/my_model

# Use tmux for persistent session (survives disconnection)
tmux new -s overfit_test

# Run overfitting test
python train_new.py training_args_overfit_test.yaml

# Detach from tmux: Press Ctrl+B, then press D
# Reattach later: tmux attach -t overfit_test
# Kill session: tmux kill-session -t overfit_test
```

### Monitor Progress:
```bash
# In another SSH session or after detaching tmux:
tail -f ~/nejm-brain-to-text/my_model/trained_models/overfit_test/training_log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Success Criteria:
```
Batch 0: total_loss: ~5.0
Batch 500: total_loss: ~2.5
Batch 1000: total_loss: ~1.5
Batch 2000: total_loss: <1.0  ← Goal!
```

---

## Part 9: Run Full Training (Adjusted for L4)

**⚠️ IMPORTANT: L4 has 24GB VRAM, your config needs ~28-32GB**

### Option 1: Reduce Batch Size for L4
```bash
# Edit training_args.yaml
cd ~/nejm-brain-to-text/my_model
nano training_args.yaml
```

**Changes:**
```yaml
dataset:
  batch_size: 32  # Reduce from 64 to fit in 24GB
  # or even 16 if still OOM
```

### Option 2: Use Gradient Checkpointing (if still OOM)
You may need to add gradient checkpointing in the model code, but try Option 1 first.

### Run Full Training:
```bash
cd ~/nejm-brain-to-text/my_model

# Use tmux for long-running job
tmux new -s full_training

# Run training
python train_new.py training_args.yaml

# Detach: Ctrl+B then D
# Check progress later: tmux attach -t full_training
```

### Monitor Training:
```bash
# Watch logs
tail -f trained_models/brain2text_audio_llm/training_log

# Check GPU usage
watch -n 1 nvidia-smi

# Should show ~20-23GB GPU memory usage with batch_size=32
```

**Estimated Time on L4:** ~15-20 hours (slower than A100)

---

## Part 10: Monitor Training Progress

### Check GPU Utilization:
```bash
# Real-time GPU monitoring
nvidia-smi dmon -s u

# Or detailed view
watch -n 1 nvidia-smi
```

### View Training Metrics:
```bash
# Training log
tail -n 50 trained_models/brain2text_audio_llm/training_log

# Look for:
# - Loss decreasing over time
# - No NaN or Inf values
# - Validation loss improving
```

### Check Disk Space:
```bash
df -h
# Make sure you have enough space for checkpoints
```

---

## Part 11: Download Results After Training

### Option A: Copy to GCS (Recommended - Backup)
```bash
# On the VM, after training completes
gsutil -m rsync -r ~/nejm-brain-to-text/my_model/trained_models \
  gs://b2txt25-data/trained_models

# Verify upload
gsutil ls gs://b2txt25-data/trained_models/
```

### Option B: Download to Local Machine
```bash
# From your LOCAL machine
gcloud compute scp --recurse \
  b2txt25-vm:~/nejm-brain-to-text/my_model/trained_models \
  /Users/Siddharth/nejm-brain-to-text/my_model/ \
  --zone=us-central1-a

# This downloads all checkpoints and logs
```

---

## Part 12: Cost Management

### Stop VM When Not Training
```bash
# From local machine or GCP Console
gcloud compute instances stop b2txt25-vm --zone=us-central1-a

# Charges: $0/hour (only pay for disk storage ~$20/month)

# Start again later:
gcloud compute instances start b2txt25-vm --zone=us-central1-a
```

### Delete VM After Training (Keep Data in GCS)
```bash
# Delete VM
gcloud compute instances delete b2txt25-vm --zone=us-central1-a

# Your data and models are safe in gs://b2txt25-data/
# Recreate VM anytime if needed
```

### Estimated Costs:
- **L4 GPU VM**: ~$0.70-1.00/hour
- **Storage (200GB disk)**: ~$20/month (when VM is stopped)
- **GCS storage (12GB data + models)**: ~$0.50/month
- **Total for full training**: ~$15-20 (15-20 hours)

---

## Quick Command Reference

```bash
# === SETUP (LOCAL MACHINE) ===
# 1. Upload data to b2txt25-data bucket (one-time, ~10-20 min)
gsutil -m rsync -r /Users/Siddharth/nejm-brain-to-text/data/hdf5_data_final \
  gs://b2txt25-data/hdf5_data_final

# 2. Upload code to VM (EXCLUDE data folder!)
gcloud compute scp --recurse \
  --exclude="data/*" \
  /Users/Siddharth/nejm-brain-to-text \
  b2txt25-vm:~/ --zone=us-central1-a

# === SETUP (ON VM) ===
# 3. SSH to b2txt25-vm
gcloud compute ssh b2txt25-vm --zone=us-central1-a

# 4. Mount GCS data on VM (makes data accessible like local files)
gcsfuse b2txt25-data ~/data

# 5. Verify mount worked
ls ~/data/hdf5_data_final/

# === TRAINING ===
# 5. Run quick test
cd ~/nejm-brain-to-text/my_model
python train_new.py training_args_quick_test.yaml

# 6. Run overfitting test
tmux new -s overfit_test
python train_new.py training_args_overfit_test.yaml

# 7. Run full training
tmux new -s full_training
python train_new.py training_args.yaml

# === MONITORING ===
# 8. Watch GPU
nvidia-smi

# 9. Monitor logs
tail -f trained_models/*/training_log

# === BACKUP & DOWNLOAD ===
# 10. Backup to GCS (on VM)
gsutil -m rsync -r ~/nejm-brain-to-text/my_model/trained_models \
  gs://b2txt25-data/trained_models

# 11. Download to local (from local machine)
gcloud compute scp --recurse \
  b2txt25-vm:~/nejm-brain-to-text/my_model/trained_models \
  /Users/Siddharth/nejm-brain-to-text/my_model/ \
  --zone=us-central1-a

# === CLEANUP ===
# 12. Stop VM
gcloud compute instances stop b2txt25-vm --zone=us-central1-a

# 13. Delete VM
gcloud compute instances delete b2txt25-vm --zone=us-central1-a
```

---

## Troubleshooting

### Issue: Out of Memory on L4
```
RuntimeError: CUDA out of memory. Tried to allocate 2.5 GB
```
**Solution:**
```yaml
# Reduce batch_size in training_args.yaml
batch_size: 16  # or even 8
```

### Issue: Data not accessible
```
FileNotFoundError: No such file or directory: '/home/.../data/hdf5_data_final'
```
**Solution:**
```bash
# Check gcsfuse mount
ls ~/data/hdf5_data_final/
# If empty, remount:
fusermount -u ~/data
gcsfuse b2txt25-data ~/data
```

### Issue: Model download fails
```
HTTPError: 401 Client Error: Unauthorized
```
**Solution:**
```bash
# Login to HuggingFace
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### Issue: Training very slow
**Solution:**
```bash
# Verify GPU is being used
nvidia-smi
# Should show python process using GPU

# Check AMP is enabled in config
grep "use_amp" training_args.yaml
# Should be: use_amp: true
```

### Issue: SSH connection lost
**Solution:**
```bash
# Reconnect
gcloud compute ssh b2txt25-vm --zone=us-central1-a

# Reattach to tmux session
tmux attach -t full_training
# Your training should still be running!
```

---

## Next Steps After Successful Deployment

### On Local Machine:
1. ✅ Upload data to `gs://b2txt25-data/hdf5_data_final` (one-time)
2. ✅ Upload code (EXCLUDING data directory) to `b2txt25-vm`

### On b2txt25-vm:
3. ✅ SSH to `b2txt25-vm`
4. ✅ Mount data with gcsfuse (data accessible at ~/data)
5. ✅ Install dependencies
6. ✅ Update config paths to point to ~/data/hdf5_data_final
7. ✅ Run quick test (5-10 min)
8. ✅ Run overfitting test (30-60 min)
9. ✅ If overfit works, run full training (~15-20 hours)
10. ✅ Backup results to GCS
11. ✅ Stop VM to save costs

**Key Point:** Data never leaves GCS. It's mounted read-only on the VM via gcsfuse.
