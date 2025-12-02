import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json

# MEMORY OPTIMIZATION: Enable expandable segments to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import apply_data_augmentations
from model_complete import Brain2TextModel

from omegaconf import OmegaConf
import editdistance  # For computing WER (same as baseline)

torch.set_float32_matmul_precision('high')  # faster matmuls on some GPUs
torch.backends.cudnn.deterministic = True  # reproducible training
torch._dynamo.config.cache_size_limit = 64


class Brain2TextTrainer:
    """
    Trainer for brain-to-text model with audio-LLM alignment.
    Based on baseline RNN trainer with adaptations for dual-loss training.
    """

    def __init__(self, args):
        self.args = args
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.use_multi_gpu = False
        self.n_gpus = 0

        self.best_val_loss = torch.inf
        self.best_val_alignment_loss = torch.inf
        self.best_val_llm_loss = torch.inf
        self.best_val_wer = torch.inf  # Track best WER for checkpointing

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.transform_args = self.args['dataset']['data_transforms']

        # Effective batch size = batch_size × gradient_accumulation_steps
        self.gradient_accumulation_steps = 4

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=True)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']:
            os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')

        if args['mode'] == 'train':
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'], 'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        #Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device and multi-GPU setup
        if not torch.cuda.is_available():
            self.logger.error("No GPU available. This training requires CUDA. Exiting.")
            sys.exit(1)

        self.n_gpus = torch.cuda.device_count()
        self.logger.info(f'Found {self.n_gpus} GPU(s) available')

        # Check if we should use multiple GPUs
        use_multi_gpu = self.args.get('use_multi_gpu', False)

        if use_multi_gpu and self.n_gpus > 1:
            self.use_multi_gpu = True
            self.device = torch.device("cuda:0")  # Primary device
            self.logger.info(f'Using DataParallel across {self.n_gpus} GPUs')
            self.logger.info(f'GPU devices: {[torch.cuda.get_device_name(i) for i in range(self.n_gpus)]}')
        else:
            # Single GPU mode
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number: {gpu_num}. Using 0.")
                gpu_num = 0

            max_gpu_index = self.n_gpus - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"GPU {gpu_num} not available. Using GPU 0.")
                gpu_num = 0

            self.device = torch.device(f"cuda:{gpu_num}")
            self.logger.info(f'Using single GPU: {self.device} ({torch.cuda.get_device_name(gpu_num)})')

        # Set seed
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize model
        self.model = Brain2TextModel(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_days=len(self.args['dataset']['sessions']),
            audio_embedding_dim=self.args['model']['audio_embedding_dim'],
            rnn_dropout=self.args['model']['rnn_dropout'],
            input_dropout=self.args['model']['input_network']['input_layer_dropout'],
            n_layers=self.args['model']['n_layers'],
            patch_size=self.args['model']['patch_size'],
            patch_stride=self.args['model']['patch_stride'],
            a2t_model_id=self.args['model']['a2t_model_id'],
            device=self.device,
            use_quantization=self.args['model'].get('use_quantization', False),
            quantization_bits=self.args['model'].get('quantization_bits', 8),
            alpha=self.args['alpha'],
            beta=self.args['beta'],
            cache_dir=self.args.get('cache_dir', 'cache/audio_embeddings'),
            logger=self.logger,
        )

        self.logger.info(f"Initialized Brain2Text model")

        # Log parameter counts (access brain_encoder directly, before DataParallel wrapping)
        total_params = sum(p.numel() for p in self.model.brain_encoder.parameters())
        self.logger.info(f"Brain encoder has {total_params:,} trainable parameters")

        day_params = sum(p.numel() for name, p in self.model.brain_encoder.named_parameters() if 'day' in name)
        self.logger.info(f"Day-specific parameters: {day_params:,} ({(day_params/total_params)*100:.2f}%)")

        # Create datasets
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_train.hdf5') 
                           for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_val.hdf5') 
                         for s in self.args['dataset']['sessions']]

        # Ensure that there are no duplicate days
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        # Split trials into train and val sets
        train_trials, _ = train_test_split_indicies(
            file_paths=train_file_paths,
            test_percentage=0,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=None,
        )

        _, val_trials = train_test_split_indicies(
            file_paths=val_file_paths,
            test_percentage=1,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=None,
        )

        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f:
            json.dump({'train': train_trials, 'val': val_trials}, f)

        # Determine if only a subset of neural features should be used
        feature_subset = self.args['dataset'].get('feature_subset', None)
        if feature_subset is not None:
            self.logger.info(f'Using feature subset: {feature_subset}')

        # Train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies=train_trials,
            split='train',
            days_per_batch=self.args['dataset']['days_per_batch'],
            n_batches=self.args['num_training_batches'],
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None, # Dataset.__getitem__() already returns batches
            shuffle=self.args['dataset']['loader_shuffle'],
            num_workers=self.args['dataset']['num_dataloader_workers'],
            pin_memory=True
        )

        # Val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split='test',
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None, # Dataset.__getitem__() already returns batches
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer with separate param groups
        self.optimizer = self.create_optimizer()

        # Learning rate scheduler
        # IMPORTANT: Adjust scheduler steps for gradient accumulation
        # With gradient accumulation, optimizer steps = num_batches / accumulation_steps
        # So scheduler should also decay over fewer steps
        effective_lr_decay_steps = self.args['lr_decay_steps'] // self.gradient_accumulation_steps
        effective_lr_decay_steps_day = self.args['lr_decay_steps_day'] // self.gradient_accumulation_steps
        effective_lr_warmup_steps = self.args['lr_warmup_steps'] // self.gradient_accumulation_steps
        effective_lr_warmup_steps_day = self.args['lr_warmup_steps_day'] // self.gradient_accumulation_steps

        self.logger.info(f"Adjusting LR scheduler for gradient accumulation:")
        self.logger.info(f"  LR decay steps: {self.args['lr_decay_steps']} → {effective_lr_decay_steps}")
        self.logger.info(f"  LR warmup steps: {self.args['lr_warmup_steps']} → {effective_lr_warmup_steps}")

        if self.args['lr_scheduler_type'] == 'cosine':
            # Store effective steps for cosine scheduler
            self.effective_lr_params = {
                'lr_decay_steps': effective_lr_decay_steps,
                'lr_decay_steps_day': effective_lr_decay_steps_day,
                'lr_warmup_steps': effective_lr_warmup_steps,
                'lr_warmup_steps_day': effective_lr_warmup_steps_day,
            }
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        elif self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=effective_lr_decay_steps,
            )
        else:
            raise ValueError(f"Invalid lr_scheduler_type: {self.args['lr_scheduler_type']}")

        # Load checkpoint if specified
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Send model to device and wrap with DataParallel if using multi-GPU
        self.model.to(self.device)

        if self.use_multi_gpu:
            self.logger.info("Wrapping model with DataParallel for multi-GPU training")
            # Only wrap the trainable brain_encoder, not the frozen LLM components
            self.model.brain_encoder = nn.DataParallel(self.model.brain_encoder)
            # Effective batch size increases by n_gpus
            effective_batch_size = self.args['dataset']['batch_size'] * self.n_gpus
            self.logger.info(f"Effective batch size with {self.n_gpus} GPUs: {effective_batch_size}")

    def create_optimizer(self):
        """
        Create optimizer with special parameter groups:
        - Biases: no weight decay
        - Day-specific layers: separate LR, no weight decay
        - Other parameters: standard LR and weight decay
        """
        bias_params = []
        day_params = []
        other_params = []

        # Get brain_encoder parameters (may be wrapped in DataParallel)
        brain_encoder = self.model.brain_encoder.module if self.use_multi_gpu else self.model.brain_encoder

        for name, p in brain_encoder.named_parameters():
            if 'bias' in name:
                bias_params.append(p)
            elif 'day_' in name or 'day_weights' in name or 'day_biases' in name:
                day_params.append(p)
            else:
                other_params.append(p)

        if len(day_params) > 0:
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias'},
                {'params': day_params, 'lr': self.args['lr_max_day'], 
                 'weight_decay': self.args['weight_decay_day'], 'group_type': 'day_layer'},
                {'params': other_params, 'group_type': 'other'}
            ]
        else:
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias'},
                {'params': other_params, 'group_type': 'other'}
            ]

        optim = torch.optim.SGD(
            param_groups,
            lr=self.args['lr_max'],
            momentum=0.9,
            weight_decay=self.args['weight_decay'],
            nesterov=True,
        )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        """Create cosine LR scheduler with warmup for each param group."""
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_max_day = self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']

        # Use effective steps that account for gradient accumulation
        if hasattr(self, 'effective_lr_params'):
            lr_decay_steps = self.effective_lr_params['lr_decay_steps']
            lr_decay_steps_day = self.effective_lr_params['lr_decay_steps_day']
            lr_warmup_steps = self.effective_lr_params['lr_warmup_steps']
            lr_warmup_steps_day = self.effective_lr_params['lr_warmup_steps_day']
        else:
            # Fallback to original values (e.g., when loading from checkpoint)
            lr_decay_steps = self.args['lr_decay_steps']
            lr_decay_steps_day = self.args['lr_decay_steps_day']
            lr_warmup_steps = self.args['lr_warmup_steps']
            lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            # After decay
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),  # biases
                lambda step: lr_lambda(step, lr_min_day / lr_max_day, lr_decay_steps_day, lr_warmup_steps_day),  # day params
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),  # other
            ]
        else:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),  # biases
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),  # other
            ]

        return LambdaLR(optim, lr_lambdas, -1)

    def compute_wer(self, predicted_texts, target_texts):
        """
        Compute Word Error Rate (WER) using edit distance.

        WER = (total_edit_distance) / (total_words_in_target)

        Edit distance counts insertions, deletions, and substitutions at word level.
        Same approach as baseline model (but at word level instead of phoneme level).

        Args:
            predicted_texts: list of str - generated text predictions
            target_texts: list of str - ground truth transcriptions

        Returns:
            wer: float - word error rate (0.0 = perfect, 1.0 = completely wrong)
            total_edit_distance: int - total edit distance across all samples
            total_words: int - total words in all target texts
        """
        total_edit_distance = 0
        total_words = 0

        for pred, target in zip(predicted_texts, target_texts):
            # Split into words (lowercase, strip whitespace)
            pred_words = pred.lower().strip().split()
            target_words = target.lower().strip().split()

            # Compute word-level edit distance using same library as baseline
            ed = editdistance.eval(target_words, pred_words)

            total_edit_distance += ed
            total_words += len(target_words)

        # Avoid division by zero
        if total_words == 0:
            return 0.0, 0, 0

        wer = total_edit_distance / total_words
        return wer, total_edit_distance, total_words

    def load_model_checkpoint(self, load_path):
        """
        Load training checkpoint - supports both new format (brain_encoder only) and old format (full model).
        """
        checkpoint = torch.load(load_path, weights_only=False)

        # Check if this is new format (brain_encoder_state_dict) or old format (model_state_dict)
        if 'brain_encoder_state_dict' in checkpoint:
            # NEW FORMAT: Only brain_encoder saved (~180MB)
            brain_encoder_state = checkpoint['brain_encoder_state_dict']

            # Load into brain_encoder (handle DataParallel wrapper)
            if self.use_multi_gpu:
                self.model.brain_encoder.module.load_state_dict(brain_encoder_state)
            else:
                self.model.brain_encoder.load_state_dict(brain_encoder_state)

            self.logger.info("Loaded checkpoint (new format: brain_encoder only)")

        elif 'model_state_dict' in checkpoint:
            # OLD FORMAT: Full model saved (~10GB) - extract only brain_encoder
            state_dict = checkpoint['model_state_dict']

            # Extract only brain_encoder parameters
            from collections import OrderedDict
            brain_encoder_state = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('brain_encoder.'):
                    # Remove 'brain_encoder.' prefix and handle module wrapper
                    if k.startswith('brain_encoder.module.'):
                        if self.use_multi_gpu:
                            brain_encoder_state[k.replace('brain_encoder.', '')] = v
                        else:
                            brain_encoder_state[k.replace('brain_encoder.module.', '')] = v
                    else:
                        if self.use_multi_gpu:
                            brain_encoder_state['module.' + k.replace('brain_encoder.', '')] = v
                        else:
                            brain_encoder_state[k.replace('brain_encoder.', '')] = v

            # Load extracted brain_encoder state
            if self.use_multi_gpu:
                self.model.brain_encoder.module.load_state_dict(brain_encoder_state)
            else:
                self.model.brain_encoder.load_state_dict(brain_encoder_state)

            self.logger.info("Loaded checkpoint (old format: extracted brain_encoder from full model)")

        else:
            raise ValueError("Checkpoint format not recognized. Expected 'brain_encoder_state_dict' or 'model_state_dict'")

        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('val_loss', torch.inf)
        self.model.to(self.device)

        # Send optimizer params to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info(f"Loaded checkpoint: {load_path}")

    def save_model_checkpoint(self, save_path, val_loss):
        """
        Save training checkpoint - ONLY TRAINABLE PARAMETERS.

        IMPORTANT: Only saves brain_encoder (~180MB) instead of entire model (~10GB).
        The frozen LLM and projector can be reloaded from HuggingFace during inference.
        """
        # Get brain_encoder state dict (handles DataParallel wrapper)
        if self.use_multi_gpu:
            brain_encoder_state = self.model.brain_encoder.module.state_dict()
        else:
            brain_encoder_state = self.model.brain_encoder.state_dict()

        checkpoint = {
            'brain_encoder_state_dict': brain_encoder_state,  # ONLY trainable params (~180MB)
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_loss': val_loss,
            # Save model config for reconstruction
            'alpha': self.model.alpha,
            'beta': self.model.beta,
        }
        torch.save(checkpoint, save_path)

        # Calculate and log checkpoint size
        checkpoint_size_mb = os.path.getsize(save_path) / (1024**2)
        self.logger.info(f"Saved checkpoint: {save_path} ({checkpoint_size_mb:.1f} MB)")

        # Save args alongside checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def train(self):
        """
        Main training loop with memory optimizations.

        MEMORY OPTIMIZATION STRATEGIES IMPLEMENTED:
        ==========================================
        1. Mixed Precision Training (AMP with bfloat16):
           - Reduces memory usage by ~50% compared to fp32
           - Uses torch.autocast for automatic mixed precision
           - bfloat16 provides better numerical stability than fp16

        2. Gradient Accumulation:
           - Accumulates gradients over multiple batches before updating weights
           - Simulates larger batch sizes without increasing memory
           - Effective batch size = batch_size × gradient_accumulation_steps
           - Reduces memory by processing smaller batches individually

        3. torch.compile:
           - Compiles the brain_encoder for 2-3x speedup
           - Uses mode='reduce-overhead' to optimize for memory efficiency
           - Reduces overhead and optimizes computation graphs

        4. Efficient Memory Management:
           - optimizer.zero_grad(set_to_none=True): Faster and more memory efficient
           - non_blocking=True in .to(device): Async GPU transfers
           - Periodic torch.cuda.empty_cache(): Prevents memory fragmentation

        5. Model-Specific Optimizations:
           - 4-bit quantization of LLM (Qwen2-Audio) when enabled
           - Only brain_encoder is trainable (frozen LLM reduces memory)
           - DataParallel for multi-GPU training (when enabled)

        Memory Savings Estimate:
        - AMP (bfloat16): ~50% reduction vs fp32
        - Gradient accumulation: Allows smaller per-batch memory
        - Quantization (4-bit): ~75% reduction for LLM weights
        - Combined: Can reduce from ~40GB to ~15GB VRAM usage
        """
        self.model.train()

        train_losses = []
        val_losses = []
        val_results = []
        val_steps_since_improvement = 0

        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)
        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            self.model.train()

            # MEMORY OPTIMIZATION: Only zero gradients at the start of accumulation cycle. Otherwise, gradient accumulation adds the gradients in a specific cycle.
            # set_to_none=True is faster and more memory efficient than setting to 0
            if i % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            start_time = time.time()

            # ===== Load batch data from dataset =====
            # MEMORY OPTIMIZATION: Use non_blocking=True for async data transfer
            # This allows computation to overlap with data transfer
            features = batch['input_features'].to(self.device, non_blocking=True)  # [batch, time, neural_dim=512]
            n_time_steps = batch['n_time_steps'].to(self.device, non_blocking=True)  # [batch] - actual timesteps per trial
            day_indicies = batch['day_indicies'].to(self.device, non_blocking=True)  # [batch] - day index for each trial

            # Decode transcriptions from byte strings (CPU operation - required for string decoding)
            transcriptions_raw = batch['transcriptions']  # [batch, max_text_len] - byte-encoded strings
            # OPTIMIZED: Vectorized decoding - list comprehension is faster than explicit loop
            # No .cpu() needed - transcriptions already on CPU from DataLoader
            # Decode using chr() to match precompute_all_embeddings.py
            target_texts = [
                ''.join([chr(c) for c in t.numpy() if c != 0]).strip()
                for t in transcriptions_raw
            ]  # List of batch_size strings, e.g., ["hello world", "foo bar", ...]

            # Use autocast for efficiency (AMP with bfloat16)
            with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):
                # Apply data augmentations ON GPU
                features, n_time_steps = apply_data_augmentations(
                    features, n_time_steps, mode='train',
                    transform_args=self.transform_args, device=self.device
                )
                # features: [batch, time, neural_dim=512] - augmented
                # n_time_steps: [batch] - may be adjusted by random_cut

                # ===== FORWARD PASS =====
                # Pipeline:
                # 1. features [batch, time, 512] → brain_encoder → brain_emb [batch, seq_len, 1280]
                # 2. target_texts → TTS → AudioTower → audio_emb [batch, seq_len_audio, 1280]
                # 3. Stage 1 loss: align brain_emb ↔ audio_emb
                # 4. brain_emb → projector [3584] → LLM → text prediction
                # 5. Stage 2 loss: LLM cross-entropy on target_texts
                total_loss, alignment_loss, llm_loss, brain_emb, audio_emb = self.model(
                    features,  # [batch, time, 512]
                    day_indicies,  # [batch]
                    target_texts  # list of batch_size strings
                )
                # total_loss: scalar - α * alignment_loss + β * llm_loss
                # alignment_loss: scalar - MSE between brain and audio embeddings
                # llm_loss: scalar - cross-entropy for text generation
                # brain_emb: [batch, seq_len, 1280] - brain embeddings in audio space
                # audio_emb: [batch, seq_len_audio, 1280] - target audio embeddings

            # ===== BACKWARD PASS =====
            # Gradients flow:
            # 1. LLM loss → (frozen projector) → (frozen LLM) BUT gradients flow back to brain_emb
            # 2. Alignment loss → brain_emb
            # 3. Both losses → brain_encoder parameters (TRAINABLE)

            # MEMORY OPTIMIZATION: Scale loss by accumulation steps for correct gradient magnitude
            # This ensures gradients are averaged across accumulation steps
            scaled_loss = total_loss / self.gradient_accumulation_steps
            scaled_loss.backward()

            # ===== OPTIMIZER STEP (only every N accumulation steps) =====
            # This reduces memory by accumulating gradients before updating weights
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (prevent exploding gradients)
                if self.args['grad_norm_clip_value'] > 0:
                    # Get brain_encoder parameters (handle DataParallel wrapper)
                    brain_encoder = self.model.brain_encoder.module if self.use_multi_gpu else self.model.brain_encoder
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        brain_encoder.parameters(),  # Only brain_encoder has gradients
                        max_norm=self.args['grad_norm_clip_value'],
                        error_if_nonfinite=True,
                        foreach=True
                    )

                # Update weights and learning rate
                self.optimizer.step()  # Update brain_encoder parameters
                self.learning_rate_scheduler.step()  # Update learning rates
            else:
                # For logging purposes when not updating
                if self.args['grad_norm_clip_value'] > 0:
                    brain_encoder = self.model.brain_encoder.module if self.use_multi_gpu else self.model.brain_encoder
                    params = brain_encoder.parameters()
                    grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params if p.grad is not None))

            train_step_duration = time.time() - start_time
            train_losses.append(total_loss.detach().item())

            # Log training progress
            if i % self.args['batches_per_train_log'] == 0:
                # Log shapes for debugging (only first time)
                if i == 0:
                    self.logger.info(
                        f'Data shapes: input={list(features.shape)} '
                        f'brain_emb={list(brain_emb.shape)} '
                        f'audio_emb={list(audio_emb.shape)}'
                    )
                    if self.args['model']['patch_size'] > 0:
                        self.logger.info(
                            f'Patching: {features.shape[1]} timesteps → {brain_emb.shape[1]} patches'
                        )

                is_update_step = (i + 1) % self.gradient_accumulation_steps == 0
                step_type = "UPDATE" if is_update_step else f"ACCUM({(i % self.gradient_accumulation_steps) + 1}/{self.gradient_accumulation_steps})"

                self.logger.info(
                    f'Train batch {i} [{step_type}]: '
                    f'total_loss: {total_loss.item():.4f} '
                    f'align_loss: {alignment_loss.item():.4f} '
                    f'llm_loss: {llm_loss.item():.4f} '
                    f'grad_norm: {grad_norm:.2f} '
                    f'time: {train_step_duration:.3f}s'
                )

            # MEMORY OPTIMIZATION: Periodically clear CUDA cache to prevent fragmentation
            # This helps maintain consistent memory usage over long training runs
            if i > 0 and i % 1000 == 0:
                torch.cuda.empty_cache()

            # Validation step
            if i % self.args['batches_per_val_step'] == 0 or i == (self.args['num_training_batches'] - 1):
                if i > 0:  # Skip validation at step 0
                    self.logger.info(f"Running validation after batch {i}")
                    start_time = time.time()
                    val_metrics = self.validation()
                    val_step_duration = time.time() - start_time

                    self.logger.info(
                        f'Val batch {i}: '
                        f'total_loss: {val_metrics["avg_total_loss"]:.4f} '
                        f'align_loss: {val_metrics["avg_alignment_loss"]:.4f} '
                        f'llm_loss: {val_metrics["avg_llm_loss"]:.4f} '
                        f'WER: {val_metrics["wer"]:.4f} ({val_metrics["total_edit_distance"]}/{val_metrics["total_words"]}) '
                        f'time: {val_step_duration:.3f}s'
                    )

                    val_losses.append(val_metrics['avg_total_loss'])
                    val_results.append(val_metrics)

                    # Check for improvement (prioritize WER, then loss as tiebreaker)
                    new_best = False
                    if val_metrics['wer'] < self.best_val_wer:
                        self.logger.info(
                            f"New best WER: {self.best_val_wer:.4f} -> {val_metrics['wer']:.4f}"
                        )
                        self.best_val_wer = val_metrics['wer']
                        self.best_val_loss = val_metrics['avg_total_loss']
                        new_best = True
                    elif val_metrics['wer'] == self.best_val_wer and val_metrics['avg_total_loss'] < self.best_val_loss:
                        self.logger.info(
                            f"Same WER but better loss: {self.best_val_loss:.4f} -> {val_metrics['avg_total_loss']:.4f}"
                        )
                        self.best_val_loss = val_metrics['avg_total_loss']
                        new_best = True

                    if new_best:
                        if save_best_checkpoint:
                            self.logger.info("Checkpointing best model")
                            self.save_model_checkpoint(
                                f'{self.args["checkpoint_dir"]}/best_checkpoint',
                                self.best_val_loss
                            )
                        val_steps_since_improvement = 0
                    else:
                        val_steps_since_improvement += 1

                    # Early stopping
                    if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                        self.logger.info(
                            f'No improvement for {early_stopping_val_steps} val steps. '
                            f'Stopping early at batch {i}'
                        )
                        break

            # Periodic checkpoint
            if i > 0 and i % 5000 == 0:
                if self.args.get('save_all_val_steps', False):
                    self.save_model_checkpoint(
                        f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}',
                        val_losses[-1] if val_losses else float('inf')
                    )

        # Training complete
        training_duration = time.time() - train_start_time
        self.logger.info(f'Best val WER: {self.best_val_wer:.5f}')
        self.logger.info(f'Best val loss: {self.best_val_loss:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model
        if self.args['save_final_model']:
            self.save_model_checkpoint(
                f'{self.args["checkpoint_dir"]}/final_checkpoint',
                val_losses[-1] if val_losses else float('inf')
            )

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_results': val_results,
        }

    def validation(self):
        """
        Run validation on held-out data.

        Evaluates model performance without gradients.
        Computes both loss metrics and WER by generating text predictions.

        Same forward pass as training but:
        - No augmentation (except smoothing which is critical for neural data)
        - No gradient computation (torch.no_grad)
        - Model in eval mode (affects dropout, batchnorm if present)
        """
        self.model.eval()

        metrics = {
            'total_losses': [],
            'alignment_losses': [],
            'llm_losses': [],
            'day_indicies': [],
            'predicted_texts': [],  # Store generated text for WER
            'target_texts': [],     # Store ground truth for WER
        }

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch in self.val_loader:
                # ===== Load batch data =====
                # MEMORY OPTIMIZATION: Use non_blocking=True for async data transfer
                features = batch['input_features'].to(self.device, non_blocking=True)  # [batch, time, neural_dim=512]
                n_time_steps = batch['n_time_steps'].to(self.device, non_blocking=True)  # [batch] - actual timesteps
                day_indicies = batch['day_indicies'].to(self.device, non_blocking=True)  # [batch] - day indices

                # Decode transcriptions from byte strings (CPU operation - required for string decoding)
                transcriptions_raw = batch['transcriptions']  # [batch, max_text_len]
                # OPTIMIZED: Vectorized decoding - list comprehension is faster than explicit loop
                # Decode using chr() to match precompute_all_embeddings.py
                target_texts = [
                    ''.join([chr(c) for c in t.numpy() if c != 0]).strip()
                    for t in transcriptions_raw
                ]  # List of batch_size strings

                # Apply transforms
                with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):
                    features, n_time_steps = apply_data_augmentations(
                        features, n_time_steps, mode='val',
                        transform_args=self.transform_args, device=self.device
                    )

                    # ===== FORWARD PASS (compute losses) =====
                    # 1. features [batch, time, 512] → brain_encoder → brain_emb [batch, seq_len, 1280]
                    # 2. target_texts → TTS → AudioTower → audio_emb [batch, seq_len_audio, 1280]
                    # 3. Alignment loss: brain_emb ↔ audio_emb
                    # 4. LLM loss: brain_emb → projector → LLM → text prediction
                    total_loss, alignment_loss, llm_loss, _, _ = self.model(
                        features,  # [batch, time, 512]
                        day_indicies,  # [batch]
                        target_texts  # list of batch_size strings
                    )
                    # total_loss: scalar
                    # alignment_loss: scalar
                    # llm_loss: scalar

                    # ===== GENERATE TEXT (for WER computation) =====
                    # Conservative max_length for short transcriptions
                    # Most sentences are < 20 words, so 40 tokens ≈ 20-40 words
                    predicted_texts = self.model.generate(
                        features,  # [batch, time, 512]
                        day_indicies,  # [batch]
                        max_length=40  # Max NEW tokens (not total length)
                    )
                    # predicted_texts: list of batch_size strings

                # Collect metrics (move to CPU for storage)
                metrics['total_losses'].append(total_loss.cpu().item())
                metrics['alignment_losses'].append(alignment_loss.cpu().item())
                metrics['llm_losses'].append(llm_loss.cpu().item())
                metrics['day_indicies'].append(day_indicies.cpu().numpy())  # [batch]

                # Store text predictions for WER computation
                metrics['predicted_texts'].extend(predicted_texts)
                metrics['target_texts'].extend(target_texts)

        # Compute average loss metrics across all validation batches
        metrics['avg_total_loss'] = np.mean(metrics['total_losses'])
        metrics['avg_alignment_loss'] = np.mean(metrics['alignment_losses'])
        metrics['avg_llm_loss'] = np.mean(metrics['llm_losses'])

        # Compute WER across all validation samples
        wer, total_edit_distance, total_words = self.compute_wer(
            metrics['predicted_texts'],
            metrics['target_texts']
        )
        metrics['wer'] = wer
        metrics['total_edit_distance'] = total_edit_distance
        metrics['total_words'] = total_words

        return metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'training_args.yaml'
    args = OmegaConf.load(config_path)
    trainer = Brain2TextTrainer(args)
    metrics = trainer.train()