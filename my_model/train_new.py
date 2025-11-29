import torch 
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

from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import apply_data_augmentations
from model_complete import Brain2TextModel

from omegaconf import OmegaConf

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

        self.best_val_loss = torch.inf
        self.best_val_alignment_loss = torch.inf
        self.best_val_llm_loss = torch.inf

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.transform_args = self.args['dataset']['data_transforms']

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

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device
        if not torch.cuda.is_available():
            self.logger.error("No GPU available. This training requires CUDA. Exiting.")
            sys.exit(1)
        
        gpu_num = self.args.get('gpu_number', 0)
        try:
            gpu_num = int(gpu_num)
        except ValueError:
            self.logger.warning(f"Invalid gpu_number: {gpu_num}. Using 0.")
            gpu_num = 0

        max_gpu_index = torch.cuda.device_count() - 1
        if gpu_num > max_gpu_index:
            self.logger.warning(f"GPU {gpu_num} not available. Using GPU 0.")
            gpu_num = 0

        self.device = torch.device(f"cuda:{gpu_num}")

        self.logger.info(f'Using device: {self.device}')

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
            t2a_model_id=self.args['model']['t2a_model_id'],
            a2t_model_id=self.args['model']['a2t_model_id'],
            device=self.device,
            alpha=self.args['alpha'],
            beta=self.args['beta'],
        )

        # Use torch.compile for 2-3x speedup
        # self.logger.info("Using torch.compile for speedup")
        # self.model = torch.compile(self.model)

        self.logger.info(f"Initialized Brain2Text model")

        # Log parameter counts
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

        # Save train/val split
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
            batch_size=None,
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
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer with separate param groups
        self.optimizer = self.create_optimizer()

        # Learning rate scheduler
        if self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        elif self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=self.args['lr_decay_steps'],
            )
        else:
            raise ValueError(f"Invalid lr_scheduler_type: {self.args['lr_scheduler_type']}")

        # Load checkpoint if specified
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Send model to device
        self.model.to(self.device)

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

        for name, p in self.model.brain_encoder.named_parameters():
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

        optim = torch.optim.AdamW(
            param_groups,
            lr=self.args['lr_max'],
            betas=(self.args['beta0'], self.args['beta1']),
            eps=self.args['epsilon'],
            weight_decay=self.args['weight_decay'],
            fused=True  # Faster optimizer (requires CUDA)
        )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        """Create cosine LR scheduler with warmup for each param group."""
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']
        lr_max_day = self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
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

    def load_model_checkpoint(self, load_path):
        """Load training checkpoint."""
        checkpoint = torch.load(load_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint: {save_path}")

        # Save args alongside checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def train(self):
        """Main training loop."""
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
            self.optimizer.zero_grad()

            start_time = time.time()

            # ===== Load batch data from dataset =====
            # Move data to device
            features = batch['input_features'].to(self.device)  # [batch, time, neural_dim=512]
            n_time_steps = batch['n_time_steps'].to(self.device)  # [batch] - actual timesteps per trial
            day_indicies = batch['day_indicies'].to(self.device)  # [batch] - day index for each trial

            # Decode transcriptions from byte strings
            transcriptions_raw = batch['transcriptions']  # [batch, max_text_len] - byte-encoded strings
            target_texts = []  # Will be list of batch_size strings
            for j in range(transcriptions_raw.shape[0]):
                text = bytes(transcriptions_raw[j].cpu().numpy()).decode('utf-8').strip()
                target_texts.append(text)  # e.g., "hello world"

            # ===== Mixed precision training with autocast =====
            # Use autocast for efficiency (AMP with bfloat16)
            with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):
                # Apply data augmentations ON GPU (smoothing, noise, etc.)
                features, n_time_steps = apply_data_augmentations(
                    features, n_time_steps, mode='train',
                    transform_args=self.transform_args, device=self.device
                )
                # features: [batch, time, neural_dim=512] - augmented
                # n_time_steps: [batch] - may be adjusted by random_cut

                # Calculate adjusted sequence lengths after patching
                # This is CRITICAL: patching changes the temporal dimension
                # Patching concatenates adjacent timesteps: [time, neural_dim] → [num_patches, patch_size*neural_dim]
                if self.args['model']['patch_size'] > 0:
                    # Formula: num_patches = (time - patch_size) / stride + 1
                    adjusted_lens = (
                        (n_time_steps - self.args['model']['patch_size'])
                        / self.args['model']['patch_stride'] + 1
                    ).to(torch.int32)  # [batch] - number of patches per trial
                else:
                    adjusted_lens = n_time_steps  # [batch] - no patching, same as input

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
                # alignment_loss: scalar - cosine/MSE between brain and audio embeddings
                # llm_loss: scalar - cross-entropy for text generation
                # brain_emb: [batch, seq_len, 1280] - brain embeddings in audio space
                # audio_emb: [batch, seq_len_audio, 1280] - target audio embeddings

            # ===== BACKWARD PASS =====
            # Gradients flow:
            # 1. LLM loss → (frozen projector) → (frozen LLM) BUT gradients flow back to brain_emb
            # 2. Alignment loss → brain_emb
            # 3. Both losses → brain_encoder parameters (TRAINABLE)
            total_loss.backward()

            # Gradient clipping (prevent exploding gradients)
            if self.args['grad_norm_clip_value'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.brain_encoder.parameters(),  # Only brain_encoder has gradients
                    max_norm=self.args['grad_norm_clip_value'],
                    error_if_nonfinite=True,
                    foreach=True
                )

            # ===== OPTIMIZER STEP =====
            self.optimizer.step()  # Update brain_encoder parameters
            self.learning_rate_scheduler.step()  # Update learning rates

            train_step_duration = time.time() - start_time
            train_losses.append(total_loss.detach().item())

            # Log training progress
            if i % self.args['batches_per_train_log'] == 0:
                # Log shapes for debugging (only first time)
                if i == 0:
                    self.logger.info(
                        f'Data shapes: '
                        f'input=[{features.shape[0]}, {features.shape[1]}, {features.shape[2]}] '  # [batch, time, 512]
                        f'brain_emb=[{brain_emb.shape[0]}, {brain_emb.shape[1]}, {brain_emb.shape[2]}] '  # [batch, seq_len, 1280]
                        f'audio_emb=[{audio_emb.shape[0]}, {audio_emb.shape[1]}, {audio_emb.shape[2]}]'  # [batch, seq_len_audio, 1280]
                    )
                    if self.args['model']['patch_size'] > 0:
                        self.logger.info(
                            f'Patching enabled: {features.shape[1]} timesteps → '
                            f'{brain_emb.shape[1]} patches '
                            f'(patch_size={self.args["model"]["patch_size"]}, '
                            f'stride={self.args["model"]["patch_stride"]})'
                        )

                self.logger.info(
                    f'Train batch {i}: '
                    f'total_loss: {total_loss.item():.4f} '
                    f'align_loss: {alignment_loss.item():.4f} '
                    f'llm_loss: {llm_loss.item():.4f} '
                    f'grad_norm: {grad_norm:.2f} '
                    f'time: {train_step_duration:.3f}s'
                )

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
                        f'time: {val_step_duration:.3f}s'
                    )

                    val_losses.append(val_metrics['avg_total_loss'])
                    val_results.append(val_metrics)

                    # Check for improvement
                    new_best = False
                    if val_metrics['avg_total_loss'] < self.best_val_loss:
                        self.logger.info(
                            f"New best val loss: {self.best_val_loss:.4f} -> {val_metrics['avg_total_loss']:.4f}"
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
        }

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch in self.val_loader:
                # ===== Load batch data =====
                features = batch['input_features'].to(self.device)  # [batch, time, neural_dim=512]
                n_time_steps = batch['n_time_steps'].to(self.device)  # [batch] - actual timesteps
                day_indicies = batch['day_indicies'].to(self.device)  # [batch] - day indices

                # Decode transcriptions from byte strings
                transcriptions_raw = batch['transcriptions']  # [batch, max_text_len]
                target_texts = []  # Will be list of batch_size strings
                for j in range(transcriptions_raw.shape[0]):
                    text = bytes(transcriptions_raw[j].cpu().numpy()).decode('utf-8').strip()
                    target_texts.append(text)  # e.g., "hello world"

                # Apply transforms (no augmentation in val, but still smooth)
                # Smoothing is CRITICAL for neural data quality
                with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):
                    features, n_time_steps = apply_data_augmentations(
                        features, n_time_steps, mode='val',
                        transform_args=self.transform_args, device=self.device
                    )
                    # features: [batch, time, neural_dim=512] - smoothed but not augmented
                    # n_time_steps: [batch] - unchanged in val mode

                    # Calculate adjusted sequence lengths after patching
                    if self.args['model']['patch_size'] > 0:
                        # Formula: num_patches = (time - patch_size) / stride + 1
                        adjusted_lens = (
                            (n_time_steps - self.args['model']['patch_size'])
                            / self.args['model']['patch_stride'] + 1
                        ).to(torch.int32)  # [batch]
                    else:
                        adjusted_lens = n_time_steps  # [batch]

                    # ===== FORWARD PASS (same as training) =====
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

                # Collect metrics (move to CPU for storage)
                metrics['total_losses'].append(total_loss.cpu().item())
                metrics['alignment_losses'].append(alignment_loss.cpu().item())
                metrics['llm_losses'].append(llm_loss.cpu().item())
                metrics['day_indicies'].append(day_indicies.cpu().numpy())  # [batch]

        # Compute average metrics across all validation batches
        metrics['avg_total_loss'] = np.mean(metrics['total_losses'])
        metrics['avg_alignment_loss'] = np.mean(metrics['alignment_losses'])
        metrics['avg_llm_loss'] = np.mean(metrics['llm_losses'])

        return metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'training_args.yaml'
    args = OmegaConf.load(config_path)
    trainer = Brain2TextTrainer(args)
    metrics = trainer.train()