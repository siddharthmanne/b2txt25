import torch
from torch import nn
import torch.nn.functional as F
from brain_encoder import BrainEncoder
from audio_target import AudioTarget
from llm_decoder_new import LLMDecoder
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class Brain2TextModel(nn.Module):
    """
    Complete brain-to-text model with two-stage alignment.

    STAGE 1: Brain → Audio Embedding Alignment
    -------------------------------------------
    Trainable path:
        brain_data → BrainEncoder (GRU + Projector) → brain_embedding[t]

    Target path (frozen):
        text_label → TTS model → audio → AudioTower → audio_embedding[t]

    Loss 1: MSE(brain_embedding[t], audio_embedding[t])

    STAGE 2: Decode to Text via Frozen Audio-LLM
    ---------------------------------------------
        aligned_brain_embedding[t] → Qwen Projector (frozen) → LLM (frozen) → text

    Loss 2: Cross-entropy(predicted_text, true_text)

    Total Loss: α * alignment_loss + β * llm_loss
    """

    def __init__(
        self,
        # BrainEncoder params
        neural_dim,
        n_units,
        n_days,
        audio_embedding_dim=1280,  # Qwen AudioTower output dim
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
        # Model IDs
        a2t_model_id="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Quantization settings
        use_quantization=False,
        quantization_bits=8,
        # LoRA settings
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=None,
        lora_bias="none",
        # Loss weights
        alpha=1.0,  # Weight for alignment loss
        beta=1.0,   # Weight for LLM loss
        # Audio cache settings
        cache_dir=None,  # Directory for precomputed audio embeddings
        logger=None,  # Logger instance
    ):
        super(Brain2TextModel, self).__init__()

        self.device = device
        self.alpha = alpha
        self.beta = beta

        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(quantization_bits == 4),
                load_in_8bit=(quantization_bits == 8),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.shared_a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                a2t_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.shared_a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                a2t_model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.shared_processor = AutoProcessor.from_pretrained(a2t_model_id)

         # MEMORY OPTIMIZATION: gradient checkpointing on LLM
        # NOTE: without gradient checkpointing, VRAM usage explodes (95+GB). When implemented, it is limited to ~65GB
        # Even though LLM is frozen, gradients flow through it to train brain_encoder
        # Checkpointing trades compute for memory (recomputes activations during backward)
        # if hasattr(self.shared_a2t_model, 'gradient_checkpointing_enable'):
            # self.shared_a2t_model.gradient_checkpointing_enable()

        # TRAINABLE: Brain encoder (outputs brain_embedding in audio space)
        self.brain_encoder = BrainEncoder(
            neural_dim=neural_dim,
            n_units=n_units,
            n_days=n_days,
            audio_embedding_dim=audio_embedding_dim,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )

        # FROZEN: Audio target (text → precomputed audio embeddings from cache)
        # NOTE: No TTS or AudioTower loaded - all embeddings are precomputed!
        self.audio_target = AudioTarget(
            cache_dir=cache_dir,
            device=device,
            logger=logger,
        )

        # FROZEN (or LoRA): LLM decoder (aligned_brain_embedding → text)
        self.llm_decoder = LLMDecoder(
            shared_a2t_model=self.shared_a2t_model,  # Same instance
            shared_processor=self.shared_processor,
            device=device,
            freeze_all=(not use_lora),  # Freeze if not using LoRA
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
            logger=logger,
        )

        # Loss functions
        self.alignment_loss_fn = nn.MSELoss()

    def forward(self, brain_data, day_idx, target_texts, n_time_steps=None):
        """
        Full training forward pass with two-stage loss.

        STAGE 1: Brain → Audio Alignment Loss
        --------------------------------------
        Train brain_encoder to map neural data into audio embedding space (1280-dim).
        Target: frozen audio embeddings from TTS → AudioTower.

        STAGE 2: LLM Decoding Loss
        ---------------------------
        Pass brain embeddings through frozen multi-modal projector (1280 → 3584).
        Then through frozen LLM to predict text.
        Gradients flow back through projector to brain_encoder.

        Args:
            brain_data: [batch, time, neural_dim=512] - Raw neural activity (may be padded)
            day_idx: [batch] - Day index for each trial (for day-specific layers)
            target_texts: list of str - Ground truth transcriptions
            n_time_steps: [batch] - Actual (unpadded) length of each brain trial

        Returns:
            total_loss: scalar - α * alignment_loss + β * llm_loss
            alignment_loss: scalar - cosine/MSE loss between brain and audio embeddings
            llm_loss: scalar - cross-entropy loss for text prediction
            brain_embedding: [batch, seq_len, 1280] - brain embeddings in audio space
            audio_embedding: [batch, seq_len_audio, 1280] - target audio embeddings
        """
        # ========== STAGE 1: Alignment Loss ==========
        # Get brain embeddings (TRAINABLE)
        brain_embedding = self.brain_encoder(brain_data, day_idx)  # [batch, seq_len, 1280]

        # Get target audio embeddings (FROZEN)
        audio_embedding, attention_mask = self.audio_target(target_texts)  # [batch, seq_len_audio, 1280], [batch, seq_len_audio]

        # Handle sequence length mismatch using interpolation
        if brain_embedding.shape[1] != audio_embedding.shape[1]:
            target_len = min(brain_embedding.shape[1], audio_embedding.shape[1])

            brain_embedding_aligned = F.interpolate(
                brain_embedding.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, target_len, 1280]

            audio_embedding_aligned = F.interpolate(
                audio_embedding.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, target_len, 1280]

            # Recompute mask based on valid lengths after interpolation
            valid_lengths = attention_mask.sum(dim=1)  # [batch]
            new_valid_lengths = (valid_lengths * target_len / audio_embedding.shape[1]).long()  # [batch]

            # OPTIMIZED: Vectorized mask creation (GPU-parallel instead of sequential loop)
            # Create sequence indices [0, 1, 2, ..., target_len-1] and broadcast comparison
            seq_indices = torch.arange(target_len, device=attention_mask.device).unsqueeze(0)  # [1, target_len]
            attention_mask_aligned = (seq_indices < new_valid_lengths.unsqueeze(1)).float()  # [batch, target_len]
            # Broadcasting: [1, target_len] < [batch, 1] → [batch, target_len]
            # Result: 1.0 for positions < valid_length, 0.0 for padding (same as original loop)
        else:
            brain_embedding_aligned = brain_embedding
            audio_embedding_aligned = audio_embedding
            attention_mask_aligned = attention_mask

        # Compute masked alignment loss (only on valid positions, not padding)
        diff = (brain_embedding_aligned - audio_embedding_aligned) ** 2  # [batch, target_len, 1280]
        diff = diff.mean(dim=-1)  # [batch, target_len] - average over embedding dim

        # Apply mask: only compute loss where attention_mask=1
        mask = (attention_mask_aligned > 0.5).float()  # [batch, target_len]
        masked_diff = diff * mask  # [batch, target_len]
        alignment_loss = masked_diff.sum() / (mask.sum() + 1e-8)  # Average only over valid positions

        # ========== STAGE 2: LLM Decoding Loss ==========
        # Pass brain embeddings through frozen LLM decoder
        # Internally: brain_embedding [batch, seq_len, 1280]
        #          → projector [batch, seq_len, 3584]
        #          → LLM with text labels
        #          → cross-entropy loss
        llm_loss, logits = self.llm_decoder.forward_train(
            brain_embedding,  # [batch, seq_len, 1280]
            target_texts  # list of str
        )
        # llm_loss: scalar
        # logits: [batch, seq_len + T_text, 156032]

        # ========== COMBINED LOSS ==========
        total_loss = self.alpha * alignment_loss + self.beta * llm_loss

        return total_loss, alignment_loss, llm_loss, brain_embedding, audio_embedding

    def generate(self, brain_data, day_idx, max_length=50):
        """
        Inference: Generate text from brain data.

        Pipeline:
        1. brain_data [batch, time, 512] → brain_encoder → [batch, seq_len, 1280]
        2. brain_embedding [batch, seq_len, 1280] → projector → [batch, seq_len, 3584]
        3. [brain_prefix, "Transcript:"] → LLM → autoregressive generation → text

        Args:
            brain_data: [batch, time, neural_dim=512] - Raw neural activity
            day_idx: [batch] - Day index for each trial
            max_length: int - Maximum NEW tokens to generate
                              ~1-2 tokens per word, so 40 tokens ≈ 20-40 words

        Returns:
            generated_texts: list of str - Predicted transcriptions
        """
        # Get brain embeddings from neural data
        brain_embedding = self.brain_encoder(brain_data, day_idx)  # [batch, seq_len, 1280]

        # Generate text via LLM decoder
        # Internally: brain_embedding → projector [3584] → LLM → autoregressive decoding
        generated_texts = self.llm_decoder.generate(brain_embedding, max_length)  # list of str

        return generated_texts  # list of str

    def get_trainable_parameters(self):
        """
        Return only trainable parameters (brain_encoder).
        Audio target and LLM decoder are frozen.
        """
        return self.brain_encoder.parameters()
