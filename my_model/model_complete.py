import torch
from torch import nn
import torch.nn.functional as F
from brain_encoder import BrainEncoder
from audio_target import AudioTarget
from llm_decoder_new import LLMDecoder
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

class Brain2TextModel(nn.Module):
    """
    Complete brain-to-text model with two-stage alignment.

    Architecture:
    ============
    STAGE 1: Brain → Audio Embedding Alignment
    -------------------------------------------
    Trainable path:
        brain_data → BrainEncoder (GRU + Projector) → brain_embedding[t]

    Target path (frozen):
        text_label → TTS → audio → AudioTower → audio_embedding[t]

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
        t2a_model_id="facebook/mms-tts-eng",
        a2t_model_id="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Loss weights
        alpha=1.0,  # Weight for alignment loss
        beta=1.0,   # Weight for LLM loss
    ):
        super(Brain2TextModel, self).__init__()

        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.shared_a2t_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            a2t_model_id,
            device_map="auto",
            torch_dtype=torch.float16
            )
        self.shared_processor = AutoProcessor.from_pretrained(a2t_model_id)
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

        # FROZEN: Audio target (text → audio → audio_embedding)
        self.audio_target = AudioTarget(
            t2a_model_id=t2a_model_id,
            shared_a2t_model=self.shared_a2t_model,  # Pass the instance
            shared_processor=self.shared_processor,
            device=device,
        )

        # FROZEN: LLM decoder (aligned_brain_embedding → text)
        self.llm_decoder = LLMDecoder(
            shared_a2t_model=self.shared_a2t_model,  # Same instance
            shared_processor=self.shared_processor,
            device=device,
            freeze_all=True,
        )

        # Loss functions
        self.alignment_loss_fn = nn.MSELoss()  # Or nn.MSELoss()

    def forward(self, brain_data, day_idx, target_texts):
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
            brain_data: [batch, time, neural_dim=512] - Raw neural activity
            day_idx: [batch] - Day index for each trial (for day-specific layers)
            target_texts: list of str - Ground truth transcriptions

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
        # Brain and audio sequences may have different temporal lengths due to:
        # - Brain: depends on trial duration and patching
        # - Audio: depends on speech duration (TTS output length)
        if brain_embedding.shape[1] != audio_embedding.shape[1]:
            # Use minimum length to avoid extrapolation
            target_len = min(brain_embedding.shape[1], audio_embedding.shape[1])  # scalar int

            # Interpolate to same length: [batch, seq_len, 1280] → [batch, target_len, 1280]
            # Need to transpose because F.interpolate operates on last dimension
            brain_embedding_aligned = F.interpolate(
                brain_embedding.transpose(1, 2),  # [batch, 1280, seq_len]
                size=target_len,  # target sequence length
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, target_len, 1280]

            audio_embedding_aligned = F.interpolate(
                audio_embedding.transpose(1, 2),  # [batch, 1280, seq_len_audio]
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, target_len, 1280]
        else:
            brain_embedding_aligned = brain_embedding  # [batch, seq_len, 1280]
            audio_embedding_aligned = audio_embedding  # [batch, seq_len, 1280]

        # Compute alignment loss on time-aligned sequences
        alignment_loss = self.alignment_loss_fn(brain_embedding_aligned, audio_embedding_aligned)  # scalar

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
        total_loss = self.alpha * alignment_loss + self.beta * llm_loss  # scalar

        return total_loss, alignment_loss, llm_loss, brain_embedding, audio_embedding
        # total_loss: scalar
        # alignment_loss: scalar
        # llm_loss: scalar
        # brain_embedding: [batch, seq_len, 1280]
        # audio_embedding: [batch, seq_len_audio, 1280]

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
            max_length: int - Maximum tokens to generate

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
