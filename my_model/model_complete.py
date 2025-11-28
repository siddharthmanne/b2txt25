import torch
from torch import nn
import torch.nn.functional as F
from brain_encoder import BrainEncoder
from audio_target import AudioTarget
from llm_decoder_new import LLMDecoder

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
            a2t_model_id=a2t_model_id,
            device=device,
        )

        # FROZEN: LLM decoder (aligned_brain_embedding → text)
        self.llm_decoder = LLMDecoder(
            a2t_model_id=a2t_model_id,
            device=device,
            freeze_all=True,
        )

        # Loss functions
        self.alignment_loss_fn = nn.CosineEmbeddingLoss()  # Or nn.MSELoss()

    def forward(self, brain_data, day_idx, target_texts):
        """
        Full training forward pass with two losses.

        Args:
            brain_data: [batch, time, neural_dim] - Neural activity
            day_idx: [batch] - Day indices for each trial
            target_texts: list of str - Ground truth transcriptions

        Returns:
            total_loss: Combined loss
            alignment_loss: Stage 1 loss (brain → audio alignment)
            llm_loss: Stage 2 loss (text generation)
            brain_embedding: [batch, time, audio_embedding_dim]
            audio_embedding: [batch, time, audio_embedding_dim]
        """
        # ========== STAGE 1: Alignment Loss ==========
        # Get brain embeddings (trainable)
        brain_embedding = self.brain_encoder(brain_data, day_idx)

        # Get target audio embeddings (frozen)
        audio_embedding = self.audio_target(target_texts)

        # Handle sequence length mismatch using interpolation
        # Brain and audio sequences may have different temporal lengths
        if brain_embedding.shape[1] != audio_embedding.shape[1]:
            # Use minimum length to avoid extrapolation
            target_len = min(brain_embedding.shape[1], audio_embedding.shape[1])

            # Interpolate to same length: [batch, time, dim] → [batch, target_len, dim]
            brain_embedding_aligned = F.interpolate(
                brain_embedding.transpose(1, 2),  # [batch, dim, time]
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, target_len, dim]

            audio_embedding_aligned = F.interpolate(
                audio_embedding.transpose(1, 2),  # [batch, dim, time]
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, target_len, dim]
        else:
            brain_embedding_aligned = brain_embedding
            audio_embedding_aligned = audio_embedding

        # Compute alignment loss on time-aligned sequences
        alignment_loss = self.alignment_loss_fn(brain_embedding_aligned, audio_embedding_aligned)

        # ========== STAGE 2: LLM Decoding Loss ==========
        # Pass aligned brain embeddings through frozen LLM decoder
        llm_loss, logits = self.llm_decoder.forward_train(
            brain_embedding,  # Use the aligned brain embedding
            target_texts
        )

        # ========== COMBINED LOSS ==========
        total_loss = self.alpha * alignment_loss + self.beta * llm_loss

        return total_loss, alignment_loss, llm_loss, brain_embedding, audio_embedding

    def generate(self, brain_data, day_idx, max_length=50):
        """
        Inference: Generate text from brain data.

        Args:
            brain_data: [batch, time, neural_dim]
            day_idx: [batch]
            max_length: Maximum text length

        Returns:
            generated_texts: list of str
        """
        # Get brain embeddings
        brain_embedding = self.brain_encoder(brain_data, day_idx)

        # Generate text via LLM decoder
        generated_texts = self.llm_decoder.generate(brain_embedding, max_length)

        return generated_texts

    def get_trainable_parameters(self):
        """
        Return only trainable parameters (brain_encoder).
        Audio target and LLM decoder are frozen.
        """
        return self.brain_encoder.parameters()
