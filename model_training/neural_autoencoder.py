"""
Neural Autoencoder Module for Brain-to-Text Alignment

This module implements a temporal autoencoder architecture that:
1. Encodes neural activity data into compact latent representations (brain embeddings)
2. Aligns these brain embeddings with text embeddings from LLMs like Llama
3. Enables downstream task prediction (e.g., phoneme/word prediction)

Pipeline overview:
    Neural data [B, T, 512] → Autoencoder → Brain embedding [B, T, latent_dim]
    Text transcription → Llama → Text embedding [B, T, 4096]
    Brain embedding → Projection MLP → Aligned embedding [B, T, 4096]
    Loss: align brain embeddings to text embeddings + reconstruction loss
    Task: Aligned embedding → Task decoder → Predictions (conditioned on audio)

Where:
    B = batch size
    T = time steps (variable length sequences)
    512 = neural features (from 256 electrodes × 2 features each)
    4096 = Llama embedding dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralAutoencoder(nn.Module):
    """
    Temporal Autoencoder for neural data with alignment to LLM embeddings.

    This autoencoder learns to compress high-dimensional neural activity patterns
    into a lower-dimensional latent space (brain embeddings) while preserving
    temporal structure. The brain embeddings are designed to be aligned with
    text embeddings from modern LLMs like Llama through contrastive learning.

    Architecture flow:
        1. Optional day-specific normalization (handles cross-day variability)
        2. Feedforward encoder layers (512 → 1024 → 1536 → 2048)
        3. Temporal modeling (Transformer/GRU/LSTM to capture temporal dependencies)
        4. Projection to latent space (2048 → latent_dim, typically 1024)
        5. Reverse path for reconstruction (latent_dim → 2048 → ... → 512)

    The latent embedding can be projected to Llama's 4096-dim space via AlignmentProjector.
    """

    def __init__(
        self,
        input_dim=512,
        latent_dim=1024,
        hidden_dims=[1024, 1536, 2048],
        temporal_model='transformer',
        n_transformer_layers=4,
        n_heads=8,
        dropout=0.1,
        use_day_layers=True,
        n_days=45
    ):
        """
        Initialize the Neural Autoencoder.

        Args:
            input_dim (int): Number of neural features per timestep (default: 512)
                            512 = 256 electrodes × 2 features (threshold crossings + spike band power)
            latent_dim (int): Dimension of compressed brain embedding (default: 1024)
                             This is the bottleneck dimension that forces the model to learn
                             a compact representation of neural activity
            hidden_dims (list): Dimensions for progressive encoder/decoder layers
                               Default [1024, 1536, 2048] gradually increases capacity
            temporal_model (str): Type of temporal modeling layer
                                 'transformer' - self-attention, captures long-range dependencies
                                 'gru' - gated recurrent unit, faster than LSTM
                                 'lstm' - long short-term memory, handles long sequences
            n_transformer_layers (int): Depth of temporal model (default: 4)
            n_heads (int): Number of attention heads for transformer (default: 8)
            dropout (float): Dropout probability for regularization (default: 0.1)
            use_day_layers (bool): Whether to use day-specific normalization layers
                                  Helps account for neural drift across recording days
            n_days (int): Number of recording days in dataset (default: 45)
        """
        super(NeuralAutoencoder, self).__init__()

        # Store hyperparameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.temporal_model = temporal_model
        self.use_day_layers = use_day_layers
        self.n_days = n_days

        # ============================================
        # Day-specific normalization layers
        # ============================================
        # Neural recordings can drift across days due to electrode movement,
        # tissue response, etc. Day-specific layers learn affine transformations
        # (weight matrix + bias) for each recording day to normalize the data
        # to a common latent space before encoding.
        if use_day_layers:
            # Softsign activation: x / (1 + |x|), similar to tanh but shallower
            self.day_layer_activation = nn.Softsign()

            # Create separate weight matrices for each day, initialized as identity
            # This allows the model to learn day-specific transformations starting
            # from a neutral initialization
            self.day_weights = nn.ParameterList(
                [nn.Parameter(torch.eye(input_dim)) for _ in range(n_days)]
            )

            # Create separate bias vectors for each day, initialized to zero
            self.day_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, input_dim)) for _ in range(n_days)]
            )

        # ============================================
        # Encoder: Pre-temporal feature extraction
        # ============================================
        # These feedforward layers process each timestep independently before
        # temporal modeling. They learn to extract relevant features from the
        # raw neural activity patterns.
        encoder_layers = []
        prev_dim = input_dim

        # Build progressive encoder: 512 → 1024 → 1536 → 2048
        # Each block: Linear → LayerNorm → GELU → Dropout
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                # Linear transformation to expand/project features
                nn.Linear(prev_dim, hidden_dim),
                # Layer normalization for stable training
                nn.LayerNorm(hidden_dim),
                # GELU activation: smooth, differentiable nonlinearity
                nn.GELU(),
                # Dropout for regularization
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder_pre_temporal = nn.Sequential(*encoder_layers)

        # Temporal modeling
        if temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dims[-1],
                nhead=n_heads,
                dim_feedforward=hidden_dims[-1] * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_transformer_layers
            )

            decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dims[-1],
                nhead=n_heads,
                dim_feedforward=hidden_dims[-1] * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.temporal_decoder = nn.TransformerEncoder(
                decoder_layer,
                num_layers=n_transformer_layers
            )

        elif temporal_model == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=hidden_dims[-1],
                hidden_size=hidden_dims[-1],
                num_layers=2,
                dropout=dropout if n_transformer_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            # Project bidirectional output back to hidden_dims[-1]
            self.temporal_encoder_proj = nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1])

            self.temporal_decoder = nn.GRU(
                input_size=latent_dim,
                hidden_size=hidden_dims[-1],
                num_layers=2,
                dropout=dropout if n_transformer_layers > 1 else 0,
                batch_first=True
            )

        elif temporal_model == 'lstm':
            self.temporal_encoder = nn.LSTM(
                input_size=hidden_dims[-1],
                hidden_size=hidden_dims[-1],
                num_layers=2,
                dropout=dropout if n_transformer_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            self.temporal_encoder_proj = nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1])

            self.temporal_decoder = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dims[-1],
                num_layers=2,
                dropout=dropout if n_transformer_layers > 1 else 0,
                batch_first=True
            )

        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Decoder: Latent -> Hidden layers -> Output
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GRU, nn.LSTM)):
                for name, param in module.named_parameters():
                    if 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def apply_day_layers(self, x, day_idx):
        """Apply day-specific normalization"""
        if not self.use_day_layers:
            return x

        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        return x

    def encode(self, x, day_idx=None, mask=None):
        """
        Encode neural data to latent brain embeddings.

        Args:
            x (tensor): Neural data [batch, time, 512]
            day_idx (tensor): Day indices for day-specific layers [batch]
            mask (tensor): Padding mask [batch, time]

        Returns:
            brain_embedding (tensor): Latent embeddings [batch, time, latent_dim]
        """
        # Apply day-specific layers if enabled
        if day_idx is not None:
            x = self.apply_day_layers(x, day_idx)

        # Encode to hidden representation
        x = self.encoder_pre_temporal(x)  # [batch, time, hidden_dims[-1]]

        # Apply temporal modeling
        if self.temporal_model == 'transformer':
            # For transformer, mask needs to be inverted: True = ignore
            attn_mask = ~mask.bool() if mask is not None else None
            x = self.temporal_encoder(x, src_key_padding_mask=attn_mask)
        elif self.temporal_model in ['gru', 'lstm']:
            # For RNNs, pack sequences if mask is provided
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )

            if self.temporal_model == 'gru':
                x, _ = self.temporal_encoder(x)
            else:  # lstm
                x, _ = self.temporal_encoder(x)

            # Unpack sequences
            if mask is not None:
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            # Project bidirectional output
            x = self.temporal_encoder_proj(x)

        # Project to latent space
        brain_embedding = self.to_latent(x)  # [batch, time, latent_dim]

        return brain_embedding

    def decode(self, brain_embedding, mask=None):
        """
        Decode latent embeddings back to neural data.

        Args:
            brain_embedding (tensor): Latent embeddings [batch, time, latent_dim]
            mask (tensor): Padding mask [batch, time]

        Returns:
            reconstruction (tensor): Reconstructed neural data [batch, time, 512]
        """
        # Apply temporal modeling in decoder
        if self.temporal_model == 'transformer':
            attn_mask = ~mask.bool() if mask is not None else None
            x = self.temporal_decoder(brain_embedding, src_key_padding_mask=attn_mask)
        elif self.temporal_model in ['gru', 'lstm']:
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    brain_embedding, lengths, batch_first=True, enforce_sorted=False
                )

            if self.temporal_model == 'gru':
                x, _ = self.temporal_decoder(brain_embedding if mask is None else x)
            else:  # lstm
                x, _ = self.temporal_decoder(brain_embedding if mask is None else x)

            if mask is not None:
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x = brain_embedding

        # Decode to neural space
        reconstruction = self.decoder(x)  # [batch, time, 512]

        return reconstruction

    def forward(self, x, day_idx=None, mask=None):
        """
        Full autoencoder forward pass.

        Args:
            x (tensor): Neural data [batch, time, 512]
            day_idx (tensor): Day indices [batch]
            mask (tensor): Padding mask [batch, time], 1 for valid, 0 for padding

        Returns:
            reconstruction (tensor): Reconstructed neural data [batch, time, 512]
            brain_embedding (tensor): Latent embeddings [batch, time, latent_dim]
        """
        brain_embedding = self.encode(x, day_idx, mask)
        reconstruction = self.decode(brain_embedding, mask)

        return reconstruction, brain_embedding


class AlignmentProjector(nn.Module):
    """
    Projects brain embeddings to text embedding space (e.g., Llama embeddings).
    This is trained jointly with the autoencoder for alignment.
    """

    def __init__(
        self,
        brain_dim=1024,
        text_dim=4096,
        hidden_dim=2048,
        n_layers=3,
        dropout=0.1
    ):
        """
        Args:
            brain_dim (int): Dimension of brain embeddings from autoencoder
            text_dim (int): Dimension of text embeddings (Llama: 4096)
            hidden_dim (int): Hidden layer dimension
            n_layers (int): Number of projection layers
            dropout (float): Dropout rate
        """
        super(AlignmentProjector, self).__init__()

        layers = []
        prev_dim = brain_dim

        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final projection to text embedding space
        layers.append(nn.Linear(prev_dim, text_dim))

        self.projector = nn.Sequential(*layers)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, brain_embedding):
        """
        Project brain embeddings to text embedding space.

        Args:
            brain_embedding (tensor): Brain embeddings [batch, time, brain_dim]

        Returns:
            aligned_embedding (tensor): Aligned embeddings [batch, time, text_dim]
        """
        return self.projector(brain_embedding)


class TaskDecoder(nn.Module):
    """
    Task-specific decoder for speech prediction conditioned on audio/text.
    Can be used for phoneme prediction, word prediction, etc.
    """

    def __init__(
        self,
        input_dim=4096,
        n_classes=40,
        hidden_dim=1024,
        n_transformer_layers=4,
        n_heads=8,
        dropout=0.1,
        condition_dim=None
    ):
        """
        Args:
            input_dim (int): Dimension of aligned embeddings
            n_classes (int): Number of output classes (e.g., phonemes)
            hidden_dim (int): Transformer hidden dimension
            n_transformer_layers (int): Number of transformer layers
            n_heads (int): Number of attention heads
            dropout (float): Dropout rate
            condition_dim (int): Dimension of conditioning signal (e.g., audio features)
        """
        super(TaskDecoder, self).__init__()

        self.condition_dim = condition_dim

        # Project input to hidden dimension
        proj_input_dim = input_dim
        if condition_dim is not None:
            proj_input_dim += condition_dim

        self.input_proj = nn.Linear(proj_input_dim, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_transformer_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, aligned_embedding, condition=None, mask=None):
        """
        Decode aligned embeddings to task predictions.

        Args:
            aligned_embedding (tensor): Aligned embeddings [batch, time, input_dim]
            condition (tensor): Optional conditioning signal [batch, time, condition_dim]
            mask (tensor): Padding mask [batch, time]

        Returns:
            logits (tensor): Class logits [batch, time, n_classes]
        """
        # Concatenate with conditioning if provided
        if condition is not None:
            x = torch.cat([aligned_embedding, condition], dim=-1)
        else:
            x = aligned_embedding

        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply transformer
        attn_mask = ~mask.bool() if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Classify
        logits = self.classifier(x)

        return logits


def create_padding_mask(lengths, max_len=None):
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths (tensor): Sequence lengths [batch]
        max_len (int): Maximum sequence length

    Returns:
        mask (tensor): Padding mask [batch, max_len], 1 for valid, 0 for padding
    """
    if max_len is None:
        max_len = lengths.max()

    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)

    return mask.float()
