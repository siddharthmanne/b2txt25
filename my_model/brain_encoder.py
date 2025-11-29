import torch
from torch import nn

class BrainEncoder(nn.Module):
    '''
    Trainable encoder: Brain data → brain_embedding

    Architecture: GRU/Transformer → Linear Projector
    Output aligns with audio embedding space (from Qwen AudioTower)
    '''
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        audio_embedding_dim,  # Target: Qwen AudioTower output dim (e.g., 1280)
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
    ):
        '''
        neural_dim (int)            - number of channels in a single timestep (e.g. 256)
        n_units (int)               - number of hidden units in each GRU layer
        n_days (int)                - number of recording days
        audio_embedding_dim (int)   - dimension of Qwen AudioTower output (target space)
        rnn_dropout (float)         - dropout in GRU layers
        input_dropout (float)       - dropout on input
        n_layers (int)              - number of GRU layers
        patch_size (int)            - number of timesteps to concat (0 to disable)
        patch_stride (int)          - stride for patching
        '''
        super(BrainEncoder, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_days = n_days
        self.audio_embedding_dim = audio_embedding_dim

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific input layers
        self.day_layer_activation = nn.Softsign()

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim

        # If using strided inputs, increase input size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # GRU encoder
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Projector: GRU hidden → audio embedding space
        self.projector = nn.Sequential(
            nn.Linear(self.n_units, audio_embedding_dim),
            nn.LayerNorm(audio_embedding_dim)
        )

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        '''
        x (tensor)       - brain data: [batch, time, neural_dim=512]
        day_idx (tensor) - day indices for each trial: [batch] (one day per trial)

        Returns:
            brain_embedding: [batch, time_or_patches, 1280]
        '''
        # Apply day-specific transformations
        # Each day has its own [neural_dim, neural_dim] weight matrix and [1, neural_dim] bias
        # Select the appropriate weight/bias for each trial's day
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)  # [batch, neural_dim=512, neural_dim=512]
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)  # [batch, 1, neural_dim=512]

        # Apply day-specific linear transformation: x @ day_weight + day_bias
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases  # [batch, time, neural_dim=512]
        x = self.day_layer_activation(x)  # [batch, time, neural_dim=512]

        # Apply dropout
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)  # [batch, time, neural_dim=512]

        # Optional patching: concatenate adjacent timesteps
        if self.patch_size > 0:
            x = x.unsqueeze(1)  # [batch, 1, time, neural_dim=512]
            x = x.permute(0, 3, 1, 2)  # [batch, neural_dim=512, 1, time]

            # Unfold creates patches of size patch_size with stride patch_stride
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batch, neural_dim=512, 1, num_patches, patch_size]
            x_unfold = x_unfold.squeeze(2)  # [batch, neural_dim=512, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batch, num_patches, patch_size, neural_dim=512]

            # Flatten patch dimension: concatenate patch_size timesteps
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)  # [batch, num_patches, patch_size*neural_dim]

        # Initialize hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()  # [n_layers=5, batch, n_units=768]

        # GRU encoding
        # Input: [batch, seq_len, input_size] where:
        #   - seq_len = time (if no patching) or num_patches (if patching)
        #   - input_size = neural_dim (no patching) or patch_size*neural_dim (patching)
        gru_output, hidden_states = self.gru(x, states)  # gru_output: [batch, seq_len, n_units=768]
                                                          # hidden_states: [n_layers=5, batch, n_units=768]

        # Project to audio embedding space: n_units → audio_embedding_dim (768 → 1280)
        brain_embedding = self.projector(gru_output)  # [batch, seq_len, 1280]

        if return_state:
            return brain_embedding, hidden_states  # [batch, seq_len, 1280], [n_layers=5, batch, n_units=768]

        return brain_embedding  # [batch, seq_len, 1280]
