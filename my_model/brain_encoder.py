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
        x (tensor)       - brain data: [batch, time, neural_dim]
        day_idx (tensor) - day indices for each example: [batch]

        Returns:
            brain_embedding: [batch, time, audio_embedding_dim]
        '''
        # Apply day-specific transformations
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Optional patching
        if self.patch_size > 0:
            x = x.unsqueeze(1)                      # [batch, 1, time, features]
            x = x.permute(0, 3, 1, 2)               # [batch, features, 1, time]

            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)          # [batch, features, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1) # [batch, num_patches, patch_size, features]

            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Initialize hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # GRU encoding
        gru_output, hidden_states = self.gru(x, states)

        # Project to audio embedding space
        brain_embedding = self.projector(gru_output)

        if return_state:
            return brain_embedding, hidden_states

        return brain_embedding
