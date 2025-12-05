import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines a direct GRU and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5,
                 patch_size = 0,
                 patch_stride = 0,
                 use_batch_norm = False,
                 batch_norm_momentum = 0.1,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_classes   (int)      - number of classes
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input
        use_batch_norm (bool)  - whether to use batch normalization
        batch_norm_momentum (float) - momentum for batch normalization running statistics
        '''
        super(GRUDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_batch_norm = use_batch_norm

        # Input dropout layer (applied directly to input, no day-specific layers)
        self.input_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Batch normalization for input (applied after patching)
        if self.use_batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(self.input_size, momentum=batch_norm_momentum)

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout,
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Batch normalization for GRU output
        if self.use_batch_norm:
            self.output_batch_norm = nn.BatchNorm1d(self.n_units, momentum=batch_norm_momentum)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

        # Enable gradient checkpointing for memory efficiency
        self.gru.gradient_checkpointing = True

    def forward(self, x, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        '''

        # Apply dropout to the input
        if self.input_dropout > 0:
            x = self.input_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0:

            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]

            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]

            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Apply batch normalization to input if enabled (using PyTorch's nn.BatchNorm1d)
        if self.use_batch_norm:
            # BatchNorm1d expects (batch, features, seq_len), so we need to permute
            x = x.permute(0, 2, 1)  # [batch, input_size, num_patches]
            x = self.input_batch_norm(x)
            x = x.permute(0, 2, 1)  # [batch, num_patches, input_size]

        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN
        output, hidden_states = self.gru(x, states)

        # Apply batch normalization to GRU output if enabled (using PyTorch's nn.BatchNorm1d)
        if self.use_batch_norm:
            # BatchNorm1d expects (batch, features, seq_len), so we need to permute
            output = output.permute(0, 2, 1)  # [batch, n_units, seq_len]
            output = self.output_batch_norm(output)
            output = output.permute(0, 2, 1)  # [batch, seq_len, n_units]

        # Compute logits
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits
        

