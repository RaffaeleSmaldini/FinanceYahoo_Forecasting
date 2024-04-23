import torch
import torch.nn as nn
import math
"""ToM == Transformer only Model"""
class ToM(nn.Module):
    def __init__(self, input_dim, n_head, dim_forward, embedding_dim, n_layers, output_dim, regularize=True):
        super(ToM, self).__init__()
        self.embedding_dim = embedding_dim
        embedding_layer = [
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU()

        ]

        # create a positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim, max_seq_length=input_dim)

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            embedding_layer.insert(1, nn.LayerNorm(input_dim*4))
            embedding_layer.insert(4, nn.LayerNorm(embedding_dim*2))
            embedding_layer.insert(7, nn.LayerNorm(embedding_dim))

        # Define the embedding layer
        self.embedding = nn.Sequential(*embedding_layer)

        # Define the transformer' encoder layer -->
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                #embedding dimension
                d_model=embedding_dim,
                #number of head
                nhead=n_head,
                #feedforward layer dimension for each layer
                dim_feedforward=dim_forward,
                batch_first=True

            ),
            num_layers=n_layers
        )

        dense_layers = [
            nn.Linear(embedding_dim, embedding_dim*4),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(embedding_dim * 4, embedding_dim),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(embedding_dim, embedding_dim // 2),  # Linear layer
            nn.ReLU(),  # ReLU
            nn.Linear(embedding_dim // 2, output_dim),  # Linear layer
            nn.ReLU(),  # ReLU
        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            dense_layers.insert(1, nn.LayerNorm(embedding_dim*4))
            dense_layers.insert(4, nn.LayerNorm(embedding_dim))
            dense_layers.insert(7, nn.LayerNorm(embedding_dim//2))
            dense_layers.insert(10, nn.LayerNorm(output_dim))

        # Define the output layer
        self.reduce_out = nn.Sequential(*dense_layers)

        self.output = nn.Sequential(
            nn.Linear(input_dim*output_dim, output_dim),  # Linear layer
            nn.ReLU(),  # ReLU activation
        )

        # define weights init
        self.init_weights()

    def forward(self, x):
        # reshape input tensor from [batch, 1, seq_len] to [batch, seq_len, 1]
        z = x.permute(0, 2, 1)
        # Embed the input tensor
        y = self.embedding(x)
        # Apply positional encoding to x
        z = self.positional_encoding(z)
        # sum positional encoding and embeddings
        x = z + y
        # Apply the transformer layer
        x = self.transformer(x)
        # Apply the output layer
        x = self.reduce_out(x)
        # Calculate the new dimension size
        new_dim = x.shape[1] * x.shape[2]
        # Reshape x to [batch, 1, new_dim]
        x = x.view(x.shape[0], 1, new_dim)
        x = self.output(x).squeeze(1)
        return x

    def init_weights(self):
        # Initialize ReLU layers using He Initialization
        for layer in self.embedding:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize ReLU layers using He Initialization
        for layer in self.reduce_out:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize ReLU layers using He Initialization
        for layer in self.output:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # NOTE: LayerNorm is often handled by PyTorch automatically, so we didn't initialize them
        # Transformer weights initialization is automatically handled by pytorch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # expected size --> [batch, seq_len, channels]
        #
        # Initialize a tensor "pe" filled with zeros.
        pe = torch.zeros(max_seq_length, d_model)
        # Create a sequence of numbers from 0 to (max_seq_length - 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # inverse div term to multiply with position. The formula is taken from the original Transformer paper
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
        # pe is registered as a buffer, which means it will be part of
        # the module's state but will not be considered a trainable parameter.

    def forward(self, x):
        # Adds the positional encodings to the input x: x.size(1) ensure that the
        # positional encodings match the actual sequence length of x.
        return x + self.pe[:, :x.size(1)]
