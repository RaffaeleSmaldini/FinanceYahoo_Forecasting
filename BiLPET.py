import torch
import torch.nn as nn
from ToM import PositionalEncoding
"""
The following code is a modified version of EBiL with Positional Encoding and Embedding layers from ToM
"""
class BiLPET(nn.Module):
    def __init__(self, input_dim, bottleneck, n_head, dim_forward, embedding_dim, n_layers, output_dim,
                 hidden_dim, layer_dim, cross_attention_heads, regularize=True):
        super(BiLPET, self).__init__()
        embedding_layer = [
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU()

        ]

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU()
        )

        # create a positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim, max_seq_length=input_dim)

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            embedding_layer.insert(1, nn.LayerNorm(input_dim*4))
            embedding_layer.insert(4, nn.LayerNorm(embedding_dim*2))
            embedding_layer.insert(7, nn.LayerNorm(embedding_dim))

        # Define the embedding layer
        self.embedding = nn.Sequential(*embedding_layer)

        # Define the transformer' encoder layer
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

        # Defining the number of layers and the nodes in each layer of the BiLSTM
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
        # LSTM layers
        self.bilstm = nn.LSTM(
            bottleneck, hidden_dim, layer_dim, batch_first=True, dropout=0.1, bidirectional=True
        )

        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -#
        # # # # dense after LSTM
        dense_layers = [
            nn.Linear(hidden_dim * 2, hidden_dim*8),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim * 8, hidden_dim * 4),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim * 2, hidden_dim//2),  # Linear layer
            nn.ReLU(),  # ReLU activation
        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            dense_layers.insert(1, nn.LayerNorm(hidden_dim*8))
            dense_layers.insert(4, nn.LayerNorm(hidden_dim * 4))
            dense_layers.insert(7, nn.LayerNorm(hidden_dim * 2))
            dense_layers.insert(10, nn.LayerNorm(hidden_dim//2))

        # Define the output layer
        self.dense_LSTM = nn.Sequential(*dense_layers)

        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -#
        # # # # dense after Encoder Transformer
        dense_layers = [
            nn.Linear(embedding_dim, embedding_dim*4),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(embedding_dim * 4, embedding_dim),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(embedding_dim, hidden_dim//2),  # Linear layer
            nn.ReLU(),  # ReLU
        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            dense_layers.insert(1, nn.LayerNorm(embedding_dim*4))
            dense_layers.insert(4, nn.LayerNorm(embedding_dim))
            dense_layers.insert(7, nn.LayerNorm(hidden_dim//2))

        # Define the output layer
        self.dense_TEnc = nn.Sequential(*dense_layers)

        self.reshape_TEnc = nn.Sequential(
            nn.Linear(input_dim*hidden_dim//2, hidden_dim//2),  # Linear layer
            nn.ReLU(),  # ReLU activation
        )
        # # # # attention
        self.multihead_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim//2,
            num_heads=cross_attention_heads
        )
        # # # # output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim*4),  # Linear layer
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim*4, output_dim),  # Linear layer
            nn.ReLU(),  # ReLU activation
        )
        # define weights init
        self.init_weights()

    def forward(self, x):
        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -# TRANSFORMER PATH
        # reshape input tensor from [batch, 1, seq_len] to [batch, seq_len, 1]
        z = x.permute(0, 2, 1)
        # Embed the input tensor
        y = self.embedding(x)
        # Apply positional encoding to x
        z = self.positional_encoding(z)
        # sum positional encoding and embeddings
        x_transfomer = z + y

        del y, z  # Delete tensors that are no longer needed
        # Apply the transformer layer
        x_transfomer = self.transformer(x_transfomer)

        out_TEnc = self.dense_TEnc(x_transfomer).squeeze(1)
        del x_transfomer  # Delete tensors that are no longer needed

        # Calculate the new dimension size
        new_dim = out_TEnc.shape[1] * out_TEnc.shape[2]
        # Reshape x to [batch, 1, new_dim]
        out_TEnc = out_TEnc.view(x.shape[0], 1, new_dim)
        out_TEnc = self.reshape_TEnc(out_TEnc).squeeze(1)      # dim :  [batch, sequence]
        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -#  LSTM PATH
        x_lstm = self.bottleneck(x)
        # Initializing hidden state for first input
        h0 = torch.zeros(self.layer_dim * 2, x_lstm.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initializing cell state for first input
        c0 = torch.zeros(self.layer_dim * 2, x_lstm.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # FP by passing: input, hidden state, and cell state into the model
        out, (hn, cn) = self.bilstm(x_lstm, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        # Apply the output layer
        out_LSTM = self.dense_LSTM(out)     # dim : [batch, sequence]
        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -# CROSS ATTENTION
        # Perform cross-attention
        cross_attention_output, _ = self.multihead_cross_attention(
            query=out_TEnc,
            key=out_LSTM,
            value=out_LSTM
        )
        del out_LSTM  # Delete tensors that are no longer needed
        # Add the cross-attention output to the Transformer output
        combined_output = out_TEnc + cross_attention_output
        del out_TEnc, cross_attention_output  # Delete tensors that are no longer needed
        # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -# OUTPUT
        out = self.output(combined_output)
        return out

    def init_weights(self):
        # Initialize ReLU layers using He Initialization
        for layer in self.embedding:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize LSTM layers using Orthogonal Initialization
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        # Initialize ReLU layers using He Initialization
        for layer in self.dense_LSTM:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize ReLU layers using He Initialization
        for layer in self.bottleneck:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize ReLU layers using He Initialization
        for layer in self.reshape_TEnc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize ReLU layers using He Initialization
        for layer in self.output:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # NOTE: LayerNorm is often handled by PyTorch automatically, so we didn't initialize them
        # Transformer weights initialization is automatically handled by pytorch