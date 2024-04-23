import torch.nn as nn
import torch
class BiL(nn.Module):
    def __init__(self, input_dim, bottleneck, hidden_dim, layer_dim, output_dim,
                 self_attention_heads=8, regularize=False):
        super(BiL, self).__init__()

        # Define the self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=bottleneck, num_heads=self_attention_heads)

        bott_layers = [
            nn.Linear(input_dim, input_dim * 2),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(input_dim * 2, input_dim),  # Linear layer
            nn.ReLU(),  # ReLU
            nn.Linear(input_dim, bottleneck),  # Linear layer
            nn.ReLU()  # ReLU activation
        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            bott_layers.insert(1, nn.LayerNorm(input_dim * 2))
            bott_layers.insert(4, nn.LayerNorm(input_dim))
            bott_layers.insert(7, nn.LayerNorm(bottleneck))

        # Create the sequential block with the defined layers
        self.bottleneck_seq = nn.Sequential(*bott_layers)

        # Defining the number of layers and the nodes in each layer of the BiLSTM
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.bilstm = nn.LSTM(
            bottleneck, hidden_dim, layer_dim, batch_first=True, dropout=0.1, bidirectional=True
        )

        dense_layers = [
            nn.Linear(hidden_dim * 2, hidden_dim),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim, hidden_dim // 2),  # Linear layer
            nn.ReLU(),  # ReLU
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Linear layer
            nn.ReLU()  # ReLU activation
        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            dense_layers.insert(1, nn.LayerNorm(hidden_dim))
            dense_layers.insert(4, nn.LayerNorm(hidden_dim // 2))
            dense_layers.insert(7, nn.LayerNorm(hidden_dim // 4))

        # Create the sequential block with the defined layers
        self.dense_seq = nn.Sequential(*dense_layers)
        # Fully connected layer
        self.out_fc = nn.Linear(hidden_dim//4, output_dim)
        # init weights
        self.init_weights()

    def forward(self, x):
        # capturing invariant feature and reducing dimensionality
        x = self.bottleneck_seq(x)

        # Apply self-attention
        attention_output, _ = self.self_attention(x, x, x)

        # Add the attention output to the original input sequence
        x = x + attention_output
        # Initializing hidden state for first input
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initializing cell state for first input
        c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # FP by passing: input, hidden state, and cell state into the model
        out, (hn, cn) = self.bilstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]

        # dense network
        out = self.dense_seq(out)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.out_fc(out)
        return out

    def init_weights(self):
        # Initialize ReLU layers using He Initialization
        for layer in self.bottleneck_seq:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # Initialize LSTM layers using Orthogonal Initialization
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        # Initialize ReLU layers using He Initialization
        for layer in self.dense_seq:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # NOTE: LayerNorm is often handled by PyTorch automatically, so we didn't initialize them
        # "" for self attention
