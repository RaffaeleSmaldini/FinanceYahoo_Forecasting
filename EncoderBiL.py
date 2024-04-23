import torch
import torch.nn as nn

class EBiL(nn.Module):
    def __init__(self, input_dim, n_head, dim_forward, embedding_dim, n_layers, output_dim,
                 hidden_dim, layer_dim, regularize=True):
        super(EBiL, self).__init__()
        self.embedding_dim = embedding_dim
        embedding_layer = [
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, embedding_dim),
            nn.ReLU()

        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            embedding_layer.insert(1, nn.LayerNorm(input_dim*2))
            embedding_layer.insert(4, nn.LayerNorm(input_dim))
            embedding_layer.insert(7, nn.LayerNorm(input_dim//2))
            embedding_layer.insert(10, nn.LayerNorm(embedding_dim))

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

        # LSTM layers
        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.1, bidirectional=True
        )

        dense_layers = [
            nn.Linear(hidden_dim * 2, hidden_dim*8),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim * 8, hidden_dim * 4),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim * 2, hidden_dim),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_dim, hidden_dim // 2),  # Linear layer
            nn.ReLU(),  # ReLU
            nn.Linear(hidden_dim // 2, output_dim),  # Linear layer
            nn.ReLU(),  # ReLU
        ]

        # Conditionally add LayerNorm layers if 'regularize' is True
        if regularize:
            dense_layers.insert(1, nn.LayerNorm(hidden_dim*8))
            dense_layers.insert(4, nn.LayerNorm(hidden_dim * 4))
            dense_layers.insert(7, nn.LayerNorm(hidden_dim * 2))
            dense_layers.insert(10, nn.LayerNorm(hidden_dim))
            dense_layers.insert(13, nn.LayerNorm(hidden_dim // 2))

        # Define the output layer
        self.output = nn.Sequential(*dense_layers)

        # define weights init
        self.init_weights()

    def forward(self, x):
        # Embed the input tensor
        x = self.embedding(x)
        # Reshape the embedded tensor
        x = x.view(-1, 1, self.embedding_dim)

        # Apply the transformer layer
        x = self.transformer(x)

        # Initializing hidden state for first input
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initializing cell state for first input
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # FP by passing: input, hidden state, and cell state into the model
        out, (hn, cn) = self.bilstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        # Apply the output layer
        out = self.output(out)
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
        for layer in self.output:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        # NOTE: LayerNorm is often handled by PyTorch automatically, so we didn't initialize them
        # Transformer weights initialization is automatically handled by pytorch
