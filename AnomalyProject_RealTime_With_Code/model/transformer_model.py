
import torch.nn as nn

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=2, hidden_dim=64):
        super(TransformerAnomalyDetector, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return self.fc(x)
