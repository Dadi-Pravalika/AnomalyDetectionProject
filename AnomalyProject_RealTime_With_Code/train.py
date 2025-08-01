
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.transformer_model import TransformerAnomalyDetector
from utils.preprocessing import load_and_preprocess

SEQ_LEN = 5
BATCH_SIZE = 4
EPOCHS = 5
DATA_PATH = "data/iscx_sample.csv"

X, Y = load_and_preprocess(DATA_PATH, SEQ_LEN)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
input_dim = X.shape[2]
model = TransformerAnomalyDetector(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = loss_fn(pred[:,-1,:], batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "transformer_model.pt")
print("✅ Model saved as transformer_model.pt")
