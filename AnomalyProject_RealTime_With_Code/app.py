from flask import Flask, jsonify, render_template
import torch
import pandas as pd
from scapy.all import sniff, get_if_list

from model.transformer_model import TransformerAnomalyDetector
from utils.preprocessing import normalize_data

app = Flask(__name__)

# Load model
model = TransformerAnomalyDetector(input_dim=6)
model.load_state_dict(torch.load("transformer_model.pt"))  # adjust path if needed
model.eval()

# STEP 1: Print all available interfaces
print("Available interfaces:")
for iface in get_if_list():
    print(" 👉", iface)

# STEP 2: Choose your interface (Pick one from above)
INTERFACE = r"\Device\NPF_{8666D0E3-D2AC-4632-A4DE-EB6B30C18FB8}"
  # ✅ Valid format

# Convert packet to feature vector (you can customize this further)
def packet_to_features(pkt):
    return [len(pkt), pkt.time % 60, 0, 0, 0, 0]

@app.route("/predict", methods=["GET"])
def predict():
    try:
        print("🔍 Sniffing real packets...")
        packets = sniff(count=10, timeout=5, iface=INTERFACE)

        if not packets:
            return jsonify({"status": "error", "message": "No packets captured. Try different interface or increase timeout."})

        print(f"✅ Captured {len(packets)} packets")
        features = [packet_to_features(pkt) for pkt in packets]
        df = pd.DataFrame(features)
        norm_df = normalize_data(df)
        X = torch.tensor(norm_df.values).float().unsqueeze(0)

        with torch.no_grad():
            prediction = model(X)
        error = torch.mean((prediction - X) ** 2).item()
        status = "Anomaly" if error > 0.02 else "Normal"

        print(f"📊 Prediction complete: Error = {error}, Status = {status}")
        return jsonify({"status": status, "error": error})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
