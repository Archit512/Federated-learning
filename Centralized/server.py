from flask import Flask, request, jsonify
import os
from train_model import train_centralized_model

app = Flask(__name__)

RECEIVED_DATA_DIR = os.path.join(os.path.dirname(__file__), "received_data")
os.makedirs(RECEIVED_DATA_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or "hospital_name" not in request.form:
        return jsonify({"error": "Missing file or hospital_name"}), 400

    file = request.files["file"]
    hospital_name = request.form["hospital_name"]

    save_path = os.path.join(RECEIVED_DATA_DIR, f"{hospital_name}.csv")
    file.save(save_path)

    print(f"[SERVER] Received dataset from {hospital_name} → saved to {save_path}")
    return jsonify({"message": f"Dataset from {hospital_name} received successfully."}), 200


@app.route("/train", methods=["GET"])
def train():
    print("[SERVER] Training triggered...")
    try:
        accuracy = train_centralized_model(RECEIVED_DATA_DIR)
        return jsonify({"centralized_accuracy": round(accuracy, 4)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[SERVER] Flask Centralized Server running on port 5000...")
    app.run(host="0.0.0.0", port=5000)