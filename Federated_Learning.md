# Technical Deep Dive: Federated Learning Architecture

This document provides a detailed explanation of the concepts, mathematical logic, and the iterative training process behind the Decentralized Heart Disease Prediction system.

## 1. Decentralization vs. Centralization
In a traditional healthcare AI setup, all hospital data is pooled into one central database. This creates a single point of failure and massive privacy concerns. In this project, we use **Federated Learning** to ensure:
* **Data Sovereignty:** Raw data never leaves the hospital's local environment.
* **Security:** Only model weight updates (mathematical gradients) are shared.
* **Efficiency:** Training happens on edge devices (the client laptops).

## 2. The Multi-Round Training Process
The global model is built through an iterative process involving local training and global synchronization. This process repeats for multiple rounds to ensure model convergence.

### Step 1: Client Selection & Global Model Distribution
At the start of each round, the central server ensures that the required number of clients (4 hospitals) are connected. The server then broadcasts the current global model weights ($W_g$) to all participating nodes.

### Step 2: Local Training (Multiple Epochs)
Each hospital node receives the global weights and initializes its local model. The hospital then trains this model on its local, private dataset for **5 epochs**. 
* **Data Privacy:** During these epochs, the model performs multiple passes over the local CSV data.
* **Optimization:** Each hospital uses its own `DataLoader` to feed features into the PyTorch model, utilizing `BCEWithLogitsLoss` and the `Adam` optimizer to refine the weights locally.

### Step 3: Transferring Weight Gradients (Updates)
Once the 5 epochs are complete, the hospital does **not** send its dataset to the server. Instead, it only transmits the **Weight Gradients** or state dictionary updates.
* This represents the "knowledge" the model gained from the local data without exposing the data itself.
* The communication is handled via the Flower framework, which serializes the weights into NumPy arrays for efficient transfer.

### Step 4: Server Aggregation (FedAvg)
The central server waits until all 4 hospital updates are received. It then performs **Federated Averaging (FedAvg)** to merge these updates into a single new global model:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{k}$$

* **Weighted Influence:** The server performs a weighted average based on the number of samples ($n_k$) each hospital contributed. A hospital with 10,000 patients has more influence on the global model than a clinic with only 100 patients.

### Step 5: Iteration (Multiple Rounds)
This process constitutes **one Federated Round**. The server redistribution marks the start of the next round. In this project, we repeat this cycle for **10 rounds**, allowing the model to learn from the diverse distributions of all 4 hospitals until the accuracy stabilizes.

## 3. The "Zero-Knowledge" Server
A critical feature of this architecture is that the central coordinator operates with **virtually no information**:
* **No Access to Data:** The server never sees a single row of the CDC heart disease dataset. It has no access to the CSV files stored on the hospital laptops.
* **No Access to Individual Models:** The server does not store or "own" individual hospital models; it only exists to calculate the average of the received weights and then discards the individual updates. 
* **Privacy by Design:** Even if the server were compromised, an attacker would find only mathematical weight values, which cannot be "reversed" to reveal specific patient medical histories.

## 4. Diagnostic Surety (The Confidence Score)
Our implementation includes a `predict()` function that goes beyond binary (0/1) labels. By applying the **Sigmoid** function to the raw model output (logits), we derive a probability percentage:

* **Logic:** `probability = torch.sigmoid(model_output)`
* **Interpretation:** "Class 1 with 82% confidence" gives healthcare providers a better sense of risk level than a simple "Heart Disease Detected."

## 5. Addressing Client Drift (Non-IID Data)
When data is **Non-IID** (e.g., Hospital A has only diabetic patients and Hospital B has only non-smokers), the local models can "drift" apart. This project explores how global aggregation can stabilize these divergent patterns into a single, robust predictor.

## 6. Computational and Storage Efficiency
Beyond the primary goal of data privacy, this Federated architecture provides significant infrastructure advantages over traditional centralized systems:

### Distributed Computation
In a centralized setup, the server must handle backpropagation and gradient calculations for the entire global dataset (millions of rows), requiring high-end GPU clusters and massive RAM. 
* **Federated Advantage:** The "heavy lifting" is distributed across client nodes (hospital laptops). The central server acts only as a coordinator, performing lightweight weight aggregation (averaging), which can be done on basic hardware.

### Minimal Storage Requirements
Centralizing medical data requires massive "Data Lakes" and expensive cloud storage buckets to hold copies of all patient records.
* **Federated Advantage:** The server never stores a single byte of patient data. It only holds the current model weights (a few megabytes). This eliminates the need for expensive central storage infrastructure.

### Bandwidth Optimization
Moving raw medical data (GBs of CSVs or images) across a network is slow and expensive. 
* **Federated Advantage:** Clients only transmit model parameters (mathematical weights), which are significantly smaller than the source datasets. This makes the system feasible even for facilities with limited internet bandwidth.

### Infrastructure Comparison

| Resource | Centralized Learning | Federated Learning |
| :--- | :--- | :--- |
| **Server Hardware** | High-end GPU / High RAM | Standard CPU / Low RAM |
| **Storage Cost** | High (Stores all raw data) | **Near Zero** (Stores weights only) |
| **Network Load** | High (Transfers raw data) | Low (Transfers model updates) |
| **Scalability** | Expensive (Needs more server power) | **Easy** (Uses existing client power) |

## 7. Future Scope
While the current implementation utilizes the **FedAvg** strategy for weight aggregation, future iterations of this project will explore **FedProx**. FedProx introduces a proximal term to the local objective function, which helps mitigate "Client Drift" in highly Non-IID (unbalanced) healthcare datasets, ensuring more stable global model convergence.
