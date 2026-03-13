# Technical Deep Dive: Federated Learning Architecture

This document provides a detailed explanation of the concepts and mathematical logic behind the Decentralized Heart Disease Prediction system.

## 1. Decentralization vs. Centralization
In a traditional healthcare AI setup, all hospital data is pooled into one central database. This creates a single point of failure and massive privacy concerns. 

In this project, we use **Federated Learning** to ensure:
* **Data Sovereignty:** Data never leaves the hospital's local environment.
* **Security:** Only model weight updates (mathematical gradients) are shared.
* **Efficiency:** Training happens on edge devices (the client laptops).

## 2. The FedAvg Algorithm
The central server utilizes **Federated Averaging (FedAvg)** to combine client updates. Instead of a simple average, it performs a weighted average based on the number of samples ($n_k$) each hospital contributed:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{k}$$

This ensures that a hospital with 10,000 patients has more influence on the global model than a clinic with only 100 patients.

## 3. Diagnostic Surety (The Confidence Score)
Our implementation includes a `predict()` function that goes beyond binary (0/1) labels. By applying the **Sigmoid** function to the raw model output (logits), we derive a probability percentage:

* **Logic:** `probability = torch.sigmoid(model_output)`
* **Interpretation:** "Class 1 with 82% confidence" gives healthcare providers a better sense of risk level than a simple "Heart Disease Detected."

## 4. Addressing Client Drift (Non-IID Data)
When data is **Non-IID** (e.g., Hospital A has only diabetic patients and Hospital B has only non-smokers), the local models can "drift" apart. This project explores how global aggregation can stabilize these divergent patterns into a single, robust predictor.

## 5. Computational and Storage Efficiency

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
