# FASA: Federated Arabic Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flower-FLWR-orange.svg)](https://flower.ai/)
[![Model](https://img.shields.io/badge/Model-AraBERT-success.svg)](https://huggingface.co/aubmindlab/bert-base-arabertv02)

## Overview
This repository contains the code and research findings for a comparative study between Centralized and Federated Learning approaches for Arabic Sentiment Analysis. The project leverages the **AraBERT** (`aubmindlab/bert-base-arabertv02`) transformer model to classify user-generated Arabic text. 

Traditional centralized training requires data aggregation, raising privacy and scalability concerns. To address this, this project implements a decentralized Federated Learning (FL) alternative, allowing collaborative model training without sharing raw data.

📄 **[Read the Full Research Paper (PDF)](./arabic_sentiment_fl.pdf)**

## Key Research Objectives
* Evaluate the performance of centralized vs. federated learning paradigms using a BERT-based model (AraBERT).
* Analyze the cross-dataset generalization capabilities on unseen external data distributions.
* Investigate the impact of noisy textual features (e.g., emojis, user mentions) by comparing model performance on original vs. cleaned data variants.

## Datasets Evaluated
The models were trained and evaluated across three distinct datasets to test in-distribution performance and cross-domain generalization:
1. **Arabic 100K Reviews (Training/Test):** User-generated reviews in Modern Standard Arabic (MSA) and dialectal Arabic, serving as the primary training distribution.
2. **SS2030 (External 1):** Arabic tweets featuring shorter text, informal language, and a high presence of noise.
3. **Arabic Twitter Sentiment Corpus (External 2):** Short, highly informal tweets with extreme dialectal variation.

## Methodology & Architecture
* **Base Model:** AraBERT equipped with a fully connected classification head and softmax activation.
* **Centralized Setup:** Trained on the aggregated 100K dataset for 10 epochs using the AdamW optimizer (Learning rate: 2e-5, Batch size: 19).
* **Federated Setup:** Implemented using the **Flower (FLWR)** framework across 3 clients. The model was trained over 10 communication rounds utilizing the **FedAvg** aggregation method.

## Key Findings & Results

### Generalization Trade-off
While the Centralized model achieved a slightly higher in-distribution accuracy on the primary test set (0.910 vs. 0.902), the Federated model demonstrated superior generalization on the unseen SS2030 dataset (0.728 vs. 0.700). The FedAvg aggregation acts as a natural global regularizer, preventing the model from overfitting to the primary training distribution.

| Dataset | Model | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Training (Test)** | Centralized | **0.910** | **0.910** |
| **Training (Test)** | Federated | 0.902 | 0.902 |
| **SS2030 (Unseen)** | Centralized | 0.700 | 0.697 |
| **SS2030 (Unseen)** | Federated | **0.728** | **0.729** |

### Noise Resilience
The study revealed that the Federated Learning paradigm is inherently more robust to user-generated noise (emojis, @mentions). 
* **Centralized:** Required data cleaning to improve performance on SS2030 (Accuracy increased from 0.700 to 0.712).
* **Federated:** Maintained highly stable performance across both raw (0.728) and cleaned (0.729) variants, indicating that collaborative weight updates are less sensitive to local textual artifacts.

## Installation & Usage

```bash
# Clone the repository
git clone [https://github.com/TananiMouhsin/FASA.git](https://github.com/TananiMouhsin/FASA.git)
cd FASA

# Install dependencies
pip install -r requirements.txt

# Run the Centralized Training Baseline
python src/train_centralized.py

# Launch the Federated Learning Simulation (Flower)
python src/run_federated.py