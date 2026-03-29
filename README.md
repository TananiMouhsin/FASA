# FASA: Federated Arabic Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flower-FLWR-orange.svg)](https://flower.ai/)
[![Model](https://img.shields.io/badge/Model-AraBERT-success.svg)](https://huggingface.co/aubmindlab/bert-base-arabertv02)

## Overview
[cite_start]This repository contains the code and research findings for a comparative study between Centralized and Federated Learning approaches for Arabic Sentiment Analysis[cite: 62, 69]. [cite_start]The project leverages the **AraBERT** (`aubmindlab/bert-base-arabertv02`) transformer model to classify user-generated Arabic text[cite: 69]. 

[cite_start]Traditional centralized training requires data aggregation, raising privacy and scalability concerns[cite: 70]. [cite_start]To address this, this project implements a decentralized Federated Learning (FL) alternative, allowing collaborative model training without sharing raw data[cite: 71].

📄 **[Read the Full Research Paper (PDF)](./arabic_sentiment_fl.pdf)**

## Key Research Objectives
* [cite_start]Evaluate the performance of centralized vs. federated learning paradigms using a BERT-based model (AraBERT)[cite: 91, 92].
* [cite_start]Analyze the cross-dataset generalization capabilities on unseen external data distributions[cite: 93].
* [cite_start]Investigate the impact of noisy textual features (e.g., emojis, user mentions) by comparing model performance on original vs. cleaned data variants[cite: 73, 94].

## Datasets Evaluated
The models were trained and evaluated across three distinct datasets to test in-distribution performance and cross-domain generalization:
1. [cite_start]**Arabic 100K Reviews (Training/Test):** User-generated reviews in Modern Standard Arabic (MSA) and dialectal Arabic, serving as the primary training distribution[cite: 105, 106, 107].
2. [cite_start]**SS2030 (External 1):** Arabic tweets featuring shorter text, informal language, and a high presence of noise[cite: 109, 111, 112].
3. [cite_start]**Arabic Twitter Sentiment Corpus (External 2):** Short, highly informal tweets with extreme dialectal variation[cite: 113, 115, 117, 118].

## Methodology & Architecture
* [cite_start]**Base Model:** AraBERT equipped with a fully connected classification head and softmax activation[cite: 137, 138, 142].
* [cite_start]**Centralized Setup:** Trained on the aggregated 100K dataset for 10 epochs using the AdamW optimizer (Learning rate: 2e-5, Batch size: 19)[cite: 147, 148, 149, 150, 151].
* [cite_start]**Federated Setup:** Implemented using the **Flower (FLWR)** framework across 3 clients[cite: 153, 155]. [cite_start]The model was trained over 10 communication rounds utilizing the **FedAvg** aggregation method[cite: 157, 158].

## Key Findings & Results

### Generalization Trade-off
[cite_start]While the Centralized model achieved a slightly higher in-distribution accuracy on the primary test set (0.910 vs. 0.902), the Federated model demonstrated superior generalization on the unseen SS2030 dataset (0.728 vs. 0.700)[cite: 177, 178, 179]. [cite_start]The FedAvg aggregation acts as a natural global regularizer, preventing the model from overfitting to the primary training distribution[cite: 179, 316, 317].

| Dataset | Model | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Training (Test)** | Centralized | **0.910** | [cite_start]**0.910** [cite: 177] |
| **Training (Test)** | Federated | 0.902 | [cite_start]0.902 [cite: 177] |
| **SS2030 (Unseen)** | Centralized | 0.700 | [cite_start]0.697 [cite: 177] |
| **SS2030 (Unseen)** | Federated | **0.728** | [cite_start]**0.729** [cite: 177] |

### Noise Resilience
[cite_start]The study revealed that the Federated Learning paradigm is inherently more robust to user-generated noise (emojis, @mentions)[cite: 321]. 
* [cite_start]**Centralized:** Required data cleaning to improve performance on SS2030 (Accuracy increased from 0.700 to 0.712)[cite: 183].
* [cite_start]**Federated:** Maintained highly stable performance across both raw (0.728) and cleaned (0.729) variants, indicating that collaborative weight updates are less sensitive to local textual artifacts[cite: 184, 321].

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