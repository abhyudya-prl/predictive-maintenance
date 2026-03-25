# 🔧 Multimodal Predictive Maintenance AI

> An end-to-end AI system that combines **LSTM sensor analysis**, **NLP maintenance log classification**, and **CNN image defect detection** into a unified engine health scoring engine — deployed as a live interactive dashboard.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)

---

## 📌 Project Overview

Traditional predictive maintenance uses only sensor data. This project goes further — it fuses **three independent AI models** trained on different data modalities to produce a single, explainable risk score (0–100) per engine, along with a recommended maintenance action.

| Model | Input | Output |
|---|---|---|
| 🧠 LSTM (PyTorch) | 30-cycle sensor time-series | Remaining Useful Life (cycles) |
| 📝 NLP (TF-IDF + Logistic Regression) | Free-text maintenance log | Fault type + severity |
| 🖼️ CNN (ResNet18 Transfer Learning) | Part inspection image | Healthy / Worn / Damaged |
| ⚡ Fusion Engine | All 3 model outputs | Risk score 0–100 + action |

---

## 🏗️ Architecture

```
Data Sources
├── Sensor data (NASA CMAPSS FD001)     → LSTM Model      → RUL prediction
├── Maintenance logs (synthetic)         → NLP Classifier  → Fault type
└── Part images (synthetic)              → CNN (ResNet18)  → Condition

                    ↓ Weighted Fusion (50% LSTM, 25% NLP, 25% CNN) ↓

              Unified Risk Score (0–100) + Maintenance Decision
```

---

## 📊 Results

| Model | Metric | Score |
|---|---|---|
| LSTM | RMSE | **15.28 cycles** |
| LSTM | MAE | **11.78 cycles** |
| NLP Classifier | Accuracy | 100% (synthetic data) |
| CNN (ResNet18) | Accuracy | 100% (synthetic data) |

> The LSTM was trained on the real NASA CMAPSS FD001 dataset (20,631 sensor readings across 100 engines). RMSE of 15.28 is competitive with research paper benchmarks (12–18 range).

---

## 🗂️ Project Structure

```
predictive-maintenance/
├── data/
│   ├── train_FD001.txt          ← NASA CMAPSS dataset
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── X_train.npy              ← Preprocessed sequences
│   ├── y_train.npy
│   └── maintenance_logs.csv     ← Synthetic NLP dataset
├── notebooks/
│   ├── 01_eda.ipynb             ← Phase 1: Data & EDA
│   ├── 02_lstm.ipynb            ← Phase 2: LSTM training
│   ├── 03_nlp.ipynb             ← Phase 3: NLP classifier
│   ├── 04_cnn.ipynb             ← Phase 4: CNN training
│   └── 05_fusion.ipynb          ← Phase 5: Fusion engine
├── models/
│   ├── lstm_best.pt             ← Trained LSTM weights
│   ├── cnn_best.pt              ← Trained ResNet18 weights
│   ├── nlp_classifier.pkl       ← TF-IDF + Logistic Regression
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
├── app/
│   └── dashboard.py             ← Phase 6: Streamlit dashboard
└── requirements.txt
```

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, torchvision (ResNet18)
- **ML:** scikit-learn (TF-IDF, Logistic Regression, MinMaxScaler)
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Dataset:** NASA CMAPSS FD001 (Turbofan Engine Degradation Simulation)

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Get the NASA CMAPSS dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place these files in `data/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

### 5. Run the notebooks in order
```
notebooks/01_eda.ipynb      → generates data/X_train.npy
notebooks/02_lstm.ipynb     → generates models/lstm_best.pt
notebooks/03_nlp.ipynb      → generates models/nlp_classifier.pkl
notebooks/04_cnn.ipynb      → generates models/cnn_best.pt
notebooks/05_fusion.ipynb   → tests the full pipeline
```

### 6. Launch the dashboard
```bash
cd app
streamlit run dashboard.py
```
Opens at `http://localhost:8501`

---

## 🧠 Key Concepts Demonstrated

- **Time-series deep learning** — LSTM with sliding window sequences on sensor data
- **Transfer learning** — ResNet18 fine-tuned for 3-class defect classification
- **NLP pipeline** — TF-IDF vectorisation + classification on maintenance logs
- **Multimodal fusion** — weighted ensemble combining 3 heterogeneous model outputs
- **Explainability** — per-model risk contribution breakdown shown in dashboard
- **Business logic layer** — raw ML outputs translated into actionable maintenance decisions

---

## 📈 Business Impact

| Scenario | Without AI | With This System |
|---|---|---|
| Engine failure detection | After breakdown | 12 cycles before failure |
| Fault identification | Manual log review | Instant NLP classification |
| Part inspection | Human visual check | Automated CNN scoring |
| Decision making | Engineer judgment | Risk score + recommended action |

---

## 🎓 About

Built as a portfolio project to demonstrate practical AI/ML skills across deep learning, NLP, and computer vision — applied to a real industrial domain.

**Dataset:** [NASA Prognostics Center — CMAPSS](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
