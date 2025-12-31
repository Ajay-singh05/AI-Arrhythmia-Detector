# 🫀 AI Arrhythmia Detector (Logistic Regression)

An end-to-end **AI-based ECG Arrhythmia Detection system** built using **Logistic Regression**. This project focuses on preprocessing raw ECG signals, extracting meaningful features, and classifying heartbeats to detect arrhythmias with high accuracy.

---

## 📌 Project Overview

Cardiac arrhythmias are irregular heart rhythms that can be life-threatening if not detected early. This project leverages **machine learning (Logistic Regression)** to automatically classify ECG signals and assist in early diagnosis.

The system performs:

* ECG signal preprocessing
* Feature extraction
* Model training using Logistic Regression
* Performance evaluation

---

## 🚀 Features

* ECG signal filtering and normalization
* Feature extraction from ECG signals
* Binary / multi-class arrhythmia classification
* Trained Logistic Regression model
* Evaluation using accuracy, precision, recall, and confusion matrix
* Modular and well-documented Jupyter notebooks

---

## 🧠 Machine Learning Model

* **Algorithm:** Logistic Regression
* **Why Logistic Regression?**

  * Simple and interpretable
  * Works well for medical classification problems
  * Fast training and inference

---

## 📂 Project Structure

```
AI-Arrhythmia-Detector/
│
├── data/
│   ├── raw/                # Raw ECG data
│   ├── processed/          # Preprocessed signals
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_signal_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_model_saving.ipynb
│   └── 07_end_to_end_pipeline.ipynb
│
├── models/
│   └── logistic_regression_model.pkl
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset

* ECG data sourced from publicly available datasets (e.g., MIT-BIH Arrhythmia Dataset)
* Signals include both normal and arrhythmic heartbeats

> **Note:** Ensure dataset licensing is followed before reuse.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-arrhythmia-detector.git
cd ai-arrhythmia-detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebooks

Open Jupyter Notebook or Jupyter Lab and run the notebooks in sequence:

```bash
jupyter notebook
```

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Sample performance:

* **Accuracy:** ~98% (may vary based on dataset and preprocessing)

---

## 🧪 Results

* Logistic Regression successfully classifies ECG signals
* High accuracy with proper preprocessing
* Demonstrates feasibility of ML-based ECG diagnosis

---

## 🛠️ Tools & Technologies

* Python 🐍
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Matplotlib
* WFDB
* Jupyter Notebook

---

## 🔮 Future Improvements

* Use advanced models (Random Forest, SVM, CNN, LSTM)
* Real-time ECG signal classification
* Web or mobile application interface
* Multi-class arrhythmia detection
* Model explainability (SHAP / LIME)

---

## 👨‍💻 Contributors

* **Abhinav Dongre** – Project Development & ML Pipeline
* **Ajay Singh** – Project Development & ML Pipeline

---



## ⭐ Acknowledgements

* MIT-BIH Arrhythmia Database
* Scikit-learn Documentation
* Open-source ML community



> 💡 *This project is intended for academic and research purposes only and should not replace professional medical dia
