# ❤️ AI-Powered ECG Arrhythmia Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Model-Random%20Forest-green)
![Framework](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📌 Project Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection of arrhythmias (irregular heartbeats) is crucial for prevention. 

This project is a **Machine Learning-based Web Application** that analyzes ECG signals from CSV files to detect arrhythmias. It uses a **Random Forest Classifier** to distinguish between **Normal** and **Abnormal** heartbeats with high accuracy. The system provides an interactive dashboard for doctors and medical staff to visualize ECG signals, calculate Heart Rate (BPM), and view a risk assessment report.

---

## 🚀 Key Features
* **Instant Diagnosis:** Classifies ECG signals as 'Normal Rhythm' or 'Arrhythmia Detected' in seconds.
* **Interactive Visualization:** Zoomable ECG graphs to analyze specific heartbeats.
* **AI Beat Analysis:** Visual markers (Red/Green dots) indicating exactly where the model detected anomalies.
* **Heart Rate Calculation:** Automatically computes the patient's BPM (Beats Per Minute).
* **Risk Score:** Provides a percentage-based risk assessment based on the ratio of abnormal beats.
* **Medical-Grade Report:** Generates a professional diagnosis card suitable for screening purposes.

---

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Frontend Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Signal Processing:** SciPy (Butterworth Filters, Peak Detection)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Streamlit Charts

---

## 📂 Project Structure
ECG_Arrhythmia_Project/
│
├── app.py                   # Main Application File (Streamlit)
├── requirements.txt         # List of dependencies
├── README.md                # Project Documentation
│
├── models/                  # Trained ML Models
│   ├── rf_ecg_model.pkl     # Random Forest Model
│   └── scaler.pkl           # Standard Scaler
│
├── notebooks/               # Jupyter Notebooks for Training
│   ├── 01_data_loading.ipynb
│   ├── ...
│   └── 08_final_prediction.ipynb
│
└── data/                    # Dataset (CSV Files)
├── normal_patient.csv
└── abnormal_patient.csv

## How to Run Locally

### 1. Prerequisites Ensure you have **Python** and **Anaconda** installed on your system.

### 2. Installation Open your terminal/command prompt and navigate to the project directory:

```bash

cd path/to/ECG_Arrhythmia_Project
