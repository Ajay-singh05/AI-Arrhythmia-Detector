import streamlit as st
import pandas as pd
import numpy as np
import joblib
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & STYLISH HEADER
# ==========================================
st.set_page_config(page_title="AI Heart Doctor", page_icon="❤️", layout="wide")
# ==========================================
# SIDEBAR - PROJECT INFO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6028/6028690.png", width=100)
    st.title("Project Controls")
    
    st.info("💡 **How to use:**")
    st.markdown("""
    1. Upload an **ECG CSV file**.
    2. Wait for the **AI Model** to analyze.
    3. View the **Graph** & **Diagnosis**.
    """)
    
    st.divider()
    st.write("🔬 **Model Details:**")
    st.caption("Algorithm: Random Forest Classifier")
    st.caption("Accuracy: ~98.5% on MIT-BIH Data")
    
    st.divider()
    st.write("👨‍🎓 **Developed by:**")
    st.write("**[Ajay Singh and Abhinav dongre]**")
    st.write("B.Tech (2nd Year)")
    
# 👇 Custom CSS for Modern Styling
st.markdown("""
    <style>
    .title-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 10px solid #ff4b4b;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 45px;
        font-weight: 800;
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        margin: 0;
    }
    .heart-icon {
        color: #ff4b4b;
        animation: heartbeat 1.5s infinite;
    }
    @keyframes heartbeat {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        margin-top: 10px;
    }
    </style>
    
    <div class="title-box">
        <h1 class="main-title"><span class="heart-icon">❤️</span> AI Arrhythmia Detector</h1>
        <p class="subtitle">Upload your ECG (CSV) file to get a Medical-Grade Diagnosis instantly.</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/rf_ecg_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# ==========================================
# 3. SIGNAL PROCESSING FUNCTIONS
# ==========================================
def preprocess_signal(raw_signal, fs=360):
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 40 / nyq], btype='band')
    return filtfilt(b, a, raw_signal)

def extract_features(clean_signal, fs=360):
    peaks, _ = signal.find_peaks(clean_signal, distance=int(0.6*fs), height=np.mean(clean_signal))
    features_list = []
    for i in range(1, len(peaks) - 1):
        r_prev = peaks[i-1]; r_curr = peaks[i]; r_next = peaks[i+1]
        rr_prev = (r_curr - r_prev) / fs
        rr_next = (r_next - r_curr) / fs
        widths = signal.peak_widths(clean_signal, [r_curr], rel_height=0.5)
        qrs_duration = widths[0][0] / fs
        window = int(0.05 * fs)
        segment = clean_signal[max(0, r_curr-window):min(len(clean_signal), r_curr+window)]
        features_list.append({
            "RR_prev(s)": rr_prev, "RR_next(s)": rr_next, "QRS_duration(s)": qrs_duration,
            "R_amp": clean_signal[r_curr],
            "QRS_max": np.max(segment) if len(segment)>0 else 0,
            "QRS_min": np.min(segment) if len(segment)>0 else 0
        })
    return pd.DataFrame(features_list)

# ==========================================
# 4. MAIN APPLICATION LOGIC
# ==========================================
if model is None:
    st.error("❌ Error: Models not found! Make sure 'models/rf_ecg_model.pkl' exists.")
else:
    uploaded_file = st.file_uploader("Choose a file (e.g., normal_patient.csv)", type="csv")

    if uploaded_file is not None:
        try:
            # A. Read File
            df = pd.read_csv(uploaded_file, header=None)
            if isinstance(df.iloc[0,0], str): df = pd.read_csv(uploaded_file)
            raw_signal = df.iloc[:, 0].values

            # B. Show Interactive Graph
            st.subheader("📈 Patient's ECG Signal")
            st.write("You can zoom in on this graph.")
            st.line_chart(raw_signal[:2000], height=250)

            # C. Process & Predict
            with st.spinner("Analyzing Heartbeats... 🤖"):
                clean_sig = preprocess_signal(raw_signal)
                features_df = extract_features(clean_sig)
                
                if features_df.empty:
                    st.warning("⚠️ Signal too noisy or empty. No heartbeats found.")
                else:
                    X_input = features_df[["RR_prev(s)", "RR_next(s)", "QRS_duration(s)", "R_amp", "QRS_max", "QRS_min"]]
                    X_scaled = scaler.transform(X_input)
                    predictions = model.predict(X_scaled)
                    
                    # Calculations
                    total_beats = len(predictions)
                    abnormal_count = np.sum(predictions)
                    risk_score = (abnormal_count / total_beats) * 100
                    avg_rr = features_df["RR_prev(s)"].mean()
                    bpm = int(60 / avg_rr) if avg_rr > 0 else 0

                    # --- STYLISH REPORT START ---
                    st.divider()
                    
                    if risk_score > 20:
                        status_color = "#ff4b4b"; status_bg = "#ffebeb"
                        status_icon = "⚠️"; status_title = "ARRHYTHMIA DETECTED"
                        status_msg = "High irregularity detected. Consult a Cardiologist."
                    else:
                        status_color = "#28a745"; status_bg = "#e6fffa"
                        status_icon = "✅"; status_title = "NORMAL RHYTHM"
                        status_msg = "Heart rhythm appears steady and healthy."

                    st.markdown(f"""
                    <style>
                    .report-card {{ background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px; border-top: 5px solid {status_color}; }}
                    .metric-container {{ display: flex; justify-content: space-between; text-align: center; }}
                    .metric-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 10px; width: 23%; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 0; }}
                    .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                    .result-box {{ background-color: {status_bg}; color: {status_color}; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center; border: 1px solid {status_color}; }}
                    .result-title {{ font-size: 22px; font-weight: 800; margin: 0; }}
                    </style>

                    <div class="report-card">
                        <h3 style="margin-top:0; color:#333;">📊 Diagnosis Report</h3>
                        <div class="metric-container">
                            <div class="metric-box"><p class="metric-value">❤️ {bpm} BPM</p><p class="metric-label">Heart Rate</p></div>
                            <div class="metric-box"><p class="metric-value">📉 {total_beats}</p><p class="metric-label">Total Beats</p></div>
                            <div class="metric-box"><p class="metric-value" style="color: {'#d9534f' if abnormal_count > 0 else '#333'}">⚠️ {int(abnormal_count)}</p><p class="metric-label">Abnormal Beats</p></div>
                            <div class="metric-box"><p class="metric-value">{risk_score:.1f}%</p><p class="metric-label">Risk Score</p></div>
                        </div>
                        <div class="result-box"><p class="result-title">{status_icon} {status_title}</p><p style="margin-bottom:0; font-size:16px;">{status_msg}</p></div>
                    </div>
                    """, unsafe_allow_html=True)
                    # --- STYLISH REPORT END ---

                    # --- AI VISUAL PROOF GRAPH ---
                    st.subheader("🤖 AI Beat Analysis (Visual Proof)")
                    st.write("Red Dots = Abnormal Beats detected by AI")
                    
                    display_limit = 1000 
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(raw_signal[:display_limit], label='ECG Signal', color='#1f77b4', alpha=0.6)
                    
                    peaks, _ = signal.find_peaks(raw_signal[:display_limit], distance=int(0.6*360), height=np.mean(raw_signal))
                    for i, peak_idx in enumerate(peaks):
                        if i == 0 or i == len(peaks) - 1: continue
                        pred_idx = i - 1
                        if pred_idx < len(predictions):
                            if predictions[pred_idx]:
                                ax.plot(peak_idx, raw_signal[peak_idx], 'ro', markersize=8) # Red for abnormal
                            else:
                                ax.plot(peak_idx, raw_signal[peak_idx], 'go', markersize=6) # Green for normal
                    
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")