import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import warnings
import time
import requests
from streamlit_lottie import st_lottie
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ============================================================
#                  CONFIGURATION & ASSETS
# ============================================================
st.set_page_config(
    page_title="Aural Sentiment Engine",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'probs' not in st.session_state:
    st.session_state.probs = None
if 'pred_idx' not in st.session_state:
    st.session_state.pred_idx = None
if 'pred_label' not in st.session_state:
    st.session_state.pred_label = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Models will be loaded directly from the local 'models/' directory

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Animation URLs
lottie_scanning = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_ndj96q.json") # Scanner
lottie_wave = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_mkmfioqr.json") # Audio wave

# ============================================================
#                        PREMIUM STYLING
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
        color: #e9ecef;
        overflow: hidden; /* Prevent global scrollbars */
    }

    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0rem !important;
        margin-top: -12rem !important;
        height: 100vh;
        overflow-y: auto; /* Allow internal scrolling if needed, but hide bar */
        scrollbar-width: none; /* Firefox */
        -ms-overflow-style: none; /* IE 10+ */
    }
    [data-testid="stAppViewBlockContainer"]::-webkit-scrollbar {
        display: none; /* Chrome/Safari */
    }

    /* Hero Title */
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -2.5rem !important;
        margin-bottom: -0.5rem;
        letter-spacing: 2px;
    }

    .hero-subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* native containers as cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        transition: all 0.4s ease !important;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 210, 255, 0.3) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.15) !important;
    }

    /* Result Glow */
    .result-glow {
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        background: rgba(0, 210, 255, 0.05);
        border: 1px solid rgba(0, 210, 255, 0.2);
        margin: 0.5rem 0;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Navigation Item Spacing - Persistent & Spacious */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]) {
        padding: 1.2rem 1.2rem !important; /* Balanced vertical/horizontal spacing */
        margin: 1.2rem 1.2rem 1.2rem 1.2rem !important; /* Centering the box */
        border-radius: 12px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        width: calc(100% - 1rem) !important; /* Fill width with breathing room */
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }
    
    /* Center the radio item content to balance the left/right gaps */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div[data-testid="stMarkdownContainer"] {
        margin-left: 0.5rem !important;
    }
    
    /* Specifically hide the empty radio label container to remove the extra box */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
        display: none !important;
    }
    
    /* Blue Hover Transition - Restored & Enhanced */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:not([data-testid="stWidgetLabel"]):hover {
        background: rgba(0, 210, 255, 0.15) !important;
        border: 1px solid rgba(0, 210, 255, 0.4) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 5px 20px rgba(0, 210, 255, 0.2) !important;
    }
    
    /* Active State Glow */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] {
        background: rgba(0, 210, 255, 0.2) !important;
        border: 1px solid rgba(0, 210, 255, 0.5) !important;
    }
    
    /* Radio item container full width */
    div[data-testid="stRadio"] {
        width: 100% !important;
    }
    div[data-testid="stRadio"] > div {
        width: 100% !important;
    }

    /* Navigation Rail mode icons visibility */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 90px !important;
        max-width: 90px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stRadio"] label {
        justify-content: center !important;
        padding: 2rem 0 !important;
        transform: none !important;
    }

    /* Button Styling - Force Blue Gradient */
    .stButton>button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff) !important;
        color: white !important;
        border: 1px solid rgba(0, 210, 255, 0.3) !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(0, 210, 255, 0.6) !important;
        transform: translateY(-3px) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }

    /* Catchy File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(0, 210, 255, 0.03) !important;
        border: 2px dashed rgba(0, 210, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border: 2px solid #00d2ff !important;
        background: rgba(0, 210, 255, 0.05) !important;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.1) !important;
    }
    
    /* File icon & text color */
    [data-testid="stFileUploadDropzone"] {
        color: #e9ecef !important;
    }
    
    /* Customization for the small 'Browse' button inside uploader */
    [data-testid="stFileUploadDropzone"] button {
        background: rgba(0, 210, 255, 0.1) !important;
        border: 1px solid rgba(0, 210, 255, 0.4) !important;
        color: #00d2ff !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        width: auto !important;
        padding: 0.4rem 1rem !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    header {background-color: rgba(0,0,0,0) !important;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
#                  CORE LOGIC / CACHING
# ============================================================
@st.cache_resource
def get_assets():
    model = load_model("models/AuralSentimentEngine_best.keras")
    scaler = joblib.load("models/scaler.joblib")
    encoder = joblib.load("models/encoder.joblib")
    return model, scaler, encoder

def extract_features_from_array(y, sr=16000):
    y = librosa.util.normalize(y)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.hstack([mfccs, chroma, contrast, zcr])

model, scaler, encoder = get_assets()

# ============================================================
#                        SIDEBAR NAV
# ============================================================
st.sidebar.markdown(f"<h1 style='font-family: Orbitron; text-align: center; font-size: 1.2rem; margin-top: -1rem; margin-bottom: 2rem;'>ASE</h1>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Home Analyze", "📊 Visual Insights", "🧠 Explainability"],
    label_visibility="collapsed"
)

# ============================================================
#                           HOME PAGE
# ============================================================
if page == "🏠 Home Analyze":
    st.markdown('<div class="hero-title">Aural Sentiment Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Advanced Neural Analysis of Human Vocal Emotion</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("📁 Data Ingestion")
        uploaded_file = st.file_uploader(
            "Drop a .wav voice file for neural processing", 
            type=["wav"],
            key=f"uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🚀 EXECUTE NEURAL SCAN"):
                    with st.empty():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if lottie_scanning:
                                st_lottie(lottie_scanning, height=250, key="scan_anim")
                        
                        y, sr = librosa.load(uploaded_file, sr=16000)
                        features = extract_features_from_array(y, sr)
                        X = scaler.transform([features])
                        probs = model.predict(X, verbose=0)[0]
                        
                        st.session_state.probs = probs
                        st.session_state.X = X
                        st.session_state.pred_idx = np.argmax(probs)
                        st.session_state.pred_label = encoder.inverse_transform([st.session_state.pred_idx])[0]
                        st.rerun()
            
            with c2:
                if st.button("🗑️ CLEAR SESSION"):
                    st.session_state.probs = None
                    st.session_state.X = None
                    st.session_state.uploader_key += 1
                    st.rerun()

    # Results Display area
    if st.session_state.probs is not None:
        with st.container(border=True):
            st.subheader("🎯 Neural Inference Report")
            
            res_l, res_r = st.columns([1, 2])
            with res_l:
                st.markdown(f"""
                    <div class="result-glow">
                        <p style="color: #8892b0; margin-bottom: 0;">PREDICTED DOMINANT EMOTION</p>
                        <h1 style="color: #00d2ff; font-family: Orbitron; margin-bottom: 0; font-size: 2rem;">{st.session_state.pred_label.upper()}</h1>
                        <h2 style="color: #3a7bd5;">{st.session_state.probs[st.session_state.pred_idx]:.2%} Confidence</h2>
                    </div>
                """, unsafe_allow_html=True)
                st.info("💡 Switch to the 'Explainability' tab via the sidebar for deep analysis.")
                
            with res_r:
                df_probs = pd.DataFrame({"Emotion": encoder.classes_, "Probability": st.session_state.probs}).set_index("Emotion")
                st.bar_chart(df_probs, color="#3a7bd5", height=280)

# ============================================================
#                        INSIGHTS PAGE
# ============================================================
elif page == "📊 Visual Insights":
    st.markdown('<div class="hero-title" style="font-size: 3rem; margin-top: -1rem;">Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle" style="margin-bottom: 1.5rem;">Global Performance Analysis</p>', unsafe_allow_html=True)

    ins1, ins2 = st.columns(2)
    with ins1:
        with st.container(border=True):
            st.markdown("##### 🎯 Confusion Matrix")
            st.image("visuals/confusion_matrix.png", use_container_width=True)

    with ins2:
        with st.container(border=True):
            st.markdown("##### 📉 Multi-Class ROC")
            st.image("visuals/roc_curve.png", use_container_width=True)

# ============================================================
#                          XAI PAGE
# ============================================================
elif page == "🧠 Explainability":
    st.markdown('<div class="hero-title" style="font-size: 2.5rem;">Explainability Lab</div>', unsafe_allow_html=True)
    
    if st.session_state.X is None:
        with st.container(border=True):
            st.warning("No Active Session Data. Please analyze an audio file in the 'Home Analyze' tab.")
            if lottie_wave:
                st_lottie(lottie_wave, height=180)
    else:
        st.markdown(f"<p class='hero-subtitle'>Deconstructing the Prediction: <b>{st.session_state.pred_label.upper()}</b></p>", unsafe_allow_html=True)
        
        with st.spinner("Decoding Neural Logic (SHAP)..."):
            X = st.session_state.X
            pred_idx = st.session_state.pred_idx
            
            background = X + np.random.normal(0, 0.01, X.shape)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                s_values = np.array(shap_values[pred_idx]).reshape(-1)
            else:
                s_values = shap_values[0, :, pred_idx].reshape(-1) if len(shap_values.shape) == 3 else shap_values[0].reshape(-1)

            feature_names = ([f"MFCC {i}" for i in range(40)] + [f"Chroma {i}" for i in range(12)] + [f"Contrast {i}" for i in range(7)] + ["ZCR"])

            col1, col2 = st.columns([1, 1])
            with col1:
                with st.container(border=True):
                    st.subheader("Top Influential Features")
                    df_shap = pd.DataFrame({"Feature": feature_names, "Impact": s_values, "Weight": np.abs(s_values)}).sort_values("Weight", ascending=False)
                    st.dataframe(df_shap.head(10), use_container_width=True)

            with col2:
                with st.container(border=True):
                    st.subheader("Feature Weight Bar")
                    # Clear previous plots
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    plt.style.use('dark_background')
                    bars = ax.barh(df_shap.head(10)["Feature"], df_shap.head(10)["Weight"], color="#00d2ff")
                    ax.invert_yaxis()
                    ax.set_xlabel("Absolute Weight", color="#8892b0")
                    ax.tick_params(colors="#8892b0")
                    
                    # Remove spines
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    
                    st.pyplot(fig, transparent=True)
                    plt.close(fig)

            with st.container(border=True):
                st.subheader("Neural Path Waterfall Chart")
                b_val = float(explainer.expected_value[pred_idx] if isinstance(explainer.expected_value, list) else (explainer.expected_value[pred_idx] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value))
                shap_exp = shap.Explanation(values=s_values, base_values=b_val, data=X[0], feature_names=feature_names)
                shap.plots.waterfall(shap_exp, max_display=12, show=False)
                plt.style.use('dark_background')
                st.pyplot(plt.gcf())
                plt.clf()