import sys, os
sys.path.insert(0, os.path.abspath('..'))

import streamlit as st
import numpy as np
import pickle, json, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)

# ── Load models (cached so they load once) ─────────────
@st.cache_resource
def load_models():
    BASE = os.path.join(os.path.dirname(__file__), '..')

    # LSTM
    class RULPredictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, 64, 2,
                                batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(32, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(1)

    with open(os.path.join(BASE, 'models/feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)

    lstm = RULPredictor(len(feature_cols))
    lstm.load_state_dict(torch.load(
        os.path.join(BASE, 'models/lstm_best.pt'),
        map_location='cpu'))
    lstm.eval()

    # NLP
    with open(os.path.join(BASE, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(BASE, 'models/nlp_classifier.pkl'), 'rb') as f:
        nlp = pickle.load(f)
    with open(os.path.join(BASE, 'models/nlp_metadata.pkl'), 'rb') as f:
        nlp_meta = pickle.load(f)

    # CNN
    cnn = models.resnet18(weights=None)
    cnn.fc = nn.Sequential(
        nn.Linear(cnn.fc.in_features, 64),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))
    cnn.load_state_dict(torch.load(
        os.path.join(BASE, 'models/cnn_best.pt'),
        map_location='cpu'))
    cnn.eval()

    with open(os.path.join(BASE, 'models/scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    X_train = np.load(os.path.join(BASE, 'data/X_train.npy'))

    return lstm, tfidf, nlp, nlp_meta, cnn, scaler, feature_cols, X_train

lstm_model, tfidf, nlp_model, nlp_meta, cnn_model, \
    scaler, feature_cols, X_train = load_models()

id_to_fault  = nlp_meta['id_to_fault']
CNN_CLASSES  = ['healthy', 'worn', 'damaged']
normalize_img = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# ── Inference helpers ──────────────────────────────────
def predict_rul(seq):
    t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return float(np.clip(lstm_model(t).item(), 0, 125))

def predict_fault(text):
    vec   = tfidf.transform([text])
    pred  = nlp_model.predict(vec)[0]
    proba = nlp_model.predict_proba(vec)[0]
    return id_to_fault[pred], float(proba[pred])

def make_img(kind):
    img = np.random.normal(
        {'healthy':0.6,'worn':0.45,'damaged':0.35}[kind],
        0.08, (64, 64))
    if kind == 'worn':
        for _ in range(4):
            r = np.random.randint(10, 54)
            img[r:r+2, 10:50] = 0.2
    elif kind == 'damaged':
        for _ in range(3):
            x, y = np.random.randint(10, 50, 2)
            img[x:x+8, y:y+8] = 0.05
    return np.clip(img, 0, 1)

def predict_cnn(img):
    t = torch.tensor(img, dtype=torch.float32)
    t = normalize_img(t.unsqueeze(0).repeat(3,1,1)).unsqueeze(0)
    with torch.no_grad():
        p = torch.nn.functional.softmax(cnn_model(t), dim=1)[0].numpy()
    return CNN_CLASSES[p.argmax()], float(p.max())

def draw_gauge(score):
    color = ('#e63946' if score >= 75 else
             '#f4a261' if score >= 50 else
             '#2a9d8f' if score >= 25 else '#57cc99')
    fig, ax = plt.subplots(figsize=(3.5, 2),
                           subplot_kw={'projection': 'polar'})
    ax.barh(0, score / 100 * np.pi, left=0,
           height=0.5, color=color, alpha=0.85)
    ax.barh(0, np.pi, left=0, height=0.5,
           color='#333', alpha=0.2)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(1)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')
    ax.text(np.pi / 2, -0.3, f'{score:.0f}',
           ha='center', va='center',
           fontsize=28, fontweight='bold', color=color)
    plt.tight_layout()
    return fig

def fuse(rul, fault, nlp_c, cond, cnn_c):
    fault_sev = {'turbine_blade_damage':95, 'bearing_wear':75,
                 'compressor_degradation':65, 'fuel_system':55,
                 'sensor_drift':30}
    cnn_sev   = {'healthy':5, 'worn':55, 'damaged':90}
    lstm_risk = (1 - rul/125) * 100
    nlp_risk  = fault_sev.get(fault, 50) * nlp_c
    cnn_risk  = cnn_sev[cond] * cnn_c
    score     = float(np.clip(0.5*lstm_risk + 0.25*nlp_risk + 0.25*cnn_risk, 0, 100))
    return score, lstm_risk, nlp_risk, cnn_risk

# ── UI ─────────────────────────────────────────────────
st.title("🔧 Multimodal Predictive Maintenance AI")
st.markdown("Combines **LSTM sensor analysis**, **NLP log classification**, and **CNN image inspection** into one unified engine health score.")
st.markdown("---")

# ── Sidebar inputs ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Engine Inputs")

    engine_id = st.text_input("Engine ID", value="ENG-001")

    seq_idx = st.slider(
        "Sensor sequence index",
        0, len(X_train)-1, 15000,
        help="Higher index = more degraded engine"
    )
    
    st.caption("Quick log examples:")
    col_a, col_b = st.columns(2)
    if col_a.button("🔴 Critical"):
        st.session_state['log_prefill'] = \
        "Crack detected on turbine blade leading edge"
    if col_b.button("✅ Healthy"):
        st.session_state['log_prefill'] = \
        "Routine inspection completed, all systems nominal"
    
    default_log = st.session_state.get(
    'log_prefill',
    "Unusual vibration in bearing housing, noise increasing"
    )

    log_text = st.text_area(
        "Maintenance log entry",
        value="Unusual vibration in bearing housing, noise increasing",
        height=100
    )

    img_condition = st.selectbox(
        "Part image condition",
        ["healthy", "worn", "damaged"]
    )

    run = st.button("🚀 Run Analysis", use_container_width=True)

# ── Main panel ─────────────────────────────────────────
if run:
    np.random.seed(42)
    sensor_seq = X_train[seq_idx]
    img_array  = make_img(img_condition)

    with st.spinner("Running all 3 models..."):
        rul              = predict_rul(sensor_seq)
        fault, nlp_conf  = predict_fault(log_text)
        cond,  cnn_conf  = predict_cnn(img_array)
        score, lr, nr, cr = fuse(rul, fault, nlp_conf, cond, cnn_conf)

    # Status colour
    if   score >= 75: status, color = "🔴 CRITICAL", "#e63946"
    elif score >= 50: status, color = "🟡 WARNING",  "#f4a261"
    elif score >= 25: status, color = "🟢 MONITOR",  "#2a9d8f"
    else:             status, color = "✅ HEALTHY",  "#57cc99"

    actions = {
        "🔴 CRITICAL": "Ground engine immediately",
        "🟡 WARNING" : "Schedule maintenance this week",
        "🟢 MONITOR" : "Inspect at next service interval",
        "✅ HEALTHY" : "No action required",
    }

    # ── Top metrics row ───────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fused Risk Score", f"{score:.1f}/100")
    c2.metric("Status", status)
    c3.metric("RUL (cycles)", f"{rul:.1f}")
    c4.metric("Action", actions[status])

    st.markdown("---")

    # ── Three model outputs ───────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 LSTM — Sensor Analysis")
        st.metric("Remaining Useful Life", f"{rul:.1f} cycles")
        st.metric("LSTM Risk Contribution", f"{lr:.1f}/100")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(sensor_seq[:, 0], color='#7c6af7', linewidth=1.5)
        ax.set_title('Sensor trace (feature 1)', fontsize=9)
        ax.set_xlabel('Timestep', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📝 NLP — Log Analysis")
        st.metric("Detected Fault", fault.replace('_', ' ').title())
        st.metric("Confidence", f"{nlp_conf*100:.1f}%")
        st.metric("NLP Risk Contribution", f"{nr:.1f}/100")
        st.info(f'"{log_text}"')

    with col3:
        st.subheader("🖼️ CNN — Image Analysis")
        st.metric("Part Condition", cond.title())
        st.metric("Confidence", f"{cnn_conf*100:.1f}%")
        st.metric("CNN Risk Contribution", f"{cr:.1f}/100")
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.imshow(img_array, cmap='gray', vmin=0, vmax=1)
        ax2.axis('off')
        ax2.set_title(f'Part: {cond}', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # ── Contribution breakdown chart ──────────────────
    st.subheader("📊 Risk Score Breakdown")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    components = [lr*0.5, nr*0.25, cr*0.25]
    bars = ax3.barh(['LSTM (50%)','NLP (25%)','CNN (25%)'],
                    components,
                    color=['#7c6af7','#2a9d8f','#e07b39'],
                    edgecolor='none')
    ax3.axvline(37.5, color='red', linestyle='--',
               alpha=0.4, label='Equal contribution line')
    ax3.set_xlim(0, 55)
    ax3.set_xlabel('Weighted contribution to final score')
    ax3.set_title(f'Engine {engine_id} — Final Risk: {score:.1f}/100 {status}',
                 fontweight='bold')
    for bar, val in zip(bars, components):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    
    st.markdown("---")
    st.subheader("📋 Analysis History")
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'Engine'  : engine_id,
        'Risk'    : score,
        'RUL'     : round(rul, 1),
        'Fault'   : fault.replace('_', ' ').title(),
        'Condition': cond.title(),
        'Status'  : status,
    })
    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

else:
    st.info("👈 Configure engine inputs in the sidebar and click **Run Analysis**")
    st.markdown("""
    ### How it works
    | Model | Input | Output |
    |-------|-------|--------|
    | 🧠 LSTM | 30-cycle sensor window | Remaining Useful Life |
    | 📝 NLP  | Maintenance log text   | Fault type + severity |
    | 🖼️ CNN  | Part image condition   | Healthy / Worn / Damaged |
    | ⚡ Fusion | All 3 outputs       | Risk score 0–100 |
    """)
