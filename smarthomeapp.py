import streamlit as st
import numpy as np
import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Management System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & background ─────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.9) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}

/* ── Header card ──────────────────────────── */
.header-card {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(79, 70, 229, 0.4);
}
.header-card h1 { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0; letter-spacing: -0.5px; }
.header-card p  { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; font-size: 1.05rem; }

/* ── Section titles ───────────────────────── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.5rem 0 0.75rem;
    border-left: 3px solid #7c3aed;
    padding-left: 0.6rem;
}

/* ── Input labels ─────────────────────────── */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
label { color: #cbd5e1 !important; font-size: 0.875rem !important; }

/* ── Predict button ───────────────────────── */
div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: #fff;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2.5rem;
    width: 100%;
    letter-spacing: 0.5px;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4);
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(79, 70, 229, 0.55);
}

/* ── Result cards ─────────────────────────── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    animation: fadeIn 0.5s ease;
}
.result-on  { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981; }
.result-off { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #ef4444; }
.result-icon { font-size: 4rem; }
.result-label { font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0; }
.result-sub   { font-size: 0.95rem; color: rgba(255,255,255,0.7); }

/* ── Confidence bar wrapper ───────────────── */
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 12px;
    margin: 0.4rem 0 1rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 12px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Metric chip ──────────────────────────── */
.metric-chip {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.82rem;
    color: #a78bfa;
    margin: 0.2rem;
}

/* ── Recommendation card ──────────────────── */
.rec-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-top: 1.25rem;
}
.rec-card h4 { color: #a78bfa; margin: 0 0 0.6rem; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px; }
.rec-item { display: flex; align-items: flex-start; gap: 0.6rem; margin: 0.45rem 0; color: #cbd5e1; font-size: 0.88rem; }
.rec-bullet { color: #34d399; font-weight: bold; flex-shrink: 0; }
.rec-bullet-warn { color: #fbbf24; font-weight: bold; flex-shrink: 0; }

/* ── Sidebar info box ─────────────────────── */
.info-box {
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.82rem;
    color: #c4b5fd;
    margin-top: 0.75rem;
}

@keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_artifacts():
    model = load_model("model.h5", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def check_files():
    missing = [f for f in ["model.h5", "scaler.pkl"] if not os.path.exists(f)]
    return missing


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>⚡ Smart Energy Management System</h1>
    <p>Artificial Neural Network · Real-time Household Power Status Prediction</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Model Info")
    st.markdown("""
    <div class="info-box">
        <b>Architecture:</b><br>
        Input → Dense(16, ReLU) → BN → Dense(8, ReLU) → Dense(1, Sigmoid)<br><br>
        <b>Optimizer:</b> Adam<br>
        <b>Loss:</b> Binary Crossentropy<br>
        <b>Dataset:</b> UCI Household Power Consumption<br><br>
        <b>Target:</b> Power_Status<br>
        &nbsp;&nbsp;🟢 ON  → above median power<br>
        &nbsp;&nbsp;🔴 OFF → below median power
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Features Used")
    features_info = {
        "🌡 Temperature": "Ambient temp (°C)",
        "💧 Humidity": "Relative humidity (%)",
        "🕐 Hour": "Hour of day (0–23)",
        "⚡ Voltage": "Supply voltage (V)",
        "🔌 Global Intensity": "Current intensity (A)",
        "📟 Sub Metering Total": "Combined sub-metering (Wh)",
        "☀ Light Availability": "Daylight present"
    }
    for feat, desc in features_info.items():
        st.markdown(f"**{feat}** — {desc}")

# ─────────────────────────────────────────────
# FILE CHECK
# ─────────────────────────────────────────────
missing = check_files()
if missing:
    st.error(f"⚠️ Missing files: {', '.join(missing)}. Please run `python train.py` first.")
    st.stop()

model, scaler = load_artifacts()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🎛 Enter Environmental & Electrical Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌡 Environmental**")
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=22.0, step=0.5)
    humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
    hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14, step=1)

with col2:
    st.markdown("**⚡ Electrical**")
    voltage = st.number_input("Voltage (V)", min_value=200.0, max_value=260.0, value=235.0, step=0.5)
    global_intensity = st.number_input("Global Intensity (A)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    sub_total = st.number_input("Sub Metering Total (Wh)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)

with col3:
    st.markdown("**☀ Lighting**")
    light_option = st.selectbox("Light Availability", options=["Yes (Daytime)", "No (Night)"])
    light_val = 1 if light_option.startswith("Yes") else 0

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Adjust all parameters to match the current household conditions, then click <b>Predict</b>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Power Status", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    input_arr = np.array([[
        temperature, humidity, hour,
        voltage, global_intensity,
        sub_total, light_val
    ]], dtype=np.float32)

    input_scaled = scaler.transform(input_arr)
    prob = float(model.predict(input_scaled, verbose=0)[0][0])
    label = int(prob >= 0.5)
    confidence = prob if label == 1 else (1 - prob)
    conf_pct = confidence * 100

    st.markdown("---")
    st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

    res_col, rec_col = st.columns([1, 1.2])

    with res_col:
        if label == 1:
            st.markdown(f"""
            <div class="result-card result-on">
                <div class="result-icon">🟢</div>
                <div class="result-label" style="color:#34d399;">POWER ON — HIGH USAGE</div>
                <div class="result-sub">Global active power is <b>above median threshold</b></div>
                <br>
                <div style="text-align:left; padding: 0 0.5rem;">
                    <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:4px;">Confidence Score</div>
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <div class="conf-bar-bg" style="flex:1;">
                            <div class="conf-bar-fill" style="width:{conf_pct:.1f}%; background:linear-gradient(90deg,#34d399,#059669);"></div>
                        </div>
                        <span style="color:#34d399; font-weight:700; font-size:1rem; min-width:50px;">{conf_pct:.1f}%</span>
                    </div>
                </div>
                <br>
                <span class="metric-chip">Raw Probability: {prob:.4f}</span>
                <span class="metric-chip">Threshold: 0.5000</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-off">
                <div class="result-icon">🔴</div>
                <div class="result-label" style="color:#f87171;">POWER OFF — LOW USAGE</div>
                <div class="result-sub">Global active power is <b>below median threshold</b></div>
                <br>
                <div style="text-align:left; padding: 0 0.5rem;">
                    <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:4px;">Confidence Score</div>
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <div class="conf-bar-bg" style="flex:1;">
                            <div class="conf-bar-fill" style="width:{conf_pct:.1f}%; background:linear-gradient(90deg,#f87171,#dc2626);"></div>
                        </div>
                        <span style="color:#f87171; font-weight:700; font-size:1rem; min-width:50px;">{conf_pct:.1f}%</span>
                    </div>
                </div>
                <br>
                <span class="metric-chip">Raw Probability: {prob:.4f}</span>
                <span class="metric-chip">Threshold: 0.5000</span>
            </div>
            """, unsafe_allow_html=True)

    with rec_col:
        if label == 1:  # HIGH usage → saving tips
            recs = [
                ("●", "Switch to energy-efficient LED lighting to reduce load."),
                ("●",
                 "Enable smart scheduling for heavy appliances (washing machine, dishwasher) during off-peak hours."),
                ("●", "Set thermostat 2–3°C lower to reduce HVAC consumption."),
                ("●", "Unplug idle chargers and standby electronics — they account for up to 10% of usage."),
                ("●", "Consider solar panels or battery storage to offset peak demand."),
                ("●", "Use power strips with surge protectors to prevent phantom loads."),
            ]
            bullet_class = "rec-bullet-warn"
            title = "⚠️ High Consumption Detected — Energy-Saving Tips"
        else:
            recs = [
                ("●", "Current consumption is within efficient range — great job!"),
                ("●", "Maintain off-peak scheduling for major appliances."),
                ("●", "Continue using natural daylight instead of artificial lighting."),
                ("●", "Verify that heaters/ACs are set to auto-off mode."),
                ("●", "Monitor sub-metering trends weekly for anomalies."),
                ("●", "Consider time-of-use tariffs to save on energy bills further."),
            ]
            bullet_class = "rec-bullet"
            title = "✅ Low Consumption — Keep It Up!"

        items_html = "".join([
            f'<div class="rec-item"><span class="{bullet_class}">{b}</span><span>{t}</span></div>'
            for b, t in recs
        ])
        st.markdown(f"""
        <div class="rec-card">
            <h4>{title}</h4>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Summary metrics row
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temperature", f"{temperature} °C")
    m2.metric("Humidity", f"{humidity} %")
    m3.metric("Voltage", f"{voltage} V")
    m4.metric("Global Intensity", f"{global_intensity} A")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Sub Metering", f"{sub_total} Wh")
    m6.metric("Hour", f"{hour}:00")
    m7.metric("Light Available", "Yes" if light_val else "No")
    m8.metric("Confidence", f"{conf_pct:.1f}%")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.8rem; padding: 1rem 0;">
    Smart Energy Management System · ANN Model · UCI Household Power Consumption Dataset<br>
    Built with TensorFlow / Keras &nbsp;|&nbsp; Streamlit &nbsp;|&nbsp; scikit-learn
</div>
""", unsafe_allow_html=True)