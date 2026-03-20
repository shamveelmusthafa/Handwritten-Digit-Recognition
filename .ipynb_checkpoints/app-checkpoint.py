import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import plotly.graph_objects as go
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME CSS
# ============================================================
def get_theme_css(dark):
    if dark:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
        .stApp { background:#0a0a0f; color:#e8e8f0; }
        section[data-testid="stSidebar"] { background:#12121a !important; border-right:1px solid #2a2a4a; }
        .main-header { background:linear-gradient(135deg,#0a0a0f,#1a1a2e,#0a0a0f); border:1px solid #00d4ff; border-radius:16px; padding:24px 32px; margin-bottom:24px; }
        .header-title { font-size:2.5rem; font-weight:800; background:linear-gradient(135deg,#00d4ff,#7c3aed,#00ff88); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; font-family:Syne,sans-serif; }
        .header-sub { color:#8888aa; font-size:0.9rem; margin-top:4px; font-family:Space Mono,monospace; }
        .metric-card { background:#1a1a2e; border:1px solid #2a2a4a; border-radius:12px; padding:20px; text-align:center; border-top:2px solid #00d4ff; }
        .metric-val { font-size:2rem; font-weight:800; color:#00d4ff; font-family:Space Mono,monospace; }
        .metric-lbl { font-size:0.75rem; color:#8888aa; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
        .pred-box { background:linear-gradient(135deg,#1a1a2e,#1a1a3e); border:2px solid #00d4ff; border-radius:20px; padding:28px; text-align:center; }
        .pred-digit { font-size:5rem; font-weight:800; font-family:Space Mono,monospace; background:linear-gradient(135deg,#00d4ff,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .pred-conf { font-size:1rem; color:#00ff88; font-family:Space Mono,monospace; margin-top:6px; }
        .sec-hdr { font-size:0.85rem; font-weight:700; color:#00d4ff; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px; padding-bottom:6px; border-bottom:1px solid #2a2a4a; font-family:Space Mono,monospace; }
        .hist-item { background:#1a1a2e; border:1px solid #2a2a4a; border-radius:8px; padding:10px 14px; margin-bottom:6px; font-family:Space Mono,monospace; font-size:0.8rem; color:#e8e8f0; }
        .badge-ok { background:rgba(0,255,136,0.15); color:#00ff88; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
        .badge-no { background:rgba(255,71,87,0.15); color:#ff4757; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
        .badge-mdl { background:rgba(124,58,237,0.15); color:#a78bfa; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
        .track-box { background:#1a1a2e; border:1px solid #2a2a4a; border-radius:10px; padding:14px; text-align:center; }
        .track-num { font-size:1.8rem; font-weight:800; font-family:Space Mono,monospace; }
        .track-lbl { font-size:0.65rem; color:#8888aa; text-transform:uppercase; letter-spacing:1px; }
        .stButton>button { background:linear-gradient(135deg,#00d4ff,#7c3aed) !important; color:white !important; border:none !important; border-radius:8px !important; }
        .empty-box { background:#1a1a2e; border:2px dashed #2a2a4a; border-radius:16px; padding:50px; text-align:center; }
        </style>"""
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
        .stApp { background:#f0f4ff; color:#1e1e2e; }
        section[data-testid="stSidebar"] { background:#ffffff !important; border-right:1px solid #e5e7eb; }
        .main-header { background:linear-gradient(135deg,#eff6ff,#f5f3ff); border:1px solid #bfdbfe; border-radius:16px; padding:24px 32px; margin-bottom:24px; }
        .header-title { font-size:2.5rem; font-weight:800; background:linear-gradient(135deg,#2563eb,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; font-family:Syne,sans-serif; }
        .header-sub { color:#6b7280; font-size:0.9rem; margin-top:4px; font-family:Space Mono,monospace; }
        .metric-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:20px; text-align:center; border-top:2px solid #2563eb; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
        .metric-val { font-size:2rem; font-weight:800; color:#2563eb; font-family:Space Mono,monospace; }
        .metric-lbl { font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
        .pred-box { background:linear-gradient(135deg,#eff6ff,#f5f3ff); border:2px solid #2563eb; border-radius:20px; padding:28px; text-align:center; }
        .pred-digit { font-size:5rem; font-weight:800; font-family:Space Mono,monospace; background:linear-gradient(135deg,#2563eb,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .pred-conf { font-size:1rem; color:#059669; font-family:Space Mono,monospace; margin-top:6px; }
        .sec-hdr { font-size:0.85rem; font-weight:700; color:#2563eb; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px; padding-bottom:6px; border-bottom:2px solid #e5e7eb; font-family:Space Mono,monospace; }
        .hist-item { background:#f8faff; border:1px solid #e5e7eb; border-radius:8px; padding:10px 14px; margin-bottom:6px; font-family:Space Mono,monospace; font-size:0.8rem; color:#1e1e2e; }
        .badge-ok { background:rgba(5,150,105,0.1); color:#059669; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
        .badge-no { background:rgba(220,38,38,0.1); color:#dc2626; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
        .badge-mdl { background:rgba(124,58,237,0.1); color:#7c3aed; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
        .track-box { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:14px; text-align:center; }
        .track-num { font-size:1.8rem; font-weight:800; font-family:Space Mono,monospace; }
        .track-lbl { font-size:0.65rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; }
        .stButton>button { background:linear-gradient(135deg,#2563eb,#7c3aed) !important; color:white !important; border:none !important; border-radius:8px !important; }
        .empty-box { background:#f8faff; border:2px dashed #e5e7eb; border-radius:16px; padding:50px; text-align:center; }
        </style>"""

# ============================================================
# SESSION STATE
# ============================================================
for k, v in {
    'dark_mode': True, 'prediction_history': [],
    'total_predictions': 0, 'correct_predictions': 0,
    'canvas_key': 0
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    return {
        'SVM (Tuned) — 98.33%': joblib.load('saved_models/svm_tuned.pkl'),
        'MLP (Tuned) — 98.05%': joblib.load('saved_models/mlp_tuned.pkl')
    }

# ============================================================
# HELPERS
# ============================================================
def preprocess_image(image):
    if image.mode != 'L':
        image = ImageOps.grayscale(image)
    image = image.resize((28, 28), Image.LANCZOS)
    arr   = np.array(image).astype('float32')
    if arr.mean() > 127:
        arr = 255 - arr
    return arr / 255.0

def predict_digit(model, img_array):
    flat = img_array.flatten().reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(flat)[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred] * 100)
    else:
        pred     = int(model.predict(flat)[0])
        scores   = model.decision_function(flat)[0]
        exp_s    = np.exp(scores - scores.max())
        proba    = exp_s / exp_s.sum()
        conf     = float(proba[pred] * 100)
    return pred, conf, proba

def gauge_chart(conf, dark):
    color = '#00ff88' if conf >= 90 else '#ffd700' if conf >= 70 else '#ff4757'
    bg    = '#1a1a2e' if dark else '#f0f4ff'
    txt   = '#e8e8f0' if dark else '#1e1e2e'
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=conf,
        title={'text': "Confidence %", 'font': {'size': 13, 'color': txt}},
        number={'suffix': '%', 'font': {'size': 26, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': txt},
            'bar': {'color': color},
            'bgcolor': bg,
            'bordercolor': color,
            'steps': [
                {'range': [0, 70],   'color': 'rgba(255,71,87,0.12)'},
                {'range': [70, 90],  'color': 'rgba(255,215,0,0.12)'},
                {'range': [90, 100], 'color': 'rgba(0,255,136,0.12)'}
            ],
            'threshold': {'line': {'color': color, 'width': 4},
                          'thickness': 0.75, 'value': conf}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor=bg, font_color=txt)
    return fig

def bar_chart(proba, dark):
    best   = int(np.argmax(proba))
    colors = ['#00d4ff' if i == best else '#7c3aed' for i in range(10)]
    bg     = '#1a1a2e' if dark else '#ffffff'
    tc     = '#e8e8f0' if dark else '#1e1e2e'
    gc     = '#2a2a4a' if dark else '#e5e7eb'
    fig    = go.Figure(go.Bar(
        x=list(range(10)), y=proba * 100,
        marker_color=colors,
        text=[f'{p*100:.1f}%' for p in proba],
        textposition='outside',
        textfont={'size': 9}
    ))
    fig.update_layout(
        title={'text': 'Confidence per Digit', 'font': {'size': 13, 'color': tc}},
        xaxis={'title': 'Digit', 'tickvals': list(range(10)), 'gridcolor': gc, 'color': tc},
        yaxis={'title': 'Conf (%)', 'range': [0, 120], 'gridcolor': gc, 'color': tc},
        paper_bgcolor=bg, plot_bgcolor=bg,
        height=260, margin=dict(l=40,r=10,t=45,b=35), font={'color': tc}
    )
    return fig

# ============================================================
# APPLY THEME
# ============================================================
st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# color shortcuts (recomputed after theme applied)
dark = st.session_state.dark_mode
ac   = '#00d4ff' if dark else '#2563eb'
tc   = '#e8e8f0' if dark else '#1e1e2e'
sc   = '#8888aa' if dark else '#6b7280'
bg   = '#1a1a2e' if dark else '#f0f4ff'
bc   = '#2a2a4a' if dark else '#e5e7eb'
gc_t = '#00ff88' if dark else '#059669'

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:12px 0 16px;'>
        <div style='font-size:2.5rem;'>🔢</div>
        <div style='font-weight:800;font-size:1rem;letter-spacing:1px;color:{tc};'>
            MNIST RECOGNIZER
        </div>
        <div style='font-size:0.7rem;color:{sc};font-family:monospace;'>
            v1.0 · SVM & MLP · 98%+ Accuracy
        </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Theme
    st.markdown("<div class='sec-hdr'>⚙️ Settings</div>", unsafe_allow_html=True)
    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.divider()

    # Model
    st.markdown("<div class='sec-hdr'>🤖 Select Model</div>", unsafe_allow_html=True)
    models    = load_models()
    sel_name  = st.selectbox("Model", list(models.keys()), index=1, label_visibility="collapsed")
    sel_model = models[sel_name]

    info_map = {
        'SVM (Tuned) — 98.33%': {'Type':'SVM','Kernel':'RBF','C':'50','Accuracy':'98.33%'},
        'MLP (Tuned) — 98.05%': {'Type':'MLP','Layers':'512→256→128','Act':'Tanh','Accuracy':'98.05%'}
    }
    rows = "".join([
        f"<div style='display:flex;justify-content:space-between;margin-bottom:5px;'>"
        f"<span style='color:{sc};font-size:0.75rem;'>{k}</span>"
        f"<span style='color:{tc};font-size:0.75rem;font-weight:700;'>{v}</span></div>"
        for k, v in info_map[sel_name].items()
    ])
    st.markdown(f"""
    <div style='background:{bg};border:1px solid {bc};border-radius:10px;
                padding:14px;margin-top:6px;font-family:monospace;'>
        {rows}
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Tracker
    st.markdown("<div class='sec-hdr'>📈 Accuracy Tracker</div>", unsafe_allow_html=True)
    total   = st.session_state.total_predictions
    correct = st.session_state.correct_predictions
    acc     = (correct / total * 100) if total > 0 else 0

    c1, c2, c3 = st.columns(3)
    for col, val, lbl, color in zip(
        [c1, c2, c3],
        [total, correct, f'{acc:.0f}%'],
        ['Total', 'Correct', 'Acc'],
        [ac, gc_t, ac]
    ):
        with col:
            st.markdown(f"""
            <div class='track-box'>
                <div class='track-num' style='color:{color};'>{val}</div>
                <div class='track-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    if total > 0:
        st.progress(acc / 100)

    if st.button("🔄 Reset Tracker", use_container_width=True):
        st.session_state.total_predictions   = 0
        st.session_state.correct_predictions = 0
        st.session_state.prediction_history  = []
        st.rerun()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class='main-header'>
    <div class='header-title'>🔢 MNIST Digit Recognizer</div>
    <div class='header-sub'>
        Draw or upload a handwritten digit · AI predicts instantly · 98%+ Accuracy
    </div>
</div>""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "✏️ Draw & Predict", "📁 Batch Prediction",
    "📋 Prediction History"
])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<div class='sec-hdr'>✏️ Input</div>", unsafe_allow_html=True)
        mode = st.radio("Input Mode", ["🎨 Draw Digit", "📁 Upload Image"], horizontal=True)
        img_array   = None
        predict_btn = False

        if mode == "🎨 Draw Digit":
            brush    = st.slider("Brush Size", 10, 40, 22)
            realtime = st.toggle("⚡ Real-time Prediction", value=True)
            sk_col   = "#FFFFFF" if dark else "#000000"
            bg_col   = "#000000" if dark else "#FFFFFF"

            canvas = st_canvas(
                fill_color=sk_col, stroke_width=brush,
                stroke_color=sk_col, background_color=bg_col,
                height=280, width=280, drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
                update_streamlit=realtime
            )

            col_p, col_c = st.columns(2)
            with col_p:
                predict_btn = st.button("🔍 Predict", use_container_width=True)
            with col_c:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.canvas_key += 1
                    st.rerun()

            if canvas.image_data is not None:
                pil       = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA')
                img_array = preprocess_image(pil)

        else:
            uploaded = st.file_uploader("Upload digit image", type=['png','jpg','jpeg'])
            if uploaded:
                pil       = Image.open(uploaded)
                st.image(pil, caption="Uploaded Image", width=280)
                img_array   = preprocess_image(pil)
                predict_btn = st.button("🔍 Predict", use_container_width=True)

        true_lbl = "Unknown"

    with right:
        st.markdown("<div class='sec-hdr'>🎯 Prediction Result</div>", unsafe_allow_html=True)

        has_ink    = img_array is not None and img_array.sum() > 0.5
        do_predict = has_ink and (predict_btn or (mode == "🎨 Draw Digit" and realtime))

        if do_predict:
            pred, conf, proba = predict_digit(sel_model, img_array)

            if true_lbl != "Unknown" and predict_btn:
                st.session_state.total_predictions   += 1
                if pred == int(true_lbl):
                    st.session_state.correct_predictions += 1
                st.session_state.prediction_history.append({
                    'prediction': pred, 'confidence': round(conf, 1),
                    'true_label': true_lbl,
                    'model'     : sel_name.split('—')[0].strip(),
                    'correct'   : pred == int(true_lbl)
                })

            st.markdown(f"""
            <div class='pred-box'>
                <div style='font-size:0.75rem;color:{sc};font-family:monospace;
                            text-transform:uppercase;letter-spacing:2px;'>
                    Predicted Digit
                </div>
                <div class='pred-digit'>{pred}</div>
                <div class='pred-conf'>Confidence: {conf:.1f}%</div>
                <div style='margin-top:5px;font-size:0.72rem;color:{sc};font-family:monospace;'>
                    {sel_name.split('—')[0].strip()}
                </div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(conf, dark), use_container_width=True)
            st.plotly_chart(bar_chart(proba, dark),  use_container_width=True)

            top3   = np.argsort(proba)[::-1][:3]
            medals = ['🥇', '🥈', '🥉']
            st.markdown("<div class='sec-hdr'>🏆 Top 3 Predictions</div>",
                        unsafe_allow_html=True)
            for medal, idx in zip(medals, top3):
                pct = proba[idx] * 100
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {bc};border-radius:8px;
                            padding:10px 14px;margin-bottom:6px;font-family:monospace;
                            display:flex;align-items:center;gap:10px;'>
                    <span style='font-size:1.3rem;'>{medal}</span>
                    <span style='font-size:1.3rem;font-weight:800;color:{ac};'>Digit {idx}</span>
                    <div style='flex:1;background:{bc};border-radius:4px;height:7px;overflow:hidden;'>
                        <div style='width:{pct}%;height:100%;background:{ac};border-radius:4px;'></div>
                    </div>
                    <span style='color:{tc};font-size:0.85rem;'>{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='empty-box'>
                <div style='font-size:3.5rem;'>✏️</div>
                <div style='color:{sc};margin-top:14px;font-family:monospace;font-size:0.9rem;'>
                    Draw or upload a digit<br>to see the prediction
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.markdown("<div class='sec-hdr'>📁 Batch Image Prediction</div>", unsafe_allow_html=True)
    st.info("Upload multiple handwritten digit images for bulk prediction")

    files = st.file_uploader("Upload Images", type=['png','jpg','jpeg'],
                              accept_multiple_files=True)
    if files:
        st.write(f"**{len(files)} image(s) uploaded**")
        if st.button("🔍 Predict All"):
            results  = []
            progress = st.progress(0)
            cols     = st.columns(min(5, len(files)))
            for i, f in enumerate(files):
                pil   = Image.open(f)
                arr   = preprocess_image(pil)
                pred, conf, proba = predict_digit(sel_model, arr)
                results.append({
                    'File': f.name, 'Prediction': pred,
                    'Confidence': f'{conf:.1f}%',
                    'Runner-up' : f'Digit {np.argsort(proba)[::-1][1]}',
                    'Model'     : sel_name.split('—')[0].strip()
                })
                with cols[i % 5]:
                    st.image(pil, width=90, caption=f"Pred:{pred} ({conf:.0f}%)")
                progress.progress((i+1)/len(files))

            st.success(f"✅ Done! Predicted {len(results)} images.")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                               "predictions.csv", "text/csv")

            conf_vals = [float(r['Confidence'].replace('%','')) for r in results]
            for col, val, lbl in zip(
                st.columns(3),
                [len(results), f'{np.mean(conf_vals):.1f}%', f'{np.max(conf_vals):.1f}%'],
                ['Images Processed', 'Avg Confidence', 'Max Confidence']
            ):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-val'>{val}</div>
                        <div class='metric-lbl'>{lbl}</div>
                    </div>""", unsafe_allow_html=True)

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.markdown("<div class='sec-hdr'>📋 Prediction History</div>", unsafe_allow_html=True)

    history = st.session_state.prediction_history

    if not history:
        st.markdown(f"""
        <div class='empty-box'>
            <div style='font-size:3rem;'>📋</div>
            <div style='color:{sc};margin-top:12px;font-family:monospace;'>
                No predictions yet · Draw or upload a digit to start!
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        total_h   = len(history)
        correct_h = sum(1 for h in history if h.get('correct') is True)
        acc_h     = (correct_h / total_h * 100) if total_h > 0 else 0

        for col, val, lbl, color in zip(
            st.columns(3),
            [total_h, correct_h, f'{acc_h:.0f}%'],
            ['Total','Correct','Accuracy'],
            [ac, gc_t, ac]
        ):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val' style='color:{color};'>{val}</div>
                    <div class='metric-lbl'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        for i, h in enumerate(reversed(history)):
            badge = ""
            if h.get('correct') is True:
                badge = "<span class='badge-ok'>✅ Correct</span>"
            elif h.get('correct') is False:
                badge = "<span class='badge-no'>❌ Wrong</span>"
            true_s = f"True:{h['true_label']} &nbsp;" if h['true_label'] != 'Unknown' else ""
            st.markdown(f"""
            <div class='hist-item'>
                <b style='color:{ac};'>#{total_h-i}</b> &nbsp;
                Predicted: <b>{h['prediction']}</b> &nbsp;
                Conf: {h['confidence']}% &nbsp;
                {true_s}
                <span class='badge-mdl'>{h['model']}</span> &nbsp;
                {badge}
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history  = []
            st.session_state.total_predictions   = 0
            st.session_state.correct_predictions = 0
            st.rerun()