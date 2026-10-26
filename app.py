"""
No-Show Risk Prediction Demo
AKSU B.Sc. Final Year Project — AK22/PHS/CSC/042

Run with:  streamlit run app.py
Needs model_A.pkl saved from the thesis notebook, copied into this same folder.

Styling note: every native widget (selectbox, number input, segmented control,
button) is restyled with CSS rules targeting Streamlit's own data-testid
attributes, with hardcoded colors. This is deliberate — it does NOT rely on
.streamlit/config.toml's theme setting (which follows the visitor's OS/browser
light-dark preference and is what caused the earlier dark-mode styling clash).
Hardcoded CSS on the widget elements themselves renders identically regardless
of the visitor's system theme.
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="No-Show Risk Predictor", page_icon="⌁", layout="centered")

# ---------------------------------------------------------------------------
# Deep CSS override — flat, restrained clinical palette, forced regardless of
# the visitor's OS/browser theme. Targets Streamlit's own data-testid markup.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{
        --text:#212b32; --text-sec:#4c6272; --bg:#f0f4f5; --panel:#ffffff;
        --border:#d8dde0; --blue:#005eb8; --orange:#e8590c; --orange-dark:#a8420a;
        --red:#d5281b; --green:#007f3b;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { max-width: 760px; padding-top: 1.6rem; padding-bottom: 3rem; }

    /* ---- top bar ---- */
    .topbar { display:flex; align-items:center; gap:10px; margin-bottom: 1.6rem; }
    .topbar .mark {
        width:26px; height:26px; border-radius:6px; background:var(--blue);
        display:flex; align-items:center; justify-content:center;
        color:#fff; font-weight:800; font-size:13px; flex-shrink:0;
    }
    .topbar .name { font-size:16px; font-weight:700; color:var(--text); }

    /* ---- section group labels ---- */
    .grouplabel {
        font-size:11.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
        color: var(--text-sec) !important; margin: 1.4rem 0 0.6rem 0;
    }
    .grouplabel.first { margin-top: 0; }

    /* ---- panel wrapper (targets the real st.container(key=...) wrapper) ---- */
    .st-key-form_panel, .st-key-result_panel {
        background: var(--panel) !important; border: 1px solid var(--border) !important;
        border-radius: 6px !important; padding: 22px 24px 18px !important; margin-bottom: 18px;
    }

    /* ---- widget labels ---- */
    [data-testid="stWidgetLabel"] p {
        font-size: 12.5px !important; font-weight: 600 !important; color: var(--text) !important;
    }

    /* ---- number input ---- */
    [data-testid="stNumberInputContainer"] {
        background: #fff !important; border: 1.5px solid var(--border) !important;
        border-radius: 4px !important;
    }
    [data-testid="stNumberInputField"] {
        background: #fff !important; color: var(--text) !important; font-size: 14px !important;
        text-align: center !important;
    }
    [data-testid="stNumberInputContainer"]:focus-within {
        border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(0,94,184,.15) !important;
    }
    /* native +/- steppers hidden — replaced with our own pair below */
    [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display: none !important; }

    /* ---- age stepper buttons (flank the number input) ---- */
    .st-key-age_dec_wrap [data-testid="stButton"] button, .st-key-age_inc_wrap [data-testid="stButton"] button {
        background: #fff !important; color: var(--text-sec) !important; border: 1.5px solid var(--border) !important;
        font-size: 16px !important; font-weight: 700 !important; padding: 0 !important;
        border-radius: 4px !important; box-shadow: none !important; height: 38px !important; min-height: 38px !important;
        letter-spacing: normal; transition: background 0.1s ease, border-color 0.1s ease;
    }
    .st-key-age_dec_wrap [data-testid="stButton"] button p, .st-key-age_inc_wrap [data-testid="stButton"] button p { color: inherit !important; font-weight: 700 !important; font-size: 16px !important; }
    .st-key-age_dec_wrap [data-testid="stButton"] button:hover, .st-key-age_inc_wrap [data-testid="stButton"] button:hover { background: #f0f4f5 !important; border-color: var(--blue) !important; color: var(--blue) !important; }
    .st-key-age_dec_wrap [data-testid="stButton"] button:active, .st-key-age_inc_wrap [data-testid="stButton"] button:active { background: #e4edf5 !important; }

    /* ---- secondary (reset) button ---- */
    .st-key-reset_wrap [data-testid="stButton"] button {
        background: #fff !important; color: var(--text-sec) !important; border: 1.5px solid var(--border) !important;
        font-size: 13px !important; font-weight: 700 !important; padding: 13px 10px !important;
        border-radius: 6px !important; box-shadow: none !important; letter-spacing: .02em;
    }
    .st-key-reset_wrap [data-testid="stButton"] button p { color: inherit !important; font-weight: 700 !important; font-size: 13px !important; }
    .st-key-reset_wrap [data-testid="stButton"] button:hover { background: #f0f4f5 !important; border-color: #c2cbd1 !important; }

    /* ---- segmented control (pill groups) ---- */
    [data-testid="stButtonGroup"] [role="radiogroup"] { gap: 5px !important; flex-wrap: wrap; }
    [data-testid="stButtonGroup"] button[role="radio"] {
        background: #fff !important; border: 1.5px solid var(--border) !important;
        border-radius: 4px !important; color: var(--text-sec) !important;
        font-weight: 600 !important; font-size: 12.5px !important;
        padding: 7px 12px !important; min-height: unset !important;
    }
    [data-testid="stButtonGroup"] button[role="radio"] p { color: inherit !important; font-size: 12.5px !important; font-weight: 600 !important; }
    [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] {
        background: var(--blue) !important; border-color: var(--blue) !important;
    }
    [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] p { color: #fff !important; }

    /* ---- selectbox ---- */
    [data-testid="stSelectbox"] [role="group"] {
        background: #fff !important; border: 1.5px solid var(--border) !important;
        border-radius: 4px !important;
    }
    [data-testid="stSelectbox"] input[role="combobox"] {
        background: #fff !important; color: var(--text) !important; font-size: 14px !important;
    }

    /* ---- primary submit button ---- */
    [data-testid="stButton"] button {
        background: var(--orange) !important; color: #fff !important; border: none !important;
        font-size: 16px !important; font-weight: 800 !important; padding: 13px 10px !important;
        border-radius: 6px !important; box-shadow: 0 4px 0 var(--orange-dark) !important;
        letter-spacing: .02em;
    }
    [data-testid="stButton"] button p { color: #fff !important; font-weight: 800 !important; font-size: 16px !important; }
    [data-testid="stButton"] button:hover { filter: brightness(1.04); }

    /* ---- result ring ---- */
    .resultcore { display:flex; align-items:center; gap: 22px; }
    .ring {
        width: 96px; height: 96px; border-radius: 50%; flex-shrink: 0;
        display:flex; align-items:center; justify-content:center; position: relative;
    }
    .ring::after { content:""; position:absolute; width:72px; height:72px; border-radius:50%; background:#fff; }
    .ring .pctnum { position:relative; z-index:1; font-size:20px; font-weight:800; color:var(--text); }
    .result-label {
        display:inline-block; font-size:12.5px; font-weight:700; padding:4px 12px;
        border-radius:3px; margin-bottom:6px;
    }
    .result-title { font-size:18px; font-weight:700; margin:0; color: var(--text); }

    .shap-title {
        font-size:11.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
        color: var(--text-sec); margin: 1.2rem 0 0.7rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1. Load the trained model (Model A only — the system in Chapters 3/4 is the
#    one with family-support factors; the ablation comparison against Model B
#    lives in Chapter 4's tables/figures, not in this live demo)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model_A.pkl", "rb") as f:
        return pickle.load(f)

model_A = load_model()

FB = ['age', 'sex_m', 'travel', 'lead', 'wait', 'missed',
      'reminded', 'accompanied', 'transport', 'cost_help', 'encouraged', 'living_c']

# Option sets match the research instrument exactly (Table 3.3) and the
# ordinal encoding the model was actually trained on (see notebook ord_cols).
SHORT = {
    'travel': ['<30 min', '30–60 min', '1–2 hrs', '>2 hrs'],
    'wait':   ['<1 hr', '1–2 hrs', '2–3 hrs', '>3 hrs'],
    'lead':   ['Same day', '1–3 days', '4–7 days', '>7 days'],
    'missed': ['0', '1', '2', '3', '4+'],
    'freq':   ['Never', 'Rarely', 'Sometimes', 'Always'],
}
LIVING_OPTIONS = ['Alone', 'Friends', 'Other', 'Other relatives', 'Parents/siblings', 'Spouse/children']


def seg(label, key, options):
    choice = st.segmented_control(label, options, default=options[0], key=key)
    return options.index(choice) if choice is not None else 0


AGE_MIN, AGE_MAX, AGE_STEP = 0, 100, 1

# Explicit default values, not just a list of keys to delete — segmented
# controls don't reliably fall back to their `default=` option after a
# plain `del st.session_state[key]`, so Reset sets each one back to its
# known starting value directly.
RESET_DEFAULTS = {
    "age": 0,
    "sex": "Male",
    "missed": SHORT['missed'][0],
    "travel": SHORT['travel'][0],
    "lead": SHORT['lead'][0],
    "wait": SHORT['wait'][0],
    "reminded": SHORT['freq'][0],
    "transport": SHORT['freq'][0],
    "encouraged": SHORT['freq'][0],
    "accompanied": SHORT['freq'][0],
    "cost_help": SHORT['freq'][0],
    "living": LIVING_OPTIONS[0],
}


def _reset_form():
    for k, v in RESET_DEFAULTS.items():
        st.session_state[k] = v


def _dec_age():
    st.session_state.age = max(AGE_MIN, st.session_state.get("age", 0) - AGE_STEP)


def _inc_age():
    st.session_state.age = min(AGE_MAX, st.session_state.get("age", 0) + AGE_STEP)


st.session_state.setdefault("age", 0)


st.markdown(
    '<div class="topbar"><div class="mark">N</div><div class="name">No-Show Risk Predictor</div></div>',
    unsafe_allow_html=True,
)

with st.container(key="form_panel"):
    st.markdown('<p class="grouplabel first">Patient</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p style="font-size:12.5px; font-weight:600; color:var(--text); margin-bottom:0.35rem;">Age</p>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns([1, 2, 1], gap="small")
        with a1:
            with st.container(key="age_dec_wrap"):
                st.button("−", key="age_dec", on_click=_dec_age, use_container_width=True)
        with a2:
            age = st.number_input(
                "Age", min_value=AGE_MIN, max_value=AGE_MAX, step=AGE_STEP,
                key="age", label_visibility="collapsed",
            )
        with a3:
            with st.container(key="age_inc_wrap"):
                st.button("+", key="age_inc", on_click=_inc_age, use_container_width=True)
    with c2:
        sex_i = seg("Sex", "sex", ["Male", "Female"])
    with c3:
        missed_i = seg("Missed appointments (12mo)", "missed", SHORT['missed'])

    st.markdown('<p class="grouplabel">This appointment</p>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        travel_i = seg("Travel time", "travel", SHORT['travel'])
    with c5:
        lead_i = seg("Booked in advance", "lead", SHORT['lead'])
    with c6:
        wait_i = seg("Usual waiting time", "wait", SHORT['wait'])

    st.markdown('<p class="grouplabel">Family support</p>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        reminded_i = seg("Reminded to attend", "reminded", SHORT['freq'])
        transport_i = seg("Transport help", "transport", SHORT['freq'])
        encouraged_i = seg("Encouraged to attend", "encouraged", SHORT['freq'])
    with c8:
        accompanied_i = seg("Accompanied", "accompanied", SHORT['freq'])
        cost_help_i = seg("Cost-of-care help", "cost_help", SHORT['freq'])
        living = st.selectbox("Living arrangement", LIVING_OPTIONS, key="living")

    st.write("")
    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        with st.container(key="reset_wrap"):
            st.button("Reset", key="reset_btn", on_click=_reset_form)
    with bcol2:
        submitted = st.button("Predict Risk", use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Result
# ---------------------------------------------------------------------------
if submitted and age < 15:
    st.warning("Please enter a valid age (15–100) before predicting.")
elif submitted:
    row = {
        'age': age, 'sex_m': sex_i, 'travel': travel_i, 'lead': lead_i, 'wait': wait_i,
        'missed': missed_i, 'reminded': reminded_i, 'accompanied': accompanied_i,
        'transport': transport_i, 'cost_help': cost_help_i, 'encouraged': encouraged_i,
        'living_c': LIVING_OPTIONS.index(living),
    }
    X = pd.DataFrame([row])
    pct = model_A.predict_proba(X[FB])[0, 1] * 100

    if pct < 33:
        label, bg, fg = "Low risk", "#e4f3ea", "#007f3b"
    elif pct < 60:
        label, bg, fg = "Moderate risk", "#fdf1e0", "#ed8b00"
    else:
        label, bg, fg = "High risk", "#fbe9e7", "#d5281b"

    ring_css = f"conic-gradient({fg} 0% {min(pct,100):.0f}%, #f0f1f2 {min(pct,100):.0f}% 100%)"
    title_text = "Likely to miss this appointment" if pct >= 50 else "Likely to attend this appointment"

    with st.container(key="result_panel"):
        st.markdown(
            f"""
            <div class="resultcore">
                <div class="ring" style="background:{ring_css};"><div class="pctnum">{pct:.0f}%</div></div>
                <div>
                    <div class="result-label" style="background:{bg}; color:{fg};">{label}</div>
                    <p class="result-title">{title_text}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="shap-title">Key factors</p>', unsafe_allow_html=True)
        try:
            import shap
            import shap.explainers._tree as _shap_tree_mod

            # Some xgboost releases (>= ~2.x) write the booster's base_score
            # as "[5E-1]" instead of a plain "0.5". Several shap releases
            # can't parse that bracketed form and raise
            # "ValueError: could not convert string to float: '[5E-1]'" from
            # inside shap's own XGBTreeModelLoader. Patch shap's internal
            # module to strip any stray brackets before parsing, regardless
            # of which shap/xgboost versions end up installed.
            if not getattr(_shap_tree_mod, "_basescore_patch_applied", False):
                _orig_float = float
                def _patched_float(x, _orig=_orig_float):
                    if isinstance(x, str):
                        x = x.strip("[]")
                    return _orig(x)
                _shap_tree_mod.float = _patched_float
                _shap_tree_mod._basescore_patch_applied = True

            explainer = shap.TreeExplainer(model_A)
            sv = explainer.shap_values(X[FB])[0]
            contrib = pd.Series(sv, index=FB).sort_values(key=abs, ascending=False).head(5)

            label_map = {
                'age': 'Age', 'sex_m': 'Sex', 'travel': 'Travel time', 'lead': 'Booked in advance',
                'wait': 'Waiting time', 'missed': 'Prior missed appointments', 'reminded': 'Reminded to attend',
                'accompanied': 'Accompanied', 'transport': 'Transport help', 'cost_help': 'Cost-of-care help',
                'encouraged': 'Encouraged to attend', 'living_c': 'Living arrangement',
            }
            max_abs = contrib.abs().max() or 1
            for feat, val in contrib.items():
                width_pct = abs(val) / max_abs * 100
                color = "#d5281b" if val > 0 else "#4c6272"
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:7px; font-size:13px;">
                        <div style="width:170px; text-align:right; color:var(--text); flex-shrink:0;">{label_map.get(feat, feat)}</div>
                        <div style="height:12px; border-radius:2px; width:{width_pct*1.4:.0f}px; background:{color};"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            st.caption("Install `shap` to see a per-prediction explanation here.")
