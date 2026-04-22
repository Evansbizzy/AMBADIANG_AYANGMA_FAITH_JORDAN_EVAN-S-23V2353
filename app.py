# =============================================================================
#  app.py — INF 232 | Analyse des Données de Transport Urbain (Taxis, Yaoundé)
#  Version : 2.1 — Bug fixes + Dashboard upgrade
#  Fixes   : trendline="ols" remplacé par numpy polyfit (statsmodels non requis)
#            Police Syne → Plus Jakarta Sans (fin de l'effet "allongé")
#  Stack   : Streamlit · Supabase · Plotly · Scikit-learn
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from streamlit_option_menu import option_menu

# ─────────────────────────────────────────────────────────────────────────────
#  0. PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TaxiData YDE — INF 232",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  1. CSS — Dark Mode v2.1
#     FIX: Syne (elongated on Linux) → Plus Jakarta Sans
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

/* ── Design Tokens ── */
:root {
    --bg-base:        #0f1117;
    --bg-card:        #181d2a;
    --bg-card-inner:  #1e2537;
    --bg-input:       #242b3d;
    --border:         rgba(99, 179, 237, 0.10);
    --border-focus:   rgba(56, 189, 248, 0.50);
    --accent:         #38bdf8;
    --accent-2:       #818cf8;
    --accent-soft:    rgba(56, 189, 248, 0.09);
    --accent-glow:    rgba(56, 189, 248, 0.20);
    --accent-glow-lg: rgba(56, 189, 248, 0.07);
    --text-primary:   #e2e8f4;
    --text-muted:     #64748b;
    --text-dim:       #334155;
    --success:        #34d399;
    --success-soft:   rgba(52, 211, 153, 0.10);
    --warning:        #fbbf24;
    --danger:         #f87171;
    --radius-lg:      16px;
    --radius:         11px;
    --radius-sm:      7px;
    --shadow-card:    0 2px 24px rgba(0,0,0,0.50), 0 1px 0 rgba(255,255,255,0.03) inset;
    --shadow-glow:    0 0 32px var(--accent-glow-lg), 0 4px 28px rgba(0,0,0,0.55);
    --font-display:   'Plus Jakarta Sans', sans-serif;
    --font-body:      'DM Sans', sans-serif;
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
#MainMenu, footer, header, .stDeployButton {
    visibility: hidden !important; display: none !important;
}
.block-container { padding: 1.4rem 2rem 3rem !important; max-width: 1320px; }

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}
h1 { font-size: 1.65rem !important; font-weight: 800 !important; }
h2 { font-size: 1.15rem !important; font-weight: 700 !important; }
h3 { font-size: 0.98rem !important; font-weight: 600 !important; }
h4 { font-size: 0.88rem !important; font-weight: 600 !important; color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.06em; }

/* ══════════════════════════════════
   NAV BAR
══════════════════════════════════ */
nav.nav {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 5px 8px !important;
    box-shadow: var(--shadow-card) !important;
    margin-bottom: 1.5rem !important;
}
nav.nav .nav-item .nav-link {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.15s ease !important;
    padding: 8px 22px !important;
    letter-spacing: 0.01em;
}
nav.nav .nav-item .nav-link:hover {
    color: var(--accent) !important; background: var(--accent-soft) !important;
}
nav.nav .nav-item .nav-link.active {
    color: #0d1117 !important; background: var(--accent) !important;
    box-shadow: 0 2px 10px var(--accent-glow) !important;
}
nav.nav .nav-item .nav-link.active i { color: #0d1117 !important; }

/* ══════════════════════════════════
   CARDS
══════════════════════════════════ */
.form-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem 2.2rem 1.8rem;
    box-shadow: var(--shadow-card);
    max-width: 620px;
    margin: 0 auto;
    transition: box-shadow 0.3s ease;
}
.form-card:hover { box-shadow: var(--shadow-glow); }

.dash-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    box-shadow: var(--shadow-card);
    margin-bottom: 0.85rem;
}

/* ══════════════════════════════════
   METRIC CARDS
══════════════════════════════════ */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.4rem !important;
    box-shadow: var(--shadow-card) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    position: relative; overflow: hidden;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px); box-shadow: var(--shadow-glow) !important;
}
/* Subtle top accent line */
div[data-testid="metric-container"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    opacity: 0.6;
}
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font-display) !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}

/* ══════════════════════════════════
   INPUTS
══════════════════════════════════ */
.stTextInput input, .stNumberInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.18s, box-shadow 0.18s;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stSlider label, .stRadio label {
    color: var(--text-muted) !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-family: var(--font-body) !important;
}

/* ══════════════════════════════════
   RADIO → Button-Group Pills
══════════════════════════════════ */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    margin-top: 4px !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px 12px !important;
    cursor: pointer !important;
    transition: all 0.14s ease !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-family: var(--font-body) !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 0 1px rgba(56,189,248,0.25) !important;
    font-weight: 600 !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child { display: none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div > p {
    font-size: 0.79rem !important; font-family: var(--font-body) !important; color: inherit !important;
}

/* ══════════════════════════════════
   BUTTONS
══════════════════════════════════ */
.stButton > button {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    padding: 0.48rem 1.1rem !important;
    transition: all 0.16s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--accent-glow) !important;
}
.btn-submit .stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border-color: var(--accent) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
}
.btn-submit .stButton > button:hover {
    background: #7dd3fc !important;
    box-shadow: 0 6px 18px rgba(56,189,248,0.35) !important;
}

/* ══════════════════════════════════
   ALERTS
══════════════════════════════════ */
.stSuccess { background: var(--success-soft) !important; border-left: 3px solid var(--success) !important; border-radius: var(--radius-sm) !important; }
.stWarning { background: rgba(251,191,36,0.08)  !important; border-left: 3px solid var(--warning) !important; border-radius: var(--radius-sm) !important; }
.stError   { background: rgba(248,113,113,0.08) !important; border-left: 3px solid var(--danger)  !important; border-radius: var(--radius-sm) !important; }
.stInfo    { background: var(--accent-soft)     !important; border-left: 3px solid var(--accent)  !important; border-radius: var(--radius-sm) !important; }

/* ══════════════════════════════════
   STEPPER
══════════════════════════════════ */
.stepper-wrap {
    display: flex; align-items: flex-start; justify-content: center;
    gap: 0; margin: 0 auto 1.8rem; max-width: 380px;
}
.step-col { display: flex; flex-direction: column; align-items: center; gap: 5px; }
.step-bubble {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-display); font-weight: 700; font-size: 0.78rem;
    border: 2px solid var(--text-dim);
    background: var(--bg-card-inner); color: var(--text-muted);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1); z-index: 1;
}
.step-bubble.active {
    border-color: var(--accent); background: var(--accent); color: #0d1117;
    box-shadow: 0 0 0 4px rgba(56,189,248,0.15), 0 0 16px var(--accent-glow);
}
.step-bubble.done {
    border-color: var(--success); background: var(--success); color: #0d1117;
}
.step-label {
    font-family: var(--font-body); font-size: 0.6rem; color: var(--text-muted);
    text-align: center; letter-spacing: 0.06em; text-transform: uppercase;
}
.step-label.active { color: var(--accent); }
.step-label.done   { color: var(--success); }
.step-connector {
    flex: 1; height: 2px; background: var(--text-dim);
    margin: 15px 3px 0; border-radius: 2px;
    transition: background 0.3s; min-width: 32px;
}
.step-connector.done { background: var(--success); }

/* ══════════════════════════════════
   STEP HEADER
══════════════════════════════════ */
.step-header {
    margin-bottom: 1.3rem; padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--border);
}
.step-header h3 { margin: 0 0 0.12rem; font-size: 0.97rem !important; }
.step-header p  { margin: 0; color: var(--text-muted); font-size: 0.76rem; }

/* ══════════════════════════════════
   RECAP BOX
══════════════════════════════════ */
.recap-box {
    background: var(--bg-card-inner); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.85rem 1.1rem;
    margin-bottom: 1.1rem; font-size: 0.8rem;
}
.recap-box strong {
    color: var(--text-muted); font-weight: 500;
    font-size: 0.66rem; text-transform: uppercase;
    letter-spacing: 0.07em; display: block; margin-bottom: 1px;
}
.recap-val { color: var(--accent); font-weight: 600; }

/* ══════════════════════════════════
   MISC
══════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.1rem 0 !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
.badge {
    display: inline-block; padding: 2px 9px;
    border-radius: 99px; font-size: 0.68rem; font-weight: 600;
    font-family: var(--font-body); letter-spacing: 0.04em;
}
.badge-info    { background: rgba(56,189,248,0.11);  color: var(--accent);  border: 1px solid rgba(56,189,248,0.2); }
.badge-success { background: rgba(52,211,153,0.11);  color: var(--success); border: 1px solid rgba(52,211,153,0.2); }
.badge-warn    { background: rgba(251,191,36,0.11);  color: var(--warning); border: 1px solid rgba(251,191,36,0.2); }
.badge-purple  { background: rgba(129,140,248,0.11); color: var(--accent-2); border: 1px solid rgba(129,140,248,0.2); }

.field-label {
    font-size: 0.66rem; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 4px; font-family: var(--font-body);
}
.spacer-sm { margin-top: 0.8rem; }
.spacer    { margin-top: 1.1rem; }

/* Section divider with label */
.section-label {
    font-family: var(--font-display); font-size: 0.8rem; font-weight: 700;
    color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.08em; margin: 1.4rem 0 0.9rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ""; flex: 1; height: 1px; background: var(--border);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  2. SUPABASE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase_client() -> Client:
    url: str = st.secrets["supabase"]["url"]
    key: str = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = get_supabase_client()
TABLE = "taxi_rides"


# ─────────────────────────────────────────────────────────────────────────────
#  3. DATA ACCESS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_all_rides() -> pd.DataFrame:
    try:
        resp = supabase.table(TABLE).select("*").execute()
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur Supabase : {e}")
        return pd.DataFrame()


def insert_ride(data: dict) -> bool:
    try:
        supabase.table(TABLE).insert(data).execute()
        fetch_all_rides.clear()
        return True
    except Exception as e:
        st.error(f"Erreur d'insertion : {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  4. RÉFÉRENTIELS
# ─────────────────────────────────────────────────────────────────────────────
QUARTIERS = [
    "Bastos", "Biyem-Assi", "Mokolo", "Ngoa-Ekellé", "Melen",
    "Mvan", "Obili", "Essos", "Tsinga", "Emana",
    "Omnisports", "Mimboman", "Nsimalen", "Nlongkak", "Cité Verte",
    "Santa Barbara", "Ebolowa-si", "Ngousso", "Etoudi", "Damase",
]
METEO_OPTIONS       = ["Ensoleillé", "Nuageux", "Pluvieux", "Très Pluvieux"]
HEURE_OPTIONS       = ["Matin (6h–9h)", "Journée (9h–17h)", "Soir (17h–20h)", "Nuit (20h–6h)"]
TRAFIC_OPTIONS      = ["Fluide", "Modéré", "Dense", "Embouteillage"]
TYPE_COURSE_OPTIONS = ["Ramassage", "Dépôt"]


# ─────────────────────────────────────────────────────────────────────────────
#  5. UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str, icon: str = ""):
    st.markdown(f"""
        <div style="margin-bottom:1.1rem;">
            <h1 style="margin:0;line-height:1.2;">{icon} {title}</h1>
            <p style="color:var(--text-muted);margin:0.2rem 0 0;font-size:0.8rem;">{subtitle}</p>
        </div><hr/>
    """, unsafe_allow_html=True)


def stepper(current: int):
    labels = ["Contexte", "Trajet", "Paiement"]
    html   = ""
    for i in range(1, 4):
        if   i < current:  b, l, ic = "done",   "done",   "✓"
        elif i == current: b, l, ic = "active", "active", str(i)
        else:              b, l, ic = "",        "",       str(i)
        html += f'<div class="step-col"><div class="step-bubble {b}">{ic}</div><span class="step-label {l}">{labels[i-1]}</span></div>'
        if i < 3:
            cc = "done" if i < current else ""
            html += f'<div class="step-connector {cc}"></div>'
    st.markdown(f'<div class="stepper-wrap">{html}</div>', unsafe_allow_html=True)


def info_badge(tag: str, variant: str, body: str):
    st.markdown(
        f'<div class="dash-card" style="border-left:3px solid var(--accent);">'
        f'<span class="badge badge-{variant}">{tag}</span>&nbsp;&nbsp;{body}</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  6. PLOTLY THEME HELPER
#     FIX: No more trendline="ols" — trendlines are drawn manually with numpy
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    title_font=dict(family="Plus Jakarta Sans", color="#e2e8f4", size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(t=44, b=32, l=16, r=16),
)


def add_trendline(fig: go.Figure, x: np.ndarray, y: np.ndarray,
                  name: str = "Tendance", color: str = "#f87171") -> go.Figure:
    """
    Ajoute une droite de régression OLS calculée avec numpy.polyfit.
    Remplace trendline='ols' de Plotly Express qui nécessite statsmodels.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return fig
    coef = np.polyfit(x[mask], y[mask], 1)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 120)
    y_line = np.polyval(coef, x_line)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name=name,
        line=dict(color=color, width=2, dash="dash"),
        hovertemplate=f"Tendance: %{{y:.0f}} FCFA<extra></extra>",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  7. COLLECTE TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_collecte_tab():
    page_header("Collecte de Données",
                "Enregistrement d'une nouvelle course de taxi à Yaoundé")

    defaults = dict(
        step=1, meteo=METEO_OPTIONS[0], heure=HEURE_OPTIONS[0],
        trafic=TRAFIC_OPTIONS[0], depart=QUARTIERS[0], arrivee=QUARTIERS[1],
        distance_km=2.0, type_course=TYPE_COURSE_OPTIONS[0], prix_fcfa=500,
        submitted=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _, col, _ = st.columns([1, 2.6, 1])

    with col:
        stepper(st.session_state.step)
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        # ── Étape 1 ──────────────────────────────────────────────────────────
        if st.session_state.step == 1:
            st.markdown("""
                <div class="step-header">
                    <h3>Contexte de la Course</h3>
                    <p>Conditions au moment de la prise en charge</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="field-label">Conditions Météorologiques</p>', unsafe_allow_html=True)
            st.session_state.meteo = st.radio(
                "Météo", METEO_OPTIONS,
                index=METEO_OPTIONS.index(st.session_state.meteo),
                horizontal=True, label_visibility="collapsed",
            )
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

            st.markdown('<p class="field-label">État du Trafic</p>', unsafe_allow_html=True)
            st.session_state.trafic = st.radio(
                "Trafic", TRAFIC_OPTIONS,
                index=TRAFIC_OPTIONS.index(st.session_state.trafic),
                horizontal=True, label_visibility="collapsed",
            )
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

            st.markdown('<p class="field-label">Plage Horaire</p>', unsafe_allow_html=True)
            st.session_state.heure = st.selectbox(
                "Heure", HEURE_OPTIONS,
                index=HEURE_OPTIONS.index(st.session_state.heure),
                label_visibility="collapsed",
            )
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
            _, bc = st.columns([3, 1])
            with bc:
                if st.button("Suivant →", key="next_1"):
                    st.session_state.step = 2; st.rerun()

        # ── Étape 2 ──────────────────────────────────────────────────────────
        elif st.session_state.step == 2:
            st.markdown("""
                <div class="step-header">
                    <h3>Informations sur le Trajet</h3>
                    <p>Origine, destination et distance parcourue</p>
                </div>
            """, unsafe_allow_html=True)

            cd, ca = st.columns(2)
            with cd:
                d_idx = QUARTIERS.index(st.session_state.depart) if st.session_state.depart in QUARTIERS else 0
                st.session_state.depart = st.selectbox("Quartier de Départ", QUARTIERS, index=d_idx)
            with ca:
                a_idx = QUARTIERS.index(st.session_state.arrivee) if st.session_state.arrivee in QUARTIERS else 1
                st.session_state.arrivee = st.selectbox("Quartier d'Arrivée", QUARTIERS, index=a_idx)

            if st.session_state.depart == st.session_state.arrivee:
                st.warning("Le départ et l'arrivée sont identiques.")

            st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
            ck, cb = st.columns([2, 1])
            with ck:
                st.session_state.distance_km = st.number_input(
                    "Distance estimée (km)", min_value=0.5, max_value=80.0,
                    value=float(st.session_state.distance_km), step=0.5,
                )
            with cb:
                km  = st.session_state.distance_km
                cat = "Courte" if km < 5 else ("Moyenne" if km < 15 else "Longue")
                st.markdown(f"""
                    <div style="margin-top:1.8rem;background:var(--bg-card-inner);
                                border:1px solid var(--border);border-radius:var(--radius-sm);
                                padding:0.5rem 0.7rem;text-align:center;">
                        <span style="font-size:0.63rem;color:var(--text-muted);text-transform:uppercase;
                                     letter-spacing:.07em;display:block;">Catégorie</span>
                        <span style="font-weight:700;font-size:0.8rem;color:var(--accent);">{cat}</span>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
            cp, _, cn = st.columns([1, 2, 1])
            with cp:
                if st.button("← Retour", key="prev_2"):
                    st.session_state.step = 1; st.rerun()
            with cn:
                if st.button("Suivant →", key="next_2"):
                    if st.session_state.depart != st.session_state.arrivee:
                        st.session_state.step = 3; st.rerun()
                    else:
                        st.error("Choisissez des quartiers différents.")

        # ── Étape 3 ──────────────────────────────────────────────────────────
        elif st.session_state.step == 3:
            st.markdown("""
                <div class="step-header">
                    <h3>Paiement de la Course</h3>
                    <p>Type de service et montant final payé au chauffeur</p>
                </div>
            """, unsafe_allow_html=True)

            m_l  = st.session_state.meteo.split(" ")[0]
            tr_l = st.session_state.trafic.split(" ")[0]
            h_l  = st.session_state.heure.split("(")[0].strip()
            st.markdown(f"""
                <div class="recap-box">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.45rem 1.5rem;">
                        <div><strong>Météo</strong><span class="recap-val">{m_l}</span></div>
                        <div><strong>Trafic</strong><span class="recap-val">{tr_l}</span></div>
                        <div><strong>Départ</strong><span class="recap-val">{st.session_state.depart}</span></div>
                        <div><strong>Arrivée</strong><span class="recap-val">{st.session_state.arrivee}</span></div>
                        <div><strong>Distance</strong><span class="recap-val">{st.session_state.distance_km} km</span></div>
                        <div><strong>Heure</strong><span class="recap-val">{h_l}</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            ct, cp2 = st.columns(2)
            with ct:
                st.session_state.type_course = st.radio(
                    "Type de Course", TYPE_COURSE_OPTIONS,
                    index=TYPE_COURSE_OPTIONS.index(st.session_state.type_course),
                    horizontal=True,
                )
            with cp2:
                st.session_state.prix_fcfa = st.number_input(
                    "Prix Final (FCFA)", min_value=100, max_value=50000,
                    value=int(st.session_state.prix_fcfa), step=50,
                )

            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
            c_prev, _, c_sub = st.columns([1, 1.2, 1.5])
            with c_prev:
                if st.button("← Retour", key="prev_3"):
                    st.session_state.step = 2; st.rerun()
            with c_sub:
                st.markdown('<div class="btn-submit">', unsafe_allow_html=True)
                if st.button("Soumettre", key="submit"):
                    record = {
                        "depart":      st.session_state.depart,
                        "arrivee":     st.session_state.arrivee,
                        "distance_km": round(float(st.session_state.distance_km), 2),
                        "heure":       st.session_state.heure,
                        "meteo":       st.session_state.meteo.split(" ")[0],
                        "trafic":      st.session_state.trafic.split(" ")[0],
                        "type_course": st.session_state.type_course,
                        "prix_fcfa":   int(st.session_state.prix_fcfa),
                    }
                    if insert_ride(record):
                        st.session_state.submitted = True
                        for k, v in defaults.items():
                            if k != "submitted":
                                st.session_state[k] = v
                        st.rerun()
                    else:
                        st.error("Insertion échouée — vérifiez la connexion Supabase.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close form-card

    if st.session_state.submitted:
        st.success("Course enregistrée avec succès dans la base de données Supabase !")
        st.session_state.submitted = False


# ─────────────────────────────────────────────────────────────────────────────
#  8. DASHBOARD TAB
#     FIX: trendline="ols" supprimé, remplacé par add_trendline() → numpy only
#     AMÉLIORÉ: graphiques supplémentaires, meilleure mise en page
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard_tab():
    page_header("Tableau de Bord",
                "Analyse descriptive et visualisation des données collectées")

    df = fetch_all_rides()
    if df.empty:
        st.info("Aucune donnée disponible. Commencez par saisir des courses dans **Collecte**.")
        return

    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["prix_fcfa"]   = pd.to_numeric(df["prix_fcfa"],   errors="coerce")
    df.dropna(subset=["distance_km", "prix_fcfa"], inplace=True)
    n = len(df)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Courses",          f"{n}")
    c2.metric("Prix moyen",       f"{df['prix_fcfa'].mean():,.0f} FCFA")
    c3.metric("Prix médian",      f"{df['prix_fcfa'].median():,.0f} FCFA")
    c4.metric("Distance moy.",    f"{df['distance_km'].mean():.1f} km")
    c5.metric("Prix / km moyen",  f"{(df['prix_fcfa']/df['distance_km']).mean():.0f} FCFA")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 : Scatter (avec trendline numpy) + Violin ──────────────────────
    section_label("Analyse Bivariée")
    col_l, col_r = st.columns(2)

    with col_l:
        # Scatter — couleur par météo, trendline via numpy (pas de statsmodels)
        fig_sc = px.scatter(
            df, x="distance_km", y="prix_fcfa", color="meteo",
            size="prix_fcfa",
            hover_data=["depart", "arrivee", "trafic", "type_course"],
            title="Distance vs Prix",
            labels={"distance_km": "Distance (km)", "prix_fcfa": "Prix (FCFA)", "meteo": "Météo"},
            color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#f87171"],
        )
        # ─ trendline manuelle ─
        fig_sc = add_trendline(
            fig_sc,
            df["distance_km"].values,
            df["prix_fcfa"].values,
            name="Tendance globale",
            color="#818cf8",
        )
        fig_sc.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r:
        # Violin — distribution des prix par état du trafic
        fig_vio = go.Figure()
        for traf in df["trafic"].unique():
            sub = df[df["trafic"] == traf]["prix_fcfa"]
            fig_vio.add_trace(go.Violin(
                y=sub, name=traf, box_visible=True, meanline_visible=True,
                points="outliers", hoverinfo="y+name",
            ))
        fig_vio.update_layout(**PLOTLY_DARK,
                              title="Distribution des Prix par Trafic",
                              yaxis_title="Prix (FCFA)", xaxis_title="État du Trafic",
                              showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── Row 2 : Histogramme overlayé + Donut ─────────────────────────────────
    section_label("Répartitions")
    ca, cb = st.columns(2)

    with ca:
        fig_h = px.histogram(
            df, x="prix_fcfa", nbins=20, color="type_course", barmode="overlay",
            title="Distribution des Prix par Type de Course",
            labels={"prix_fcfa": "Prix (FCFA)", "type_course": "Type"},
            color_discrete_map={"Ramassage": "#38bdf8", "Dépôt": "#34d399"}, opacity=0.72,
        )
        fig_h.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig_h, use_container_width=True)

    with cb:
        mc = df["meteo"].value_counts().reset_index()
        mc.columns = ["meteo", "count"]
        fig_d = px.pie(
            mc, names="meteo", values="count",
            title="Répartition par Météo",
            color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#f87171"],
            hole=0.52,
        )
        fig_d.update_traces(textposition="outside", textinfo="percent+label")
        fig_d.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Row 3 : Bar groupé moyen + Heatmap ───────────────────────────────────
    section_label("Analyse Croisée")
    cc, cd = st.columns(2)

    with cc:
        # Prix moyen par heure × type de course
        grp = df.groupby(["heure", "type_course"])["prix_fcfa"].mean().reset_index()
        grp["prix_fcfa"] = grp["prix_fcfa"].round(0)
        fig_bar = px.bar(
            grp, x="heure", y="prix_fcfa", color="type_course", barmode="group",
            title="Prix Moyen par Plage Horaire et Type",
            labels={"heure": "Heure", "prix_fcfa": "Prix moyen (FCFA)", "type_course": "Type"},
            color_discrete_map={"Ramassage": "#38bdf8", "Dépôt": "#34d399"},
            text_auto=True,
        )
        fig_bar.update_traces(texttemplate="%{y:.0f}", textposition="outside")
        fig_bar.update_layout(**PLOTLY_DARK, xaxis_tickangle=-20)
        st.plotly_chart(fig_bar, use_container_width=True)

    with cd:
        # Heatmap prix moyen — Météo × Trafic
        pivot = df.pivot_table(
            values="prix_fcfa", index="meteo", columns="trafic", aggfunc="mean"
        ).round(0)
        if not pivot.empty:
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="Blues",
                text=pivot.values.astype(int),
                texttemplate="%{text} FCFA",
                hoverongaps=False,
            ))
            fig_heat.update_layout(**PLOTLY_DARK,
                                   title="Prix Moyen (FCFA) — Météo × Trafic",
                                   xaxis_title="État du Trafic",
                                   yaxis_title="Météo")
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Données brutes ────────────────────────────────────────────────────────
    with st.expander("Voir les données brutes"):
        display_cols = [c for c in
                        ["depart","arrivee","distance_km","heure",
                         "meteo","trafic","type_course","prix_fcfa"]
                        if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values("prix_fcfa", ascending=False),
            use_container_width=True, height=260,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  9. MODÈLES IA TAB
# ─────────────────────────────────────────────────────────────────────────────
MIN_REG = 10
MIN_CLF = 15


def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["prix_fcfa"]   = pd.to_numeric(df["prix_fcfa"],   errors="coerce")
    df.dropna(subset=["distance_km","prix_fcfa","meteo","trafic","type_course"], inplace=True)
    df["meteo_enc"]  = LabelEncoder().fit_transform(df["meteo"])
    df["trafic_enc"] = LabelEncoder().fit_transform(df["trafic"])
    df["type_enc"]   = LabelEncoder().fit_transform(df["type_course"])
    return df


def render_models_tab():
    page_header("Modèles d'Intelligence Artificielle",
                "Régression linéaire et classification supervisée")

    df_raw = fetch_all_rides()
    if df_raw.empty:
        st.info("Aucune donnée disponible."); return

    df = _encode_df(df_raw)
    n  = len(df)

    if n < MIN_REG:
        st.warning(f"Minimum **{MIN_REG} observations** requises. Actuellement : **{n}**.")
        return

    # ── Modèle 1 — Régression Linéaire Simple ────────────────────────────────
    section_label("Modèle 1 — Régression Linéaire Simple")
    info_badge("RLS", "info",
        "Prédit le <strong>prix</strong> d'une course à partir de la seule <strong>distance (km)</strong>.")

    X1 = df[["distance_km"]].values
    y1 = df["prix_fcfa"].values
    m1 = LinearRegression().fit(X1, y1)
    r2_1 = r2_score(y1, m1.predict(X1))
    a, b = m1.coef_[0], m1.intercept_

    ceq, cr2 = st.columns(2)
    with ceq:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown(f"""
**Équation :**
$$\\hat{{Prix}} = {a:.1f} \\times d + {b:.1f}$$

Chaque km supplémentaire augmente le prix d'environ **{a:.0f} FCFA**.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    with cr2:
        q = "Très bon" if r2_1 >= 0.75 else ("Moyen" if r2_1 >= 0.50 else "Faible")
        st.metric("Score R²", f"{r2_1:.4f}")
        st.caption(f"Qualité : {q}")
        st.progress(min(max(r2_1, 0.0), 1.0))

    # Scatter + trendline numpy (pas de statsmodels)
    fig_r1 = go.Figure()
    fig_r1.add_trace(go.Scatter(
        x=df["distance_km"], y=df["prix_fcfa"], mode="markers",
        name="Données réelles",
        marker=dict(color="#38bdf8", size=8, opacity=0.75,
                    line=dict(color="#0d1117", width=0.5)),
    ))
    x_line = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 120)
    y_line = m1.predict(x_line.reshape(-1, 1))
    fig_r1.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name="Droite ajustée",
        line=dict(color="#f87171", width=2.5, dash="dash"),
    ))
    fig_r1.update_layout(**PLOTLY_DARK,
                         title="Régression Linéaire Simple — Distance vs Prix",
                         xaxis_title="Distance (km)", yaxis_title="Prix (FCFA)")
    st.plotly_chart(fig_r1, use_container_width=True)
    st.divider()

    # ── Modèle 2 — Régression Linéaire Multiple ──────────────────────────────
    section_label("Modèle 2 — Régression Linéaire Multiple")
    info_badge("RLM", "purple",
        "Prédit le <strong>prix</strong> en combinant distance, trafic <em>et</em> météo.")

    X2 = df[["distance_km","trafic_enc","meteo_enc"]].values
    y2 = df["prix_fcfa"].values
    m2 = LinearRegression().fit(X2, y2)
    r2_2  = r2_score(y2, m2.predict(X2))
    coefs = m2.coef_

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Multiple",      f"{r2_2:.4f}")
    c2.metric("Coeff. Distance",  f"{coefs[0]:.2f} FCFA/km")
    c3.metric("Δ R² vs Modèle 1", f"{r2_2 - r2_1:+.4f}",
              help="Gain en pouvoir explicatif par rapport à la régression simple.")

    st.markdown(f"""
**Équation :**
$$\\hat{{Prix}} = {coefs[0]:.2f} \\cdot d + {coefs[1]:.2f} \\cdot Trafic + {coefs[2]:.2f} \\cdot Météo + {m2.intercept_:.2f}$$

> *Trafic* et *Météo* sont encodés numériquement (LabelEncoder). Leur interprétation directe est limitée, mais leur inclusion améliore la précision globale.
    """)
    st.divider()

    # ── Modèle 3 — Classification k-NN ───────────────────────────────────────
    section_label("Modèle 3 — Classification Supervisée (k-NN)")
    info_badge("CLF", "success",
        "Classifie le <strong>type de course</strong> (Ramassage / Dépôt) selon prix et distance.")

    if n < MIN_CLF:
        st.warning(f"Minimum {MIN_CLF} observations requises (actuellement : {n})."); return

    Xc = df[["prix_fcfa","distance_km"]].values
    yc = df["type_enc"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xc, yc, test_size=0.25, random_state=42, stratify=yc
    )

    k   = max(3, int(np.sqrt(n)))
    k   = k if k % 2 != 0 else k + 1
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    y_pred   = knn.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)

    le_t    = LabelEncoder().fit(df["type_course"])
    classes = le_t.classes_

    c1, c2 = st.columns(2)
    c1.metric("Précision (Accuracy)", f"{accuracy*100:.1f} %")
    c2.metric("Voisins k utilisés",   str(k))

    report = classification_report(y_te, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    st.markdown("**Rapport de Classification :**")
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)

    df_plot             = df.copy()
    df_plot["Prédiction"] = le_t.inverse_transform(knn.predict(Xc))
    fig_cls = px.scatter(
        df_plot, x="distance_km", y="prix_fcfa",
        color="Prédiction", symbol="type_course",
        title=f"Classification k-NN (k={k}) — Prix vs Distance",
        labels={"distance_km":"Distance (km)","prix_fcfa":"Prix (FCFA)","type_course":"Classe réelle"},
        color_discrete_map={"Ramassage":"#38bdf8","Dépôt":"#34d399"},
    )
    fig_cls.update_layout(**PLOTLY_DARK)
    st.plotly_chart(fig_cls, use_container_width=True)

    # Prédicteur interactif
    st.markdown("#### Tester le Modèle en Temps Réel")
    s1, s2 = st.columns(2)
    with s1: p_dist = st.slider("Distance (km)", 0.5, 50.0, 5.0, 0.5)
    with s2: p_prix = st.slider("Prix (FCFA)",   100, 20000, 1000, 50)

    pred  = le_t.inverse_transform(knn.predict([[p_prix, p_dist]]))[0]
    proba = knn.predict_proba([[p_prix, p_dist]])[0].max()
    bv    = "badge-info" if pred == "Ramassage" else "badge-success"

    st.markdown(f"""
        <div class="dash-card" style="text-align:center;margin-top:0.4rem;">
            <p style="color:var(--text-muted);font-size:0.71rem;margin:0 0 0.5rem;">
                Résultat de la prédiction</p>
            <span class="badge {bv}" style="font-size:0.95rem;padding:5px 22px;">{pred}</span>
            <p style="color:var(--text-muted);font-size:0.71rem;margin:0.5rem 0 0;">
                Confiance :
                <strong style="color:var(--accent);">{proba*100:.1f} %</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  10. APP SHELL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:0.5rem;padding:0 0.1rem;">
    <div>
        <h1 style="margin:0;font-size:1.45rem;line-height:1.2;">
            TaxiData <span style="color:var(--accent)">YDE</span>
        </h1>
        <p style="color:var(--text-muted);margin:0.12rem 0 0;font-size:0.73rem;letter-spacing:0.02em;">
            INF 232 — Analyse de Données · Université de Yaoundé I
        </p>
    </div>
    <span class="badge badge-info">v2.1</span>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Collecte", "Dashboard", "Modèles IA"],
    icons=["pencil-square", "bar-chart-line", "cpu"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "5px 8px",
            "background-color": "#181d2a",
            "border": "1px solid rgba(99,179,237,0.10)",
            "border-radius": "16px",
            "margin-bottom": "1.4rem",
        },
        "icon": {"color": "#64748b", "font-size": "13px"},
        "nav-link": {
            "font-family": "Plus Jakarta Sans, sans-serif",
            "font-size": "0.81rem",
            "font-weight": "600",
            "color": "#64748b",
            "border-radius": "9px",
            "--hover-color": "rgba(56,189,248,0.09)",
        },
        "nav-link-selected": {
            "background-color": "#38bdf8",
            "color": "#0d1117",
            "font-weight": "700",
            "box-shadow": "0 2px 10px rgba(56,189,248,0.25)",
        },
    },
)

# ─────────────────────────────────────────────────────────────────────────────
#  11. ROUTING
# ─────────────────────────────────────────────────────────────────────────────
if   selected == "Collecte":   render_collecte_tab()
elif selected == "Dashboard":  render_dashboard_tab()
elif selected == "Modèles IA": render_models_tab()
