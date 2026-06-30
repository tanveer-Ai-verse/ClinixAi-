"""
🏥 ClinixAi — Autonomous Multimodal Medical Case Intelligence System
=========================================================================
Pipeline: Audio → Whisper ASR → Biomedical NER (Groq LLM) →
          Vision Analysis (Groq Vision) → Differential Diagnosis →
          PDF Report Generation

Backend : Groq LLaMA-3.3-70B-Versatile (text/diagnosis) +
          Groq Llama-3.2-11B-Vision (medical imaging)
Stack   : Streamlit · Whisper · Plotly · FPDF2 · Pandas
"""

import streamlit as st
import os, json, time, io, base64, textwrap, re
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from fpdf import FPDF

# ── Optional / heavy dependencies loaded defensively ──────────────────────────
try:
    from groq import Groq
    GROQ_LIB_OK = True
except ImportError:
    GROQ_LIB_OK = False

try:
    import whisper
    WHISPER_LIB_OK = True
except ImportError:
    WHISPER_LIB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='ClinixAi — Medical AI',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─────────────────────────────────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background: linear-gradient(135deg, #0f1117 0%, #141920 50%, #0d1117 100%); }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
  }

  /* Main header */
  .hero-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(154,103,255,0.08) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(99,179,237,0.04) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(154,103,255,0.04) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #9a67ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem;
  }
  .hero-sub { color: #8892a4; font-size: 0.95rem; margin: 0; }

  /* Cards */
  .metric-card {
    background: rgba(17,24,39,0.8);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 700; color: #63b3ed; margin: 0; }
  .metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.25rem 0 0; }

  .section-card {
    background: rgba(17,24,39,0.7);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
  }
  .section-title {
    font-size: 0.78rem; font-weight: 600;
    color: #63b3ed;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 0 0 1rem;
  }

  /* Diagnosis pills */
  .diag-pill {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #93c5fd;
    margin: 3px;
    font-weight: 500;
  }
  .diag-pill.high { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.3); color: #fca5a5; }
  .diag-pill.med  { background: rgba(251,191,36,0.1); border-color: rgba(251,191,36,0.3); color: #fcd34d; }
  .diag-pill.low  { background: rgba(34,197,94,0.1);  border-color: rgba(34,197,94,0.3);  color: #86efac; }

  /* Entity badges */
  .ent-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
  }
  .ent-sym  { background: rgba(167,139,250,0.15); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.3); }
  .ent-med  { background: rgba(52,211,153,0.12);  color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
  .ent-anat { background: rgba(251,146,60,0.12);  color: #fdba74; border: 1px solid rgba(251,146,60,0.3); }
  .ent-val  { background: rgba(99,179,237,0.12);  color: #93c5fd; border: 1px solid rgba(99,179,237,0.3); }

  /* Step indicator */
  .step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #63b3ed, #9a67ff);
    border-radius: 50%;
    font-size: 0.75rem; font-weight: 700; color: white;
    margin-right: 10px; flex-shrink: 0;
  }

  /* Streamlit overrides */
  .stButton>button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    width: 100%;
  }
  .stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
  }
  .stTextArea textarea {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stFileUploader {
    background: rgba(17,24,39,0.5);
    border: 1px dashed rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 1rem;
  }
  .stProgress > div > div { background: linear-gradient(90deg, #63b3ed, #9a67ff) !important; }
  div[data-testid="stMarkdownContainer"] p { color: #d1d5db; }
  h1, h2, h3 { color: #f3f4f6 !important; }
  label, .stSelectbox label, .stSlider label { color: #9ca3af !important; font-size: 0.85rem !important; }
  .stSelectbox > div > div {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
  }
  .stAlert { border-radius: 10px !important; }
  [data-testid="stExpander"] {
    background: rgba(17,24,39,0.6);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
  }
  .stTab [data-baseweb="tab"] { color: #9ca3af !important; }
  .stTab [aria-selected="true"] { color: #63b3ed !important; border-bottom-color: #63b3ed !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Groq client — secure initialization via st.secrets (cached, never hardcoded)
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODEL = 'llama-3.3-70b-versatile'           # primary text / diagnosis model
GROQ_VISION_MODEL = 'llama-3.2-11b-vision-preview'  # medical imaging model


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Initialize the Groq client from Streamlit secrets. Returns None on failure."""
    if not GROQ_LIB_OK:
        return None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except (KeyError, FileNotFoundError):
        return None
    except Exception:
        return None


client = get_groq_client()
GROQ_READY = client is not None

if not GROQ_READY:
    st.error(
        "⚠️ **GROQ_API_KEY is not configured.** ClinixAi's AI features "
        "(entity extraction, differential diagnosis, imaging analysis) require "
        "a valid Groq API key.\n\n"
        "Add it under **Settings → Secrets** in Streamlit Cloud, or in a local "
        "`.streamlit/secrets.toml` file. See the README for instructions."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: extract structured entities from clinical text (Biomedical NER)
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities(text: str) -> dict:
    """Extract symptoms, medications, anatomy, vitals, labs, and history via Groq LLM NER."""
    empty = {'symptoms': [], 'medications': [], 'anatomical_sites': [],
              'vital_signs': [], 'lab_values': [], 'patient_history': []}
    if not GROQ_READY:
        return empty
    prompt = f"""
You are a clinical NLP expert. Extract medical entities from the clinical note below.
Return ONLY valid JSON with these keys:
  symptoms (list), medications (list), anatomical_sites (list),
  vital_signs (list), lab_values (list), patient_history (list)

Clinical note:
{text}
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1, max_tokens=800
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except Exception:
            return empty
    except Exception as e:
        st.warning(f"⚠️ Entity extraction failed: {e}. Continuing with no entities detected.")
        return empty

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: generate differential diagnosis
# ─────────────────────────────────────────────────────────────────────────────
def generate_differential(entities: dict, lab_df, image_desc: str, note: str) -> dict:
    """Generate a structured differential diagnosis report via Groq LLM."""
    if not GROQ_READY:
        return {'error': 'AI unavailable',
                'raw': 'GROQ_API_KEY is not configured. Please add it in Streamlit Secrets.'}
    try:
        lab_text = lab_df.to_string(index=False) if lab_df is not None else 'No lab data provided.'
    except Exception:
        lab_text = 'No lab data provided.'
    prompt = f"""
You are a senior attending physician generating a structured differential diagnosis.
Return ONLY valid JSON with this exact structure:
{{
  "patient_summary": "2-3 sentence clinical summary",
  "differential_diagnoses": [
    {{"rank": 1, "diagnosis": "name", "probability": 85, "confidence": "high",
      "supporting_evidence": ["evidence1", "evidence2"],
      "against_evidence": ["against1"],
      "recommended_workup": ["test1", "test2"]}}
  ],
  "immediate_actions": ["action1", "action2"],
  "red_flags": ["flag1"],
  "overall_assessment": "paragraph",
  "safety_netting": "paragraph"
}}

Clinical Note: {note}
Entities: {json.dumps(entities)}
Lab Results:\n{lab_text}
Imaging: {image_desc}
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.15, max_tokens=2000
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except Exception:
            return {'error': 'Parse error', 'raw': raw}
    except Exception as e:
        return {'error': 'API error', 'raw': f'Groq API call failed: {e}'}

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: describe image via Groq vision
# ─────────────────────────────────────────────────────────────────────────────
def describe_image(image_bytes: bytes, mime: str = 'image/jpeg') -> str:
    """Analyse a medical image (X-ray, CT, MRI, fundus) via Groq's vision model."""
    if not GROQ_READY:
        return "⚠️ Imaging analysis unavailable — GROQ_API_KEY is not configured."
    try:
        b64 = base64.b64encode(image_bytes).decode()
        resp = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url',
                     'image_url': {'url': f'data:{mime};base64,{b64}'}},
                    {'type': 'text',
                     'text': 'You are a radiologist. Describe key findings in this medical image relevant to diagnosis. Be specific about abnormalities, distributions, and severity.'}
                ]
            }],
            temperature=0.1, max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Imaging analysis failed: {e}. Proceeding without imaging findings."

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: transcribe audio with Whisper (cached resource for responsiveness)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Whisper ASR model…")
def load_whisper():
    """Load and cache the Whisper 'base' model so it is only loaded once per session."""
    if not WHISPER_LIB_OK:
        return None
    return whisper.load_model('base')


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text using the cached Whisper model."""
    try:
        model = load_whisper()
        if model is None:
            return "⚠️ Whisper is not available in this environment. Please type the clinical note instead."
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        st.error(f"⚠️ Audio transcription failed: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: generate PDF report
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(report: dict, note: str) -> bytes:
    """Build a downloadable PDF clinical report from the diagnosis JSON."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(13, 17, 23)
        pdf.rect(0, 0, 210, 297, 'F')

        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(99, 179, 237)
        pdf.cell(0, 15, 'ClinixAi Medical Report', ln=True, align='C')

        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  Model: {GROQ_MODEL}', ln=True, align='C')
        pdf.ln(4)

        def section(title):
            pdf.set_fill_color(30, 41, 59)
            pdf.set_text_color(99, 179, 237)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 8, title, ln=True, fill=True)
            pdf.ln(1)

        def body(txt):
            pdf.set_text_color(209, 213, 219)
            pdf.set_font('Helvetica', '', 9)
            safe = (txt or '').encode('latin-1', errors='replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe)
            pdf.ln(2)

        section('PATIENT SUMMARY')
        body(report.get('patient_summary', ''))

        section('DIFFERENTIAL DIAGNOSES')
        for d in report.get('differential_diagnoses', []):
            pdf.set_text_color(248, 250, 252)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, f"{d['rank']}. {d['diagnosis']}  -  {d['probability']}% ({d['confidence']} confidence)", ln=True)
            body('  Supporting: ' + ', '.join(d.get('supporting_evidence', [])))
            body('  Workup: ' + ', '.join(d.get('recommended_workup', [])))

        section('IMMEDIATE ACTIONS')
        body('\n'.join(f'• {a}' for a in report.get('immediate_actions', [])))

        section('RED FLAGS')
        body('\n'.join(f'⚠ {f}' for f in report.get('red_flags', [])))

        section('OVERALL ASSESSMENT')
        body(report.get('overall_assessment', ''))

        section('SAFETY NETTING')
        body(report.get('safety_netting', ''))

        section('ORIGINAL CLINICAL NOTE')
        body(note)

        return bytes(pdf.output())
    except Exception as e:
        st.error(f"⚠️ PDF generation failed: {e}")
        # Return a minimal valid PDF so the download button doesn't break
        fallback = FPDF()
        fallback.add_page()
        fallback.set_font('Helvetica', '', 12)
        fallback.cell(0, 10, 'ClinixAi Report — PDF generation encountered an error.', ln=True)
        return bytes(fallback.output())

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: probability gauge chart
# ─────────────────────────────────────────────────────────────────────────────
def make_probability_chart(diagnoses: list) -> go.Figure:
    names = [d['diagnosis'] for d in diagnoses]
    probs = [d['probability'] for d in diagnoses]
    colors = ['#ef4444' if p >= 70 else '#f59e0b' if p >= 40 else '#22c55e' for p in probs]

    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation='h',
        marker_color=colors,
        text=[f'{p}%' for p in probs],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Probability: %{x}%<extra></extra>'
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#9ca3af'),
        xaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.05)',
                   ticksuffix='%', color='#6b7280'),
        yaxis=dict(autorange='reversed', color='#d1d5db'),
        margin=dict(l=0, r=10, t=10, b=10),
        height=max(200, len(diagnoses) * 50)
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: radar chart for entity counts
# ─────────────────────────────────────────────────────────────────────────────
def make_entity_radar(entities: dict) -> go.Figure:
    cats = ['Symptoms', 'Medications', 'Anatomy', 'Vitals', 'Labs', 'History']
    vals = [
        len(entities.get('symptoms', [])),
        len(entities.get('medications', [])),
        len(entities.get('anatomical_sites', [])),
        len(entities.get('vital_signs', [])),
        len(entities.get('lab_values', [])),
        len(entities.get('patient_history', []))
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill='toself',
        fillcolor='rgba(99,179,237,0.15)',
        line=dict(color='#63b3ed', width=2),
        marker=dict(color='#63b3ed', size=6)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, max(vals + [1])],
                            gridcolor='rgba(255,255,255,0.07)', color='#6b7280'),
            angularaxis=dict(color='#9ca3af')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#9ca3af'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center; padding: 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:2.5rem;">🏥</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.1rem; font-weight:700; color:#63b3ed;">ClinixAi</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; color:#6b7280;">Medical AI Intelligence</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,179,237,0.15); margin: 1rem 0;">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem;">Pipeline Stages</div>', unsafe_allow_html=True)

    steps = [
        ('🎙️', 'ASR', 'Whisper'),
        ('🔬', 'NER', 'Groq LLM'),
        ('🖼️', 'Vision', 'Llama Vision'),
        ('📊', 'Tabular', 'Statistical'),
        ('🧠', 'Synthesis', 'Groq 70B'),
        ('📄', 'Report', 'PDF Export'),
    ]
    for icon, name, model in steps:
        st.markdown(f'''
        <div style="display:flex; align-items:center; gap:10px;
                    background:rgba(17,24,39,0.5); border:1px solid rgba(99,179,237,0.1);
                    border-radius:8px; padding:8px 12px; margin-bottom:6px;">
            <span style="font-size:1.1rem;">{icon}</span>
            <div>
                <div style="font-size:0.82rem; font-weight:600; color:#e5e7eb;">{name}</div>
                <div style="font-size:0.7rem; color:#6b7280;">{model}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,179,237,0.15); margin: 1rem 0;">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.72rem; color:#6b7280;">Model: <span style="color:#63b3ed;">{GROQ_MODEL}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.72rem; color:#6b7280; margin-top:4px;">Vision: <span style="color:#63b3ed;">{GROQ_VISION_MODEL}</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; color:#6b7280; margin-top:4px;">ASR: <span style="color:#63b3ed;">Whisper base</span></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(99,179,237,0.15); margin: 1rem 0;">', unsafe_allow_html=True)
    if not GROQ_READY:
        st.markdown('<div style="font-size:0.72rem; color:#ef4444; text-align:center;">🔴 AI backend offline — configure GROQ_API_KEY in Secrets.</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#374151; text-align:center;">⚠️ For research purposes only. Not a substitute for clinical judgment.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="hero-header">
  <div class="hero-title">Autonomous Medical Case Intelligence</div>
  <div class="hero-sub">Multimodal AI pipeline · Audio + Vision + Labs + NLP → Differential Diagnosis</div>
</div>
''', unsafe_allow_html=True)

# ─── Input tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(['🎙️ Clinical Note', '🖼️ Medical Image', '📊 Lab Results', '⚙️ Settings'])

with tab1:
    col_a, col_b = st.columns([1, 1], gap='medium')
    with col_a:
        st.markdown('<div class="section-card"><div class="section-title">Audio Recording</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader('Upload doctor\'s voice note', type=['mp3', 'wav', 'ogg', 'm4a'], key='audio')
        if audio_file:
            st.audio(audio_file)
            st.info('Audio loaded — will be transcribed via Whisper on analysis.')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card"><div class="section-title">Or Type Clinical Note</div>', unsafe_allow_html=True)
        clinical_note = st.text_area(
            'Clinical note',
            height=180,
            placeholder='Patient is a 58-year-old male presenting with 3 days of productive cough, fever 38.9°C, right lower lobe dullness on percussion. PMH: hypertension, type 2 diabetes. Current meds: metformin 1g BD, amlodipine 5mg. SpO2 94% on air. WBC 14.2, CRP 87...',
            label_visibility='collapsed'
        )
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card"><div class="section-title">Medical Imaging</div>', unsafe_allow_html=True)
    img_file = st.file_uploader('Upload X-ray, CT slice, MRI, or fundus image', type=['jpg', 'jpeg', 'png', 'bmp'], key='img')
    if img_file:
        try:
            img = Image.open(img_file)
            col_i1, col_i2 = st.columns([1, 1])
            with col_i1:
                st.image(img, caption='Uploaded scan', use_column_width=True)
            with col_i2:
                st.markdown('<div style="color:#9ca3af; font-size:0.85rem;">Image details</div>', unsafe_allow_html=True)
                st.markdown(f'**Size:** {img.size[0]} × {img.size[1]} px')
                st.markdown(f'**Mode:** {img.mode}')
                st.markdown(f'**Format:** {img_file.type}')
        except Exception as e:
            st.error(f"⚠️ Could not load image: {e}")
    else:
        st.markdown('<div style="text-align:center; padding:2rem; color:#4b5563;">📷 No image uploaded — imaging section will be skipped</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card"><div class="section-title">Laboratory Results</div>', unsafe_allow_html=True)
    lab_mode = st.radio('Input method', ['Manual entry', 'Upload CSV'], horizontal=True)

    lab_df = None
    if lab_mode == 'Manual entry':
        default_labs = pd.DataFrame({
            'Test': ['WBC', 'Haemoglobin', 'Platelets', 'CRP', 'Creatinine', 'Sodium', 'Potassium', 'SpO2'],
            'Value': [14.2, 12.8, 210, 87.0, 98.0, 138.0, 4.1, 94.0],
            'Unit': ['×10⁹/L', 'g/dL', '×10⁹/L', 'mg/L', 'µmol/L', 'mmol/L', 'mmol/L', '%'],
            'Reference Range': ['4–11', '13–17', '150–400', '<5', '60–110', '135–145', '3.5–5.0', '95–100'],
            'Status': ['HIGH', 'LOW', 'Normal', 'HIGH', 'Normal', 'Normal', 'Normal', 'LOW']
        })
        edited = st.data_editor(default_labs, use_container_width=True, num_rows='dynamic')
        lab_df = edited
    else:
        csv_file = st.file_uploader('Upload CSV with Test, Value, Unit columns', type=['csv'])
        if csv_file:
            try:
                lab_df = pd.read_csv(csv_file)
                st.dataframe(lab_df, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Could not parse CSV file: {e}")
                lab_df = None

    if lab_df is not None:
        try:
            abnormal = lab_df[lab_df.get('Status', pd.Series()) == 'HIGH'].shape[0] if 'Status' in lab_df.columns else 0
            low = lab_df[lab_df.get('Status', pd.Series()) == 'LOW'].shape[0] if 'Status' in lab_df.columns else 0
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(lab_df)}</div><div class="metric-label">Tests</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ef4444;">{abnormal}</div><div class="metric-label">High</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#f59e0b;">{low}</div><div class="metric-label">Low</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute lab summary metrics: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-card"><div class="section-title">Analysis Settings</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_diagnoses = st.slider('Max differential diagnoses', 3, 8, 5)
        include_workup = st.toggle('Include recommended workup', value=True)
    with col_s2:
        specialty = st.selectbox('Clinical specialty context',
            ['General Medicine', 'Emergency Medicine', 'Cardiology',
             'Pulmonology', 'Neurology', 'Gastroenterology', 'Infectious Disease'])
        risk_level = st.selectbox('Patient risk profile', ['Low', 'Moderate', 'High', 'Critical'])
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Analyse button ───────────────────────────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_analysis = st.button('🧠 Run ClinixAi Analysis', use_container_width=True)

# ─── Analysis pipeline ────────────────────────────────────────────────────────
if run_analysis:
    note_text = clinical_note.strip()

    # Step 1: Audio transcription
    if audio_file and not note_text:
        try:
            with st.status('🎙️ Transcribing audio with Whisper...', expanded=True) as status:
                audio_bytes = audio_file.read()
                tmp_path = '/tmp/clinixai_audio_input.' + audio_file.name.split('.')[-1]
                with open(tmp_path, 'wb') as f:
                    f.write(audio_bytes)
                note_text = transcribe_audio(tmp_path)
                status.update(label='✅ Transcription complete', state='complete')
            if note_text:
                st.markdown(f'<div class="section-card"><div class="section-title">Transcription</div><p style="color:#d1d5db; font-size:0.9rem;">{note_text}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Audio processing error: {e}")
            note_text = ""

    if not note_text:
        st.error('Please provide a clinical note (typed or via audio upload).')
        st.stop()

    progress = st.progress(0, text='Starting analysis...')

    # Step 2: Entity extraction
    progress.progress(15, text='🔬 Extracting clinical entities...')
    entities = extract_entities(note_text)

    # Step 3: Image analysis
    image_desc = 'No imaging provided.'
    if img_file:
        progress.progress(35, text='🖼️ Analysing medical image with vision AI...')
        try:
            img_file.seek(0)
            image_desc = describe_image(img_file.read(), img_file.type)
        except Exception as e:
            image_desc = f"⚠️ Imaging analysis failed: {e}"

    # Step 4: Differential diagnosis
    progress.progress(60, text='🧠 Generating differential diagnosis...')
    report = generate_differential(entities, lab_df, image_desc, note_text)

    progress.progress(85, text='📄 Building report...')
    pdf_bytes = generate_pdf(report, note_text)

    progress.progress(100, text='✅ Analysis complete!')
    time.sleep(0.4)
    progress.empty()

    # ─── Results ─────────────────────────────────────────────────────────────
    if 'error' in report:
        st.error(f'⚠️ {report.get("error", "Analysis error")}: {report.get("raw", "")}')
        st.stop()

    st.markdown('<hr style="border-color:rgba(99,179,237,0.15); margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.4rem; font-weight:700; color:#f3f4f6; margin-bottom:1.5rem;">📋 Analysis Results</div>', unsafe_allow_html=True)

    # Patient summary
    st.markdown(f'''
    <div class="section-card">
        <div class="section-title">Patient Summary</div>
        <p style="color:#d1d5db; font-size:0.95rem; line-height:1.7; margin:0;">{report.get("patient_summary", "")}</p>
    </div>
    ''', unsafe_allow_html=True)

    # Metrics row
    diags = report.get('differential_diagnoses', [])
    red_flags = report.get('red_flags', [])
    actions = report.get('immediate_actions', [])
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(diags)}</div><div class="metric-label">Differentials</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-val">{diags[0]["probability"] if diags else 0}%</div><div class="metric-label">Top Diagnosis</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ef4444;">{len(red_flags)}</div><div class="metric-label">Red Flags</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(actions)}</div><div class="metric-label">Immediate Actions</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Two-column layout: chart + entities
    col_r1, col_r2 = st.columns([3, 2], gap='medium')

    with col_r1:
        st.markdown('<div class="section-card"><div class="section-title">Differential Probabilities</div>', unsafe_allow_html=True)
        if diags:
            try:
                st.plotly_chart(make_probability_chart(diags), use_container_width=True, config={'displayModeBar': False})
            except Exception as e:
                st.warning(f"⚠️ Could not render probability chart: {e}")
        else:
            st.info("No differential diagnoses were returned.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="section-card"><div class="section-title">Entity Coverage</div>', unsafe_allow_html=True)
        try:
            st.plotly_chart(make_entity_radar(entities), use_container_width=True, config={'displayModeBar': False})
        except Exception as e:
            st.warning(f"⚠️ Could not render entity radar: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Diagnosis detail cards
    st.markdown('<div class="section-card"><div class="section-title">Detailed Differentials</div>', unsafe_allow_html=True)
    for d in diags:
        conf = d.get('confidence', 'low')
        pill_cls = 'high' if conf == 'high' else 'med' if conf == 'moderate' else 'low'
        evidence_html = ''.join(f'<div style="font-size:0.8rem; color:#9ca3af; padding:2px 0;">✓ {e}</div>' for e in d.get('supporting_evidence', []))
        workup_html = ''.join(f'<span class="diag-pill">{w}</span>' for w in d.get('recommended_workup', []))
        against_html = ''.join(f'<div style="font-size:0.8rem; color:#6b7280; padding:2px 0;">✗ {e}</div>' for e in d.get('against_evidence', []))
        st.markdown(f'''
        <div style="background:rgba(17,24,39,0.5); border:1px solid rgba(99,179,237,0.12);
                    border-radius:10px; padding:1rem; margin-bottom:0.75rem;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.2rem; font-weight:700; color:#63b3ed;">{d["rank"]}.</span>
                    <span style="font-size:1rem; font-weight:600; color:#f3f4f6;">{d["diagnosis"]}</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="diag-pill {pill_cls}">{conf} confidence</span>
                    <span style="font-size:1.1rem; font-weight:700; color:#63b3ed;">{d["probability"]}%</span>
                </div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
                <div><div style="font-size:0.72rem; color:#63b3ed; margin-bottom:4px;">SUPPORTING</div>{evidence_html}</div>
                <div><div style="font-size:0.72rem; color:#ef4444; margin-bottom:4px;">AGAINST</div>{against_html}</div>
            </div>
            <div style="margin-top:0.75rem;"><div style="font-size:0.72rem; color:#9ca3af; margin-bottom:6px;">RECOMMENDED WORKUP</div>{workup_html}</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Entity tags
    st.markdown('<div class="section-card"><div class="section-title">Extracted Entities</div>', unsafe_allow_html=True)
    ent_cols = st.columns(3)
    ent_map = [
        ('Symptoms', entities.get('symptoms', []), 'ent-sym', ent_cols[0]),
        ('Medications', entities.get('medications', []), 'ent-med', ent_cols[1]),
        ('Anatomy', entities.get('anatomical_sites', []), 'ent-anat', ent_cols[2]),
    ]
    for label, items, cls, col in ent_map:
        with col:
            tags = ''.join(f'<span class="ent-badge {cls}">{i}</span>' for i in items) or '<span style="color:#4b5563; font-size:0.8rem;">None detected</span>'
            st.markdown(f'<div style="margin-bottom:0.5rem; font-size:0.72rem; color:#6b7280;">{label}</div>{tags}', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Red flags & actions
    col_rf, col_ac = st.columns(2, gap='medium')
    with col_rf:
        st.markdown('<div class="section-card"><div class="section-title">🚨 Red Flags</div>', unsafe_allow_html=True)
        for f in red_flags:
            st.markdown(f'<div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:8px; padding:8px 12px; margin-bottom:6px; font-size:0.85rem; color:#fca5a5;">⚠️ {f}</div>', unsafe_allow_html=True)
        if not red_flags:
            st.markdown('<div style="color:#22c55e; font-size:0.85rem;">✅ No immediate red flags identified</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ac:
        st.markdown('<div class="section-card"><div class="section-title">⚡ Immediate Actions</div>', unsafe_allow_html=True)
        for i, a in enumerate(actions, 1):
            st.markdown(f'<div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:8px;"><span style="background:linear-gradient(135deg,#2563eb,#7c3aed); color:white; border-radius:50%; width:20px; height:20px; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:700; flex-shrink:0;">{i}</span><span style="font-size:0.85rem; color:#d1d5db; line-height:1.5;">{a}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Overall assessment
    st.markdown(f'''
    <div class="section-card">
        <div class="section-title">Overall Assessment</div>
        <p style="color:#d1d5db; font-size:0.92rem; line-height:1.75; margin:0;">{report.get("overall_assessment", "")}</p>
    </div>
    ''', unsafe_allow_html=True)

    # Imaging findings
    if img_file:
        st.markdown(f'''
        <div class="section-card">
            <div class="section-title">🖼️ Imaging Analysis (AI Radiologist)</div>
            <p style="color:#d1d5db; font-size:0.9rem; line-height:1.7; margin:0;">{image_desc}</p>
        </div>
        ''', unsafe_allow_html=True)

    # Safety netting
    st.markdown(f'''
    <div style="background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.2);
                border-radius:12px; padding:1.25rem; margin-bottom:1.5rem;">
        <div style="font-size:0.78rem; font-weight:600; color:#fcd34d;
                    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Safety Netting</div>
        <p style="color:#fde68a; font-size:0.88rem; line-height:1.65; margin:0;">{report.get("safety_netting", "")}</p>
    </div>
    ''', unsafe_allow_html=True)

    # PDF download
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label='📄 Download Full PDF Report',
            data=pdf_bytes,
            file_name=f'ClinixAi_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf',
            mime='application/pdf',
            use_container_width=True
        )

    st.markdown('''
    <div style="text-align:center; padding:1.5rem 0; color:#374151; font-size:0.75rem;">
        ClinixAi · Autonomous Multimodal Medical Case Intelligence System ·
        Powered by Groq · For research & educational use only
    </div>
    ''', unsafe_allow_html=True)
