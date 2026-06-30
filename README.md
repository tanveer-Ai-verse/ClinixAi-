# 🏥 ClinixAi — Autonomous Multimodal Medical Case Intelligence System

Transform a clinical note, voice recording, medical image, and lab results into a structured AI-generated differential diagnosis — complete with probability rankings, red-flag detection, recommended workup, and a downloadable PDF report.

**Pipeline:** Audio → Whisper ASR → Biomedical NER (Groq LLM) → Vision Analysis (Groq Vision) → Differential Diagnosis Synthesis → PDF Report

> ⚠️ **For research and educational use only. Not a substitute for clinical judgment or a licensed medical professional.**

---

## 🚀 Features

- **🎙️ Clinical Note Intake** — type a note directly, or upload an audio recording (MP3/WAV/OGG/M4A) transcribed automatically via OpenAI Whisper
- **🔬 Biomedical NER** — extracts symptoms, medications, anatomical sites, vital signs, lab values, and patient history via Groq LLM
- **🖼️ Medical Imaging** — upload an X-ray, CT slice, MRI, or fundus image for AI radiologist-style findings via Groq's vision model
- **📊 Lab Results** — manual entry with an editable data table, or upload a CSV; auto-computed high/low abnormality counts
- **⚙️ Analysis Settings** — configurable specialty context, patient risk profile, max differential count, and workup inclusion
- **🧠 Differential Diagnosis Engine** — ranked diagnoses with probability, confidence level, supporting/against evidence, and recommended workup
- **🚨 Red Flag & Immediate Action Detection** — surfaces urgent findings and next steps prominently
- **📈 Visual Analytics** — probability bar chart and entity-coverage radar chart (Plotly)
- **📄 PDF Report Export** — full, formatted clinical report ready to download

---

## 🏗️ Project Structure

```
clinixai/
├── app.py             ← Main Streamlit application
├── requirements.txt   ← Python dependencies (PyPI only)
├── packages.txt        ← System-level dependency (ffmpeg, for Whisper)
└── README.md           ← This file
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/clinixai.git
cd clinixai
pip install -r requirements.txt
```

Whisper also requires the **ffmpeg** command-line tool on your system:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

Create `.streamlit/secrets.toml` in your project root:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com). Then run:

```bash
streamlit run app.py
```

> ⚠️ Add `.streamlit/secrets.toml` to your `.gitignore` — never commit API keys to GitHub.

---

## ☁️ Deploying to Streamlit Cloud

1. Push `app.py`, `requirements.txt`, `packages.txt`, and `README.md` to a GitHub repository (do **not** include `secrets.toml`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → set main file to `app.py` → **Deploy**
3. **Set your API key:** open your app's **⋮ menu → Settings → Secrets**, paste `GROQ_API_KEY = "gsk_your_key_here"`, then click **Save** — the app restarts automatically with AI features enabled.

`packages.txt` tells Streamlit Cloud to install **ffmpeg** at the system level automatically — Whisper needs it to decode audio, and it cannot be installed via `requirements.txt` since it isn't a Python package.

---

## 🛡️ Deployment Notes

- No hardcoded API keys anywhere in the source — the Groq client is loaded only via `st.secrets["GROQ_API_KEY"]`, cached with `@st.cache_resource`, with a clear on-screen warning if the key is missing rather than a crash.
- All `pyngrok` / `subprocess.Popen` tunnel code from the original notebook has been removed — Streamlit Cloud serves the app directly, no tunnel is needed.
- The Whisper model is loaded once via `@st.cache_resource` so the interface stays responsive across reruns instead of reloading the model on every interaction.
- `requirements.txt` contains only standard PyPI package names — no direct binary/wheel URLs.
- Every AI call, audio transcription, image load, CSV parse, and PDF build is wrapped in `try/except`, so a single failure (bad file, API hiccup, malformed AI response) shows a friendly message instead of crashing the app.
- Two Groq models are used intentionally: `llama-3.3-70b-versatile` for all text-based reasoning (NER, differential diagnosis) and `llama-3.2-11b-vision-preview` for the one image-analysis call, since the 70B text model cannot process images — preserving the original imaging feature exactly as it worked before.

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI Backend | Groq · LLaMA-3.3-70B-Versatile (text) + Llama-3.2-11B-Vision (imaging) |
| Speech-to-Text | OpenAI Whisper (`base` model) |
| Visualization | Plotly |
| PDF Generation | fpdf2 |
| Data Handling | Pandas · NumPy |
| Image Handling | Pillow |

---

<div align="center"><strong>🏥 ClinixAi · Autonomous Multimodal Medical Case Intelligence System</strong></div>
