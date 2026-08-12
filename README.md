# 🛡️ MindGuard AI

**Mental Health Awareness & Suicide Prevention Agent**  
Powered by **IBM watsonx.ai** and **IBM Granite Models**

---

## Overview

MindGuard AI is a multi-agent Flask web application designed to provide empathetic, AI-driven mental health support. It combines five specialised agents orchestrated by a central pipeline, backed by IBM Granite large language models running on IBM watsonx.ai.

> ⚠️ **Disclaimer:** MindGuard AI provides educational and emotional support only. It is **not** a substitute for professional medical or psychological care. In a crisis, please contact a qualified professional or a crisis helpline immediately.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **Mental Health Awareness Agent** | Education on anxiety, depression, stress & mindfulness |
| 💚 **Emotional Support Agent** | Empathetic support & evidence-based coping strategies |
| 🔍 **Distress Detection Agent** | Risk classification & early-warning signal analysis |
| 🌿 **Prevention & Wellness Agent** | Personalised wellness & self-care plan generation |
| 🤝 **Human Support Connector** | Professional resources & crisis helplines by region |
| 📄 **RAG Document Ingestion** | Upload PDF / TXT files to ground AI responses |
| 📊 **Live Risk Meter** | Real-time distress score with visual indicator |
| 🤖 **IBM Granite Integration** | `ibm/granite-4-h-small` via watsonx.ai SDK |

---

## Tech Stack

- **Backend:** Python 3 · Flask
- **AI / LLM:** IBM watsonx.ai · IBM Granite Models (`ibm/granite-4-h-small`)
- **RAG:** Custom vector store with cosine-similarity retrieval (NumPy)
- **PDF Parsing:** PyPDF2
- **Frontend:** Bootstrap 5 · Bootstrap Icons · Vanilla JS
- **Config:** python-dotenv

---

## Project Structure

```
mindguard-ai/
├── app.py                 # Main Flask app — agents, RAG, routes
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Single-page frontend (Jinja2)
├── static/
│   └── style.css          # External stylesheet
├── .env.example           # Environment variable template
└── .gitignore
```

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone <repo-url>
cd mindguard-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your IBM credentials:

```bash
cp .env.example .env
```

```dotenv
WATSONX_API_KEY=your_ibm_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> Get your credentials from [IBM watsonx.ai](https://dataplatform.cloud.ibm.com/).

### 4. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WATSONX_API_KEY` | *(required)* | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | *(required)* | watsonx.ai project ID |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | Regional endpoint |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the SPA |
| `POST` | `/api/chat` | Main chat — runs full agent pipeline |
| `POST` | `/api/wellness` | Generate a personalised wellness plan |
| `POST` | `/api/upload` | Ingest a PDF or TXT file into the RAG store |
| `GET` | `/api/resources` | Fetch crisis resources list |
| `GET` | `/api/status` | Health-check — model & config status |

---

## Agent Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────┐
│     Orchestrator Agent      │ ← routes to relevant sub-agents
└────────────┬────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
Awareness          Emotional
Agent              Support Agent
    │                 │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
Distress          Wellness
Detection         Agent
Agent                │
    │                ▼
    └──► Support Connector Agent
              │
              ▼
         Final Response
```

---

## Crisis Resources

MindGuard AI surfaces crisis helplines for multiple regions including India, USA, UK, Canada, and Australia, accessible at any time via the Resources panel in the UI.

---

## License

This project is intended for educational and demonstration purposes. See `LICENSE` for details.
