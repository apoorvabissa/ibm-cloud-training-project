# =============================================================================
# MindGuard AI – Mental Health Awareness & Suicide Prevention Agent
# =============================================================================
# Built with:
#   - Python / Flask
#   - IBM watsonx.ai Studio
#   - IBM Granite Models (granite-13b-chat-v2)
#   - Agentic AI Architecture (5 Specialized Agents + Master Orchestrator)
#   - Lightweight RAG System (PDF / TXT document ingestion)
#
# Suitable for:
#   IBM SkillsBuild  |  Hackathons  |  Academic Projects  |  AI Showcases
#
# Author  : MindGuard AI Team
# License : Apache 2.0
# =============================================================================

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import re
import io
import sys
import json
import math
import textwrap
import traceback
from datetime import datetime

# Force UTF-8 output on Windows (cp1252 terminals reject emoji/special chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Third-Party (install via requirements below) ──────────────────────────────
#   pip install flask ibm-watsonx-ai PyPDF2 numpy python-dotenv
from flask import Flask, request, jsonify, render_template_string

# Load .env file automatically if present (python-dotenv)
# Must happen BEFORE os.getenv() calls below
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("[INFO] .env loaded successfully")
except ImportError:
    pass  # python-dotenv not installed — fall back to OS environment variables

# IBM watsonx.ai SDK
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_SDK_AVAILABLE = True
except ImportError:
    WATSONX_SDK_AVAILABLE = False
    print("[WARN] ibm-watsonx-ai SDK not found. AI responses will use fallback mode.")

# PDF text extraction
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARN] PyPDF2 not found. PDF upload will be disabled.")

# Lightweight vector similarity (no external vector DB needed)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[WARN] numpy not found. RAG will use keyword-based retrieval.")

# =============================================================================
# 1. FLASK APPLICATION SETUP
# =============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# =============================================================================
# 2. IBM WATSONX.AI CONFIGURATION
# =============================================================================
# Credentials are read from environment variables for security.
# Set these before running:
#   export WATSONX_API_KEY="your-api-key"
#   export WATSONX_PROJECT_ID="your-project-id"
#   export WATSONX_URL="https://us-south.ml.cloud.ibm.com"

# Read credentials fresh — load_dotenv() has already run above
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM Granite Model ID — the core reasoning engine for all agents
GRANITE_MODEL_ID = "ibm/granite-4-h-small"

# Initialise the watsonx.ai model client (global singleton)
_watsonx_model = None

def _get_watsonx_model():
    """
    Lazily initialise and return the IBM Granite model client.
    All five agents share this single client to avoid redundant connections.
    """
    global _watsonx_model
    if _watsonx_model is not None:
        return _watsonx_model
    if not WATSONX_SDK_AVAILABLE:
        return None
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        print("[WARN] WATSONX_API_KEY or WATSONX_PROJECT_ID not set.")
        return None
    try:
        # ── IBM watsonx.ai Integration Point ────────────────────────────────
        # Uses the modern chat-completions API (/ml/v1/text/chat)
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY,
        )
        _watsonx_model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={"max_tokens": 1024},   # max_tokens is the only param needed for chat API
        )
        print(f"[INFO] Connected to IBM watsonx.ai — model: {GRANITE_MODEL_ID}")
        # ────────────────────────────────────────────────────────────────────
        return _watsonx_model
    except Exception as exc:
        print(f"[ERROR] Failed to initialise watsonx.ai model: {exc}")
        return None


def generate_response(prompt: str, max_tokens: int = 1024) -> str:
    """
    Core IBM watsonx.ai / IBM Granite Models call.
    Every agent routes its prompt through this function.

    Uses the modern chat-completions API (model.chat) which is the current
    recommended approach for IBM Granite on watsonx.ai.
    Falls back to a safe placeholder when SDK is unavailable.
    """
    model = _get_watsonx_model()
    if model is None:
        # Fallback — useful during development without credentials
        return (
            "[Demo Mode — IBM watsonx.ai not connected]\n\n"
            "To enable AI responses, set WATSONX_API_KEY, "
            "WATSONX_PROJECT_ID, and WATSONX_URL environment variables, "
            "then restart the application."
        )
    try:
        # ── IBM Granite Chat API ─────────────────────────────────────────────
        # model.chat() uses /ml/v1/text/chat — the current supported endpoint.
        # The older generate_text() used /ml/v1/text/generation which is deprecated.
        response = model.chat(
            messages=[{"role": "user", "content": prompt}],
            params={"max_tokens": max_tokens},
        )
        # Extract the assistant reply from the chat-completions response dict
        content = response["choices"][0]["message"]["content"]
        # ────────────────────────────────────────────────────────────────────
        return content.strip() if content else "I'm here to help. Could you tell me more?"
    except Exception as exc:
        print(f"[ERROR] generate_response: {exc}")
        return "I encountered an issue processing your request. Please try again."


# =============================================================================
# 3. LIGHTWEIGHT RAG SYSTEM
# =============================================================================
# A simple in-memory RAG that:
#   1. Accepts PDF or TXT uploads
#   2. Splits text into chunks
#   3. Creates TF-IDF-style keyword embeddings (no external vector DB)
#   4. Retrieves the most relevant passages given a query
#   5. Injects context into Granite Model prompts

rag_documents = []   # list of {"title": str, "chunks": [str], "vectors": [dict]}


def _tokenize(text: str) -> list:
    """Lowercase word tokeniser — used for keyword-based retrieval."""
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def _build_vector(tokens: list) -> dict:
    """
    Build a simple term-frequency vector from a token list.
    No external libraries required.
    """
    vec = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    total = sum(vec.values()) or 1
    return {k: v / total for k, v in vec.items()}


def _cosine_similarity(v1: dict, v2: dict) -> float:
    """Compute cosine similarity between two sparse TF vectors."""
    if NUMPY_AVAILABLE:
        keys = list(set(v1) | set(v2))
        a = np.array([v1.get(k, 0) for k in keys], dtype=float)
        b = np.array([v2.get(k, 0) for k in keys], dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    else:
        shared = set(v1) & set(v2)
        dot = sum(v1[k] * v2[k] for k in shared)
        norm_a = math.sqrt(sum(x ** 2 for x in v1.values()))
        norm_b = math.sqrt(sum(x ** 2 for x in v2.values()))
        return dot / (norm_a * norm_b) if norm_a * norm_b else 0.0


def ingest_document(title: str, raw_text: str) -> int:
    """
    Parse, chunk, and index a document for RAG retrieval.
    Returns the number of chunks created.
    """
    # Split into ~300-word chunks with 50-word overlap
    words = raw_text.split()
    chunk_size, overlap = 300, 50
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    vectors = [_build_vector(_tokenize(c)) for c in chunks]
    rag_documents.append({"title": title, "chunks": chunks, "vectors": vectors})
    print(f"[RAG] Ingested '{title}' — {len(chunks)} chunks")
    return len(chunks)


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top-k most relevant passages for a given query.
    Used to augment agent prompts with domain-specific knowledge.
    """
    if not rag_documents:
        return ""

    query_vec = _build_vector(_tokenize(query))
    scored = []
    for doc in rag_documents:
        for i, (chunk, vec) in enumerate(zip(doc["chunks"], doc["vectors"])):
            score = _cosine_similarity(query_vec, vec)
            scored.append((score, doc["title"], chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top or top[0][0] < 0.01:
        return ""

    context_parts = []
    for score, title, chunk in top:
        context_parts.append(f"[Source: {title}]\n{chunk}")

    return "\n\n".join(context_parts)


# =============================================================================
# 4. FIVE SPECIALIZED AI AGENTS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Mental Health Awareness Agent
# ─────────────────────────────────────────────────────────────────────────────
def awareness_agent(user_query: str) -> dict:
    """
    Educates users about mental health topics:
    anxiety, depression, stress, burnout, mindfulness, self-care, and more.
    Powered by IBM Granite Models.
    """
    # Retrieve any relevant RAG context
    rag_context = retrieve_context(user_query)
    context_block = (
        f"\n\nRelevant Resource Context:\n{rag_context}\n"
        if rag_context else ""
    )

    # ── IBM Granite Prompt — Awareness Agent ────────────────────────────────
    prompt = textwrap.dedent(f"""
        You are the MindGuard Mental Health Awareness Agent, powered by IBM Granite AI.
        Your role is to educate users about mental health in a clear, compassionate,
        and evidence-based way.

        Topics you cover: anxiety, depression, stress, burnout, emotional wellness,
        mindfulness, self-care, and healthy coping strategies.
        {context_block}
        User Question: {user_query}

        Provide an informative, empathetic, and easy-to-understand response.
        Use bullet points where helpful. Keep the tone calm and supportive.
        End with one encouraging sentence.
    """).strip()
    # ────────────────────────────────────────────────────────────────────────

    response = generate_response(prompt)
    return {
        "agent": "Mental Health Awareness Agent",
        "icon": "🧠",
        "reason": "Query is educational — asking about a mental health concept or condition.",
        "response": response,
        "rag_used": bool(rag_context),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — Emotional Support Agent
# ─────────────────────────────────────────────────────────────────────────────
def emotional_support_agent(user_query: str) -> dict:
    """
    Provides empathetic, non-judgmental conversational support.
    Acknowledges emotions and encourages healthy coping mechanisms.
    """
    rag_context = retrieve_context(user_query)
    context_block = (
        f"\n\nRelevant Coping Resource:\n{rag_context}\n"
        if rag_context else ""
    )

    # ── IBM Granite Prompt — Emotional Support Agent ─────────────────────────
    prompt = textwrap.dedent(f"""
        You are the MindGuard Emotional Support Agent, an empathetic AI companion
        powered by IBM Granite.
        Your role is to listen actively, validate feelings, and provide compassionate
        support without judgment.

        Guidelines:
        - Acknowledge the user's emotions explicitly
        - Never minimise or dismiss feelings
        - Suggest healthy coping strategies when appropriate
        - Encourage professional help if emotions seem severe
        - Keep language warm, gentle, and supportive
        {context_block}
        User Message: {user_query}

        Respond with genuine empathy. Validate their feelings first, then gently
        offer one or two coping suggestions. End with the following disclaimer
        on a new line:
        "MindGuard AI provides educational and emotional support. It is not a
        substitute for professional medical or psychological care."
    """).strip()
    # ────────────────────────────────────────────────────────────────────────

    response = generate_response(prompt)
    return {
        "agent": "Emotional Support Agent",
        "icon": "💚",
        "reason": "User is expressing an emotional state and seeking support.",
        "response": response,
        "rag_used": bool(rag_context),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — Distress Detection Agent
# ─────────────────────────────────────────────────────────────────────────────

# Keyword patterns mapped to severity weights
_DISTRESS_PATTERNS = {
    "critical": [
        r"\bsuicid\w*\b", r"\bend\s+my\s+life\b", r"\bkill\s+myself\b",
        r"\bwant\s+to\s+die\b", r"\bno\s+reason\s+to\s+live\b",
        r"\bgoodbye\s+forever\b", r"\bself.harm\b", r"\bself.hurt\b",
    ],
    "high": [
        r"\bhopeless\b", r"\bworthless\b", r"\bgiving\s+up\b",
        r"\bcan.t\s+go\s+on\b", r"\bno\s+way\s+out\b", r"\btrapped\b",
        r"\bburden\b", r"\bempty\s+inside\b", r"\bnumb\b",
    ],
    "moderate": [
        r"\boverwhelmed\b", r"\bexhausted\b", r"\banxious\b",
        r"\bdepressed\b", r"\blonely\b", r"\bisolated\b",
        r"\bcrying\b", r"\bcan.t\s+sleep\b", r"\bpanic\b",
    ],
    "low": [
        r"\bstressed\b", r"\bworried\b", r"\bsad\b",
        r"\btired\b", r"\bfrustrated\b", r"\bupset\b",
    ],
}


def _compute_risk_score(text: str) -> tuple:
    """
    Rule-based initial risk scoring before IBM Granite analysis.
    Returns (score 0-100, detected_keywords list, preliminary_level str).
    """
    text_lower = text.lower()
    score = 0
    keywords_found = []

    weights = {"critical": 40, "high": 20, "moderate": 10, "low": 5}
    for level, patterns in _DISTRESS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += weights[level]
                keywords_found.append(level)

    score = min(score, 100)

    if score >= 60 or "critical" in keywords_found:
        level = "High Risk"
    elif score >= 30 or "high" in keywords_found:
        level = "Moderate Risk"
    elif score >= 10 or "moderate" in keywords_found:
        level = "Low Risk"
    else:
        level = "Minimal Risk"

    return score, keywords_found, level


def distress_detection_agent(user_query: str) -> dict:
    """
    Analyses the user's message for distress indicators.
    Classifies risk level and generates recommended next steps.
    Powered by IBM Granite Models for nuanced reasoning.
    """
    score, keywords, preliminary_level = _compute_risk_score(user_query)

    # ── IBM Granite Prompt — Distress Detection Agent ────────────────────────
    prompt = textwrap.dedent(f"""
        You are the MindGuard Distress Detection Agent, powered by IBM Granite AI.
        Your task is to carefully analyse the following message for signs of
        emotional distress, mental health struggles, or crisis indicators.

        Preliminary rule-based score: {score}/100 ({preliminary_level})
        Detected signal categories: {", ".join(set(keywords)) if keywords else "none"}

        User Message: {user_query}

        Please provide:
        1. RISK LEVEL: (Minimal Risk / Low Risk / Moderate Risk / High Risk)
        2. RISK SCORE: A refined score from 0 to 100
        3. KEY INDICATORS: List the specific emotional indicators you detected
        4. EXPLANATION: A brief, compassionate explanation of your assessment
        5. NEXT STEPS: 3 concrete recommended actions for this risk level

        Be sensitive, non-alarmist, and always err on the side of caution.
        Format clearly with the numbered headings above.
    """).strip()
    # ────────────────────────────────────────────────────────────────────────

    granite_analysis = generate_response(prompt)

    return {
        "agent": "Distress Detection Agent",
        "icon": "🔍",
        "reason": "Message contains emotional content requiring risk assessment.",
        "response": granite_analysis,
        "risk_score": score,
        "risk_level": preliminary_level,
        "rag_used": False,
    }


def detect_risk(text: str) -> dict:
    """
    Lightweight risk detection helper called by the orchestrator
    to decide if the Support Connector Agent should be invoked.
    Returns {"level": str, "score": int, "is_crisis": bool}.
    """
    score, keywords, level = _compute_risk_score(text)
    return {
        "level": level,
        "score": score,
        "is_crisis": "critical" in keywords or score >= 60,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — Prevention & Wellness Agent
# ─────────────────────────────────────────────────────────────────────────────
def wellness_agent(user_query: str, mood: str = "", stress_level: str = "") -> dict:
    """
    Generates a personalised wellness plan based on the user's current
    mood, stress level, and emotional state.
    Powered by IBM Granite Models.
    """
    rag_context = retrieve_context(user_query + " " + mood)
    context_block = (
        f"\n\nWellness Resource Context:\n{rag_context}\n"
        if rag_context else ""
    )

    user_state = user_query
    if mood:
        user_state += f"\nCurrent Mood: {mood}"
    if stress_level:
        user_state += f"\nStress Level: {stress_level}"

    # ── IBM Granite Prompt — Wellness Agent ──────────────────────────────────
    prompt = textwrap.dedent(f"""
        You are the MindGuard Prevention & Wellness Agent, powered by IBM Granite AI.
        Your role is to create a personalised, actionable wellness plan that helps
        users proactively manage their mental health.
        {context_block}
        User State: {user_state}

        Generate a Personalised Wellness Plan that includes:
        🌬️ BREATHING EXERCISE: A specific technique with steps
        🧘 MEDITATION PRACTICE: A short guided meditation suggestion
        📓 JOURNALING PROMPT: One reflective writing prompt for today
        😴 SLEEP TIP: One actionable sleep improvement tip
        🏃 MOVEMENT ACTIVITY: A light physical activity recommendation
        🌟 DAILY AFFIRMATION: One powerful positive affirmation

        Keep each section brief (2-3 sentences), practical, and encouraging.
        The plan should feel personalised to their current emotional state.
    """).strip()
    # ────────────────────────────────────────────────────────────────────────

    plan = generate_response(prompt)
    return {
        "agent": "Prevention & Wellness Agent",
        "icon": "🌿",
        "reason": "User needs proactive wellness strategies and self-care guidance.",
        "response": plan,
        "rag_used": bool(rag_context),
    }


def generate_wellness_plan(mood: str, stress_level: str) -> str:
    """Convenience wrapper used by the /api/wellness endpoint."""
    result = wellness_agent(
        f"I am feeling {mood} with a stress level of {stress_level}.",
        mood=mood,
        stress_level=stress_level,
    )
    return result["response"]


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — Human Support Connector Agent
# ─────────────────────────────────────────────────────────────────────────────

# Hardcoded crisis resources — no external API calls required
CRISIS_RESOURCES = {
    "International": [
        {"name": "International Association for Suicide Prevention",
         "contact": "https://www.iasp.info/resources/Crisis_Centres/"},
        {"name": "Crisis Text Line (US/UK/CA/IE)", "contact": "Text HOME to 741741"},
        {"name": "Befrienders Worldwide", "contact": "https://www.befrienders.org"},
    ],
    "United States": [
        {"name": "988 Suicide & Crisis Lifeline",
         "contact": "Call or text 988 (available 24/7)"},
        {"name": "NAMI Helpline", "contact": "1-800-950-NAMI (6264)"},
        {"name": "SAMHSA Helpline", "contact": "1-800-662-4357"},
    ],
    "India": [
        {"name": "iCall (TISS)", "contact": "9152987821"},
        {"name": "Vandrevala Foundation", "contact": "1860-2662-345 (24/7)"},
        {"name": "AASRA", "contact": "91-22-27546669"},
    ],
    "United Kingdom": [
        {"name": "Samaritans", "contact": "116 123 (free, 24/7)"},
        {"name": "Mind Infoline", "contact": "0300 123 3393"},
        {"name": "PAPYRUS HOPEline UK", "contact": "0800 068 4141"},
    ],
}


def support_connector_agent(user_query: str, risk_level: str = "Low Risk") -> dict:
    """
    Recommends professional support, counselling services, support groups,
    and crisis helplines. Escalates when High Risk is detected.
    """
    is_crisis = risk_level == "High Risk"

    # ── IBM Granite Prompt — Support Connector Agent ─────────────────────────
    prompt = textwrap.dedent(f"""
        You are the MindGuard Human Support Connector Agent, powered by IBM Granite AI.
        Your role is to compassionately connect users with appropriate professional
        mental health support.

        Current Risk Assessment: {risk_level}
        User Message: {user_query}

        Provide:
        1. PROFESSIONAL SUPPORT RECOMMENDATION: What type of professional would be
           most helpful (therapist, psychiatrist, counsellor, GP, crisis line, etc.)
        2. WHY THIS MATTERS: A brief, caring explanation of why professional support
           is beneficial
        3. HOW TO TAKE THE FIRST STEP: 2-3 concrete, gentle action steps to seek help
        4. SUPPORTIVE MESSAGE: One warm, encouraging closing message

        {"IMPORTANT: This is a HIGH RISK situation. Lead with crisis resources and "
         "urge immediate professional contact." if is_crisis else ""}

        Always end with: "MindGuard AI is not a substitute for professional care.
        Reaching out for help is a sign of strength, not weakness."
    """).strip()
    # ────────────────────────────────────────────────────────────────────────

    guidance = generate_response(prompt)

    return {
        "agent": "Human Support Connector Agent",
        "icon": "🤝",
        "reason": (
            "High distress detected — escalating to professional support resources."
            if is_crisis else
            "User may benefit from professional or peer support guidance."
        ),
        "response": guidance,
        "crisis_resources": CRISIS_RESOURCES,
        "is_crisis": is_crisis,
        "rag_used": False,
    }


# =============================================================================
# 5. MASTER ORCHESTRATOR AGENT
# =============================================================================
def orchestrate_agents(user_query: str) -> dict:
    """
    The brain of MindGuard AI.

    Workflow:
      1. Perform quick risk detection on the incoming message.
      2. Classify the query intent (awareness / support / wellness / crisis).
      3. Route to the appropriate specialized agent(s).
      4. When multiple agents are relevant, combine their outputs.
      5. Always append distress detection results for transparency.

    Returns a comprehensive response dict consumed by the Flask API.
    """
    query_lower = user_query.lower()
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Step 1: Quick risk scan ───────────────────────────────────────────────
    risk_info = detect_risk(user_query)

    # ── Step 2: Intent classification (keyword heuristic) ────────────────────
    awareness_keywords = [
        "what is", "explain", "define", "tell me about", "how does",
        "symptoms", "signs of", "difference between", "causes of",
        "anxiety", "depression", "burnout", "ptsd", "bipolar",
        "mindfulness", "meditation", "therapy",
    ]
    wellness_keywords = [
        "help me", "tips", "how to", "strategies", "exercise",
        "breathing", "sleep", "routine", "plan", "activities",
        "improve", "better", "wellness", "self-care", "relax",
    ]
    support_keywords = [
        "i feel", "i am feeling", "i'm feeling", "feel", "feeling",
        "lonely", "overwhelmed", "nobody", "nobody understands",
        "stressed about", "sad", "crying", "hopeless", "numb",
        "lost", "empty",
    ]

    is_awareness = any(kw in query_lower for kw in awareness_keywords)
    is_wellness  = any(kw in query_lower for kw in wellness_keywords)
    is_support   = any(kw in query_lower for kw in support_keywords)
    is_crisis    = risk_info["is_crisis"]

    # ── Step 3 & 4: Route and combine ────────────────────────────────────────
    primary_result    = None
    secondary_results = []
    agents_activated  = []

    if is_crisis:
        # Crisis path — distress detection + support connector always fire
        primary_result = distress_detection_agent(user_query)
        agents_activated.append(primary_result["agent"])

        connector = support_connector_agent(user_query, risk_level="High Risk")
        secondary_results.append(connector)
        agents_activated.append(connector["agent"])

    elif is_awareness:
        primary_result = awareness_agent(user_query)
        agents_activated.append(primary_result["agent"])

    elif is_wellness:
        primary_result = wellness_agent(user_query)
        agents_activated.append(primary_result["agent"])

    elif is_support:
        primary_result = emotional_support_agent(user_query)
        agents_activated.append(primary_result["agent"])

        # Always run distress detection alongside support
        distress = distress_detection_agent(user_query)
        secondary_results.append(distress)
        agents_activated.append(distress["agent"])

        # Escalate to connector if Moderate+ risk
        if risk_info["score"] >= 30:
            connector = support_connector_agent(user_query, risk_level=risk_info["level"])
            secondary_results.append(connector)
            agents_activated.append(connector["agent"])

    else:
        # Default — emotional support with distress check
        primary_result = emotional_support_agent(user_query)
        agents_activated.append(primary_result["agent"])

        distress = distress_detection_agent(user_query)
        secondary_results.append(distress)
        agents_activated.append(distress["agent"])

    # ── Step 5: Build orchestration summary ──────────────────────────────────
    orchestration_note = (
        f"Orchestrator activated {len(agents_activated)} agent(s): "
        f"{', '.join(agents_activated)}. "
        f"Risk Score: {risk_info['score']}/100 ({risk_info['level']})."
    )

    return {
        "timestamp": timestamp,
        "user_query": user_query,
        "orchestration_note": orchestration_note,
        "agents_activated": agents_activated,
        "risk_info": risk_info,
        "primary_agent": primary_result,
        "secondary_agents": secondary_results,
    }


# =============================================================================
# 6. FLASK API ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Main chat endpoint.
    Receives a user message and runs it through the orchestrator.
    Returns a JSON response consumed by the front-end.
    """
    data = request.get_json(silent=True) or {}
    user_query = (data.get("message") or "").strip()

    if not user_query:
        return jsonify({"error": "Message is required."}), 400

    try:
        result = orchestrate_agents(user_query)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/wellness", methods=["POST"])
def api_wellness():
    """
    Dedicated wellness plan endpoint.
    Accepts mood and stress_level to generate a personalised plan.
    """
    data         = request.get_json(silent=True) or {}
    mood         = (data.get("mood") or "neutral").strip()
    stress_level = (data.get("stress_level") or "moderate").strip()

    try:
        plan = generate_wellness_plan(mood, stress_level)
        return jsonify({"plan": plan})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    RAG document ingestion endpoint.
    Accepts PDF or TXT files and indexes them for retrieval-augmented generation.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    filename = file.filename or "document"
    raw_text = ""

    if filename.lower().endswith(".pdf"):
        if not PDF_SUPPORT:
            return jsonify({"error": "PyPDF2 not installed. Cannot process PDF files."}), 400
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            raw_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            return jsonify({"error": f"PDF parsing failed: {exc}"}), 400

    elif filename.lower().endswith(".txt"):
        raw_text = file.read().decode("utf-8", errors="ignore")

    else:
        return jsonify({"error": "Only PDF and TXT files are supported."}), 400

    if not raw_text.strip():
        return jsonify({"error": "No readable text found in document."}), 400

    chunk_count = ingest_document(filename, raw_text)
    return jsonify({
        "message": f"Document '{filename}' ingested successfully.",
        "chunks": chunk_count,
        "total_documents": len(rag_documents),
    })


@app.route("/api/resources", methods=["GET"])
def api_resources():
    """Return the hardcoded crisis resources for the Support Panel."""
    return jsonify(CRISIS_RESOURCES)


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health check — returns connection status and model info."""
    model_ready = _get_watsonx_model() is not None
    return jsonify({
        "status": "running",
        "watsonx_connected": model_ready,
        "model": GRANITE_MODEL_ID,
        "rag_documents": len(rag_documents),
        "pdf_support": PDF_SUPPORT,
    })


# =============================================================================
# 7. SINGLE-PAGE APPLICATION — HTML / CSS / JS (Bootstrap 5)
# =============================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MindGuard AI – Mental Health Awareness & Suicide Prevention</title>
<!-- Bootstrap 5 CDN -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"/>
<!-- Bootstrap Icons -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet"/>
<style>
  /* ── Global ─────────────────────────────────────────────────── */
  :root {
    --primary:   #4a90a4;
    --secondary: #6c8ebf;
    --accent:    #7cb9a8;
    --soft-bg:   #f0f7f9;
    --card-bg:   #ffffff;
    --text:      #2c3e50;
    --muted:     #6c757d;
    --border:    #d4e6ec;
    --crisis:    #dc3545;
    --high:      #fd7e14;
    --moderate:  #ffc107;
    --low:       #28a745;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--soft-bg);
    color: var(--text);
    margin: 0; padding: 0;
  }

  /* ── Header ─────────────────────────────────────────────────── */
  .app-header {
    background: linear-gradient(135deg, #2c5f6e 0%, #4a90a4 50%, #7cb9a8 100%);
    color: #fff; padding: 1.5rem 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.15);
  }
  .app-header h1 { font-size: 1.75rem; font-weight: 700; margin: 0; }
  .app-header p  { margin: .25rem 0 0; opacity: .85; font-size: .9rem; }
  .ibm-badge {
    display: inline-flex; align-items: center; gap: .35rem;
    background: rgba(255,255,255,.2); border-radius: 20px;
    padding: .3rem .85rem; font-size: .8rem; font-weight: 600;
  }

  /* ── Layout ─────────────────────────────────────────────────── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 1.25rem;
    padding: 1.25rem;
    max-width: 1400px;
    margin: 0 auto;
  }
  @media (max-width: 900px) {
    .main-grid { grid-template-columns: 1fr; }
  }

  /* ── Cards ───────────────────────────────────────────────────── */
  .mg-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    overflow: hidden;
  }
  .mg-card-header {
    padding: .75rem 1.1rem;
    font-weight: 600; font-size: .92rem;
    display: flex; align-items: center; gap: .5rem;
    border-bottom: 1px solid var(--border);
    background: var(--soft-bg);
  }

  /* ── Chat ────────────────────────────────────────────────────── */
  #chatBox {
    height: 420px; overflow-y: auto;
    padding: 1rem;
    display: flex; flex-direction: column; gap: .75rem;
    scroll-behavior: smooth;
  }
  .bubble {
    max-width: 80%; padding: .75rem 1rem;
    border-radius: 16px; line-height: 1.55;
    font-size: .88rem; white-space: pre-wrap;
  }
  .bubble-user {
    background: var(--primary); color: #fff;
    align-self: flex-end; border-bottom-right-radius: 4px;
  }
  .bubble-ai {
    background: #e8f4f7; color: var(--text);
    align-self: flex-start; border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }
  .bubble-crisis {
    background: #fff3f3; color: #842029;
    border: 1px solid #f5c2c7;
  }
  .bubble-system {
    background: #fff8e1; font-size: .8rem;
    color: var(--muted); align-self: center;
    border: 1px dashed #ffd54f;
  }

  /* ── Chat input ──────────────────────────────────────────────── */
  .chat-input-wrap {
    padding: .75rem 1rem; border-top: 1px solid var(--border);
    display: flex; gap: .5rem;
  }
  #chatInput {
    flex: 1; border-radius: 22px; border: 1.5px solid var(--border);
    padding: .6rem 1rem; font-size: .88rem; resize: none;
    transition: border-color .2s;
    font-family: inherit;
  }
  #chatInput:focus { outline: none; border-color: var(--primary); }
  #sendBtn {
    background: var(--primary); color: #fff;
    border: none; border-radius: 22px;
    padding: .6rem 1.3rem; cursor: pointer;
    font-weight: 600; font-size: .88rem;
    transition: background .2s;
    white-space: nowrap;
  }
  #sendBtn:hover { background: #3a7a8e; }
  #sendBtn:disabled { background: #aaa; cursor: not-allowed; }

  /* ── Agent Panel ─────────────────────────────────────────────── */
  .agent-card {
    border-radius: 10px; padding: .75rem 1rem;
    margin-bottom: .6rem; border: 1px solid var(--border);
    font-size: .83rem; transition: all .25s;
    cursor: default;
  }
  .agent-card.active {
    border-color: var(--primary);
    background: #e8f4f7;
    box-shadow: 0 0 0 2px rgba(74,144,164,.25);
  }
  .agent-card .agent-name { font-weight: 700; font-size: .87rem; }
  .agent-card .agent-reason { color: var(--muted); margin-top: .2rem; }
  .agent-pulse {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: var(--accent);
    animation: pulse 1.2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .4; transform: scale(.7); }
  }

  /* ── Risk Badge ──────────────────────────────────────────────── */
  .risk-badge {
    display: inline-block; border-radius: 20px;
    padding: .3rem .9rem; font-weight: 700;
    font-size: .82rem; letter-spacing: .5px;
  }
  .risk-minimal  { background: #d4edda; color: #155724; }
  .risk-low      { background: #d4edda; color: #155724; }
  .risk-moderate { background: #fff3cd; color: #856404; }
  .risk-high     { background: #f8d7da; color: #721c24; }

  /* ── Risk Meter ──────────────────────────────────────────────── */
  .risk-meter { height: 10px; border-radius: 5px; overflow: hidden;
                background: #e9ecef; margin: .4rem 0; }
  .risk-fill  { height: 100%; border-radius: 5px;
                transition: width .6s ease; }
  .risk-fill-minimal  { background: #28a745; }
  .risk-fill-low      { background: #28a745; }
  .risk-fill-moderate { background: #ffc107; }
  .risk-fill-high     { background: #dc3545; }

  /* ── Wellness Panel ──────────────────────────────────────────── */
  .wellness-item {
    border-left: 3px solid var(--accent);
    padding: .5rem .75rem; margin-bottom: .5rem;
    background: #f8fffe; border-radius: 0 8px 8px 0;
    font-size: .83rem; line-height: 1.55;
  }

  /* ── Resource Cards ──────────────────────────────────────────── */
  .resource-region { margin-bottom: .75rem; }
  .resource-region h6 { font-size: .82rem; font-weight: 700;
                         color: var(--primary); margin-bottom: .35rem; }
  .resource-item {
    font-size: .8rem; padding: .3rem 0;
    border-bottom: 1px solid #f0f0f0; display: flex;
    justify-content: space-between; align-items: flex-start; gap: .5rem;
  }
  .resource-item:last-child { border-bottom: none; }

  /* ── Upload Zone ─────────────────────────────────────────────── */
  .upload-zone {
    border: 2px dashed var(--border); border-radius: 10px;
    text-align: center; padding: 1.2rem;
    color: var(--muted); font-size: .83rem; cursor: pointer;
    transition: border-color .2s;
  }
  .upload-zone:hover { border-color: var(--primary); }

  /* ── Status Bar ──────────────────────────────────────────────── */
  .status-dot { width: 8px; height: 8px; border-radius: 50%;
                display: inline-block; }
  .status-dot.connected    { background: #28a745; }
  .status-dot.disconnected { background: #dc3545; }

  /* ── Typing indicator ────────────────────────────────────────── */
  .typing-dots span {
    display: inline-block; width: 6px; height: 6px;
    background: var(--muted); border-radius: 50%; margin: 0 1px;
    animation: blink 1.2s infinite;
  }
  .typing-dots span:nth-child(2) { animation-delay: .2s; }
  .typing-dots span:nth-child(3) { animation-delay: .4s; }
  @keyframes blink {
    0%, 80%, 100% { opacity: .2; }
    40% { opacity: 1; }
  }

  /* ── Footer ──────────────────────────────────────────────────── */
  footer {
    text-align: center; font-size: .75rem;
    color: var(--muted); padding: 1rem;
    border-top: 1px solid var(--border); margin-top: 1rem;
  }

  /* ── Scrollbar ───────────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #c0d4d9; border-radius: 10px; }
</style>
</head>
<body>

<!-- ════════════════════ HEADER ════════════════════ -->
<header class="app-header">
  <div class="d-flex justify-content-between align-items-start flex-wrap gap-2">
    <div>
      <h1>🛡️ MindGuard AI</h1>
      <p>Mental Health Awareness &amp; Suicide Prevention Agent</p>
    </div>
    <div class="d-flex flex-column align-items-end gap-1">
      <span class="ibm-badge">
        <svg width="14" height="14" viewBox="0 0 32 32" fill="#fff">
          <rect x="0" y="6" width="32" height="3"/><rect x="0" y="12" width="32" height="3"/>
          <rect x="0" y="18" width="32" height="3"/><rect x="0" y="24" width="32" height="3"/>
        </svg>
        IBM watsonx.ai Studio
      </span>
      <span class="ibm-badge">🤖 IBM Granite Models</span>
      <div class="d-flex align-items-center gap-2 mt-1">
        <span class="status-dot" id="statusDot"></span>
        <small id="statusText" style="font-size:.75rem;opacity:.85">Checking…</small>
      </div>
    </div>
  </div>
</header>

<!-- ════════════════════ DISCLAIMER BANNER ════════════════════ -->
<div class="alert alert-warning mb-0 py-2 px-3 rounded-0" style="font-size:.8rem;text-align:center;">
  ⚠️ <strong>Important:</strong> MindGuard AI provides educational and emotional support only.
  It is <strong>not</strong> a substitute for professional medical or psychological care.
  If you are in crisis, please contact a helpline immediately.
</div>

<!-- ════════════════════ MAIN CONTENT ════════════════════ -->
<main>
<div class="main-grid">

  <!-- LEFT COLUMN -->
  <div class="d-flex flex-column gap-3">

    <!-- Chat Card -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-chat-heart-fill text-primary"></i> Mental Health Chat
        <span class="badge bg-primary ms-auto" style="font-size:.7rem;">AI-Powered</span>
      </div>

      <div id="chatBox">
        <!-- Welcome message -->
        <div class="bubble bubble-ai">
          <strong>👋 Welcome to MindGuard AI</strong><br><br>
          I'm here to support your mental health journey. You can:<br>
          • Ask about anxiety, depression, stress, or burnout<br>
          • Share how you're feeling — I'm here to listen<br>
          • Request coping strategies or a wellness plan<br>
          • Get connected to professional resources<br><br>
          <em>How are you feeling today?</em>
        </div>
      </div>

      <div class="chat-input-wrap">
        <textarea id="chatInput" rows="1" placeholder="Type your message…" maxlength="1000"></textarea>
        <button id="sendBtn" onclick="sendMessage()">
          <i class="bi bi-send-fill"></i> Send
        </button>
      </div>

      <!-- Quick prompts -->
      <div style="padding:.5rem 1rem .75rem;display:flex;gap:.4rem;flex-wrap:wrap;">
        <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:.75rem;"
                onclick="quickPrompt('What is anxiety and how does it feel?')">What is anxiety?</button>
        <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:.75rem;"
                onclick="quickPrompt('I feel overwhelmed and stressed today.')">I feel overwhelmed</button>
        <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:.75rem;"
                onclick="quickPrompt('Give me breathing exercises for stress relief.')">Breathing exercises</button>
        <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:.75rem;"
                onclick="quickPrompt('I feel lonely and nobody understands me.')">Feeling lonely</button>
        <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:.75rem;"
                onclick="quickPrompt('How can mindfulness help with depression?')">Mindfulness tips</button>
      </div>
    </div>

    <!-- Agent Orchestration Visualization -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-diagram-3-fill" style="color:var(--secondary)"></i>
        Agentic AI Workflow — Agent Orchestration
      </div>
      <div style="padding:1rem;">
        <p class="text-muted mb-2" style="font-size:.8rem;">
          The Master Orchestrator routes your query to the most relevant specialized agent(s).
        </p>
        <div id="agentPipeline">
          <!-- Populated by JS -->
          <div class="text-center text-muted py-3" style="font-size:.83rem;">
            <i class="bi bi-arrow-down-circle" style="font-size:1.5rem;"></i><br>
            Send a message to see which agents activate.
          </div>
        </div>
        <div id="orchestrationNote" class="mt-2" style="font-size:.78rem;color:var(--muted);"></div>
      </div>
    </div>

    <!-- Wellness Dashboard -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-flower1" style="color:var(--accent)"></i>
        Prevention &amp; Wellness Plan Generator
      </div>
      <div style="padding:1rem;">
        <div class="row g-2 mb-3">
          <div class="col-6">
            <label class="form-label" style="font-size:.82rem;font-weight:600;">Current Mood</label>
            <select class="form-select form-select-sm" id="moodSelect">
              <option value="anxious">😰 Anxious</option>
              <option value="sad">😔 Sad</option>
              <option value="stressed" selected>😤 Stressed</option>
              <option value="overwhelmed">😵 Overwhelmed</option>
              <option value="tired">😴 Tired</option>
              <option value="neutral">😐 Neutral</option>
              <option value="hopeful">🙂 Hopeful</option>
            </select>
          </div>
          <div class="col-6">
            <label class="form-label" style="font-size:.82rem;font-weight:600;">Stress Level</label>
            <select class="form-select form-select-sm" id="stressSelect">
              <option value="low">🟢 Low</option>
              <option value="moderate" selected>🟡 Moderate</option>
              <option value="high">🔴 High</option>
              <option value="severe">⚫ Severe</option>
            </select>
          </div>
        </div>
        <button class="btn btn-sm w-100" style="background:var(--accent);color:#fff;font-weight:600;"
                onclick="generateWellnessPlan()">
          <i class="bi bi-magic"></i> Generate My Wellness Plan (IBM Granite AI)
        </button>
        <div id="wellnessOutput" class="mt-3"></div>
      </div>
    </div>

  </div>

  <!-- RIGHT COLUMN -->
  <div class="d-flex flex-column gap-3">

    <!-- Risk Detection Panel -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-shield-exclamation text-danger"></i>
        Distress Detection Dashboard
      </div>
      <div style="padding:1rem;">
        <div class="d-flex justify-content-between align-items-center mb-1">
          <span style="font-size:.82rem;font-weight:600;">Risk Level</span>
          <span id="riskBadge" class="risk-badge risk-minimal">—</span>
        </div>
        <div class="risk-meter">
          <div class="risk-fill risk-fill-minimal" id="riskFill" style="width:0%"></div>
        </div>
        <div class="d-flex justify-content-between" style="font-size:.75rem;color:var(--muted);">
          <span>0</span><span>Risk Score: <strong id="riskScore">—</strong></span><span>100</span>
        </div>
        <div id="riskExplanation" class="mt-2" style="font-size:.8rem;color:var(--muted);">
          Send a message to see the real-time distress assessment.
        </div>
      </div>
    </div>

    <!-- Active Agents Panel (summary) -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-cpu-fill" style="color:var(--secondary)"></i>
        Specialized AI Agents
      </div>
      <div style="padding:.75rem;">
        <div class="agent-card" id="agentCard1">
          <div class="d-flex gap-2 align-items-start">
            <span>🧠</span>
            <div>
              <div class="agent-name">Mental Health Awareness Agent</div>
              <div class="agent-reason">Education on anxiety, depression, stress &amp; mindfulness</div>
            </div>
          </div>
        </div>
        <div class="agent-card" id="agentCard2">
          <div class="d-flex gap-2 align-items-start">
            <span>💚</span>
            <div>
              <div class="agent-name">Emotional Support Agent</div>
              <div class="agent-reason">Empathetic support &amp; coping strategies</div>
            </div>
          </div>
        </div>
        <div class="agent-card" id="agentCard3">
          <div class="d-flex gap-2 align-items-start">
            <span>🔍</span>
            <div>
              <div class="agent-name">Distress Detection Agent</div>
              <div class="agent-reason">Risk classification &amp; early warning signals</div>
            </div>
          </div>
        </div>
        <div class="agent-card" id="agentCard4">
          <div class="d-flex gap-2 align-items-start">
            <span>🌿</span>
            <div>
              <div class="agent-name">Prevention &amp; Wellness Agent</div>
              <div class="agent-reason">Personalised wellness &amp; self-care plans</div>
            </div>
          </div>
        </div>
        <div class="agent-card" id="agentCard5">
          <div class="d-flex gap-2 align-items-start">
            <span>🤝</span>
            <div>
              <div class="agent-name">Human Support Connector</div>
              <div class="agent-reason">Professional resources &amp; crisis helplines</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Support Resources Panel -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-telephone-fill text-success"></i>
        Crisis &amp; Support Resources
      </div>
      <div style="padding:.75rem;max-height:280px;overflow-y:auto;" id="resourcesPanel">
        <div class="text-center text-muted py-2" style="font-size:.8rem;">Loading resources…</div>
      </div>
    </div>

    <!-- RAG Document Upload -->
    <div class="mg-card">
      <div class="mg-card-header">
        <i class="bi bi-database-fill-up" style="color:var(--secondary)"></i>
        RAG Knowledge Base
      </div>
      <div style="padding:.75rem;">
        <p style="font-size:.78rem;color:var(--muted);margin-bottom:.6rem;">
          Upload WHO guidelines, coping strategy PDFs, or mental health resources
          to enhance AI responses with domain-specific knowledge.
        </p>
        <label class="upload-zone d-block" for="ragUpload">
          <i class="bi bi-cloud-upload" style="font-size:1.4rem;"></i><br>
          Click to upload PDF or TXT<br>
          <small style="font-size:.72rem;">Max 16 MB</small>
        </label>
        <input type="file" id="ragUpload" accept=".pdf,.txt" class="d-none"
               onchange="uploadDocument(this)"/>
        <div id="uploadStatus" class="mt-2" style="font-size:.78rem;"></div>
        <div id="ragDocList" style="font-size:.78rem;margin-top:.5rem;color:var(--muted);">
          No documents uploaded yet.
        </div>
      </div>
    </div>

  </div>
</div>
</main>

<footer>
  🛡️ MindGuard AI — Powered by <strong>IBM watsonx.ai Studio</strong> &amp;
  <strong>IBM Granite Models</strong><br>
  <span style="font-size:.7rem;">
    For educational &amp; demonstration purposes · IBM SkillsBuild · Hackathons · Academic Projects
  </span>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
// ══════════════════════════════════════════════════════════════════════
// MindGuard AI — Front-End JavaScript
// ══════════════════════════════════════════════════════════════════════

const agentIconMap = {
  "Mental Health Awareness Agent":  "agentCard1",
  "Emotional Support Agent":        "agentCard2",
  "Distress Detection Agent":       "agentCard3",
  "Prevention & Wellness Agent":    "agentCard4",
  "Human Support Connector Agent":  "agentCard5",
};

// ── Initialise on page load ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkStatus();
  loadResources();
  // Auto-resize textarea
  const inp = document.getElementById("chatInput");
  inp.addEventListener("input", () => {
    inp.style.height = "auto";
    inp.style.height = Math.min(inp.scrollHeight, 100) + "px";
  });
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
});

// ── Status check ─────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    const dot  = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    if (d.watsonx_connected) {
      dot.className  = "status-dot connected";
      text.textContent = "IBM watsonx.ai Connected";
    } else {
      dot.className  = "status-dot disconnected";
      text.textContent = "Demo Mode (no credentials)";
    }
  } catch {
    document.getElementById("statusText").textContent = "Status unknown";
  }
}

// ── Quick prompt ─────────────────────────────────────────────────────
function quickPrompt(text) {
  document.getElementById("chatInput").value = text;
  sendMessage();
}

// ── Send message ─────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById("chatInput");
  const msg   = input.value.trim();
  if (!msg) return;

  input.value = "";
  input.style.height = "auto";
  document.getElementById("sendBtn").disabled = true;

  // Render user bubble
  appendBubble(msg, "user");

  // Show typing indicator
  const typingId = appendTyping();

  try {
    const res  = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json();

    removeBubble(typingId);

    if (data.error) {
      appendBubble("⚠️ " + data.error, "ai");
      return;
    }

    // ── Update agent pipeline visualization ──────────────────────────
    updateAgentPipeline(data);

    // ── Update risk dashboard ─────────────────────────────────────────
    if (data.risk_info) {
      updateRiskPanel(data.risk_info);
    }

    // ── Render primary agent response ─────────────────────────────────
    if (data.primary_agent) {
      const pa = data.primary_agent;
      const cls = data.risk_info && data.risk_info.is_crisis ? "bubble-crisis" : "ai";
      const html = `<strong>${pa.icon} ${pa.agent}</strong>\n\n${pa.response}`;
      appendBubble(html, cls);
    }

    // ── Render secondary agent responses ─────────────────────────────
    if (data.secondary_agents && data.secondary_agents.length > 0) {
      for (const sa of data.secondary_agents) {
        if (sa.agent === "Distress Detection Agent") continue; // shown in dashboard
        const html = `<strong>${sa.icon} ${sa.agent}</strong>\n\n${sa.response}`;
        appendBubble(html, "ai");

        // Render crisis resources inline if connector fired
        if (sa.crisis_resources && sa.is_crisis) {
          renderCrisisResourcesBubble(sa.crisis_resources);
        }
      }
    }

    // ── Highlight active agents ───────────────────────────────────────
    highlightAgents(data.agents_activated || []);

  } catch (err) {
    removeBubble(typingId);
    appendBubble("⚠️ Network error: " + err.message, "ai");
  } finally {
    document.getElementById("sendBtn").disabled = false;
    document.getElementById("chatInput").focus();
  }
}

// ── Bubble helpers ───────────────────────────────────────────────────
function appendBubble(text, type) {
  const box  = document.getElementById("chatBox");
  const div  = document.createElement("div");
  const uid  = "b" + Date.now() + Math.random();
  div.id     = uid;
  div.className = `bubble bubble-${type}`;
  // For user bubbles escape HTML; for AI bubbles allow safe tags (bold, br, em)
  if (type === "user") {
    div.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
  } else {
    div.innerHTML = sanitizeAI(text);
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return uid;
}

function appendTyping() {
  const box = document.getElementById("chatBox");
  const div = document.createElement("div");
  const uid = "t" + Date.now();
  div.id    = uid;
  div.className = "bubble bubble-ai typing-dots";
  div.innerHTML = '<span></span><span></span><span></span>';
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return uid;
}

function removeBubble(uid) {
  const el = document.getElementById(uid);
  if (el) el.remove();
}

function escapeHtml(t) {
  return t.replace(/&/g,"&amp;").replace(/</g,"&lt;")
          .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// Allow <strong>, <em>, <br> from AI responses; escape everything else
function sanitizeAI(text) {
  // First escape all HTML
  let safe = escapeHtml(text);
  // Then selectively restore safe tags
  safe = safe.replace(/&lt;strong&gt;/gi, "<strong>")
             .replace(/&lt;\/strong&gt;/gi, "</strong>")
             .replace(/&lt;em&gt;/gi, "<em>")
             .replace(/&lt;\/em&gt;/gi, "</em>")
             .replace(/&lt;br\s*\/?&gt;/gi, "<br>");
  // Convert newlines to <br>
  safe = safe.replace(/\n/g, "<br>");
  return safe;
}

// ── Agent pipeline visualization ─────────────────────────────────────
function updateAgentPipeline(data) {
  const panel = document.getElementById("agentPipeline");
  const note  = document.getElementById("orchestrationNote");

  const agents = [];
  if (data.primary_agent)   agents.push(data.primary_agent);
  if (data.secondary_agents) agents.push(...data.secondary_agents);

  if (!agents.length) {
    panel.innerHTML = '<div class="text-muted text-center" style="font-size:.82rem;">No agents activated.</div>';
    return;
  }

  let html = '<div class="d-flex align-items-center gap-2 flex-wrap">';
  agents.forEach((a, i) => {
    html += `
      <div class="text-center" style="flex:1;min-width:80px;">
        <div style="font-size:1.3rem;">${a.icon}</div>
        <div style="font-size:.72rem;font-weight:600;color:var(--primary);margin-top:.2rem;">
          ${a.agent.replace(" Agent","").replace(" Connector","")}</div>
        <div style="font-size:.65rem;color:var(--muted);">Active</div>
      </div>`;
    if (i < agents.length - 1) {
      html += `<i class="bi bi-arrow-right" style="color:var(--border);font-size:.9rem;"></i>`;
    }
  });
  html += "</div>";
  panel.innerHTML = html;
  note.textContent = data.orchestration_note || "";
}

// ── Highlight agent cards ─────────────────────────────────────────────
function highlightAgents(activeNames) {
  // Reset all
  Object.values(agentIconMap).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active");
  });
  // Activate matching
  activeNames.forEach(name => {
    const id = agentIconMap[name];
    if (id) {
      const el = document.getElementById(id);
      if (el) el.classList.add("active");
    }
  });
}

// ── Risk panel ───────────────────────────────────────────────────────
function updateRiskPanel(riskInfo) {
  const badge  = document.getElementById("riskBadge");
  const fill   = document.getElementById("riskFill");
  const score  = document.getElementById("riskScore");
  const expl   = document.getElementById("riskExplanation");

  const level = riskInfo.level || "Minimal Risk";
  const sc    = riskInfo.score || 0;
  const slug  = level.toLowerCase().replace(/\s+/g, "-").replace("-risk","");

  badge.textContent  = level;
  badge.className    = `risk-badge risk-${slug}`;
  fill.style.width   = sc + "%";
  fill.className     = `risk-fill risk-fill-${slug}`;
  score.textContent  = sc + "/100";

  if (riskInfo.is_crisis) {
    expl.innerHTML = `<span style="color:#dc3545;font-weight:600;">
      ⚠️ Crisis signals detected. Please see the Support Resources panel immediately.
    </span>`;
  } else if (level === "Moderate Risk") {
    expl.textContent = "Moderate distress indicators present. Consider speaking with a professional.";
  } else if (level === "Low Risk") {
    expl.textContent = "Some stress indicators detected. Wellness strategies recommended.";
  } else {
    expl.textContent = "No significant distress indicators detected.";
  }
}

// ── Crisis resources inline bubble ───────────────────────────────────
function renderCrisisResourcesBubble(resources) {
  const box = document.getElementById("chatBox");
  const div = document.createElement("div");
  div.className = "bubble bubble-crisis";

  let html = "<strong>🆘 Emergency Crisis Resources</strong><br><br>";
  for (const [region, items] of Object.entries(resources)) {
    html += `<strong>${region}</strong><br>`;
    items.forEach(r => {
      html += `• ${r.name}: <em>${r.contact}</em><br>`;
    });
    html += "<br>";
  }
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ── Wellness plan ─────────────────────────────────────────────────────
async function generateWellnessPlan() {
  const mood  = document.getElementById("moodSelect").value;
  const stress = document.getElementById("stressSelect").value;
  const out   = document.getElementById("wellnessOutput");

  out.innerHTML = '<div class="text-muted" style="font-size:.8rem;"><i class="bi bi-hourglass-split"></i> IBM Granite is crafting your plan…</div>';

  try {
    const res  = await fetch("/api/wellness", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mood, stress_level: stress }),
    });
    const data = await res.json();

    if (data.error) { out.textContent = "Error: " + data.error; return; }

    // Parse and render sections
    const plan = data.plan || "";
    const lines = plan.split(/\n+/).filter(l => l.trim());
    let html = "";
    lines.forEach(line => {
      if (line.match(/^[🌬🧘📓😴🏃🌟]/u)) {
        html += `<div class="wellness-item">${escapeHtml(line)}</div>`;
      } else {
        html += `<p style="font-size:.82rem;margin:.3rem 0 0;">${escapeHtml(line)}</p>`;
      }
    });
    out.innerHTML = html || `<p style="font-size:.82rem;">${escapeHtml(plan)}</p>`;
  } catch (err) {
    out.textContent = "Network error: " + err.message;
  }
}

// ── Load crisis resources ─────────────────────────────────────────────
async function loadResources() {
  try {
    const res  = await fetch("/api/resources");
    const data = await res.json();
    const panel = document.getElementById("resourcesPanel");

    let html = "";
    for (const [region, items] of Object.entries(data)) {
      html += `<div class="resource-region">
        <h6><i class="bi bi-geo-alt-fill"></i> ${region}</h6>`;
      items.forEach(r => {
        html += `<div class="resource-item">
          <span>${r.name}</span>
          <span style="color:var(--primary);font-weight:600;white-space:nowrap;">${r.contact}</span>
        </div>`;
      });
      html += "</div>";
    }
    panel.innerHTML = html;
  } catch {
    document.getElementById("resourcesPanel").textContent = "Failed to load resources.";
  }
}

// ── RAG document upload ───────────────────────────────────────────────
async function uploadDocument(input) {
  const file = input.files[0];
  if (!file) return;

  const status = document.getElementById("uploadStatus");
  status.innerHTML = `<span class="text-muted"><i class="bi bi-arrow-repeat"></i> Uploading ${file.name}…</span>`;

  const form = new FormData();
  form.append("file", file);

  try {
    const res  = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      status.innerHTML = `<span class="text-danger">❌ ${data.error}</span>`;
    } else {
      status.innerHTML = `<span class="text-success">✅ ${data.message} (${data.chunks} chunks)</span>`;
      const list = document.getElementById("ragDocList");
      list.textContent = `${data.total_documents} document(s) in knowledge base.`;
    }
  } catch (err) {
    status.innerHTML = `<span class="text-danger">❌ Upload failed: ${err.message}</span>`;
  } finally {
    input.value = "";
  }
}
</script>
</body>
</html>
"""

# =============================================================================
# 8. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  MindGuard AI — Mental Health Awareness & Suicide Prevention")
    print("  Powered by IBM watsonx.ai Studio & IBM Granite Models")
    print("=" * 65)
    print(f"  Model        : {GRANITE_MODEL_ID}")
    print(f"  watsonx URL  : {WATSONX_URL}")
    print(f"  API Key Set  : {'Yes [OK]' if WATSONX_API_KEY else 'No - Demo Mode'}")
    print(f"  Project ID   : {WATSONX_PROJECT_ID or 'Not set - Demo Mode'}")
    print(f"  PDF Support  : {'Yes (PyPDF2)' if PDF_SUPPORT else 'No (install PyPDF2)'}")
    print(f"  NumPy RAG    : {'Yes' if NUMPY_AVAILABLE else 'No (keyword fallback)'}")
    print("=" * 65)
    print("  Open: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=True, host="0.0.0.0", port=5000)
