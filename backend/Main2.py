"""
AI Therapist Backend — FastAPI + MongoDB + Ollama (phi3)
=========================================================
POST /api/auth/signup              — register
POST /api/auth/login               — login → JWT
POST /api/sessions/start           — start session (new or returning)
POST /api/sessions/voice-message   — WAV → Whisper → LLM reply
POST /api/sessions/end             — summarise full chat + save
GET  /api/sessions/history         — list past sessions
POST /api/sessions/detect-emotion  — Pipeline B: text → Hartmann NLP → emotion probs
POST /api/sessions/acoustic-emotion — Pipeline A: WAV chunk → librosa → emotion probs
"""

import os, hashlib, uuid, base64, tempfile
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pymongo import MongoClient
import jwt
import ollama

# ── Optional: faster-whisper ──────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    WHISPER_OK = True
    print("[Whisper] Model loaded OK.")
except Exception as e:
    WHISPER_OK = False
    print(f"[Warning] faster-whisper unavailable: {e}")

# ── Pipeline B: Hartmann emotion model ────────────────────────────────────────
NLP_MODEL     = "j-hartmann/emotion-english-distilroberta-base"
_nlp_pipeline = None
NLP_OK        = False

def _load_nlp():
    global _nlp_pipeline, NLP_OK
    if _nlp_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            print(f"[NLP] Loading Hartmann model '{NLP_MODEL}' ...")
            _nlp_pipeline = hf_pipeline(
                "text-classification",
                model=NLP_MODEL,
                top_k=None,
                device=device,
            )
            NLP_OK = True
            print("[NLP] Hartmann model ready.")
        except Exception as e:
            print(f"[Warning] Hartmann NLP unavailable: {e}")
    return _nlp_pipeline

# ── Pipeline A: Librosa acoustic features ─────────────────────────────────────
try:
    import librosa
    import numpy as np
    LIBROSA_OK = True
    print("[Librosa] Available.")
except Exception as e:
    LIBROSA_OK = False
    print(f"[Warning] librosa unavailable: {e}")

EMOTION_KEYS_PY = ['neutral', 'happy', 'excited', 'calm', 'sad', 'anger']
W_ACOUSTIC      = 0.40
W_SEMANTIC      = 0.60
SAMPLE_RATE     = 16_000

_HARTMANN_MAP = {
    'anger':   'anger',
    'disgust': 'anger',
    'fear':    'sad',
    'joy':     'happy',
    'neutral': 'neutral',
    'sadness': 'sad',
    'surprise':'excited',
}

def _pipeline_acoustic(audio_np) -> dict:
    import numpy as np
    import librosa

    chunk = audio_np.flatten().astype(np.float32)

    if len(chunk) < 512:
        v = 1.0 / len(EMOTION_KEYS_PY)
        return {k: v for k in EMOTION_KEYS_PY}

    rms  = float(np.mean(librosa.feature.rms(y=chunk)))
    zcr  = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=SAMPLE_RATE)))
    roll = float(np.mean(librosa.feature.spectral_rolloff(
                    y=chunk, sr=SAMPLE_RATE, roll_percent=0.85)))
    bw   = float(np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=SAMPLE_RATE)))
    mfccs = librosa.feature.mfcc(y=chunk, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc  = float(np.mean(mfccs[0]))

    if rms < 0.001:
        probs = {k: (1.0 if k == 'neutral' else 0.0) for k in EMOTION_KEYS_PY}
    else:
        raw = {
            'anger':   (rms * 180) * (zcr * 45) * (bw / 900),
            'excited': (rms * 90)  * (cent / 1400) * (roll / 1800),
            'happy':   (rms * 55)  * (cent / 1100) * max(0.1, (mfcc + 35) / 35),
            'calm':    max(0, (1.0 - rms * 75)) * max(0, (1.0 - zcr * 35)),
            'sad':     max(0, (1.0 - rms * 55)) * max(0, (1.0 - cent / 1400)),
            'neutral': 0.25,
        }
        total = sum(max(v, 1e-4) for v in raw.values())
        probs = {k: max(raw[k], 1e-4) / total for k in EMOTION_KEYS_PY}

    print(f"[Pipeline A] rms={rms:.4f} → {max(probs, key=probs.get).upper()}")
    return probs


def _pipeline_semantic_text(text: str) -> dict:
    nlp = _load_nlp()
    if nlp is None:
        v = 1.0 / len(EMOTION_KEYS_PY)
        return {k: v for k in EMOTION_KEYS_PY}

    preds = nlp(text[:512])[0]
    sp = {k: 0.0 for k in EMOTION_KEYS_PY}
    for item in preds:
        mapped = _HARTMANN_MAP.get(item['label'].lower(), 'neutral')
        sp[mapped] += item['score']

    total = sum(sp.values())
    if total:
        sp = {k: v / total for k, v in sp.items()}

    print(f"[Pipeline B] \"{text[:60]}\" → {max(sp, key=sp.get).upper()}")
    return sp


def _fuse(acoustic_probs: dict, semantic_probs: dict, has_speech: bool) -> tuple:
    if has_speech:
        fused = {k: W_ACOUSTIC * acoustic_probs[k] + W_SEMANTIC * semantic_probs[k]
                 for k in EMOTION_KEYS_PY}
    else:
        fused = dict(acoustic_probs)

    total = sum(fused.values())
    if total:
        fused = {k: v / total for k, v in fused.items()}

    emotion = max(fused, key=fused.get)
    return emotion, fused

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("JWT_SECRET", "serene-secret-change-in-prod")
ALGORITHM   = "HS256"
TOKEN_HOURS = 72
LLM_MODEL   = os.getenv("LLM_MODEL", "phi3")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
_db       = MongoClient(MONGO_URI)["therapist_app"]

users_col    = _db["users"]
sessions_col = _db["sessions"]
messages_col = _db["messages"]

app = FastAPI(title="Serene AI Therapist API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
security = HTTPBearer()

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def _make_token(uid):
    exp = datetime.utcnow() + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": uid, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def _decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except Exception:
        raise HTTPException(401, "Invalid token")

def get_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    uid  = _decode_token(creds.credentials)
    user = users_col.find_one({"_id": uid})
    if not user: raise HTTPException(401, "User not found")
    return user

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class AuthReq(BaseModel):
    name: str
    password: str

class VoiceReq(BaseModel):
    session_id: str
    audio_b64:  str
    emotion:    str = "neutral"

class ChatMsg(BaseModel):
    role:    str
    content: str

class EndReq(BaseModel):
    session_id:   str
    chat_history: Optional[List[ChatMsg]] = None

class EmotionTextReq(BaseModel):
    text: str

class AcousticChunkReq(BaseModel):
    audio_b64: str

# ═══════════════════════════════════════════════════════════════════════════════
# THERAPIST SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
#
# Three-phase therapeutic arc:
#
#   PHASE 1 — UNDERSTAND (exchanges 0–2)
#     Listen. Reflect. Ask at most one focused question per turn.
#     Do NOT advise. Just understand the real situation.
#
#   PHASE 2 — CONNECT (exchanges 3–5)
#     Name what you're noticing. Connect the threads. Normalise.
#     Help the person see their situation more clearly.
#     Reduce questions — only ask if something important is still missing.
#
#   PHASE 3 — ACT (exchanges 6+)
#     Stop questioning. Give 2–3 concrete, specific, personalised action steps.
#     Tailored to THEIR exact situation — never generic.
#     Explain WHY each step helps them specifically.
#     Ask only: which one feels most doable right now?
#
THERAPIST_SYSTEM = """ You are a compassionate and skilled therapist.
Strictly Start the conversation with:
"Hi, it’s good to see you. How have you been feeling lately?" nothing else.
Your goal is to understand the user deeply through conversation, not to immediately give advice.
Use short, natural responses (2–4 lines max)
Follow this approach:
- Ask open-ended, thoughtful questions
- Validate feelings ("That sounds difficult", "I’m glad you shared that")
- slowly explore root causes (past, pressure, beliefs)
- Identify negative thought patterns gently
- Do NOT rush to solutions early
- Build trust step-by-step like a real therapy session
- Use short, natural responses (2–4 lines max)
- Ask one question at a time

As the conversation progresses:
- Reflect back what the user says
- Help them notice patterns (like avoidance, self-criticism)
- Gradually guide them toward small, practical, achievable steps
- End with 2–3 simple actionable suggestions

Tone:
- Calm, non-judgmental, human-like
- Never robotic or overly formal
- Never dismiss feelings

Make sure you communicate and never explain why you asked that question.
 """


# ── Greetings ─────────────────────────────────────────────────────────────────
NEW_USER_GREETING_PROMPT = [
    {"role": "system", "content": THERAPIST_SYSTEM},
    {
        "role": "user",
        "content": (
            """ You are a compassionate and skilled therapist.
Strictly Start the conversation with:
"Hi, it’s good to see you. How have you been feeling lately?" nothing else.
Your goal is to understand the user deeply through conversation, not to immediately give advice.
Use short, natural responses (2–4 lines max)
Follow this approach:
- Ask open-ended, thoughtful questions
- Validate feelings ("That sounds difficult", "I’m glad you shared that")
- slowly explore root causes (past, pressure, beliefs)
- Identify negative thought patterns gently
- Do NOT rush to solutions early
- Build trust step-by-step like a real therapy session
- Use short, natural responses (2–4 lines max)
- Ask one question at a time

As the conversation progresses:
- Reflect back what the user says
- Help them notice patterns (like avoidance, self-criticism)
- Gradually guide them toward small, practical, achievable steps
- End with 2–3 simple actionable suggestions

Tone:
- Calm, non-judgmental, human-like
- Never robotic or overly formal
- Never dismiss feelings

Make sure you communicate and never explain why you asked that question.
 """
        ),
    },
]

def returning_user_greeting_prompt(past_summary: str) -> list:
    return [
        {"role": "system", "content": THERAPIST_SYSTEM},
        {
            "role": "system",
            "content": (
                "You already know this client from a previous session. "
                "Here is what you know — do NOT quote it or say 'last session'. "
                "Simply let it inform how you greet them and what you notice:\n\n"
                + past_summary
            ),
        },
        {
            "role": "user",
            "content": (
                "The client has returned. Greet them like someone you genuinely know and care about. "
                "Acknowledge where they were without quoting the notes. "
                "Ask ONE warm, specific question that shows you remember them. "
                "2–3 sentences maximum."
            ),
        },
    ]


def _llm(messages: list) -> str:
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def _count_user_exchanges(db_history: list) -> int:
    """Count completed user turns in the conversation."""
    return sum(1 for h in db_history if h.get("role") == "user")


def _build_conversation_messages(session: dict, db_history: list, emotion: str) -> list:
    msgs = [{"role": "system", "content": THERAPIST_SYSTEM}]

    # Past session memory
    past = session.get("past_summary_used", "")
    if past:
        msgs.append({
            "role": "system",
            "content": (
                "What you already know about this client from a prior session "
                "(use for continuity — do NOT reference directly):\n\n" + past
            ),
        })

    # Phase hint — tells LLM where it is in the arc
    exchange_count = _count_user_exchanges(db_history)

    if exchange_count <= 2:
        phase_hint = (
            "[PHASE: UNDERSTAND] You are still learning this person's situation. "
            "Reflect what they've shared. Acknowledge their specific emotion. "
            "Ask at most ONE question if you genuinely need more context. "
            "Do NOT give any advice or suggestions yet. Just understand."
        )
    elif exchange_count <= 5:
        phase_hint = (
            "[PHASE: CONNECT] You now have a reasonable picture of their situation. "
            "Connect the threads of what they've shared. Name the pattern you're seeing. "
            "Help them feel deeply understood — not just heard. "
            "Normalise their experience. Reduce questions to only what's truly missing. "
            "You can begin gently hinting at what might help, but hold off on full advice."
        )
    else:
        phase_hint = (
            "[PHASE: ACTION] You understand this person's situation well. "
            "Stop asking questions — it is time to help them move forward. "
            "Give 2–3 concrete, specific, actionable steps tailored EXACTLY to what they described. "
            "Not generic advice. Specific to their words and situation. "
            "For each step, briefly explain WHY it will help them specifically. "
            "End with: 'Which of these feels most realistic to start with?' — just this one question."
        )

    msgs.append({"role": "system", "content": phase_hint})

    # Full conversation history
    for h in db_history:
        role = "assistant" if h["role"] == "assistant" else "user"
        msgs.append({"role": role, "content": h["content"]})

    # Emotion tone calibration (internal — never shown to user)
    emotion_tones = {
        "sad":     "The client sounds sad or low. Be especially slow, gentle, and warm. Let them feel completely heard before anything else.",
        "anger":   "The client sounds frustrated or angry. Acknowledge that directly first — don't minimise or rush past it. Meet them where they are.",
        "excited": "The client sounds anxious or activated. Help them slow down. Ground them before diving into content.",
        "happy":   "The client sounds relatively okay today. You can be slightly lighter in tone while still going deep.",
        "calm":    "The client sounds calm and reflective. You can go deeper and more exploratory.",
    }
    tone_note = emotion_tones.get(emotion, "")
    if tone_note:
        msgs.append({
            "role": "system",
            "content": f"[Internal tone note — never mention this]: {tone_note}",
        })

    return msgs


def _summarise(transcript: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a therapist's case-note writer. "
            "Write a concise 4–6 sentence summary in third person covering: "
            "the client's main presenting concern, their emotional state during the session, "
            "key themes and patterns that emerged, any action steps that were discussed, "
            "and what still needs attention in future sessions. "
            "This summary will be used to seed memory for the next session."
        )},
        {"role": "user", "content": f"Session transcript:\n{transcript}"},
    ]
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"].strip()


def _transcribe(path: str) -> str:
    if not WHISPER_OK:
        return ""
    segments, _ = _whisper.transcribe(
        path, beam_size=5, language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    return " ".join(s.text for s in segments).strip()

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/signup")
def signup(body: AuthReq):
    name = body.name.strip()
    if not name: raise HTTPException(400, "Name cannot be empty")
    if users_col.find_one({"name": name}): raise HTTPException(400, "Username taken")
    uid = hashlib.md5(name.encode()).hexdigest()
    users_col.insert_one({
        "_id": uid, "name": name,
        "password_hash": _hash(body.password),
        "created_at": datetime.utcnow(),
    })
    return {"token": _make_token(uid), "name": name}

@app.post("/api/auth/login")
def login(body: AuthReq):
    user = users_col.find_one({"name": body.name.strip()})
    if not user or user["password_hash"] != _hash(body.password):
        raise HTTPException(401, "Invalid credentials")
    return {"token": _make_token(user["_id"]), "name": user["name"]}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/sessions/start")
def start_session(user=Depends(get_user)):
    uid = user["_id"]
    now = datetime.utcnow()

    past = sessions_col.find_one(
        {"user_id": uid, "status": "completed", "summary": {"$exists": True}},
        sort=[("ended_at", -1)],
    )
    past_summary    = past["summary"] if past else None
    has_past_memory = past_summary is not None

    greeting_msgs = returning_user_greeting_prompt(past_summary) if has_past_memory else NEW_USER_GREETING_PROMPT
    greeting = _llm(greeting_msgs)

    sid = str(uuid.uuid4())
    sessions_col.insert_one({
        "_id": sid, "user_id": uid, "status": "active",
        "started_at": now, "has_past_memory": has_past_memory,
        "past_summary_used": past_summary or "",
    })

    messages_col.insert_one({
        "session_id": sid, "role": "assistant", "content": greeting,
        "chat_generated": greeting, "audio_transcript": None,
        "emotion_detected": "neutral", "timestamp": now, "ts": now,
    })

    return {"session_id": sid, "greeting": greeting, "has_past_memory": has_past_memory}


@app.post("/api/sessions/voice-message")
def voice_message(body: VoiceReq, user=Depends(get_user)):
    session = sessions_col.find_one({"_id": body.session_id, "user_id": user["_id"]})
    if not session:                    raise HTTPException(404, "Session not found")
    if session["status"] != "active": raise HTTPException(400, "Session not active")
    if not WHISPER_OK:                 raise HTTPException(503, "faster-whisper not installed")

    try:
        wav_bytes = base64.b64decode(body.audio_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes); tmp = f.name

    try:
        transcript = _transcribe(tmp)
        acoustic_probs = {k: (1.0/len(EMOTION_KEYS_PY)) for k in EMOTION_KEYS_PY}
        if LIBROSA_OK:
            try:
                import librosa as _librosa, numpy as _np
                audio_np, _ = _librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
                acoustic_probs = _pipeline_acoustic(audio_np)
            except Exception as e:
                print(f"[Pipeline A] error: {e}")
    finally:
        try: os.remove(tmp)
        except: pass

    if not transcript:
        raise HTTPException(422, "No speech detected")

    now = datetime.utcnow()

    semantic_probs = {k: (1.0/len(EMOTION_KEYS_PY)) for k in EMOTION_KEYS_PY}
    try:
        semantic_probs = _pipeline_semantic_text(transcript)
    except Exception as e:
        print(f"[Pipeline B] error: {e}")

    has_speech = bool(transcript.strip())
    detected_emotion, _ = _fuse(acoustic_probs, semantic_probs, has_speech)
    print(f"[Fusion] → FUSED={detected_emotion.upper()}")

    messages_col.insert_one({
        "session_id": body.session_id, "role": "user", "content": transcript,
        "chat_generated": None, "audio_transcript": transcript,
        "emotion_detected": detected_emotion, "timestamp": now, "ts": now,
    })

    history  = list(messages_col.find({"session_id": body.session_id}, sort=[("ts", 1)]))
    llm_msgs = _build_conversation_messages(session, history, detected_emotion)
    reply    = _llm(llm_msgs)

    now2 = datetime.utcnow()
    messages_col.insert_one({
        "session_id": body.session_id, "role": "assistant", "content": reply,
        "chat_generated": reply, "audio_transcript": None,
        "emotion_detected": detected_emotion, "timestamp": now2, "ts": now2,
    })

    return {"transcript": transcript, "reply": reply, "emotion": detected_emotion}


@app.post("/api/sessions/end")
def end_session(body: EndReq, user=Depends(get_user)):
    session = sessions_col.find_one({"_id": body.session_id, "user_id": user["_id"]})
    if not session: raise HTTPException(404, "Session not found")

    now = datetime.utcnow()

    if body.chat_history and len(body.chat_history) > 0:
        lines = [f"{'Client' if m.role == 'user' else 'Therapist'}: {m.content}"
                 for m in body.chat_history]
        transcript = "\n".join(lines)
    else:
        history = list(messages_col.find({"session_id": body.session_id}, sort=[("ts", 1)]))
        transcript = "\n".join(
            f"{'Client' if h['role'] == 'user' else 'Therapist'}: {h['content']}"
            for h in history
        )

    try:
        summary = _summarise(transcript) if transcript.strip() else "Empty session."
    except Exception as e:
        summary = f"Session ended. Summary error: {e}"

    msg_count = messages_col.count_documents({"session_id": body.session_id})

    sessions_col.update_one(
        {"_id": body.session_id},
        {"$set": {
            "status": "completed", "ended_at": now, "summary": summary,
            "full_transcript": transcript, "message_count": msg_count,
        }},
    )
    return {"summary": summary}


@app.get("/api/sessions/history")
def session_history(user=Depends(get_user)):
    rows = list(sessions_col.find({"user_id": user["_id"]}, sort=[("started_at", -1)], limit=20))
    return [
        {
            "session_id":      r["_id"],
            "status":          r.get("status", "unknown"),
            "started_at":      r["started_at"].isoformat() if r.get("started_at") else "",
            "ended_at":        r["ended_at"].isoformat()   if r.get("ended_at")   else "",
            "message_count":   r.get("message_count", 0),
            "summary":         r.get("summary", ""),
            "has_past_memory": r.get("has_past_memory", False),
        }
        for r in rows
    ]


@app.post("/api/sessions/detect-emotion")
def detect_emotion(body: EmotionTextReq):
    text = body.text.strip()
    if not text:
        flat = 1.0 / len(EMOTION_KEYS_PY)
        return {"emotion": "neutral", "probs": {k: flat for k in EMOTION_KEYS_PY}}
    try:
        sp = _pipeline_semantic_text(text)
        return {"emotion": max(sp, key=sp.get), "probs": sp}
    except Exception as e:
        print(f"[detect-emotion] error: {e}")
        flat = 1.0 / len(EMOTION_KEYS_PY)
        return {"emotion": "neutral", "probs": {k: flat for k in EMOTION_KEYS_PY}}


@app.post("/api/sessions/acoustic-emotion")
def acoustic_emotion(body: AcousticChunkReq):
    """Pipeline A — librosa on a live chunk. Hot path, no auth needed."""
    if not LIBROSA_OK:
        flat = 1.0 / len(EMOTION_KEYS_PY)
        return {"emotion": "neutral", "probs": {k: flat for k in EMOTION_KEYS_PY}}
    try:
        wav_bytes = base64.b64decode(body.audio_b64)
        import librosa as _librosa, io
        audio_np, _ = _librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
        probs   = _pipeline_acoustic(audio_np)
        emotion = max(probs, key=probs.get)
        return {"emotion": emotion, "probs": probs}
    except Exception as e:
        print(f"[acoustic-emotion] error: {e}")
        flat = 1.0 / len(EMOTION_KEYS_PY)
        return {"emotion": "neutral", "probs": {k: flat for k in EMOTION_KEYS_PY}}


@app.get("/")
def root(): return {"status": "Serene AI Therapist API running"}