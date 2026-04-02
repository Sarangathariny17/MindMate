"""
AI Therapist Backend — FastAPI + MongoDB + Ollama
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

    print(f"[Pipeline A] rms={rms:.4f} -> {max(probs, key=probs.get).upper()}")
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

    print(f"[Pipeline B] \"{text[:60]}\" -> {max(sp, key=sp.get).upper()}")
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
LLM_MODEL   = os.getenv("LLM_MODEL", "llava:7b")

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

# =============================================================================
# AUTH HELPERS
# =============================================================================
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

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
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

# =============================================================================
# THERAPIST PROMPTS
# =============================================================================
#
# SESSION ARC — 10 user exchanges, then the session closes with a personalised
# action plan. The model blends its curiosity naturally into each response
# rather than firing questions as a checklist.
#
# What the model learns across the 10 exchanges:
#   Turn 1  — Overall feeling right now
#   Turn 2  — The main thing weighing on them
#   Turn 3  — The situation / context behind it
#   Turn 4  — Emotional impact day-to-day
#   Turn 5  — Whether this is recurring or new
#   Turn 6  — The core thought / belief driving the feeling
#   Turn 7  — What they've already tried or been avoiding
#   Turn 8  — What a better state looks and feels like
#   Turn 9  — What's been blocking them from getting there
#   Turn 10 — The smallest realistic shift they can imagine today
#   Close   — 2-3 specific, personalised action steps + warm close
#

THERAPIST_SYSTEM = """You are a warm, skilled therapist having a focused one-on-one session.

HARD RULES — never break these:
- Never send an incomplete message. Always finish your thought.
- Keep every response to 2-4 lines maximum. Short, human, natural.
- Ask only ONE question per message. Never stack two questions.
- Never explain why you are asking something.
- Never use bullet points, numbered lists, or headers in your replies.
- Never sound robotic, clinical, or overly formal.
- Always validate before you ask — acknowledge what the person just said before moving forward.
- Do NOT give advice or action steps until the ACTION phase (exchange 10+).
- Never repeat a question already asked in this conversation.
- Always complete your sentence before stopping — never cut off mid-thought.

YOUR GOAL:
Across exactly 10 exchanges, naturally understand the person's situation well enough
to give them specific, personalised action steps at the end.
Each response should feel like a natural continuation of a real conversation,
not a scripted checklist. Weave your curiosity into genuine reactions to what they share.

TONE:
- Calm, warm, non-judgmental
- Sounds like a real person who genuinely cares
- Never dismisses or minimises feelings
- Natural phrases: "That makes sense", "I hear you", "That sounds really hard",
  "I'm glad you shared that", "That takes courage to sit with"
"""


# ── Greeting prompts ───────────────────────────────────────────────────────────

NEW_USER_GREETING_PROMPT = [
    {"role": "system", "content": THERAPIST_SYSTEM},
    {
        "role": "user",
        "content": (
            "A new client has just arrived for their first session. "
            "Open with exactly this greeting and nothing else: "
            "\"Hi, it's good to see you. How have you been feeling lately?\""
        ),
    },
]


def returning_user_greeting_prompt(past_summary: str) -> list:
    return [
        {"role": "system", "content": THERAPIST_SYSTEM},
        {
            "role": "system",
            "content": (
                "This client has seen you before. Here is what you know from the last session. "
                "Let it inform your warmth and awareness — do NOT quote it or say "
                "'last session' or 'last time'. Just let it shape how you see them:\n\n"
                + past_summary
            ),
        },
        {
            "role": "user",
            "content": (
                "The client has returned. Greet them like someone you genuinely know and care about. "
                "In 2-3 sentences, gently acknowledge where they were and ask one warm, specific "
                "question that shows you remember them. Do not quote your notes or say 'last session'."
            ),
        },
    ]


# ── LLM call ──────────────────────────────────────────────────────────────────

def _llm(messages: list) -> str:
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def _count_user_exchanges(db_history: list) -> int:
    """Count how many user turns have happened in this session."""
    return sum(1 for h in db_history if h.get("role") == "user")


# ── Per-turn phase hints ───────────────────────────────────────────────────────

def _get_phase_hint(exchange_count: int) -> str:
    """
    Injected as an internal system message each turn. Tells the model exactly
    where it is in the arc and what gap to fill — without the user ever seeing it.
    """
    hints = {
        1: (
            "[TURN 1] The client just responded to your opening. "
            "Acknowledge what they said warmly in one sentence. "
            "Then invite them to share the main thing that's been on their mind. "
            "One question only. No advice."
        ),
        2: (
            "[TURN 2] You know their general mood. "
            "Validate what they shared. Then ask about the specific situation or context "
            "behind what they're going through — what's actually been happening in their life? "
            "One question only. No advice."
        ),
        3: (
            "[TURN 3] You understand the situation now. "
            "Reflect back what they described so they feel heard. "
            "Then gently ask how this has been affecting them emotionally day-to-day — "
            "how does it show up in how they feel or function? One question only. No advice."
        ),
        4: (
            "[TURN 4] You understand the emotional weight they're carrying. "
            "Name it specifically — don't be vague. "
            "Then ask whether this feels like something new or a pattern they've been through before. "
            "One question only. No advice."
        ),
        5: (
            "[TURN 5] You're getting a sense of whether this is a recurring pattern. "
            "Acknowledge that gently. "
            "Then ask about the thought or belief their mind lands on when they're in this feeling — "
            "what does their inner voice say is happening or wrong with them or the situation? "
            "One question only. No advice."
        ),
        6: (
            "[TURN 6] You've heard the thought pattern. "
            "Name it without judgment and normalise it — this is something many people experience. "
            "Then ask what they've already tried to deal with this, or what they've been avoiding. "
            "One question only. No advice."
        ),
        7: (
            "[TURN 7] You know what they've tried or avoided. "
            "Validate that — it makes sense given where they are. "
            "Then ask what things would actually look and feel like if this shifted — "
            "paint it in their terms: what would a better state mean for their day-to-day life? "
            "One question only. No advice."
        ),
        8: (
            "[TURN 8] You understand where they want to be. "
            "Reflect that vision back so they feel it's real and worth moving toward. "
            "Then ask what they feel has been the main thing getting in the way — "
            "the block, the fear, or what keeps pulling them back. One question only. No advice."
        ),
        9: (
            "[TURN 9] You understand their block. "
            "Acknowledge it — it's real and it makes sense that it's hard. "
            "This is your final question before you move into helping them act. "
            "Ask: given everything they've shared today, what feels like the smallest "
            "possible shift that could actually make a difference for them right now? "
            "One question only. After their answer, you will give them their action plan."
        ),
    }

    if exchange_count <= 9:
        return hints.get(exchange_count, hints[1])

    # Turn 10+ — close with a personalised action plan
    return (
        "[TURN 10+ — ACTION PLAN] You now understand this person's situation deeply. "
        "Stop asking questions entirely. "
        "Deliver 2-3 action steps that are SPECIFIC to everything they've shared — "
        "not generic advice, but steps rooted in their exact words, their situation, "
        "their pattern, and their desired state. "
        "For each step, explain in one sentence why it will specifically help THEM. "
        "Write this as warm, natural prose — no bullet points, no numbered lists, no headers. "
        "End with a single closing sentence that affirms their capacity to move through this. "
        "Do not ask any more questions. This is the close of the session."
    )


def _build_conversation_messages(session: dict, db_history: list, emotion: str) -> list:
    msgs = [{"role": "system", "content": THERAPIST_SYSTEM}]

    # Past session memory (silent continuity)
    past = session.get("past_summary_used", "")
    if past:
        msgs.append({
            "role": "system",
            "content": (
                "What you already know about this client from a prior session "
                "(use for continuity — do NOT reference directly):\n\n" + past
            ),
        })

    # Phase hint — tells the model what to do this turn
    exchange_count = _count_user_exchanges(db_history)
    msgs.append({"role": "system", "content": _get_phase_hint(exchange_count)})

    # Full conversation history
    for h in db_history:
        role = "assistant" if h["role"] == "assistant" else "user"
        msgs.append({"role": role, "content": h["content"]})

    # Emotion tone calibration — internal, never shown to user
    emotion_tones = {
        "sad": (
            "[Tone] The client sounds sad or low right now. "
            "Be slow, gentle, and especially warm. Give them space. "
            "Let them feel fully heard before anything else."
        ),
        "anger": (
            "[Tone] The client sounds frustrated or angry. "
            "Acknowledge that directly and without flinching — do not minimise or skip past it. "
            "Meet them exactly where they are."
        ),
        "excited": (
            "[Tone] The client sounds anxious or activated. "
            "Help them slow down with your pace. Ground them gently before going deeper."
        ),
        "happy": (
            "[Tone] The client sounds relatively okay today. "
            "Match their lighter energy while still going beneath the surface."
        ),
        "calm": (
            "[Tone] The client sounds calm and reflective. "
            "You can go deeper and more exploratory — they're ready for it."
        ),
    }
    tone_note = emotion_tones.get(emotion, "")
    if tone_note:
        msgs.append({"role": "system", "content": tone_note})

    return msgs


def _summarise(transcript: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a compassionate therapist writing private case notes after a session. "
                "Write a focused 4-6 sentence summary in the third person. Cover: "
                "the client's main presenting concern, their emotional state during the session, "
                "the core thought pattern or belief that emerged, "
                "what they have tried or been avoiding, "
                "what their desired state looks like, "
                "the action steps that were recommended, "
                "and what still needs gentle attention in future sessions. "
                "Be specific to this person's exact words and situation — never write generic notes. "
                "This summary will be used to personalise their next session."
            ),
        },
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

# =============================================================================
# AUTH ENDPOINTS
# =============================================================================
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

# =============================================================================
# SESSION ENDPOINTS
# =============================================================================
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
    print(f"[Fusion] -> FUSED={detected_emotion.upper()}")

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