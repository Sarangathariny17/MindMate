"""
Serene AI Therapist Backend — FastAPI + MongoDB + Ollama
=========================================================
POST /api/auth/signup              — register + create personal profile
POST /api/auth/login               — login → JWT
POST /api/sessions/start           — start session (new or returning)
POST /api/sessions/voice-message   — WAV → Whisper → LLM reply
POST /api/sessions/text-message    — plain text → LLM reply  (new)
POST /api/sessions/end             — summarise + save
GET  /api/sessions/history         — list past sessions
POST /api/sessions/detect-emotion  — text → Hartmann NLP → emotion probs
POST /api/sessions/acoustic-emotion— WAV chunk → librosa → emotion probs
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
    roll = float(np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=SAMPLE_RATE, roll_percent=0.85)))
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

    return max(fused, key=fused.get), fused

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
profiles_col = _db["profiles"]   # ← personal profile space per user

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
# PERSONAL PROFILE — created on signup, updated after every session
# ═══════════════════════════════════════════════════════════════════════════════
def _get_profile(uid: str) -> dict:
    p = profiles_col.find_one({"_id": uid})
    if not p:
        p = {
            "_id": uid,
            "name": "",
            "sessions_completed": 0,
            "dominant_emotions": [],
            "recurring_themes": [],
            "past_remedies": [],
            "life_context": {},      # age, job, relationships — gathered over time
            "last_updated": None,
        }
        profiles_col.insert_one(p)
    return p

def _update_profile(uid: str, patch: dict):
    profiles_col.update_one({"_id": uid}, {"$set": patch}, upsert=True)

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

class TextReq(BaseModel):
    session_id: str
    text:       str
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
# THE THERAPY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# 10-turn arc. Each turn has a HIDDEN GOAL stored here.
# The LLM only sees the goal, not the label — so output stays human.
#
# Turns 1-9  → one question per turn, ≤100 words, warm + curious
# Turn 10    → NO question. Full remedy woven as natural conversation.
#
# After each user turn we also extract a structured "fact" and save it
# to MongoDB (profile + session). This drives adaptive follow-up questions.
#
# ── What each turn must accomplish — injected as a hard instruction ──────────
# The model sees the full conversation history + these instructions.
# Each turn label tells the model EXACTLY what it has already learned
# and what NEW ground to cover — preventing any repetition.
#
# ── Master system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Serene — a warm, psychologically-informed companion. You are having a real, flowing conversation with one person. This is NOT a scripted intake form. You are a supportive friend who understands psychology deeply.

YOUR CORE JOB:
- Listen carefully to everything the person says
- Respond directly and specifically to what they just said — never ignore it
- If they answer a question, acknowledge their answer before anything else
- Gradually move from understanding → support → concrete advice as the conversation develops
- By message 3-4, start weaving in practical suggestions and coping strategies
- By message 5+, be primarily supportive and advisory — give real, actionable help

CONVERSATION FLOW (adapt naturally, don't follow rigidly):
- Early (messages 1-2): Understand what's going on. Ask one gentle clarifying question if needed.
- Middle (messages 3-4): You have enough context. Acknowledge their situation deeply, start offering perspective and 1-2 concrete strategies.
- Later (messages 5+): Be their supportive advisor. Give specific, practical psychological techniques. Answer their questions directly. If they ask "why not now?" — do it NOW. If they say "help me" — help them immediately.

RESPONSE RULES:
1. MAXIMUM 80 WORDS. Hard limit. Every single response.
2. Output ONLY natural spoken words. No bullet points, no numbered lists, no headers, no asterisks, no labels.
3. Always respond to what they JUST said first. Never skip past their message.
4. If they ask a direct question — answer it directly before anything else.
5. If they want to try something now — guide them through it now.
6. Contractions. Short sentences. Warm but direct.
7. Sound like a caring friend who happens to know psychology — not a therapist running a session.

ADVICE QUALITY:
- Give REAL, NAMED psychological techniques: CBT thought records, box breathing, 5-4-3-2-1 grounding, behavioural activation, cognitive defusion, progressive muscle relaxation, etc.
- Tie every technique to something they specifically said
- Make it actionable TODAY, not someday
- If guiding an exercise — guide it step by step in plain language

BANNED: bullet points, numbered lists, headers, "validate", "reframe", "unpack", "hold space", "journey", "empower", "closure", "narrative", therapy-speak of any kind.

CONTINUITY: This is ONE ongoing conversation. Remember everything they said. Use their exact words back to them. Never ask about something they already told you."""


def _build_messages(session: dict, db_history: list, profile: dict, emotion: str, turn_num: int) -> list:
    """Build the full adaptive prompt for this turn."""

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Profile / cross-session memory ────────────────────────────────────────
    profile_lines = []
    if profile.get("recurring_themes"):
        profile_lines.append("Themes from past sessions: " + ", ".join(profile["recurring_themes"]))
    if profile.get("life_context"):
        ctx = profile["life_context"]
        bits = [f"{k}: {v}" for k, v in ctx.items() if v]
        if bits:
            profile_lines.append("Known context: " + ", ".join(bits))
    if profile_lines:
        msgs.append({
            "role": "system",
            "content": "BACKGROUND (never mention directly — just let it inform your warmth):\n" + "\n".join(profile_lines)
        })

    # ── Past session memory ───────────────────────────────────────────────────
    past = session.get("past_summary_used", "")
    if past:
        msgs.append({
            "role": "system",
            "content": "From a previous conversation (don't reference it directly):\n" + past
        })

    # ── Conversation stage guidance ───────────────────────────────────────────
    if turn_num <= 2:
        stage = (
            "STAGE: Early conversation. You're still learning what's going on. "
            "Respond warmly to what they said, then ask ONE gentle clarifying question if you need more context. "
            "If they've already told you the core issue clearly, skip the question and show you understood."
        )
    elif turn_num <= 4:
        stage = (
            "STAGE: You understand their situation now. Shift toward support and advice. "
            "Acknowledge what they're going through with genuine empathy, then offer 1-2 specific, practical suggestions. "
            "Ask a follow-up only if truly necessary. Prioritise being helpful over gathering more information."
        )
    else:
        stage = (
            "STAGE: Ongoing support. Be their advisor. Give concrete psychological techniques tied to what they said. "
            "Answer their questions directly and immediately. If they want to try an exercise — do it right now, guide them through it. "
            "No more information-gathering questions unless they bring up something completely new."
        )

    msgs.append({"role": "system", "content": stage})

    # ── Emotion calibration ───────────────────────────────────────────────────
    tone_map = {
        "sad":     "They sound low and heavy. Be slow, gentle, close. Don't rush to fix — but do offer real help.",
        "anger":   "They sound frustrated. Acknowledge the frustration first, then help.",
        "excited": "They sound anxious or overwhelmed. Be calm and steady. Help them slow down.",
        "happy":   "They seem okay. Keep it light but still go deep and useful.",
        "calm":    "They're reflective. Match that tone — go thoughtful and substantive.",
    }
    tone = tone_map.get(emotion, "")
    if tone:
        msgs.append({"role": "system", "content": f"[Tone — never mention this]: {tone}"})

    # ── Full conversation history ─────────────────────────────────────────────
    for h in db_history:
        msgs.append({
            "role": "assistant" if h["role"] == "assistant" else "user",
            "content": h["content"]
        })

    # ── Final output instruction ──────────────────────────────────────────────
    msgs.append({
        "role": "system",
        "content": (
            "Now write your response. "
            "Start by directly addressing what they just said. "
            "Be warm, specific, and genuinely useful. "
            "Maximum 80 words. No bullet points. No lists. Pure natural conversation."
        )
    })

    return msgs


def _extract_facts_prompt(conversation_so_far: list) -> str:
    """
    Ask the LLM to extract structured facts from the conversation
    and return them as a short JSON blob for MongoDB profile storage.
    """
    history_text = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Serene'}: {h['content']}"
        for h in conversation_so_far
    )
    return (
        "Read this therapy conversation and extract any facts mentioned by the user. "
        "Return ONLY a compact JSON object with these keys (omit if not mentioned): "
        "age, occupation, relationship_status, living_situation, main_concern, "
        "duration, daily_impact, support_network, coping_tried, root_cause_guess, "
        "good_day_description, small_step.\n\n"
        f"Conversation:\n{history_text}\n\n"
        "Return only valid JSON. No explanation."
    )


def _extract_and_save_facts(uid: str, session_id: str, db_history: list):
    """Run fact extraction and persist to profile.life_context."""
    try:
        prompt = _extract_facts_prompt(db_history)
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise JSON extractor. Output only valid JSON."},
                {"role": "user",   "content": prompt}
            ]
        )
        raw = response["message"]["content"].strip()
        import json, re
        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        facts = json.loads(raw)
        if isinstance(facts, dict):
            patch = {f"life_context.{k}": v for k, v in facts.items() if v}
            patch["last_updated"] = datetime.utcnow()
            _update_profile(uid, patch)
            print(f"[Profile] Facts saved for {uid}: {list(facts.keys())}")
    except Exception as e:
        print(f"[Profile] Fact extraction failed: {e}")


def _summarise(transcript: str, profile: dict) -> str:
    """Generate session summary + extract recurring themes for the profile."""
    messages = [
        {"role": "system", "content": (
            "You are a therapist's case-note writer. "
            "Write a concise 4-6 sentence summary in third person covering: "
            "the client's main presenting concern, emotional state, key themes, "
            "any action steps discussed, and what still needs attention. "
            "This will seed memory for the next session. Be specific — use the client's own words."
        )},
        {"role": "user", "content": f"Session transcript:\n{transcript}"},
    ]
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"].strip()


def _extract_themes(transcript: str) -> list:
    """Pull out 2-4 recurring themes as short strings for the profile."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Return only a JSON array of 2-4 short theme strings. No explanation."},
                {"role": "user", "content": f"Extract the main emotional/life themes from this session:\n{transcript}"}
            ]
        )
        import json, re
        raw = re.sub(r"```json|```", "", response["message"]["content"].strip()).strip()
        themes = json.loads(raw)
        if isinstance(themes, list):
            return [str(t) for t in themes[:4]]
    except Exception as e:
        print(f"[Themes] extraction failed: {e}")
    return []


def _llm(messages: list) -> str:
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        text = response["message"]["content"].strip()
        # Hard-trim to ~80 words as a safety net
        words = text.split()
        if len(words) > 85:
            text = " ".join(words[:80])
            # Try to end on a complete sentence
            for punct in ['.', '?', '!']:
                idx = text.rfind(punct)
                if idx > len(text) * 0.6:
                    text = text[:idx+1]
                    break
        return text
    except Exception as e:
        return f"[LLM error: {e}]"


def _transcribe(path: str) -> str:
    if not WHISPER_OK:
        return ""
    segments, _ = _whisper.transcribe(
        path, beam_size=5, language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    return " ".join(s.text for s in segments).strip()


def _count_user_turns(db_history: list) -> int:
    return sum(1 for h in db_history if h.get("role") == "user")


# ── Greeting prompts ──────────────────────────────────────────────────────────
NEW_USER_GREETING = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": (
        "Generate the very first message to a brand new client. "
        "Warm, human, no therapy-speak. Tell them this is just a conversation — "
        "no pressure, no judgment. End with one open question: what's been on their mind? "
        "Under 60 words. No labels."
    )},
]

def _returning_greeting(past_summary: str, profile: dict) -> list:
    ctx_bits = []
    if profile.get("life_context"):
        ctx = profile["life_context"]
        if ctx.get("main_concern"):
            ctx_bits.append(f"their main concern was: {ctx['main_concern']}")
        if ctx.get("small_step"):
            ctx_bits.append(f"they wanted to try: {ctx['small_step']}")
    ctx_note = (" You know " + "; ".join(ctx_bits) + ".") if ctx_bits else ""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": (
            f"You've spoken with this person before.{ctx_note} "
            f"Here's what you know from last time (don't quote it directly): {past_summary}"
        )},
        {"role": "user", "content": (
            "Generate a warm returning greeting. Let them feel you remember them and genuinely care. "
            "Ask how things have been since you last spoke. Under 60 words. Human. No labels."
        )},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/signup")
def signup(body: AuthReq):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    if users_col.find_one({"name": name}):
        raise HTTPException(400, "Username taken")

    uid = hashlib.md5(name.encode()).hexdigest()
    now = datetime.utcnow()

    users_col.insert_one({
        "_id": uid, "name": name,
        "password_hash": _hash(body.password),
        "created_at": now,
    })

    # ── Create personal profile space ─────────────────────────────────────────
    profiles_col.insert_one({
        "_id": uid,
        "name": name,
        "sessions_completed": 0,
        "dominant_emotions": [],
        "recurring_themes": [],
        "past_remedies": [],
        "life_context": {},
        "created_at": now,
        "last_updated": now,
    })
    print(f"[Signup] New user '{name}' — profile space created.")

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
    uid     = user["_id"]
    now     = datetime.utcnow()
    profile = _get_profile(uid)

    past = sessions_col.find_one(
        {"user_id": uid, "status": "completed", "summary": {"$exists": True}},
        sort=[("ended_at", -1)],
    )
    past_summary    = past["summary"] if past else None
    has_past_memory = past_summary is not None

    if has_past_memory:
        greeting_msgs = _returning_greeting(past_summary, profile)
    else:
        greeting_msgs = NEW_USER_GREETING

    greeting = _llm(greeting_msgs)

    sid = str(uuid.uuid4())
    sessions_col.insert_one({
        "_id": sid, "user_id": uid, "status": "active",
        "started_at": now, "has_past_memory": has_past_memory,
        "past_summary_used": past_summary or "",
        "turn_count": 0,
    })
    messages_col.insert_one({
        "session_id": sid, "role": "assistant", "content": greeting,
        "emotion_detected": "neutral", "ts": now,
    })

    return {"session_id": sid, "greeting": greeting, "has_past_memory": has_past_memory}


def _handle_user_message(session_id: str, user: dict, text: str, emotion: str) -> dict:
    """
    Shared core for both voice and text messages.
    Saves user turn → extracts facts → builds LLM prompt → saves reply.
    """
    uid     = user["_id"]
    session = sessions_col.find_one({"_id": session_id, "user_id": uid})
    if not session:
        raise HTTPException(404, "Session not found")
    if session["status"] != "active":
        raise HTTPException(400, "Session not active")

    now = datetime.utcnow()

    # Save user message
    messages_col.insert_one({
        "session_id": session_id, "role": "user", "content": text,
        "emotion_detected": emotion, "ts": now,
    })

    # Full history for context
    history = list(messages_col.find({"session_id": session_id}, sort=[("ts", 1)]))
    user_turn_count = _count_user_turns(history)

    # Determine turn number (used for conversation stage, not capped)
    turn_num = user_turn_count

    # Extract facts every 3 turns to keep profile updated
    if user_turn_count % 3 == 0:
        _extract_and_save_facts(uid, session_id, history)

    profile = _get_profile(uid)
    llm_msgs = _build_messages(session, history, profile, emotion, turn_num)
    reply    = _llm(llm_msgs)

    # Detect emotion on reply context too
    try:
        semantic_probs = _pipeline_semantic_text(text)
        detected_emotion = max(semantic_probs, key=semantic_probs.get)
    except Exception:
        detected_emotion = emotion

    now2 = datetime.utcnow()
    messages_col.insert_one({
        "session_id": session_id, "role": "assistant", "content": reply,
        "emotion_detected": detected_emotion, "ts": now2,
    })

    # Update session turn count
    sessions_col.update_one({"_id": session_id}, {"$inc": {"turn_count": 1}})

    is_remedy = turn_num >= 4

    return {
        "reply":      reply,
        "emotion":    detected_emotion,
        "turn":       turn_num,
        "is_remedy":  is_remedy,
    }


@app.post("/api/sessions/text-message")
def text_message(body: TextReq, user=Depends(get_user)):
    """Accept plain text input — no audio required."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty")

    # Run semantic emotion detection on the text
    emotion = body.emotion
    try:
        sp = _pipeline_semantic_text(text)
        emotion = max(sp, key=sp.get)
    except Exception:
        pass

    result = _handle_user_message(body.session_id, user, text, emotion)
    result["transcript"] = text
    return result


@app.post("/api/sessions/voice-message")
def voice_message(body: VoiceReq, user=Depends(get_user)):
    if not WHISPER_OK:
        raise HTTPException(503, "faster-whisper not installed")

    try:
        wav_bytes = base64.b64decode(body.audio_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name

    try:
        transcript   = _transcribe(tmp)
        acoustic_probs = {k: (1.0/len(EMOTION_KEYS_PY)) for k in EMOTION_KEYS_PY}
        if LIBROSA_OK:
            try:
                import librosa as _librosa
                audio_np, _ = _librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
                acoustic_probs = _pipeline_acoustic(audio_np)
            except Exception as e:
                print(f"[Pipeline A] error: {e}")
    finally:
        try: os.remove(tmp)
        except: pass

    if not transcript:
        raise HTTPException(422, "No speech detected")

    semantic_probs = {k: (1.0/len(EMOTION_KEYS_PY)) for k in EMOTION_KEYS_PY}
    try:
        semantic_probs = _pipeline_semantic_text(transcript)
    except Exception as e:
        print(f"[Pipeline B] error: {e}")

    detected_emotion, _ = _fuse(acoustic_probs, semantic_probs, bool(transcript.strip()))
    print(f"[Fusion] → {detected_emotion.upper()}")

    result = _handle_user_message(body.session_id, user, transcript, detected_emotion)
    result["transcript"] = transcript
    return result


@app.post("/api/sessions/end")
def end_session(body: EndReq, user=Depends(get_user)):
    uid     = user["_id"]
    session = sessions_col.find_one({"_id": body.session_id, "user_id": uid})
    if not session:
        raise HTTPException(404, "Session not found")

    now = datetime.utcnow()

    if body.chat_history and len(body.chat_history) > 0:
        transcript = "\n".join(
            f"{'Client' if m.role == 'user' else 'Serene'}: {m.content}"
            for m in body.chat_history
        )
    else:
        history = list(messages_col.find({"session_id": body.session_id}, sort=[("ts", 1)]))
        transcript = "\n".join(
            f"{'Client' if h['role'] == 'user' else 'Serene'}: {h['content']}"
            for h in history
        )

    profile = _get_profile(uid)

    try:
        summary = _summarise(transcript, profile) if transcript.strip() else "Empty session."
    except Exception as e:
        summary = f"Session ended. Summary error: {e}"

    # Extract themes and update profile
    themes = _extract_themes(transcript)
    if themes:
        existing = profile.get("recurring_themes", [])
        merged   = list(dict.fromkeys(existing + themes))[:10]  # dedupe, cap at 10
        _update_profile(uid, {
            "recurring_themes":  merged,
            "sessions_completed": profile.get("sessions_completed", 0) + 1,
            "last_updated":       now,
        })

    # Run final full fact extraction
    history_docs = list(messages_col.find({"session_id": body.session_id}, sort=[("ts", 1)]))
    _extract_and_save_facts(uid, body.session_id, history_docs)

    msg_count = messages_col.count_documents({"session_id": body.session_id})
    sessions_col.update_one(
        {"_id": body.session_id},
        {"$set": {
            "status": "completed", "ended_at": now, "summary": summary,
            "full_transcript": transcript, "message_count": msg_count,
        }},
    )

    return {"summary": summary, "themes": themes}


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


@app.get("/api/profile")
def get_profile_endpoint(user=Depends(get_user)):
    """Return the user's personal profile space."""
    p = _get_profile(user["_id"])
    p.pop("_id", None)
    return p


# ── Emotion detection endpoints (unchanged) ───────────────────────────────────
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
def root():
    return {"status": "Serene AI Therapist API running"}