# ═══════════════════════════════════════════════════════════════════════════
#  KALAK SHETRA — AI BACKEND (API-powered)
#  • Background removal  → remove.bg API
#  • Speech-to-Text      → Sarvam AI (Saaras v3)
#  • Text-to-Speech      → Sarvam AI (Bulbul v3)
#  • Translation         → Google Translate (free)
#  • Chat assistant      → Grok (xAI)
#  No local AI models → fits easily in Render free tier (512 MB)
# ═══════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import io
import base64
import requests
from openai import OpenAI

# ── Sarvam AI SDK ──
try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None


app = FastAPI(title="Kalak Shetra AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys (set these in Render → Environment) ──
REMOVEBG_KEY    = os.getenv("REMOVEBG_API_KEY", "")
SARVAM_KEY      = os.getenv("SARVAM_API_KEY", "")
XAI_KEY         = os.getenv("XAI_API_KEY", "")

# ── Initialize clients (lazy, so missing keys don't crash boot) ──
grok = None
if XAI_KEY:
    grok = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")

sarvam = None
if SARVAM_KEY and SarvamAI is not None:
    sarvam = SarvamAI(api_subscription_key=SARVAM_KEY)


# ─────────────────────────────────────────────────────────────
#  ROOT — health check
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "removebg_configured": bool(REMOVEBG_KEY),
        "sarvam_configured": bool(SARVAM_KEY),
        "grok_configured": bool(XAI_KEY),
        "note": "All AI services are external APIs. No local models.",
    }


# ─────────────────────────────────────────────────────────────
#  /enhance — background removal via remove.bg
# ─────────────────────────────────────────────────────────────
@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    if not REMOVEBG_KEY:
        return JSONResponse(
            {"error": "REMOVEBG_API_KEY not set"}, status_code=503)

    try:
        raw = await file.read()
        resp = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": ("image.jpg", raw, "image/jpeg")},
            data={"size": "auto", "format": "jpg", "bg_color": "ffffff"},
            headers={"X-Api-Key": REMOVEBG_KEY},
            timeout=60,
        )
        if resp.status_code != 200:
            return JSONResponse(
                {"error": f"remove.bg {resp.status_code}: {resp.text[:200]}"},
                status_code=resp.status_code,
            )

        b64 = base64.b64encode(resp.content).decode()
        return {"image": b64}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  /translate — Google Translate (free, no key)
# ─────────────────────────────────────────────────────────────
LANG = {"en": "en", "en_US": "en", "hi": "hi", "hi_IN": "hi",
        "te": "te", "te_IN": "te", "auto": "auto"}


@app.post("/translate")
async def translate(text: str = Form(...),
                    source: str = Form("auto"),
                    target: str = Form("en")):
    try:
        s = LANG.get(source, "auto")
        t = LANG.get(target, "en")
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": s, "tl": t, "dt": "t", "q": text}
        r = requests.get(url, params=params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
        return {"translated": translated, "source": s, "target": t}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  /speech-to-text — Sarvam AI (Saaras v3)
#  Accepts audio file → returns transcript
# ─────────────────────────────────────────────────────────────
@app.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form("te-IN"),
):
    if sarvam is None:
        return JSONResponse(
            {"error": "SARVAM_API_KEY not set"}, status_code=503)

    try:
        audio_bytes = await file.read()
        # Sarvam SDK expects a file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        response = sarvam.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            language_code=language,
            mode="transcribe",
        )
        return {"transcript": response.transcript}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  /text-to-speech — Sarvam AI (Bulbul v3)
#  Accepts text → returns base64-encoded audio
# ─────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    language: str = "te-IN"   # 'te-IN' | 'hi-IN' | 'en-IN'
    speaker: str = "anushka"  # Sarvam voice name


@app.post("/text-to-speech")
async def text_to_speech(req: TTSRequest):
    if sarvam is None:
        return JSONResponse(
            {"error": "SARVAM_API_KEY not set"}, status_code=503)

    try:
        response = sarvam.text_to_speech.convert(
            text=req.text,
            model="bulbul:v3",
            target_language_code=req.language,
            speaker=req.speaker,
        )
        # Sarvam returns audios: [base64_string, ...]
        audio_b64 = response.audios[0] if response.audios else None
        if not audio_b64:
            return JSONResponse(
                {"error": "No audio returned"}, status_code=500)
        return {"audio": audio_b64, "format": "wav"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  /chat — Grok assistant
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


SYSTEM_PROMPT = (
    "You are Kalak Shetra's helpful seller assistant for Indian artisans. "
    "You help weavers, potters, and craftspeople with pricing advice, "
    "product descriptions, festival selling tips, and general business guidance. "
    "Reply in the same language the user speaks (Telugu, Hindi, or English). "
    "Keep answers short, practical, and friendly."
)


@app.post("/chat")
async def chat(req: ChatRequest):
    if grok is None:
        return JSONResponse(
            {"error": "XAI_API_KEY not set"}, status_code=503)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        msgs.append({"role": m.role, "content": m.content})

    try:
        resp = grok.chat.completions.create(
            model="grok-3-mini-fast",
            messages=msgs,
            temperature=0.7,
            max_tokens=400,
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)