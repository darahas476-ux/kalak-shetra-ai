# ═══════════════════════════════════════════════════════════════════════════
#  KALAK SHETRA — AI BACKEND
#  • Background removal  → remove.bg API
#  • Speech-to-Text      → Sarvam REST (Saarika v2)
#  • Text-to-Speech      → Sarvam REST (Bulbul v3)
#  • Translation         → Google Translate (free)
#  • Chat assistant      → xAI Grok (official xai-sdk)
# ═══════════════════════════════════════════════════════════════════════════

import os, io, base64, requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

# ── Official xAI SDK ──
from xai_sdk import Client
from xai_sdk.chat import system, user, assistant

app = FastAPI(title="Kalak Shetra AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REMOVEBG_KEY = os.getenv("REMOVEBG_API_KEY", "")
SARVAM_KEY   = os.getenv("SARVAM_API_KEY", "")
XAI_KEY      = os.getenv("XAI_API_KEY", "")

# ── Grok model ──
GROK_MODEL = "grok-4.6"

# ── Sarvam Bulbul v3 speakers ──
# bulbul:v2 was deprecated. v3 speakers: priya, ishita, shubh, aditya, etc.
SARVAM_SPEAKERS = ["priya", "ishita", "shubh", "aditya"]
SARVAM_TTS_MODEL = "bulbul:v3"

# ── Initialize Grok client ──
grok_client = None
if XAI_KEY:
    grok_client = Client(api_key=XAI_KEY)


@app.get("/")
def root():
    return {
        "status": "ok",
        "removebg_configured": bool(REMOVEBG_KEY),
        "sarvam_configured": bool(SARVAM_KEY),
        "grok_configured": bool(XAI_KEY),
        "grok_model": GROK_MODEL,
        "grok_sdk": "xai-sdk (official)",
        "sarvam_tts_model": SARVAM_TTS_MODEL,
    }


# ─────────────────────────────────────────────────────────────
#  ENHANCE — remove.bg
# ─────────────────────────────────────────────────────────────
@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    if not REMOVEBG_KEY:
        return JSONResponse({"error": "REMOVEBG_API_KEY not set"}, status_code=503)
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
                {"error": f"remove.bg {resp.status_code}: {resp.text[:300]}"},
                status_code=500,
            )
        return {"image": base64.b64encode(resp.content).decode()}
    except Exception as e:
        return JSONResponse({"error": f"enhance failed: {e}"}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  TRANSLATE
# ─────────────────────────────────────────────────────────────
LANG = {"en":"en","en_US":"en","hi":"hi","hi_IN":"hi",
        "te":"te","te_IN":"te","auto":"auto"}

@app.post("/translate")
async def translate(text: str = Form(...), source: str = Form("auto"),
                    target: str = Form("en")):
    try:
        s = LANG.get(source, "auto")
        t = LANG.get(target, "en")
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client":"gtx","sl":s,"tl":t,"dt":"t","q":text}
        r = requests.get(url, params=params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        translated = "".join(seg[0] for seg in data[0] if seg and seg[0])
        return {"translated": translated}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  SPEECH-TO-TEXT — Sarvam Saarika v2
# ─────────────────────────────────────────────────────────────
@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...),
                         language: str = Form("te-IN")):
    if not SARVAM_KEY:
        return JSONResponse({"error": "SARVAM_API_KEY not set"}, status_code=503)
    try:
        audio_bytes = await file.read()
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "saarika:v2", "language_code": language}
        headers = {"api-subscription-key": SARVAM_KEY}

        r = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            files=files, data=data, headers=headers, timeout=60,
        )
        if r.status_code != 200:
            return JSONResponse(
                {"error": f"Sarvam STT {r.status_code}: {r.text[:400]}"},
                status_code=500,
            )
        return {"transcript": r.json().get("transcript", "")}
    except Exception as e:
        return JSONResponse({"error": f"STT failed: {e}"}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  TEXT-TO-SPEECH — Sarvam Bulbul v3
# ─────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    language: str = "te-IN"
    speaker: str = ""

def _sarvam_tts_call(text: str, lang: str, speaker: str) -> dict:
    try:
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            json={
                "text": text,
                "target_language_code": lang,
                "speaker": speaker,
                "model": SARVAM_TTS_MODEL,
            },
            headers={
                "api-subscription-key": SARVAM_KEY,
                "Content-Type": "application/json",
            },
            timeout=45,
        )
        if r.status_code != 200:
            return {"ok": False, "error": f"{r.status_code}: {r.text[:200]}"}
        audios = r.json().get("audios", [])
        if not audios:
            return {"ok": False, "error": "no audio in response"}
        return {"ok": True, "audio": audios[0], "speaker": speaker}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/text-to-speech")
async def text_to_speech(req: TTSRequest):
    if not SARVAM_KEY:
        return JSONResponse({"error": "SARVAM_API_KEY not set"}, status_code=503)

    candidates = [req.speaker] if req.speaker else []
    candidates += [s for s in SARVAM_SPEAKERS if s not in candidates]

    last_error = None
    for spk in candidates:
        result = _sarvam_tts_call(req.text, req.language, spk)
        if result["ok"]:
            return {"audio": result["audio"], "format": "wav", "speaker": result["speaker"]}
        last_error = result["error"]

    return JSONResponse(
        {"error": f"All speakers failed. Last: {last_error}"},
        status_code=500,
    )


# ─────────────────────────────────────────────────────────────
#  CHAT — Grok via official xai-sdk
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

SYSTEM_PROMPT = (
    "You are Kalak Shetra's helpful seller assistant for Indian artisans. "
    "Help with pricing, product descriptions, festival tips, business guidance. "
    "Reply in the same language the user speaks (Telugu, Hindi, or English). "
    "Keep answers short, practical, and friendly."
)

@app.post("/chat")
async def chat(req: ChatRequest):
    if grok_client is None:
        return JSONResponse({"error": "XAI_API_KEY not set"}, status_code=503)
    try:
        # Build the chat using the official xai-sdk
        chat = grok_client.chat.create(
            model=GROK_MODEL,
            messages=[system(SYSTEM_PROMPT)],
        )
        for m in req.messages:
            if m.role == "user":
                chat.append(user(m.content))
            elif m.role == "assistant":
                chat.append(assistant(m.content))

        response = chat.sample()
        return {"reply": response.content, "model": GROK_MODEL}
    except Exception as e:
        return JSONResponse({"error": f"Grok failed: {e}"}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  DEBUG
# ─────────────────────────────────────────────────────────────
@app.get("/debug")
def debug():
    results = {"grok_sdk": "xai-sdk (official)"}

    # Grok test using official SDK
    if grok_client:
        try:
            chat = grok_client.chat.create(
                model=GROK_MODEL,
                messages=[system("You reply tersely.")],
            )
            chat.append(user("Say OK"))
            response = chat.sample()
            results["grok"] = {"ok": True, "reply": response.content, "model": GROK_MODEL}
        except Exception as e:
            results["grok"] = {"ok": False, "error": str(e)[:400]}
    else:
        results["grok"] = {"ok": False, "error": "XAI_API_KEY not configured"}

    # Sarvam TTS test
    if SARVAM_KEY:
        tts_results = []
        for spk in SARVAM_SPEAKERS:
            r = _sarvam_tts_call("నమస్కారం", "te-IN", spk)
            tts_results.append({"speaker": spk, "ok": r["ok"],
                                "error": r.get("error")})
        results["sarvam_tts"] = tts_results
    else:
        results["sarvam_tts"] = {"ok": False, "error": "SARVAM_API_KEY not configured"}

    results["removebg"] = {"configured": bool(REMOVEBG_KEY)}

    return results