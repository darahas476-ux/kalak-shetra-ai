# ═══════════════════════════════════════════════════════════════════════════
#  KALAK SHETRA — AI BACKEND (REST-only, no SDKs)
# ═══════════════════════════════════════════════════════════════════════════

import os, io, base64, requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from openai import OpenAI

app = FastAPI(title="Kalak Shetra AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REMOVEBG_KEY = os.getenv("REMOVEBG_API_KEY", "")
SARVAM_KEY = os.getenv("SARVAM_API_KEY", "")
XAI_KEY = os.getenv("XAI_API_KEY", "")

grok = None
if XAI_KEY:
    grok = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")


@app.get("/")
def root():
    return {
        "status": "ok",
        "removebg_configured": bool(REMOVEBG_KEY),
        "sarvam_configured": bool(SARVAM_KEY),
        "grok_configured": bool(XAI_KEY),
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
#  TRANSLATE — Google (free)
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
#  SPEECH-TO-TEXT — Sarvam REST API
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
                {"error": f"Sarvam STT {r.status_code}: {r.text[:300]}"},
                status_code=500,
            )
        return {"transcript": r.json().get("transcript", "")}
    except Exception as e:
        return JSONResponse({"error": f"STT failed: {e}"}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  TEXT-TO-SPEECH — Sarvam REST API
# ─────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    language: str = "te-IN"
    speaker: str = "meera"

@app.post("/text-to-speech")
async def text_to_speech(req: TTSRequest):
    if not SARVAM_KEY:
        return JSONResponse({"error": "SARVAM_API_KEY not set"}, status_code=503)
    try:
        payload = {
            "text": req.text,
            "target_language_code": req.language,
            "speaker": req.speaker,
            "model": "bulbul:v2",
        }
        headers = {
            "api-subscription-key": SARVAM_KEY,
            "Content-Type": "application/json",
        }
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            json=payload, headers=headers, timeout=45,
        )
        if r.status_code != 200:
            return JSONResponse(
                {"error": f"Sarvam TTS {r.status_code}: {r.text[:300]}"},
                status_code=500,
            )
        audios = r.json().get("audios", [])
        if not audios:
            return JSONResponse({"error": "No audio in response"}, status_code=500)
        return {"audio": audios[0], "format": "wav"}
    except Exception as e:
        return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  CHAT — Grok
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

SYSTEM_PROMPT = (
    "You are Kalak Shetra's helpful seller assistant for Indian artisans. "
    "Help with pricing, product descriptions, festival tips, business guidance. "
    "Reply in the same language the user speaks. Keep answers short and practical."
)

@app.post("/chat")
async def chat(req: ChatRequest):
    if grok is None:
        return JSONResponse({"error": "XAI_API_KEY not set"}, status_code=503)
    try:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.messages:
            msgs.append({"role": m.role, "content": m.content})

        resp = grok.chat.completions.create(
            model="grok-2-1212",       # proven working model
            messages=msgs,
            temperature=0.7,
            max_tokens=400,
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        return JSONResponse({"error": f"Grok failed: {e}"}, status_code=500)


# ─────────────────────────────────────────────────────────────
#  DEBUG — test all APIs
# ─────────────────────────────────────────────────────────────
@app.get("/debug")
def debug():
    results = {}

    # Grok
    if grok:
        try:
            r = grok.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10,
            )
            results["grok"] = {"ok": True, "reply": r.choices[0].message.content}
        except Exception as e:
            results["grok"] = {"ok": False, "error": str(e)[:400]}
    else:
        results["grok"] = {"ok": False, "error": "XAI_API_KEY not configured"}

    # Sarvam TTS test
    if SARVAM_KEY:
        try:
            r = requests.post(
                "https://api.sarvam.ai/text-to-speech",
                json={
                    "text": "నమస్కారం",
                    "target_language_code": "te-IN",
                    "speaker": "meera",
                    "model": "bulbul:v2",
                },
                headers={
                    "api-subscription-key": SARVAM_KEY,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            results["sarvam_tts"] = {
                "ok": r.status_code == 200,
                "status": r.status_code,
                "body_preview": r.text[:400],
            }
        except Exception as e:
            results["sarvam_tts"] = {"ok": False, "error": str(e)[:400]}
    else:
        results["sarvam_tts"] = {"ok": False, "error": "SARVAM_API_KEY not configured"}

    # remove.bg — just report configured
    results["removebg"] = {"configured": bool(REMOVEBG_KEY)}

    return results