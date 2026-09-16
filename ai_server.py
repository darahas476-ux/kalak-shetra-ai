# ═══════════════════════════════════════════════════════════════════════════
#  KALAK SHETRA — AI BACKEND
#  Endpoints: /enhance (bg removal) · /translate · /chat (Grok)
#  Optimized for Render free tier (512 MB RAM)
# ═══════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove, new_session
from deep_translator import GoogleTranslator
from pydantic import BaseModel
from typing import List, Optional
import io, base64, os
from openai import OpenAI

app = FastAPI(title="Kalak Shetra AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Grok client (xAI exposes an OpenAI-compatible endpoint) ──
GROK_KEY = os.getenv("XAI_API_KEY", "")
grok = OpenAI(
    api_key=GROK_KEY if GROK_KEY else "dummy",
    base_url="https://api.x.ai/v1",
) if GROK_KEY else None

# ── Preload rembg model at startup (fits in 512 MB with isnet-general-use) ──
_session = None

@app.on_event("startup")
async def startup():
    global _session
    print("Loading rembg model...")
    _session = new_session("isnet-general-use")  # lighter than u2net
    # warm up with a tiny image so first real call is instant
    dummy = Image.new("RGB", (32, 32), (255, 255, 255))
    buf = io.BytesIO()
    dummy.save(buf, format="PNG")
    remove(buf.getvalue(), session=_session)
    print("Model ready.")

@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": _session is not None,
        "grok_configured": grok is not None,
    }

# ─────────────────────────────────────────────────────────────
#  /enhance  — background removal + sharpen
# ─────────────────────────────────────────────────────────────
@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        if max(img.size) > 800:
            img.thumbnail((800, 800), Image.LANCZOS)

        img = remove(img, session=_session)

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Brightness(img).enhance(1.05)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return JSONResponse({"image": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────────────────────
#  /translate
# ─────────────────────────────────────────────────────────────
LANG = {"en":"en","en_US":"en","hi":"hi","hi_IN":"hi",
        "te":"te","te_IN":"te","auto":"auto"}

@app.post("/translate")
async def translate(text: str = Form(...),
                    source: str = Form("auto"),
                    target: str = Form("en")):
    try:
        s = LANG.get(source, "auto")
        t = LANG.get(target, "en")
        return {"translated": GoogleTranslator(source=s, target=t).translate(text)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────────────────────
#  /chat  — Grok-powered seller assistant
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
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
        raise HTTPException(status_code=503, detail="Grok not configured. Set XAI_API_KEY.")

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        msgs.append({"role": m.role, "content": m.content})

    try:
        resp = grok.chat.completions.create(
            model="grok-3-mini-fast",  # cheapest + fastest
            messages=msgs,
            temperature=0.7,
            max_tokens=400,
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)