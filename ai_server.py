# ═══════════════════════════════════════════════════════════════════════════
#  KALAK-SHETRA AI — FastAPI backend (Gemini + remove.bg only)
# ═══════════════════════════════════════════════════════════════════════════
import os
import io
import time
import json
import base64
import logging
import traceback
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("kalak-shetra")

# ─── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kalak-Shetra AI",
    description="AI backend for Kalakriti artisan marketplace",
    version="21.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  ENV HELPERS  (case-insensitive lookup)
# ═══════════════════════════════════════════════════════════════════════════
def env_get(*names: str) -> Optional[str]:
    """Return the first non-empty env var matching any of the given names (case-insensitive)."""
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    # case-insensitive fallback
    lower_map = {k.lower(): v for k, v in os.environ.items()}
    for n in names:
        v = lower_map.get(n.lower())
        if v and v.strip():
            return v.strip()
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI KEY ROTATOR
# ═══════════════════════════════════════════════════════════════════════════
class GeminiKeyRotator:
    """
    Round-robin over multiple Gemini API keys.
    Reads env vars: GEMINI_API_KEY_1 ... GEMINI_API_KEY_9 (+ GEMINI_API_KEY fallback)
    Case-insensitive: gemini_api_1, GEMINI_API_KEY_1, Gemini_Api_1 all work.
    """
    def __init__(self):
        self.keys: List[str] = []
        seen = set()

        for i in range(1, 10):
            k = env_get(f"GEMINI_API_KEY_{i}", f"gemini_api_{i}")
            if k and k not in seen:
                self.keys.append(k)
                seen.add(k)

        legacy = env_get("GEMINI_API_KEY", "gemini_api_key")
        if legacy and legacy not in seen:
            self.keys.append(legacy)

        self.current_index = 0
        self.failed_until: Dict[str, float] = {}
        log.info(f"🔑 Gemini rotator loaded {len(self.keys)} key(s)")

    def current(self) -> Optional[str]:
        if not self.keys:
            return None
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            if self.failed_until.get(key, 0) > now:
                continue
            return key
        return min(self.keys, key=lambda k: self.failed_until.get(k, 0))

    def mark_failed(self, key: str, cooldown_seconds: int = 3600):
        self.failed_until[key] = time.time() + cooldown_seconds
        log.warning(f"⚠️  Key ...{key[-6:]} cooled for {cooldown_seconds}s")

    def mark_ok(self, key: str):
        self.failed_until.pop(key, None)

    def stats(self):
        now = time.time()
        return [
            {
                "suffix": f"...{k[-6:]}",
                "available": self.failed_until.get(k, 0) <= now,
                "cooldown_remaining": max(0, int(self.failed_until.get(k, 0) - now)),
            }
            for k in self.keys
        ]

    def count(self) -> int:
        return len(self.keys)


gemini_rotator = GeminiKeyRotator()


def _is_quota_error(err: str) -> bool:
    e = err.lower()
    return any(x in e for x in [
        "429", "quota", "rate limit", "rate_limit",
        "resource_exhausted", "resource exhausted",
        "permission_denied", "permission denied",
        "invalid api key", "api key not valid",
        "401", "403",
    ])


def gemini_generate(
    prompt: Any,
    model_name: str = "gemini-3.6-flash",
    image_bytes: Optional[bytes] = None,
    max_retries: Optional[int] = None,
) -> str:
    """Call Gemini with automatic key rotation on quota errors."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise HTTPException(status_code=500, detail="google-generativeai not installed")

    if gemini_rotator.count() == 0:
        raise HTTPException(status_code=500, detail="No Gemini API keys configured")

    attempts = max_retries or gemini_rotator.count()
    last_error = None

    for attempt in range(attempts):
        key = gemini_rotator.current()
        if not key:
            break
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)

            if image_bytes is not None:
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                parts = [prompt, img] if isinstance(prompt, str) else [*prompt, img]
                response = model.generate_content(parts)
            else:
                response = model.generate_content(prompt)

            text = getattr(response, "text", None)
            if not text:
                try:
                    text = response.candidates[0].content.parts[0].text
                except Exception:
                    text = ""

            gemini_rotator.mark_ok(key)
            return text or ""

        except Exception as e:
            err = str(e)
            last_error = err
            if _is_quota_error(err):
                gemini_rotator.mark_failed(key, cooldown_seconds=3600)
                log.warning(f"🔁 Gemini key failed, rotating ({attempt + 2}/{attempts})")
                continue
            log.error(f"Gemini non-quota error: {err}")
            raise

    raise HTTPException(
        status_code=503,
        detail=f"All Gemini keys exhausted. Last error: {last_error}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"


class PricePredictRequest(BaseModel):
    title: str
    description: str = ""
    category: str = "Handicrafts"


# ═══════════════════════════════════════════════════════════════════════════
#  ROOT / HEALTH / DEBUG
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "kalak-shetra-ai",
        "version": "21.1.0",
        "gemini_keys": gemini_rotator.count(),
        "removebg_configured": bool(env_get("REMOVEBG_API_KEY", "remove_bg", "REMOVE_BG_API_KEY")),
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/debug")
async def debug():
    rbg = env_get("REMOVEBG_API_KEY", "remove_bg", "REMOVE_BG_API_KEY")
    return {
        "service": "kalak-shetra-ai",
        "version": "21.1.0",
        "gemini": {
            "total_keys": gemini_rotator.count(),
            "keys": gemini_rotator.stats(),
        },
        "removebg": {
            "configured": bool(rbg),
            "key_suffix": (rbg or "")[-6:],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  /analyze-product
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/analyze-product")
async def analyze_product(
    file: UploadFile = File(...),
    user_desc: str = Form(""),
):
    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        prompt = f"""You are an expert at Indian handicrafts, sarees, jewelry, and traditional crafts.

Analyze this product image and respond with ONLY a valid JSON object (no markdown, no code fences).

The user described it as: "{user_desc}" (may be empty).

Return JSON with these exact keys:
{{
  "title": "short catchy title (max 60 chars)",
  "description_en": "2-3 sentence English description (100-180 chars)",
  "description_hi": "same description in Hindi (Devanagari)",
  "category": "one of: Sarees, Dresses, Handicrafts, Jewelry",
  "price_low": integer (INR, realistic low estimate),
  "price_high": integer (INR, realistic high estimate),
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": float between 0 and 1
}}

Return only the JSON."""

        raw = gemini_generate(prompt, image_bytes=img_bytes).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(raw[start : end + 1])
            else:
                raise

        return JSONResponse(parsed)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/analyze-product error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  /enhance — remove background
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        api_key = env_get("REMOVEBG_API_KEY", "remove_bg", "REMOVE_BG_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="remove.bg API key not configured")

        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.remove.bg/v1.0/removebg",
                headers={"X-Api-Key": api_key},
                files={"image_file": ("image.jpg", img_bytes, "image/jpeg")},
                data={"size": "auto"},
            )

        if resp.status_code != 200:
            log.error(f"remove.bg {resp.status_code}: {resp.text[:200]}")
            return JSONResponse({
                "image": base64.b64encode(img_bytes).decode(),
                "warning": f"remove.bg returned {resp.status_code}, using original",
            })

        return JSONResponse({
            "image": base64.b64encode(resp.content).decode(),
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/enhance error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  /chat
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        if not req.messages:
            raise HTTPException(status_code=400, detail="No messages")

        system = (
            "You are Kalakriti Assistant, a helpful AI for Indian artisans "
            "selling sarees, dresses, handicrafts, and jewelry. "
            "Help with pricing, photography tips, festival sale planning, "
            "and product descriptions. Be warm and concise. "
            "Reply in the same language the user writes in (English, Hindi, Telugu)."
        )

        lines = []
        for m in req.messages:
            role = "User" if m.role == "user" else "Assistant"
            lines.append(f"{role}: {m.content}")

        prompt = f"{system}\n\n" + "\n".join(lines) + "\nAssistant:"

        reply = gemini_generate(prompt, model_name="gemini-3.6-flash")
        return {"reply": reply.strip()}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  /predict-price
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/predict-price")
async def predict_price(req: PricePredictRequest):
    try:
        prompt = f"""You are a pricing expert for Indian handicrafts. Based on:

Title: {req.title}
Description: {req.description}
Category: {req.category}

Return ONLY valid JSON:
{{
  "price_low": integer in INR,
  "price_high": integer in INR,
  "price_recommended": integer in INR (midpoint),
  "reasoning": "one sentence in English"
}}"""

        raw = gemini_generate(prompt, model_name="gemini-3.6-flash").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            parsed = json.loads(raw[start : end + 1]) if start != -1 and end > start else {}

        return JSONResponse(parsed)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/predict-price error: {e}")
        base = {"Sarees": 2500, "Dresses": 1400, "Jewelry": 900}.get(req.category, 800)
        return {
            "price_low": base,
            "price_high": int(base * 1.6),
            "price_recommended": int(base * 1.3),
            "reasoning": "Estimated from category baseline.",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  /translate
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/translate")
async def translate(req: TranslateRequest):
    try:
        prompt = (
            f"Translate the following text to language code '{req.target_lang}'. "
            f"Return ONLY the translation, no explanation, no quotes.\n\n"
            f"Text: {req.text}"
        )
        translated = gemini_generate(prompt, model_name="gemini-3.6-flash")
        return {"translated": translated.strip()}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/translate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  ERROR HANDLER + STARTUP
# ═══════════════════════════════════════════════════════════════════════════
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


@app.on_event("startup")
async def on_startup():
    rbg = env_get("REMOVEBG_API_KEY", "remove_bg", "REMOVE_BG_API_KEY")
    log.info("═" * 60)
    log.info("🚀 Kalak-Shetra AI v21.1.0 starting")
    log.info(f"   Gemini keys: {gemini_rotator.count()}")
    log.info(f"   remove.bg: {'✅' if rbg else '❌'}")
    log.info("═" * 60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
