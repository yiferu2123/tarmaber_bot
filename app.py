"""
🇪🇹 Amharic Hate Speech Detection — HuggingFace Space
Combines:
  • /predict  — ML inference API (used by bot.py internally)
  • /webhook  — Telegram webhook receiver
  • /          — Health check

Deploy as a Docker Space on HuggingFace.
Required Secrets (Settings → Repository Secrets):
  BOT_TOKEN        — Telegram bot token
  WEBHOOK_URL      — Full public URL of THIS Space, e.g. https://YIFER-tarmaber-bot.hf.space
  HF_TOKEN         — HuggingFace token (to load private model)
  HF_REPO_ID       — HuggingFace model repo, e.g. YIFER/amharic-hate-speech
  CONFIDENCE_THRESHOLD — (optional) default 0.70
"""

import os
import asyncio
import logging

import torch
import httpx
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
from telegram import Update, Bot

# ── Bot logic lives in bot.py (same container) ──────────────────────────────
from bot import build_application, WEBHOOK_URL, BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════
#  LOAD ML MODEL
# ════════════════════════════════════════════
HF_TOKEN  = os.environ.get("HF_TOKEN", "").strip() or None
REPO_ID   = os.environ.get("HF_REPO_ID", "YIFER/amharic-hate-speech").strip()

logger.info(f"Loading model {REPO_ID} …")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN)
model     = AutoModelForSequenceClassification.from_pretrained(REPO_ID, token=HF_TOKEN)
model.eval()
logger.info("✅  Model loaded.")

# ════════════════════════════════════════════
#  PTB APPLICATION (shared instance)
# ════════════════════════════════════════════
ptb_app = build_application()

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
FULL_WEBHOOK = WEBHOOK_URL.rstrip("/") + WEBHOOK_PATH

# Track initialization state
_bot_initialized = False

# ════════════════════════════════════════════
#  REGISTER WEBHOOK (background task)
# ════════════════════════════════════════════
async def register_webhook():
    """Register webhook with Telegram API using raw HTTP (no PTB dependency)."""
    global _bot_initialized
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook"
    params = {
        "url": FULL_WEBHOOK,
        "drop_pending_updates": True,
    }
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=params)
                data = resp.json()
                if data.get("ok"):
                    logger.info(f"🔗  Webhook registered: {FULL_WEBHOOK}")
                    # Now initialize PTB
                    await ptb_app.initialize()
                    await ptb_app.start()
                    _bot_initialized = True
                    logger.info("🟢  Bot is ready to receive updates!")
                    return
                else:
                    logger.warning(f"Webhook registration returned: {data}")
        except Exception as e:
            logger.warning(f"⚠️  Webhook attempt {attempt+1}/5 failed: {e}")
        await asyncio.sleep(5)
    logger.error("❌  Could not register webhook after 5 attempts.")

# ════════════════════════════════════════════
#  FASTAPI LIFESPAN  (startup / shutdown)
# ════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: launch webhook registration as background task ──
    logger.info("🚀  Space is starting. Registering webhook in background …")
    asyncio.create_task(register_webhook())
    yield
    # ── Shutdown ─────────────────────────────
    if _bot_initialized:
        try:
            await ptb_app.bot.delete_webhook()
            await ptb_app.stop()
            await ptb_app.shutdown()
        except Exception as e:
            logger.warning(f"Shutdown cleanup error: {e}")
    logger.info("👋  Space stopped.")

# ════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════
app = FastAPI(title="Amharic Hate Speech Bot + API", lifespan=lifespan)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "🟢 Space is running!",
        "bot_ready": _bot_initialized,
        "webhook": FULL_WEBHOOK,
    }

# ── Telegram webhook receiver ─────────────────────────────────────────────────
@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if not _bot_initialized:
        logger.warning("⏳  Bot not yet initialized, dropping update.")
        return Response(status_code=200)
    body = await request.json()
    update = Update.de_json(body, ptb_app.bot)
    await ptb_app.process_update(update)
    return Response(status_code=200)

# ── ML inference endpoint ─────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=-1)

    return {
        "is_hate_speech": bool(predicted_class.item() == 1),
        "confidence": float(confidence.item()),
    }
