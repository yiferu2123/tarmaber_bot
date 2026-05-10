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
import logging

import torch
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
from telegram import Update

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

# ════════════════════════════════════════════
#  FASTAPI LIFESPAN  (startup / shutdown)
# ════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────
    await ptb_app.initialize()
    await ptb_app.bot.set_webhook(
        url=FULL_WEBHOOK,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )
    logger.info(f"🔗  Webhook registered: {FULL_WEBHOOK}")
    await ptb_app.start()
    yield
    # ── Shutdown ─────────────────────────────
    await ptb_app.bot.delete_webhook()
    await ptb_app.stop()
    await ptb_app.shutdown()
    logger.info("👋  Webhook deleted, bot stopped.")

# ════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════
app = FastAPI(title="Amharic Hate Speech Bot + API", lifespan=lifespan)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "🟢 Amharic Hate Speech Space is running!", "webhook": FULL_WEBHOOK}

# ── Telegram webhook receiver ─────────────────────────────────────────────────
@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
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
