"""
🇪🇹 Amharic Hate Speech Detection — Telegram Bot (Cloud Version)
Bot: @tarmaber_bot
Hosting: Railway.app (24/7 online)
Model:   HuggingFace Hub (private repo)

Strike System:
  Strike 1  →  Delete + Warn
  Strike 2  →  Delete + Warn + Mute 1 hour
  Strike 3+ →  Delete + Mute 24 hours
"""

import os
import json
import logging
import torch
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update, ChatPermissions
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ════════════════════════════════════════════
#  CONFIGURATION  (loaded from env variables)
# ════════════════════════════════════════════
BOT_TOKEN   = os.environ.get("BOT_TOKEN", "").strip()
HF_REPO_ID  = os.environ.get("HF_REPO_ID", "").strip()   # e.g. "yourname/amharic-hate-speech"
HF_TOKEN    = os.environ.get("HF_TOKEN", "").strip()      # HuggingFace read token

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
STRIKES_FILE = "user_strikes.json"

# ── Startup validation ────────────────────────────────────────
missing = [k for k, v in {"BOT_TOKEN": BOT_TOKEN, "HF_REPO_ID": HF_REPO_ID, "HF_TOKEN": HF_TOKEN}.items() if not v]
if missing:
    raise SystemExit(f"❌  Missing required environment variables: {', '.join(missing)}\n"
                     f"    Set them in Railway → Variables tab.")

# ════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════
#  LOAD MODEL FROM HUGGINGFACE HUB
# ════════════════════════════════════════════
logger.info(f"⏳  Downloading model from HuggingFace: {HF_REPO_ID} …")
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
model     = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
logger.info(f"✅  Model loaded on {DEVICE.upper()}.")

LABEL_MAP = {0: "መልካም", 1: "ጥላቻ"}

# ════════════════════════════════════════════
#  STRIKE PERSISTENCE
# ════════════════════════════════════════════
def load_strikes() -> dict:
    if os.path.exists(STRIKES_FILE):
        with open(STRIKES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_strikes(strikes: dict):
    with open(STRIKES_FILE, "w", encoding="utf-8") as f:
        json.dump(strikes, f, indent=2)

user_strikes: dict = load_strikes()

# ════════════════════════════════════════════
#  MODEL INFERENCE
# ════════════════════════════════════════════
def predict(text: str) -> tuple[str, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_idx   = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][pred_idx].item()
    return LABEL_MAP[pred_idx], confidence

# ════════════════════════════════════════════
#  HELPER
# ════════════════════════════════════════════
async def safe_delete(message):
    try:
        await message.delete()
        return True
    except Exception as e:
        logger.warning(f"Could not delete message: {e}")
        return False

# ════════════════════════════════════════════
#  MAIN MESSAGE HANDLER
# ════════════════════════════════════════════
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message is None or not message.text:
        return
    if message.chat.type not in ("group", "supergroup"):
        return

    text    = message.text.strip()
    user    = message.from_user
    chat_id = message.chat_id
    user_id = user.id

    if len(text.split()) < 2:
        return

    label, confidence = predict(text)
    logger.info(
        f"[{message.chat.title}] @{user.username}: "
        f"'{text[:40]}' → {label} ({confidence:.2%})"
    )

    if label != "ጥላቻ" or confidence < CONFIDENCE_THRESHOLD:
        return

    key = f"{chat_id}:{user_id}"
    user_strikes[key] = user_strikes.get(key, 0) + 1
    strike = user_strikes[key]
    save_strikes(user_strikes)

    mention = f"@{user.username}" if user.username else user.first_name
    await safe_delete(message)

    if strike == 1:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ {mention} — Your message was removed for hate speech "
                f"that violates our community rules.\n"
                f"🔴 This is your *1st warning*. Please respect all members."
            ),
            parse_mode="Markdown",
        )

    elif strike == 2:
        until = datetime.now() + timedelta(hours=1)
        try:
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=until,
            )
            mute_text = "muted for *1 hour*"
        except Exception as e:
            logger.error(f"Mute failed: {e}")
            mute_text = "(mute failed — check bot admin rights)"

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"🔇 {mention} — Repeated hate speech detected.\n"
                f"Your message was removed and you have been {mute_text}.\n"
                f"🔴 This is your *2nd warning* — next violation = 24-hour mute."
            ),
            parse_mode="Markdown",
        )

    else:
        until = datetime.now() + timedelta(hours=24)
        try:
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=until,
            )
            mute_text = "muted for *24 hours*"
        except Exception as e:
            logger.error(f"Mute failed: {e}")
            mute_text = "(mute failed — check bot admin rights)"

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"🚫 {mention} — Continued violations. "
                f"Message removed and you have been {mute_text}.\n"
                f"Strike count: *{strike}*. Continued violations may result in removal."
            ),
            parse_mode="Markdown",
        )

# ════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════
def main():
    logger.info("🤖  Starting @tarmaber_bot (Cloud Mode) …")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("🟢  Bot is polling for messages 24/7 …")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
