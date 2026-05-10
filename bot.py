"""
🇪🇹 Amharic Hate Speech Detection — Telegram Bot (HuggingFace Spaces / Webhook)
Bot: @tarmaber_bot
Hosting: HuggingFace Spaces (Docker, port 7860)
Brain:   Same Space — /predict endpoint (served by FastAPI)

Webhook flow:
  Telegram → POST /webhook/<BOT_TOKEN> → FastAPI → python-telegram-bot

Strike System:
  Strike 1  →  Delete + Warn
  Strike 2  →  Delete + Warn + Mute 1 hour
  Strike 3+ →  Delete + Mute 24 hours
"""

import os
import json
import logging
from datetime import datetime, timedelta

import requests
from telegram import Update, ChatPermissions
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ════════════════════════════════════════════
#  CONFIGURATION  (loaded from env variables)
# ════════════════════════════════════════════
BOT_TOKEN   = os.environ.get("BOT_TOKEN", "").strip()
API_URL     = os.environ.get("API_URL", "http://localhost:7860/predict").strip()
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()   # e.g. https://YIFER-tarmaber-bot.hf.space

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
STRIKES_FILE = "user_strikes.json"

if not BOT_TOKEN:
    raise SystemExit(
        "❌  Missing required environment variable: BOT_TOKEN\n"
        "    Set it in the HuggingFace Space → Settings → Repository Secrets."
    )
if not WEBHOOK_URL:
    raise SystemExit(
        "❌  Missing required environment variable: WEBHOOK_URL\n"
        "    Set it to the full public URL of your HuggingFace Space,\n"
        "    e.g.  https://YIFER-tarmaber-bot.hf.space"
    )

# ════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
#  MODEL INFERENCE (via /predict on same Space)
# ════════════════════════════════════════════
def predict(text: str) -> tuple[str, float]:
    try:
        response = requests.post(API_URL, json={"text": text}, timeout=15)
        response.raise_for_status()
        data = response.json()
        label      = "ጥላቻ" if data["is_hate_speech"] else "መልካም"
        confidence = float(data["confidence"])
        return label, confidence
    except Exception as e:
        logger.error(f"API Error: {e}")
        return "መልካም", 0.0

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
#  BUILD THE APPLICATION (shared instance)
# ════════════════════════════════════════════
def build_application() -> Application:
    """Build and return the configured PTB Application (no polling started)."""
    logger.info("🤖  Building @tarmaber_bot (Webhook / HuggingFace Mode) …")
    app = Application.builder().token(BOT_TOKEN).updater(None).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info(f"🧠  Brain API: {API_URL}")
    return app

# ════════════════════════════════════════════
#  ENTRY POINT  (only used for local polling)
# ════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("🔄  Running in local POLLING mode (dev/test only) …")
    local_app = Application.builder().token(BOT_TOKEN).build()
    local_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    local_app.run_polling(allowed_updates=Update.ALL_TYPES)
