"""
🛡️ Amharic Hate Speech Moderator Bot
Configured for deployment on Render.com as a FREE Web Service (runs 24/7)
Uses the Hugging Face Inference API for predictions.
"""

import os
import logging
import asyncio
from datetime import datetime
import threading
import httpx
from flask import Flask

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest

# ── Flask Web Server (For Render Free Tier health check) ──────────
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Bot is running live 24/7!"

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────
MODEL_ID = "mekonnena/amharic-hate-speech-classifier"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {}
if HF_API_TOKEN:
    headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

# Moderation warning and ban tracker
warn_history = {}

# Stats tracker
stats = {
    "total_predictions": 0,
    "hate_detected": 0,
    "not_hate_detected": 0,
    "telegram_users": set(),
    "start_time": datetime.now(),
    "messages_deleted": 0,
    "warnings_sent": 0,
    "bans_1hr": 0,
    "bans_24hr": 0,
    "bans_perm": 0,
}

LABEL_MAP = {
    "LABEL_0": {"name": "✅ Not Hate Speech", "am": "ጥላቻ አይደለም", "emoji": "✅"},
    "LABEL_1": {"name": "🚨 Hate Speech", "am": "የጥላቻ ንግግር", "emoji": "🚨"},
}

def predict(text: str) -> dict:
    if not text or not text.strip():
        return {"error": "Empty text"}
    try:
        payload = {"inputs": text.strip()}
        with httpx.Client(timeout=10.0) as client:
            response = client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

        if isinstance(response_json, list) and len(response_json) > 0:
            if isinstance(response_json[0], list):
                result = response_json[0][0]
            else:
                result = response_json[0]
        else:
            return {"error": "Invalid API response"}

        label_key = result["label"]
        confidence = result["score"]
        label_info = LABEL_MAP.get(label_key, {"name": label_key, "am": "", "emoji": "❓"})

        stats["total_predictions"] += 1
        if label_key == "LABEL_1":
            stats["hate_detected"] += 1
        else:
            stats["not_hate_detected"] += 1

        return {
            "label": label_info["name"],
            "label_am": label_info["am"],
            "emoji": label_info["emoji"],
            "confidence": confidence,
            "raw_label": label_key,
        }
    except Exception as e:
        logger.error(f"Inference API Error: {e}")
        return {"error": str(e)}

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats["telegram_users"].add(update.effective_user.id)
    welcome = (
        "🛡️ *Amharic Hate Speech Detector Bot*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "ሰላም! 👋 Welcome!\n\n"
        "I can detect hate speech in Amharic text using AI.\n"
        "የጥላቻ ንግግርን በ AI እለያለሁ።\n\n"
        "📝 *How to use:*\n"
        "Simply send me any Amharic text and I'll classify it!\n"
        "📋 *Commands:*\n"
        "/start  — This message\n"
        "/help   — Help & usage guide\n"
        "/stats  — View statistics\n"
        "/about  — About this bot\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⬇️ *Send Amharic text below to get started!*"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 *Help — Amharic Hate Speech Detector*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🔍 *Classification:*\n"
        "Send any Amharic text and I'll tell you if it's:\n"
        "  ✅ Not Hate Speech (ጥላቻ አይደለም)\n"
        "  🚨 Hate Speech (የጥላቻ ንግግር)\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = datetime.now() - stats["start_time"]
    hours = int(uptime.total_seconds() // 3600)
    minutes = int((uptime.total_seconds() % 3600) // 60)
    stats_text = (
        "📊 *Bot Statistics*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⏱️ Uptime: {hours}h {minutes}m\n"
        f"🔢 Total Classifications: {stats['total_predictions']}\n"
        f"✅ Not Hate Speech: {stats['not_hate_detected']}\n"
        f"🚨 Hate Speech Detected: {stats['hate_detected']}\n"
        f"👥 Unique Users: {len(stats['telegram_users'])}\n\n"
        f"🗑️ Messages Deleted: {stats['messages_deleted']}\n"
        f"⚠️ Warnings Issued: {stats['warnings_sent']}\n"
        f"⏳ 1h Bans: {stats['bans_1hr']}\n"
        f"⏰ 24h Bans: {stats['bans_24hr']}\n"
        f"🚫 Permanent Bans: {stats['bans_perm']}"
    )
    await update.message.reply_text(stats_text, parse_mode="Markdown")

async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = (
        "ℹ️ *About This Bot*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🛡️ *Amharic Hate Speech Detector*\n\n"
        "This bot uses a fine-tuned XLM-RoBERTa model\n"
        "to detect hate speech in Amharic (አማርኛ) text.\n"
    )
    await update.message.reply_text(about_text, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    text = update.message.text
    user = update.effective_user
    chat = update.effective_chat
    stats["telegram_users"].add(user.id)

    logger.info(f"Telegram | User {user.id} ({user.first_name}) in Chat {chat.id}: classifying text")

    result = predict(text)
    if "error" in result:
        if chat.type == "private":
            await update.message.reply_text("⚠️ Failed to process text. Please try again later.")
        return

    is_group = chat.type in ["group", "supergroup"]
    if is_group:
        if result["raw_label"] == "LABEL_1":
            # Skip if user is admin/creator
            try:
                member = await context.bot.get_chat_member(chat_id=chat.id, user_id=user.id)
                if member.status in ["creator", "administrator"]:
                    return
            except Exception as e:
                logger.error(f"Failed to check chat member status: {e}")

            # Delete the hate speech message
            try:
                await update.message.delete()
                stats["messages_deleted"] += 1
            except Exception as e:
                logger.error(f"Failed to delete message: {e}")

            history_key = f"{chat.id}:{user.id}"
            user_warns = warn_history.get(history_key, 0) + 1
            warn_history[history_key] = user_warns
            now_epoch = int(datetime.utcnow().timestamp())

            if user_warns == 1:
                stats["warnings_sent"] += 1
                warn_msg = (
                    f"⚠️ **Warning 1 / የመጀመሪያ ማስጠንቀቂያ**\n"
                    f"User: {user.mention_markdown_v2()}\n"
                    f"Your message was automatically deleted because it was flagged as hate speech in Amharic\\.\n"
                )
                await context.bot.send_message(chat_id=chat.id, text=warn_msg, parse_mode="MarkdownV2")

            elif user_warns == 2:
                stats["warnings_sent"] += 1
                stats["bans_1hr"] += 1
                ban_duration = 3600
                until_date = datetime.fromtimestamp(now_epoch + ban_duration)
                warn_msg = (
                    f"🚨 **Warning 2 / ሁለተኛ ማስጠንቀቂያ**\n"
                    f"User: {user.mention_markdown_v2()}\n"
                    f"You have been banned for **1 hour** for posting hate speech\\.\n"
                )
                try:
                    await context.bot.ban_chat_member(chat_id=chat.id, user_id=user.id, until_date=until_date)
                    await context.bot.send_message(chat_id=chat.id, text=warn_msg, parse_mode="MarkdownV2")
                except Exception as e:
                    logger.error(f"Failed to ban user: {e}")
                    await context.bot.send_message(chat_id=chat.id, text=f"❌ Failed to ban {user.first_name} but message deleted (needs admin rights).")

            elif user_warns == 3:
                stats["warnings_sent"] += 1
                stats["bans_24hr"] += 1
                ban_duration = 86400
                until_date = datetime.fromtimestamp(now_epoch + ban_duration)
                warn_msg = (
                    f"🚫 **Warning 3 / ሶስተኛ ማስጠንቀቂያ**\n"
                    f"User: {user.mention_markdown_v2()}\n"
                    f"You have been banned for **24 hours** for posting hate speech\\.\n"
                )
                try:
                    await context.bot.ban_chat_member(chat_id=chat.id, user_id=user.id, until_date=until_date)
                    await context.bot.send_message(chat_id=chat.id, text=warn_msg, parse_mode="MarkdownV2")
                except Exception as e:
                    logger.error(f"Failed to ban user: {e}")

            else:
                stats["bans_perm"] += 1
                warn_msg = (
                    f"❌ **Permanent Ban / ዘላቂ እገዳ**\n"
                    f"User: {user.mention_markdown_v2()}\n"
                    f"You have been permanently banned from this group for repeated violations\\.\n"
                )
                try:
                    await context.bot.ban_chat_member(chat_id=chat.id, user_id=user.id)
                    await context.bot.send_message(chat_id=chat.id, text=warn_msg, parse_mode="MarkdownV2")
                except Exception as e:
                    logger.error(f"Failed to permanently ban user: {e}")
        return

    # DM (Private chat) — show classification result
    conf_bar = "█" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
    response = (
        f"{result['emoji']} *Classification Result*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🏷️ *Result:* {result['label']}\n"
        f"📊 *Confidence:* {result['confidence']:.1%}\n"
        f"    [{conf_bar}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    keyboard = [[InlineKeyboardButton("🔍 Classify Another", callback_data="classify_another")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(response, parse_mode="Markdown", reply_markup=reply_markup)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "classify_another":
        await query.message.reply_text("📝 Send me another Amharic text to classify!")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Telegram error: {context.error}")

async def run_bot():
    req = HTTPXRequest(connect_timeout=15.0, read_timeout=15.0)
    app = Application.builder().token(TELEGRAM_TOKEN).request(req).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_error_handler(error_handler)

    logger.info("Initializing Telegram Bot...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("✅ Telegram Bot is running!")

    while True:
        await asyncio.sleep(3600)

def start_bot_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_bot())

if __name__ == "__main__":
    # 1. Start Bot in background thread
    bot_thread = threading.Thread(target=start_bot_thread, daemon=True)
    bot_thread.start()

    # 2. Run Web server on main thread for Render health check
    run_web_server()
