---
title: Tarmaber Bot - Amharic Hate Speech Detection
emoji: 🇪🇹
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🇪🇹 Tarmaber Bot — Amharic Hate Speech Detection

A Telegram bot that **automatically detects and moderates Amharic hate speech** in group chats using a fine-tuned transformer model.

## Architecture

```
Telegram Group Chat
        │
        ▼  (webhook)
┌──────────────────────────┐
│   HuggingFace Space      │
│   ┌────────────────────┐ │
│   │  FastAPI (uvicorn)  │ │
│   │                     │ │
│   │  POST /webhook/...  │──── receives Telegram updates
│   │  POST /predict      │──── ML inference (internal)
│   │  GET  /             │──── health check
│   └────────────────────┘ │
│   ┌────────────────────┐ │
│   │  Transformer Model │ │
│   │  (BERT-based)      │ │
│   └────────────────────┘ │
└──────────────────────────┘
```

## Strike System

| Strike | Action |
|--------|--------|
| 1st    | Delete message + Warning |
| 2nd    | Delete + Warn + Mute 1 hour |
| 3rd+   | Delete + Mute 24 hours |

## Required Environment Variables

Set these as **Repository Secrets** in the HuggingFace Space settings:

| Variable | Description |
|----------|-------------|
| `BOT_TOKEN` | Telegram bot token from @BotFather |
| `WEBHOOK_URL` | Full public URL of this Space |
| `HF_REPO_ID` | Model repo (e.g. `YIFER/amharic-hate-speech`) |
| `HF_TOKEN` | HuggingFace read token |
| `CONFIDENCE_THRESHOLD` | Min confidence to flag (default: 0.70) |

## Deployment

This repo auto-deploys to HuggingFace Spaces via GitHub webhook.
Push to `main` → HF Space rebuilds automatically.
