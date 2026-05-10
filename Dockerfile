# Dockerfile for HuggingFace Space (Docker SDK)
# Serves: FastAPI (uvicorn) on port 7860
#   /predict  — ML inference
#   /webhook  — Telegram webhook
#   /         — Health check

FROM python:3.10-slim

# HuggingFace Spaces require a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies (heavy PyTorch layer cached first)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user bot.py .
COPY --chown=user app.py .
COPY --chown=user user_strikes.json* ./

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
