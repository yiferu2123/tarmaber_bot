FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch first (forces pip to ONLY look at the lightweight CPU version)
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the bot's code
COPY . .

# Run the bot
CMD ["python", "bot.py"]
