from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Load secrets
TOKEN_ENV = os.environ.get("HF_TOKEN")
TOKEN = TOKEN_ENV.strip() if TOKEN_ENV else None
REPO_ID = os.environ.get("HF_REPO_ID", "YIFER/amharic-hate-speech").strip()

print(f"Loading model {REPO_ID}...")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, token=TOKEN)
model.eval()
print("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "AI Model is running!"}

@app.post("/predict")
def predict(req: TextRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=-1)
    
    return {
        "is_hate_speech": bool(predicted_class.item() == 1),
        "confidence": float(confidence.item())
    }
