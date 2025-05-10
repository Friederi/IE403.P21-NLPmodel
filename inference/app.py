# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
import numpy as np
import os

from model.model import BertWithScalarFeatures
from Annotation_Untils import singular_comment_without_annotation
from DataCleaning_Untils import clean_comment

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.getcwd(), "model", "best_model.pt")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertWithScalarFeatures(scalar_feature_dim=16, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

class CommentRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(req: CommentRequest):
    text = clean_comment(req.text)
    scalars = singular_comment_without_annotation(text)

    scalar_features = np.array([
        scalars['hate_score'],
        scalars['toxicity'],
        scalars['obscene'],
        scalars['identity_attack'],
        scalars['insult'],
        scalars['threat'],
        scalars['sexual_explicit'],
        1.0 if scalars['sentiment'] == 'negative' else 0.0,
        1.0 if scalars['sentiment'] == 'neutral' else 0.0,
        1.0 if scalars['sentiment'] == 'positive' else 0.0,
        1.0 if scalars['emotion'] == 'anger' else 0.0,
        1.0 if scalars['emotion'] == 'fear' else 0.0,
        1.0 if scalars['emotion'] == 'joy' else 0.0,
        1.0 if scalars['emotion'] == 'love' else 0.0,
        1.0 if scalars['emotion'] == 'sadness' else 0.0,
        1.0 if scalars['emotion'] == 'surprise' else 0.0
    ], dtype=np.float32)
    
    scalar_features = torch.tensor(scalar_features).unsqueeze(0).to(device)

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, scalar_features=scalar_features)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

    return {
        "text": text,
        "predicted_class": predicted_class,
        "confidence": confidence
    }
