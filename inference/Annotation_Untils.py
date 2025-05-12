# %% [markdown]
# # Library import

# %%
import pandas as pd
import torch
import os

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

pd.set_option('display.max_colwidth', None)

# %% [markdown]
#MODEL_DIR = os.getenv("MODEL_DIR", "./model/pretrained_model")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, "models", "pretrained_model")

ROBERTA_BASE_MODEL_PATH = os.path.join(PRETRAINED_DIR, "twitter-roberta-base-sentiment")
TOXIC_BERT_MODEL_PATH = os.path.join(PRETRAINED_DIR, "toxic-bert")
BERT_BASE_MODEL_PATH = os.path.join(PRETRAINED_DIR, "bert-base-uncased-emotion")
ROBERTA_HATE_MODEL_PATH = os.path.join(PRETRAINED_DIR, "roberta-hate-speech")


# %%
sentiment_pipe = pipeline("sentiment-analysis", model=ROBERTA_BASE_MODEL_PATH, local_files_only=True)
toxicity_pipe = pipeline("text-classification", model=TOXIC_BERT_MODEL_PATH, top_k=None, local_files_only=True)

# %%
emotion_model = AutoModelForSequenceClassification.from_pretrained(BERT_BASE_MODEL_PATH, local_files_only=True)
emotion_tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODEL_PATH, local_files_only=True)
emotion_model.eval()

def get_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return label_list[predicted_class]

# %%
hate_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_HATE_MODEL_PATH, local_files_only=True)
hate_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_HATE_MODEL_PATH, local_files_only=True)
hate_model.eval()

def get_hate_score(text):
    try:
        inputs = hate_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = hate_model(**inputs)
            scores = torch.sigmoid(outputs.logits)
        return scores[0][0].item()
    except:
        return 0.0

# %%
def get_sentiment(text):
    try:
        result = sentiment_pipe(text)[0]
        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }
        return label_map.get(result["label"], "UNKNOWN")
    except:
        return "UNKNOWN"

def get_toxicity(text):
    try:
        result = toxicity_pipe(text)[0]
        return {r['label'].lower(): r['score'] for r in result}
    except:
        return {}

# %%
def annotate_comment(comments):

    ensemble_data = []

    for comment in tqdm(comments, desc="Annotating comments"):
        result = {
            "comment": comment,
            "sentiment": get_sentiment(comment),
            "hate_score": get_hate_score(comment),
            "emotion": get_emotion(comment)
        }

        toxicity_scores = get_toxicity(comment)
        for key in ['toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']:
            result[key] = toxicity_scores.get(key, 0.0)

        result["label"] = "unknown"  
        ensemble_data.append(result)

    return ensemble_data

# %%
def auto_label(row):
    if (row['sentiment'] == 'negative' and row['emotion'] == 'anger') and (
        row['hate_score'] > 0.9 or
        row['toxicity'] > 0.8 or
        row['insult'] > 0.5 or
        row['threat'] > 0.4 or
        row['identity_attack'] > 0.5
    ):
        return "delete"
    
    if row['toxicity'] > 0.6 or \
        row['insult'] > 0.6 or \
        row['identity_attack'] > 0.6 or \
        row['obscene'] > 0.6 or \
        row['threat'] > 0.6 or \
        row['sexual_explicit'] > 0.6:

        return "delete"
    
    if row['emotion'] == 'anger' and row['toxicity'] > 0.4:
        return "delete"

    return "keep"

# %%
def singular_comment_with_annotation(comment):
    result = {
        "comment": comment,
        "sentiment": get_sentiment(comment),
        "hate_score": get_hate_score(comment),
        "emotion": get_emotion(comment)
    }

    toxicity_scores = get_toxicity(comment)
    for key in ['toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']:
        result[key] = toxicity_scores.get(key, 0.0)
    
    result["label"] = auto_label(result)
    return result

# %%
def singular_comment_without_annotation(comment):
    result = {
        "comment": comment,
        "sentiment": get_sentiment(comment),
        "hate_score": get_hate_score(comment),
        "emotion": get_emotion(comment)
    }

    toxicity_scores = get_toxicity(comment)
    for key in ['toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']:
        result[key] = toxicity_scores.get(key, 0.0)
    
    return result