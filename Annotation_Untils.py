# %% [markdown]
# # Library import

# %%
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

pd.set_option('display.max_colwidth', None)

# %% [markdown]
# ## Calling all pretrained model

# %%
sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
toxicity_pipe = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

# %%
emotion_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
emotion_model.eval()  # Set the model to evaluation mode

def get_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return label_list[predicted_class]

# %%
hate_model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
hate_tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
hate_model.eval()  # Set the model to evaluation mode

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
    """
    Annotate a batch of comments with scalar features like sentiment, toxicity, emotion, etc.

    Args:
        comments (list[str]): List of raw text comments.

    Returns:
        list[dict]: List of annotated comment dictionaries.
    """
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
    # Flag strong negative + hate or toxicity
    if (row['sentiment'] == 'negative' and row['emotion'] == 'anger') and (
        row['hate_score'] > 0.9 or
        row['toxicity'] > 0.8 or
        row['insult'] > 0.5 or
        row['threat'] > 0.4 or
        row['identity_attack'] > 0.5
    ):
        return "delete"

    # Even if sentiment is neutral or positive, still delete if too toxic
    
    if row['toxicity'] > 0.6 or \
        row['hate_score'] > 0.97 or \
        row['insult'] > 0.6 or \
        row['identity_attack'] > 0.6 or \
        row['obscene'] > 0.6 or \
        row['threat'] > 0.6 or \
        row['sexual_explicit'] > 0.6:

        return "delete"
    

    # Emotional cue: very angry + some toxicity
    if row['emotion'] == 'anger' and row['toxicity'] > 0.4:
        return "delete"

    # Everything else is kept
    return "keep"

# %%
def analyze_moderation_output(df, save_csv=True, csv_name="dashboard_summary.csv"):
    print("🔎 Starting moderation evaluation...")

    # 1. Label Distribution
    print("\n🔢 Label distribution (%):")
    label_dist = df["label"].value_counts(normalize=True) * 100
    print(label_dist)

    # 2. Sentiment vs Final Label
    print("\n🧠 Sentiment vs Final Label:")
    sentiment_vs_label = pd.crosstab(df['sentiment'], df['label'])
    print(sentiment_vs_label)

    # 3. Emotion vs Final Label
    print("\n❤️ Emotion vs Final Label:")
    emotion_vs_label = pd.crosstab(df['emotion'], df['label'])
    print(emotion_vs_label)

    # 4. Toxicity Stats
    print("\n📊 Toxicity Score Summary:")
    toxic_cols = ['toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
    toxic_stats = df[toxic_cols].describe()
    print(toxic_stats)

    # 5. Suspected False Positives
    print("\n🤔 Potential False Positives (NEGATIVE but non-toxic):")
    false_positives = df[
        (df['label'] == 'delete') &
        (df['toxicity'] < 0.2) &
        (df['sentiment'] == 'NEGATIVE')
    ][['comment', 'sentiment', 'toxicity', 'emotion', 'label']]
    print(false_positives.head(10))

    # 6. Visualization
    print("\n📊 Generating visualization...")
    sns.set(style="whitegrid")

    plt.figure(figsize=(7,4))
    sns.countplot(x='sentiment', hue='label', data=df)
    plt.title("Sentiment vs Moderation Label")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    sns.countplot(x='emotion', hue='label', data=df, order=df['emotion'].value_counts().index)
    plt.title("Emotion vs Moderation Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if save_csv:
        dashboard = {
            "Label Distribution (%)": label_dist,
            "Sentiment vs Label": sentiment_vs_label,
            "Emotion vs Label": emotion_vs_label,
            "Toxicity Stats": toxic_stats
        }

        with pd.ExcelWriter(csv_name.replace(".csv", ".xlsx")) as writer:
            label_dist.to_frame(name="Percentage").to_excel(writer, sheet_name="Label Dist")
            sentiment_vs_label.to_excel(writer, sheet_name="Sentiment vs Label")
            emotion_vs_label.to_excel(writer, sheet_name="Emotion vs Label")
            toxic_stats.to_excel(writer, sheet_name="Toxicity Stats")
            false_positives.to_excel(writer, sheet_name="Suspected False Positives")

        print(f"\n✅ Dashboard summary saved to: {csv_name.replace('.csv', '.xlsx')}")

    return false_positives

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